"""
Module: matching_utils.py

Purpose:
This module provides a collection of utilities and core logic for the invoice line
item matching process. It includes functionalities for extracting potential part
numbers from text, managing and querying manufacturer data (with caching),
finding manufacturer names within descriptions, and orchestrating the analysis
of a product description to identify candidate parts from various data sources
(database or Azure AI Search). It plays a crucial role in preparing data for the
scoring and final match determination stages.

High-Level Design:
- Part Number Extraction: The `PartNumberExtractor` class implements a rule-based
  system for identifying potential manufacturer part numbers within a given text.
  It uses a series of regex patterns (rulesets) and exclusion lists to find and
  validate candidates. It also calculates an "effective length" for part numbers,
  which is used in scoring.
- Manufacturer Data Management:
    - `read_manufacturer_data`: Fetches manufacturer names (unclean and clean versions)
      from a database (via `sql_utils`). Implements an in-memory cache with a TTL
      (Time-To-Live) and a threading lock (`_manufacturer_data_populate_lock`) to
      prevent race conditions and reduce database load during concurrent requests.
    - `find_manufacturers_in_description`: Identifies which manufacturer names from the
      cached list appear as whole words in a given product description, applying logic
      to prefer longer, more specific matches when overlaps occur.
    - `update_manufacturer_dict`: Augments the cached manufacturer dictionary with any
      new manufacturer names found in parts data retrieved during an analysis.
- Description Analysis Pipeline (`analyze_description`): This is a central asynchronous
  function that:
    1. Cleans the input description.
    2. Generates a vector embedding for the cleaned description using an LLM.
    3. Extracts potential part numbers using `PartNumberExtractor`.
    4. Fetches candidate parts data based on these part numbers from a primary data
       source ("database" or "azure_search") with a fallback mechanism to the other
       source if the primary fails (`get_parts_data_with_fallback`).
    5. Retrieves and updates manufacturer data.
    6. Identifies manufacturers mentioned in the description.
    7. Enriches the retrieved parts data with cleaned manufacturer names, match types,
       and calculates cosine similarity between the input description's embedding and
       the item descriptions' embeddings (either pre-computed or calculated on-the-fly).
    8. Filters enriched data to keep only the highest similarity row per ItemID.
- Data Source Abstraction: The `analyze_description` and `get_parts_data_with_fallback`
  functions support fetching data from either a SQL database or Azure AI Search,
  providing flexibility in data retrieval strategies.
- Asynchronous Operations: Core data fetching operations (`read_manufacturer_data`,
  `get_parts_data`, `analyze_description`, `get_parts_data_with_fallback`, and
  Azure Search calls) are asynchronous (`async def`) to handle I/O-bound tasks
  efficiently.

The module integrates with other components like `sql_utils` for database interaction,
`llm` for embedding generation and similarity calculation, `sdp` for database session
management, and `azure_search_utils` for Azure AI Search interactions.
"""

import re
import threading
import time
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Pattern, Set, Tuple

import numpy as np
import pandas as pd

from azure_search_utils import AzureSearchUtils
from constants import MfrNameMatchType, MfrRelationshipType
from llm import LLM
from logger import logger
from sdp import SDP
from sql_utils import get_manufacturer_data, get_parts_data
from utils import clean_description, remove_separators

# A reasonable upper limit for how many words a manufacturer name can have.
# Prevents excessive processing on very long descriptions.
MAX_WORDS_IN_MFR_NAME = 8

# --- Module-level cache storage and lock ---
_manufacturer_data_cache = {
    "data": None,
    "expiry_time": 0.0,  # Using time.monotonic() for expiry
    "ttl_seconds": 3600,  # Cache TTL: 1 hour (3600 seconds)
}
_manufacturer_data_populate_lock = threading.Lock()  # To ensure only one thread populates


class PartNumberExtractor:
    """
    Extracts potential manufacturer part numbers from text using a series of
    predefined regex rulesets and exclusion patterns.
    """

    def __init__(self) -> None:
        """
        Initializes the PartNumberExtractor by compiling regex patterns for
        various part number rulesets and exclusion criteria.
        """
        raw_exclusions = [
            # Sizes
            r"\b\d{1,4}-?(?:FT|IN)\b",  # e.g., 100FT, 14FT
            r"\b\d+(?:-\d+)?/\d+(?:-)?IN\b",  # e.g., 1-1/4-IN, 1/2IN
            r"\b\d{1,2}-\d{1,2}/\d{1,2}\b",  # Fractions like 2-1/8, 1-5/8
            r"\b\d{1,2}-\d{1,2}\b",  # e.g., 12-23
            r"\b\d{1,2}/\d{1,2}\b",  # e.g., 12/34
            r"\b\d-?PLY\b",  # e.g., 1-PLY, 2PLY
            r"\b\d{1,3}X\d{1,3}\b",  # e.g., 24X32
            r"\b(?:\d{1}/\d{1}|\d{1,3})(?:IN|FT)X\d{1,3}(?:IN|FT)\b",  # e.g., 1/2INX600FT, 2INX600FT
            r"\b\d{1,2}X\d{1,2}IN\b",  # e.g., 10X13IN
            r"\b(?:\d{1}/\d{1}|\d{1,3})X\d{1,3}FT\b",  # e.g., 3/4X66FT, 4X66FT
            r"\b\d{1,3}X\d{1,3}X(?:\d{1}/\d{1}|\d{1,3})\b",  # e.g., 8X40X8, 2X4X8, 4X8X1/2
            # Counts
            r"\b\d{1,3}[-/]?(?:PC|PACK|PK|CT|MIL|BX|BOX)\b",  # e.g., 1-PC, 25PACK, 100/PK, 200CT, 1MIL
            # Dates and years
            r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])/((?:20)?\d{2})\b",  # Matches dates like 3/20/25, 03/20/2025
            r"\b((?:20)?\d{2})/(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])\b",  # Matches 2025/3/20 or 25/3/20
            r"\b20(2[0-9]|30)\b",  # Years 2020-2030
            # Electrical values
            r"\b\d{1,4}-?V\b|\b\d{1,4}-?A(?:MP)?\b|\b\d{1,4}-?W\b",  # e.g., 115V, 600V, 200-A, 400AMP, 5760W
            r"\bCAT\d\b",  # e.g., CAT6
        ]

        # Define patterns for each ruleset
        raw_rulesets = [
            # Ruleset 1: Numbers only (4-50 digits)
            r"(?:^|[^0-9])(\d{4,50})(?=$|[^0-9])",
            # Ruleset 2: Numbers and '-'
            r"(?:^|(?<![0-9-]))([-0-9]{4,50})(?![0-9-])",
            # Ruleset 3: Numbers and '/'
            r"(?:^|(?<![0-9/]))([/0-9]{4,50})(?![0-9/])",
            # Ruleset 4: Numbers and '-' or '/'
            r"(?:^|[^0-9/-])([0-9/-]{4,50})(?=$|[^0-9/-])",
            # Ruleset 5: Numbers and letters (capital only)
            r"(?:^|[^A-Z0-9])([A-Z0-9]{4,50})(?=[^A-Z0-9]|$)",
            # Ruleset 6: Numbers, letters, and '-'
            r"(?:^|[^A-Z0-9-])([A-Z0-9-]{4,50})(?![A-Z0-9-])",
            # Ruleset 7: Numbers, letters, and '/'
            r"(?:^|[^A-Z0-9/])([A-Z0-9/]{4,50})(?![A-Z0-9/])",
            # Ruleset 8: Numbers, letters, '-', and '/'
            r"(?:^|[^A-Z0-9/-])([A-Z0-9/-]{4,50})(?![A-Z0-9/-])",
            # Ruleset 9: Numbers and dots only
            r"(?:^|[^0-9\.])([0-9\.]{4,50})(?=[^0-9\.]|$)",
            # Ruleset 10: Numbers, letters, and dots
            r"(?:^|[^A-Z0-9\.])([A-Z0-9\.]{4,50})(?=[^A-Z0-9\.]|$)",
        ]

        # Compile all patterns
        self.exclusion_patterns = [re.compile(pattern) for pattern in raw_exclusions]
        self.ruleset_patterns = [re.compile(pattern) for pattern in raw_rulesets]

    def is_excluded(self, text: str) -> bool:
        """
        Checks if the given text matches any of the predefined exclusion patterns.

        Exclusion patterns are typically for common measurements, counts, dates,
        or electrical values that might otherwise be mistaken for part numbers.

        Args:
            text (str): The text string to check against exclusion patterns.

        Returns:
            bool: True if the text matches an exclusion pattern, False otherwise.
        """
        return any(pattern.fullmatch(text) is not None for pattern in self.exclusion_patterns)

    @staticmethod
    def effective_length(part_number: str) -> int:
        """
        Calculates the "effective length" of a potential part number string.

        The effective length is determined by:
        - Starting with the actual length of the string.
        - Deducting counts of common separators ('-', '/', '.').
        - Deducting for excessive repetitions of the same character (more than 2
          consecutive occurrences).
        - Adding bonus points if the part number contains multiple capital letters.

        This length is used as a factor in scoring the quality of a part number match.

        Args:
            part_number (str): The part number string.

        Returns:
            int: The calculated effective length. Returns 0 if part_number is None or empty.
        """
        if not part_number:
            return 0

        effective_length = len(part_number)

        # Deduct occurrences of '-' and '/' and '.'
        effective_length -= part_number.count("-") + part_number.count("/") + part_number.count(".")

        # Deduct excess repetitions
        i = 0
        while i < len(part_number):
            char = part_number[i]
            count = 1

            # Count consecutive occurrences
            j = i + 1
            while j < len(part_number) and part_number[j] == char:
                count += 1
                j += 1

            # Deduct if more than 2 consecutive occurrences
            if count > 2:
                effective_length -= count - 2

            # Move to the next different character
            i = j

        # Check capital letters
        letter_count = sum(1 for char in part_number if "A" <= char <= "Z")

        if letter_count >= 2:
            effective_length += 1

        if letter_count >= 4:
            effective_length += 1

        return effective_length

    def is_valid_match(self, match: str) -> bool:
        """
        Validates if a potential part number string is a valid match.

        A match is considered valid if:
        1. Its `effective_length` is 4 or more.
        2. It contains at least one digit.
        3. It does not match any of the exclusion patterns defined in `is_excluded`.

        Args:
            match (str): The potential part number string to validate.

        Returns:
            bool: True if the match is considered valid, False otherwise.
        """
        if self.effective_length(match) < 4:
            return False

        # Must contain at least one number
        if not any(char.isdigit() for char in match):
            return False

        # Check against exclusion patterns
        if self.is_excluded(match):
            return False

        return True

    def find_matches(self, pattern: Pattern[str], text: str) -> Set[str]:
        """
        Finds all occurrences of a given regex `pattern` that are
        validated by `is_valid_match`. This function now only returns the
        raw, valid matches, deferring the creation of variations to the caller.
        """
        matches = set()
        for match in pattern.finditer(text):
            # Extract match, strip leading/trailing separators, and validate it.
            match_text = match.group(1).strip("-/.")
            if self.is_valid_match(match_text):
                matches.add(match_text)
        return matches

    def extract_part_numbers(self, description: str) -> Tuple[List[str], List[str]]:
        """
        Extracts potential part numbers and identifies UIPNCs.

        This function now performs two key roles:
        1. It generates all possible variations of valid part numbers (with and
           without separators) for a comprehensive database lookup.
        2. It generates and filters a separate list of fully normalized candidates
           to produce the "Unique Independent Part Number Candidates" (UIPNCs),
           which are used for advanced confidence scoring.

        Args:
            description (str): The input text from which to extract part numbers.

        Returns:
            Tuple[List[str], List[str]]:
            - A list of all candidate part number variations.
            - A list of the identified UIPNC strings.
        """
        if not description:
            return [], []

        # --- Step 1: Gather all unique, raw, valid matches from all rulesets ---
        raw_valid_matches: Set[str] = set()
        for pattern in self.ruleset_patterns:
            raw_valid_matches.update(self.find_matches(pattern, description))

        # --- Step 2: Create variations for database lookup and collect normalized candidates ---
        all_match_variations: Set[str] = set()
        normalized_candidates: Set[str] = set()

        for raw_match in raw_valid_matches:
            # Add all variations for the main parts lookup
            all_match_variations.add(raw_match)
            all_match_variations.add(remove_separators(raw_match, remove_dot=False))

            # The fully normalized version is a candidate for UIPNC
            normalized_version = remove_separators(raw_match, remove_dot=True)
            all_match_variations.add(normalized_version)
            normalized_candidates.add(normalized_version)

        # --- Step 3: Filter normalized candidates to find UIPNCs ---
        # A UIPNC is a maximal-length string that isn't a substring of another candidate.
        uipnc_list: List[str] = []
        # Sort by length (descending) to ensure we process longer strings first.
        sorted_normalized = sorted(normalized_candidates, key=len, reverse=True)

        for candidate in sorted_normalized:
            # A candidate is a UIPNC if it's not a substring of any *already accepted* UIPNC.
            if not any(candidate in accepted_uipnc for accepted_uipnc in uipnc_list):
                uipnc_list.append(candidate)

        return list(all_match_variations), uipnc_list


_part_number_extractor_instance = PartNumberExtractor()


class SpecialCharIgnoringDict(dict):
    """
    A dictionary that normalizes its keys before any operation by removing
    a predefined set of special characters. It assumes keys are already
    in the correct case (e.g., uppercase).
    """

    # Define the characters you want to ignore in a key
    _chars_to_remove = "&-/\\_.,() "

    # For performance, create a translation table once.
    # This is faster than calling .replace() in a loop.
    _translation_table = str.maketrans("", "", _chars_to_remove)

    @staticmethod
    def normalize(key):
        """Applies the normalization logic to a given key."""
        if not isinstance(key, str):
            return key  # Don't normalize non-string keys

        # Use the pre-built translation table for efficiency
        return key.strip().upper().translate(SpecialCharIgnoringDict._translation_table)

    def __init__(self, *args, **kwargs):
        """Initializes the dictionary, ensuring all initial keys are normalized
        and handling extended data structures for new features."""
        self._extended_details: Dict[str, Dict[str, Any]] = {}
        super().__init__()
        initial_data = dict(*args, **kwargs)
        for key, value in initial_data.items():
            # --- Smart handling of dict values for new data structure ---
            normalized_key = self.normalize(key)
            if isinstance(value, dict) and "CleanName" in value:
                # This is the new extended data structure from read_manufacturer_data
                # e.g., {'CleanName': '...', 'ParentCleanName': '...', 'BeginningOnlyFlag': ...}
                clean_name = value["CleanName"]
                # 1. Store simple mapping (key -> CleanName string) for backward compatibility
                #    This ensures legacy code like `dict.get(unclean_name)` still works.
                super().__setitem__(normalized_key, clean_name)
                # 2. Store the full extended details internally for new features
                self._extended_details[normalized_key] = value
            else:
                # Standard population (e.g., UncleanName -> CleanName string)
                # This maintains compatibility with any existing simple mappings.
                super().__setitem__(normalized_key, value)

    def __setitem__(self, key, value):
        """Sets an item, storing it under the normalized key."""
        normalized_key = self.normalize(key)
        super().__setitem__(normalized_key, value)

    def __getitem__(self, key):
        """Gets an item by its normalized key."""
        normalized_key = self.normalize(key)
        return super().__getitem__(normalized_key)

    def __contains__(self, key):
        """Checks for the existence of a normalized key."""
        normalized_key = self.normalize(key)
        return super().__contains__(normalized_key)

    def get(self, key, default=None):
        """
        Overrides the default .get() method to use key normalization.
        """
        normalized_key = self.normalize(key)
        # Use the parent class's .get() method to perform the actual lookup
        # on the normalized key and correctly handle the default value.
        return super().get(normalized_key, default)

    def get_details(self, key, default=None) -> Optional[Dict[str, Any]]:
        """
        Retrieves the extended details dictionary for a given key.

        This method provides access to the full manufacturer information
        including ParentCleanName and BeginningOnlyFlag, intended for use
        by new code that requires the enhanced data structure.

        Args:
            key (str): The key (UncleanName or CleanName) to look up.
            default (Any, optional): The value to return if the key is not found
                                     or has no extended details. Defaults to None.

        Returns:
            Optional[Dict[str, Any]]: The dictionary containing extended details
                                     (CleanName, ParentCleanName, BeginningOnlyFlag),
                                     or the `default` value if the key is not found.

        Example:
            details = manufacturer_dict.get_details("SOME-UNCLEAN_NAME")
            if details:
                parent = details.get("ParentCleanName")
                is_beg_only = details.get("BeginningOnlyFlag", False)

            # With default
            details = manufacturer_dict.get_details("UNKNOWN_MFR", {})
        """
        normalized_key = self.normalize(key)
        return self._extended_details.get(normalized_key, default)

    def copy(self):
        """
        Creates a shallow copy of the dictionary, ensuring the copy is also
        an instance of SpecialCharIgnoringDict and retains extended details.

        --- UPDATE FOR EXTENDED DATA ---
        - The copy will have the same key-value mappings in its main dictionary part.
        - The copy will also have a copy of the internal `_extended_details` dictionary,
        preserving the extended information for all keys that had it.
        """
        new_instance = type(self)(self)  # This creates the instance AND populates main dict

        # Ensure the new instance also gets the extended details from the original
        new_instance._extended_details = self._extended_details.copy()

        return new_instance


def extract_part_numbers(description: str) -> Tuple[List[str], List[str]]:
    """
    Wrapper function that uses a single, pre-compiled instance of `PartNumberExtractor`
    to extract part numbers from a description string.

    Args:
        description (str): The input description text.

    Returns:
        Tuple[List[str], List[str]]:
        - A list of all candidate part number variations.
        - A list of the identified UIPNC strings.
    """
    return _part_number_extractor_instance.extract_part_numbers(description)


def _process_manufacturer_row(row: pd.Series, raw_manufacturer_dict: Dict[str, Dict[str, Any]]) -> None:
    """
    Helper function to process a single row of manufacturer data and update
    the raw dictionary for compatibility with the updated SpecialCharIgnoringDict.
    """
    unclean_name = row["UncleanName"]
    clean_name = row["CleanName"]
    parent_clean_name = row["ParentCompanyName"] if pd.notna(row["ParentCompanyName"]) and row["ParentCompanyName"] else None
    beginning_only_flag = row["BeginningOnlyFlag"]

    # Data structure to store for each key (this is the new extended data structure)
    details = {
        "CleanName": clean_name,
        "ParentCleanName": parent_clean_name,  # Store ParentCompanyName directly
        "BeginningOnlyFlag": beginning_only_flag,
    }

    # Map UncleanName to its details structure
    # The updated SpecialCharIgnoringDict.__init__ will handle storing the CleanName string
    # for backward compatibility and the details dict for get_details.
    if unclean_name:
        raw_manufacturer_dict[unclean_name] = details

    # # Map CleanName to its details structure
    # if clean_name and clean_name != unclean_name:
    #     raw_manufacturer_dict[clean_name] = details


def _ensure_clean_manufacturers_exist(manufacturer_dict: Dict[str, Dict[str, Any]], result_df: pd.DataFrame) -> None:
    """
    Ensures that every CleanName exists as a key.
    Uses the 'First-Win' strategy: if the key exists, it is NOT updated.
    """
    # 1. Drop duplicates to ensure we only look at unique CleanNames (keeping the first occurrence found)
    unique_df = result_df.drop_duplicates(subset=["CleanName"])

    # 2. Create dictionary of ONLY items that are currently missing from the main dict
    new_clean_entries = {
        clean_name: {"CleanName": clean_name, "ParentCleanName": parent_name, "BeginningOnlyFlag": False}
        for clean_name, parent_name in zip(unique_df["CleanName"], unique_df["ParentCompanyName"])
        if clean_name and clean_name not in manufacturer_dict
    }

    # 3. Bulk update (only adds the missing ones)
    manufacturer_dict.update(new_clean_entries)


def _ensure_parent_manufacturers_exist(manufacturer_dict: Dict[str, Dict[str, Any]], parent_names: pd.Series) -> None:
    """
    Ensures that every parent company name exists as a key in the manufacturer dictionary.

    This function iterates through unique parent company names. If a parent name is not
    already a key in the provided dictionary (i.e., it wasn't an UncleanName or
    CleanName itself), it's added with a default entry.

    Args:
        manufacturer_dict (Dict[str, Dict[str, Any]]): The dictionary of manufacturer
            data, which will be modified in place.
        parent_names (pd.Series): A pandas Series containing all parent company names
            from the original data.
    """
    # Create a dictionary containing only the parent companies that are missing.
    new_parents = {
        parent_name: {"CleanName": parent_name, "ParentCleanName": parent_name, "BeginningOnlyFlag": False}
        for parent_name in parent_names.dropna().unique()
        if parent_name and parent_name not in manufacturer_dict
    }

    # Update the main dictionary with the new entries.
    manufacturer_dict.update(new_parents)


async def read_manufacturer_data(sdp: SDP) -> SpecialCharIgnoringDict:
    """
    Asynchronously reads manufacturer data, processes it into a SpecialCharIgnoringDict,
    and caches it in memory with a TTL.
    The function uses a module-level cache (`_manufacturer_data_cache`) and a
    threading lock (`_manufacturer_data_populate_lock`) to ensure that the
    database is queried only once per TTL period, even with concurrent calls.
    If the cache is valid, it's returned immediately. Otherwise, the lock is
    acquired, the cache is re-checked (double-checked locking pattern), and if
    still invalid/expired, data is fetched from the database via `get_manufacturer_data`,
    processed, and stored in the cache.
    The processing involves:
    - Dropping rows with missing UncleanName or CleanName.
    - Normalizing all names (UncleanName, CleanName) to uppercase and stripping whitespace.
    - Filtering out empty names.
    - Converting AIMatchIndicator to a boolean BeginningOnlyFlag.
    - Creating a dictionary mapping both UncleanName and CleanName to a dictionary
      containing CleanName, ParentCleanName (as fetched), and BeginningOnlyFlag.
     - Ensuring every ParentCleanName also exists as a key via a helper function.
    - Wrapping the result in a SpecialCharIgnoringDict to enable flexible lookups.
    Args:
        sdp (SDP): An instance of the SDP (SQL Data Platform/Provider) class,
                   used for database access via `get_manufacturer_data`.
    Returns:
        SpecialCharIgnoringDict: An instance of a custom dictionary that maps
                                 manufacturer names (UncleanName or CleanName) to
                                 a dictionary of their details.
                                 It normalizes lookup keys by removing special
                                 characters (e.g., '-', '/', spaces).
                                 Returns an empty SpecialCharIgnoringDict if
                                 fetching or processing fails, or if the lock
                                 acquisition times out.
    """
    current_time = time.monotonic()
    # Optimistic check (no lock)
    if _manufacturer_data_cache["data"] is not None and current_time < _manufacturer_data_cache["expiry_time"]:
        return _manufacturer_data_cache["data"]
    # Acquire threading.Lock to handle cache population
    if _manufacturer_data_populate_lock.acquire(blocking=True, timeout=10.0):  # 10-second timeout to acquire lock
        try:
            # Double-check cache condition *inside* the lock
            current_time = time.monotonic()  # Re-fetch current time
            if _manufacturer_data_cache["data"] is not None and current_time < _manufacturer_data_cache["expiry_time"]:
                return _manufacturer_data_cache["data"]
            # If still no valid cache, then populate
            fetch_reason = "Cache empty"
            if _manufacturer_data_cache["data"] is not None:  # Implies it's stale
                fetch_reason = "Cache expired"
            # Calculate what the new expiry time will be, relative to wall clock time for logging readability
            # Note: current_time is monotonic. We use time.time() for human-readable future expiry.
            new_expiry_wall_time_readable = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(time.time() + _manufacturer_data_cache["ttl_seconds"])
            )
            logger.debug(
                f"Fetching manufacturer data from database ({fetch_reason}). "
                f"New cache expiry intended: {new_expiry_wall_time_readable}."
            )
            # --- Fetch data from SDP ---
            result_df = await get_manufacturer_data(sdp)  # Actual async DB call
            # --- Data processing starts ---
            if result_df.empty:
                logger.warning("Manufacturer data DataFrame is empty after fetch.")
                raw_manufacturer_dict = {}
            else:
                # --- Drop rows with missing critical columns (UncleanName, CleanName) ---
                result_df = result_df.dropna(subset=["UncleanName", "CleanName"]).copy()

                def normalize(name: Optional[str]) -> str:
                    return name.strip().upper() if name else ""

                # --- Normalize relevant columns ---
                result_df["UncleanName"] = result_df["UncleanName"].map(normalize)
                result_df["CleanName"] = result_df["CleanName"].map(normalize)
                result_df["ParentCompanyName"] = result_df["ParentCompanyName"].map(normalize)

                # --- Filter out empty names after normalization ---
                result_df = result_df[result_df["CleanName"] != ""]
                result_df = result_df[result_df["UncleanName"] != ""]

                # --- Convert AIMatchIndicator to boolean BeginningOnlyFlag ---
                def indicator_to_bool(indicator: Optional[str]) -> bool:
                    return indicator is not None and indicator.strip().upper() == "B"

                result_df["BeginningOnlyFlag"] = result_df["AIMatchIndicator"].map(indicator_to_bool)
                result_df = result_df.drop(columns=["AIMatchIndicator"])

                # 1. Create the raw dictionary mapping UncleanName/CleanName -> details
                raw_manufacturer_dict: Dict[str, Dict[str, Any]] = {}
                # --- Use helper function for row processing ---
                result_df.apply(lambda row: _process_manufacturer_row(row, raw_manufacturer_dict), axis=1)

                # 2. Ensure all CleanNames exist as keys (without overwriting)
                _ensure_clean_manufacturers_exist(raw_manufacturer_dict, result_df)

                # 3. Ensure all ParentCleanNames also exist as keys.
                _ensure_parent_manufacturers_exist(raw_manufacturer_dict, result_df["ParentCompanyName"])

            # 4. Create an instance of the new smart dictionary
            #    This is the object that will be cached and returned
            manufacturer_map = SpecialCharIgnoringDict(raw_manufacturer_dict)
            _manufacturer_data_cache["data"] = manufacturer_map  # Cache the smart dictionary
            _manufacturer_data_cache["expiry_time"] = current_time + _manufacturer_data_cache["ttl_seconds"]
            return manufacturer_map  # Return the smart dictionary
        except Exception as e:
            logger.error(f"Error populating manufacturer data cache: {str(e)}. Returning empty dictionary.", exc_info=True)
            return SpecialCharIgnoringDict()  # Return empty SpecialCharIgnoringDict instance
        finally:
            _manufacturer_data_populate_lock.release()
    else:
        logger.error(
            "Timed out acquiring lock for populating manufacturer data cache. "
            "This might indicate a stuck thread holding the lock or very high contention."
        )
        current_time = time.monotonic()
        if _manufacturer_data_cache["data"] is not None and current_time < _manufacturer_data_cache["expiry_time"]:
            return _manufacturer_data_cache["data"]
        elif _manufacturer_data_cache["data"] is not None:
            logger.warning("Returning stale manufacturer data due to lock acquisition timeout.")
            return _manufacturer_data_cache["data"]
        logger.warning("No cached/stale manufacturer data available after lock acquisition timeout. Returning empty dictionary.")
        return SpecialCharIgnoringDict()  # Return empty SpecialCharIgnoringDict instance


def mfr_eq_type(
    clean_name_1: str, clean_name_2: str, manufacturer_data_dict: SpecialCharIgnoringDict  # Takes the full dict as an argument
) -> MfrRelationshipType:
    """
    Determines the specific type of relationship between two manufacturers based on their CleanNames.
    Requires the full manufacturer data dictionary containing ParentCleanName information.

    Args:
        clean_name_1 (str): The CleanName of the first manufacturer.
        clean_name_2 (str): The CleanName of the second manufacturer.
        manufacturer_data_dict (SpecialCharIgnoringDict): The full manufacturer data dict
            containing details like ParentCleanName for each CleanName/UncleanName key.

    Returns:
        MfrRelationshipType: The type of relationship (DIRECT, PARENT, CHILD, SIBLING, NOT_EQUIVALENT).
    """
    if not clean_name_1 or not clean_name_2:
        return MfrRelationshipType.NOT_EQUIVALENT

    clean_name_1 = clean_name_1.strip().upper()
    clean_name_2 = clean_name_2.strip().upper()

    if clean_name_1 == clean_name_2:
        return MfrRelationshipType.DIRECT

    # Get details for both manufacturers from the dictionary
    details_1 = manufacturer_data_dict.get_details(clean_name_1, {})
    details_2 = manufacturer_data_dict.get_details(clean_name_2, {})

    # Ensure details are dictionaries
    if not isinstance(details_1, dict) or not isinstance(details_2, dict):
        logger.warning(f"mfr_eq_type: Unexpected data format for {clean_name_1} or {clean_name_2}. Expected dict.")
        return MfrRelationshipType.NOT_EQUIVALENT

    parent_clean_name_1 = details_1.get("ParentCleanName")
    parent_clean_name_2 = details_2.get("ParentCleanName")

    # 2. Parent/Child Relationship
    # Check if clean_name_1 is the parent of clean_name_2
    if parent_clean_name_2 and parent_clean_name_2 == clean_name_1:
        return MfrRelationshipType.PARENT
    # Check if clean_name_2 is the parent of clean_name_1
    if parent_clean_name_1 and parent_clean_name_1 == clean_name_2:
        return MfrRelationshipType.CHILD

    # 3. Sibling Relationship
    # Different CleanNames, same Parent (and parents exist and are not None/empty)
    if parent_clean_name_1 and parent_clean_name_2 and parent_clean_name_1 == parent_clean_name_2:
        return MfrRelationshipType.SIBLING

    # No recognized relationship
    return MfrRelationshipType.NOT_EQUIVALENT


class MatchType(Enum):
    """
    Defines the type of a manufacturer name match within a description.
    """

    WHOLE_WORD = 1  # e.g., "FESTO" in "A FESTO PART"
    EMBEDDED = 2  # e.g., "PS" in "PS-24V"
    INVALID = 3  # e.g., "PS" in "-PS-" or "AMP" in "CHAMPION"


def _classify_match_in_description(phrase: str, description: str, seps: Set[str]) -> Tuple[Optional[MatchType], int, int]:
    """
    Finds a phrase in a description using a flexible regex and classifies its boundaries.

    Args:
        phrase (str): The candidate phrase to search for (e.g., "LEVITON MANUFACTURING").
        description (str): The original, full description string.
        seps (Set[str]): A set of separator characters.

    Returns:
        A tuple containing the MatchType, start index, and end index of the best find.
        Returns (None, -1, -1) if no valid match is found.
    """
    # Create a flexible regex from the phrase (e.g., "A B" -> r"A[\W_]+B").
    parts = re.findall(r"\w+", phrase)
    if not parts:
        return None, -1, -1
    pattern = r"[\W_]+".join([re.escape(part) for part in parts])

    try:
        # Find the first occurrence of this pattern in the description.
        match = re.search(pattern, description, re.IGNORECASE)
        if not match:
            return None, -1, -1
    except re.error:
        return None, -1, -1  # Invalid regex pattern

    start, end = match.span()

    # --- Perform boundary classification ---
    char_before = description[start - 1] if start > 0 else " "
    char_after = description[end] if end < len(description) else " "

    if char_before.isalnum() or char_after.isalnum():
        return MatchType.INVALID, start, end

    CLEAR_BOUNDARIES = {
        "(",
        ")",
        ",",
        ".",
    }  # Treat parens, commas, and dots as "Clear" boundaries, effectively equivalent to spaces.
    left_is_clear = (start == 0) or char_before.isspace() or char_before in CLEAR_BOUNDARIES
    right_is_clear = (end == len(description)) or char_after.isspace() or char_after in CLEAR_BOUNDARIES

    left_is_separator = char_before in seps
    right_is_separator = char_after in seps

    if left_is_clear and right_is_clear:
        return MatchType.WHOLE_WORD, start, end
    elif (left_is_clear and right_is_separator) or (left_is_separator and right_is_clear):
        return MatchType.EMBEDDED, start, end
    elif left_is_separator and right_is_separator:
        return MatchType.INVALID, start, end

    return MatchType.INVALID, start, end


def find_manufacturers_in_description(manufacturer_dict: SpecialCharIgnoringDict, part_description: str) -> Dict[str, str]:
    """
    Identifies the best manufacturer candidates from a description using a multi-step
    process that preserves the original implementation's core logic while adding
    WHOLE_WORD/EMBEDDED classification.
    """
    if not part_description or not manufacturer_dict:
        return {}

    SEPARATORS = {"-", "/", "."}

    # Step 1: Find ALL potential manufacturer matches and classify them.
    # This loop structure is from the original implementation.
    description_cleaned = re.sub(r"[^\w\s]", " ", part_description)
    words = description_cleaned.split()
    num_words = len(words)
    all_found_matches = []  # Stores (matched_text, clean_name, match_type)

    for i in range(num_words):
        for j in range(1, MAX_WORDS_IN_MFR_NAME + 1):
            if i + j > num_words:
                break
            candidate_phrase = " ".join(words[i : i + j])

            manufacturer_details = manufacturer_dict.get_details(candidate_phrase)
            if manufacturer_details:
                clean_name = manufacturer_details.get("CleanName")
                beginning_only = manufacturer_details.get("BeginningOnlyFlag", False)

                # We found a potential manufacturer. Now, find its location in the
                # original string and classify its boundaries.
                match_type, start, end = _classify_match_in_description(candidate_phrase, part_description, SEPARATORS)

                if match_type and match_type != MatchType.INVALID:
                    # Check BeginningOnlyFlag constraint.
                    actual_text = part_description[start:end]
                    if beginning_only and not beginning_of_description(actual_text, part_description):
                        continue

                    all_found_matches.append((actual_text, clean_name, match_type))

    if not all_found_matches:
        return {}

    # Step 2: Filter out matches that are substrings of longer matches.
    # This logic is restored directly from the original implementation.
    all_found_matches.sort(key=lambda x: len(x[0]), reverse=True)

    filtered_matches = []
    processed_text = set()
    for text, clean_name, match_type in all_found_matches:
        if text not in processed_text:
            filtered_matches.append((text, clean_name, match_type))
            # Mark all substrings of this text as processed to avoid including them later.
            for i in range(len(text)):
                for j in range(i + 1, len(text) + 1):
                    processed_text.add(text[i:j])

    # Step 3: Implement the two-pass priority logic (WHOLE_WORD > EMBEDDED).
    whole_word_matches = [m for m in filtered_matches if m[2] == MatchType.WHOLE_WORD]

    if whole_word_matches:
        candidates_to_process = whole_word_matches
    else:
        candidates_to_process = [m for m in filtered_matches if m[2] == MatchType.EMBEDDED]

    if not candidates_to_process:
        return {}

    # Step 4: Group by CleanName and select the final winner.
    # This logic is also restored from the original implementation.
    grouped_matches = defaultdict(list)
    for matched_text, clean_name, _ in candidates_to_process:
        grouped_matches[clean_name].append(matched_text)

    final_matches: Dict[str, str] = {}
    for clean_name, candidates in grouped_matches.items():
        beginning_candidates = [text for text in candidates if beginning_of_description(text, part_description)]

        priority_candidates = beginning_candidates if beginning_candidates else candidates
        if priority_candidates:
            winner = max(priority_candidates, key=len)
            final_matches[clean_name] = winner

    return final_matches


async def analyze_description(
    sdp: SDP,
    llm: LLM,
    description: str,
    data_source: Literal["database", "azure_search"] = "azure_search",
    search_utils: Optional[AzureSearchUtils] = None,
    desc_embedding: Optional[np.ndarray] = None,  # Pre-calculated embedding
) -> Tuple[pd.DataFrame, Dict[str, str], SpecialCharIgnoringDict, Optional[np.ndarray]]:
    """
    Analyzes a product description to identify matching parts from a data source,
    find relevant manufacturers, and calculate description similarity scores.

    This is a core orchestration function that performs the following steps:
    1. Cleans the input `description`.
    2. Obtains a vector embedding for the cleaned description (uses `desc_embedding` if provided,
       otherwise generates it using the `llm`).
    3. Extracts potential part numbers from the cleaned description.
    4. Fetches parts data for these part numbers using `get_parts_data_with_fallback`,
       trying the specified `data_source` first.
    5. Retrieves manufacturer data (unclean to clean name mappings and details)
       using `read_manufacturer_data`.
    6. Updates this manufacturer dictionary with any new MfrNames found in the fetched `parts_df`.
    7. Identifies which manufacturers from the updated dictionary are present in the cleaned description.
    8. If parts data was found and embedding was successful, enriches `parts_df` by:
        - Adding 'CleanName' and 'UncleanName' columns.
        - Determining 'MfrNameMatchType'.
        - Calculating 'DescriptionSimilarity' between the input description's embedding
          and each part's item description embedding (either pre-computed from Azure Search
          if 'ItemDescription_vector' column exists, or calculated on-the-fly).
    9. Filters the enriched DataFrame to keep only the row with the highest
       'DescriptionSimilarity' for each unique 'ItemID'.

    Args:
        sdp (SDP): An SDP instance for database operations.
        llm (LLM): An LLM instance for generating embeddings and calculating similarity.
        description (str): The product description string to analyze.
        data_source (Literal["database", "azure_search"]): The primary data source to query
            for parts. Defaults to "database".
        search_utils (Optional[AzureSearchUtils]): An instance of `AzureSearchUtils`,
            required if `data_source` is "azure_search" or if it's a fallback option.
            Defaults to None.
        desc_embedding (Optional[np.ndarray]): A pre-calculated vector embedding for the
            *cleaned* `description`. If provided, the `llm.get_embeddings` call is skipped.
            If None, the embedding is generated internally. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, Dict[str, str], SpecialCharIgnoringDict, Optional[np.ndarray]]:
            - DataFrame: Enriched and filtered parts data. Empty if no parts found or
              if description was empty. Columns include "ItemID", "MfrPartNum", "MfrName",
              "UPC", "UNSPSC", "AKPartNum", "DescriptionID", "ItemDescription",
              "CleanName", "UncleanName", "MfrNameMatchType", "DescriptionSimilarity".
            - Dict[str, str]: Manufacturers found in the description (CleanName: UncleanName).
            - SpecialCharIgnoringDict: The full manufacturer data dictionary containing
              all known manufacturer variants and their details (CleanName, ParentCleanName,
              BeginningOnlyFlag). This is the dictionary returned by `read_manufacturer_data`.
            - Optional[np.ndarray]: The vector embedding of the input description. None if
              embedding generation failed or description was empty.
    """

    # Define standard empty DataFrame to avoid code duplication
    def create_empty_results_df():
        return pd.DataFrame(
            columns=[
                "ItemID",
                "MfrPartNum",
                "MfrName",
                "UPC",
                "UNSPSC",
                "AKPartNum",
                "ItemSourceName",
                "DescriptionID",
                "ItemDescription",
                "DescSourceName",
                "CleanName",
                "UncleanName",
                "MfrNameMatchType",
                "DescriptionSimilarity",
            ]
        )

    # Clean the description
    cleaned_description = clean_description(description)

    # Fetch manufacturer data early
    manufacturer_dict = await read_manufacturer_data(sdp)

    # Handle empty description case
    if not cleaned_description or cleaned_description == "":
        logger.warning("EXACT MATCHING: Description is empty after cleaning")
        return create_empty_results_df(), {}, manufacturer_dict, None

    # If desc_embedding is not provided, calculate it
    if desc_embedding is None and cleaned_description:
        try:
            desc_embedding = llm.get_embeddings(cleaned_description)[0]
        except Exception as e:
            logger.error(f"analyze_description: Error generating embedding: {str(e)}", exc_info=True)
            desc_embedding = None
    # If desc_embedding was provided (not None), it's used directly.
    # If cleaned_description was empty and desc_embedding was None, desc_embedding remains None.

    # Step 1: Extract part numbers from the description
    part_numbers, uipnc_list = extract_part_numbers(cleaned_description)

    # Step 2: Get parts data with fallback mechanism
    parts_df = await get_parts_data_with_fallback(sdp, part_numbers, data_source, search_utils)

    # Step 3: Update manufacturer dictionary with any new names from parts data (using the fetched dict)
    manufacturer_dict = update_manufacturer_dict(manufacturer_dict, parts_df)

    # Step 4: Find manufacturers mentioned in the description
    manufacturers_in_desc = find_manufacturers_in_description(manufacturer_dict, cleaned_description)

    if not parts_df.empty and desc_embedding is not None:
        # Step 5: Append data to parts DataFrame, pass the embedding
        result_df = enrich_parts_data(
            parts_df, manufacturer_dict, manufacturers_in_desc, cleaned_description, llm, desc_embedding
        )
        # Step 6: Keep only the row with highest DescriptionSimilarity for each ItemID
        result_df = keep_highest_similarity_rows(result_df)
    else:
        result_df = create_empty_results_df()

    return result_df, manufacturers_in_desc, manufacturer_dict, desc_embedding, uipnc_list


async def get_parts_data_with_fallback(
    sdp: SDP,
    part_numbers: List[str],
    primary_source: Literal["database", "azure_search"],
    search_utils: Optional[AzureSearchUtils] = None,
) -> pd.DataFrame:
    """
    Asynchronously retrieves parts data based on a list of part numbers, trying a
    `primary_source` first and falling back to a secondary source if the primary fails.

    - If `primary_source` is "database": Tries `get_parts_data` (SQL query). If it fails,
      and `search_utils` is provided, it falls back to
      `search_utils.get_parts_data_from_index` (Azure AI Search).
    - If `primary_source` is "azure_search": Tries `search_utils.get_parts_data_from_index`.
      If it fails, it falls back to `get_parts_data` (SQL query).

    Logs actions, successes, failures, and fallback attempts.

    Args:
        sdp (SDP): An SDP instance for database access.
        part_numbers (List[str]): A list of part numbers to search for.
        primary_source (Literal["database", "azure_search"]): The preferred data
                                                              source to query first.
        search_utils (Optional[AzureSearchUtils]): An instance of `AzureSearchUtils`.
            Required if "azure_search" is the primary or fallback source.

    Returns:
        pd.DataFrame: A DataFrame containing the parts data found. Returns an empty
                      DataFrame with a predefined schema if `part_numbers` is empty,
                      if `search_utils` is required but not provided for an Azure Search
                      operation, or if all attempted data sources fail.
                      Expected columns: "ItemID", "MfrPartNum", "MfrName", "UPC",
                      "UNSPSC", "AKPartNum", "DescriptionID", "ItemDescription".
    """

    def get_empty_dataframe() -> pd.DataFrame:
        columns = [
            "ItemID",
            "MfrPartNum",
            "MfrName",
            "UPC",
            "UNSPSC",
            "AKPartNum",
            "ItemSourceName",
            "DescriptionID",
            "ItemDescription",
            "DescSourceName",
        ]
        return pd.DataFrame(columns=columns)

    # Validate inputs
    if not part_numbers:
        return get_empty_dataframe()

    if primary_source == "azure_search" and search_utils is None:
        logger.warning(
            "AzureSearchUtils object is required for Azure Search but was not provided. Falling back to database source."
        )
        primary_source = "database"

    # Try primary source first
    if primary_source == "database":
        try:
            logger.info("Retrieving parts data from database")
            parts_df = await get_parts_data(sdp, part_numbers)
            logger.info(f"Database query completed successfully, returned {len(parts_df)} results")
            return parts_df
        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")

            # Only try fallback if search_utils is available
            if search_utils is not None:
                try:
                    logger.warning("Falling back to Azure AI Search after database failure")
                    parts_df = await search_utils.get_parts_data_from_index(part_numbers)
                    logger.info(f"Azure AI Search fallback completed, returned {len(parts_df)} results")
                    return parts_df
                except Exception as fallback_error:
                    logger.error(f"Azure AI Search fallback also failed: {str(fallback_error)}")
            else:
                logger.error("No fallback source available (search_utils is None)")

    else:  # primary_source == "azure_search"
        try:
            logger.info("Retrieving parts data from Azure AI Search")
            parts_df = await search_utils.get_parts_data_from_index(part_numbers)
            logger.info(f"Azure AI Search query completed successfully, returned {len(parts_df)} results")
            return parts_df
        except Exception as e:
            logger.error(f"Azure AI Search query failed: {str(e)}")

            try:
                logger.warning("Falling back to database after Azure AI Search failure")
                parts_df = await get_parts_data(sdp, part_numbers)
                logger.info(f"Database fallback completed, returned {len(parts_df)} results")
                return parts_df
            except Exception as fallback_error:
                logger.error(f"Database fallback also failed: {str(fallback_error)}")

    # If we get here, both primary and fallback sources failed
    logger.error("All data sources failed, returning empty DataFrame")
    return get_empty_dataframe()


def update_manufacturer_dict(manufacturer_dict: Dict[str, str], parts_df: pd.DataFrame) -> SpecialCharIgnoringDict:
    """
    Updates a given manufacturer dictionary with new manufacturer names found in a
    DataFrame of parts data.

    It iterates through unique 'MfrName' values in `parts_df`. If a manufacturer name
    is not already a key in `manufacturer_dict`, it's added with itself as the value.
    The function creates a copy of the original dictionary only if changes are needed,
    to avoid unnecessary object creation.

    Args:
        manufacturer_dict (Dict[str, str]): The initial dictionary mapping unclean
                                            manufacturer names to clean names.
        parts_df (pd.DataFrame): DataFrame containing parts data, expected to have
                                 an "MfrName" column.

    Returns:
        Dict[str, str]: The updated manufacturer dictionary. If `parts_df` is empty
                        or contains no new manufacturer names, the original
                        `manufacturer_dict` is returned.
    """
    if parts_df.empty:
        return manufacturer_dict

    # Find all new manufacturer names
    new_names = [mfr_name for mfr_name in parts_df["MfrName"].unique() if mfr_name and mfr_name not in manufacturer_dict]
    if not new_names:
        # No changes needed; return original
        return manufacturer_dict

    # Copy only if new names need to be added
    updated_dict = manufacturer_dict.copy()
    for mfr_name in new_names:
        updated_dict[mfr_name] = mfr_name

    return updated_dict


def determine_manufacturer_match_type(
    has_been_linked: bool,
    equivalent_manufacturers_found_count: int,  # Number of found manufacturers equivalent to the part
    total_manufacturers_found_count: int,  # Total number of distinct manufacturers found in description
) -> MfrNameMatchType:
    """
    Determines the `MfrNameMatchType` based on the linking outcome and the nature
    of manufacturers found in the description, considering manufacturer relationships.

    This function refines the logic to differentiate between cases where multiple
    found manufacturer names are all related to the matched part (a single conceptual
    group) versus cases where unrelated manufacturers are also present.

    Args:
        has_been_linked (bool): True if the part was successfully linked to one of
                                the manufacturers found in the description.
        equivalent_manufacturers_found_count (int): The number of distinct manufacturers found in the
                                description that are equivalent (Direct, Parent, Child,
                                Sibling) to the part's own CleanName.
        total_manufacturers_found_count (int): The total number of distinct manufacturers identified
                           in the input description.

    Returns:
        MfrNameMatchType:
        - `SINGLE_MATCH`: A specific manufacturer for the part was found, and all
          manufacturers found in the description are equivalent to the part
          (representing a single conceptual group/entity).
        - `MULTIPLE_MATCHES_ONE_VALID`: A specific manufacturer for the part was found,
          but the description also contained other manufacturers that are either
          not equivalent or represent a distinct group.
        - `NO_VALID_MATCHES`: No specific manufacturer for this part was found,
          but manufacturers were present in the description.
        - `NO_MANUFACTURERS_FOUND`: No manufacturers were identified in the description
          at all.
    """
    if total_manufacturers_found_count == 0:
        return MfrNameMatchType.NO_MANUFACTURERS_FOUND

    if has_been_linked:
        # A manufacturer was successfully linked to this part
        if equivalent_manufacturers_found_count == total_manufacturers_found_count:
            # All found manufacturers are equivalent to this part - single conceptual group
            return MfrNameMatchType.SINGLE_MATCH
        else:  # equivalent_manufacturers_found_count < total_manufacturers_found_count
            # Some found manufacturers are not equivalent - multiple distinct entities
            return MfrNameMatchType.MULTIPLE_MATCHES_ONE_VALID
    else:  # not has_been_linked
        # No manufacturer was linked to this part
        if equivalent_manufacturers_found_count > 0:
            # But some equivalent manufacturers were found (conflicts for other parts, maybe)
            return MfrNameMatchType.NO_VALID_MATCHES
        else:  # equivalent_manufacturers_found_count == 0
            # No equivalent manufacturers found
            if total_manufacturers_found_count > 0:
                return MfrNameMatchType.NO_VALID_MATCHES
            # Redundant else, included for explicitness matching original logic
            return MfrNameMatchType.NO_MANUFACTURERS_FOUND


def enrich_parts_data(
    parts_df: pd.DataFrame,
    full_manufacturer_data_dict: SpecialCharIgnoringDict,
    manufacturers_in_desc: Dict[str, str],
    description: str,
    llm: LLM,
    desc_embedding: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Enriches a DataFrame of parts data with additional information, including
    manufacturer details and description similarity scores.
    - Adapts to the new structure of `full_manufacturer_data_dict` (SpecialCharIgnoringDict
      containing detailed manufacturer data dicts as values).
    - Correctly extracts 'CleanName' from the detailed data structure.
    - Uses `mfr_eq_type` to link parts to manufacturers found in the description,
      supporting parent/child/sibling relationships for the `UncleanName` assignment
      and `MfrNameMatchType` determination.
    - Prioritizes manufacturers found at the beginning of the description, then by name length.

    The enrichment process involves:
    1. Mapping 'MfrName' in `parts_df` to 'CleanName' using `full_manufacturer_data_dict`.
    2. Mapping the new 'CleanName' to 'UncleanName' using `manufacturers_in_desc`
       AND `mfr_eq_type` to find the most suitable equivalent manufacturer
       (Direct, Parent, Child, Sibling), prioritizing those at the beginning of the
       description, then the longest name.
    3. Determining and adding the 'MfrNameMatchType' for each part based on the
       presence of its 'UncleanName' and the nature of manufacturers found in the description,
       using `mfr_eq_type` and `determine_manufacturer_match_type`.
    4. Calculating 'DescriptionSimilarity':
        - If `desc_embedding` (for the input description) is not provided, it's generated.
        - Item description embeddings are retrieved:
            - If `parts_df` has an 'ItemDescription_vector' column (e.g., from Azure Search),
              those pre-computed embeddings are used.
            - Otherwise, embeddings are generated on-the-fly for each 'ItemDescription'.
        - Cosine similarity is computed between `desc_embedding` and each item's embedding.

    Args:
        parts_df (pd.DataFrame): DataFrame containing parts data. Expected columns:
                                 "MfrName", "ItemDescription". May optionally contain
                                 "ItemDescription_vector".
        full_manufacturer_data_dict (SpecialCharIgnoringDict): The full manufacturer data dictionary
            mapping all known MfrNames/UncleanNames to a dictionary of their details (CleanName,
            ParentCleanName, BeginningOnlyFlag). This is the dictionary returned by `read_manufacturer_data`.
        manufacturers_in_desc (Dict[str, str]): Dictionary of manufacturers specifically
                                                found in the input description
                                                ({CleanName_found: UncleanName_found_in_description}).
        description (str): The original product description text.
        llm (LLM): An LLM instance for generating embeddings and calculating similarity.
        desc_embedding (Optional[np.ndarray]): Pre-calculated vector embedding for the
            input `description`. If None, it will be generated. Defaults to None.

    Returns:
        pd.DataFrame: The enriched DataFrame with new columns: "CleanName", "UncleanName",
                      "MfrNameMatchType", and "DescriptionSimilarity".
    """

    # Helper to extract the 'CleanName' string from the manufacturer details dictionary,
    # providing a consistent way to handle the nested data structure.
    def get_clean_name(mfr_name: str) -> Optional[str]:
        mfr_details = full_manufacturer_data_dict.get_details(mfr_name)
        if isinstance(mfr_details, dict):
            return mfr_details.get("CleanName")
        elif isinstance(mfr_details, str):  # Fallback for older data structures
            return mfr_details
        else:
            return None

    # Step 1: Map the raw 'MfrName' from the parts data to its 'CleanName'.
    parts_df["CleanName"] = parts_df["MfrName"].apply(get_clean_name)

    # Define a helper lambda for sorting manufacturer candidates by length (descending).
    def sort_key_len_desc(candidates):
        return sorted(candidates, key=lambda x: len(x[0]), reverse=True)

    def enrich_row(row) -> pd.Series:
        """
        Processes a single candidate part (row) to determine its UncleanName
        and MfrNameMatchType based on complex relationship and conflict rules.
        """
        # --- Data Extraction for the Current Row ---
        part_clean_name = row["CleanName"]
        item_description_db = row.get("ItemDescription", "")

        # --- False-Positive Filtering ---
        # For this specific part, identify which manufacturers from the input description
        # are "valid" by filtering out any that are false positives (i.e., appear in this
        # part's own database description).
        valid_mfrs_for_row = {}
        if part_clean_name:
            valid_mfrs_for_row = {
                clean_name: unclean_name
                for clean_name, unclean_name in manufacturers_in_desc.items()
                # Keep a manufacturer if it meets either of these criteria:
                # 1. It IS equivalent to the part's own manufacturer (it's the match).
                # 2. It is NOT a false positive (it's a valid competitor).
                if (
                    mfr_eq_type(part_clean_name, clean_name, full_manufacturer_data_dict)
                    or not is_false_positive_manufacturer(unclean_name, item_description_db)
                )
            }

        total_valid_manufacturers_found_count = len(valid_mfrs_for_row)

        # --- Edge Case Handling ---
        # If the part has no CleanName or if no valid manufacturers were found after
        # filtering, we can determine the match type directly.
        if not part_clean_name or total_valid_manufacturers_found_count == 0:
            match_type_enum = determine_manufacturer_match_type(
                has_been_linked=False,
                equivalent_manufacturers_found_count=0,
                total_manufacturers_found_count=total_valid_manufacturers_found_count,
            )
            return pd.Series([None, match_type_enum.value], index=["UncleanName", "MfrNameMatchType"])

        # --- Find and Prioritize Equivalent Manufacturers ---
        # Identify all valid manufacturers that are equivalent (Direct, Parent, Child, Sibling)
        # to the current part's manufacturer.
        equivalent_candidates = []
        equivalent_mfr_count = 0
        for found_clean_name, found_unclean_name in valid_mfrs_for_row.items():
            if mfr_eq_type(part_clean_name, found_clean_name, full_manufacturer_data_dict):
                equivalent_mfr_count += 1
                equivalent_candidates.append((found_unclean_name, found_clean_name))

        # From the equivalent candidates, select the best one to be the 'UncleanName'.
        # The priority is: 1) Appears at the beginning of the description, 2) Longest name.
        unclean_name_to_assign = None
        if equivalent_candidates:
            beginning_candidates = [(u, c) for u, c in equivalent_candidates if beginning_of_description(u, description)]
            if beginning_candidates:
                # If any are at the beginning, the winner is the LONGEST of those.
                unclean_name_to_assign = sort_key_len_desc(beginning_candidates)[0][0]
            else:
                # Otherwise, the winner is the LONGEST of all equivalent candidates.
                unclean_name_to_assign = sort_key_len_desc(equivalent_candidates)[0][0]

        # --- Determine Final MfrNameMatchType ---
        # Use the validated counts (which exclude false positives) to get the correct match type.
        match_type_enum = determine_manufacturer_match_type(
            unclean_name_to_assign is not None,  # Was a link successfully made?
            equivalent_mfr_count,  # How many equivalents were found?
            total_valid_manufacturers_found_count,  # How many valid competitors were there?
        )

        return pd.Series([unclean_name_to_assign, match_type_enum.value], index=["UncleanName", "MfrNameMatchType"])

    # Step 2: Apply the enrichment logic to every row in the DataFrame.
    enrichment_results = parts_df.apply(enrich_row, axis=1)
    parts_df = pd.concat([parts_df, enrichment_results], axis=1)

    # Step 3: Calculate Description Similarity
    # This block computes the cosine similarity between the input description's vector
    # and each candidate part's description vector.
    if desc_embedding is None:
        desc_embedding = llm.get_embeddings(description)[0]

    # Use pre-computed embeddings from Azure Search if available; otherwise, compute them on the fly.
    if "ItemDescription_vector" in parts_df.columns:
        item_embeddings = parts_df["ItemDescription_vector"].tolist()
    else:
        item_descriptions = parts_df["ItemDescription"].tolist()
        item_embeddings = llm.get_embeddings(item_descriptions)

    parts_df["DescriptionSimilarity"] = [
        llm.cosine_similarity(desc_embedding, item_embedding) for item_embedding in item_embeddings
    ]

    return parts_df


def keep_highest_similarity_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a DataFrame to keep only the row with the highest 'DescriptionSimilarity'
    for each unique 'ItemID'.

    If multiple parts share the same 'ItemID' (e.g., different variations or data
    source entries), this function ensures that only the one most similar to the
    input description (based on 'DescriptionSimilarity') is retained.

    Args:
        df (pd.DataFrame): DataFrame containing parts data. Expected columns:
                           "ItemID", "DescriptionSimilarity".

    Returns:
        pd.DataFrame: A DataFrame where each "ItemID" appears at most once,
                      corresponding to the entry with the maximum
                      "DescriptionSimilarity" for that ItemID.
    """
    # Sort by ItemID and DescriptionSimilarity (descending)
    sorted_df = df.sort_values(by=["ItemID", "DescriptionSimilarity"], ascending=[True, False])

    # Keep first occurrence of each ItemID (which will be the one with highest similarity)
    result = sorted_df.drop_duplicates(subset=["ItemID"], keep="first")

    return result


def is_false_positive_manufacturer(manufacturer_name: Optional[str], item_description: Optional[str]) -> bool:
    """
    Checks if a manufacturer name is a "false positive" by determining if it appears
    as a whole word within a given item description. This is used to prevent a
    manufacturer from being incorrectly flagged as a "conflict" if its name is
    naturally part of the candidate part's description in the database.

    Args:
        manufacturer_name (Optional[str]): The manufacturer name from the input to check.
        item_description (Optional[str]): The database item description to search within.

    Returns:
        bool: True if the manufacturer_name is found within the item_description
              (and is thus a false positive), False otherwise.
    """
    if not manufacturer_name or not item_description:
        return False

    # Normalize both strings to handle separators consistently and ensure reliable
    # whole-word matching.
    name_to_find = re.sub(r"[^\w\s]", " ", manufacturer_name).strip()
    description_to_search = re.sub(r"[^\w\s]", " ", item_description)

    # If the name becomes empty after normalization (e.g., it was just "-"), it can't match.
    if not name_to_find:
        return False

    try:
        # The \b markers create word boundaries, ensuring "AMP" doesn't match "CHAMPION".
        pattern = rf"\b{re.escape(name_to_find)}\b"
        if re.search(pattern, description_to_search, re.IGNORECASE):
            return True  # It's a false positive.
    except re.error as e:
        # Log the error but treat it as a non-match to avoid breaking the flow.
        logger.warning(f"Regex error in is_false_positive_manufacturer for name '{name_to_find}': {e}")

    return False  # It's not a false positive.


def beginning_of_description(text_to_find: Optional[str], description: str) -> bool:
    """
    Checks if `text_to_find` appears at the very beginning of the `description` string
    as a whole word or phrase, handling variations in separators like spaces or hyphens.

    The match is case-insensitive. It constructs a flexible regex pattern to ensure
    that 'S STRUT' can match 'S-STRUT' at the start of a description, while still
    respecting word boundaries to prevent 'S' from matching 'STRUT'.

    Args:
        text_to_find (Optional[str]): The text string to search for at the beginning.
                                      If None or empty, returns False.
        description (str): The description string to check.

    Returns:
        bool: True if `text_to_find` is found as a whole-word prefix of `description`,
              False otherwise.
    """
    if not text_to_find:
        return False

    # Normalize the text_to_find by replacing all non-word characters with a space,
    # then escape it for regex safety. This makes the pattern robust to separator
    # differences (e.g., 'S-STRUT' vs 'S STRUT').
    text_pattern = re.escape(re.sub(r"[^\w]+", " ", text_to_find).strip())

    # Build the final regex pattern. Replace literal spaces in the pattern with a regex
    # that matches one or more non-word characters or whitespace ([\W_]+).
    # This allows 'S STRUT' to match 'S-STRUT', 'S.STRUT', etc.
    pattern_string = text_pattern.replace(r"\ ", r"[\W_]+")

    # Anchor to the start of the string (^) and ensure it's not followed by a
    # word character (?!\w) to form a complete word/phrase match.
    final_pattern = rf"^{pattern_string}(?!\w)"

    try:
        if re.search(final_pattern, description, re.IGNORECASE):
            return True
    except re.error as e:
        # This should be rare but provides robustness.
        logger.warning(f"Regex error in beginning_of_description. Text: '{text_to_find}', Pattern: '{final_pattern}', Error: {e}")

    return False
