"""
Module: matching_scores.py

Purpose:
This module is central to the invoice line item matching solution, focusing on
calculating and evaluating confidence scores for potential matches between input
invoice line item data (description, manufacturer) and catalog/database part data.
It implements a sophisticated scoring system based on various factors including
part number characteristics, manufacturer name matching, UNSPSC codes, and
description similarity. The module defines a pipeline of normalization, filtering,
and adjustment steps to identify the best possible match from a set of candidates.

High-Level Design:
- Configuration-Driven Scoring: Utilizes an external YAML configuration file
  (`confidences.yml`) to define weights, thresholds, and points for different
  aspects of the matching logic. This allows for fine-tuning the scoring behavior
  without code changes.
- Multi-Factor Confidence Calculation: Confidence scores are derived for part numbers,
  manufacturer names, and UNSPSC codes. These scores consider:
    - Part Number: Effective length, match type within the description (e.g., exact, partial).
    - Manufacturer Name: Presence, conflicts with other potential manufacturers in the
      input description, position in the description (beginning bonus/penalty).
    - Description Similarity: Pre-calculated similarity scores are used and normalized.
- Filtering Pipeline: Applies a series of filtering steps to a DataFrame of potential
  matches:
    1. `normalize_part_numbers`: Standardizes part numbers and sorts/deduplicates candidates.
    2. `filter_by_unclean_name`: Filters records based on the presence/absence of
       'UncleanName' for a given normalized part number.
    3. `filter_by_description_similarity`: Filters records based on description similarity
       thresholds, handling scenarios with and without 'UncleanName'.
    4. `filter_and_adjust_by_confidence`: Filters records based on overall confidence scores
       relative to the top score and applies adjustments.
- Score Adjustment: The `adjust_confidence_scores` function refines the confidence
  scores of the top candidate by considering the scores of competing candidates with
  different part numbers, manufacturer names, or UNSPSC codes.
- Best Match Selection: The main entry point `best_match_score` orchestrates the
  calculation and filtering pipeline to return the single best matching record.
- Helper Functions: Includes various helper functions for specific calculations, such as
  `get_part_number_points`, `get_manufacturer_name_points`, `normalize_similarity_score`,
  `are_unspsc_compatible`, `mpn_match_type`, and `beginning_of_description` for regex-based
  position checking.
- Manufacturer Name Processing: The `process_manufacturer_dict` function handles scenarios
  where multiple manufacturer names might be extracted from the input, selecting the most
  plausible one and assigning an initial manufacturer confidence score.

The module relies heavily on pandas DataFrames for data manipulation and processing of
candidate matches. It aims to provide a robust and configurable mechanism for determining
the most confident match for an invoice line item.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from constants import LocalFiles, MfrNameMatchType, MfrRelationshipType, StageNames, SubStageNames
from matching_utils import (
    PartNumberExtractor,
    SpecialCharIgnoringDict,
    beginning_of_description,
    is_false_positive_manufacturer,
    mfr_eq_type,
)
from utils import clean_description, load_yaml, remove_separators


def get_config() -> Dict[str, Any]:
    """
    Loads and returns the specific configuration for the 'complete_match' stage
    from the confidences YAML file.

    This function reads the configuration from `LocalFiles.CONFIDENCES_FILE` and
    navigates to the nested dictionary corresponding to the 'complete_match'
    stage and sub-stage.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters
                        for the complete match scoring logic.
    """
    conf_yaml = load_yaml(LocalFiles.CONFIDENCES_FILE)
    return conf_yaml[StageNames.COMPLETE_MATCH][SubStageNames.COMPLETE_MATCH]


def normalize_part_numbers(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes manufacturer part numbers, sorts the DataFrame, and removes duplicates.

    Normalization involves removing common separators from 'MfrPartNum' to create
    a 'NormalizedPartNum' column. The DataFrame is then sorted based on this
    normalized part number, 'CleanName', and various confidence scores in descending
    order. Duplicates based on 'NormalizedPartNum' and 'CleanName' are removed,
    keeping the first occurrence (which, due to sorting, has the highest scores).

    Args:
        result_df (pd.DataFrame): DataFrame containing potential matches.
                                  Expected columns: 'MfrPartNum', 'CleanName',
                                  'PartNumberConfidenceScore',
                                  'ManufacturerNameConfidenceScore',
                                  'UNSPSCConfidenceScore'.

    Returns:
        pd.DataFrame: The DataFrame with normalized part numbers, sorted, and
                      deduplicated.
    """
    # Normalize MfrPartNum for sorting and deduplication
    result_df["NormalizedPartNum"] = result_df["MfrPartNum"].apply(remove_separators)

    # Sort by normalized part number, clean name, and confidence scores
    result_df = result_df.sort_values(
        by=[
            "NormalizedPartNum",
            "CleanName",
            "PartNumberConfidenceScore",
            "ManufacturerNameConfidenceScore",
            "UNSPSCConfidenceScore",
        ],
        ascending=[True, True, False, False, False],
    )

    # Remove duplicates, keeping the first occurrence (highest confidence score)
    result_df = result_df.drop_duplicates(subset=["NormalizedPartNum", "CleanName"], keep="first")

    return result_df


def filter_by_unclean_name(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters records based on the presence of 'UncleanName' for identical 'NormalizedPartNum'.

    If a 'NormalizedPartNum' group contains at least one record where 'UncleanName'
    is populated, this function removes all records from that same group where
    'UncleanName' is empty or NaN. This prioritizes matches where a specific
    manufacturer name from the database ('UncleanName') is available.

    Args:
        result_df (pd.DataFrame): DataFrame to be filtered.
                                  Expected columns: 'NormalizedPartNum', 'UncleanName'.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if result_df.empty:
        return result_df

    # Identify part numbers that have at least one record with populated UncleanName
    part_nums_with_unclean_names = result_df.loc[~result_df["UncleanName"].isna(), "NormalizedPartNum"].unique()

    # Create a mask for rows to keep (either has populated UncleanName or has a part number not in our list)
    mask_to_drop = result_df["NormalizedPartNum"].isin(part_nums_with_unclean_names) & result_df["UncleanName"].isna()

    # Drop the rows that match our criteria
    if mask_to_drop.any():
        result_df = result_df[~mask_to_drop]

    return result_df


def filter_by_description_similarity(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters records based on 'DescriptionSimilarity' scores using configured thresholds.

    This function applies two main filtering scenarios based on 'DescriptionSimilarity':
    1.  For groups of items with the same 'NormalizedPartNum' where all records
        have an empty 'UncleanName': If any record in such a group has a
        'DescriptionSimilarity' above a high threshold, records in that same
        group with similarity below a low threshold are removed.
    2.  For groups of items with the same 'NormalizedPartNum' where at least one
        record has a populated 'UncleanName' AND its 'DescriptionSimilarity' is
        above a high threshold: Records in that same group with similarity below
        a low threshold are removed.

    Thresholds ('high_threshold', 'low_threshold') are loaded from the configuration.

    Args:
        result_df (pd.DataFrame): DataFrame to be filtered.
                                  Expected columns: 'NormalizedPartNum', 'UncleanName',
                                  'DescriptionSimilarity'.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if result_df.empty:
        return result_df

    # Load configuration
    config = get_config()
    high_threshold = config["DESCRIPTION_SIMILARITY"]["high_threshold"]
    low_threshold = config["DESCRIPTION_SIMILARITY"]["low_threshold"]

    # Make a copy to avoid modifying the original during iteration
    filtered_df = result_df.copy()

    def create_masks(df: pd.DataFrame) -> tuple:
        """Helper to generate aligned boolean masks"""
        has_unclean_name = (~df["UncleanName"].isna()) & (df["UncleanName"] != "")
        high_sim = df["DescriptionSimilarity"] > high_threshold
        low_sim = df["DescriptionSimilarity"] < low_threshold
        return has_unclean_name, high_sim, low_sim

    # Initial masks (pre-Scenario 1 processing)
    has_unclean_name, high_similarity, low_similarity = create_masks(filtered_df)

    # --- Scenario 1: Process part numbers where all records have empty UncleanName ---
    # Identify part numbers where all records have empty UncleanName
    part_nums_all_empty_mask = filtered_df.groupby("NormalizedPartNum")["UncleanName"].apply(
        lambda x: x.isna().all() or (x == "").all()
    )
    part_nums_all_empty = part_nums_all_empty_mask[part_nums_all_empty_mask].index.tolist()

    # Find part numbers with at least one high similarity record
    high_sim_part_nums_s1 = filtered_df[(filtered_df["NormalizedPartNum"].isin(part_nums_all_empty)) & high_similarity][
        "NormalizedPartNum"
    ].unique()

    # Drop low similarity records for these part numbers
    if len(high_sim_part_nums_s1) > 0:
        mask_to_drop_s1 = filtered_df["NormalizedPartNum"].isin(high_sim_part_nums_s1) & low_similarity
        filtered_df = filtered_df[~mask_to_drop_s1]

    # Refresh masks after DataFrame modification
    has_unclean_name, high_similarity, low_similarity = create_masks(filtered_df)

    # --- Scenario 2: Process part numbers where at least one record has populated UncleanName ---
    # Identify part numbers that have at least one record with both populated UncleanName AND high similarity
    high_sim_with_name_part_nums = filtered_df[has_unclean_name & high_similarity]["NormalizedPartNum"].unique()

    # Drop low similarity records for these part numbers
    if len(high_sim_with_name_part_nums) > 0:
        mask_to_drop_s2 = filtered_df["NormalizedPartNum"].isin(high_sim_with_name_part_nums) & low_similarity
        filtered_df = filtered_df[~mask_to_drop_s2]

    return filtered_df


def filter_by_part_number_substring(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out items where the part number is a substring of another item's
    part number within the same 'CleanName' group.

    This function groups the DataFrame by 'CleanName' and applies a filter to
    each group. The filter identifies and removes any row whose 'NormalizedPartNum'
    is a proper substring of a longer 'NormalizedPartNum' within the same group.

    Args:
        result_df (pd.DataFrame): The DataFrame to filter. It is expected to have
                                  'CleanName' and 'NormalizedPartNum' columns.

    Returns:
        pd.DataFrame: The filtered DataFrame, where shorter, substring-based
                      part number matches have been removed in favor of longer,
                      more complete ones within the same manufacturer group.
    """
    if result_df.empty or len(result_df) < 2:
        return result_df

    def filter_group(group: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a single group to find and remove substring part numbers.
        """
        # No need to process groups with only one item
        if len(group) < 2:
            return group

        # Get unique part numbers and sort them by length, longest first.
        # This provides a slight optimization for the substring check.
        part_nums = sorted(group["NormalizedPartNum"].unique(), key=len, reverse=True)

        # Identify part numbers that are proper substrings of any other part number in the group.
        # The 'any()' construct is efficient as it stops searching as soon as a match is found.
        pns_to_drop = {pn for pn in part_nums if any(pn in other_pn and pn != other_pn for other_pn in part_nums)}

        if not pns_to_drop:
            return group

        # Return the original group frame with the substring part numbers filtered out.
        return group[~group["NormalizedPartNum"].isin(pns_to_drop)]

    # Apply the filtering function to each 'CleanName' group.
    # .reset_index(drop=True) restores the original index structure.
    return result_df.groupby("CleanName", group_keys=False).apply(filter_group)


def filter_and_adjust_by_confidence(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters records based on relative part number confidence and adjusts scores.

    The function first sorts the DataFrame by confidence scores. It then filters out
    records whose 'PartNumberConfidenceScore' is below a certain factor of the top
    record's score (defined by 'top_score_relative_filter_factor' in config).
    After this initial filtering, it calls `adjust_confidence_scores` to refine the
    scores of the remaining top candidates. Finally, it checks if the (new) top record
    meets a minimum absolute confidence threshold ('min_confidence_threshold' from config)
    in at least one of its score categories (PartNumber, ManufacturerName, or UNSPSC).
    If not, an empty DataFrame is returned.

    Args:
        result_df (pd.DataFrame): DataFrame to be filtered and adjusted.
                                  Expected columns: 'PartNumberConfidenceScore',
                                  'ManufacturerNameConfidenceScore', 'UNSPSCConfidenceScore'.

    Returns:
        pd.DataFrame: The filtered and adjusted DataFrame, potentially empty if no
                      records meet the final confidence criteria.
    """
    if result_df.empty:
        return result_df

    # Load configuration
    config = get_config()
    min_confidence_threshold = config["min_confidence_threshold"]
    top_score_relative_filter_factor = config["top_score_relative_filter_factor"]

    # Sort by confidence scores to prepare for filtering/adjustment
    result_df = result_df.sort_values(
        by=["PartNumberConfidenceScore", "ManufacturerNameConfidenceScore", "UNSPSCConfidenceScore"],
        ascending=[False, False, False],
    )

    top_record_confidence = result_df.iloc[0]["PartNumberConfidenceScore"]

    # Remove records with confidence score less than the configured factor of top record's score
    result_df = result_df[result_df["PartNumberConfidenceScore"] > (top_record_confidence * top_score_relative_filter_factor)]

    # Adjust confidence scores based on second-best record
    result_df = adjust_confidence_scores(result_df)

    # After adjustment, check if the top record has at least one confidence score >= min_confidence_threshold
    # If not, return an empty DataFrame
    if not result_df.empty:
        top_record = result_df.iloc[0]
        has_sufficient_confidence = (
            top_record["PartNumberConfidenceScore"] >= min_confidence_threshold
            or top_record["ManufacturerNameConfidenceScore"] >= min_confidence_threshold
            or top_record["UNSPSCConfidenceScore"] >= min_confidence_threshold
        )

        if not has_sufficient_confidence:
            return pd.DataFrame(columns=result_df.columns)

    return result_df


def apply_final_score_capping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies final scoring rules, like capping scores for implicit matches.

    This helper function inspects the top record of the scored and filtered
    DataFrame. If the manufacturer match was implicit (i.e., the 'UncleanName'
    column is empty), it caps the 'ManufacturerNameConfidenceScore' according
    to a configured threshold.

    Args:
        df (pd.DataFrame): The DataFrame after all scoring and adjustments,
                           with the best candidate as the first row.

    Returns:
        pd.DataFrame: The DataFrame with the final capping rule applied to its
                      top record.
    """
    if df.empty:
        return df

    config = get_config()
    mfr_conf = config.get("MANUFACTURER_CONFIDENCE", {})
    implicit_match_score_cap = mfr_conf.get("implicit_match_score_cap", 90)  # Default to 90

    # Examine the UncleanName of the top-scoring record
    top_record_index = df.index[0]
    unclean_name = df.loc[top_record_index, "UncleanName"]

    # If the manufacturer match was implicit, apply the cap.
    if pd.isna(unclean_name) or not unclean_name:
        current_score = df.loc[top_record_index, "ManufacturerNameConfidenceScore"]
        # Use .loc for safe, direct assignment
        df.loc[top_record_index, "ManufacturerNameConfidenceScore"] = min(current_score, implicit_match_score_cap)

    return df


def check_and_set_verified_flag(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for a unique, confirmed manufacturer match from a verified source
    and adds an 'is_verified_flag' column.

    The 'is_verified_flag' is set to True only for the specific record where:
    1. 'UncleanName' is populated.
    2. There is exactly one such record in the DataFrame.
    3. The 'ItemSourceName' for that record is in 'VERIFIED_SOURCES'.
    """

    if result_df.empty:
        return result_df

    # 1. Initialize the column to False for all rows
    result_df["IsVerified"] = False

    # Load verified sources
    config = get_config()
    # Convert to set for faster lookup
    verified_sources = set(config.get("VERIFIED_SOURCES", []))

    # 2. Identify rows with a confirmed manufacturer match
    #    (UncleanName is not NA and not empty string)
    manufacturer_matches_mask = pd.notna(result_df["UncleanName"]) & (result_df["UncleanName"] != "")

    # 3. Check if there is EXACTLY ONE match
    if manufacturer_matches_mask.sum() == 1:
        # Get the index of the single matching record
        # We use .index[0] to get the label of the row
        match_index = result_df[manufacturer_matches_mask].index[0]

        # Get the source name for this specific row
        source_name = result_df.at[match_index, "ItemSourceName"]

        # 4. Check if the source is in the verified list
        if pd.notna(source_name) and source_name in verified_sources:
            # Set True ONLY for this specific row
            result_df.at[match_index, "IsVerified"] = True

    return result_df


def best_match_score(
    result_df: pd.DataFrame,
    manufacturers_in_desc: Dict[str, str],  # {CleanName: UncleanName_found_in_description}
    description: str,
    full_manufacturer_data_dict: SpecialCharIgnoringDict,  # The full dict from read_manufacturer_data
    uipnc_list: List[str],
) -> Optional[pd.DataFrame]:
    """
    Calculates confidence scores for potential matches and returns the top matching record.
    This function orchestrates the entire scoring and filtering pipeline:
    1. Calculates initial confidence scores using `calculate_initial_confidence_scores`,
       now providing the list of UIPNCs for contextual scoring.
    2. Normalizes part numbers and deduplicates using `normalize_part_numbers`.
    3. Applies sequential filters: `filter_by_unclean_name`, `filter_by_description_similarity`,
       and `filter_and_adjust_by_confidence`.
    4. Removes the temporary 'NormalizedPartNum' column.
    5. Returns the single top-scoring record, or None if no suitable match is found.

    Args:
        result_df (pd.DataFrame): DataFrame containing potential part matches from a
                                  database or catalog. Expected columns include
                                  'MfrPartNum', 'UncleanName', 'ItemDescription',
                                  'DescriptionSimilarity', 'UNSPSC', 'CleanName'.
        manufacturers_in_desc (Dict[str, str]): A dictionary mapping cleaned manufacturer
                                                names to unclean (original) manufacturer
                                                names extracted from the input description.
        description (str): The original input description string being analyzed.
        full_manufacturer_data_dict (SpecialCharIgnoringDict): The full manufacturer data
            dictionary from `read_manufacturer_data`, containing extended information
            like ParentCleanName and BeginningOnlyFlag. Required for `mfr_eq_type`.
        uipnc_list (List[str]): A list of "Unique Independent Part Number Candidates"
                                extracted from the input description. This is used by
                                the scoring function to apply contextual bonuses or
                                penalties (e.g., a bonus if there's only one
                                candidate, a penalty if there are many).

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the single best matching record
                                with all calculated confidence scores. Returns None if
                                the initial `result_df` is empty or if no record
                                survives the filtering pipeline.
    """
    if result_df.empty:
        return None
    # Clean description once if needed globally (e.g., for mpn_match_type)
    cleaned_item_description = clean_description(description)
    # Calculate initial confidence scores for each row
    result_df = calculate_initial_confidence_scores(
        result_df, manufacturers_in_desc, cleaned_item_description, full_manufacturer_data_dict, uipnc_list
    )

    # Normalize and perform initial sorting/deduplication
    result_df = normalize_part_numbers(result_df)
    # Apply filters in sequence
    result_df = filter_by_unclean_name(result_df)
    result_df = filter_by_description_similarity(result_df)
    result_df = filter_by_part_number_substring(result_df)

    # Continue with confidence-based filtering and adjustment
    result_df = filter_and_adjust_by_confidence(result_df)

    # Set the is_verified_flag based on the state of the DataFrame after filtering.
    result_df = check_and_set_verified_flag(result_df)

    # Apply final score capping rules using the helper function
    result_df = apply_final_score_capping(result_df)

    # Drop the temporary normalized part number column
    result_df = result_df.drop(columns=["NormalizedPartNum"])

    # Return only the top record
    return result_df.iloc[[0]] if not result_df.empty else None


def calculate_initial_confidence_scores(
    df: pd.DataFrame,
    manufacturers_in_desc: Dict[str, str],  # {CleanName: UncleanName_found_in_description}
    description: str,
    full_manufacturer_data_dict: SpecialCharIgnoringDict,  # The full dict from read_manufacturer_data with ParentCleanName etc.
    uipnc_list: List[str],
) -> pd.DataFrame:
    """
    Calculates initial confidence scores for each candidate part.

    This function computes scores for PartNumber, ManufacturerName, and UNSPSC.
    It has been enhanced to apply a more flexible, contextual score adjustment
    to the PartNumber points based on the list of "Unique Independent Part Number
    Candidates" (UIPNCs) found in the description:

    - A flat bonus is applied if the part number is a confirmed UIPNC.
    - A weighted penalty is applied based on the ambiguity of the description. The
      penalty is proportional to the ratio of the total effective length of all
      UIPNCs versus the effective length of the part number being scored.

    This replaces the previous discrete bonus/penalty and match_type systems with
    a more holistic approach to description ambiguity.
    """
    # --- Load Scoring Configuration ---
    config = get_config()
    part_conf = config["PART_NUMBER_CONFIDENCE"]
    mfr_conf = config["MANUFACTURER_CONFIDENCE"]
    unspsc_conf = config["UNSPSC_CONFIDENCE"]

    # Load weights for score calculations
    part_number_base_weight = part_conf["base_weight"]
    part_number_weight = part_conf["weight"]
    manufacturer_weight_for_part = part_conf["manufacturer_weight"]
    desc_sim_weight_for_part = part_conf["desc_sim_weight"]
    manufacturer_base_weight = mfr_conf["base_weight"]
    part_weight_for_manufacturer = mfr_conf["part_weight"]
    manufacturer_weight = mfr_conf["weight"]
    desc_sim_weight_for_manufacturer = mfr_conf["desc_sim_weight"]
    unspsc_base_weight = unspsc_conf["base_weight"]
    unspsc_part_weight = unspsc_conf["part_weight"]
    unspsc_manufacturer_weight = unspsc_conf["manufacturer_weight"]
    unspsc_desc_weight = unspsc_conf["desc_sim_weight"]

    # --- Load UIPNC scoring parameters ---
    uipnc_bonus = part_conf["uipnc_match_bonus"]
    uipnc_penalty_unit = part_conf["uipnc_penalty_unit"]

    # Initialize score columns
    df["PartNumberConfidenceScore"] = 0
    df["ManufacturerNameConfidenceScore"] = 0
    df["UNSPSCConfidenceScore"] = 0
    df["ManufacturerMatchRelationship"] = MfrRelationshipType.NOT_EQUIVALENT.name

    unclean_to_clean_in_desc = {v: k for k, v in manufacturers_in_desc.items()}

    # --- Pre-calculate sum of UIPNC lengths for efficiency ---
    sum_of_uipnc_lengths = sum(PartNumberExtractor.effective_length(u) for u in uipnc_list)

    # --- Process Each Candidate Part (Row) ---
    for idx, row in df.iterrows():
        # Conflict and relationship logic
        item_description_db = row.get("ItemDescription", "")
        unclean_name = row.get("UncleanName")
        clean_name_db = row.get("CleanName")
        relevant_conflicting_unclean_names_for_row = [
            unclean
            for unclean, clean in unclean_to_clean_in_desc.items()
            if not mfr_eq_type(clean_name_db, clean, full_manufacturer_data_dict)
            and not is_false_positive_manufacturer(unclean, item_description_db)
        ]
        clean_name_for_unclean = unclean_to_clean_in_desc.get(unclean_name)
        row_relationship_type = (
            mfr_eq_type(clean_name_db, clean_name_for_unclean, full_manufacturer_data_dict)
            if clean_name_db and clean_name_for_unclean
            else MfrRelationshipType.NOT_EQUIVALENT
        )
        df.at[idx, "ManufacturerMatchRelationship"] = row_relationship_type.name

        # --- Flexible Part Number Scoring Logic ---
        matched_part_eff_len = PartNumberExtractor.effective_length(row["MfrPartNum"])

        # 1. Calculate base points from effective length only
        part_number_points = get_part_number_points(matched_part_eff_len)

        # 2. Calculate Bonus
        normalized_part_num = remove_separators(row["MfrPartNum"], remove_dot=True)
        bonus = uipnc_bonus if normalized_part_num in uipnc_list else 0

        # 3. Calculate Penalty
        penalty_factor = (sum_of_uipnc_lengths / matched_part_eff_len) - 1
        penalty = round(max(0, penalty_factor) * uipnc_penalty_unit)

        # 4. Apply adjustment and cap the final points between 0 and 100
        adjusted_points = part_number_points + bonus - penalty
        part_number_points = max(0, min(100, adjusted_points))

        # Other score components
        desc_sim_points = normalize_similarity_score(row["DescriptionSimilarity"], config["DESCRIPTION_SIMILARITY"]["mid_point"])
        manufacturer_name_points = get_manufacturer_name_points(
            unclean_name, relevant_conflicting_unclean_names_for_row, description
        )

        # Score total calculations
        part_number_total = (
            part_number_base_weight
            + part_number_weight * part_number_points
            + manufacturer_weight_for_part * manufacturer_name_points
            + desc_sim_weight_for_part * desc_sim_points
        )
        df.at[idx, "PartNumberConfidenceScore"] = round(max(0, part_number_total))
        manufacturer_total = (
            manufacturer_base_weight
            + part_weight_for_manufacturer * part_number_points
            + manufacturer_weight * manufacturer_name_points
            + desc_sim_weight_for_manufacturer * desc_sim_points
        )
        df.at[idx, "ManufacturerNameConfidenceScore"] = round(max(0, manufacturer_total))
        if pd.notna(row["UNSPSC"]) and row["UNSPSC"]:
            unspsc_total = (
                unspsc_base_weight
                + unspsc_part_weight * part_number_points
                + unspsc_manufacturer_weight * manufacturer_name_points
                + unspsc_desc_weight * desc_sim_points
            )
            df.at[idx, "UNSPSCConfidenceScore"] = round(max(0, unspsc_total))

    return df


def adjust_confidence_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts the confidence scores of the top record in the DataFrame based on other
    competing records.

    This function modifies the 'PartNumberConfidenceScore',
    'ManufacturerNameConfidenceScore', and 'UNSPSCConfidenceScore' of the
    first record (df.iloc[0]) only.
    - Part Number Score: Reduced if there's another record with a different 'MfrPartNum'
      and a high 'PartNumberConfidenceScore'.
    - Manufacturer Score: Adjusted based on records with different 'CleanName'. If no
      such competing records, it gets a boost.
    - UNSPSC Score: Adjusted if there are records with "incompatible" 'UNSPSC' codes
      (checked by `are_unspsc_compatible`). If all 'UNSPSC' codes are the same or
      compatible, it gets a boost.

    The adjustment formulas aim to penalize the top score proportionally to the
    strength of a strong competitor. Scores are clamped between 0 and 100.

    Args:
        df (pd.DataFrame): DataFrame with initial confidence scores, sorted with the
                           best candidate at df.iloc[0]. Expected columns: 'MfrPartNum',
                           'CleanName', 'UNSPSC', 'PartNumberConfidenceScore',
                           'ManufacturerNameConfidenceScore', 'UNSPSCConfidenceScore'.

    Returns:
        pd.DataFrame: The DataFrame with adjusted confidence scores for its top record.
                      If the input DataFrame has only one row, scores are clamped and rounded.
                      If empty, returns the empty DataFrame.
    """
    if df.empty:
        return df

    # Precompute column indices once
    PART_COL = df.columns.get_loc("PartNumberConfidenceScore")
    MANU_COL = df.columns.get_loc("ManufacturerNameConfidenceScore")
    UNSPSC_COL = df.columns.get_loc("UNSPSCConfidenceScore")

    def update_scores(part, manu, unspsc):
        """Centralized score normalization and update logic"""
        # Clamp and round values
        clamped_part = round(max(0, min(100, part)))
        clamped_manu = round(max(0, min(100, manu)))
        clamped_unspsc = round(max(0, min(100, unspsc)))

        # Update DataFrame in-place
        df.iloc[0, PART_COL] = clamped_part
        df.iloc[0, MANU_COL] = clamped_manu
        df.iloc[0, UNSPSC_COL] = clamped_unspsc
        return df

    # Single record case
    if len(df) == 1:
        return update_scores(df.iloc[0, PART_COL], df.iloc[0, MANU_COL], df.iloc[0, UNSPSC_COL])

    # Multi-record logic
    top_record = df.iloc[0]

    # Get original confidence scores
    original_part_number_score = top_record["PartNumberConfidenceScore"]
    original_manufacturer_score = top_record["ManufacturerNameConfidenceScore"]
    original_unspsc_score = top_record["UNSPSCConfidenceScore"]

    # Adjust Part Number Confidence Score
    # Find record with highest PartNumberConfidenceScore where MfrPartNum is different
    different_part_records = df[df["MfrPartNum"] != top_record["MfrPartNum"]]
    if not different_part_records.empty:
        other_record = different_part_records.loc[different_part_records["PartNumberConfidenceScore"].idxmax()]
        other_part_number_score = other_record["PartNumberConfidenceScore"]
        if original_part_number_score > 0:
            adjusted_part_number = original_part_number_score - other_part_number_score * other_part_number_score / (
                2 * original_part_number_score
            )
        else:
            adjusted_part_number = 0
    else:
        adjusted_part_number = original_part_number_score

    # Adjust Manufacturer Name Confidence Score
    # Find record with highest ManufacturerNameConfidenceScore where CleanName is different
    different_mfr_records = df[df["CleanName"] != top_record["CleanName"]]
    if not different_mfr_records.empty:
        other_record = different_mfr_records.loc[different_mfr_records["ManufacturerNameConfidenceScore"].idxmax()]
        other_manufacturer_score = other_record["ManufacturerNameConfidenceScore"]
        if original_manufacturer_score > 0:
            adjusted_manufacturer = original_manufacturer_score - other_manufacturer_score * other_manufacturer_score / (
                2 * original_manufacturer_score
            )
        else:
            adjusted_manufacturer = 0
    else:
        adjusted_manufacturer = (original_manufacturer_score + 100) / 2

    # Adjust UNSPSC Confidence Score
    if pd.notna(top_record["UNSPSC"]) and top_record["UNSPSC"]:
        # Check if all UNSPSC values are the same (direct comparison)
        if len(df["UNSPSC"].unique()) == 1:
            adjusted_unspsc = (original_unspsc_score + 100) / 2
        else:
            # Find records with incompatible UNSPSC
            incompatible_unspsc_records = []
            for _, record in df.iloc[1:].iterrows():
                if pd.notna(record["UNSPSC"]) and record["UNSPSC"]:
                    if not are_unspsc_compatible(top_record["UNSPSC"], record["UNSPSC"]):
                        incompatible_unspsc_records.append(record)

            if incompatible_unspsc_records:
                # Find the record with highest UNSPSCConfidenceScore among incompatible ones
                other_record = max(incompatible_unspsc_records, key=lambda record: record["UNSPSCConfidenceScore"])
                other_unspsc_score = other_record["UNSPSCConfidenceScore"]
                if original_unspsc_score > 0:
                    adjusted_unspsc = original_unspsc_score - other_unspsc_score * other_unspsc_score / (
                        2 * original_unspsc_score
                    )
                else:
                    adjusted_unspsc = 0
            else:
                adjusted_unspsc = original_unspsc_score
    else:
        adjusted_unspsc = original_unspsc_score

    # Update the top record with adjusted scores
    return update_scores(adjusted_part_number, adjusted_manufacturer, adjusted_unspsc)


def get_part_number_points(effective_len: int) -> int:
    """
    Calculates points for part number confidence based purely on its effective length.

    Points are awarded based on the adjusted effective length according to predefined
    brackets in the configuration ('length_points').

    Args:
        effective_len (int): The effective character length of the part number.

    Returns:
        int: The calculated points for the part number.
    """
    # Load configuration
    config = get_config()
    length_points = config["PART_NUMBER_CONFIDENCE"]["length_points"]

    # Return points based on effective length brackets
    if effective_len < 4:
        return length_points["lt_4"]
    elif effective_len == 4:
        return length_points["eq_4"]
    elif effective_len == 5:
        return length_points["eq_5"]
    elif effective_len == 6:
        return length_points["eq_6"]
    elif effective_len == 7:
        return length_points["eq_7"]
    elif effective_len == 8:
        return length_points["eq_8"]
    elif effective_len == 9:
        return length_points["eq_9"]
    elif effective_len == 10:
        return length_points["eq_10"]
    else:  # > 10
        return length_points["gt_10"]


def get_manufacturer_name_points(
    unclean_name: Optional[str], relevant_conflicting_unclean_names: List[str], description: str
) -> int:
    """
    Calculates points for manufacturer name matching.

    This function determines points based on:
    - Whether the database record has an `unclean_name`.
    - The number and length of `relevant_conflicting_unclean_names` (manufacturer names
      from the input description that are different from `unclean_name` and not found
      within the DB record's own `ItemDescription`).
    - Whether `unclean_name` or any `relevant_conflicting_unclean_names` appear at the
      beginning of the input `description` (checked via `beginning_of_description`),
      applying bonuses or penalties accordingly.
    - The length of `unclean_name` itself (short names might get a penalty, long names a bonus).

    Points are derived from various nested settings in the configuration file under
    'MANUFACTURER_CONFIDENCE.match_points'. The final score is capped at 100.

    Args:
        unclean_name (Optional[str]): The manufacturer name from the database row
                                      being evaluated. Can be None or empty.
        relevant_conflicting_unclean_names (List[str]): List of unclean manufacturer
            names from the input description that are considered direct conflicts
            for this specific database row.
        description (str): The original input description string.

    Returns:
        int: Calculated points for manufacturer name match, between 0 and 100.
    """
    config = get_config()
    mfr_config = config["MANUFACTURER_CONFIDENCE"]["match_points"]
    beginning_bonus_penalty_amount = mfr_config["beginning_of_description_bonus"]

    points = 0
    beginning_adjustment = 0

    has_unclean_name = pd.notna(unclean_name) and bool(unclean_name) and unclean_name.strip() != ""

    # 1. Calculate beginning of description bonus or penalty
    if has_unclean_name and beginning_of_description(unclean_name, description):  # type: ignore
        beginning_adjustment += beginning_bonus_penalty_amount
    else:
        # Penalty check: iterate over relevant_conflicting_unclean_names.
        # These names are already confirmed to be:
        #   a) Not the 'unclean_name' of the current DB row.
        #   b) Not found in the ItemDescription of the current DB row.
        for conflicting_mfr_name in relevant_conflicting_unclean_names:
            if beginning_of_description(conflicting_mfr_name, description):
                beginning_adjustment -= beginning_bonus_penalty_amount
                break  # Apply penalty once for the first conflicting name found at the beginning

    # --- The rest of the logic for calculating points based on unclean_name presence and conflicts ---
    if has_unclean_name:
        primary_match_config = mfr_config["primary_match_found"]
        conflict_scores_config = mfr_config["primary_match_conflicts"]

        # Normalize conflicting names before checking length for consistent scoring
        # normalized_conflicts = [SpecialCharIgnoringDict._normalize_key(name) for name in relevant_conflicting_unclean_names]
        short_conflicts = [name for name in relevant_conflicting_unclean_names if len(name) < 4]
        long_conflicts = [name for name in relevant_conflicting_unclean_names if len(name) >= 4]

        num_short_conflicts = len(short_conflicts)
        num_long_conflicts = len(long_conflicts)

        base_points = 0
        if num_long_conflicts == 0:
            if num_short_conflicts == 0:
                base_points = primary_match_config["no_conflicts_base"]
            elif num_short_conflicts == 1:
                base_points = conflict_scores_config["one_short_conflict_score"]
            else:
                base_points = conflict_scores_config["multiple_short_conflicts_score"]
        elif num_long_conflicts == 1:
            if num_short_conflicts == 0:
                base_points = conflict_scores_config["one_long_conflict_score"]
            else:
                base_points = conflict_scores_config["one_long_one_or_more_short_conflicts_score"]
        else:
            base_points = conflict_scores_config["multiple_long_conflicts_score"]

        # Base the length adjustment on the NORMALIZED primary unclean name
        # normalized_unclean_name = SpecialCharIgnoringDict._normalize_key(unclean_name)
        name_length = len(unclean_name)
        length_adjustment_for_unclean_name = 0
        if name_length < 4:
            length_adjustment_for_unclean_name = primary_match_config["primary_short_penalty"]
        elif name_length > 6:
            length_adjustment_for_unclean_name = primary_match_config["primary_long_bonus"]

        points = base_points + length_adjustment_for_unclean_name + beginning_adjustment

    else:
        # --- No Primary Match (unclean_name from DB row is missing or empty) ---
        no_primary_config = mfr_config["no_primary_match"]

        other_count = len(relevant_conflicting_unclean_names)
        other_short_count = len([name for name in relevant_conflicting_unclean_names if len(name) < 4])

        if other_count == 0:
            points = no_primary_config["no_others_score"]
        else:
            long_others_count = other_count - other_short_count
            if long_others_count == 0:
                points = no_primary_config["only_short_others_score"]
            elif long_others_count == 1:
                points = no_primary_config["one_long_other_score"]
            else:
                points = no_primary_config["multiple_long_others_score"]

        points += beginning_adjustment  # Apply penalty if any conflicting name was at beginning

    return min(100, points)  # Ensure points don't exceed 100


def normalize_similarity_score(similarity: float, mid_similarity: float = 0.55) -> int:
    """
    Normalizes a raw similarity score (assumed to be between 0 and 1) to a 0-100 scale.

    The normalization uses a `mid_similarity` point. Scores below `mid_similarity`
    can result in negative normalized scores before rounding, but the function effectively
    scales the range [`mid_similarity`, 1.0] to [0, 100]. The result is rounded to
    the nearest integer.

    Args:
        similarity (float): The raw similarity score, typically between 0.0 and 1.0.
        mid_similarity (float): The raw similarity score that should map to 0
                                on the normalized scale. Defaults to 0.55.

    Returns:
        int: The normalized similarity score, rounded to the nearest integer.
             Typically, this aims for a 0-100 range where scores at or above
             `mid_similarity` are positive.
    """
    return round((similarity - mid_similarity) / (1 - mid_similarity) * 100)


def are_unspsc_compatible(unspsc1: str, unspsc2: str) -> bool:
    """
    Checks if two UNSPSC (United Nations Standard Products and Services Code)
    values are compatible.

    Compatibility is defined as:
    1. They share a common prefix of non-zero digits.
    2. After this common prefix, if they differ, the differing digits must lead to
       one UNSPSC code having only zeros for the rest of its length, while the other
       continues or also has zeros.
    3. If digits differ and neither is '0' at the first point of difference, they
       are considered incompatible.

    Example:
        "44121706" and "44120000" are compatible (common prefix "4412", then one has "1706", other has "0000").
        "42000000" and "44000000" are not compatible (differ at the second digit, neither is '0').
        "12345678" and "12340000" are compatible.
        "12345000" and "12340000" are compatible (common prefix "1234").

    Args:
        unspsc1 (str): The first UNSPSC code string.
        unspsc2 (str): The second UNSPSC code string.

    Returns:
        bool: True if the UNSPSC codes are compatible, False otherwise.
    """
    # Convert to strings if they're not already
    unspsc1 = str(unspsc1)
    unspsc2 = str(unspsc2)

    # Get the common prefix (before the first different character)
    common_prefix_len = 0
    for i in range(min(len(unspsc1), len(unspsc2))):
        if unspsc1[i] == unspsc2[i]:
            common_prefix_len += 1
        elif unspsc1[i] != "0" and unspsc2[i] != "0":
            # Different non-zero digits means incompatible
            return False
        else:
            break

    # Check if the remaining characters in both strings are all zeros
    suffix1 = unspsc1[common_prefix_len:]
    suffix2 = unspsc2[common_prefix_len:]

    return all(c == "0" for c in suffix1) or all(c == "0" for c in suffix2)


# Helper function for calculating base score based on length
def calculate_base_mfr_score(unclean_name: str, single_mfr_conf: Dict[str, int]) -> int:
    """
    Calculates a base manufacturer score based on the length of the `unclean_name`.

    This function is typically used when processing a single manufacturer extracted
    from the input description, before considering conflicts or other factors.
    It uses length-based scoring brackets defined in the 'single_mfr_conf'
    dictionary (part of the main configuration).

    Args:
        unclean_name (str): The manufacturer name whose length determines the score.
        single_mfr_conf (Dict[str, int]): A configuration dictionary containing
                                          keys like 'short_name', 'name_4_chars', etc.,
                                          with corresponding point values.

    Returns:
        int: The base score assigned based on the length of `unclean_name`.
    """
    name_length = len(unclean_name)
    if name_length < 4:
        return single_mfr_conf["short_name"]
    elif name_length == 4:
        return single_mfr_conf["name_4_chars"]
    elif name_length == 5:
        return single_mfr_conf["name_5_chars"]
    elif name_length == 6:
        return single_mfr_conf["name_6_chars"]
    elif name_length == 7:
        return single_mfr_conf["name_7_chars"]
    else:  # 8 characters or more
        return single_mfr_conf["name_8plus_chars"]


def process_manufacturer_dict(manufacturer_dict: Dict[str, str], description: str) -> Optional[pd.DataFrame]:
    """
    Processes a dictionary of manufacturer names extracted from an input description
    to determine a single "best" manufacturer and assign it a confidence score.

    This function handles cases with zero, one, or multiple potential manufacturers:
    - If `manufacturer_dict` is empty, returns None.
    - If one manufacturer: Calculates a base score using `calculate_base_mfr_score`.
    - If multiple manufacturers:
        - Prioritizes matches found at the beginning of the `description` (checked by
          `beginning_of_description`). If multiple beginning matches, the one with the
          longest `UncleanName` is chosen.
        - If no beginning matches, prioritizes the manufacturer with the longest `UncleanName`.
          If there's a tie in length for the top candidates, no manufacturer is chosen,
          and the score effectively remains 0.
        - If a manufacturer is chosen from multiple candidates, penalties are applied to its
          base score based on the number of other conflicting manufacturers.

    A bonus (`beginning_bonus`) is added if the finally chosen `UncleanName` is at the
    beginning of the `description`. The final score is clamped between 0 and 100.
    Returns a single-row DataFrame populated with the chosen manufacturer's details
    and its 'ManufacturerNameConfidenceScore'. Other scores are default (e.g., 0).

    Args:
        manufacturer_dict (Dict[str, str]): Dictionary of {CleanName: UncleanName}
                                            manufacturer names found in the input.
        description (str): The original input description string.

    Returns:
        Optional[pd.DataFrame]: A single-row DataFrame with schema matching other
                                parts of the system, containing the chosen manufacturer
                                and its confidence score. Returns None if
                                `manufacturer_dict` is empty.
    """
    if not manufacturer_dict:
        return None

    config = get_config()
    single_mfr_conf = config["SINGLE_MANUFACTURER_CONFIDENCE"]

    # Extract configuration values
    beginning_bonus = single_mfr_conf["beginning_of_description_bonus"]
    penalty_single_conflict = single_mfr_conf["penalty_for_single_conflict"]
    penalty_multiple_conflicts = single_mfr_conf["penalty_for_multiple_conflicts"]

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
        "CleanName",
        "UncleanName",
        "MfrNameMatchType",
        "DescriptionSimilarity",
        "PartNumberConfidenceScore",
        "ManufacturerNameConfidenceScore",
        "UNSPSCConfidenceScore",
    ]

    data = {
        "ItemID": None,
        "MfrPartNum": None,
        "MfrName": None,
        "UPC": None,
        "UNSPSC": None,
        "AKPartNum": None,
        "ItemSourceName": None,
        "DescriptionID": None,
        "ItemDescription": None,
        "DescSourceName": None,
        "CleanName": None,
        "UncleanName": None,
        "MfrNameMatchType": None,
        "DescriptionSimilarity": None,
        "PartNumberConfidenceScore": 0,
        "ManufacturerNameConfidenceScore": 0,
        "UNSPSCConfidenceScore": 0,
    }

    current_mfr_score = 0

    if len(manufacturer_dict) == 1:
        clean_name, unclean_name = next(iter(manufacturer_dict.items()))
        data["CleanName"] = clean_name
        data["UncleanName"] = unclean_name
        data["MfrNameMatchType"] = MfrNameMatchType.SINGLE_MATCH

        current_mfr_score = calculate_base_mfr_score(unclean_name, single_mfr_conf)

    elif len(manufacturer_dict) > 1:
        # Prioritize item matching beginning of description
        beginning_matches: List[Tuple[str, str]] = []
        for cn, un in manufacturer_dict.items():
            if un and beginning_of_description(un, description):  # Check if 'un' is not None or empty
                beginning_matches.append((cn, un))

        if beginning_matches:
            # If multiple items match the beginning, pick the one with the longest unclean_name
            beginning_matches.sort(key=lambda x: len(x[1]), reverse=True)
            # The first item after sorting is the chosen one
            clean_name, unclean_name = beginning_matches[0]
            data["CleanName"] = clean_name
            data["UncleanName"] = unclean_name
            data["MfrNameMatchType"] = MfrNameMatchType.MULTIPLE_MATCHES_ONE_VALID
        else:
            # Pick by length if no beginning-of-description match
            sorted_entries = sorted(manufacturer_dict.items(), key=lambda x: len(x[1]), reverse=True)

            # Check if the top two entries have the same length (for their unclean_name)
            if len(sorted_entries) > 1 and len(sorted_entries[0][1]) == len(sorted_entries[1][1]):
                # If tied by length, do nothing, score remains 0.
                # data["UncleanName"] remains None.
                pass
            else:
                # Take the entry with the longest unclean name
                clean_name, unclean_name = sorted_entries[0]
                data["CleanName"] = clean_name
                data["UncleanName"] = unclean_name
                data["MfrNameMatchType"] = MfrNameMatchType.MULTIPLE_MATCHES_ONE_VALID

        # If a manufacturer was chosen and UncleanName is populated:
        if data["UncleanName"]:
            # Calculate base score based on chosen unclean_name length
            current_mfr_score = calculate_base_mfr_score(data["UncleanName"], single_mfr_conf)

            # Apply penalty based on the total number of manufacturers (which implies number of conflicts)
            num_total_manufacturers = len(manufacturer_dict)
            if num_total_manufacturers == 2:  # This means 1 conflict
                current_mfr_score -= penalty_single_conflict
            elif num_total_manufacturers > 2:  # This means 2 or more conflicts
                current_mfr_score -= penalty_multiple_conflicts
            # No penalty if num_total_manufacturers is 1, but this block is for len(manufacturer_dict) > 1.

    # Apply bonus if UncleanName was determined and is at the beginning of description
    # This applies to single match case as well, and for multiple matches chosen by either method.
    if data["UncleanName"] and beginning_of_description(data["UncleanName"], description):
        current_mfr_score += beginning_bonus

    # Ensure score is within 0-100 range
    data["ManufacturerNameConfidenceScore"] = round(max(0, min(100, current_mfr_score)))

    # Create DataFrame with the populated data
    df = pd.DataFrame([data], columns=columns)

    return df
