"""
Module: semantic_matching.py

Purpose:
This module implements semantic matching logic, primarily designed to find relevant
product information (UNSPSC codes and manufacturer names) based on a given product
description or its vector embedding. It leverages Azure AI Search for performing
vector-based semantic searches and then applies a weighted scoring mechanism to
the search results to determine the most confident match for UNSPSC and manufacturer.
The module is configurable through a YAML file for various thresholds and weights,
allowing for tunable matching behavior.

High-Level Design:
- Configuration-Driven: Search parameters (similarity thresholds, number of results)
  and confidence calculation parameters (mid-points, exponential factors, minimum
  thresholds) are loaded from an external YAML configuration file.
- Semantic Search Orchestration (`semantic_match_by_description`):
    - Accepts either a textual description (which is then converted to an embedding
      using an LLM) or a pre-calculated embedding vector.
    - Performs an iterative vector search using `AzureSearchUtils` to fetch a
      set of distinct items that meet a similarity threshold (`_fetch_distinct_items`).
    - Optionally cleans manufacturer names found in search results using a mapping
      obtained via `matching_utils.read_manufacturer_data` if an SDP (database)
      connection is provided.
    - Calculates exponential weights for search results based on their similarity
      scores (`_calculate_weights`).
    - Processes the weighted results to determine the most likely manufacturer name
      and its confidence score (`_process_manufacturer_with_weights`).
    - Processes the weighted results to determine the most likely UNSPSC code and its
      confidence score, considering UNSPSC hierarchy (`_process_unspsc_with_weights`).
- Weighted Scoring:
    - Manufacturer names and UNSPSC codes are scored based on the sum of weights of
      search results they appear in, factored by the maximum search score for items
      containing them.
- UNSPSC Hierarchy Handling:
    - The UNSPSC processing logic generates parent (class, family) variants for each
      UNSPSC code found in search results.
    - It then selects the best UNSPSC variant by considering confidence scores at
      different hierarchy levels and applying configured thresholds and rules for
      preferring more specific or significantly more confident generic codes.
- Helper Functions: The module is structured with several private helper functions
  to encapsulate specific logic steps, such as embedding generation, fetching
  distinct search items, calculating weights, processing manufacturer/UNSPSC,
  and handling UNSPSC variants and hierarchy.
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from azure_search_utils import AzureSearchUtils
from constants import LocalFiles, StageNames, SubStageNames
from exceptions import InvoiceProcessingError
from llm import LLM
from logger import logger
from matching_utils import read_manufacturer_data
from sdp import SDP
from utils import load_yaml


def get_config() -> Dict[str, Any]:
    """
    Loads and returns the SEMANTIC_SEARCH configuration section from the
    confidences YAML file specified in `LocalFiles.CONFIDENCES_FILE`.

    This function navigates the loaded YAML structure to extract the specific
    dictionary corresponding to the 'SEMANTIC_SEARCH' stage and sub-stage.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters
                        for semantic search and matching logic.
    """
    conf_yaml = load_yaml(LocalFiles.CONFIDENCES_FILE)
    return conf_yaml[StageNames.SEMANTIC_SEARCH][SubStageNames.SEMANTIC_SEARCH]


async def semantic_match_by_description(
    description: Union[str, List[float]],
    azure_search_utils: AzureSearchUtils,
    sdp: Optional[SDP] = None,
    llm: Optional[LLM] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Performs a semantic search and returns both the best match summary and the
    top N raw search results.

    Returns:
        Tuple[Dict[str, Any], List[Dict[str, Any]]]:
            1. The best match summary (UNSPSC, MfrName, scores).
            2. A list of the top N raw search result dictionaries (including Part Number).
    """
    # Load configuration
    config = get_config()
    search_conf = config["SEARCH"]
    confidence_conf = config["CONFIDENCE"]

    # Get configuration parameters
    similarity_threshold = search_conf["similarity_threshold"]
    top_results = search_conf["top_results"]
    max_results = search_conf["max_results"]
    mid_point = config["WEIGHTS"]["mid_point"]
    exp_factor = config["WEIGHTS"]["exp_factor"]
    min_confidence_threshold = config["CONFIDENCE"]["min_confidence_threshold"]
    single_match_similarity_threshold = confidence_conf.get("single_match_similarity_threshold", 80)

    # Get RAG parameters (default to 0.85 if missing)
    rag_config = config.get("RAG_CONTEXT", {})
    min_example_similarity = rag_config.get("min_example_similarity", 0.85)

    # Step 1: Get embedding if description is a string
    embedding = _get_embedding(description, llm)

    # Step 2: Run iterative vector search to get distinct ItemIDs
    # UPDATED: Added "MfrPartNum" to select_fields
    results = _fetch_distinct_items(
        azure_search_utils=azure_search_utils,
        embedding=embedding,
        top_distinct_results=top_results,
        similarity_threshold=similarity_threshold,
        initial_fetch_size=max_results,
        select_fields=["ItemID", "DescriptionID", "MfrName", "UNSPSC", "ItemDescription", "MfrPartNum"],
    )

    # Initialize return data
    result_data = {}
    top_n_items = []

    # If no results found, return empty data
    if len(results) == 0:
        return result_data, top_n_items

    results["MfrName"] = results["MfrName"].apply(lambda x: x.strip() if isinstance(x, str) else x)
    results["UNSPSC"] = results["UNSPSC"].apply(lambda x: x.strip() if isinstance(x, str) else x)
    results["MfrPartNum"] = results["MfrPartNum"].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Initialize IsMfrClean flag to False by default
    results["IsMfrClean"] = False

    # If sdp is provided, clean manufacturer names in the results dataframe
    if sdp is not None:
        try:
            # Get manufacturer mapping data
            manufacturer_mapping = await read_manufacturer_data(sdp)

            # Track if any manufacturer name was cleaned
            mfr_cleaned = False

            # Update MfrName values and track if any were actually cleaned
            def clean_and_track(mfr_name):
                nonlocal mfr_cleaned
                if pd.notna(mfr_name) and mfr_name in manufacturer_mapping:
                    mfr_cleaned = True
                    return manufacturer_mapping[mfr_name]
                return mfr_name

            # Apply cleaning
            results["MfrName"] = results["MfrName"].apply(clean_and_track)

            # Set the flag based on whether any cleaning happened
            results["IsMfrClean"] = mfr_cleaned

            logger.debug(f"Applied manufacturer name cleaning using mapping data. Cleaned: {mfr_cleaned}")
        except Exception as e:
            logger.warning(f"Failed to apply manufacturer mapping: {str(e)}", exc_info=True)
            # Continue processing with original manufacturer names

    # Use helper to filter raw items for Cache (RAG Context)
    if not results.empty:
        top_n_items = _filter_results_for_rag_context(results, min_example_similarity)

    # Single result case - check if we need to apply the single match factor and similarity threshold
    is_single_match = len(results) == 1
    single_match_factor = confidence_conf["single_match_factor"]

    # For single match, check if it meets the specific similarity threshold
    if is_single_match and results["@search.score"].iloc[0] * 100 < single_match_similarity_threshold:
        return result_data, top_n_items

    # Calculate weights for each result based on similarity score
    results = _calculate_weights(results, mid_point, exp_factor)

    # Process manufacturer name
    mfr_results = _process_manufacturer_with_weights(results, min_confidence_threshold, is_single_match, single_match_factor)
    if mfr_results:
        result_data.update(mfr_results)

    # Process UNSPSC
    unspsc_results = _process_unspsc_with_weights(results, config, is_single_match, single_match_factor)
    if unspsc_results:
        result_data.update(unspsc_results)

    return result_data, top_n_items


def _filter_results_for_rag_context(results: pd.DataFrame, min_similarity: float) -> List[Dict[str, Any]]:
    """
    Filters search results for use as RAG context examples based on similarity threshold
    and diversity criteria.
    """
    top_n_items = []

    if results.empty:
        return top_n_items

    seen_combinations = set()

    for _, row in results.iterrows():
        # 1. Similarity Threshold
        # We assume scores are in the 0.0 - 1.0 range based on the config convention.
        score = row.get("@search.score", 0)

        if score < min_similarity:
            break

        # 2. Diversity Filter (Deduplicate by Mfr + UNSPSC)
        mfr = row.get("MfrName")
        unspsc = row.get("UNSPSC")

        # Normalize for keying
        mfr_key = str(mfr).strip().upper() if pd.notna(mfr) else "NONE"
        unspsc_key = str(unspsc).strip() if pd.notna(unspsc) else "NONE"
        key = (mfr_key, unspsc_key)

        if key in seen_combinations:
            continue

        seen_combinations.add(key)

        # Add to list (converting to dict, replacing NaNs with None)
        row_dict = row.where(pd.notnull(row), None).to_dict()
        top_n_items.append(row_dict)

        # Cap at 3 examples for RAG
        if len(top_n_items) >= 3:
            break

    return top_n_items


def _calculate_weights(results: pd.DataFrame, mid_point: float, exp_factor: float) -> pd.DataFrame:
    """
    Calculates an exponential weight for each search result in the DataFrame.

    The weight is based on the '@search.score' (similarity score) of the result,
    using the formula: `weight = exp(exp_factor * (score - mid_point))`.
    This gives higher weights to scores significantly above the `mid_point`.

    Args:
        results (pd.DataFrame): DataFrame containing search results. Must have
                                an '@search.score' column.
        mid_point (float): The similarity score threshold where scores start to
                           gain significant weight.
        exp_factor (float): An exponential factor that controls the steepness
                            of the weighting curve.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'weight' column.
                      A copy of the original DataFrame is modified.
    """
    # Create a copy to avoid modifying the original
    df = results.copy()

    # Calculate weight using the exponential formula
    df["weight"] = df["@search.score"].apply(lambda score: math.exp(exp_factor * (score - mid_point)))

    return df


def _process_manufacturer_with_weights(
    results: pd.DataFrame, min_confidence_threshold: float, is_single_match: bool, single_match_factor: float
) -> Dict[str, Any]:
    """
    Determines the most confident manufacturer name from weighted search results.

    It groups the search results by 'MfrName', calculates a confidence score for
    each manufacturer based on the sum of weights of its associated results and
    the maximum search score among them. The manufacturer with the highest
    confidence is chosen. If `is_single_match` is true, the confidence score is
    multiplied by `single_match_factor`.

    Args:
        results (pd.DataFrame): DataFrame of search results, including 'MfrName',
                                '@search.score', 'weight', and 'IsMfrClean' columns.
        min_confidence_threshold (float): The minimum confidence score required for
                                          a manufacturer to be returned.
        is_single_match (bool): Flag indicating if the overall search yielded only
                                a single distinct item.
        single_match_factor (float): A factor to adjust the confidence score if
                                     `is_single_match` is true.

    Returns:
        Dict[str, Any]: A dictionary containing "MfrName",
                        "ManufacturerNameConfidenceScore", and "IsMfrClean" for the
                        best manufacturer if its score exceeds the threshold.
                        Returns an empty dictionary otherwise.
    """
    # Skip processing if no MfrName in results or all values are NaN
    if "MfrName" not in results.columns or results["MfrName"].isna().all():
        return {}

    # Filter out rows with NaN manufacturer names
    valid_results = results.dropna(subset=["MfrName"])
    if valid_results.empty:
        return {}

    # Calculate total weight of all results
    total_weight = results["weight"].sum()

    # Group by manufacturer name and calculate confidence scores
    mfr_confidence = {}
    mfr_is_clean = {}  # Track IsMfrClean status for each manufacturer name

    for mfr_name, group in valid_results.groupby("MfrName"):
        # Skip empty manufacturer names
        if pd.isna(mfr_name) or not mfr_name:
            continue

        # Sum of weights for this manufacturer
        group_weight = group["weight"].sum()

        # Get maximum search score for this manufacturer
        max_search_score = group["@search.score"].max()

        # Calculate confidence score using max search score * weighted proportion
        confidence = max_search_score * (group_weight / total_weight) * 100

        # Store manufacturer name and its confidence score
        mfr_confidence[mfr_name] = confidence

        # Store the IsMfrClean status for this manufacturer
        # Any True value among rows with this manufacturer takes precedence
        mfr_is_clean[mfr_name] = group["IsMfrClean"].any()

    # If no valid manufacturer names found
    if not mfr_confidence:
        return {}

    # Find manufacturer with highest confidence
    best_mfr = max(mfr_confidence.items(), key=lambda x: x[1])
    mfr_name = best_mfr[0]
    confidence_score = best_mfr[1]

    # Apply single match factor if needed
    if is_single_match:
        confidence_score *= single_match_factor

    # Only return if confidence is high enough
    if confidence_score > min_confidence_threshold:
        return {
            "MfrName": mfr_name,
            "ManufacturerNameConfidenceScore": round(confidence_score),
            "IsMfrClean": mfr_is_clean[mfr_name],
        }

    return {}


def _process_unspsc_with_weights(
    results: pd.DataFrame, config: Dict[str, Any], is_single_match: bool, single_match_factor: float
) -> Dict[str, Any]:
    """
    Determines the most confident UNSPSC code from weighted search results,
    considering UNSPSC hierarchy.

    1. Generates UNSPSC variants (commodity, class, family levels) for all unique
       UNSPSCs in the results.
    2. Calculates a confidence score for each variant based on the weights of
       matching rows and their max search scores.
    3. Selects the best UNSPSC variant using `_select_best_unspsc_variant`, which
       applies hierarchical rules and configured thresholds.
    4. If `is_single_match` is true, the confidence score of the selected UNSPSC is
       multiplied by `single_match_factor`.

    Args:
        results (pd.DataFrame): DataFrame of search results, including 'UNSPSC',
                                '@search.score', and 'weight' columns.
        config (Dict[str, Any]): The semantic search configuration dictionary,
                                 used to access UNSPSC level thresholds and other
                                 UNSPSC-specific settings.
        is_single_match (bool): Flag indicating if the overall search yielded only
                                a single distinct item.
        single_match_factor (float): A factor to adjust the confidence score if
                                     `is_single_match` is true.

    Returns:
        Dict[str, Any]: A dictionary containing "UNSPSC" and "UNSPSCConfidenceScore"
                        for the best UNSPSC code if its score exceeds the configured
                        minimum confidence threshold. Returns an empty dictionary otherwise.
    """
    # Skip processing if no UNSPSC in results or all values are NaN
    if "UNSPSC" not in results.columns or results["UNSPSC"].isna().all():
        return {}

    # Filter out rows with NaN UNSPSC
    valid_results = results.dropna(subset=["UNSPSC"])
    if valid_results.empty:
        return {}

    min_confidence_threshold = config["CONFIDENCE"]["min_confidence_threshold"]
    level_thresholds = config["UNSPSC"]["level_thresholds"]
    generic_delta_percentage = config["UNSPSC"]["generic_delta_percentage"]

    # Calculate total weight of all results
    total_weight = results["weight"].sum()

    # Generate all UNSPSC variants
    unspsc_variants = _generate_unspsc_variants(valid_results)

    # Calculate confidence scores for each UNSPSC variant
    variant_confidence = {}
    for variant, code_level in unspsc_variants.items():
        # Get all rows that match this variant at the specified level
        matching_rows = _get_matching_rows_for_unspsc_variant(valid_results, variant, code_level)

        if not matching_rows.empty:
            # Sum of weights for this UNSPSC variant
            variant_weight = matching_rows["weight"].sum()

            # Get maximum search score for this UNSPSC variant
            max_search_score = matching_rows["@search.score"].max()

            # Calculate confidence score using max search score * weighted proportion
            confidence = max_search_score * (variant_weight / total_weight) * 100

            # Store UNSPSC variant and its confidence score
            variant_confidence[(variant, code_level)] = confidence

    # If no valid UNSPSC variants found
    if not variant_confidence:
        return {}

    # Find the best UNSPSC variant based on confidence scores and hierarchical rules
    best_unspsc = _select_best_unspsc_variant(variant_confidence, level_thresholds, generic_delta_percentage)

    # If no suitable UNSPSC found
    if not best_unspsc:
        return {}

    unspsc_code, confidence_score = best_unspsc

    # Apply single match factor if needed
    if is_single_match:
        confidence_score *= single_match_factor

    # Only return if confidence is high enough
    if confidence_score > min_confidence_threshold:
        return {"UNSPSC": unspsc_code, "UNSPSCConfidenceScore": round(confidence_score)}

    return {}


def _generate_unspsc_variants(results: pd.DataFrame) -> Dict[str, int]:
    """
    Generates all relevant UNSPSC variants (commodity, class, family) from the
    UNSPSC codes present in the search results.

    For each unique, valid 8-digit UNSPSC code found in the 'UNSPSC' column of
    the `results` DataFrame, it generates:
    - The original code (level 3, commodity).
    - The class-level code (first 6 digits + "00", level 2).
    - The family-level code (first 4 digits + "0000", level 1).

    Args:
        results (pd.DataFrame): DataFrame containing search results with an 'UNSPSC'
                                column.

    Returns:
        Dict[str, int]: A dictionary mapping each unique UNSPSC variant string to its
                        hierarchy level (3 for commodity, 2 for class, 1 for family).
    """
    variants = {}

    for unspsc in results["UNSPSC"].dropna().unique():
        if not unspsc or not isinstance(unspsc, str) or len(unspsc) < 8:
            continue

        # Original UNSPSC (Commodity level)
        variants[unspsc] = 3

        # Class level (last 2 digits are zeros)
        class_variant = unspsc[:-2] + "00"
        variants[class_variant] = 2

        # Family level (last 4 digits are zeros)
        family_variant = unspsc[:-4] + "0000"
        variants[family_variant] = 1

    return variants


def _get_matching_rows_for_unspsc_variant(results: pd.DataFrame, variant: str, level: int) -> pd.DataFrame:
    """
    Filters the results DataFrame to find rows that match a given UNSPSC variant
    at a specific hierarchy level.

    - Level 3 (Commodity): Matches rows where 'UNSPSC' is an exact match to `variant`.
    - Level 2 (Class): Matches rows where the first 6 digits of 'UNSPSC' match the
      first 6 digits of `variant`.
    - Level 1 (Family): Matches rows where the first 4 digits of 'UNSPSC' match the
      first 4 digits of `variant`.

    Args:
        results (pd.DataFrame): The DataFrame of search results with an 'UNSPSC' column.
        variant (str): The UNSPSC variant string to match against.
        level (int): The hierarchy level of the `variant` (3, 2, or 1).

    Returns:
        pd.DataFrame: A DataFrame containing only the rows that match the specified
                      UNSPSC variant at the given level. Returns an empty DataFrame
                      if the level is invalid.
    """
    if level == 3:  # Commodity level (exact match)
        return results[results["UNSPSC"] == variant]
    elif level == 2:  # Class level (first 6 digits match)
        prefix = variant[:6]
        return results[results["UNSPSC"].str.startswith(prefix, na=False)]
    elif level == 1:  # Family level (first 4 digits match)
        prefix = variant[:4]
        return results[results["UNSPSC"].str.startswith(prefix, na=False)]
    else:
        return pd.DataFrame()


def _select_best_unspsc_variant(
    variant_confidence: Dict[Tuple[str, int], float], level_thresholds: List[int], generic_delta_percentage: float
) -> Optional[Tuple[str, float]]:
    """
    Selects the best UNSPSC variant from a dictionary of variants and their
    confidence scores, applying hierarchical preference rules and thresholds.

    The selection logic prioritizes more specific UNSPSC levels (commodity > class > family)
    if their confidence scores meet configured `level_thresholds`. It also considers
    whether a more generic level (e.g., class) is "significantly better" (by
    `generic_delta_percentage`) than a more specific level (e.g., commodity)
    before choosing the generic one. If no variant meets any specific threshold criteria,
    the variant with the overall highest confidence score is chosen.

    Args:
        variant_confidence (Dict[Tuple[str, int], float]): A dictionary mapping
            (UNSPSC_variant_string, hierarchy_level) tuples to their calculated
            confidence scores.
        level_thresholds (List[int]): A list of confidence score thresholds, typically
            ordered for different levels of specificity (e.g., for commodity, class, family).
            The function iterates through these, applying them to corresponding levels.
        generic_delta_percentage (float): The percentage by which a more generic
            UNSPSC variant's confidence must exceed a more specific one's confidence
            to be preferred.

    Returns:
        Optional[Tuple[str, float]]: A tuple containing the selected UNSPSC code string
                                     and its confidence score. Returns None if
                                     `variant_confidence` is empty.
    """
    # Sort variants by hierarchy level (highest/most specific first) and confidence score (highest first)
    sorted_variants = sorted(
        variant_confidence.items(), key=lambda x: (-x[0][1], -x[1])  # Sort by -level (descending), then -confidence (descending)
    )

    if not sorted_variants:
        return None

    # Check specific levels against thresholds
    for threshold in level_thresholds:
        # First check Commodity level (level 3)
        commodity_variants = [(v, c) for ((v, l), c) in variant_confidence.items() if l == 3 and c >= threshold]
        if commodity_variants:
            best_commodity = max(commodity_variants, key=lambda x: x[1])
            return best_commodity

        # Then check Class level (level 2)
        class_variants = [(v, c) for ((v, l), c) in variant_confidence.items() if l == 2 and c >= threshold]
        if class_variants:
            best_class = max(class_variants, key=lambda x: x[1])

            # Compare with best commodity variant
            best_commodity = max(
                [(v, c) for ((v, l), c) in variant_confidence.items() if l == 3], key=lambda x: x[1], default=(None, 0)
            )

            # If there's a commodity variant and the class isn't significantly better, use commodity
            if best_commodity[0] and (best_class[1] - best_commodity[1]) < (best_commodity[1] * generic_delta_percentage / 100):
                return best_commodity
            return best_class

        # Finally check Family level (level 1)
        family_variants = [(v, c) for ((v, l), c) in variant_confidence.items() if l == 1 and c >= threshold]
        if family_variants:
            best_family = max(family_variants, key=lambda x: x[1])

            # Compare with best class and commodity variants
            best_higher_level = max(
                [(v, c) for ((v, l), c) in variant_confidence.items() if l > 1], key=lambda x: x[1], default=(None, 0)
            )

            # If there's a higher level variant and the family isn't significantly better, use higher level
            if best_higher_level[0] and (best_family[1] - best_higher_level[1]) < (
                best_higher_level[1] * generic_delta_percentage / 100
            ):
                return best_higher_level
            return best_family

    # If no variant meets the thresholds, return the variant with highest confidence score
    best_variant = sorted_variants[0][0]  # (variant, level)
    best_score = sorted_variants[0][1]  # confidence score

    return (best_variant[0], best_score)  # (UNSPSC code, confidence score)


def _get_embedding(description: Union[str, List[float]], llm: Optional[LLM] = None) -> List[float]:
    """
    Retrieves or generates the vector embedding for a given description.

    If `description` is already a list of floats (assumed to be an embedding),
    it's returned directly. If `description` is a string, the `llm` object
    is used to generate its embedding.

    Args:
        description (Union[str, List[float]]): The product description as a string,
            or its pre-calculated vector embedding.
        llm (Optional[LLM]): An LLM instance. Required if `description` is a string.
            Defaults to None.

    Returns:
        List[float]: The vector embedding.

    Raises:
        InvoiceProcessingError: If `description` is a string and `llm` is not provided.
    """
    if isinstance(description, str):
        if llm is None:
            raise InvoiceProcessingError("SEMANTIC MATCH: LLM object must be provided when Description is a string")
        return llm.get_embeddings([description])[0]
    else:
        return description


def _fetch_distinct_items(
    azure_search_utils: AzureSearchUtils,
    embedding: List[float],
    top_distinct_results: int,
    similarity_threshold: float,
    initial_fetch_size: int,
    select_fields: List[str],
) -> pd.DataFrame:
    """
    Fetches documents iteratively from Azure AI Search to retrieve up to
    `top_distinct_results` unique items based on 'ItemID'.

    This function ensures that each returned item meets the `similarity_threshold`
    and represents the highest scoring version if multiple documents share the same 'ItemID'.
    It performs searches in batches, excluding already found 'ItemID's in subsequent
    queries, until the desired number of distinct items is collected or no more
    suitable items can be found.

    Args:
        azure_search_utils (AzureSearchUtils): Instance of `AzureSearchUtils` for
                                               performing search queries.
        embedding (List[float]): The vector embedding to use for the search.
        top_distinct_results (int): The target number of distinct 'ItemID's to return.
        similarity_threshold (float): The minimum '@search.score' (similarity)
                                      a result must have to be considered.
        initial_fetch_size (int): The number of candidates to fetch in each search
                                  iteration. This is also used as a minimum batch size.
        select_fields (List[str]): A list of field names to retrieve from Azure Search.
                                   'ItemID' will be added if not present.

    Returns:
        pd.DataFrame: A DataFrame containing up to `top_distinct_results` distinct items,
                      sorted by '@search.score' in descending order. Each row corresponds
                      to the highest scoring document for a unique 'ItemID' that met
                      the criteria. Returns an empty DataFrame if no items are found.
    """
    if "ItemID" not in select_fields:
        # Ensure ItemID is always selected for deduplication logic
        select_fields = ["ItemID"] + [f for f in select_fields if f != "ItemID"]

    collected_items = {}  # Dictionary to store the best item found for each ItemID {item_id: {data}}
    found_item_ids: Set[str] = set()  # Set to keep track of ItemIDs already found

    while len(found_item_ids) < top_distinct_results:
        num_needed = top_distinct_results - len(found_item_ids)
        # Determine how many to fetch in this iteration - fetch a larger batch
        # to increase chances of finding new distinct items above the threshold.
        num_to_fetch = max(initial_fetch_size, num_needed * 2)  # Fetch at least initial_fetch_size or double the remaining needed

        # --- Construct Vector Query ---
        vector_query = {"vector": embedding, "fields": "ItemDescription_vector", "k_nearest_neighbors": num_to_fetch}

        # --- Construct Filter Expression ---
        # Base filter to exclude records with no UNSPSC
        current_filter = "UNSPSC ne null"

        # Build the full filter expression
        if found_item_ids:
            # Build OData filter to exclude already found ItemIDs.
            item_id_filter_parts = [f"ItemID ne {item_id}" for item_id in found_item_ids]
            item_id_filter = " and ".join(item_id_filter_parts)
            current_filter = f"{current_filter} and {item_id_filter}"

        # --- Perform Search ---
        try:
            current_results_df = azure_search_utils.search(
                vector_query=vector_query,
                filter_expression=current_filter,
                select=select_fields,
                top=num_to_fetch,  # Fetch the calculated batch size
            )
        except Exception as e:
            # Log the error appropriately
            logger.error(f"Error during Azure Search query: {e}")
            break  # Exit loop on search error

        # --- Process Results ---
        if current_results_df.empty or "@search.score" not in current_results_df.columns:
            # No more results returned by the search query
            break

        # Filter by similarity threshold
        qualified_results = current_results_df[current_results_df["@search.score"] >= similarity_threshold]

        if qualified_results.empty:
            # No results in this batch met the threshold
            # We might have exhausted all possible candidates above the threshold
            break

        # Sort by score descending to easily pick the best for each ItemID
        qualified_results = qualified_results.sort_values(by="@search.score", ascending=False)

        new_items_found_in_batch = 0
        for _, row in qualified_results.iterrows():
            item_id = row["ItemID"]

            # Check if this ItemID is new OR if this row has a higher score than previously found for this ItemID
            # (The second condition shouldn't happen if we exclude correctly, but good for robustness)
            if item_id not in found_item_ids:
                # Store the data for this item (as a dictionary or Series)
                collected_items[item_id] = row.to_dict()
                found_item_ids.add(item_id)
                new_items_found_in_batch += 1
                if len(found_item_ids) >= top_distinct_results:
                    break  # Stop processing this batch if we have enough distinct items

        if new_items_found_in_batch == 0:
            # If we fetched results above threshold but none had new ItemIDs,
            # it implies we've likely found all reachable distinct ItemIDs above threshold.
            break

        # Check if the search returned fewer results than requested, implies exhaustion
        if len(current_results_df) < num_to_fetch:
            break

    # --- Final Preparation ---
    if not collected_items:
        return pd.DataFrame()  # Return empty DataFrame if nothing was found

    # Convert the collected items back to a DataFrame
    final_results_list = list(collected_items.values())
    final_df = pd.DataFrame(final_results_list)

    # Sort the final distinct results by score and take the top ones needed
    final_df = final_df.sort_values(by="@search.score", ascending=False)
    final_df = final_df.head(top_distinct_results).reset_index(drop=True)

    return final_df
