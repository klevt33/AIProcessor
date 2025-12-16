# test_real_semantic_match.py

import asyncio

import pandas as pd

from azure_search_utils import AzureSearchUtils

# --- Direct Imports of Your Project Modules ---
# These must be available in your Python environment.
from config import Config
from llm import LLM
from logger import logger  # Assuming semantic_matching.py and its utils use this
from matching_utils import read_manufacturer_data  # Assuming this is how semantic_matching.py imports it
from sdp import SDP

# --- Imports from semantic_matching.py (the module under test) ---
from semantic_matching import (  # Import any other constants or utility functions from semantic_matching if they were; previously accessed directly by the test script logic (though most were internal to semantic_match_by_description)
    _calculate_weights,
    _fetch_distinct_items,
    _get_embedding,
    _process_manufacturer_with_weights,
    _process_unspsc_with_weights,
    get_config,
)

# Note: LocalFiles, StageNames, SubStageNames, load_yaml will be used internally by get_config()

# --- Test Input ---
# !!! MODIFY THIS DESCRIPTION TO TEST DIFFERENT SCENARIOS !!!
TEST_DESCRIPTION_STRING = "WIEG SC080804NK SCPB 8X8X4 NO KO"
# TEST_DESCRIPTION_STRING = "Industrial strength blue widgets" # Another example


async def run_real_semantic_match_by_description():
    """
    Mimics the semantic_match_by_description function step-by-step using real components.
    Prints intermediate results.
    """
    print("--- Test Script for semantic_match_by_description (REAL COMPONENTS) ---")
    print(f'Input Description: "{TEST_DESCRIPTION_STRING}"\n')

    # 1. Initialize real config, search utils, LLM, and SDP
    # This will fail if configurations are missing or services are unavailable.
    print("Step 0: Initializing real Config, SDP, AzureSearchUtils, LLM objects...")
    try:
        config_obj = Config()
        sdp_obj = SDP(config_obj)  # Set to None if SDP is optional and you want to test without it
        search_utils_obj = AzureSearchUtils(config_obj)
        llm_obj = LLM(config_obj)
        print("Successfully initialized Config, SDP, AzureSearchUtils, LLM objects.\n")
    except Exception as e:
        print(f"CRITICAL ERROR during initialization: {e}")
        print("Please ensure all configurations (API keys, endpoints, DB connections) are correct and services are accessible.")
        print("The script cannot proceed.")
        return

    # --- Start Mimicking semantic_match_by_description ---

    # 2. Load configuration from semantic_matching.py's get_config()
    print("Step 1: Load SEMANTIC_SEARCH configuration using get_config()")
    try:
        sm_config = get_config()  # This will call utils.load_yaml internally
        print(f"Loaded SEMANTIC_SEARCH config: {sm_config}\n")
    except Exception as e:
        print(f"CRITICAL ERROR loading semantic search config via get_config(): {e}")
        print("Ensure LocalFiles.CONFIDENCES_FILE is correctly set up and accessible.")
        return

    # Extract config parameters as done in the original function
    search_conf = sm_config["SEARCH"]
    confidence_conf = sm_config["CONFIDENCE"]
    similarity_threshold = search_conf["similarity_threshold"]
    top_distinct_results_config = search_conf["top_results"]
    initial_fetch_size_config = search_conf["max_results"]
    mid_point_config = sm_config["WEIGHTS"]["mid_point"]
    exp_factor_config = sm_config["WEIGHTS"]["exp_factor"]
    min_confidence_threshold_config = sm_config["CONFIDENCE"]["min_confidence_threshold"]
    single_match_similarity_config = confidence_conf.get("single_match_similarity_threshold", 80)

    # 3. Get embedding for the description string
    print("Step 2: Get embedding using _get_embedding() with real LLM...")
    try:
        embedding_vector = _get_embedding(TEST_DESCRIPTION_STRING, llm_obj)
        print(f"Generated Embedding (first 10 dims): {embedding_vector[:10]}... (Total Dims: {len(embedding_vector)})\n")
    except Exception as e:
        print(f"ERROR during embedding generation: {e}")
        print("Check LLM service and configuration.")
        return

    # 4. Run iterative vector search to get distinct ItemIDs using _fetch_distinct_items()
    print("Step 3: Fetch distinct items using _fetch_distinct_items() with real Azure Search...")
    select_fields_for_search = ["ItemID", "DescriptionID", "MfrName", "UNSPSC", "ItemDescription"]
    print("  Parameters for _fetch_distinct_items:")
    print(f"    embedding: (vector of len {len(embedding_vector)})")
    print(f"    top_distinct_results: {top_distinct_results_config}")
    print(f"    similarity_threshold: {similarity_threshold}")
    print(f"    initial_fetch_size: {initial_fetch_size_config}")
    print(f"    select_fields: {select_fields_for_search}")

    try:
        results_df = _fetch_distinct_items(
            azure_search_utils=search_utils_obj,
            embedding=embedding_vector,
            top_distinct_results=top_distinct_results_config,
            similarity_threshold=similarity_threshold,
            initial_fetch_size=initial_fetch_size_config,
            select_fields=select_fields_for_search,
        )
        print("Results from _fetch_distinct_items (after deduplication, thresholding, and limiting to top results):")
        if results_df.empty:
            print("  DataFrame is empty.")
        else:
            print(results_df.to_string())
        print("\n")
    except Exception as e:
        print(f"ERROR during Azure Search fetch: {e}")
        print("Check Azure Search service, index, and configuration.")
        return

    # Initialize final result dictionary
    final_match_data = {}

    # 5. Handle case where no results are found
    print("Step 4: Check if results are empty after fetching")
    if results_df.empty:
        print("No results found after _fetch_distinct_items. Mimicked function would return empty dict.")
        print(f"Final Mimicked Result: {final_match_data}")
        return
    print("Results are not empty. Proceeding...\n")

    # 6. Clean MfrName and UNSPSC strings (strip whitespace)
    print("Step 5: Clean MfrName and UNSPSC strings (strip whitespace)")
    if "MfrName" in results_df.columns:
        results_df["MfrName"] = results_df["MfrName"].apply(lambda x: x.strip() if isinstance(x, str) else x)
    if "UNSPSC" in results_df.columns:
        results_df["UNSPSC"] = results_df["UNSPSC"].apply(lambda x: x.strip() if isinstance(x, str) else x)
    print("Results DataFrame after stripping whitespace from MfrName and UNSPSC:")
    if not results_df.empty:
        print(results_df.to_string())
    else:
        print("  DataFrame is empty.")
    print("\n")

    # Initialize IsMfrClean flag
    results_df["IsMfrClean"] = False

    # 7. Manufacturer name cleaning using SDP mapping (if sdp_obj is provided)
    print("Step 6: Clean manufacturer names using SDP mapping (if sdp_obj is provided and active)")
    # To test without SDP, you can comment out the sdp_obj initialization or set sdp_obj = None
    if sdp_obj is not None:
        print("  SDP object provided. Attempting real manufacturer name cleaning.")
        try:
            manufacturer_mapping = await read_manufacturer_data(sdp_obj)  # Real call
            print(
                f"  Manufacturer mapping loaded (Count: {len(manufacturer_mapping)}). First 5:"
                f" {dict(list(manufacturer_mapping.items())[:5])}"
            )

            any_mfr_name_cleaned_in_df = False

            def temp_clean_and_track_mfr(mfr_name_val):
                nonlocal any_mfr_name_cleaned_in_df
                if pd.notna(mfr_name_val) and mfr_name_val in manufacturer_mapping:
                    any_mfr_name_cleaned_in_df = True
                    return manufacturer_mapping[mfr_name_val]
                return mfr_name_val

            if "MfrName" in results_df.columns:
                results_df["MfrName"] = results_df["MfrName"].apply(temp_clean_and_track_mfr)

            results_df["IsMfrClean"] = any_mfr_name_cleaned_in_df

            print(f"  Were any manufacturer names in the DataFrame cleaned? {any_mfr_name_cleaned_in_df}")
            print("  Results DataFrame after manufacturer name cleaning attempt:")
            if not results_df.empty:
                print(results_df.to_string())
            else:
                print("  DataFrame is empty.")
            print("\n")
        except Exception as e:
            logger.warning(f"Mimic: Failed to apply real manufacturer mapping: {str(e)}", exc_info=True)
            print(f"  WARNING: Error during real manufacturer mapping: {e}. Continuing with original names.\n")
    else:
        print("  SDP object not provided or not active. Skipping manufacturer name cleaning.\n")

    # 8. Single result case - check similarity threshold
    print("Step 7: Handle single match: check if its score meets 'single_match_similarity_threshold'")
    is_single_match_df = len(results_df) == 1
    if not results_df.empty and "@search.score" in results_df.columns:
        results_df["@search.score"] = results_df["@search.score"].astype(float)
    else:
        print("  @search.score column missing or DataFrame empty before single match check. This might indicate an issue.")

    if is_single_match_df:
        score_of_single_match = results_df["@search.score"].iloc[0] * 100
        print(f"  Single match found. Score: {score_of_single_match:.2f}, Threshold: {single_match_similarity_config}")
        if score_of_single_match < single_match_similarity_config:
            print("  Single match score is below single_match_similarity_threshold. Mimicked function would return empty dict.")
            print(f"Final Mimicked Result: {final_match_data}")  # Should be empty
            return
        else:
            print("  Single match score meets the threshold. Proceeding.")
    else:
        print(f"  Not a single match case (found {len(results_df)} results). Proceeding.")
    print("\n")

    if results_df.empty:
        print("Results DataFrame is empty before weight calculation. Mimicked function would return empty dict.")
        print(f"Final Mimicked Result: {final_match_data}")
        return

    # 9. Calculate weights for each result
    print("Step 8: Calculate weights for results using _calculate_weights()")
    print(f"  Parameters for _calculate_weights: mid_point={mid_point_config}, exp_factor={exp_factor_config}")
    results_df_with_weights = _calculate_weights(results_df.copy(), mid_point_config, exp_factor_config)
    print("Results DataFrame with 'weight' column added:")
    if not results_df_with_weights.empty:
        print(results_df_with_weights.to_string())
    else:
        print("  DataFrame is empty.")
    print("\n")

    is_single_match_for_confidence_factor = len(results_df_with_weights) == 1
    single_match_conf_factor = confidence_conf["single_match_factor"]
    print("Step 8.5: Determine if single match for applying confidence adjustment factor")
    print(f"  is_single_match_for_confidence_factor: {is_single_match_for_confidence_factor}")
    print(f"  single_match_factor (for confidence): {single_match_conf_factor}\n")

    # 10. Process manufacturer name with weights
    print("Step 9: Process manufacturer names with weights using _process_manufacturer_with_weights()")
    print(
        f"  Parameters: min_confidence_threshold={min_confidence_threshold_config},"
        f" is_single_match={is_single_match_for_confidence_factor}, single_match_factor={single_match_conf_factor}"
    )
    mfr_results_dict = _process_manufacturer_with_weights(
        results_df_with_weights, min_confidence_threshold_config, is_single_match_for_confidence_factor, single_match_conf_factor
    )
    print(f"Manufacturer processing results: {mfr_results_dict}")
    if mfr_results_dict:
        final_match_data.update(mfr_results_dict)
    print("\n")

    # 11. Process UNSPSC with weights
    print("Step 10: Process UNSPSC codes with weights using _process_unspsc_with_weights()")
    print(
        "  Parameters: config (relevant parts will be used by function),"
        f" is_single_match={is_single_match_for_confidence_factor}, single_match_factor={single_match_conf_factor}"
    )
    print(f"    min_confidence_threshold (from config): {sm_config['CONFIDENCE']['min_confidence_threshold']}")
    print(f"    UNSPSC settings (from config): {sm_config['UNSPSC']}")
    unspsc_results_dict = _process_unspsc_with_weights(
        results_df_with_weights, sm_config, is_single_match_for_confidence_factor, single_match_conf_factor
    )
    print(f"UNSPSC processing results: {unspsc_results_dict}")
    if unspsc_results_dict:
        final_match_data.update(unspsc_results_dict)
    print("\n")

    # 12. Final assembled result data
    print("Step 11: Assembled Final Mimicked Result Data")
    print(final_match_data)
    print("\n--- Test Script Finished ---")


if __name__ == "__main__":
    # Ensure Pandas displays enough columns/rows for DataFrames
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", 150)  # Increased for potentially long descriptions

    asyncio.run(run_real_semantic_match_by_description())
