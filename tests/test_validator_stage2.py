import asyncio
import json
import os

# Temporarily add project root to path to allow imports
import sys
from unittest.mock import AsyncMock, MagicMock

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ai_engine import AIEngine
from config import Config
from constants import Logs
from logger import logger

# --- Test Configuration ---
# This is the classic "false positive" scenario the validator is designed to catch.
TEST_DESCRIPTION = "replacement blade for a DEWALT DCE151B cable stripper"


async def run_integration_test():
    """
    Initializes a real AIEngine and runs a single invoice line item
    through the entire pipeline to test the Context Validator end-to-end.
    """
    logger.info("--- Starting AI Engine Integration Test ---")

    # 1. Set up the real configuration and AI Engine
    # We assume this script runs from the project root.
    # app_root = os.path.dirname(__file__)
    config = Config()

    # The sdp object is used in async initialization, so we must use AsyncMock.
    mock_sdp = AsyncMock()

    # --- MINIMALLY VIABLE MOCK DATA ---
    # The COMPLETE_MATCH stage needs manufacturer data from the DB to correctly
    # populate 'CleanName' when a manufacturer is not found in the text.

    # 1. Mock for the Category Classifier mapping (can be empty)
    category_df_schema = pd.DataFrame(columns=["CTGY_ID", "CTGY_NM", "PRNT_CTGY_ID"])

    # 2. Mock for the Manufacturer data lookup. We provide the one entry our test needs.
    manufacturer_data = {
        "CleanName": ["DEWALT"],
        "UncleanName": ["DEWALT"],
        "ParentCompanyName": ["DEWALT"],
        "AIMatchIndicator": [""],
    }
    manufacturer_df_schema = pd.DataFrame(manufacturer_data)

    # 3. Use 'side_effect' to return the correct DataFrame for each awaited call.
    # The first sdp.fetch_data call is for categories, the second is for manufacturers.
    mock_sdp.fetch_data.side_effect = [category_df_schema, manufacturer_df_schema]

    # As a fallback, any other calls will get a generic empty DataFrame.
    mock_sdp.fetch_data_as_dataframe.return_value = pd.DataFrame()

    engine = AIEngine(config=config, sdp=mock_sdp)
    await engine.async_init()  # Initialize manufacturer data, etc.

    # 2. Create a mock invoice detail object (ivce_dtl) to carry our test data
    ivce_dtl = MagicMock()
    ivce_dtl.ITM_LDSC = TEST_DESCRIPTION
    ivce_dtl.RNTL_IND = "N"
    ivce_dtl.CLN_MFR_AI_NM = None  # No pre-filled data from RPA

    # Define which fields the pipeline should try to extract
    ivce_dtl.fields = [Logs.MFR_NAME, Logs.PRT_NUM, Logs.UNSPSC]

    # This mapping simulates a typical run where all stages are active
    ivce_dtl.stage_mapping = {
        "CLASSIFICATION": ["DESCRIPTION_CLASSIFIER", "LOT_CLASSIFIER", "RENTAL_CLASSIFIER"],
        "SEMANTIC_SEARCH": ["SEMANTIC_SEARCH"],
        "COMPLETE_MATCH": ["COMPLETE_MATCH"],
        "CONTEXT_VALIDATOR": ["CONTEXT_VALIDATOR"],
        "FINETUNED_LLM": ["FINETUNED_LLM"],
        "EXTRACTION_WITH_LLM_AND_WEBSEARCH": ["AZURE_AI_AGENT_WITH_BING_SEARCH"],
    }

    logger.info(f"Processing test description: '{TEST_DESCRIPTION}'")

    # 3. Run the entire extraction pipeline
    stage_results_obj = await engine.process_description(ivce_dtl=ivce_dtl)

    # 4. Print the results for analysis
    logger.info("--- AI Engine Processing Complete ---")

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(json.dumps(stage_results_obj.final_results, indent=2))

    print("\n" + "=" * 80)
    print("FULL STAGE-BY-STAGE BREAKDOWN")
    print("=" * 80)
    # Use default=str to handle any non-serializable objects like datetimes
    print(json.dumps(stage_results_obj.results, indent=2, default=str))
    print("=" * 80)


if __name__ == "__main__":
    # Ensure asyncio event loop is managed correctly
    try:
        asyncio.run(run_integration_test())
    except Exception as e:
        logger.error(f"An error occurred during the integration test: {e}", exc_info=True)
