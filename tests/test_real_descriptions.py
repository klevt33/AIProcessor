import asyncio

import pandas as pd

from azure_search_utils import AzureSearchUtils
from config import Config
from llm import LLM
from logger import logging
from matching_scores import best_match_score, process_manufacturer_dict
from matching_utils import analyze_description
from sdp import SDP
from utils import clean_description

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Constants for source of data
SOURCE = "database"  # 'azure_search' or 'database'
MIN_ID = 905679
MAX_ID = 905700
OUTPUT_FILE = r"C:\Repo\Spend Report AI API\tests\results-new.csv"


async def fetch_and_process_data():
    """
    Fetch data from the database, process it, and save the results to a CSV file.
    """
    print("Initializing configuration and services...")
    config = Config()
    sdp = SDP(config)
    llm = LLM(config)
    search_utils = AzureSearchUtils(config)

    # SQL query to retrieve data
    sql_query = f"""
        SELECT IVCE_DTL_UID, ITM_LDSC, MFR_NM, MFR_PRT_NUM, AKS_PRT_NUM, UPC_CD
        FROM [SDPDWH].[RPAO].[IVCE_DTL]
        WHERE IVCE_DTL_UID BETWEEN {MIN_ID} AND {MAX_ID}
          AND ITM_LDSC IS NOT NULL
          AND LEN(ITM_LDSC) > 7
    """

    print(f"Fetching data from database where IVCE_DTL_UID is between {MIN_ID} and {MAX_ID}...")
    db_data = await sdp.fetch_data(sql_query)

    # Convert fetched data into a DataFrame
    db_df = pd.DataFrame(db_data, columns=["IVCE_DTL_UID", "ITM_LDSC", "MFR_NM", "MFR_PRT_NUM", "AKS_PRT_NUM", "UPC_CD"])

    if db_df.empty:
        print("No data retrieved from the database.")
        return

    print(f"Retrieved {len(db_df)} records from the database.")

    # List to store processed results
    results_list = []

    # Process each description in the database
    for index, row in db_df.iterrows():
        print(f"\nProcessing record {index + 1}/{len(db_df)}: IVCE_DTL_UID={row['IVCE_DTL_UID']}")

        # Call analyze_description
        try:
            results_df, manufacturers_dict, full_manufacturer_data_dict, desc_embedding, uipnc_list = await analyze_description(
                sdp, llm, row["ITM_LDSC"], SOURCE, search_utils
            )

            # Call best_match_score
            best_match = None
            if not results_df.empty:
                best_match = best_match_score(
                    results_df, manufacturers_dict, row["ITM_LDSC"], full_manufacturer_data_dict, uipnc_list
                )

            # Call process_manufacturer_dict if no best match is found
            mfr_df = None
            if best_match is None and manufacturers_dict:
                mfr_df = process_manufacturer_dict(manufacturers_dict, clean_description(row["ITM_LDSC"]))

            # Capture required fields
            result = {
                "IVCE_DTL_UID": row["IVCE_DTL_UID"],
                "ITM_LDSC": row["ITM_LDSC"],
                "MFR_NM": row["MFR_NM"],
                "MFR_PRT_NUM": row["MFR_PRT_NUM"],
                "AKS_PRT_NUM": row["AKS_PRT_NUM"],
                "UPC_CD": row["UPC_CD"],
                "ItemID": best_match["ItemID"].iloc[0] if best_match is not None and "ItemID" in best_match else None,
                "MfrPartNum": best_match["MfrPartNum"].iloc[0] if best_match is not None and "MfrPartNum" in best_match else None,
                "MfrName": best_match["MfrName"].iloc[0] if best_match is not None and "MfrName" in best_match else None,
                "UPC": best_match["UPC"].iloc[0] if best_match is not None and "UPC" in best_match else None,
                "UNSPSC": best_match["UNSPSC"].iloc[0] if best_match is not None and "UNSPSC" in best_match else None,
                "AKPartNum": best_match["AKPartNum"].iloc[0] if best_match is not None and "AKPartNum" in best_match else None,
                "DescriptionID": (
                    best_match["DescriptionID"].iloc[0] if best_match is not None and "DescriptionID" in best_match else None
                ),
                "ItemDescription": (
                    best_match["ItemDescription"].iloc[0] if best_match is not None and "ItemDescription" in best_match else None
                ),
                "CleanName": best_match["CleanName"].iloc[0] if best_match is not None and "CleanName" in best_match else None,
                # "UncleanName": (
                #     best_match["UncleanName"].iloc[0] if best_match is not None and "UncleanName" in best_match else None
                # ),
                "UncleanName": (
                    best_match["MfrNameMatchType"].iloc[0]
                    if best_match is not None and "MfrNameMatchType" in best_match
                    else None
                ),
                "DescriptionSimilarity": (
                    best_match["DescriptionSimilarity"].iloc[0]
                    if best_match is not None and "DescriptionSimilarity" in best_match
                    else None
                ),
                "PartNumberConfidenceScore": (
                    best_match["PartNumberConfidenceScore"].iloc[0]
                    if best_match is not None and "PartNumberConfidenceScore" in best_match
                    else None
                ),
                "ManufacturerNameConfidenceScore": (
                    best_match["ManufacturerNameConfidenceScore"].iloc[0]
                    if best_match is not None and "ManufacturerNameConfidenceScore" in best_match
                    else None
                ),
                "UNSPSCConfidenceScore": (
                    best_match["UNSPSCConfidenceScore"].iloc[0]
                    if best_match is not None and "UNSPSCConfidenceScore" in best_match
                    else None
                ),
            }

            # Add manufacturer dictionary results if available
            if mfr_df is not None:
                result.update(
                    {
                        "CleanName": mfr_df["CleanName"].iloc[0] if "CleanName" in mfr_df.columns else None,
                        "UncleanName": mfr_df["UncleanName"].iloc[0] if "UncleanName" in mfr_df.columns else None,
                        "MfrNameMatchType": mfr_df["MfrNameMatchType"].iloc[0] if "MfrNameMatchType" in mfr_df.columns else None,
                        "ManufacturerNameConfidenceScore": (
                            mfr_df["ManufacturerNameConfidenceScore"].iloc[0]
                            if "ManufacturerNameConfidenceScore" in mfr_df.columns
                            else None
                        ),
                    }
                )

            results_list.append(result)

        except Exception as e:
            print(f"Error processing record IVCE_DTL_UID={row['IVCE_DTL_UID']}: {str(e)}")
            continue

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results_list)

    # Save results to CSV file
    print(f"\nSaving results to {OUTPUT_FILE}...")
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("Processing completed successfully.")


if __name__ == "__main__":
    # Run the async function
    asyncio.run(fetch_and_process_data())
