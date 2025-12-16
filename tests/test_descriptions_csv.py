import asyncio
import csv  # Import csv module for access to quote constants
import os

import pandas as pd

from azure_search_utils import AzureSearchUtils
from config import Config
from llm import LLM
from logger import logging
from matching_scores import best_match_score, process_manufacturer_dict
from matching_utils import analyze_description
from sdp import SDP
from semantic_matching import semantic_match_by_description
from utils import clean_description

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Constants for source of data
SOURCE = "azure_search"
INPUT_FILE = r"C:\Users\LevtovKirill(Perfici\Downloads\QA-ExactMatchB.csv"


async def fetch_and_process_data():
    """
    Fetch data from a CSV file, process it, and save the results to a CSV file.
    """
    print("Initializing configuration and services...")
    config = Config()
    sdp = SDP(config)
    llm = LLM(config)
    search_utils = AzureSearchUtils(config)

    # Determine output file name based on input file name
    base_name = os.path.splitext(INPUT_FILE)[0]
    OUTPUT_FILE = f"{base_name}-results.csv"

    print(f"Reading descriptions from {INPUT_FILE}...")
    try:
        # Read CSV file with special configuration for multi-line fields
        input_df = pd.read_csv(
            INPUT_FILE,
            header=None,
            names=["ITM_LDSC"],
            quotechar='"',  # Specify quote character
            quoting=csv.QUOTE_ALL,  # Quote all fields using csv module's constant
            escapechar="\\",  # Handle escaped quotes
            lineterminator="\n",  # Explicit line terminator
            doublequote=True,  # Double quotes within quoted fields are treated as single quotes
            skip_blank_lines=True,  # Skip any blank lines
        )
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        # If the standard read_csv fails, try an alternative approach
        try:
            print("Attempting alternative read method...")
            # Read the entire file as a single string
            with open(INPUT_FILE, "r", encoding="utf-8") as f:
                content = f.read()

            # Manual parsing to handle multi-line entries
            # Split by new lines not within quotes

            # Replace newlines within quotes with a special marker
            in_quotes = False
            processed_content = ""
            for char in content:
                if char == '"':
                    in_quotes = not in_quotes
                    processed_content += char
                elif char == "\n" and in_quotes:
                    processed_content += " "  # Replace newline with space if inside quotes
                else:
                    processed_content += char

            # Create DataFrame from processed content
            from io import StringIO

            input_df = pd.read_csv(
                StringIO(processed_content),
                header=None,
                names=["ITM_LDSC"],
                quotechar='"',
                quoting=csv.QUOTE_ALL,  # Use csv module constant
            )

        except Exception as e2:
            print(f"Alternative read method failed: {str(e2)}")

            # Last resort: read line by line and manually process
            print("Attempting final read method...")
            try:
                descriptions = []
                with open(INPUT_FILE, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                current_desc = ""
                in_quotes = False

                for line in lines:
                    # Count quotes to determine if we're inside a quoted string
                    for char in line:
                        if char == '"':
                            in_quotes = not in_quotes

                    if current_desc:
                        current_desc += " " + line.strip()
                    else:
                        current_desc = line.strip()

                    # If we're not in quotes after processing the line, add the description
                    if not in_quotes:
                        descriptions.append(current_desc)
                        current_desc = ""

                # Add the last description if any
                if current_desc:
                    descriptions.append(current_desc)

                input_df = pd.DataFrame(descriptions, columns=["ITM_LDSC"])
            except Exception as e3:
                print(f"Final read method failed: {str(e3)}")
                return

    if input_df.empty:
        print("No data in the input CSV file.")
        return

    # Print all entries to debug
    print(f"Retrieved {len(input_df)} descriptions from the CSV file.")
    for idx, desc in enumerate(input_df["ITM_LDSC"]):
        print(f"Entry {idx + 1}: {desc[:50]}..." if len(desc) > 50 else f"Entry {idx + 1}: {desc}")

    # List to store processed results
    results_list = []

    # Process each description in the CSV
    for index, row in input_df.iterrows():
        description = row["ITM_LDSC"]
        print(f"\nProcessing record {index + 1}/{len(input_df)}")
        print(f"Description: {description[:100]}..." if len(description) > 100 else f"Description: {description}")

        try:
            # Step 1: Analyze description using existing logic
            cleaned_description = clean_description(description)
            results_df, manufacturers_dict, full_manufacturer_data_dict, description_embedding, uipnc_list = (
                await analyze_description(sdp, llm, cleaned_description, SOURCE, search_utils)
            )

            # Step 2: Call semantic_match_by_description for additional matching
            semantic_result = await semantic_match_by_description(
                description=description_embedding, azure_search_utils=search_utils, sdp=sdp
            )

            # Step 3: Combine results from both functions
            best_match = None
            if not results_df.empty:
                best_match = best_match_score(
                    results_df, manufacturers_dict, cleaned_description, full_manufacturer_data_dict, uipnc_list
                )

            mfr_df = None
            if best_match is None and manufacturers_dict:
                mfr_df = process_manufacturer_dict(manufacturers_dict, cleaned_description)

            # Capture required fields
            result = {
                "Test Description": description,
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
                "UncleanName": (
                    best_match["UncleanName"].iloc[0] if best_match is not None and "UncleanName" in best_match else None
                ),
                "MfrNameMatchType": (
                    best_match["MfrNameMatchType"].iloc[0]
                    if best_match is not None and "MfrNameMatchType" in best_match
                    else None
                ),
                "DescriptionSimilarity": (
                    float(best_match["DescriptionSimilarity"].iloc[0])
                    if best_match is not None and "DescriptionSimilarity" in best_match
                    else None
                ),
                "PartNumberConfidenceScore": (
                    float(best_match["PartNumberConfidenceScore"].iloc[0])
                    if best_match is not None and "PartNumberConfidenceScore" in best_match
                    else None
                ),
                "ManufacturerNameConfidenceScore": (
                    float(best_match["ManufacturerNameConfidenceScore"].iloc[0])
                    if best_match is not None and "ManufacturerNameConfidenceScore" in best_match
                    else None
                ),
                "UNSPSCConfidenceScore": (
                    float(best_match["UNSPSCConfidenceScore"].iloc[0])
                    if best_match is not None and "UNSPSCConfidenceScore" in best_match
                    else None
                ),
                # Add semantic_match_by_description results
                "Semantic_MfrName": semantic_result.get("MfrName", None),
                "Semantic_MfrNameConfidenceScore": (
                    float(semantic_result.get("ManufacturerNameConfidenceScore", 0))
                    if semantic_result.get("ManufacturerNameConfidenceScore") is not None
                    else None
                ),
                "Semantic_UNSPSC": semantic_result.get("UNSPSC", None),
                "Semantic_UNSPSCConfidenceScore": (
                    float(semantic_result.get("UNSPSCConfidenceScore", 0))
                    if semantic_result.get("UNSPSCConfidenceScore") is not None
                    else None
                ),
            }

            # Add manufacturer dictionary results if available
            if mfr_df is not None:
                result.update(
                    {
                        "CleanName": mfr_df["CleanName"].iloc[0] if "CleanName" in mfr_df.columns else None,
                        "UncleanName": mfr_df["UncleanName"].iloc[0] if "UncleanName" in mfr_df.columns else None,
                        "MfrNameMatchType": (
                            mfr_df["MfrNameMatchType"].iloc[0].value
                            if "MfrNameMatchType" in mfr_df.columns and hasattr(mfr_df["MfrNameMatchType"].iloc[0], "value")
                            else mfr_df["MfrNameMatchType"].iloc[0] if "MfrNameMatchType" in mfr_df.columns else None
                        ),
                        "ManufacturerNameConfidenceScore": (
                            float(mfr_df["ManufacturerNameConfidenceScore"].iloc[0])
                            if "ManufacturerNameConfidenceScore" in mfr_df.columns
                            else None
                        ),
                    }
                )

            results_list.append(result)

        except Exception as e:
            print(f"Error processing record at index {index}: {str(e)}")
            # Add to results list with just the description and error
            results_list.append({"Test Description": description, "Error": str(e)})
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
