import asyncio

from azure_search_utils import AzureSearchUtils
from config import Config
from llm import LLM
from matching_scores import best_match_score, process_manufacturer_dict
from matching_utils import analyze_description
from sdp import SDP
from utils import clean_description

# logging.basicConfig(level=logging.INFO)
SOURCE = "azure_search"  # 'azure_search' or 'database'


async def run_test_cases():
    """
    Run test cases for analyze_description function with different sample descriptions
    to test various scenarios of the function's output.
    """
    print("Initializing configuration and services...")
    config = Config()
    sdp = SDP(config)
    llm = LLM(config)
    search_utils = AzureSearchUtils(config)

    # Test cases with different descriptions
    test_cases = [
        # {"name": "Case 1: Overall scoring", "description": "521711234E STL-CTY 52171-1/2-3/4-E 4SQ BOX"},
        # {"name": "Case 2: Overall scoring", "description": "E940L-2-HD PVC 3 INCH COUPLING (E940L)"},
        # {
        #     "name": "Case 3: Testing MFR Name spelling",
        #     "description": "LEVITON MANUFACTURING (COMPANY) INC. UPDLC-S10 UPC DPLX LC-LC SM 10M",
        # },
        # {
        #     "name": "Case 4: Testing MFR Name in the beginning",
        #     "description": (
        #         'STRUT ACC 125 320848 320848 UNIVERSAL STRAP 1 1/4" UNIVERSAL STRAP ASSEMBLY G-STRUT GC114-125 HAYDON C-1104-1'
        #         " 1/4 S-STRUT 703-1-1/4EG"
        #     ),
        # },
        # {"name": "Case 5: Overall scoring", "description": "POW 04441-PWR POWERS 3/8X4 RND HD TOGGLE BOLT"},
        # {"name": "Case 6: Overall scoring", "description": "PT2097-W P&S 2097-GRY RADIANT SELF TEST GFCI 20A 125V GRY"},
        # {
        #     "name": "Case 7: Overall scoring",
        #     # "description": "OZG 5125S 1-1/4 S/SCR EMT CPLG 1-1/4 IN EMT SSCR CPLG SET-SCREW COUPLING"
        #     "description": "CROUSEH TP403 4SQ 2-1/8D BOX 1/2-3/4 KO 25/BX",
        # },
        # {
        #     "name": "Case 8: Overall scoring",
        #     "description": (
        #         "OZG 5050S 1/2 S/SCR STEEL EMT CONN SC50RKON 1/2 S/SCR STEEL EMT COUPLING CS 25/250 ALT EQUAL: KONKORE SK50RKON"
        #         " W/BOLT #0"
        #     ),
        # },
        # {"name": "Case 9: Overall scoring", "description": "LEV 2433 53011 LKG CONN L16-20R"},
        # {
        #     "name": "Case 6: PN Scoring",
        #     "description": "0781-4226 VICTOR GF150-50-580 FLOWGAUGE"
        # },
        # {
        #     "name": "Case 1: Testing new MFR matching logic",
        #     "description": "GRN JNO  123  33M  ACUITYY 24-BWH White Universal Baffle 123"
        # },
        # {
        #     "name": "Case 2: Testing new MFR matching logic",
        #     "description": "GRN PJ262W HUBBELL WIRING DEVICES WALLPLATE M-SIZE 2G 2 RECT WH"
        # },
        # {
        #     "name": "Case 3: Testing MFR scoring",
        #     "description": "HCFAP 5858-42-00 WIC. HCFAP 5858-42-00 12/2 STR 250C BLK WHT GRN W/#10 ALUM BOND ALUM JKT - STOCK IN CHINO, CA (PLUS FREIGHT) - PROCURED ITEM"
        # },
        # {
        #     "name": "Case 4: False positive test",
        #     "description": "SEPCO S1110 1/2\" STL SS EMT CONN 62007"
        # },
        # {"name": "Case 1: Parent MFR", "description": "FESTO 551375/FESTO SMT-10M-PS-24V-E-0,3-L-M 8D PROXIMITY SE"},
        # {"name": "Case 2: Parent MFR", "description": "551375/FESTO SMT-10M-PS-24V-E-0,3-L-M 8D PROXIMITY SE"},
        # {"name": "Case 3: Parent MFR", "description": "551375/FESTO SMT-10M- PS-24V-E-0,3-L-M 8D PROXIMITY SE"},
        # {"name": "Case 4: Parent MFR", "description": "551375/FESTO. SMT-10M-PS -24V-E-0,3-L-M 8D PROXIMITY SE"},
        # {
        #     "name": "Case 10: Verified flag",
        #     "description": (
        #         "ACME L3300T-10-PG HALF SLOT 13/16\" X 1-5/8\" STRUT 12GA 10' CHANNEL PRE GALVANIZED [BESC1316X10X12GASLZG90]"
        #     ),
        # },
        {"name": "Case 10: Verified flag", "description": "CULLY / MINERALLAC CULLY 60328J 1/4 X 1-3/4 TAPCON HWH CONCRETEF"}
    ]

    # Run all test cases
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Running {case['name']}")
        print(f"Description: {case['description']}")
        print("-" * 80)

        # try:
        # Call analyze_description
        if True:
            description = clean_description(case["description"])
            results_df, manufacturers_dict, full_manufacturer_data_dict, _, uipnc_list = await analyze_description(
                sdp, llm, description, SOURCE, search_utils
            )

            # Print results
            print(f"\nResults for Test Case {i}:")
            print("\nDataFrame Results:")
            if results_df.empty:
                print("No matching parts found.")
            else:
                print(f"Found {len(results_df)} matching part(s):")
                # Print relevant columns in a more readable format
                display_cols = [
                    "ItemID",
                    "MfrPartNum",
                    "MfrName",
                    "ItemDescription",
                    "DescSourceName",
                    "DescriptionSimilarity",
                    "CleanName",
                    "UncleanName",
                    "MfrNameMatchType",
                    # "UNSPSC",
                ]
                available_cols = [col for col in display_cols if col in results_df.columns]

                # Make a copy of the DataFrame for display purposes to avoid modifying the original
                display_df = results_df[available_cols].copy()

                # Truncate ItemDescription if longer than 50 characters
                if "ItemDescription" in display_df.columns:
                    display_df["ItemDescription"] = display_df["ItemDescription"].apply(
                        lambda x: f"{x[:50]}..." if isinstance(x, str) and len(x) > 50 else x
                    )

                print(display_df.to_string())

            print("\nManufacturer Dictionary:")
            if not manufacturers_dict:
                print("No manufacturers found in description.")
            else:
                print(f"Found {len(manufacturers_dict)} manufacturer(s):")
                for clean_name, unclean_name in manufacturers_dict.items():
                    print(f"  {clean_name}: {unclean_name}")

            # Execute best_match_score and print results
            print("\nBest Match Score Results:")
            if results_df.empty:
                print("No data available for best match scoring.")

                # NEW CODE: Try to process manufacturer dictionary if best_match_score has no results
                if manufacturers_dict:
                    print("\nProcessing manufacturer dictionary:")
                    mfr_df = process_manufacturer_dict(manufacturers_dict, description)
                    if mfr_df is not None:
                        print("Generated DataFrame from manufacturer dictionary:")
                        # Display relevant columns from the generated DataFrame
                        display_cols = ["CleanName", "UncleanName", "MfrNameMatchType", "ManufacturerNameConfidenceScore"]
                        print(mfr_df[display_cols].to_string())
                    else:
                        print("No DataFrame could be generated from manufacturer dictionary.")
            else:
                best_match = best_match_score(
                    results_df, manufacturers_dict, case["description"], full_manufacturer_data_dict, uipnc_list
                )
                if best_match is None:
                    print("No best match found.")

                    # Try to process manufacturer dictionary if best_match is None
                    if manufacturers_dict:
                        print("\nProcessing manufacturer dictionary:")
                        mfr_df = process_manufacturer_dict(manufacturers_dict, description)
                        if mfr_df is not None:
                            print("Generated DataFrame from manufacturer dictionary:")
                            # Display relevant columns from the generated DataFrame
                            display_cols = ["CleanName", "UncleanName", "MfrNameMatchType", "ManufacturerNameConfidenceScore"]
                            print(mfr_df[display_cols].to_string())
                        else:
                            print("No DataFrame could be generated from manufacturer dictionary.")
                else:
                    # Get all columns for display
                    confidence_cols = ["PartNumberConfidenceScore", "ManufacturerNameConfidenceScore", "UNSPSCConfidenceScore"]
                    # Add any original columns that might be useful
                    display_cols = ["ItemID", "MfrPartNum", "MfrName", "ManufacturerMatchRelationship", "IsVerified"]
                    # Combine columns, ensuring they exist in the DataFrame
                    all_cols = [col for col in display_cols + confidence_cols if col in best_match.columns]

                    print("Top matching record with confidence scores:")
                    print(best_match[all_cols].to_string())

        # except Exception as e:
        #     print(f"Error running test case: {str(e)}")

    print(f"\n{'=' * 80}")
    print("All test cases completed.")


if __name__ == "__main__":
    # Run the async test function
    asyncio.run(run_test_cases())
