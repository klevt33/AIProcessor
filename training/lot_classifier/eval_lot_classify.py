import asyncio
import os
import sys
import time

import pandas as pd

"""
Use this script to run the LLM to use as classifier for the LOT data.
The previous run results are stored in Spend Report Share Point.
"""


WEBJOB_VERSION = 1
WEBJOB_NAME = ""
WEBJOB_ID = 1

semaphore = asyncio.Semaphore(30)  # max 10 concurrent requests


lot_classifier_prompt = """
You are a text classifier that decides whether the given input description belongs to "LOT" item or not.

If the input is ambiguous or cannot be confidently classified, use the label "uncertain".

Format the response strictly as a JSON object. No additional text, explanations, or disclaimers - only return JSON in this structure:
```json
{{
  "output": "<Yes | No | uncertain>",
  "confidence": <score between 0 and 1>
}}
```
Where "confidence" is your best estimate of how sure you are about the classification.

Examples:

Input: "20,250 Lumens - 150 Watt - 5000 Kelvin - LED Parking Lot Fixture 400 Watt Metal Halide Equal - Type V - Excel Series Mounting Hardware Sold Separately - 120-277 Volt - PLTS-11975"
Output:
{{
  "output": "No",
  "confidence": 1.0
}}

Input: "Quazite PC0608HA0017 6 x 8 Heavy Duty Polymer Concrete Cover To Read "ELECTRIC", Tier 15, For Driveways, Parking Lots & Off Roadway Applications) 6 x 8 Inch PC Covers"
Output:
{{
  "output": "No",
  "confidence": 1.0
}}

Input: "Siemens 52PL4E2P  30MM Red Pilot Light 120V Full Voltage Includes: (2) Extra Lenses - Green/ Amber (2) Legend Plates - ^On^/ ^Stop^ NEMA 1 3 4 4X 12 A600 Contacts"
Output:
{{
  "output": "No",
  "confidence": 1.0
}}


Input: "LOT  COOPER LIGHTING LLC COOPER LIGHTING SOLUTIONS - EPHESUS"
Output:
{{
  "output": "Yes",
  "confidence": 1.0
}}

Input: " SIEMENS LOT: 3 - TYPE TC.GEN1.XFMR: 1001 TLK38Q XFMR LUG KIT, THREE PHASE, 500 KVA"
Output:
{{
  "output": "Yes",
  "confidence": 1.0
}}

Input: "SQUARE D LOT SHIP ON 12/12 SCHNEIDER ELECTRIC USA INC Consisting of:"
Output:
{{
  "output": "Yes",
  "confidence": 1.0
}}


Input: "76334 ITASCA FP & KE TYPE SH1FN KENALL RMCA-4-FL/TR-0/0-45L35K-DCC-277-SYM /B-1-LEL-DLN-NCM-SP PART OF LOT: KENALL M"
Output:
{{
  "output": "Yes",
  "confidence": 1.0
}}

Input: "GENL PARTIAL LOT BILLING CONSIST OF REF SLT INT N"
Output:
{{
  "output": "Yes",
  "confidence": 1.0
}}

Input: "SCHNEIDER ELECTRIC USA INC LOT SCHNEIDER 7003564286"
Output:
{{
  "output": "Yes",
  "confidence": 1.0
}}

Input: "LOT OF EATON PER P2UN0516X4K1-0002 Consists of the following components: 1 EA of CUTLMISC1H EZB2042RMIA 001B Type: PANEL G1 1 EA of CUTLMISC1H EZT2042SMIA 001T Type: PANEL"
Output:
{{
  "output": "Yes",
  "confidence": 1.0
}}

Input: "3M 27-1/2X66 1/2 Inch x 66 Foot Glasscloth Tape, Heat Stable Insulation For Furnace/ Oven Controls, Motor Leads & Switches"
Output:
{{
  "output": "No",
  "confidence": 1.0
}}

Now classify the following inputs, give output for each:

Input: {input_description}
Output:

"""


def setup_env():
    """
    Add project root to sys.path
    """
    global APP_ROOT

    environment = "local"
    print("ENV Running: ", environment)

    if environment == "local":
        # Project root is three levels up.
        file_path = os.path.abspath(__file__)
        root = os.path.abspath(os.path.join(file_path, "..", "..", "..", ".."))
        if os.path.isdir(root) and root not in sys.path:
            sys.path.insert(0, root)
            APP_ROOT = root
            print(f"[INFO] Added to sys.path: {root}")

    else:
        # ─── Locate the folder that contains config.py ───────────────────────────────
        candidates = [os.path.join(os.environ.get("HOME", "/home"), "site", "wwwroot"), "/app", os.getcwd()]

        for root in candidates:
            if os.path.isfile(os.path.join(root, "config.py")):
                sys.path.insert(0, root)
                APP_ROOT = root
                print(f"[INFO] Added to sys.path: {root}")
                break
        else:
            raise ImportError("Cannot find config.py in any of the known locations: " + ", ".join(candidates))


def get_llm_prompt():
    return get_llm_prompt


async def get_llm_response(query):
    from langchain.prompts import PromptTemplate

    from llm import LLM
    from utils import extract_json

    prompt = get_llm_prompt()
    prompt_template = PromptTemplate(input_variables=["input_description"], template=prompt)

    llms = LLM(config=config)

    chain = prompt_template | llms.aoai_gpt4o_finetuned

    description = query

    # Get response from LLM
    response = await llms.get_llm_response(chain=chain, params={"input_description": description})

    logger.debug(f"LLM response - {response.content}")

    # Extract JSON from LLM response
    if response.content and response.content.strip() == Constants.EMPTY_STRING:
        logger.warning(f"FINETUNED LLM: LLM did not return any JSON. : {response.content}")
        return "failure", response.content
    else:
        results_json = extract_json(response.content)
        # logger.info(f'Json from LLM : {results_json}')
        return "success", results_json


async def get_data():
    # This file is currently in share point.
    file = "LOT_DATA_CLEANED.xlsx"

    lot_df = pd.read_excel(file)
    lot_df = lot_df[lot_df["LOT CAUGHT BY RPA?"] == "N"]
    lot_df = lot_df[(lot_df["LOT? "] == "Y") | (lot_df["LOT? "] == "N") | (lot_df["LOT? "] == "MATERIAL")]
    # logger.info(f'Total {len(lot_df)} lot items to be checked')

    df_a = lot_df[lot_df["LOT? "] == "MATERIAL"]
    df_b = lot_df[lot_df["LOT? "] == "Y"]
    df_c = lot_df[lot_df["LOT? "] == "N"]

    # Un-Comment below to evaluate only few random data.
    # sample_rows = 10
    # df_a = lot_df[lot_df['LOT? '] == 'MATERIAL'].sample(n=sample_rows, random_state=42)
    # df_b = lot_df[lot_df['LOT? '] == 'Y'].sample(n=sample_rows, random_state=42)
    # df_c = lot_df[lot_df['LOT? '] == 'N'] # only 17 rows.

    sample_rest_df = pd.concat([df_a, df_b, df_c], ignore_index=True)
    sample_rest_df = sample_rest_df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Total {len(sample_rest_df)} lot items to be checked")

    return sample_rest_df


async def process_row(idx, row):
    async with semaphore:
        print(f"Processing row : {idx}")
        result = {}
        try:
            start = time.perf_counter()
            description = row["ITM_LDSC"]
            logger.info(f" {idx} - Desc: {description}")
            result, result_map = await get_llm_response(description)
            elapsed = time.perf_counter() - start
            logger.info(f" {idx} - LLM: {result} - {result_map}, Time : {elapsed:0.2f} secs")
            if result == "success":
                if "output" in result_map and "confidence" in result_map:
                    return {"ITM_LDSC": description, "prediction": result_map["output"], "confidence": result_map["confidence"]}
                else:
                    if "confidence" in result_map and "output" not in result_map:
                        return {
                            "ITM_LDSC": description,
                            "prediction": "No Label prediction in LLM",
                            "confidence": result_map["confidence"],
                        }
                    if "output" in result_map and "confidence" not in result_map:
                        return {
                            "ITM_LDSC": description,
                            "prediction": result_map["output"],
                            "confidence": "No confidence from LLM",
                        }
                    if "output" not in result_map and "confidence" not in result_map:
                        return {
                            "ITM_LDSC": description,
                            "prediction": "No Label prediction in LLM",
                            "confidence": "No confidence from LLM",
                        }
            else:
                return {"ITM_LDSC": description, "prediction": "LLM Error: invalid json", "confidence": "0"}
        except Exception:
            return {"ITM_LDSC": description, "prediction": "LLM Error", "confidence": "0"}


async def main(config, logger):
    sample_test_df = await get_data()

    tasks = [process_row(idx, row) for idx, row in sample_test_df.iterrows()]
    results = await asyncio.gather(*tasks)

    # Convert to DataFrame
    output_file = "llm_lot_classifier_test.xlsx"
    results_df = pd.DataFrame(results)
    final_df = pd.merge(sample_test_df, results_df, on="ITM_LDSC")
    final_df.to_excel(output_file, index=False)


async def main1(config, logger):
    lot_df = await get_data()
    op_file = "LOT_DATA_CLEANED_op.xlsx"

    # df_a = lot_df[lot_df['LOT? '] == 'MATERIAL'].sample(n=sample_rows, random_state=42)
    # df_b = lot_df[lot_df['LOT? '] == 'Y'].sample(n=sample_rows, random_state=42)
    df_a = lot_df[lot_df["LOT? "] == "MATERIAL"]
    df_b = lot_df[lot_df["LOT? "] == "Y"]
    df_c = lot_df[lot_df["LOT? "] == "N"]  # only 17 rows.

    sample_rest_df = pd.concat([df_a, df_b, df_c], ignore_index=True)
    sample_rest_df = sample_rest_df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Total {len(sample_rest_df)} lot items to be checked")
    predictions = []
    confidence = []

    try:
        count = 1
        # for description in sample_rest_df['ITM_LDSC']:
        for idx, row in sample_rest_df.iterrows():
            # description = "LOT BILL FOR VCC CONTROLS PACKAGE"
            try:
                start = time.perf_counter()
                description = row["ITM_LDSC"]
                logger.info(f" {idx} - Desc: {description}")
                result, result_map = await get_llm_response(description)
                elapsed = time.perf_counter() - start
                logger.info(f" {idx} - Desc: {description}, LLM: {result} - {result_map}, Time : {elapsed:0.2f} secs")
                if result == "success":
                    if "output" in result_map:
                        predictions.append(result_map["output"])
                    else:
                        predictions.append("No label predicted")
                    if "confidence" in result_map:
                        confidence.append(int(result_map["confidence"]) * 100)
                    else:
                        predictions.append("No confidence from LLM")
                else:
                    predictions.append("LLM Error: invalid json")
                    confidence.append(0)  # just padding for data frame
                count += 1
            except Exception:
                predictions.append("LLM Error")
                confidence.append(0)  # just padding for data frame
                continue
    except Exception:
        predictions.append("LLM Error")
        confidence.append(0)  # just padding for data frame

    sample_rest_df["LOT_Predicted"] = predictions
    sample_rest_df["Confidence"] = confidence
    sample_rest_df.to_excel(op_file, index=False)


if __name__ == "__main__":

    # Load the app files

    print("setting up env!")
    setup_env()

    from ai_engine import AIEngine
    from config import Config
    from constants import Constants
    from logger import get_default_logger, job_id_var
    from sdp import SDP

    print("loading config")
    CONFIG = Config(app_root=APP_ROOT)  # Load application configuration
    print("APP VERSION Running: ", CONFIG.app_version, "WEBJOB VERSION Running: ", WEBJOB_VERSION)

    config = Config()
    sdp = SDP(config=config)
    ai_engine = AIEngine(config=config, sdp=sdp)

    logger = get_default_logger(name=WEBJOB_NAME, azure_conn_str=CONFIG.APP_INSIGHTS_CONN_STRING)  # Initialize logger
    job_id_var.set(WEBJOB_ID)  # Store the request ID in the context variable

    asyncio.run(main(config, logger))
