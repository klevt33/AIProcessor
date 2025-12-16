import asyncio
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import time

from ai_engine import AIEngine
from config import Config
from sdp import SDP
from utils import clean_text_for_llm

path = r"C:\\Users\\VamsiMalneedi(Aspire\\M_Vamsi\\Projects\\Spend_Report\\laban-testing\\"
input_file = "llm_test_data1.xlsx"
output_file = "llm_test_data_output1.xlsx"

# Load Excel
df = pd.read_excel(os.path.join(path, input_file))  # Replace with your actual file path
output_data = []

config = Config()
sdp = SDP(config=config)
ai_engine = AIEngine(config=config, sdp=sdp)
semaphore = asyncio.Semaphore(5)  # max 10 concurrent requests


async def test_llm_stage(description):
    cleaned_description = clean_text_for_llm(ai_engine.lemmatizer, description)
    finetuned_stage_details = await ai_engine.ai_stages.extract_from_finetuned_llm(sdp, cleaned_description)
    # print(finetuned_stage_details.details)
    return finetuned_stage_details.details


# Async wrapper to time each LLM call
async def process_row(row):
    async with semaphore:
        start = time.perf_counter()
        result = await test_llm_stage(row["ITM_LDSC_CLEAN"])
        end = time.perf_counter()
        duration = round(end - start, 2)

        # Flatten the result and add time
        return {
            "IVCE_DTL_UID": row["IVCE_DTL_UID"],
            "manufacturer_name": result.get("manufacturer_name"),
            "part_number": result.get("part_number"),
            "unspsc": result.get("unspsc"),
            "description": result.get("description"),
            "conf_mfr_name": result.get("confidence_score", {}).get("manufacturer_name"),
            "conf_part_number": result.get("confidence_score", {}).get("part_number"),
            "conf_unspsc": result.get("confidence_score", {}).get("unspsc"),
            "processing_time_sec": duration,
        }


async def main():
    tasks = [process_row(row) for _, row in df.iterrows()]
    results = await asyncio.gather(*tasks)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    final_df = pd.merge(df, results_df, on="IVCE_DTL_UID")
    final_df.to_excel(os.path.join(path, output_file), index=False)


start = time.perf_counter()
asyncio.run(main())
end = time.perf_counter()
duration = round(end - start, 2)
print("Total duration of parallel processing of all invoices ", duration)
