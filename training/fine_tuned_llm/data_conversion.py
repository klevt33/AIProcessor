import json

import pandas as pd
from sqlalchemy import create_engine

# Input Description:
# 3548 - 909LABOR Fuel Filter
# Format the response strictly as a JSON object. No additional text, explanations, or disclaimers - only return JSON in this structure:
# ```json
# {
#     "ManufacturerName": "string",
#     "PartNumber": "string",
#     "UNSPSC": "string",
# }
# ```
# If any attribute is unavailable, return an empty string for that attribute.

query = """
        SELECT d.[IVCE_PRDT_LDSC], i.[MFR_PRT_NUM], i.[MFR_NM], i.[UNSPSC_CD]
        FROM AIML.IVCE_XCTN_LLM_TRNL_PRDT_REF d
        JOIN AIML.IVCE_XCTN_LLM_TRNL_MFR_REF i
            ON d.[IVCE_XCTN_LLM_TRNL_MFR_REF_UID] = i.[IVCE_XCTN_LLM_TRNL_MFR_REF_UID]
        WHERE d.[REC_ACTV_IND] = 'Y';
        """

sqlalchemy_connection_string = (
    "mssql+pyodbc://svc_aks-ai-spendreport-devl:h6ey8MwVgA6KymJuhe3uduyF@akssdpdevl.dc8f289d8f4f."
    "database.windows.net,1433/SDPDWH?driver=ODBC+Driver+17+for+SQL+Server"
)
sqlalchemy_engine = create_engine(sqlalchemy_connection_string, fast_executemany=True)

# Fetch data
with sqlalchemy_engine.connect() as conn:
    df = pd.read_sql(query, conn)

print("Total records: ", len(df))

user_prompt = """
Format the response strictly as a JSON object. No additional text, explanations, or disclaimers - only return JSON in this structure:
```json
{
    "ManufacturerName": "string",
    "PartNumber": "string",
    "UNSPSC": "string",
}
```
If any attribute is unavailable, return an empty string for that attribute."""


def prepare_user_prompt(description):
    return f"""
Input Description:
{description}
Format the response strictly as a JSON object. No additional text, explanations, or disclaimers - only return JSON in this structure:
```json
{{
    "ManufacturerName": "string",
    "PartNumber": "string",
    "UNSPSC": "string",
}}
```
If any attribute is unavailable, return an empty string for that attribute."""


def prepare_json(mfr_nm, mfr_pn, unspsc):
    return f"""```json
{{
    "ManufacturerName": "{mfr_nm}",
    "PartNumber": "{mfr_pn}",
    "UNSPSC": "{unspsc}"
}}
```"""


# Convert to GPT-4o training format
def format_gpt4o(row):
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant in finding and structuring information about electrical parts.",
            },
            {"role": "user", "content": prepare_user_prompt(row["IVCE_PRDT_LDSC"])},
            {"role": "assistant", "content": prepare_json(row["MFR_NM"], row["MFR_PRT_NUM"], row["UNSPSC_CD"])},
        ]
    }


# Apply formatting
training_data = df.apply(format_gpt4o, axis=1).tolist()

print("Final records: ", len(training_data))

# Save as JSONL
with open("training_data.jsonl", "w") as f:
    for entry in training_data:
        f.write(f"{json.dumps(entry)}\n")

print("Training data saved as training_data.jsonl")

# az openai fine-tune job list --resource-group AKS-RG-RPA_AI_Dev --workspace-name spend_report

# az openai fine-tune job show --resource-group AKS-RG-RPA_AI_Dev --name <your-job-name>
