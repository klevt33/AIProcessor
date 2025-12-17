# # -------------------------------------------------------------------------
# # Copyright (c) Microsoft Corporation. All rights reserved.
# # Licensed under the MIT License. See License.txt in the project root for
# # license information.
# # --------------------------------------------------------------------------

import re

import nltk
import pandas as pd
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Constants
AZURE_ENDPOINT = "https://aks-spendreport-costnet-classifier-dev.cognitiveservices.azure.com/"
AZURE_KEY = os.environ["TEXT_ANALYTICS_KEY"]
PROJECT_NAME = "description-classifier-03-19-25"
DEPLOYMENT_NAME = "description-classifier-03-20-25"
# INPUT_FILE = r"C:\\Users\\VamsiMalneedi(Aspire\\M_Vamsi\\Projects\\Spend_Report\\data\\Invoice_test_data_for_AI.xlsx"
INPUT_FILE = r"C:\\Users\\VamsiMalneedi(Aspire\\M_Vamsi\\Projects\\Spend_Report\\laban-testing\\classifier-test-data2.xlsx"
OUTPUT_FILE = "classification_test_results.xlsx"
BATCH_SIZE = 25  # Number of documents to process per request

# Initialize NLP Tools
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()

# Azure Client
text_analytics_client = TextAnalyticsClient(endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY))


def clean_text(text):
    text = str(text)
    # Allowed special characters
    allowed_chars = set(".-/'\"")
    replace_char = " "

    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\$", " dollars ", text)  # Replace $ with "dollars"
    text = re.sub(r"%", " percent ", text)  # Replace % with "percent"
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)  # Remove periods that are not between digits
    text = re.sub(r"[^a-zA-Z0-9\s" + re.escape("".join(allowed_chars)) + "]", replace_char, text)  # Remove unwanted characters
    text = re.sub(r"\s+", " ", text).strip()  # Replace multiple spaces with a single space

    # Tokenize, remove stopwords, and apply lemmatization
    words = text.split()
    # words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords & lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize
    cleaned_text = " ".join(words)
    return cleaned_text


def classify_documents(df):
    """Classifies documents using Azure's Text Analytics API."""
    df["ITM_LDSC_CLEAN"] = df["ITM_LDSC"].astype(str).map(clean_text)
    results = []

    # Process documents in batches
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Processing Batches"):
        batch = df.iloc[i : i + BATCH_SIZE]
        documents = batch["ITM_LDSC_CLEAN"].tolist()

        poller = text_analytics_client.begin_single_label_classify(
            documents=documents, project_name=PROJECT_NAME, deployment_name=DEPLOYMENT_NAME
        )
        document_results = poller.result()

        # Collect results
        for idx, (doc, classification_result) in enumerate(zip(documents, document_results)):
            row = batch.iloc[idx]
            if classification_result.kind == "CustomDocumentClassification":
                classification = classification_result.classifications[0]
                results.append(
                    {
                        "IVCE_DTL_UID": row["IVCE_DTL_UID"],
                        "ITM_LDSC": row["ITM_LDSC"],
                        "ITM_LDSC_CLEAN": doc,
                        "CLASS": classification.category,
                        "CONFIDENCE": classification.confidence_score,
                    }
                )
            elif classification_result.is_error:
                results.append(
                    {
                        "IVCE_DTL_UID": row["IVCE_DTL_UID"],
                        "ITM_LDSC_CLEAN": doc,
                        "ERROR": classification_result.error.code,
                        "MESSAGE": classification_result.error.message,
                    }
                )

    # Convert to DataFrame and save
    final_df = pd.DataFrame(results)
    final_df.to_excel(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    df = pd.read_excel(INPUT_FILE)
    classify_documents(df)

# import os
# import re
# import pandas as pd
# import nltk
# import asyncio
# from tqdm.asyncio import tqdm
# from azure.core.credentials import AzureKeyCredential
# from azure.ai.textanalytics.aio import TextAnalyticsClient
# from nltk.stem import WordNetLemmatizer

# # Constants
# AZURE_ENDPOINT = "https://aks-spendreport-costnet-classifier-dev.cognitiveservices.azure.com/"
# AZURE_KEY = os.environ.get("TEXT_ANALYTICS_KEY")  # Replace with your actual key
# PROJECT_NAME = "description-classifier-03-19-25"
# DEPLOYMENT_NAME = "description-classifier-03-20-25"
# INPUT_FILE = r"C:\\Users\\VamsiMalneedi(Aspire\\M_Vamsi\\Projects\\Spend_Report\\data\\Invoice_test_data_for_AI.xlsx"
# OUTPUT_FILE = "classification_test_results.xlsx"
# BATCH_SIZE = 10  # Number of documents per request

# # Initialize NLP Tools
# nltk.download('wordnet')
# lemmatizer = WordNetLemmatizer()

# def clean_text(text):
#     text = str(text)
#     # Allowed special characters
#     allowed_chars = set(".-/'\"")
#     replace_char = " "

#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'\$', ' dollars ', text)  # Replace $ with "dollars"
#     text = re.sub(r'%', ' percent ', text)   # Replace % with "percent"
#     text = re.sub(r'(?<!\d)\.(?!\d)', '', text)  # Remove periods that are not between digits
#     text = re.sub(r"[^a-zA-Z0-9\s" + re.escape("".join(allowed_chars)) + "]", replace_char, text)  # Remove unwanted characters
#     text = re.sub(r"\s+", " ", text).strip()  # Replace multiple spaces with a single space

#     # Tokenize, remove stopwords, and apply lemmatization
#     words = text.split()
#     # words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords & lemmatize
#     cleaned_text = " ".join([lemmatizer.lemmatize(word) for word in words])  # Lemmatize
#     return cleaned_text

# async def classify_batch(text_analytics_client, batch, results):
#     """Classifies a batch of documents asynchronously."""
#     documents = batch["ITM_LDSC_CLEAN"].tolist()
#     poller = await text_analytics_client.begin_single_label_classify(
#         documents=documents,
#         project_name=PROJECT_NAME,
#         deployment_name=DEPLOYMENT_NAME
#     )
#     document_results = await poller.result()

#     for idx, (doc, classification_result) in enumerate(zip(documents, document_results)):
#         row = batch.iloc[idx]
#         if classification_result.kind == "CustomDocumentClassification":
#             classification = classification_result.classifications[0]
#             results.append({
#                 "IVCE_DTL_UID": row['IVCE_DTL_UID'],
#                 "ITM_LDSC_CLEAN": doc,
#                 "CLASS": classification.category,
#                 "CONFIDENCE": classification.confidence_score
#             })
#         elif classification_result.is_error:
#             results.append({
#                 "IVCE_DTL_UID": row['IVCE_DTL_UID'],
#                 "ITM_LDSC_CLEAN": doc,
#                 "ERROR": classification_result.error.code,
#                 "MESSAGE": classification_result.error.message
#             })

# async def classify_documents(df):
#     """Processes all documents asynchronously in batches."""
#     df["ITM_LDSC_CLEAN"] = df["ITM_LDSC"].astype(str).map(clean_text)
#     results = []

#     # Initialize async Azure client
#     async with TextAnalyticsClient(
#         endpoint=AZURE_ENDPOINT,
#         credential=AzureKeyCredential(AZURE_KEY),
#     ) as text_analytics_client:

#         tasks = []
#         for i in range(0, len(df), BATCH_SIZE):
#             batch = df.iloc[i:i + BATCH_SIZE]
#             tasks.append(classify_batch(text_analytics_client, batch, results))

#         # Run all batches asynchronously
#         await tqdm.gather(*tasks, desc="Processing Batches")

#     # Save results to Excel
#     final_df = pd.DataFrame(results)
#     final_df.to_excel(OUTPUT_FILE, index=False)
#     print(f"Results saved to {OUTPUT_FILE}")

# if __name__ == "__main__":
#     df = pd.read_excel(INPUT_FILE)
#     asyncio.run(classify_documents(df))
