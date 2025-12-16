import asyncio
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd


# --- Custom Formatter (Keep or simplify as needed) ---
class EnhancedFormatter(logging.Formatter):
    def format(self, record):  # noqa
        try:
            task = asyncio.current_task()
            record.taskName = task.get_name() if task else "NoAsyncTask"
        except RuntimeError:
            record.taskName = "NoTask/NoLoop"
        return super().format(record)


# --- Standard Imports and Setup ---
try:
    from lemmatizer import get_lemmatizer

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from ai_engine import AIEngine
    from config import Config
    from sdp import SDP
    from utils import clean_text_for_llm

    # Attempt to import common Azure exception type - replace if yours is different
    try:
        from azure.core.exceptions import HttpResponseError
    except ImportError:
        HttpResponseError = None  # Define as None if not available, handle generically
        print("Warning: azure.core.exceptions.HttpResponseError not found. Will handle errors generically.")

except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

# --- Logging Configuration ---
log_format = "%(asctime)s - %(levelname)s - %(threadName)s [%(taskName)s] - %(message)s"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = EnhancedFormatter(log_format)
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# --- Configuration ---
PATH = r"C:\\Users\\LevtovKirill(Perfici\Downloads\\"
INPUT_FILE = "llm_test_data1.xlsx"
OUTPUT_FILE = "llm_websearch_test_data_output.xlsx"
MAX_WORKERS = 10

# --- Global Initializations (Ensure Thread Safety!) ---
try:
    config = Config()
    sdp = SDP(config=config)
    ai_engine = AIEngine(config=config, sdp=sdp)
    lemmatizer = get_lemmatizer()
    df = pd.read_excel(os.path.join(PATH, INPUT_FILE))
    if "ITM_LDSC_CLEAN" not in df.columns:
        raise ValueError("Input Excel file must contain the 'ITM_LDSC_CLEAN' column.")
    # df = df.head(50)

except FileNotFoundError:
    logger.error(f"Input file not found at {os.path.join(PATH, INPUT_FILE)}")
    sys.exit(1)
except ValueError as ve:
    logger.error(f"Data Error: {ve}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Initialization failed: {e}", exc_info=True)
    sys.exit(1)


# --- Core Async Function (Unchanged) ---
async def run_websearch_async(item_description):
    logger.info(f"Starting web search stage for description: {item_description[:50]}...")
    cleaned_description = clean_text_for_llm(lemmatizer, item_description)
    websearch_stage_details = await ai_engine.ai_stages.extract_with_ai_agents_from_websearch(sdp, cleaned_description)
    if websearch_stage_details.status != "success":
        logger.warning(
            f"Websearch resulted in status: {websearch_stage_details.status}. Msg:"
            f" {websearch_stage_details.details.get('message', 'N/A')}"
        )
    else:
        logger.info(f"Successfully completed web search stage for: {item_description[:50]}...")
    return websearch_stage_details.details


# --- Synchronous Wrapper for Thread Pool (MODIFIED) ---
def process_row_sync_wrapper(row_data):
    description = row_data.get("ITM_LDSC_CLEAN", "")
    row_id = row_data.get("IVCE_DTL_UID", "UNKNOWN_ID")
    threading.current_thread().name

    if not description:
        logger.warning(f"Skipping row {row_id} due to empty description.")
        return {"IVCE_DTL_UID": row_id, "error": "Empty description", "processing_time_sec": 0}

    logger.info(f"Thread acquired job for row ID: {row_id}")
    start_time = time.perf_counter()
    result_details = {}
    error_info = {}  # Dictionary to store specific error details

    try:
        result_details = asyncio.run(run_websearch_async(description))
        if not isinstance(result_details, dict):
            logger.error(f"Unexpected result type: {type(result_details)}. Forcing empty dict.")
            # Store generic error info
            error_info = {"error_type": str(type(result_details)), "error_message": "Unexpected result type from async function"}
            result_details = {}  # Ensure it's a dict for merging later

    # --- MODIFIED EXCEPTION HANDLING ---
    except Exception as e:
        logger.error(f"Error processing row ID {row_id}: {e}", exc_info=False)  # Keep basic log

        # Store basic error info
        error_info = {"error_type": type(e).__name__, "error_message": str(e)}  # Get class name string

        # Attempt to extract more details, especially for Azure/HTTP errors
        # Check if it's the imported HttpResponseError (or duck-type if import failed)
        is_http_error = HttpResponseError and isinstance(e, HttpResponseError)
        # Fallback: check common attributes if import failed or it's a different HTTP error type
        has_response_attr = hasattr(e, "response")
        has_status_code_attr = hasattr(e, "status_code")

        if is_http_error or has_response_attr or has_status_code_attr:
            logger.info(f"Extracting details for HTTP/Azure error on row ID {row_id}:")

            status_code = getattr(e, "status_code", None)
            error_info["error_status_code"] = status_code
            logger.info(f"  Status Code: {status_code}")

            response = getattr(e, "response", None)
            if response:
                headers = getattr(response, "headers", {})
                error_info["error_headers"] = str(headers)  # Store headers as string
                logger.info(f"  Response Headers: {headers}")  # Log full headers for analysis

                # Extract specific useful headers if they exist
                retry_after = headers.get("Retry-After") or headers.get("retry-after")
                if retry_after:
                    error_info["error_retry_after"] = retry_after
                    logger.info(f"    Retry-After: {retry_after}")

                # Add other Azure-specific headers you might expect
                # e.g., x_ms_retry_after_ms = headers.get('x-ms-retry-after-ms') ...

                # Try to get response body/text
                try:
                    body = response.text  # Or response.content for bytes
                    error_info["error_body"] = body
                    # Log body carefully, maybe only for specific status codes like 429
                    if status_code == 429:
                        logger.info(f"  Response Body (Rate Limit): {body[:500]}...")  # Log truncated body
                    # Could also try parsing as JSON if appropriate:
                    # try:
                    #     body_json = response.json()
                    #     logger.info(f"  Response Body JSON: {body_json}")
                    # except json.JSONDecodeError:
                    #     logger.info(f"  Response Body Text: {body}")
                except Exception as body_err:
                    logger.warning(f"  Could not read response body: {body_err}")
                    error_info["error_body"] = "[Could not read body]"

            # Check for structured error object (common in Azure SDK)
            error_obj = getattr(e, "error", None)
            if error_obj:
                error_code = getattr(error_obj, "code", None)
                error_message_detail = getattr(error_obj, "message", None)
                error_info["error_details_code"] = error_code
                error_info["error_details_message"] = error_message_detail
                logger.info(f"  Structured Error Code: {error_code}")
                logger.info(f"  Structured Error Message: {error_message_detail}")

        else:
            # Log traceback for non-HTTP errors to understand their origin
            logger.exception(f"Caught non-HTTP exception type '{error_info['error_type']}' processing row ID {row_id}:")
            error_info["error_traceback"] = "See logs for details"  # Indicate traceback is in logs

        # Ensure result_details is empty if an error occurred
        result_details = {}
    # --- END OF MODIFIED EXCEPTION HANDLING ---

    finally:
        end_time = time.perf_counter()
        duration = round(end_time - start_time, 2)
        logger.info(f"Thread finished processing row ID: {row_id} in {duration}s")

        # Prepare results, merging base info, successful results (if any), and error info
        output_row = {
            "IVCE_DTL_UID": row_id,
            "manufacturer_name": result_details.get("manufacturer_name"),
            "conf_mfr_name": result_details.get("confidence_score", {}).get("manufacturer_name"),
            "part_number": result_details.get("part_number"),
            "conf_part_number": result_details.get("confidence_score", {}).get("part_number"),
            "unspsc": result_details.get("unspsc"),
            "conf_unspsc": result_details.get("confidence_score", {}).get("unspsc"),
            "source": result_details.get("web_search_url"),
            "description": result_details.get("description"),
            "processing_time_sec": duration,
            "ai_status_message": result_details.get("message"),  # Message from successful run
            # Add the captured error details - keys will only exist if an error occurred
            **error_info,
        }
    return output_row


# --- Main Execution Logic (Unchanged - handles results including error fields) ---
def main():
    logger.info(f"Starting threaded processing for {len(df)} rows with MAX_WORKERS={MAX_WORKERS}...")
    all_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_rowid = {
            executor.submit(process_row_sync_wrapper, row): row.get("IVCE_DTL_UID", f"Index_{idx}") for idx, row in df.iterrows()
        }
        for future in as_completed(future_to_rowid):
            row_id = future_to_rowid[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as exc:
                # This catches errors *in the executor or future handling itself*,
                # errors *during* the sync wrapper call are handled inside it now.
                logger.error(f"Row ID {row_id} generated an unhandled exception AT EXECUTOR LEVEL: {exc}", exc_info=True)
                all_results.append(
                    {"IVCE_DTL_UID": row_id, "error_message": f"Unhandled executor exception: {exc}", "processing_time_sec": -1}
                )

    logger.info(f"Finished processing {len(all_results)} results.")
    if not all_results:
        logger.warning("No results were processed.")
        return

    results_df = pd.DataFrame(all_results)
    # Ensure expected columns exist even if no errors occurred for consistency
    error_cols = [
        "error_type",
        "error_message",
        "error_status_code",
        "error_headers",
        "error_retry_after",
        "error_body",
        "error_details_code",
        "error_details_message",
        "error_traceback",
    ]
    for col in error_cols:
        if col not in results_df.columns:
            results_df[col] = None  # Add column with nulls if missing

    final_df = pd.merge(df, results_df, on="IVCE_DTL_UID", how="left")

    output_path = os.path.join(PATH, OUTPUT_FILE)
    try:
        final_df.to_excel(output_path, index=False)
        logger.info(f"Successfully saved results to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to Excel: {e}", exc_info=True)


# --- Script Execution ---
if __name__ == "__main__":
    total_start_time = time.perf_counter()
    logger.info("Script starting...")
    main()
    total_end_time = time.perf_counter()
    total_duration = round(total_end_time - total_start_time, 2)
    logger.info(f"Total script duration: {total_duration} seconds")
