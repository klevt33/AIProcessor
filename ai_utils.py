"""
## Overview
This module provides functionality for processing LLM and web search results, calculating confidence scores, and selecting
the most relevant result based on predefined criteria. AI utility functions are included mainly in this module.

"""

import asyncio
import random
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd

from agents import Agents
from constants import Constants, LocalFiles, Logs, SpecialCases, StageNames, SubStageNames
from exceptions import InvalidJsonResponseError, InvoiceProcessingError, MissingRequiredFieldError, TruncatedJsonError
from logger import logger
from matching_utils import read_manufacturer_data
from utils import extract_json, get_alphanumeric, is_not_empty, is_not_null, load_yaml, remove_accents


async def get_clean_mfr_name(sdp, mfr_name):
    """
    Retrieve a cleaned manufacturer name based on the provided name.

    This function checks if the given manufacturer name exists in the manufacturer data
    retrieved from the `read_manufacturer_data` function. If a match is found, it returns
    the cleaned name along with a flag indicating success. If no match is found, it returns
    the original name and a flag indicating failure.

    Args:
        sdp (Any): The source data provider or context required to fetch manufacturer data.
        mfr_name (str): The manufacturer name to be cleaned.

    Returns:
        tuple: A tuple containing:
            - str: The cleaned manufacturer name if found, otherwise the original name.
            - bool: A flag indicating whether the cleaning was successful (True) or not (False).
    """
    # Fetch the manufacturer data using the provided source data provider (sdp).
    mfr_dict = await read_manufacturer_data(sdp=sdp)

    # Attempt to retrieve the cleaned name from the manufacturer dictionary.
    # value = mfr_dict.get(mfr_name.strip().upper())
    value = mfr_dict.get(mfr_name)

    # Return the cleaned name and success flag if found, otherwise return the original name and failure flag.
    return (value, True) if value is not None and is_not_empty(string=mfr_name) else (mfr_name, False)


# --- Helper for JSON Extraction/Validation ---
def extract_and_validate_json(response_text: str | None, agent_type: str, thread_id: str | None):
    """
    Extracts JSON from the provided response text and validates that the result is not empty.

    Raises:
        InvalidJsonResponseError: If the response is empty, contains non-retriable invalid JSON, or yields an empty result.
        TruncatedJsonError: If the JSON appears to be structurally incomplete/truncated (retriable).
    """
    logger.debug(f"AGENT WEB_SEARCH (Helper): Validating JSON for {agent_type}. Thread: {thread_id}")

    if not response_text:
        log_msg = f"AGENT WEB_SEARCH (Helper): {agent_type.capitalize()} agent response was empty or None."
        if thread_id:
            log_msg += f" Thread: {thread_id}"
        logger.warning(log_msg)

        # FIX: Formatted details into the message string. Removed 'response=' keyword argument.
        raise TruncatedJsonError(
            f"Agent returned an empty or null response (treating as retryable truncation). Thread: {thread_id}"
        )

    # Detect Dangling Markdown Fence
    stripped_resp = response_text.strip()
    if stripped_resp.endswith("```json") or stripped_resp == "```":
        logger.warning(
            f"AGENT WEB_SEARCH (Helper): Detected dangling markdown fence. Treating as truncation. Thread: {thread_id}"
        )

        # Formatted details into the message string
        snippet = stripped_resp[-100:] if len(stripped_resp) > 100 else stripped_resp
        raise TruncatedJsonError(f"Response ends with dangling markdown fence. Snippet: {snippet}")

    try:
        extracted_data = extract_json(response_text)
        logger.debug(f"AGENT WEB_SEARCH (Helper): Successfully validated JSON for {agent_type}. Thread: {thread_id}")
        return extracted_data

    except TruncatedJsonError:
        # Re-raise explicit truncation errors from extract_json so the retry loop catches them
        raise
    except (InvoiceProcessingError, TypeError, ValueError) as extract_ex:
        log_msg = f"AGENT WEB_SEARCH (Helper): Failed JSON validation for {agent_type} due to "
        f"underlying extraction error: {extract_ex}."
        if thread_id:
            log_msg += f" Thread: {thread_id}"
        logger.warning(log_msg)

        # NOTE: InvalidJsonResponseError DOES accept thread_id and response arguments in exceptions.py, so this is valid.
        raise InvalidJsonResponseError(
            f"Agent completed but failed during JSON extraction/validation for {agent_type}: {extract_ex}",
            thread_id=thread_id,
            response=response_text,
        ) from extract_ex


def _get_tokens_of_string(token_list, value, strip_chars=' \n"'):
    """
    Finds the contiguous token index range in token_list whose concatenation
    equals `value`, handling token fragmentation and cleanup robustly.

    Args:
        token_list (list[str]): raw tokens from the model
        value (str): exact target substring to match
        strip_chars (str): characters to strip from the edges of the reconstruction

    Returns:
        List[int]: indices [start, ..., end] of the matching tokens, or [] if none.
    """
    if not value:
        return []

    value_len = len(value)
    # Pre-calculate normalized value (no spaces) for fallback matching
    value_normalized = "".join(value.split())

    for start_idx in range(len(token_list)):
        current_concat = ""
        current_indices = []

        for i in range(start_idx, len(token_list)):
            raw_tok = token_list[i]
            current_concat += raw_tok
            current_indices.append(i)

            # Clean the current reconstruction
            # 1. Unescape quotes (double AND single)
            clean_concat = current_concat.replace('\\"', '"').replace("\\'", "'")
            # 2. Strip edge chars (quotes/spaces)
            clean_concat = clean_concat.strip(strip_chars)

            # Check 1: Exact Match
            if clean_concat == value:
                return current_indices

            # Check 2: Normalized Match (Ignore internal spacing)
            # This handles cases like "20 '" vs "20'"
            if "".join(clean_concat.split()) == value_normalized:
                return current_indices

            # Check 3: Case-insensitive fallback (Exact length)
            if len(clean_concat) == value_len and clean_concat.lower() == value.lower():
                return current_indices

            # Stop condition: If we've gone significantly past the length
            if len(clean_concat) > value_len + 10:
                break

    return []


async def calculate_confidence_for_finetuned_llm(sdp, response, results_json, fields):
    """
    Calculates confidence scores for fields extracted by a fine-tuned LLM (Language Model).

    This function processes the response metadata from the LLM, extracts tokens and log probabilities,
    and calculates confidence scores for specific fields such as Manufacturer Name, Part Number, and UNSPSC Code.
    It also cleans the Manufacturer Name using an external function and returns the processed details.

    Args:
        sdp: An external service or object used for cleaning the Manufacturer Name (Source Data Provider).
        response: The response object from the LLM execution (expected to contain 'response_metadata' with 'logprobs').
        results_json (dict): A dictionary containing the extracted field values parsed from the LLM's JSON response.
        fields (list): A list of field keys (e.g., Logs.MFR_NAME) that need to be processed for this item.

    Returns:
        tuple: A tuple containing:
            - details (dict): A dictionary containing processed values (cleaned manufacturer, etc.) and a nested
                              'confidence_score' dictionary with scores for each field.
            - cln_mfr_flag (bool | None): A flag indicating whether the Manufacturer Name was successfully cleaned/mapped
                                          against the database. Returns None if manufacturer processing was not requested.
    """

    LOG_PREFIX = "[LLM_CONFIDENCE_CALC]"

    def calculate_confidence(df, token_list, value):
        if not value:
            return 0.0

        indices = _get_tokens_of_string(token_list, value)
        if not indices:
            # If we cannot align tokens, we cannot calculate confidence.
            return 0.0

        try:
            # Grab logprobs as floats
            logprobs = df.iloc[indices]["logprob"].to_numpy(dtype=float)

            # clip any absurdly large positives (avoid exp overflow)
            logprobs = np.clip(logprobs, a_min=None, a_max=0.0)

            # turn into probs, filter out anything non-finite
            probs = np.exp(logprobs)
            probs = probs[np.isfinite(probs)]

            if probs.size == 0:
                return 0.0

            mean_p = float(np.mean(probs))
            # clamp [0,1]
            conf = max(0.0, min(mean_p, 1.0))
            return round(conf, 2)

        except Exception as e:
            # Log specific value that failed
            logger.error(
                f"{LOG_PREFIX} Token confidence calculation failed for value='{value}'. Indices found: {indices}. Error: {str(e)}"
            )
            return 0.0

    def get_value_and_confidence(field_name):
        value = results_json.get(field_name)
        if value and is_not_empty(value):
            conf = calculate_confidence(df, token_list, value=str(value))
        else:
            value = None
            conf = 0.0
        return value, conf

    # Extract tokens and log probabilities from the response metadata
    try:
        tokens = response.response_metadata["logprobs"]["content"]
        df = pd.DataFrame(tokens)
        token_list = list(df["token"])
    except Exception as e:
        # Log metadata keys to help debug missing/malformed structure
        meta_keys = response.response_metadata.keys() if hasattr(response, "response_metadata") else "N/A"
        logger.error(
            f"{LOG_PREFIX} Failed to extract logprobs from response. Metadata keys available: {meta_keys}. Error: {str(e)}"
        )
        # Return empty confidence structure so pipeline can continue (with 0 confidence)
        return {Logs.CONFIDENCE: {}}, None

    details = {Logs.CONFIDENCE: {}}
    cln_mfr_flag = None

    # Calculate confidence scores for specific fields
    if Logs.MFR_NAME in fields:
        mfr_name, name_conf = get_value_and_confidence("ManufacturerName")

        mfr_name = remove_accents(mfr_name.strip().upper()) if mfr_name else ""

        # Clean the Manufacturer Name
        cln_mfr_name, cln_mfr_flag = await get_clean_mfr_name(sdp, mfr_name)

        details.update({Logs.MFR_NAME: cln_mfr_name, Logs.UNCLN_MFR_NAME: mfr_name})
        details[Logs.CONFIDENCE].update({Logs.MFR_NAME: round(name_conf * 100, 2)})

    if Logs.PRT_NUM in fields:
        part_number, pn_conf = get_value_and_confidence("PartNumber")

        details.update({Logs.PRT_NUM: part_number})
        details[Logs.CONFIDENCE].update({Logs.PRT_NUM: round(pn_conf * 100, 2)})

    if Logs.UNSPSC in fields:
        unspsc_code, code_conf = get_value_and_confidence("UNSPSC")
        details.update({Logs.UNSPSC: unspsc_code})
        details[Logs.CONFIDENCE].update({Logs.UNSPSC: round(code_conf * 100, 2)})

    return details, cln_mfr_flag


async def calculate_confidence_for_web_search_results_with_ranking(sdp, web_results_ranking_json, ivce_dtl):
    """
    Calculates confidence scores for web search results based on various ranking criteria.

    This function processes web search results, evaluates multiple scoring factors such as part number match,
    manufacturer match, description similarity, and UNSPSC code, and computes an overall confidence score
    for each result. It uses predefined scoring configurations and external manufacturer data.

    Args:
        sdp: An external service or object used for reading manufacturer data.
        web_results_ranking_json (list or dict): A list of dictionaries (or a single dictionary)
                                                 containing web search results and their ranking details.

    Returns:
        pd.DataFrame: A DataFrame containing the web search results with calculated scores and confidence values.
                      Returns None if no valid results are found.

    Raises:
        KeyError: If required keys are missing in the web search results or configuration file.
    """

    # Function to calculate mfr match score
    def get_mfr_match_score(match_type, mm_conf_yaml, mfr_nm, mfr_dict):
        """
        Calculates the manufacturer match score based on match type and manufacturer data.

        Args:
            match_type (str): The type of match (e.g., exact, likely, mismatch).
            mm_conf_yaml (dict): Configuration dictionary for manufacturer match scoring.
            mfr_nm (str): Manufacturer name from the web search result.
            mfr_dict (dict): Dictionary of existing manufacturers.

        Returns:
            int: The calculated manufacturer match score.
        """
        score_map = {
            Constants.EXACT: mm_conf_yaml["exact"],
            Constants.LIKELY: mm_conf_yaml["likely"],
            Constants.POSSIBLE: mm_conf_yaml["possible"],
            Constants.NOT_DETECTED: mm_conf_yaml["not_detected"],
            Constants.MISMATCH: mm_conf_yaml["mismatch"],
        }
        score = score_map.get(match_type, mm_conf_yaml["not_detected"])

        # Add bonus score if the manufacturer exists in the dictionary
        if mfr_dict.get(mfr_nm) is not None:
            score += mm_conf_yaml["existing_mfr"]
        return score

    # Function to calculate part number length score
    def get_part_number_length_score(part_number, pnl_conf_yaml):
        """
        Calculates the part number length score based on its length and alphanumeric properties.

        Args:
            part_number (str): The part number from the web search result.
            pnl_conf_yaml (dict): Configuration dictionary for part number length scoring.

        Returns:
            int: The calculated part number length score.
        """
        part_number = str(part_number)
        length = len(part_number)

        if length < 4:
            score = pnl_conf_yaml["lt_4"]
        elif length <= 5:
            score = pnl_conf_yaml["eq_4_5"]
        elif length <= 7:
            score = pnl_conf_yaml["eq_6_7"]
        else:
            score = pnl_conf_yaml["gt_7"]

        if part_number.isalnum():
            score += pnl_conf_yaml["alphanumeric"]
        return score

    # Function to calculate part number match score
    def get_part_number_match_score(match_type, pnm_conf_yaml):
        """
        Calculates the part number match score based on match type.

        Args:
            match_type (str): The type of match (e.g., exact, strong, partial).
            pnm_conf_yaml (dict): Configuration dictionary for part number match scoring.

        Returns:
            int: The calculated part number match score.
        """
        score_map = {
            Constants.EXACT: pnm_conf_yaml["exact"],
            Constants.STRONG: pnm_conf_yaml["strong"],
            Constants.PARTIAL: pnm_conf_yaml["partial"],
            Constants.POSSIBLE: pnm_conf_yaml["possible"],
            Constants.NO_MATCH: pnm_conf_yaml["no_match"],
        }
        return score_map.get(match_type, 0)

    # Function to calculate description similarity score
    def get_description_similarity_score(match_type, ds_conf_yaml):
        """
        Calculates the description similarity score based on match type.

        Args:
            match_type (str): The type of match (e.g., exact, very high, medium).
            ds_conf_yaml (dict): Configuration dictionary for description similarity scoring.

        Returns:
            int: The calculated description similarity score.
        """
        score_map = {
            Constants.EXACT: ds_conf_yaml["exact"],
            Constants.VERY_HIGH: ds_conf_yaml["very_high"],
            Constants.HIGH: ds_conf_yaml["high"],
            Constants.MEDIUM: ds_conf_yaml["medium"],
            Constants.LOW: ds_conf_yaml["low"],
        }
        return score_map.get(match_type, 0)

    # Load confidence scoring configuration
    conf_yaml = load_yaml(LocalFiles.CONFIDENCES_FILE)
    conf_yaml = conf_yaml[StageNames.EXTRACTION_WITH_LLM_AND_WEBSEARCH][SubStageNames.AZURE_AI_AGENT_WITH_BING_SEARCH]

    # Ensure web_results_ranking_json is a list of dicts for DataFrame creation
    if isinstance(web_results_ranking_json, dict):
        data_for_df = [web_results_ranking_json]
    else:
        data_for_df = web_results_ranking_json

    # Handle cases where web_results_ranking_json might be None or empty list
    if not data_for_df:
        logger.info("No results found in web search ranking to process.")
        return None

    df = pd.DataFrame(data_for_df)

    # Remove empty objects if exists
    # Check if "Source" column exists before trying to operate on it
    if "Source" in df.columns:
        df["Source"] = df["Source"].replace(Constants.EMPTY_STRING, pd.NA)
        df.dropna(subset=["Source"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    if len(df) == 0:
        logger.info("No results found in web search after initial processing.")
        return None
    else:
        # calculate data sources confidence
        # Ensure 'PriorityMatch' column exists before using it
        if "PriorityMatch" in df.columns:
            p_count = len(df[df["PriorityMatch"].astype(bool)])
            np_count = len(df) - p_count

            if p_count > 1 or (p_count == 1 and np_count > 0):
                ds_conf = 25
            elif p_count == 1:
                ds_conf = 10
            elif np_count > 0:
                ds_conf = np_count * 5
            else:
                ds_conf = 0
        else:  # Default if 'PriorityMatch' is missing
            ds_conf = 0
            logger.warning("Column 'PriorityMatch' not found in ranking results. Defaulting DataSourcesScore to 0.")

        df["DataSourcesScore"] = ds_conf

        # Precompute booleans to avoid repeating checks, ensure columns exist
        has_part_number = "PartNumber" in df.columns and df["PartNumber"].notna() & (
            df["PartNumber"].astype(str).str.strip() != Constants.EMPTY_STRING
        )
        has_mfr = "ManufacturerName" in df.columns and df["ManufacturerName"].notna() & (
            df["ManufacturerName"].astype(str).str.strip() != Constants.EMPTY_STRING
        )
        has_unspsc = "UNSPSC" in df.columns and df["UNSPSC"].notna() & (
            df["UNSPSC"].astype(str).str.strip() != Constants.EMPTY_STRING
        )

        # Part Number Length Score
        if "PartNumber" in df.columns:
            df["PartNumberLengthScore"] = df["PartNumber"].apply(
                lambda x: int(get_part_number_length_score(x, conf_yaml[Constants.PART_NUM_LENGTH]))
            )
        else:
            df["PartNumberLengthScore"] = 0
            logger.warning("Column 'PartNumber' not found. 'PartNumberLengthScore' set to 0.")

        # Part Number Match Score
        df["PartNumberMatchScore"] = 0
        if "PartNumberMatch" in df.columns and has_part_number.any():
            df.loc[has_part_number, "PartNumberMatchScore"] = df.loc[has_part_number, "PartNumberMatch"].apply(
                lambda x: int(get_part_number_match_score(x, conf_yaml[Constants.PART_NUM_MATCH]))
            )
        elif "PartNumberMatch" not in df.columns:
            logger.warning("Column 'PartNumberMatch' not found. 'PartNumberMatchScore' remains 0.")

        # Description Similarity Score
        if "ItemDescriptionMatch" in df.columns:
            df["DescriptionSimilarityScore"] = df["ItemDescriptionMatch"].apply(
                lambda x: int(get_description_similarity_score(x, conf_yaml[Constants.DESCRIPTION_SIMILARITY]))
            )
        else:
            df["DescriptionSimilarityScore"] = 0
            logger.warning("Column 'ItemDescriptionMatch' not found. 'DescriptionSimilarityScore' set to 0.")

        # MFR match Score
        mfr_dict = await read_manufacturer_data(sdp=sdp)
        df["ManufacturerMatchScore"] = 0
        if "ManufacturerMatch" in df.columns and "ManufacturerName" in df.columns:
            df["ManufacturerMatchScore"] = df.apply(
                lambda row: int(
                    get_mfr_match_score(
                        row["ManufacturerMatch"], conf_yaml[Constants.MFR_MATCH], row["ManufacturerName"], mfr_dict
                    )
                ),
                axis=1,
            )
        else:
            logger.warning("Columns 'ManufacturerMatch' or 'ManufacturerName' not found. 'ManufacturerMatchScore' remains 0.")

        # Calculate individual scores
        df["PartNumberScore"] = 0
        if (
            isinstance(has_part_number, pd.Series) and has_part_number.any()
        ):  # Check if there are any valid part numbers before calculating
            df.loc[has_part_number, "PartNumberScore"] = (
                df.loc[has_part_number, "PartNumberMatchScore"]
                + df.loc[has_part_number, "PartNumberLengthScore"]
                + df.loc[has_part_number, "DescriptionSimilarityScore"]
                + df.loc[has_part_number, "DataSourcesScore"]
            )

        df["MfrScore"] = 0
        if isinstance(has_mfr, pd.Series) and has_mfr.any():  # Check if there are any valid manufacturers
            df.loc[has_mfr, "MfrScore"] = (
                df.loc[has_mfr, "ManufacturerMatchScore"]
                + df.loc[has_mfr, "DescriptionSimilarityScore"]
                + df.loc[has_mfr, "DataSourcesScore"]
            )

        df["UnspscScore"] = 0
        if isinstance(has_unspsc, pd.Series) and has_unspsc.any():  # Check if there are any valid UNSPSCs
            df.loc[has_unspsc, "UnspscScore"] = (
                df.loc[has_unspsc, "DescriptionSimilarityScore"] + df.loc[has_unspsc, "DataSourcesScore"]
            )
            df.loc[has_unspsc, "UnspscScore"] = df.loc[has_unspsc, "UnspscScore"] * 2

        # Calculate overall confidence
        if ivce_dtl.is_special_case and ivce_dtl.special_case_type == SpecialCases.CASE_1:
            weightages = conf_yaml[Constants.WEIGHTAGE][SpecialCases.CASE_1]
        else:
            weightages = conf_yaml[Constants.WEIGHTAGE][Constants.CASE]

        df["Confidence"] = (
            df["PartNumberScore"] * weightages["prt_num"]
            + df["MfrScore"] * weightages["mfr"]
            + df["UnspscScore"] * weightages["unspsc"]
        )

        # Ensure 'ID' column exists for logging, or handle its absence
        log_columns = ["PartNumberScore", "MfrScore", "UnspscScore", "Confidence"]
        if "ID" in df.columns:
            log_columns.insert(0, "ID")

        # Log only existing columns from the desired list
        existing_log_columns = [col for col in log_columns if col in df.columns]
        if existing_log_columns:
            logger.debug("\n" + str(df[existing_log_columns]))
        else:
            logger.debug("Confidence calculation complete, but no standard ID/Score columns to log.")

        return df


async def get_higher_confidence_web_search_result(sdp, df, fields):
    """
    Selects the web search result with the highest confidence score, applying tie-breaking logic if necessary.

    This function filters the rows with the maximum confidence score from the given DataFrame. If multiple rows
    have the same confidence score, it applies a series of tie-breaking criteria based on mean scores, part number
    scores, manufacturer scores, and priority match flags. Finally, it selects the first row if ties still remain.

    Args:
        sdp: An external service or object used for cleaning manufacturer names.
        df (pd.DataFrame): A DataFrame containing web search results with calculated confidence scores.

    Returns:
        tuple: A tuple containing:
            - pd.Series: The selected row with the highest confidence score.
            - bool: A flag indicating whether the manufacturer name was cleaned.

    Raises:
        KeyError: If required columns are missing in the DataFrame.
    """
    # Filter rows with max confidence
    max_conf = df["Confidence"].max()
    df_max = df[df["Confidence"] == max_conf].copy()

    # Apply tie-breaking logic if multiple rows have the same confidence score
    if len(df_max) > 1:
        # Calculate the mean of Manufacturer Score and Part Number Score
        df_max["MeanScore"] = (df_max["MfrScore"] + df_max["PartNumberScore"]) / 2
        max_mean = df_max["MeanScore"].max()
        df_max = df_max[df_max["MeanScore"] == max_mean]

        # Further tie-breaking based on Part Number Score
        if len(df_max) > 1:
            max_part = df_max["PartNumberScore"].max()
            df_max = df_max[df_max["PartNumberScore"] == max_part]

        # Further tie-breaking based on Manufacturer Score
        if len(df_max) > 1:
            max_mfr = df_max["MfrScore"].max()
            df_max = df_max[df_max["MfrScore"] == max_mfr]

        # Further tie-breaking based on Priority Match flag
        if len(df_max) > 1:
            if df_max["PriorityMatch"].any():
                df_max = df_max[df_max["PriorityMatch"].astype(bool)]

        # Further tie-breaking based on UNSPSC Score
        if len(df_max) > 1:
            max_mfr = df_max["UnspscScore"].max()
            df_max = df_max[df_max["UnspscScore"] == max_mfr]

    # Select the first row if ties still remain
    selected_row = df_max.iloc[0].copy()

    # Get clean mfr mapping
    cln_mfr_flag = None
    if Logs.MFR_NAME in fields:
        mfr_name_raw = selected_row.get("ManufacturerName")
        mfr_name = remove_accents(mfr_name_raw.strip().upper()) if mfr_name_raw else ""
        cln_mfr_name, cln_mfr_flag = await get_clean_mfr_name(sdp, mfr_name)
        selected_row["UncleanManufacturerName"] = mfr_name
        selected_row["ManufacturerName"] = cln_mfr_name

    # logger.debug("\n" + str(selected_row))
    return selected_row, cln_mfr_flag


def validate_search_results_schema(results_json: any):
    """
    Validates that each item in the agent's response contains the 'Source' key.

    Raises:
        MissingRequiredFieldError: If any item is missing the 'Source' key.
    """
    if not results_json:
        return

    items = [results_json] if isinstance(results_json, dict) else results_json

    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and "Source" not in item:
                raise MissingRequiredFieldError("Search result item is missing the required 'Source' field.")


async def run_agent_and_get_json_with_retry(
    agents: Agents, prompt: str, agent: Any, agent_type: str, validator: Optional[Callable[[Any], None]] = None
) -> Tuple[Any, str, str, str]:
    """
    Runs an agent, validates the response is valid JSON, and retries on specific failures.

    This function encapsulates a robust retry loop that handles two main failure scenarios:
    1.  **SDK-Level Failures:** The underlying `agents.run()` method has its own comprehensive
        retry logic for transient network issues, rate limits, etc.
    2.  **Truncated JSON Responses:** If `agents.run()` completes successfully but returns a
        response that appears to be a truncated or malformed JSON object/array, this
        function will trigger a full retry of the agent call.

    It will NOT retry on responses that are valid non-JSON text or responses that parse
    to an empty JSON object/list, treating these as final outcomes.

    Args:
        agents (Agents): The configured Agents instance which contains retry settings
                         and the method to run the agent.
        prompt (str): The prompt to send to the agent.
        agent (Any): The initialized agent object (e.g., an AzureAgent instance) to run.
        agent_type (str): A string identifier for the agent (e.g., "web search", "ranking")
                          used for clear logging.

    Returns:
        A tuple containing:
        - parsed_json (Any): The successfully extracted and validated JSON data (dict or list).
        - assistant_response (str): The raw text response from the assistant.
        - thread_id (str): The ID of the thread used for the interaction.
        - run_status (str): The final status of the successful agent run (e.g., "completed").

    Raises:
        InvalidJsonResponseError: If the agent run fails with a non-completed status, if the
                                  response is non-retriable invalid JSON, or if all retries
                                  for a truncated JSON are exhausted.
    """
    last_exception = None
    parsed_json_on_fail = None
    assistant_response, run_status, thread_id = None, None, None

    for attempt in range(agents.max_agent_retries):
        try:
            # Step 1: Run the agent
            assistant_response, run_status, thread_id = await agents.run(prompt=prompt, agent=agent)
            logger.debug(
                f"RETRY_HELPER: Agent '{agent_type}' run finished on attempt {attempt + 1}/{agents.max_agent_retries}. "
                f"Status: {run_status}. Thread: {thread_id}"
            )

            # Step 2: Check status
            if run_status != Constants.COMPLETED_lower:
                logger.error(
                    f"RETRY_HELPER: Agent '{agent_type}' run failed with non-completed status '{run_status}'. Thread: {thread_id}"
                )

                # --- INCLUDE ERROR DETAILS IN EXCEPTION ---
                error_context = assistant_response if assistant_response else "No details provided"
                raise InvalidJsonResponseError(
                    f"Agent run for '{agent_type}' failed with status: {run_status}. Error: {error_context}. Thread: {thread_id}"
                )

            # Step 3: Extract and Validate
            parsed_json = extract_and_validate_json(assistant_response, agent_type, thread_id)
            parsed_json_on_fail = parsed_json

            if validator:
                validator(parsed_json)

            return parsed_json, assistant_response, thread_id, run_status

        except TruncatedJsonError as e:
            last_exception = e
            logger.warning(
                f"RETRY_HELPER: Retriable truncated JSON from '{agent_type}' on attempt "
                f"{attempt + 1}/{agents.max_agent_retries}. Thread: {thread_id}. Error: {e}"
            )

            if attempt + 1 >= agents.max_agent_retries:
                logger.error(f"RETRY_HELPER: Max retries reached for '{agent_type}'. Failing. Thread: {thread_id}")
                # InvalidJsonResponseError accepts thread_id/response kwargs
                raise InvalidJsonResponseError(
                    f"Agent '{agent_type}' failed to return valid JSON after {agents.max_agent_retries} retries.",
                    thread_id=thread_id,
                    response=assistant_response,
                ) from e
            else:
                wait_time = agents.default_retry_wait_seconds + random.uniform(0, 5.0)
                logger.debug(f"RETRY_HELPER: Waiting {wait_time:.2f} seconds before next attempt...")
                await asyncio.sleep(wait_time)

        except MissingRequiredFieldError as e:
            last_exception = e
            logger.warning(
                f"RETRY_HELPER: Retriable 'MissingRequiredFieldError' from '{agent_type}' "
                f"on attempt {attempt + 1}/{agents.max_agent_retries}. "
                f"Thread: {thread_id}. Details: {e}"
            )

            # Check if this is the final attempt
            if attempt + 1 >= agents.max_agent_retries:
                logger.warning(
                    "RETRY_HELPER: Max retries reached for missing 'Source' key. "
                    f"Patching data with 'N/A' and returning as success. Thread: {thread_id}"
                )

                items_to_patch = [parsed_json_on_fail] if isinstance(parsed_json_on_fail, dict) else parsed_json_on_fail

                if isinstance(items_to_patch, list):
                    for item in items_to_patch:
                        if isinstance(item, dict) and "Source" not in item:
                            item["Source"] = Constants.MISSING_SOURCE

                return parsed_json_on_fail, assistant_response, thread_id, run_status
            else:
                wait_time = agents.default_retry_wait_seconds + random.uniform(1.0, 5.0)
                logger.debug(f"RETRY_HELPER: Waiting {wait_time:.2f} seconds before next attempt...")
                await asyncio.sleep(wait_time)

        except InvalidJsonResponseError:
            logger.error(f"RETRY_HELPER: Non-retriable error for '{agent_type}'. Failing. Thread: {thread_id}")
            raise

    # Final Catch-all
    raise InvalidJsonResponseError(
        f"Exited retry loop for '{agent_type}' without success after {agents.max_agent_retries} attempts.",
        thread_id=thread_id,
        response=assistant_response,
    ) from last_exception


def check_if_rpa_can_boost_confidence(ivce_dtl, field, ai_value):
    """
    Determine whether RPA-derived data matches the AI-extracted value for a given field.

    This function looks up a corresponding cleaned attribute on the `ivce_dtl` object
    based on a predefined mapping for manufacturer name and part number. It then
    compares the RPA-derived value to the AI-extracted value (after stripping to
    alphanumeric characters) and returns True if they match, indicating that the
    RPA value can be used to boost the AI's confidence.

    Parameters
    ----------
    ivce_dtl : InvoiceDetail
        The invoice-line detail object containing RPA-cleaned fields (e.g. `CLN_MFR_NM`,
        `CLN_MFR_PRT_NUM`).
    field : str
        The AI-extracted field key (should be one of `Logs.MFR_NAME` or
        `Logs.PRT_NUM`).
    ai_value : str
        The value produced by the AI stage for comparison.

    Returns
    -------
    bool
        True if:
          - `field` is in the mapping,
          - the corresponding `ivce_dtl` attribute exists and is non-null/non-empty,
          - and the alphanumeric-normalized RPA value equals the normalized `ai_value`.
        Otherwise, False.

    Notes
    -----
    - Uses `get_alphanumeric(...)` to strip non-alphanumeric characters before comparison.
    - Returns False immediately if `field` is not in the internal mapping.
    """
    mapping = {Logs.MFR_NAME: "CLN_MFR_NM", Logs.PRT_NUM: "CLN_MFR_PRT_NUM"}
    can_boost = False

    # get field value using mapping
    if field not in mapping:
        return can_boost
    rpa_value = getattr(ivce_dtl, mapping[field], None)

    # compare normalized strings
    if (
        ai_value is not None
        and is_not_null(ai_value)
        and is_not_empty(ai_value)
        and rpa_value is not None
        and is_not_null(rpa_value)
        and is_not_empty(rpa_value)
    ):
        # Only if both are valid do we proceed to the safe comparison.
        can_boost = get_alphanumeric(rpa_value).upper() == get_alphanumeric(ai_value).upper()

    return can_boost


def build_context_augmented_description(target_description, semantic_search_results, manufacturer_aliases, attempt=1):
    """
    Constructs the full input string for the Fine-Tuned LLM based on the attempt strategy.

    Strategies:
    - Attempt 1: Examples (find patterns to use for extraction) + Input (for extraction).
    - Attempt 2: Examples (for reference only) + Input (for extraction).
    - Attempt 3: No Examples + Input (Standard Label).

    Args:
        target_description (str): The raw input description to extract from.
        semantic_search_results (list): List of similar items found in DB.
        manufacturer_aliases (dict): Dictionary of {Official_Name: Alias_Token}.
        attempt (int): Current retry attempt number (1-based).

    Returns:
        tuple: (full_prompt_string, trace_metadata_dict)
    """
    parts = []

    # Initialize Trace Metadata
    trace_metadata = {
        "is_mfr_dict_provided": bool(manufacturer_aliases),
        "prompt_strategy": "no_context",
        "examples_count": 0,
        "examples_used": [],
    }

    # 1. Manufacturer Hints (All Attempts)
    if manufacturer_aliases:
        parts.append(
            "Manufacturer Alias to Official ManufacturerName Dictionary (use Official ManufacturerName if applicable to Input"
            " Description):"
        )
        # Dictionary is {Official_Name: Alias_Token}. Format required: Alias -> Official
        for official_name, alias_token in manufacturer_aliases.items():
            if official_name and alias_token:
                parts.append(f"{alias_token} -> {official_name}")
        parts.append("")  # Spacer

    # 2. Examples (Attempts 1 & 2 Only)
    if attempt < 3 and semantic_search_results:
        if attempt == 1:
            header = "Examples (find patterns to use for extraction):"
            trace_metadata["prompt_strategy"] = "patterns_for_extraction"
        else:
            header = "Examples (for reference only):"
            trace_metadata["prompt_strategy"] = "reference_only"

        example_lines = []
        examples_trace = []

        for item in semantic_search_results:
            # Limit to top 3
            if len(example_lines) >= 3:
                break

            desc = item.get("ItemDescription") or ""
            # Construct the exact dictionary string format requested
            output_dict = {
                "ManufacturerName": item.get("MfrName") or "",
                "PartNumber": item.get("MfrPartNum") or "",
                "UNSPSC": item.get("UNSPSC") or "",
            }
            # Format: 'Input String' -> {'Key': 'Value', ...}
            example_lines.append(f"'{desc}' -> {output_dict}")

            # Save for trace
            examples_trace.append({"input": desc, "output": output_dict})

        if example_lines:
            parts.append(header)
            parts.extend(example_lines)
            parts.append("")  # Spacer
            trace_metadata["examples_count"] = len(example_lines)
            trace_metadata["examples_used"] = examples_trace

    elif attempt >= 3:
        trace_metadata["prompt_strategy"] = "no_context"

    # 3. Target Input Description
    # Attempts 1 & 2 use the "nudge" suffix. Attempt 3 uses standard label.
    if attempt < 3:
        input_header = "Input Description (for extraction):"
    else:
        input_header = "Input Description:"

    parts.append(input_header)
    parts.append(target_description)

    return "\n".join(parts), trace_metadata


async def execute_hardened_llm_request(
    llm_instance, chain, target_description, semantic_search_results=None, manufacturer_aliases=None, max_retries=3
):
    """
    Executes LLM chain with Staged Prompt Strategy (Aggressive -> Passive -> Zero Context).
    Performs validation but does NOT inject system feedback on retries.

    Args:
        llm_instance (LLM): The LLM client wrapper.
        chain (Runnable): The LangChain runnable (Prompt + LLM).
        target_description (str): The raw input description.
        semantic_search_results (list): Context examples.
        manufacturer_aliases (dict): Manufacturer hints.
        max_retries (int): Number of attempts (strategies).

    Returns:
        tuple: (response_object, results_json, trace_metadata)

    Raises:
        Exception: Re-raises the last exception if all retries fail.
    """
    last_error = None

    for attempt in range(max_retries):
        current_attempt = attempt + 1

        # 1. Build the full input string (Context + Target) dynamically per attempt strategy
        full_input_string, trace_metadata = build_context_augmented_description(
            target_description=target_description,
            semantic_search_results=semantic_search_results,
            manufacturer_aliases=manufacturer_aliases,
            attempt=current_attempt,
        )

        # # Debug Log
        # logger.info(f"\n{'='*20} FULL PROMPT (ATTEMPT {current_attempt}) {'='*20}\n{full_input_string}\n{'='*60}")

        try:
            # 2. Execute
            # Note: The prompt template now expects a single 'description' variable containing everything
            response = await llm_instance.get_llm_response(chain=chain, params={"description": full_input_string})

            # 3. Validate Content
            if not response.content or not response.content.strip():
                raise ValueError("LLM returned empty content")

            results_json = extract_json(response.content)
            if not results_json:
                raise ValueError("Failed to extract valid JSON from response")

            # 4. Validate UNSPSC
            unspsc = results_json.get("UNSPSC")
            if unspsc and is_not_empty(str(unspsc)):
                clean_unspsc = str(unspsc).strip()
                if not (clean_unspsc.isdigit() and len(clean_unspsc) == 8):
                    raise ValueError(f"Extracted UNSPSC '{unspsc}' is not a valid 8-digit code.")

            # 5. Validate Part Number (Hard Constraint)
            # Check for GENERIC exception
            is_generic = False
            if "GENERIC" in target_description.upper():
                is_generic = True
            elif manufacturer_aliases:
                for official in manufacturer_aliases.keys():
                    if official and "GENERIC" in str(official).upper():
                        is_generic = True
                        break

            if not is_generic:
                pn = results_json.get("PartNumber")
                if pn and is_not_empty(str(pn)):
                    norm_pn = get_alphanumeric(str(pn)).upper()
                    norm_target = get_alphanumeric(target_description).upper()

                    if norm_pn not in norm_target:
                        raise ValueError(f"Extracted Part Number '{pn}' not found in Input Description.")

            # If we get here, validation passed
            return response, results_json, trace_metadata

        except Exception as e:
            last_error = e
            logger.warning(
                f"FINETUNED_LLM_EXEC: Attempt {current_attempt}/{max_retries} failed validation/execution. Error: {str(e)}"
            )
            # Simply continue to next loop iteration (which changes the prompt strategy)
            await asyncio.sleep(1)

    # Re-raise the last error if all retries fail
    raise last_error
