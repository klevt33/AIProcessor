"""
## Overview
This Python file contains utility functions for various purposes, including data processing, text cleaning,
configuration management, and token validation. Below is a categorized summary of the functions:
"""

import json
import os
import re
import unicodedata
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict
from zoneinfo import ZoneInfo

import jwt
import pandas as pd
import requests
import yaml as yaml
from aiohttp import web
from fastapi import HTTPException

from constants import AzureAppConfig, Constants, LocalFiles
from exceptions import InvoiceProcessingError, TruncatedJsonError
from logger import logger


def get_current_datetime_cst_obj():
    """Get current datetime object in Central Time."""
    return datetime.now(ZoneInfo("America/Chicago"))


def get_current_datetime_cst():
    """Get current datetime string in Central Time."""
    return str(get_current_datetime_cst_obj())


def generate_uuid() -> str:
    """
    Generates a new UUID each time function called
    """
    return str(uuid.uuid4())


def update_app_last_restart_time(config):
    """
    Updates the APP_RESTART_TIME application setting on an Azure Web App.

    This function:
      1. Reads the existing application settings (StringDictionary) from `config.azure_webapp`.
      2. Sets or updates the "APP_RESTART_TIME" key to the current CST timestamp.
      3. Pushes the merged settings back via `config.azure_web_client.web_apps.update_application_settings`.

    Args:
        config: A configuration-like object that must have:
            - azure_webapp: a StringDictionary (the result of list_application_settings)
            - azure_web_client: an authenticated WebSiteManagementClient
            - WA_RG: the resource group name (string)
            - WA_APP_NAME: the Web App name (string)

    Raises:
        Exception: Any errors returned by the Azure SDK (these will be re-raised after logging).
    """
    try:
        restart_time = get_current_datetime_cst()
        config.set_azure_app_config_value(AzureAppConfig.MAIN_APP_RESTART_TIME, config.app_config_label, restart_time)
        logger.info(f"Successfully set APP_RESTART_TIME={restart_time}")
    except Exception:
        logger.error("Failed to update APP_RESTART_TIME in Azure App Service settings.", exc_info=True)


def load_env_config(environment: str, app_root=Constants.EMPTY_STRING):
    """
    Load environment-specific configuration from a YAML file.

    Args:
        environment (str): Name of the environment (e.g., "dev", "prod").

    Returns:
        dict: Parsed YAML configuration as a dictionary.
    """
    path = os.path.join("env", environment + ".env.yaml")

    if not os.path.isabs(path) and app_root != Constants.EMPTY_STRING:
        path = os.path.join(app_root, path)

    try:
        with open(rf"{path}", "r") as file:
            return yaml.safe_load(file) or {}

    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in file {path}: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to load YAML file at {path}, App root: {app_root}, Error: {str(e)}")
    return {}


def load_yaml(path: str, app_root: str = Constants.EMPTY_STRING):
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML content as a dictionary.
    """
    # Join app_root and path only if path is relative and app_root is provided
    if not os.path.isabs(path) and app_root != Constants.EMPTY_STRING:
        path = os.path.join(app_root, path)

    try:
        with open(path, "r") as file:
            return yaml.safe_load(file) or {}
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in file {path}: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to load YAML file at {path}, App root: {app_root}, Error: {str(e)}")

    return {}


def dump_yaml(path: str, data: Dict):
    """
    Write data to a YAML file.

    Args:
        path (str): Path to the YAML file.
        data (Dict): Data to write to the file.
    """
    with open(rf"{path}", "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def read_txt_file(path: str):
    """
    Read the contents of a text file.

    Args:
        path (str): Path to the text file.

    Returns:
        str: Contents of the file.
    """
    with open(path, "r", encoding="utf-8") as file:
        file_content = file.read()
    return file_content


def convert_to_str(obj):
    """
    Convert an object to a string. If the object is a dictionary or list, convert it to JSON.

    Args:
        obj: Object to convert.

    Returns:
        str: String representation of the object.
    """
    if isinstance(obj, (dict, list)):
        obj = json.dumps(obj)
    else:
        obj = str(obj)
    return obj


def list_to_str(lst):
    """
    Convert a list to a numbered string.

    Args:
        lst (list): List to convert.

    Returns:
        str: Numbered string representation of the list.
    """
    return "\n".join([str(i) + ". " + str(j) for i, j in enumerate(lst)])


def is_not_empty(string):
    """
    Check if a string is not empty.

    Args:
        string (str): String to check.

    Returns:
        bool: True if the string is not empty, False otherwise.
    """
    return string.strip() != Constants.EMPTY_STRING


def is_not_null(string):
    """
    Check if a string is not Null.

    Args:
        string (str): String to check.

    Returns:
        bool: True if the string is not Null, False otherwise.
    """
    return string.strip().upper() != Constants.NULL_STRING


def is_not_undefined(value):
    """
    Check if a Value is not UNDEFINED.

    Args:
        value: Value to check.

    Returns:
        bool: True if the value is not UNDEFINED, False otherwise.
    """
    return str(value).strip().upper() != Constants.UNDEFINED


def have_min_length(string, length):
    """
    Check if a string has a minimum length.

    Args:
        string (str): String to check.
        length (int): Minimum length.

    Returns:
        bool: True if the string has at least the specified length, False otherwise.
    """
    return len(string.strip()) >= length if string else False


def get_spl_cases(app_root):
    special_cases_yaml = load_yaml(path=LocalFiles.SPECIAL_CASES_FILE, app_root=app_root)
    special_cases = special_cases_yaml[Constants.CASES] if Constants.CASES in special_cases_yaml else {}
    return special_cases


def extract_json(content: str):
    """
    Extract a JSON object or array from a string, potentially wrapped in markdown code fences.
    Args:
        content (str): The string potentially containing JSON.
    Returns:
        dict or list: The parsed Python object (dict or list).
    Raises:
        InvoiceProcessingError: If no valid JSON is found or if JSON decoding fails for non-retriable reasons.
        TruncatedJsonError: If a JSONDecodeError occurs on what appears to be a truncated JSON object/array.
        TypeError: If content is not a string or bytes-like object.
    """
    if not isinstance(content, (str, bytes)):
        logger.error(f"Invalid input type for extract_json: {type(content)}. Content: {str(content)[:200]}")
        raise TypeError(f"Expected string or bytes-like object, got {type(content)}")

    if isinstance(content, bytes):  # Decode if bytes
        try:
            content_str = content.decode("utf-8")
        except UnicodeDecodeError as ude:
            logger.error(f"Failed to decode bytes content to UTF-8: {ude}. Preview: {content[:200]}")
            raise InvoiceProcessingError(message=f"Content decode error: {ude}", original_exception=ude)
    else:
        content_str = content

    try:
        # Normalize line endings and strip leading/trailing whitespace from the whole content
        processed_content = content_str.strip()
        json_to_parse = processed_content

        # Check for and strip markdown code fences for JSON
        if processed_content.startswith("```json") and processed_content.endswith("```"):
            json_to_parse = processed_content[len("```json") : -len("```")].strip()
        elif processed_content.startswith("```") and processed_content.endswith("```"):
            json_to_parse = processed_content[len("```") : -len("```")].strip()
    except Exception as e:
        logger.error(f"Error in JSON string Normalization. {str(e)}", exc_info=True)
        raise InvoiceProcessingError(message="Error in JSON string Normalization.")

    if not json_to_parse:
        logger.warning("Content for JSON parsing is empty after stripping potential markdown fences.")
        logger.debug(f"Original content that led to empty parse string (length: {len(content_str)}):\n{content_str}")
        raise InvoiceProcessingError(message="JSON content is empty after stripping markdown fences.")

    try:
        extracted_data = json.loads(json_to_parse)
        return extracted_data
    except json.JSONDecodeError as decode_err:
        logger.warning(
            f"JSONDecodeError: Failed to decode JSON string: {decode_err}. "
            f"Attempted to parse (preview first 200 chars): '{json_to_parse[:200]}...'"
        )
        # Create a temporary string for the check, handling a potential unclosed markdown fence.
        check_str = json_to_parse.lstrip()
        if check_str.startswith("```json"):
            check_str = check_str[len("```json") :].lstrip()
        elif check_str.startswith("```"):
            check_str = check_str[len("```") :].lstrip()

        # Now check if the potentially cleaned string starts like a JSON object or array.
        if check_str.startswith(("{", "[")):
            # This is the RETRIABLE path
            logger.warning("Detected what appears to be a truncated JSON. Raising retriable error.")
            raise TruncatedJsonError(
                message=f"Potentially truncated JSON structure found: {decode_err}", original_exception=decode_err
            ) from decode_err
        else:
            # This is the NON-RETRIABLE path (e.g., plain text)
            raise InvoiceProcessingError(
                message=f"Invalid JSON structure found (likely a text response): {decode_err}", original_exception=decode_err
            ) from decode_err


def get_rental_flag(rental_flag):
    """
    Handle rental flag with possible values. Converts 'None' to 'N'
    Args:
        rental_flag (str): Y or N or None

    Returns:
        str: Y or N
    """
    return Constants.Y if rental_flag and rental_flag == Constants.Y else Constants.N


def apply_rental_flag(rental_flag, rpa_rental_flag):
    """
    Handle rental flag with possible values. Set only if RPA says it is not RENTAL
    Args:
        rental_flag (str): Y or N
        rpa_rental_flag (str): Y or N

    Returns:
        str: Y or N
    """
    if rpa_rental_flag == Constants.N:
        return rental_flag

    return rpa_rental_flag


def is_valid_aks_part_number(akp: str) -> bool:
    """
    Checks if ArchKey part number is not None and starts with 'AK' followed by digits
    Args:
        akp: AK Part number
    """
    pattern = re.compile(r"^AK\d+$")
    return True if akp and pattern.match(akp) else False


def is_valid_upc_number(upc: str) -> bool:
    """
    Checks if UPC number is not None and only digits of length 8-12
    Args:
        upc: UPC number
    """
    pattern = re.compile(r"^\d{8,12}$")
    return True if upc and pattern.match(upc) else False


def is_mfr_available_from_rpa(mfr_nm):
    """
    Checks if a MFR name is not [Null, None, '', UNDEFINED].
    Args:
        mfr_nm: MFR Name from RPA
    """
    return True if mfr_nm and is_not_empty(mfr_nm) and is_not_null(mfr_nm) and is_not_undefined(mfr_nm) else False


def make_rental(name):
    """
    Converts normal name to Rental name by appending the key with format required
    """
    return " - ".join([Constants.RENTAL, name])


def validate_token(CONFIG, token):
    """
    Validate a JWT token and extract claims.

    Args:
        CONFIG: Configuration object containing JWKS URL, audience, and issuer.
        token (str): JWT token to validate.

    Returns:
        dict: Decoded token payload.

    Raises:
        HTTPException: If the token is invalid or expired.
    """
    # logger.debug(CONFIG.WA_JWKS_URL)
    # logger.debug(requests.get(CONFIG.WA_JWKS_URL))
    try:
        # Fetch the JWKS
        response = requests.get(CONFIG.WA_JWKS_URL)
        response.raise_for_status()  # Raise an error for HTTP status codes >= 400

        # Attempt to parse the JSON
        jwks = response.json()

    except requests.exceptions.RequestException:
        raise HTTPException(status_code=500, detail="Unable to fetch JWKS.")
    except requests.exceptions.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JWKS response.")

    unverified_header = jwt.get_unverified_header(token)
    rsa_key = {}

    for key in jwks["keys"]:
        if key["kid"] == unverified_header["kid"]:
            rsa_key = {"kty": key["kty"], "kid": key["kid"], "use": key["use"], "n": key["n"], "e": key["e"]}
            break

    if not rsa_key:
        logger.error("Invalid token key.", exc_info=True)
        raise web.HTTPUnauthorized(reason="Invalid token key.")

    try:
        payload = jwt.decode(
            token,
            key=jwt.algorithms.RSAAlgorithm.from_jwk(rsa_key),
            algorithms=["RS256"],
            audience=CONFIG.WA_AUDIENCE,
            issuer=[CONFIG.WA_ISSUER_V1, CONFIG.WA_ISSUER_V2],
        )
        return payload
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(status_code=401, detail=f"Token expired: {str(e)}")
    except jwt.InvalidTokenError as e:
        logger.error(
            f"Token: {token}\nRSA Key: {rsa_key}\nAudience: {CONFIG.WA_AUDIENCE}\nIssuer: {CONFIG.WA_ISSUER_V1},"
            f" {CONFIG.WA_ISSUER_V2}",
            exc_info=True,
        )
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


def get_alphanumeric(text):
    """
    Remove all special characters and return an alphanumeric string.

    Args:
        text (str): Input string.

    Returns:
        str: Alphanumeric string.
    """
    return re.sub(r"[^a-zA-Z0-9]", "", text)


def clean_text_for_classification(lemmatizer, text):
    """
    Clean text for classification by removing unwanted characters and applying lemmatization.

    Args:
        lemmatizer: Lemmatizer object for word lemmatization.
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    logger.debug(f"CLASSIFICATION: Input text: {text}")

    # Allowed special characters
    allowed_chars = set(".-/'\"")

    # Replace character
    replace_char = " "

    # Convert to lowercase
    text = text.lower()

    # Replace $ with "dollars"
    text = re.sub(r"\$", " dollars ", text)

    # Replace % with "percent"
    text = re.sub(r"%", " percent ", text)

    # Remove periods that are not between digits
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)

    # Remove unwanted characters
    text = re.sub(r"[^a-zA-Z0-9\s" + re.escape("".join(allowed_chars)) + "]", replace_char, text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize, remove stopwords, and apply lemmatization
    words = text.split()

    # Remove stopwords & lemmatize
    # words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]

    cleaned_text = " ".join(words)
    logger.debug(f"CLASSIFICATION: Cleaned text: {cleaned_text}")

    return cleaned_text


def clean_text_for_llm(lemmatizer, text):
    """
    Clean text for LLM (Language Model) by removing unwanted characters.

    Args:
        lemmatizer: Lemmatizer object for word lemmatization.
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    logger.debug(f"LLM: Input text: {text}")

    # Allowed special characters
    allowed_chars = set(".-/'\"")

    # Replace character
    replace_char = " "

    # Convert to lowercase
    # text = text.lower()

    # Replace $ with "dollars"
    text = re.sub(r"\$", " dollars ", text)

    # Replace % with "percent"
    text = re.sub(r"%", " percent ", text)

    # Remove periods that are not between digits
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)

    # Remove unwanted characters
    text = re.sub(r"[^a-zA-Z0-9\s" + re.escape("".join(allowed_chars)) + "]", replace_char, text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize, remove stopwords, and apply lemmatization
    # words = text.split()

    # Remove stopwords & lemmatize
    # words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Lemmatize words
    # words = [lemmatizer.lemmatize(word) for word in words]

    # cleaned_text = " ".join(words)

    logger.debug(f"LLM: Cleaned text: {text}")

    return text


def clean_description(description: str) -> str:
    """
    Clean a description string by removing unwanted characters and formatting.

    Args:
        description (str): Input description.

    Returns:
        str: Cleaned description.
    """
    # Validate the input type
    if not isinstance(description, str):
        raise InvoiceProcessingError("Function clean_description: Input must be a string.")

    # Convert to uppercase
    description = description.upper()

    # Replace _, *, |, =, @, ^ with spaces
    description = re.sub(r"[_*|=@^]", " ", description)

    # Replace line breaks (\r, \n, or both) with a space
    description = re.sub(r"[\r\n]+", " ", description)

    # Remove leading and trailing spaces, '-' and '.'
    description = description.strip(" -.")

    # Replace multiple spaces with a single space
    description = re.sub(r"\s+", " ", description).strip()

    return description


def remove_separators(input_string: str, remove_dot: bool = True) -> str:
    """
    Remove '-' and '/' characters from the input string. Optionally remove '.' characters.

    Args:
        input_string (str): The string to process.
        remove_dot (bool, optional): Whether to remove '.' characters. Defaults to True.

    Returns:
        str: Processed string with specified separators removed.
    """
    # Validate the input type
    if not isinstance(input_string, str):
        raise InvoiceProcessingError("Function remove_separators: Input must be a string.")

    # Remove the specified characters using str.replace()
    result = input_string.replace("-", "").replace("/", "")
    if remove_dot:
        result = result.replace(".", "")

    return result


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame by removing vector columns and columns starting with '@'.

    Args:
        df (pd.DataFrame): DataFrame to clean.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Drop any column containing "vector"
    df = df.loc[:, ~df.columns.str.contains("vector")]

    # Drop any column starting with '@'
    df = df.loc[:, ~df.columns.str.startswith("@")]

    return df


def clean_dictionary(dictionary: Dict) -> Dict:
    """
    Clean a dictionary by removing keys with empty, null, or NaN values.
    Keys starting with '@' are assumed to be already removed during DataFrame cleaning.
    Converts enum values to their string representation.

    Args:
        dictionary (Dict): Dictionary to clean.

    Returns:
        Dict: Cleaned dictionary with enums converted to strings
    """
    if not dictionary:
        return {}

    # Create a new dictionary with only non-empty, non-null, non-NaN values
    cleaned_dict = {}

    for key, value in dictionary.items():
        # Handle nested dictionaries
        if isinstance(value, dict):
            cleaned_nested = clean_dictionary(value)
            if cleaned_nested:  # Only add if the cleaned nested dict is not empty
                cleaned_dict[key] = cleaned_nested

        # Handle enum values
        elif isinstance(value, Enum):
            cleaned_dict[key] = value.value

        # Handle empty strings, None, NaN
        elif value is not None and value != "" and not (isinstance(value, float) and pd.isna(value)):
            cleaned_dict[key] = value

    return cleaned_dict


def find_max_fitting_index(lines: list[bytes], max_bytes: int) -> int:
    """
    Finds the maximum index N such that the first N lines (in bytes)
    cumulatively do not exceed the given max_bytes limit.

    Performs a binary search to find the maximum number of lines that can be included
    without exceeding the specified byte size limit.

    Parameters:
        lines (list[bytes]): List of pre-encoded lines (e.g., JSONL lines as bytes).
        max_bytes (int): Maximum allowed cumulative size in bytes.

    Returns:
        int: The largest index `n` such that the first `n` lines fit within `max_bytes`.
    """
    left, right = 1, len(lines)
    best = 0

    while left <= right:
        mid = (left + right) // 2
        size = sum(len(line) for line in lines[:mid])

        if size <= max_bytes:
            best = mid
            left = mid + 1
        else:
            right = mid - 1

    return best


def remove_accents(input_str: str) -> str:
    """
    Removes diacritical marks from a string by converting them to their
    base Latin form.

    This is a hardened, hybrid implementation that handles environment-specific
    issues and special character transliterations using a highly efficient
    single-pass replacement method.
    """
    # Step 1: Define all special-cased character replacements.
    replacements = {"ß": "ss", "æ": "ae", "Æ": "AE", "ø": "o", "Ø": "O", "þ": "th", "Þ": "Th", "ð": "d", "Ð": "D"}

    # Step 2: Create a regular expression from the dictionary keys.
    # The pattern will be something like: 'ß|æ|Æ|ø|Ø...'
    pattern = re.compile("|".join(replacements.keys()))

    # Step 3: Run the standard unicode normalization first.
    nfkd_form = unicodedata.normalize("NFKD", input_str)

    # Step 4: Use the single-pass re.sub() to perform all replacements.
    # For each match, the lambda function looks up the replacement in the dict.
    replaced_form = pattern.sub(lambda m: replacements[m.group(0)], nfkd_form)

    # Step 5: Encode to ASCII to strip the remaining combining marks.
    return replaced_form.encode("ascii", "ignore").decode("utf-8")


def get_stage_and_sub_stage_names_from_log_stage(log_stage):
    names = log_stage.split(" - ")  # stage - sub stage
    return names[0], names[1]


def get_webapp_access_token(config, retry_count=3):
    """
    This method queries the azure to request a new access token.
    This token can be used to make API call to the Web App.

    Parameters:
        retry_count (int): Numbers of times to retry the API on failure.
    """
    token_headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}

    token_data = {
        "client_id": config.WA_AUDIENCE,
        "client_secret": config.WA_ACCESS_CLIENT_SECRET,
        "scope": config.WA_API_ACCESS_SCOPE,
        "grant_type": "client_credentials",
    }

    access_token_url = f"https://login.microsoftonline.com/{config.WA_TENANT_ID}/oauth2/v2.0/token"

    try:
        for attempt in range(retry_count):
            token_response = requests.post(access_token_url, headers=token_headers, data=token_data)
            if token_response.status_code == 200:
                access_token = token_response.json().get("access_token")
                return access_token

            if attempt < (retry_count - 1):
                logger.error(
                    f"Failed to create access token {token_response}, retrying for {retry_count - attempt - 1} more times"
                )
        else:
            # failure and all retries done.
            raise Exception(f"Failed to create access token - {token_response}")

    except Exception as e:
        logger.error(f"Failed to create access token , error - {str(e)}", exc_info=True)
        raise e
