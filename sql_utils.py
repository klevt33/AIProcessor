"""
## Overview
This module provides a set of asynchronous functions for interacting with invoice-related data in a database.
It includes functionality for fetching, processing, and updating invoice details, manufacturer data, parts data,
and tracking information. The module also includes utility functions for handling data deduplication,
batch processing, and cleaning.
"""

import json
from copy import deepcopy
from typing import Iterator, List, Literal

import pandas as pd

from constants import Constants, DatabaseObjects, DataStates, DescriptionCategories, Logs, TrainingDataVersions
from exceptions import InvoiceProcessingError
from logger import logger
from sdp import retry_on_deadlock
from utils import clean_description, get_current_datetime_cst


async def count_number_in_trainable_table(sdp, number: str, field: Literal["AKP", "UPC"]) -> int:
    """
    Checks if the UPC number or AK part number is already present in the training table.

    Args:
        sdp: Database connection/session object.
        number (str): UPC number or AK part number to look for.
        type (Literal): Which column to check ('AKP' or 'UPC').

    Returns:
        int: Count of matching records in the training table.
    """
    try:
        # Map column name based on type
        column_map = {Constants.AKP: "AKS_PRT_NUM", Constants.UPC: "UPC_CD"}

        if field not in column_map:
            raise ValueError(f"Invalid type provided: {field}")

        column_name = column_map[field]

        query = f"""
            SELECT COUNT(*) AS CNT
            FROM {DatabaseObjects.TBL_IVCE_XCTN_LLM_TRNL_MFR_REF} WITH (NOLOCK)
            WHERE {column_name} = '{number}'
        """

        df = await sdp.fetch_data(query)
        return int(df["CNT"][0])

    except Exception as e:
        logger.error(f"Error in query execution: {str(e)}")
        raise


# Function to fetch invoice data based on invoice header ID
async def get_invoice_data(sdp, invoice_header_id):
    """
    Fetches invoice data from the database for a given invoice header ID.

    Args:
        sdp: Database connection/session object.
        invoice_header_id (int): The ID of the invoice header to fetch data for.

    Returns:
        Dict[str, Any]: A dictionary containing invoice data.
    """
    df = None
    try:
        query = f"""SELECT *
                    FROM {DatabaseObjects.TBL_INVC_DTL} WITH (NOLOCK)
                    WHERE IVCE_HDR_ID={invoice_header_id}"""

        df = await sdp.fetch_data(query)

    except Exception as e:
        logger.error(f"Error in query execution: {str(e)}")
        raise e
    return df


async def get_invoice_duplicate_detail_ids(sdp, invoice_num: str, invoice_detail_id: int):
    """
    Fetches deduplicated invoice data from the database for a given invoice number.
    Groups duplicate line items (based on descriptive fields) and aggregates all detail IDs.

    Args:
        sdp: Database connection/session object.
        invoice_num (str): The invoice number to fetch data for.
        invoice_detail_id (int): The detail ID to fetch data for.

    Returns:
        pd.DataFrame: A DataFrame containing deduplicated invoice data with all detail IDs.
    """
    df = None
    try:
        # Shared column list
        columns = """
            ITM_LDSC,
            MFR_NM,
            MFR_PRT_NUM,
            IVCE_NUM
        """
        # IVCE_LNE_STAT - removed from condition

        query = f"""
        WITH Filtered AS (
            SELECT *
            FROM {DatabaseObjects.TBL_INVC_DTL} WITH (NOLOCK)
            WHERE IVCE_NUM = '{invoice_num}'
        ),
        DistinctDesc AS (
            SELECT
                {columns},
                STRING_AGG(CAST(IVCE_DTL_UID AS VARCHAR(20)), ',')
                    WITHIN GROUP (ORDER BY IVCE_DTL_UID) AS DTL_IDS
            FROM Filtered
            GROUP BY {columns}
        )
        SELECT
            {columns},
            DTL_IDS
        FROM DistinctDesc
        WHERE DTL_IDS like '%{invoice_detail_id}%';
        """

        df = await sdp.fetch_data(query)

        if df is not None and len(df) > 0:
            # Prepare IDs list
            dup_ids = [int(i) for i in df["DTL_IDS"].iloc[0].split(",")]

            # Delete current detail ID
            dup_ids.remove(invoice_detail_id)
            return dup_ids
        else:
            raise InvoiceProcessingError("Unable to check duplicate IDs")

    except Exception as e:
        logger.error(f"Error in query execution for IVCE_NUM={invoice_num} and IVCE_DTL_UID={invoice_detail_id}: {str(e)}")
        raise e


# Function to fetch detailed invoice data for a list of invoice detail IDs
async def get_invoice_detail_data(sdp, invoice_detail_ids):
    """
    Fetches detailed invoice data for a list of invoice detail IDs and identifies missing IDs.

    Args:
        sdp: Database connection/session object.
        invoice_detail_ids (List[int]): List of invoice detail IDs to fetch data for.

    Returns:
        Tuple[List[Dict[str, Any]], List[int]]:
            - List of dictionaries containing invoice detail data.
            - List of IDs that were not found in the database.
    """
    # Convert list to comma-separated string
    id_list_str = ", ".join(map(str, invoice_detail_ids))

    df = None
    try:
        query = f"""
            SELECT
                [IVCE_DTL_UID],
                [IVCE_NUM],
                [IVCE_HDR_ID],
                [ITM_LDSC],
                [CLN_MFR_AI_NM],
                [CLN_MFR_NM],
                [MFR_NM],
                [CLN_MFR_PRT_NUM],
                [MFR_PRT_NUM],
                [AKS_PRT_NUM],
                [UPC_CD],
                [RNTL_IND],
                [IVCE_VRFN_IND],
                [CTGY_ID]
            FROM {DatabaseObjects.TBL_INVC_DTL} WITH (NOLOCK)
            WHERE IVCE_DTL_UID IN ({id_list_str})
        """

        # query = f"""SELECT
        #                 dt.[IVCE_DTL_UID],
        #                 dt.[IVCE_HDR_ID],
        #                 dt.[ITM_LDSC],
        #                 dt.[CLN_MFR_AI_NM],
        #                 dt.[CLN_MFR_NM],
        #                 dt.[MFR_NM],
        #                 dt.[CLN_MFR_PRT_NUM],
        #                 dt.[MFR_PRT_NUM],
        #                 dt.[AKS_PRT_NUM],
        #                 dt.[UPC_CD],
        #                 hdr.[RNTL_IND]
        #             FROM {DatabaseObjects.TBL_INVC_DTL} AS dt WITH (NOLOCK)
        #             LEFT JOIN {DatabaseObjects.TBL_IVCE_HDR} AS hdr WITH (NOLOCK)
        #                 ON dt.IVCE_HDR_ID = hdr.IVCE_HDR_UID
        #             WHERE dt.IVCE_DTL_UID IN ({id_list_str})"""

        df = await sdp.fetch_data(query)

        # Extract returned IDs from the DataFrame
        returned_ids = set(df["IVCE_DTL_UID"].tolist())

        # Find missing IDs
        input_ids_set = set(invoice_detail_ids)
        missing_ids = input_ids_set - returned_ids

    except Exception as e:
        logger.error(f"Error in query execution: {str(e)}")
        raise e
    return df, missing_ids


async def get_classifier_categories_data(sdp):
    """
    Fetches classifier category details and mapping with their parents.
    If parent is Null means it doesn't have any parent class.
    """
    df = None
    try:
        query = f"SELECT * FROM {DatabaseObjects.TBL_IVCE_XCTN_CLSFR_CTGY_DTL} WITH (NOLOCK)"
        df = await sdp.fetch_data(query)

    except Exception as e:
        logger.error(f"Error in query execution: {str(e)}")
        raise e
    return df


# Function to fetch manufacturer data
async def get_manufacturer_data(sdp):
    """
    Fetches manufacturer data from the database, including unclean and clean names,
    parent company name, and AI match indicator.

    Args:
        sdp: Database connection/session object.

    Returns:
        List[Dict[str, Any]] | pd.DataFrame: A list of dictionaries or DataFrame
                                             containing manufacturer data.
                                     Columns: UncleanName, CleanName,
                                              ParentCompanyName, AIMatchIndicator
                                     Returns an empty DataFrame with these columns
                                     if query fails.
    """
    try:
        # --- UPDATE: Include PRNT_CMPY_NM and AI_MCH_IND in the SELECT ---
        # --- UPDATE: Use AIMatchIndicator as the alias for AI_MCH_IND ---
        query = (
            "SELECT MFR_NM as UncleanName, "
            "       CLN_MFR_NM as CleanName, "
            "       PRNT_CMPY_NM as ParentCompanyName, "
            "       AI_MCH_IND as AIMatchIndicator "
            f"FROM {DatabaseObjects.TBL_MFR_DTL} WITH (NOLOCK) "
            "WHERE (AI_MCH_IND IS NULL OR AI_MCH_IND = '' OR AI_MCH_IND <> 'N') "
            "ORDER BY ISNULL(AI_MCH_IND, '')"
        )
        df = await sdp.fetch_data(query)

        expected_columns = {"UncleanName", "CleanName", "ParentCompanyName", "AIMatchIndicator"}
        if isinstance(df, pd.DataFrame):
            if not expected_columns.issubset(df.columns):
                logger.warning(
                    f"get_manufacturer: Expected columns {expected_columns} not fully present in result. "
                    f"Got columns: {df.columns.tolist() if not df.empty else '[]'}"
                )
                # Return empty DataFrame with expected structure
                return pd.DataFrame(columns=list(expected_columns))

        return df

    except Exception as e:
        logger.error(f"Error fetching manufacturer  {str(e)}", exc_info=True)
        # Return empty DataFrame with required columns if query fails
        # --- UPDATE: Match the new column list in the return ---
        return pd.DataFrame(columns=["UncleanName", "CleanName", "ParentCompanyName", "AIMatchIndicator"])


# Function to fetch parts data based on manufacturer part numbers
async def get_parts_data(sdp, mfr_part_numbers, active_only=False):
    """
    Fetch parts data from the database based on manufacturer part numbers.

    Args:
        sdp: An instance of the SDP class for database access
        mfr_part_numbers: List of Manufacturer Part Numbers (list of strings)
        active_only: Boolean, if True, only fetch active descriptions (default: False)

    Returns:
        DataFrame containing parts data

    Raises:
        Exception: If there's an error fetching data from the database
    """

    def get_empty_dataframe():
        columns = [
            "ItemID",
            "MfrPartNum",
            "MfrName",
            "UPC",
            "UNSPSC",
            "AKPartNum",
            "DescriptionID",
            "ItemDescription",
            "ItemSourceName",
            "DescSourceName",
        ]
        return pd.DataFrame(columns=columns)

    # Handle empty or invalid mfr_part_numbers
    if not mfr_part_numbers or not isinstance(mfr_part_numbers, list):
        return get_empty_dataframe()

    # Filter out None or empty strings from the input list
    mfr_part_numbers = [pn.strip() for pn in mfr_part_numbers if pn and isinstance(pn, str)]

    # If no valid part numbers remain after filtering, return an empty DataFrame
    if not mfr_part_numbers:
        return get_empty_dataframe()

    # Build the part numbers string for the SQL query
    part_numbers_str = ", ".join(f"'{pn}'" for pn in mfr_part_numbers)

    # Build the JOIN condition with optional active_only filter
    join_condition = "i.IVCE_XCTN_LLM_TRNL_MFR_REF_UID = d.IVCE_XCTN_LLM_TRNL_MFR_REF_UID"
    if active_only:
        join_condition += " AND d.REC_ACTV_IND  = 'Y'"

    # Construct the SQL query
    query = f"""
    SELECT
        i.IVCE_XCTN_LLM_TRNL_MFR_REF_UID as ItemID,
        i.MFR_PRT_NUM as MfrPartNum,
        i.MFR_NM as MfrName,
        i.UPC_CD as UPC,
        i.UNSPSC_CD as UNSPSC,
        i.AKS_PRT_NUM as AKPartNum,
        i.SRC_NM as ItemSourceName,
        d.IVCE_XCTN_LLM_TRNL_PRDT_REF_UID as DescriptionID,
        d.IVCE_PRDT_LDSC as ItemDescription,
        d.SRC_NM as DescSourceName
    FROM {DatabaseObjects.TBL_IVCE_XCTN_LLM_TRNL_MFR_REF} i WITH (NOLOCK)
    LEFT JOIN {DatabaseObjects.TBL_IVCE_XCTN_LLM_TRNL_PRDT_REF} d WITH (NOLOCK)
        ON {join_condition}
    WHERE i.MFR_PRT_NUM IN ({part_numbers_str})
    """

    # Execute the query and return the result
    # Let exceptions propagate to be caught by the calling function
    df = await sdp.fetch_data(query)
    return df


async def get_all_index_data(sdp, batch_size=1000, active_only=False):
    """
    Fetch all data needed for the search index with support for batching and deduplication.

    Args:
        sdp: An instance of the SDP class for database access
        batch_size: Number of records to fetch per batch (default: 1000)
        active_only: Boolean, if True, only fetch active descriptions (default: False)

    Returns:
        A tuple containing:
        - A generator yielding DataFrames containing batches of unique records
        - The total number of records before deduplication
    """
    # Build the JOIN condition
    join_condition = "i.IVCE_XCTN_LLM_TRNL_MFR_REF_UID = d.IVCE_XCTN_LLM_TRNL_MFR_REF_UID"

    # Basic filter for non-empty descriptions
    where_clause = "WHERE d.IVCE_PRDT_LDSC IS NOT NULL AND d.IVCE_PRDT_LDSC <> ''"

    # # Add filter for non-empty UNSPSC
    # where_clause += " AND i.UNSPSC_CD IS NOT NULL AND i.UNSPSC_CD <> ''"

    # Add active_only filter to WHERE clause if needed
    if active_only:
        where_clause += " AND d.REC_ACTV_IND = 'Y'"

    # Count total records to process
    count_query = f"""
    SELECT COUNT(*) as total_count
    FROM {DatabaseObjects.TBL_IVCE_XCTN_LLM_TRNL_MFR_REF} i WITH (NOLOCK)
    INNER JOIN {DatabaseObjects.TBL_IVCE_XCTN_LLM_TRNL_PRDT_REF} d WITH (NOLOCK)
        ON {join_condition}
    {where_clause}
    """

    df_count = await sdp.fetch_data(count_query)
    total_count = df_count["total_count"].iloc[0]

    logger.debug(f"Total records to process: {total_count}")

    return batch_processor(sdp, join_condition, where_clause, batch_size), total_count


# Function to strip whitespace from specified columns in a DataFrame
def strip_dataframe_columns(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Applies the strip() method to specified columns of a Pandas DataFrame in-place.
    Handles potential non-string values or NaNs gracefully.

    Args:
        df: The Pandas DataFrame to modify.
        columns: A list of column names to apply stripping to.
    """
    for col in columns:
        if col in df.columns:
            # Check if the column could potentially contain strings
            if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == "object":
                # Use the vectorized .str.strip() which handles NaN/None safely
                df[col] = df[col].str.strip()


async def batch_processor(sdp, join_condition, where_clause, batch_size):
    """
    Processes data in batches with buffering to handle deduplication across batches.
    This generator fetches data in chunks larger than the requested batch_size to buffer
    records with the same ItemID, ensuring all records for a given ItemID are processed together

    Args:
        sdp: Database connection/session object.
        join_condition (str): SQL join condition for the query.
        where_clause (str): SQL where clause for filtering data.
        batch_size (int): Number of records to process per batch.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing processed data.
    """
    # Use a larger internal batch size to reduce database calls
    internal_batch_size = batch_size * 5
    internal_batch_size = 5000 if internal_batch_size < 5000 else internal_batch_size
    offset = 0
    buffer = pd.DataFrame()

    string_columns_to_strip = ["MfrPartNum", "MfrName", "UPC", "UNSPSC", "AKPartNum"]

    while True:
        # Construct the SQL query for the current internal batch
        query = f"""
        SELECT
            i.IVCE_XCTN_LLM_TRNL_MFR_REF_UID as ItemID,
            i.MFR_PRT_NUM as MfrPartNum,
            i.MFR_NM as MfrName,
            i.UPC_CD as UPC,
            i.UNSPSC_CD as UNSPSC,
            i.AKS_PRT_NUM as AKPartNum,
            i.SRC_NM as ItemSourceName,
            d.IVCE_XCTN_LLM_TRNL_PRDT_REF_UID as DescriptionID,
            d.IVCE_PRDT_LDSC as ItemDescription,
            d.SRC_NM as DescSourceName,
            i.REC_UPDD_DTTM as ItemLastModified,
            d.REC_UPDD_DTTM as DescLastModified
        FROM {DatabaseObjects.TBL_IVCE_XCTN_LLM_TRNL_MFR_REF} i WITH (NOLOCK)
        INNER JOIN {DatabaseObjects.TBL_IVCE_XCTN_LLM_TRNL_PRDT_REF} d WITH (NOLOCK)
            ON {join_condition}
        {where_clause}
        ORDER BY i.IVCE_XCTN_LLM_TRNL_MFR_REF_UID, CAST(d.IVCE_XCTN_LLM_TRNL_PRDT_REF_UID AS VARCHAR(20))
        OFFSET {offset} ROWS
        FETCH NEXT {internal_batch_size} ROWS ONLY
        """

        # Execute the query
        df = await sdp.fetch_data(query)

        # If no data returned, we're done
        if df.empty:
            # Yield any remaining buffered data
            if not buffer.empty:
                deduplicated = deduplicate_descriptions(buffer)
                # Process the remaining buffer in batch_size chunks
                for i in range(0, len(deduplicated), batch_size):
                    yield deduplicated.iloc[i : i + batch_size]
            break

        # Convert ItemID to integer
        df["ItemID"] = df["ItemID"].astype(int)

        # Cast DescriptionID to match string type in the index
        df["DescriptionID"] = df["DescriptionID"].astype(str)

        # Apply stripping to specified columns
        strip_dataframe_columns(df, string_columns_to_strip)

        # Clean the ItemDescription field and add a cleaned version for deduplication
        df["ItemDescription"] = df["ItemDescription"].apply(clean_description)

        # Append new data to buffer
        buffer = pd.concat([buffer, df])

        # Process complete ItemID groups from the buffer
        processed_buffer, remaining_buffer = extract_complete_item_groups(buffer, df)
        buffer = remaining_buffer

        if not processed_buffer.empty:
            # Deduplicate the processed buffer
            deduplicated = deduplicate_descriptions(processed_buffer)

            # Yield the deduplicated data in batch_size chunks
            for i in range(0, len(deduplicated), batch_size):
                yield deduplicated.iloc[i : i + batch_size]

        # Update offset for next query
        offset += internal_batch_size


def extract_complete_item_groups(buffer, current_batch):
    """
    Extract complete ItemID groups from the buffer.

    A complete group is one where all records for a given ItemID are in the buffer
    and not split across batches.

    Args:
        buffer: DataFrame containing all buffered records
        current_batch: DataFrame containing the most recently fetched batch

    Returns:
        tuple: (processed_buffer, remaining_buffer)
    """
    # Get the set of ItemIDs in the current batch
    current_batch_item_ids = set(current_batch["ItemID"].unique())

    # Get all unique ItemIDs in the buffer
    all_buffer_item_ids = set(buffer["ItemID"].unique())

    # Find ItemIDs that are complete (not in the current batch)
    complete_item_ids = all_buffer_item_ids - current_batch_item_ids

    # Extract complete groups
    if complete_item_ids:
        mask = buffer["ItemID"].isin(complete_item_ids)
        processed_buffer = buffer[mask].copy()
        remaining_buffer = buffer[~mask].copy()
        return processed_buffer, remaining_buffer
    else:
        # No complete groups yet
        return pd.DataFrame(), buffer


def deduplicate_descriptions(df):
    """
    Deduplicate records by keeping only unique ItemID and ItemDescription combinations.

    For each ItemID, keep only the first occurrence of each unique cleaned description.
    """
    # Sort by ItemID and DescriptionID to ensure consistent results
    df = df.sort_values(["ItemID", "DescriptionID"])

    # Group by ItemID and ItemDescription and keep the first occurrence
    df = df.drop_duplicates(subset=["ItemID", "ItemDescription"], keep="first")

    return df


def update_final_mfr_nm_to_update_in_sdp(data):
    if Logs.CATEGORY in data and Logs.MFR_NAME in data:
        mfr_nm = data[Logs.MFR_NAME]
        category = data[Logs.CATEGORY]

        # If Class is BAD, write UNCLASSIFIED
        if category == DescriptionCategories.BAD:
            mfr_nm = DescriptionCategories.UNCLASSIFIED

        if category == DescriptionCategories.LOT:
            if data[Logs.MFR_NAME] == Constants.UNDEFINED or not data[Logs.IS_MFR_CLEAN]:
                mfr_nm = DescriptionCategories.LOT
                data[Logs.MFR_NAME] = DescriptionCategories.LOT
                data[Logs.IS_MFR_CLEAN] = True

        # If Class is not MATERIAL or LOT, write class in MFR_NM
        if category not in [DescriptionCategories.MATERIAL, DescriptionCategories.LOT]:
            mfr_nm = category

        # Uncomment this portion to avoid writing UNDEFINED for non-clean mfr in invoice table
        # else:
        #     mfr_nm = data[Logs.MFR_NAME]

        # Comment this
        if data[Logs.IS_MFR_CLEAN]:
            mfr_nm = data[Logs.MFR_NAME]
        else:
            mfr_nm = Constants.UNDEFINED

        # Update the value so that auto-population gets it
        data[Logs.MFR_NAME] = mfr_nm

    elif Logs.MFR_NAME in data:
        if not data[Logs.IS_MFR_CLEAN]:
            data[Logs.MFR_NAME] = Constants.UNDEFINED

    elif Logs.CATEGORY in data:
        data[Logs.IS_MFR_CLEAN] = True
        category = data[Logs.CATEGORY]
        mfr_nm = deepcopy(category)

        # If Class is BAD, write UNCLASSIFIED
        if category == DescriptionCategories.BAD:
            mfr_nm = DescriptionCategories.UNCLASSIFIED
        elif category == DescriptionCategories.MATERIAL:
            del data[Logs.IS_MFR_CLEAN]
            mfr_nm = Constants.UNDEFINED

        # Update the value so that auto-population gets it
        data[Logs.MFR_NAME] = mfr_nm


def _normalize_stage_results(stage_results: dict | object) -> dict:
    """
    A helper function to ensure stage_results is always a consistent dictionary.

    It handles two cases:
    1. A FLAT dictionary read from Cosmos DB (the writer service path).
    2. A NESTED StageResults object from the live AI pipeline (the API path).

    It always returns a dictionary with 'final_results', 'status', and 'message' keys.
    """
    if isinstance(stage_results, dict):
        # The incoming dictionary from Cosmos DB IS the final results.
        # We wrap it in the structure that the downstream functions expect.
        result = {
            "final_results": stage_results,
            "status": stage_results.get("status", "success"),
            "message": stage_results.get("message", ""),
        }
    else:
        # It's a StageResults object, so extract the nested .final_results attribute.
        result = {
            "final_results": getattr(stage_results, "final_results", {}),
            "status": getattr(stage_results, "status", "success"),
            "message": getattr(stage_results, "message", ""),
        }

    return result


# Function to generate query and values for updating invoice details
async def get_invoice_details_query_and_values(invoice_detail_id: int, normalized_results: dict):
    """
    Dynamically builds an UPDATE query for invoice detail.
    Includes only fields present in `data`, always includes audit fields and WHERE condition.

    Args:
        invoice_detail (int): The invoice detail object containing the unique ID.
        normalized_results (dict): The stage results containing final results and other metadata.

    Returns:
        tuple: A tuple containing the SQL query string and the values to be used in the query.
    """
    # data = deepcopy(stage_results.final_results)
    data = deepcopy(normalized_results["final_results"])

    # Fix issue with this call
    if Logs.MFR_NAME in data or Logs.CATEGORY_ID in data:
        update_final_mfr_nm_to_update_in_sdp(data)

    # Map Logs/keys to DB columns
    db_fields = {
        Logs.MFR_NAME: ("CLN_MFR_AI_NM", str),
        Logs.PRT_NUM: ("CLN_MFR_PRT_NUM", str),
        Logs.AKS_PRT_NUM: ("CLN_AKS_PRT_NUM", str),
        Logs.UPC: ("CLN_UPC_NUM", str),
        Logs.DESCRIPTION: ("CLN_DESC_LDSC", str),
        Logs.UNSPSC: ("CLN_UNSPSC_CD", str),
        Logs.CATEGORY_ID: ("CTGY_ID", str),
        Logs.IVCE_LINE_STATUS: ("IVCE_LNE_STAT", str),
        Constants.IS_RENTAL: ("RNTL_IND", str),
        Logs.IS_VERIFIED: ("IVCE_VRFN_IND", str),
    }

    set_parts = []
    values = []

    # Add only fields that exist in data
    for key, (col, transform) in db_fields.items():
        if key in data and data[key] is not None:
            set_parts.append(f"{col} = ?")
            values.append(transform(data[key]))

    # Always include audit fields
    set_parts.append("REC_UPDD_DTTM = GETDATE()")
    set_parts.append("REC_UPDD_BY_ID = ?")
    values.append(DataStates.AI_USER)

    # Add WHERE
    where_clause = "IVCE_DTL_UID = ?"
    values.append(invoice_detail_id)

    # Build query
    query = f"""
    UPDATE {DatabaseObjects.TBL_INVC_DTL}
    SET {', '.join(set_parts)}
    WHERE {where_clause};
    """
    return query, tuple(values)


# Function to generate query and values for updating invoice tracking
async def get_invoice_tracking_query_and_values(
    invoice_detail_id: int, normalized_results: dict, is_duplicate: bool, parent_detail_id: int
):
    """
    Prepares dynamic UPDATE and INSERT queries for a safe "upsert" operation.

    This function generates the necessary SQL and parameters for a two-step
    UPDATE-then-INSERT pattern. When executed within a single transaction, this
    provides a highly reliable and predictable method for creating or updating records.
    """
    # data = deepcopy(stage_results.final_results)
    data = deepcopy(normalized_results["final_results"])
    status = normalized_results["status"]
    message = normalized_results["message"]

    if is_duplicate:
        data.update({Logs.PRNT_DTL_ID: parent_detail_id})

    # status = stage_results.status

    optional_ai_fields = {
        Logs.DESCRIPTION: "CLN_LDSC",
        Logs.CATEGORY: "CLN_CTGRY",
        Logs.CATEGORY_ID: "CTGY_ID",
        Logs.CONF_CATEGORY: "CLN_CTGRY_CONF",
        Logs.MFR_NAME: "CLN_MFR_NM",
        Logs.IS_MFR_CLEAN: "IS_MFR_CLEAN",
        Logs.PRT_NUM: "CLN_MFR_PRT_NUM",
        Logs.UNSPSC: "CLN_UNSPSC",
        Logs.UPC: "CLN_UPC",
        Logs.AKS_PRT_NUM: "CLN_AKS_PRT_NUM",
        Constants.IS_RENTAL: "IS_RENTAL",
        Logs.IS_VERIFIED: "IVCE_VRFN_IND",
    }

    # Dynamically build AI data dict
    ai_data = {}
    for key, out_key in optional_ai_fields.items():
        if key in data and data[key] is not None:
            ai_data[out_key] = data[key]

    ai_data["STATUS"] = status

    if status == Constants.ERROR_lower:
        # ai_data["MESSAGE"] = stage_results.message
        ai_data["MESSAGE"] = message

    ai_data_str = json.dumps(ai_data, default=str)

    # Define your optional db columns -> python params: (DB columns, transformation func)
    optional_db_fields = {
        Logs.CONF_MFR_NAME: ("IVCE_CLN_MFR_NM_CONFDC_PERC", int),
        Logs.CONF_PRT_NUM: ("IVCE_CLN_MFR_PRT_NUM_CONFDC_PERC", int),
        Logs.CONF_UNSPSC: ("IVCE_CLN_UNSPSC_CONFDC_PERC", int),
        Logs.STAGE_MFR_NAME: ("IVCE_CLN_MFR_NM_IDFN_STEP_NM", lambda v: "AI-" + v),
        Logs.STAGE_PRT_NUM: ("IVCE_CLN_MFR_PRT_NUM_IDFN_STEP_NM", lambda v: "AI-" + v),
        Logs.STAGE_AKS_PRT_NUM: ("IVCE_CLN_AKS_PRT_NUM_IDFN_STEP_NM", lambda v: "AI-" + v),
        Logs.STAGE_UPC: ("IVCE_CLN_UPC_IDFN_STEP_NM", lambda v: "AI-" + v),
        Logs.STAGE_UNSPSC: ("IVCE_CLN_UNSPSC_IDFN_STEP_NM", lambda v: "AI-" + v),
        Logs.STAGE_CATEGORY: ("IVCE_CLN_CTGY_IDFN_STEP_NM", lambda v: "AI-" + v),
        Logs.WEB_SEARCH_URL: ("IVCE_AI_SRCH_URL_LDSC", str),
        Logs.PRNT_DTL_ID: ("PRNT_IVCE_DTL_ID", str),
    }

    # -------------------
    # UPDATE query parts
    # -------------------
    update_parts = []
    update_values = []

    for key, (col, transform) in optional_db_fields.items():
        if key in data and data[key] is not None:
            update_parts.append(f"[{col}] = ?")
            update_values.append(transform(data[key]))

    # Always include payload + audit
    update_parts.append("[IVCE_AI_XCTD_DAT_DESC] = ?")
    update_values.append(ai_data_str)
    update_parts.append("[REC_UPDD_DTTM] = GETDATE()")
    update_parts.append("[REC_UPDD_BY_ID] = ?")
    update_values.append(DataStates.AI_USER)

    update_query = f"""
    UPDATE {DatabaseObjects.TBL_IVCE_TRKG_MSTR} WITH (UPDLOCK, HOLDLOCK, ROWLOCK)
    SET {', '.join(update_parts)}
    WHERE IVCE_DTL_ID = ?
    """

    update_params = [*update_values, str(invoice_detail_id)]

    # -------------------
    # INSERT query parts
    # -------------------
    insert_cols = ["IVCE_DTL_ID"]
    insert_vals = ["?"]
    insert_values = [str(invoice_detail_id)]

    for key, (col, transform) in optional_db_fields.items():
        if key in data and data[key] is not None:
            insert_cols.append(f"[{col}]")
            insert_vals.append("?")
            insert_values.append(transform(data[key]))

    # Always include payload + audit
    insert_cols.append("[IVCE_AI_XCTD_DAT_DESC]")
    insert_vals.append("?")
    insert_values.append(ai_data_str)

    insert_cols.extend(["[REC_CRTD_DTTM]", "[REC_CRTD_BY_ID]", "[REC_UPDD_DTTM]", "[REC_UPDD_BY_ID]"])
    insert_vals.extend(["GETDATE()", "?", "GETDATE()", "?"])
    insert_values.extend([DataStates.AI_USER, DataStates.AI_USER])

    insert_query = f"""
    INSERT INTO {DatabaseObjects.TBL_IVCE_TRKG_MSTR}
        ({', '.join(insert_cols)})
    VALUES ({', '.join(insert_vals)})
    """

    # -------------------
    # Return both
    # -------------------
    return update_query, tuple(update_params), insert_query, tuple(insert_values)


# @retry_on_deadlock(max_retries=3, backoff_factor=1.0)
# async def update_invoice_detail_and_tracking_values_by_id(
#     sdp, invoice_detail_id: int, stage_results: dict, is_duplicate=False, parent_detail_id=None
# ):
#     """
#     Atomically updates both the invoice details and invoice tracking tables
#     for a given invoice_detail_id using the provided data.
#     """
#     try:
#         # Prepare queries and values
#         ivce_dtl_query, ivce_dtl_values = await get_invoice_details_query_and_values(invoice_detail_id, stage_results)

#         update_q, update_p, insert_q, insert_p = await get_invoice_tracking_query_and_values(
#             invoice_detail_id, stage_results, is_duplicate, parent_detail_id
#         )

#         # Execute both updates in a single transaction
#         with sdp.sqlalchemy_engine.begin() as conn:
#             # Step 1: Update the main invoice detail table
#             await sdp.update_data(ivce_dtl_query, ivce_dtl_values, conn=conn, retry=False)
#             logger.info(f"Updated invoice detail {invoice_detail_id} successfully.")

#             # Step 2: Execute the robust UPDATE/INSERT pattern for the tracking table
#             # Attempt to UPDATE first.
#             update_result = await sdp.update_data(update_q, update_p, conn=conn, retry=False)

#             # If no rows were affected by the update, the record doesn't exist.
#             # We can now safely INSERT it.
#             if update_result.rowcount == 0:
#                 await sdp.update_data(insert_q, insert_p, conn=conn, retry=False)

#             logger.info(f"Upserted invoice tracking details for {invoice_detail_id} successfully.")

#     except Exception as e:
#         logger.error(f"Error during atomic update for invoice detail {invoice_detail_id}: {str(e)}")
#         raise


@retry_on_deadlock(max_retries=3, backoff_factor=2.0)
async def update_invoice_detail_and_tracking_values_by_id(
    sdp, invoice_detail_id: int, stage_results: dict, is_duplicate=False, parent_detail_id=None, conn=None
):
    """
    Atomically updates both invoice details and tracking tables.

    This function is designed for flexibility:
    - If a 'conn' (SQLAlchemy connection) is provided, it will execute its
      operations within that existing transaction. This is essential for batch processing.
    - If 'conn' is None, it will create and manage its own self-contained transaction.
    """
    try:
        # 1. Prepare all necessary SQL queries and parameters (no changes here).
        # Normalize the data ONCE at the beginning of the process.
        normalized_results = _normalize_stage_results(stage_results)

        # Pass the ALREADY NORMALIZED data to the helper functions.
        ivce_dtl_query, ivce_dtl_values = await get_invoice_details_query_and_values(invoice_detail_id, normalized_results)
        update_q, update_p, insert_q, insert_p = await get_invoice_tracking_query_and_values(
            invoice_detail_id, normalized_results, is_duplicate, parent_detail_id
        )

        # Define a helper function to perform the database writes.
        async def perform_writes(connection):
            await sdp.update_data(ivce_dtl_query, ivce_dtl_values, conn=connection, retry=False)

            update_result = await sdp.update_data(update_q, update_p, conn=connection, retry=False)
            if update_result.rowcount == 0:
                await sdp.update_data(insert_q, insert_p, conn=connection, retry=False)

        # 2. Decide whether to use the provided transaction or create a new one.
        if conn:
            # A connection was passed in, so execute within that existing transaction.
            await perform_writes(conn)
        else:
            # No connection was passed, so create a new self-contained transaction.
            with sdp.sqlalchemy_engine.begin() as new_conn:
                await perform_writes(new_conn)

        if is_duplicate:
            logger.info(f"Upserted DUPLICATE invoice details for {invoice_detail_id} (Parent: {parent_detail_id}) successfully.")
        else:
            logger.info(f"Upserted PARENT invoice details for {invoice_detail_id} successfully.")

    except Exception as e:
        logger.error(f"Error during atomic update for invoice detail {invoice_detail_id}: {str(e)}")
        raise


async def get_invoice_tracking_upsert_queries_and_values(
    invoice_detail_id: int, stage_results: dict, is_duplicate: bool, parent_detail_id: int
):
    """
    Prepares dynamic UPDATE and INSERT queries for a safe and atomic upsert operation.

    This function generates the necessary SQL and parameters to first attempt an
    UPDATE on a record. If no rows are affected (meaning the record does not
    exist), a subsequent INSERT can be performed. This two-step pattern, when
    wrapped in a transaction, provides a highly reliable and predictable method
    for creating or updating records, avoiding concurrency issues.

    Args:
        invoice_detail_id (int): The unique ID of the invoice detail.
        stage_results (dict): The results from the processing stages, containing
                              final data and metadata.
        is_duplicate (bool): Flag indicating if the line is a duplicate.
        parent_detail_id (int): The parent detail ID if it's a duplicate.

    Returns:
        tuple: A tuple containing four elements:
               - The parameterized UPDATE query string.
               - A tuple of parameters for the UPDATE query.
               - The parameterized INSERT query string.
               - A tuple of parameters for the INSERT query.
    """
    data = deepcopy(stage_results.final_results)
    if is_duplicate:
        data.update({Logs.PRNT_DTL_ID: parent_detail_id})

    # status = stage_results.status
    status = stage_results["status"]

    # --- 1. Prepare the JSON payload for the tracking column ---
    # This dictionary contains a detailed snapshot of the AI processing results.
    optional_ai_fields = {
        Logs.DESCRIPTION: "CLN_LDSC",
        Logs.CATEGORY: "CLN_CTGRY",
        Logs.CATEGORY_ID: "CTGY_ID",
        Logs.CONF_CATEGORY: "CLN_CTGRY_CONF",
        Logs.MFR_NAME: "CLN_MFR_NM",
        Logs.IS_MFR_CLEAN: "IS_MFR_CLEAN",
        Logs.PRT_NUM: "CLN_MFR_PRT_NUM",
        Logs.UNSPSC: "CLN_UNSPSC",
        Logs.UPC: "CLN_UPC",
        Logs.AKS_PRT_NUM: "CLN_AKS_PRT_NUM",
        Constants.IS_RENTAL: "IS_RENTAL",
        Logs.IS_VERIFIED: "IVCE_VRFN_IND",
    }

    # Dynamically build AI data dict
    ai_data = {}
    for key, out_key in optional_ai_fields.items():
        if key in data and data[key] is not None:
            ai_data[out_key] = data[key]

    ai_data["STATUS"] = status

    if status == Constants.ERROR_lower:
        ai_data["MESSAGE"] = stage_results.message

    ai_data_str = json.dumps(ai_data, default=str)

    # --- 2. Define mappings from input data keys to database columns ---
    # This allows for dynamically building queries with only the available data.
    optional_db_fields = {
        Logs.CONF_MFR_NAME: ("IVCE_CLN_MFR_NM_CONFDC_PERC", int),
        Logs.CONF_PRT_NUM: ("IVCE_CLN_MFR_PRT_NUM_CONFDC_PERC", int),
        Logs.CONF_UNSPSC: ("IVCE_CLN_UNSPSC_CONFDC_PERC", int),
        Logs.STAGE_MFR_NAME: ("IVCE_CLN_MFR_NM_IDFN_STEP_NM", lambda v: "AI-" + v),
        Logs.STAGE_PRT_NUM: ("IVCE_CLN_MFR_PRT_NUM_IDFN_STEP_NM", lambda v: "AI-" + v),
        Logs.STAGE_AKS_PRT_NUM: ("IVCE_CLN_AKS_PRT_NUM_IDFN_STEP_NM", lambda v: "AI-" + v),
        Logs.STAGE_UPC: ("IVCE_CLN_UPC_IDFN_STEP_NM", lambda v: "AI-" + v),
        Logs.STAGE_UNSPSC: ("IVCE_CLN_UNSPSC_IDFN_STEP_NM", lambda v: "AI-" + v),
        Logs.STAGE_CATEGORY: ("IVCE_CLN_CTGY_IDFN_STEP_NM", lambda v: "AI-" + v),
        Logs.WEB_SEARCH_URL: ("IVCE_AI_SRCH_URL_LDSC", str),
        Logs.PRNT_DTL_ID: ("PRNT_IVCE_DTL_ID", str),
    }

    # --- 3. Build the UPDATE Query ---
    update_parts = []
    update_values = []
    for key, (col, transform) in optional_db_fields.items():
        if key in data and data[key] is not None:
            update_parts.append(f"[{col}] = ?")
            update_values.append(transform(data[key]))

    # Always include the JSON payload and standard audit fields in the update
    update_parts.extend(["[IVCE_AI_XCTD_DAT_DESC] = ?", "[REC_UPDD_DTTM] = GETDATE()", "[REC_UPDD_BY_ID] = ?"])
    update_values.extend([ai_data_str, DataStates.AI_USER])

    # The locking hints (UPDLOCK, HOLDLOCK, ROWLOCK) are critical for preventing
    # deadlocks and race conditions in a concurrent environment.
    update_query = f"""
    UPDATE {DatabaseObjects.TBL_IVCE_TRKG_MSTR} WITH (UPDLOCK, HOLDLOCK, ROWLOCK)
    SET {', '.join(update_parts)}
    WHERE IVCE_DTL_ID = ?
    """
    update_params = tuple(update_values + [str(invoice_detail_id)])

    # --- 4. Build the INSERT Query ---
    insert_cols = ["IVCE_DTL_ID"]
    insert_placeholders = ["?"]
    insert_values = [str(invoice_detail_id)]
    for key, (col, transform) in optional_db_fields.items():
        if key in data and data[key] is not None:
            insert_cols.append(f"[{col}]")
            insert_placeholders.append("?")
            insert_values.append(transform(data[key]))

    # Always include the JSON payload and standard audit fields in the insert
    insert_cols.extend(["[IVCE_AI_XCTD_DAT_DESC]", "[REC_CRTD_DTTM]", "[REC_CRTD_BY_ID]", "[REC_UPDD_DTTM]", "[REC_UPDD_BY_ID]"])
    insert_placeholders.extend(["?", "GETDATE()", "?", "GETDATE()", "?"])
    insert_values.extend([ai_data_str, DataStates.AI_USER, DataStates.AI_USER])

    insert_query = f"""
    INSERT INTO {DatabaseObjects.TBL_IVCE_TRKG_MSTR} ({', '.join(insert_cols)})
    VALUES ({', '.join(insert_placeholders)})
    """
    insert_params = tuple(insert_values)

    return update_query, update_params, insert_query, insert_params


async def bulk_update_invoice_line_status(sdp, invoice_detail_ids, new_status):
    """
    Bulk updates the invoice detail rows with new line status

    Args:
        invoice_detail_ids (list): List of detail IDs
        new_status (str): New status to updated set to.
    """
    try:
        placeholders = ",".join("?" for _ in invoice_detail_ids)

        query = f"""
            UPDATE {DatabaseObjects.TBL_INVC_DTL}
            SET IVCE_LNE_STAT = ?
            WHERE DTL_UID IN ({placeholders})
        """

        # First param is status, rest are DTL_UIDs
        values = (new_status, *invoice_detail_ids)

        # Call your existing update_data function
        await sdp.update_data(query, values)

    except Exception as e:
        logger.error(f"Error during update for invoice detail IDs {invoice_detail_ids}: {str(e)}")


async def get_training_data(sdp, training_data_version: str = TrainingDataVersions.NEW) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """
    Returns the dataframe of the training data.

    Parameters:
        num_rows (int): Number of rows to be considered for training.
        training_data_version (str): the rows of this value will be picked for training.
    """
    df = None
    try:
        query = f"""
                    SELECT
                        d.[IVCE_PRDT_LDSC],
                        i.[MFR_PRT_NUM],
                        i.[MFR_NM],
                        i.[UNSPSC_CD],
                        d.[IVCE_XCTN_LLM_TRNL_PRDT_REF_UID]
                    FROM AIML.IVCE_XCTN_LLM_TRNL_PRDT_REF d WITH (NOLOCK)
                    JOIN AIML.IVCE_XCTN_LLM_TRNL_MFR_REF i WITH (NOLOCK)
                        ON d.[IVCE_XCTN_LLM_TRNL_MFR_REF_UID] = i.[IVCE_XCTN_LLM_TRNL_MFR_REF_UID]
                    WHERE
                        d.[REC_ACTV_IND] = 'Y' AND
                        d.[TRNG_DAT_VRSN_NUM] = '{training_data_version}'
                    ORDER BY
                        d.[IVCE_XCTN_LLM_TRNL_PRDT_REF_UID] ASC;
                """
        df = await sdp.fetch_data(query)

    except Exception as e:
        logger.error(f"Error in query execution: {str(e)}")
        raise e
    return df


async def bulk_update_training_version(sdp, ref_uuid_list, version_name):
    """
    Updates the product reference table with training version number for the given
    list of id's.

    Parameters:
        training_version_name (str): The training version name to be updated.
        ref_uuid_list (list): the list of values of the REF_UID values for which the
                        training version needs to be updated.

    Returns:
        tuple: A tuple containing the SQL query string and the values to be used in the query.
    """
    try:
        ref_uid_placeholder = ", ".join("?" for _ in ref_uuid_list)
        version_update_query = f"""
                    UPDATE AIML.IVCE_XCTN_LLM_TRNL_PRDT_REF
                    SET
                        TRNG_DAT_VRSN_NUM = ?
                    WHERE IVCE_XCTN_LLM_TRNL_PRDT_REF_UID IN ({ref_uid_placeholder})"""

        values = (version_name, *ref_uuid_list)

        await sdp.update_data(version_update_query, values)
        logger.info("Updated version for current batch successfully.")

    except Exception as e:
        logger.error(f"Error during batch training version update: {str(e)}")
        raise


async def bulk_update_training_version_with_temptable(sdp, ref_uuid_list: list, version_name: str):
    try:
        now = get_current_datetime_cst()
        rows = [(uid, version_name, "AI_USER", now) for uid in ref_uuid_list]

        with sdp.sqlalchemy_engine.begin() as conn:
            # Step 1: Create temp table
            logger.info("Creating temp table")
            conn.exec_driver_sql(
                """
                CREATE TABLE #UpdateRefTable (
                    IVCE_XCTN_LLM_TRNL_PRDT_REF_UID INT NOT NULL,
                    TRNG_DAT_VRSN_NUM VARCHAR(50),
                    [REC_UPDD_BY_ID] [varchar](20) NULL,
                    [REC_UPDD_DTTM] [datetime] NULL
                )
            """
            )

            # Step 2: Insert using raw DBAPI connection (for fast_executemany)
            logger.info("Inserting rows into temp table")
            raw_conn = conn.connection
            raw_cursor = raw_conn.cursor()
            raw_cursor.fast_executemany = True
            raw_cursor.executemany(
                """
                INSERT INTO #UpdateRefTable (
                    IVCE_XCTN_LLM_TRNL_PRDT_REF_UID,
                    TRNG_DAT_VRSN_NUM,
                    REC_UPDD_BY_ID,
                    REC_UPDD_DTTM
                )
                VALUES (?, ?, ?, GETDATE())
            """,
                rows,
            )

            # Step 3: Execute join-based update
            logger.info("Joining rows for update")
            conn.exec_driver_sql(
                """
                UPDATE TGT
                SET
                    TGT.TRNG_DAT_VRSN_NUM = TMP.TRNG_DAT_VRSN_NUM,
                    TGT.REC_UPDD_BY_ID = TMP.REC_UPDD_BY_ID,
                    TGT.REC_UPDD_DTTM = TMP.REC_UPDD_DTTM
                FROM AIML.IVCE_XCTN_LLM_TRNL_PRDT_REF TGT
                JOIN #UpdateRefTable TMP
                    ON TGT.IVCE_XCTN_LLM_TRNL_PRDT_REF_UID = TMP.IVCE_XCTN_LLM_TRNL_PRDT_REF_UID
            """
            )

            # Step 4: Drop temp table
            logger.info("Dropping temp table")
            conn.exec_driver_sql("DROP TABLE #UpdateRefTable")

    except Exception as e:
        logger.error(f"Error during bulk training version update: {str(e)}")
        raise


async def get_finetuned_llm_training_versions(sdp):
    """
    Fetches distinct training versions from table

    Returns:
        df: DataFrame of versions
        col_name: Column name of versions in table
    """
    col_name = "TRNG_DAT_VRSN_NUM"
    query = f"""
        SELECT
            DISTINCT {col_name}
        FROM
            AIML.IVCE_XCTN_LLM_TRNL_PRDT_REF WITH (NOLOCK)
    """

    df = await sdp.fetch_data(query)
    return df, col_name


async def get_non_material_records(sdp, invoice_line_status="RC-AI", number_of_days=1):
    """
    Queries and returns the data of all the rows from the IVCE_DTL
    table which are material records and with the given IVCE_LNE_STAT status.
    Only the records classified as Non-Material will be queried.
    Only the records from the previous day will be queried.

    Parameters:
        sdp (SDP): the sdp data base object to make query.
        invoice_line_status (list): List of unique identifier to indicate the rows to be updated.
        number_of_days (str): Integer indicating the previous number of days records to be fetched.
                              Example: '1' - Previous day records, '2' - Previous 2 days records.
    """
    query = f"""
        SELECT
            IVCE_DTL_UID, IVCE_LNE_STAT, REC_UPDD_BY_ID, CLN_MFR_AI_NM, REC_UPDD_DTTM
        FROM
            RPAO.IVCE_DTL
        WHERE
            REC_UPDD_DTTM >= DATEADD(DAY, -{number_of_days}, CAST(GETDATE() AS DATE))  -- previous day 12:00 AM
            AND REC_UPDD_DTTM < CAST(GETDATE() AS DATE)                                -- current day 12:00 AM (exclusive)
            AND IVCE_LNE_STAT = '{invoice_line_status}'
            AND ( CLN_MFR_AI_NM IN ('TAX', 'FEE', 'LABOR','FREIGHT','BAD', 'UNCLASSIFIED') OR CLN_MFR_AI_NM LIKE 'RENTAL - %')
        ORDER BY
            REC_CRTD_DTTM ASC;
    """
    df = await sdp.fetch_data(query)
    return df


async def get_material_records(sdp, invoice_line_status="RC-AI", number_of_days=1):
    """
    Queries and returns the data of all the rows from the
    IVCE_DTL table which are material records and with the
    given IVCE_LNE_STAT status.
    Only the records classified as Material will be queried.
    Only the records from the previous day will be queried.

    Parameters:
        sdp (SDP): the sdp data base object to make query.
        invoice_line_status (list): List of unique identifier to indicate the rows to be updated.
        number_of_days (str): Integer indicating the previous number of days records to be fetched.
                              Example: '1' - Previous day records, '2' - Previous 2 days records.
    """
    query = f"""
        SELECT
            IVCE_DTL_UID, IVCE_LNE_STAT, REC_UPDD_BY_ID, CLN_MFR_AI_NM, REC_UPDD_DTTM
        FROM
            RPAO.IVCE_DTL
        WHERE
            REC_UPDD_DTTM >= DATEADD(DAY, -{number_of_days}, CAST(GETDATE() AS DATE))  -- previous day 12:00 AM
            AND REC_UPDD_DTTM < CAST(GETDATE() AS DATE)                                -- current day 12:00 AM (exclusive)
            AND IVCE_LNE_STAT = '{invoice_line_status}'
            AND (CLN_MFR_AI_NM NOT IN ('TAX', 'FEE', 'LABOR','FREIGHT','BAD', 'UNCLASSIFIED')
            AND CLN_MFR_AI_NM NOT LIKE 'RENTAL - %')
        ORDER BY
            REC_CRTD_DTTM ASC;
    """
    df = await sdp.fetch_data(query)
    return df


async def bulk_update_rc_status_with_temptable(sdp, ref_uuid_list: list, rc_status: str):
    """
    Updates the RPAO.IVCE_DTL table with the the given status for
    the records of ref_uuid_list list.
    This method uses temp table to update all the records at once.

    Parameters:
        sdp (SDP): the sdp data base object to make query.
        ref_uuid_list (list): List of unique identifier to indicate the rows to be updated.
        rc_status (str): The IVCE_LNE_STAT column value to be updated.
    """
    try:
        rows = [(uid, rc_status) for uid in ref_uuid_list]
        with sdp.sqlalchemy_engine.begin() as conn:
            # Step 1: Create temp table
            conn.exec_driver_sql(
                """
                CREATE TABLE #UpdateRefTable (
                    IVCE_DTL_UID BIGINT,
                    IVCE_LNE_STAT VARCHAR(10)
                )
            """
            )

            # Step 2: Insert using raw DBAPI connection (for fast_executemany)
            raw_conn = conn.connection
            raw_cursor = raw_conn.cursor()
            raw_cursor.fast_executemany = True
            raw_cursor.executemany(
                """
                INSERT INTO #UpdateRefTable (IVCE_DTL_UID, IVCE_LNE_STAT)
                VALUES (?, ?)
            """,
                rows,
            )

            # Step 3: Execute join-based update
            conn.exec_driver_sql(
                """
                UPDATE TGT
                SET TGT.IVCE_LNE_STAT = TMP.IVCE_LNE_STAT
                FROM RPAO.IVCE_DTL TGT
                JOIN #UpdateRefTable TMP
                    ON TGT.IVCE_DTL_UID = TMP.IVCE_DTL_UID
            """
            )

            # Step 4: Drop temp table
            conn.exec_driver_sql("DROP TABLE #UpdateRefTable")

    except Exception as e:
        logger.error(f"Error during bulk training version update: {str(e)}")
        raise
