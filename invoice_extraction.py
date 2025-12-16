# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
""" "
## Overview
This module provides functionality for processing invoice details using AI-based tools and Cosmos DB.
It includes classes and methods for handling invoice extraction, job creation, and detailed invoice processing.
The module is designed to work asynchronously and supports parallel processing with configurable worker limits.
"""

from typing import Any

from api_schemas import Requests, Responses
from cdb import CDB
from cdb_utils import CDB_Utils
from config import Config
from constants import Constants, Logs
from exceptions import InvoiceProcessingError
from logger import logger
from pipelines import Pipelines
from sdp import SDP
from sql_utils import get_invoice_data
from utils import get_current_datetime_cst, is_not_empty, is_not_null


class InvoiceExtractor:
    """
    Handles the extraction and processing of invoices using AI-based tools.

    Args:
        config (Config): Configuration object containing settings for the system, such as
            maximum workers, database connection details, and AI engine parameters.

    Attributes:
        config (Config): The configuration object passed during initialization.
        cdb (CDB): Instance of the Cosmos DB handler initialized with the provided config.
        sdp (SDP): Instance of the SDP handler initialized with the provided config.
    """

    def __init__(self, config: Config):
        self.config = config
        self.cdb = CDB(config=config)
        self.sdp = SDP(config=config)
        self.pipelines = Pipelines(config=config, sdp=self.sdp)
        self.cdb_utils = CDB_Utils(cdb=self.cdb, config=self.config)

    def refresh_connections(self):
        """
        Refresh objects as required
        """
        self.sdp = SDP(config=self.config)

    async def process_all_invoices_request(self, request):
        # logger.debug(f"User query: {user_query}")
        result = {}
        return result

    async def process_invoice_request(self, request_body: Requests.ProcessInvoiceRequest):
        # load invc from SDP
        # df = await get_invoice_data(sdp=self.sdp, invoice_header_id=request.invoice_id)
        _ = await get_invoice_data(sdp=self.sdp, invoice_header_id=request_body.invoice_id)
        # iterate through each detail
        #   # run ai engine
        #   # update details
        return {}

    async def process_invoice_details_request(self, request_body: Requests.ProcessInvoiceDetailRequest):
        """
        Processes an invoice details request by creating jobs in Cosmos DB and returning
        a response with the status and message.

        Args:
            request_body (ProcessInvoiceDetailRequest): The request object containing metadata
                and a list of invoice detail IDs to be processed.

        Returns:
            ProcessInvoiceDetailResponse: A response object containing the status, reply ID,
                and a message summarizing the outcome of the job creation process.

        Workflow:
            1. Create jobs in Cosmos DB using the provided request and invoice detail IDs.
            2. Generate a response object with the status set to "success" and the message
            returned from the job creation process.
            3. Return the response object.

        Notes:
            - The `create_jobs` method handles the job creation process and returns a message
            summarizing the outcome.
            - The response object includes the `replyToId` to correlate the response with the
            original request.

        """
        try:
            # Step 1: Fire the async background task to create jobs in Cosmos DB
            message = await self.cdb_utils.create_jobs_in_cosmos_queue(
                request_body=request_body, detail_ids=request_body.invoice_detail_ids
            )

            # Step 2: Update the message in the API request in container
            self.cdb.patch_document(
                container=self.cdb.ai_api_requests_container,
                doc_id=request_body.api_request_uuid,
                operations=[{"op": "add", "path": f"/{Constants.MESSAGE}", "value": message}],
                # partition_key="/id",  # Looks like a bug
                partition_key=request_body.api_request_uuid,  # This should fix the error
            )

            # Step 3: Continue immediately generating a response object
            response = Responses.ProcessInvoiceDetailResponse(replyToId=request_body.id, status="success", message=message)

        except Exception as e:
            raise InvoiceProcessingError(message=f"Error occurred in processing request. {str(e)}", original_exception=e)
        # Step 5: Return the response
        return response

    async def process_single_invoice_detail(self, request_details, ivce_dtl):
        """
        Processes a single invoice detail asynchronously, executes AI-based processing,
        and logs the results to Cosmos DB.

        Args:
            request_details (dict): Metadata and details related to the request.
            ivce_dtl (InvoiceDetail): The invoice detail object to be processed.

        Workflow:
            1. Log the start of the invoice detail processing.
            2. Initialize the AI engine and process the invoice detail using `process_description`.
            3. Generate a response object based on the processing results.
            4. Prepare and upload the process logs and results to Cosmos DB.
            5. Handle any exceptions during processing or logging, and update the response accordingly.

        Returns:
            None

        Exception Handling:
            - Logs errors that occur during processing or Cosmos DB operations.
            - Raises exceptions if logging to Cosmos DB fails.

        Notes:
            - The AI engine processes the invoice detail and returns stage results, including
            intermediate logs and final results.
            - Process logs and results are stored in the Cosmos DB `ai_logs_container`.

        """

        def validate_and_update_description(ivce_dtl: InvoiceDetail) -> InvoiceDetail:
            try:
                ivce_dtl.ITM_RPA_LDSC = ivce_dtl.ITM_LDSC
                part_num = ivce_dtl.MFR_PRT_NUM
                mfr_name = ivce_dtl.MFR_NM
                description = str(ivce_dtl.ITM_LDSC)

                if part_num is not None and is_not_null(part_num) and is_not_empty(part_num) and part_num not in description:
                    description = " ".join([part_num, description])

                if mfr_name is not None and is_not_null(mfr_name) and is_not_empty(mfr_name) and mfr_name not in description:
                    description = " ".join([mfr_name, description])

                ivce_dtl.ITM_LDSC = description

            except Exception as e:
                raise InvoiceProcessingError(
                    message=f"Error occurred in validating and updating description. {str(e)}", original_exception=e
                )

            return ivce_dtl

        try:
            # Step 1: Log the start of processing
            logger.info("-" * 80)
            logger.info(f"Processing invoice detail with ID {ivce_dtl.IVCE_DTL_UID}")
            request_details[Logs.START_TIME] = get_current_datetime_cst()

            # Step 2: Initialize AI engine and process the invoice detail
            # ai_engine = AIEngine(config=self.config, sdp=self.sdp)
            ivce_dtl = validate_and_update_description(ivce_dtl)

            # stage_results = await ai_engine.process_description(request_details, ivce_dtl)
            pipeline_results = await self.pipelines.process(request_details, ivce_dtl)
            stage_results = pipeline_results.stage_results
            logger.info(f"Invoice processing is completed for {ivce_dtl.IVCE_DTL_UID}")

            # # Step 3: Generate a response object based on processing results
            if stage_results.status == Constants.SUCCESS_lower:
                message = "Processed invoice detail."
            else:
                message = stage_results.message
            response = Responses.ProcessInvoiceDetailResponse(
                replyToId=request_details["id"], status=stage_results.status, message=message
            )

        except Exception as e:
            # Handle errors during processing
            logger.error(f"Error in process_single_invoice_detail(): {str(e)}", exc_info=True)
            response = Responses.ProcessInvoiceDetailResponse(
                replyToId=request_details["id"],
                status=Constants.ERROR_lower,
                message=f"Error in process_single_invoice_detail(): {str(e)}",
            )

        try:
            process_logs = getattr(stage_results, "results", {})
            process_output = getattr(stage_results, "final_results", {})

            # Step 4: Prepare and upload process logs to Cosmos DB
            document = await self.cdb_utils.prepare_document_for_ai_logs(
                request_details=request_details,
                response=response,
                process_logs=process_logs,
                process_output=process_output,
                post_process_details=pipeline_results.post_process_details,
                invoice_details=ivce_dtl,
                pre_process_details=pipeline_results.pre_process_details,
            )
            # self.cdb.add_item(self.cdb.ai_logs_container, document)
            self.cdb.update_document(container=self.cdb.ai_logs_container, document=document)
            logger.info(f"Invoice process logs are uploaded to Cosmos DB container {ivce_dtl.IVCE_DTL_UID}")

        except Exception as e:
            # Handle errors during logging to Cosmos DB
            logger.error(f"Error in process_single_invoice_detail(): {str(e)}", exc_info=True)
            raise e


class InvoiceDetail:
    """
    Represents an invoice detail object, initializing its attributes based on the
    provided dictionary data.

    Args:
        data (dict): A dictionary containing key-value pairs to initialize the attributes
            of the `InvoiceDetail` instance.

    Raises:
        TypeError: If the provided `data` is not a dictionary.

    Attributes:
        Dynamically created attributes based on the keys and values in the `data` dictionary.

    Example:
        data = {
            "IVCE_DTL_UID": "12345",
            "amount": 100.50,
            "status": "pending"
        }
        invoice_detail = InvoiceDetail(data)
        print(invoice_detail.IVCE_DTL_UID)  # Output: "12345"
        print(invoice_detail.amount)       # Output: 100.50
        print(invoice_detail.status)       # Output: "pending"
    """

    def __init__(self, data):
        if isinstance(data, dict):  # Ensure data is a single dictionary
            for key, value in data.items():
                setattr(self, key, value)
        else:
            raise TypeError("Expected a single dictionary, but got a list.")

    def __getattr__(self, name: str) -> Any:
        # This tells mypy: any attribute is allowed
        return None
