# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
## Overview
This file serves as the entry point for the invoice processing API application.
It uses FastAPI to expose endpoints for processing invoices, managing worker threads,
and handling authentication and logging middleware.
"""

import asyncio
import tracemalloc
from functools import wraps

import uvicorn

tracemalloc.start()

from contextlib import asynccontextmanager

from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from fastapi import FastAPI, HTTPException, Request
from opentelemetry import trace
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from starlette.middleware.cors import CORSMiddleware

from api_schemas import Requests, Responses
from config import Config
from constants import APIPaths, Constants, Environments
from exceptions import InvoiceProcessingError
from invoice_extraction import InvoiceExtractor
from logger import logger
from middlewares import CosmosLoggingMiddleware, JWTAuthMiddleware, LoggingMiddleware
from sql_writer import SqlWriterService
from utils import update_app_last_restart_time
from worker import Worker

# Load Configuration
CONFIG = Config()  # Load application configuration
# logger = get_default_logger(level=CONFIG.log_level, azure_conn_str=CONFIG.APP_INSIGHTS_CONN_STRING)  # Initialize logger

# overwrite APP ENV
if CONFIG.environment != Environments.LOCAL:
    update_app_last_restart_time(config=CONFIG)

# Create the Invoice Extractor instance and pass it Worker class
ie = InvoiceExtractor(config=CONFIG)

# Get the shared SDP and CDB instances from the InvoiceExtractor
sdp_instance = ie.sdp
cdb_instance = ie.cdb
sql_writer = SqlWriterService(config=CONFIG, sdp=sdp_instance, cdb=cdb_instance)

worker = Worker(invoice_extractor=ie, sql_writer=sql_writer)
worker.refresh_config()


def inject_api_request_uuid(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        body: Requests.BaseRequest = kwargs.get("request_body")
        if body:
            body.api_request_uuid = kwargs.get("request", {}).get("state", {}).get("api_request_uuid")
        return await func(*args, **kwargs)

    return wrapper


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifespan of the FastAPI application, including starting and stopping worker threads.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Allows the application to run while managing resources.
    """
    # global worker

    # # --- Initialize Agents Singleton Early ---
    # try:
    #     # This will trigger Agents.__init__ if it hasn't run yet, or return the existing instance.
    #     Agents(config=CONFIG)
    #     logger.info("Agents singleton initialization completed.")
    # except Exception as e:
    #     logger.critical(f"CRITICAL: Failed to initialize Agents singleton during app startup: {e}", exc_info=True)

    # Start worker thread
    # launches your supervisor + worker threads

    worker.start_worker_thread()

    try:
        yield
    finally:
        # Stop the SQL Writer Service gracefully
        try:
            await asyncio.to_thread(sql_writer.stop)
        except Exception as e:
            logger.error(f"Error stopping SQL Writer Service: {e}", exc_info=True)

        logger.info("Supervisor Worker shut down.")  # Log shutdown of worker threads
        # # --- Shutdown Agents Coordinator ---
        # agents_singleton = Agents._instance
        # if agents_singleton is not None and getattr(agents_singleton, '_initialized', False):
        #     logger.info("Agents instance found and was initialized. Proceeding with shutdown.")
        #     try:
        #         agents_singleton.shutdown()
        #     except Exception as e:
        #         logger.error(f"Error during Agents coordinator shutdown: {e}", exc_info=True)
        # else:
        #     logger.info("Agents instance was not created or not fully initialized during app runtime. "
        #                 "Skipping Agents shutdown.")
        # logger.info("Application shutdown sequence finished.")


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


if CONFIG.APP_INSIGHTS_CONN_STRING:
    # Setup OTLP TracerProvider + Azure exporter
    trace.set_tracer_provider(TracerProvider())
    exporter = AzureMonitorTraceExporter.from_connection_string(CONFIG.APP_INSIGHTS_CONN_STRING)
    span_processor = BatchSpanProcessor(exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

    # Adding ASGI middleware
    app.add_middleware(OpenTelemetryMiddleware)

# Adding middlewares
app.add_middleware(LoggingMiddleware)  # Middleware for logging requests and responses

if CONFIG.environment != Environments.LOCAL:
    # CORS Middleware (before JWT)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all domains (configure as needed)
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # JWT Middleware for authentication
    app.add_middleware(JWTAuthMiddleware, config=CONFIG)

app.add_middleware(CosmosLoggingMiddleware, cdb_utils=ie.cdb_utils)


@app.get(
    APIPaths.GET_WORKER_STATUS,
    response_model=Responses.GetWorkerStatusResponse,
    responses={401: {"model": Responses.ErrorResponse}, 403: {"model": Responses.ErrorResponse}},
)
def worker_alive(request_body: Requests.GetWorkerStatusRequest, request: Request):
    """
    Endpoint to check the status of the supervisor and worker threads.

    Returns:
        dict: A dictionary indicating whether the supervisor and worker threads are alive.
    """
    return Responses.GetWorkerStatusResponse(
        supervisor=getattr(worker, "supervisor_thread", None) and worker.supervisor_thread.is_alive(),
        worker=getattr(worker, "worker_thread", None) and worker.worker_thread.is_alive(),
        status="success",
        message=Constants.EMPTY_STRING,
    )


@app.post(
    APIPaths.PROCESS_ALL_INVOICES,
    response_model=Responses.ProcessAllInvoicesResponse,
    responses={401: {"model": Responses.ErrorResponse}, 403: {"model": Responses.ErrorResponse}},
)
@inject_api_request_uuid
async def process_all_invoices(request_body: Requests.ProcessAllInvoicesRequest, request: Request):
    """
    Endpoint to process all invoices in the system.

    Args:
        request (ProcessAllInvoicesRequest): The request object containing invoice processing details.

    Returns:
        ProcessAllInvoicesResponse: A response object indicating the status of the operation.

    Raises:
        HTTPException: If an internal server error occurs.
    """
    try:
        # Process invoices
        await ie.process_all_invoices_request(request_body)
        worker.queue_request_id(request_body.id)

        return Responses.ProcessAllInvoicesResponse(status="success", message="Processed all invoices.")

    except Exception as e:
        logger.error(f"Error in API method execution: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post(
    APIPaths.PROCESS_INVOICE,
    response_model=Responses.ProcessInvoiceResponse,
    responses={401: {"model": Responses.ErrorResponse}, 403: {"model": Responses.ErrorResponse}},
)
@inject_api_request_uuid
async def process_invoice(request_body: Requests.ProcessInvoiceRequest, request: Request):
    """
    Endpoint to process a single invoice.

    Args:
        request (ProcessInvoiceRequest): The request object containing invoice processing details.

    Returns:
        ProcessInvoiceResponse: A response object indicating the status of the operation.

    Raises:
        HTTPException: If an internal server error occurs.
    """
    try:
        # Process invoices
        await ie.process_invoice_request(request_body)
        worker.queue_request_id(request_body.id)

        return Responses.ProcessInvoiceResponse(status="success", message="Processed invoice.")

    except Exception as e:
        logger.error(f"Error in API method execution: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post(
    APIPaths.PROCESS_INVOICE_DETAILS,
    response_model=Responses.ProcessInvoiceDetailResponse,
    responses={401: {"model": Responses.ErrorResponse}, 403: {"model": Responses.ErrorResponse}},
)
@inject_api_request_uuid
async def process_invoice_details(request_body: Requests.ProcessInvoiceDetailRequest, request: Request):
    """
    Endpoint to process multiple details in invoice.

    Args:
        request (ProcessInvoiceDetailRequest): The request object containing invoice detail processing details.

    Returns:
        ProcessInvoiceDetailResponse: A response object indicating the status of the operation.

    Raises:
        ErrorResponse: If an invoice processing error occurs.
        HTTPException: If an internal server error occurs.
    """
    try:
        # Process invoices
        response = await ie.process_invoice_details_request(request_body=request_body)
        worker.queue_request_id(request_body.id)
        return response

    except InvoiceProcessingError as e:
        logger.error(f"Invoice Processing Error: {e.message}")
        return Responses.ErrorResponse(status="failed", message=e.message)

    except Exception as e:
        logger.error(f"Error in API method execution: {str(e)}", exc_info=True)
        # raise HTTPException(status_code=500, detail="Internal Server Error")
        return Responses.ErrorResponse(status="failed", message="Internal Server Error", error_detail=str(e))


@app.post(
    APIPaths.RELOAD_APP_CONFIG,
    response_model=Responses.ReloadAppConfigResponse,
    responses={401: {"model": Responses.ErrorResponse}, 403: {"model": Responses.ErrorResponse}},
)
@inject_api_request_uuid
async def reload_app_config(request_body: Requests.ReloadAppConfigRequest, request: Request):
    """
    Endpoint to process reload the app config values.

    Args:
        request (ReloadAppConfigRequest): The request object containing request id.

    Returns:
        ReloadAppConfigResponse: A response object indicating the status of the operation.

    Raises:
        ErrorResponse: If an invoice processing error occurs.
        HTTPException: If an internal server error occurs.
    """
    try:
        logger.info("Refreshing the Azure APP configuration")
        CONFIG.refresh_from_azure_app_config()
        worker.refresh_config()
        # ie.config = CONFIG
        worker.config = CONFIG
        worker._has_app_refreshed = True

        return Responses.ReloadAppConfigResponse(status="success", message="App config reload.")

    except Exception as e:
        logger.error(f"Error in API method execution: {str(e)}", exc_info=True)
        # raise HTTPException(status_code=500, detail="Internal Server Error")
        return Responses.ErrorResponse(status="failed", message="Internal Server Error", error_detail=str(e))


@app.post(
    APIPaths.GET_CONFIG_SNAPSHOT,
    response_model=Responses.GetConfigSnapshotResponse,
    responses={401: {"model": Responses.ErrorResponse}, 403: {"model": Responses.ErrorResponse}},
)
@inject_api_request_uuid
async def get_config_snapshot(request_body: Requests.GetConfigSnapshotRequest, request: Request):
    """
    Endpoint to fetch CONFIG object that app is currently using

    Returns:
        GetConfigSnapshotResponse: A response object indicating dictionary of config object.

    Raises:
        ErrorResponse: If an invoice processing error occurs.
        HTTPException: If an internal server error occurs.
    """
    try:
        logger.info("Sending CONFIG snapshot...")
        return Responses.GetConfigSnapshotResponse(config=CONFIG.__dict__, status="success", message="App config reload.")

    except Exception as e:
        logger.error(f"Error in API method execution: {str(e)}", exc_info=True)
        # raise HTTPException(status_code=500, detail="Internal Server Error")
        return Responses.ErrorResponse(status="failed", message="Internal Server Error", error_detail=str(e))


@app.post(
    APIPaths.GET_COSMOS_QUEUE_STATUS,
    response_model=Responses.ProcessInvoiceDetailResponse,
    responses={401: {"model": Responses.ErrorResponse}, 403: {"model": Responses.ErrorResponse}},
)
@inject_api_request_uuid
async def get_cosmos_queue_status(body: Requests.ProcessInvoiceDetailRequest, request: Request):

    try:
        # container = await get_container()

        # count_pending = container.query_items(
        #     "SELECT VALUE COUNT(1) FROM c WHERE c.status = 'pending'",
        #     enable_cross_partition_query=True
        # )
        # count_done = container.query_items(
        #     "SELECT VALUE COUNT(1) FROM c WHERE c.status = 'done'",
        #     enable_cross_partition_query=True
        # )

        # pending = [x async for x in count_pending][0]
        # done = [x async for x in count_done][0]

        # return {"pending": pending, "done": done}
        pass

    except InvoiceProcessingError as e:
        logger.error(f"Invoice Processing Error: {e.message}")
        return Responses.ErrorResponse(status="failed", message=e.message)

    except Exception as e:
        logger.error(f"Error in API method execution: {str(e)}", exc_info=True)
        # raise HTTPException(status_code=500, detail="Internal Server Error")
        return Responses.ErrorResponse(status="failed", message="Internal Server Error", error_detail=str(e))


if __name__ == "__main__":
    """
    Entry point for the application. Starts the FastAPI server using Uvicorn.
    """
    # uvicorn app:app --reload --host 0.0.0.0 --port 8000
    logger.debug(f"\nENVIRONMENT: {CONFIG.environment}, LOG_LEVEL: {CONFIG.log_level}\n")
    uvicorn.run(app, log_level=CONFIG.log_level.lower())
