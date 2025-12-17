import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from queue import Queue
from typing import Iterator, Optional

import pandas as pd

from cdb_utils import CDB_Utils
from config import Config
from constants import Constants, CosmosLogStatus, Environments, Logs
from exceptions import InvoiceProcessingError
from invoice_extraction import InvoiceDetail, InvoiceExtractor
from logger import logger
from sql_utils import get_invoice_detail_data
from sql_writer import SqlWriterService
from utils import get_current_datetime_cst


class PriorityRequestStatus(StrEnum):
    NOT_QUEUED = "NOT_IN_QUEUE"  # Not yet seen request submitted
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"  # Started processing part of the request
    COMPLETED = "COMPLETED"  # Started processing part of the request and found no more


status_mapper = {
    CosmosLogStatus.DONE_lower: PriorityRequestStatus.COMPLETED,
    CosmosLogStatus.PROCESSING_lower: PriorityRequestStatus.IN_PROGRESS,
    CosmosLogStatus.QUEUED_lower: PriorityRequestStatus.QUEUED,
    CosmosLogStatus.PENDING_lower: PriorityRequestStatus.QUEUED,
    CosmosLogStatus.ERROR_lower: PriorityRequestStatus.QUEUED,
}


class Worker:
    """
    Handles the queuing and processing of invoices using Cosmos DB.

    Args:
        invoice_extractor (Invoice_Extractor): Object for processing invoices

    Attributes:
        config (Config): The configuration object passed during initialization.
        cdb (CDB): Instance of the Cosmos DB handler initialized with the provided config.
        sdp (SDP): Instance of the SDP handler initialized with the provided config.
        BUFFER_SIZE (int): Maximum size of the local buffer for pending jobs.
        REFILL_THRESH (int): Threshold for refilling the buffer when jobs are processed.
        POLL_INTERVAL (int): Time interval (in seconds) between polling for new jobs.
        MAX_WORKERS (int): Maximum number of workers for parallel processing, adjusted based
            on the buffer size and configuration settings.
        pending_queue (Queue): Local buffer for storing pending jobs, with a maximum size
            defined by `BUFFER_SIZE`.

    Notes:
        - The `MAX_WORKERS` attribute is adjusted to ensure it does not exceed the buffer size
          or fall below 1. Warnings are logged if adjustments are made.
        - The `pending_queue` is used to manage jobs locally before processing.

    Example:
        workers = Workers(config=config)
        print(workers.MAX_WORKERS)  # Output depends on the buffer size and config settings.
    """

    def __init__(self, invoice_extractor: InvoiceExtractor, sql_writer: SqlWriterService):
        self.ie = invoice_extractor
        self.config = self.ie.config
        self.cdb_utils = invoice_extractor.cdb_utils
        self.cdb = self.ie.cdb
        self.sdp = self.ie.sdp
        self.sql_writer = sql_writer
        self.req_tracker = RequestsTracker(cdb_utils=self.cdb_utils, config=self.config)

        self.update_settings_from_config()
        # Overriding this flag because it is the start of the webapp
        self._has_app_refreshed = False

    def update_settings_from_config(self):
        self.configure_local_queue()
        self.configure_max_workers()
        self.recreate_queue_and_tpe()
        self.should_pause_processing = self.config.worker_settings.get(Constants.SHOULD_PAUSE_PROCESSING, False)

    def configure_local_queue(self):
        """
        Configuration for job processing
        """
        worker_settings = self.config.worker_settings
        self.BUFFER_SIZE = worker_settings.get(Constants.BUFFER_SIZE, 40)  # 40 # Buffer size of pending jobs in queue

        self.REFILL_THRESH = worker_settings.get(Constants.REFILL_THRESH, 50)  # 20 # when 50 have been processed
        if self.REFILL_THRESH >= self.BUFFER_SIZE:
            self.REFILL_THRESH = int(self.BUFFER_SIZE * 0.8)
            logger.warning(
                f"Refill Threshold is force set to '{self.REFILL_THRESH}' to be lessthan the buffer size {self.BUFFER_SIZE}."
            )

        self.POLL_INTERVAL = worker_settings.get(Constants.POLL_INTERVAL, 5)  # seconds between polls
        if self.POLL_INTERVAL <= 1:
            self.POLL_INTERVAL = 2
            logger.warning(f"poll interval is force set to '{self.POLL_INTERVAL}'.")

    def configure_max_workers(self):
        """
        Adjust MAX_WORKERS based on buffer size and configuration
        """
        config_workers = self.config.worker_settings.get(Constants.MAX_WORKERS, 1)
        if config_workers > self.BUFFER_SIZE:
            self.MAX_WORKERS = self.BUFFER_SIZE - 1
            logger.warning(f"Max workers are force set to '{(self.BUFFER_SIZE - 1)}' to be lessthan the buffer size.")

        elif config_workers <= 0:
            self.MAX_WORKERS = 1
            logger.warning("Max workers are force set to '1' which makes sequential processing.")

        else:
            self.MAX_WORKERS = config_workers

    def recreate_queue_and_tpe(self):
        """
        Recreate queue with the new BUFFER_SIZE and recreate new ThreadPoolExecutor with new MAX_WORKERS
        """
        # Local buffer for pending jobs
        self.pending_queue: Queue = Queue(maxsize=self.BUFFER_SIZE)
        self.executor = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)
        self._has_app_refreshed = True

    def refresh_config(self):
        self.req_tracker.refresh_config()

    def queue_request_id(self, request_id: str):
        """Request id is ready to be processed, it has been added to the jobs in cosmos"""
        self._set_request_id_status(request_id, PriorityRequestStatus.QUEUED)

    def _set_request_id_status(self, request_id: str, status: PriorityRequestStatus):
        self.req_tracker.set_request_id_status(request_id, status)

    def start_worker_thread(self):
        """
        Spawn the daemon thread that runs `self._supervise_worker`.
        """
        sup = threading.Thread(target=self._supervise_worker, daemon=True)
        sup.start()
        self.supervisor_thread = sup

    # def _supervise_worker(self):
    #     """
    #     Loop forever: spawn the real worker, wait for it to exit, then restart.
    #     """
    #     while True:
    #         worker = threading.Thread(target=self._worker_loop, daemon=True)
    #         worker.start()
    #         self.worker_thread = worker
    #         worker.join()  # block here until _worker_loop returns/crashes
    #         logger.warning("Worker thread stopped; restarting in 5s…")
    #         time.sleep(5)

    def _supervise_worker(self):
        """
        Supervisor loop that monitors and restarts both the main worker
        and the SQL writer threads if they crash.
        """
        logger.info("Main supervisor thread started. Monitoring both Worker and SQL Writer.")

        # Start the SQL Writer service for the first time
        self.sql_writer.start()

        # Start the main Worker thread for the first time
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        # Loop forever to monitor the health of both threads
        while True:
            try:
                # Use join with timeout to efficiently wait for worker death
                # This blocks until worker dies OR 30 seconds pass (whichever comes first)
                self.worker_thread.join(timeout=30)

                # If we get here, either:
                # 1. Worker died (is_alive() = False), or
                # 2. Timeout elapsed (is_alive() = True)

                if not self.worker_thread.is_alive():
                    logger.warning("Worker thread has stopped unexpectedly. Restarting in 5s...")
                    time.sleep(5)  # Backoff before restart
                    self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
                    self.worker_thread.start()

                # Check SQL Writer health (can't use join since we don't control its thread directly)
                if not self.sql_writer.is_running():
                    logger.warning("SQL Writer thread has stopped unexpectedly. Restarting in 5s...")
                    time.sleep(5)  # Backoff before restart
                    try:
                        self.sql_writer.stop()
                    except Exception as stop_err:
                        logger.error(f"Error stopping SQL Writer during restart: {stop_err}", exc_info=True)
                    finally:
                        try:
                            self.sql_writer.start()
                        except Exception as start_err:
                            logger.critical(f"Failed to restart SQL Writer: {start_err}", exc_info=True)

            except Exception as e:
                logger.critical(f"Supervisor loop encountered an error: {e}", exc_info=True)
                time.sleep(5)  # Backoff on error

    def _fetch_jobs(self, n):
        """Fetch jobs to process in priority order

        Args:
            n (int): Number of jobs to fetch

        Returns:
            list: list of jobs returned
        """
        # Where condition
        base_condition = f" WHERE c.status = '{CosmosLogStatus.PENDING_lower}' and c.request_details.sys_name = "
        base_condition = (
            f" WHERE c.status = '{CosmosLogStatus.PENDING_lower}' AND IS_DEFINED(c.api_request_uuid) = true and"
            " c.request_details.sys_name = "
        )
        if self.config.environment == Environments.LOCAL:
            base_condition += f"'{self.config.local_sys_name}'"
        else:
            base_condition += "'web'"

        # Order condition
        order = " ORDER BY c.created_at ASC"
        all_jobs = []
        # Request id's that we've processed, that way we don't try to refind them without the request_id filter
        request_ids_processed = set()
        request_ids = self.req_tracker.get_request_ids()
        while len(all_jobs) < n:
            condition = base_condition
            try:
                request_id = next(request_ids)
            except StopIteration:
                request_id = None

            if request_id:
                # add to where condition
                condition += f" and c.request_details.request_id = '{request_id}'"
                request_ids_processed.add(request_id)
            elif request_ids_processed:
                # Don't get the jobs that we already retrieved
                request_ids_processed_str = [f"'{request_id}'" for request_id in request_ids_processed]
                # add to where condition
                condition += f" and c.request_details.request_id not in ({', '.join(request_ids_processed_str)})"

            # Order condition
            order = " ORDER BY c.created_at ASC"

            # Read jobs from cosmos db queue
            n -= len(all_jobs)
            jobs = self.cdb.get_documents(
                container=self.cdb.ai_jobs_container, top=n, where_condition=condition, order_condition=order
            )
            if not jobs:
                if request_id:
                    # Mark request id as done
                    self.req_tracker.set_request_id_status(request_id, PriorityRequestStatus.IN_PROGRESS)
                    continue
                logger.debug("There are NO pending jobs.", extra={"dd_excluded": True})
                break
            else:
                # if not request_id:
                #     # Check if we are processing any that somehow didn't get marked as queued
                #     request_ids = {job.get("request_details", {}).get("request_id", None) for job in jobs}
                #     for request_id in request_ids:
                #         if request_id:
                #             self._set_request_id_status(request_id, PriorityRequestStatus.IN_PROGRESS)
                all_jobs.extend(jobs)
                if not request_id:
                    break

        if not all_jobs:
            # Mark request id as completed if filtered by request_id
            # Call again
            # If this log message modified, need to update skip_exact_logs list in logger.py
            logger.debug("There are NO pending jobs.", extra={"dd_excluded": True})
            return []
        else:
            logger.debug(f"Currently, found at least {len(all_jobs)} pending jobs/ids", extra={"dd_excluded": True})
        return all_jobs

    def _fetch_and_prepare_jobs(self, n):
        """
        Fetches up to `n` pending documents from Cosmos DB, marks them as 'queued',
        and enqueues them locally for further processing. Documents already marked as
        'queued' or 'processing' or 'error' are skipped.

        Args:
            n (int): The maximum number of pending documents to fetch.

        Raises:
            InvoiceProcessingError: Raised if invoice detail data cannot be found for the fetched jobs.

        Process:
            1. Fetch up to `n` pending documents from Cosmos DB based on the environment configuration.
            2. Extract `invoice_detail_ids` from the fetched documents.
            3. Retrieve detailed invoice data using the extracted IDs.
            4. Build a mapping of invoice details for further processing.
            5. Update the status of each job in Cosmos DB and enqueue the job locally.

        Notes:
            - The function uses Cosmos DB for fetching and updating job statuses.
            - Local environment jobs are filtered differently from non-local environments.
            - Missing invoice details are logged as errors and marked accordingly in Cosmos DB.
            - Jobs are enqueued locally for further processing, ensuring the local queue does not overflow.

        """
        # If this log message modified, need to update skip_exact_logs list in logger.py
        logger.debug("Polling Cosmos DB...", extra={"dd_excluded": True})

        # Step 1: Fetch up to `n` pending documents based on environment configuration
        jobs = self._fetch_jobs(n)

        if not jobs:
            # If this log message modified, need to update skip_exact_logs list in logger.py
            logger.debug("There are NO pending jobs.", extra={"dd_excluded": True})
            return
        else:
            logger.info(f"Currently, there are {len(jobs)} pending jobs/ids")

        # Step 2: Extract `invoice_detail_ids` from the fetched documents
        invoice_ids = [doc["id"].split("~")[-1] for doc in jobs]

        # Step 3: Fetch detailed invoice data using the extracted IDs
        df, missing = asyncio.run(get_invoice_detail_data(sdp=self.sdp, invoice_detail_ids=invoice_ids))
        if df is None:
            raise InvoiceProcessingError("Invoice detail ID not found.")

        # Step 4: Build a mapping of invoice details for further processing
        invoice_detail_map = {str(rec["IVCE_DTL_UID"]): InvoiceDetail(rec) for rec in df.to_dict(orient="records")}

        for job in jobs:
            ivce_id = job["id"].split("~")[-1]
            ivce_dtl = invoice_detail_map.get(ivce_id)

            if ivce_dtl is None:
                # Should not happen, but guard
                logger.error(f"No detail for {ivce_id}; marking error.")
                job.update({"status": "error", "message": "Detail not found", "updated_at": get_current_datetime_cst()})
                self.cdb.update_document(document=job, container=self.cdb.ai_jobs_container)
                continue

            # Mark the job in Cosmos DB to prevent re-fetching
            job.update({"status": CosmosLogStatus.QUEUED_lower, "updated_at": get_current_datetime_cst()})
            self.cdb.update_document(document=job, container=self.cdb.ai_jobs_container)

            orig = job["request_details"]
            request_details = {
                Logs.ID: str(orig["request_id"]),
                "total_invoice_detail_ids": orig[Logs.TOTAL_IDS_IN_REQUEST],
                # "invoice_details_ids_missing_in_db": len(missing),
                Logs.API_REQUEST_UUID: job[Logs.API_REQUEST_UUID],
                Logs.CLASSIFY: orig["classify"],
                Logs.SYS_NAME: orig["sys_name"],
            }

            # Add to Req Tracker
            self.req_tracker.add_request(
                api_req_uuid=request_details[Logs.API_REQUEST_UUID],
                req_id=request_details[Logs.ID],
                count=orig["total_ids_in_req"],
            )

            # Enqueue the job locally (blocks only if the queue is full)
            self.pending_queue.put((job, request_details, ivce_dtl))

        logger.info(f"Fetched {len(jobs)} jobs into local queue (size now {self.pending_queue.qsize()})")

    def _worker_loop(self):
        """
        Rolling thread-pool worker loop that manages job processing with dynamic scheduling.
        The loop ensures that up to `MAX_WORKERS` jobs are always in flight, and dynamically
        refills the job queue based on thresholds and buffer size.

        Workflow:
            1. Fetch an initial batch of jobs to fill the local queue up to `BUFFER_SIZE`.
            2. Continuously schedule jobs for processing using a thread pool executor.
            3. Reap completed futures and handle any exceptions raised during job execution.
            4. Dynamically refill the job queue based on the `REFILL_THRESH` or when both
            the in-flight jobs and the local queue are empty.
            5. Back off periodically to avoid excessive polling.

        Key Parameters:
            - `MAX_WORKERS`: Maximum number of jobs that can be processed concurrently.
            - `REFILL_THRESH`: Threshold for the number of jobs processed since the last fetch
            that triggers a refill.
            - `BUFFER_SIZE`: Maximum size of the local job queue.
            - `POLL_INTERVAL`: Time interval (in seconds) to wait before the next iteration.

        Behavior:
            - If `N ≤ MAX_WORKERS`: All jobs run in one batch, followed by a refill when both
            the in-flight jobs and the queue are empty.
            - If `MAX_WORKERS < N ≤ REFILL_THRESH`: Dynamic scheduling drains the queue without
            hitting the threshold refill, followed by a final refill when both are empty.
            - If `REFILL_THRESH < N ≤ BUFFER_SIZE`: One threshold-based refill occurs after
            processing 20 jobs, followed by a final refill when both are empty.
            - If `N > BUFFER_SIZE`: Initial fetch fills the queue to `BUFFER_SIZE`, and subsequent
            refills occur every 20 completions until Cosmos DB is exhausted.

        Notes:
            - The function uses a thread pool executor to manage concurrent job processing.
            - Jobs are fetched from Cosmos DB and enqueued locally for processing.
            - Exception handling ensures that crashed jobs are logged without halting the loop.

        """
        in_flight = set()  # Set[Future]
        processed_since_fetch = 0
        logger.info("Sync worker thread started")

        # Step 1: Initial fill of the local queue
        self._fetch_and_prepare_jobs(self.BUFFER_SIZE)

        while True:
            # Step 2: Schedule jobs up to `MAX_WORKERS`
            while len(in_flight) < self.MAX_WORKERS and not self.pending_queue.empty():
                doc, req_det, ivce_dtl = self.pending_queue.get_nowait()

                # Submit the job to the executor; the thread will immediately mark it 'processing'
                fut = self.executor.submit(self._sync_handle, doc, req_det, ivce_dtl)
                in_flight.add(fut)

            # Step 3: Reap completed futures
            done = {f for f in in_flight if f.done()}
            for f in done:
                in_flight.remove(f)
                processed_since_fetch += 1
                try:
                    request_id = f.result()
                    if request_id:
                        self.req_tracker.increment_processed(request_id)
                except Exception as e:
                    logger.exception(f"Job crashed {str(e)}")

            if self._has_app_refreshed and len(in_flight) > 0:
                continue

            if self._has_app_refreshed:
                self._has_app_refreshed = False
                self.update_settings_from_config()

            # Step 4: Refill the queue if needed
            # Conditions for refill:
            #   a) Processed jobs exceed the `REFILL_THRESH`
            #   b) Both in-flight jobs and the local queue are empty
            if not self.should_pause_processing:
                if processed_since_fetch >= self.REFILL_THRESH or (not in_flight and self.pending_queue.empty()):
                    to_fetch = self.BUFFER_SIZE - self.pending_queue.qsize()
                    if to_fetch > 0:
                        self._fetch_and_prepare_jobs(to_fetch)
                    processed_since_fetch = 0

            # Step 5: Back off to avoid excessive spinning
            time.sleep(self.POLL_INTERVAL)

    def _sync_handle(self, doc, request_details, ivce_dtl) -> str | None:
        """
        Handles the processing of a single document within a thread pool. The function
        marks the document as 'processing', processes the invoice detail, and updates
        the document status in Cosmos DB based on the outcome.

        Args:
            doc (dict): The document fetched from Cosmos DB that needs processing.
            request_details (dict): Metadata and details related to the request.
            ivce_dtl (InvoiceDetail): The invoice detail object associated with the document.

        Workflow:
            1. Mark the document as 'processing' in Cosmos DB.
            2. Process the invoice detail using `process_single_invoice_detail`.
            3. On success, mark the document as 'done' in Cosmos DB.
            4. On failure, log the error, mark the document as 'error', and update Cosmos DB.

        Notes:
            - The function uses `asyncio.run` to process the invoice detail asynchronously.
            - Exception handling ensures that errors are logged and the document status is updated
            to 'error' in Cosmos DB.

        """
        try:
            # Step 1: Mark the document as 'processing'
            doc.update({"status": CosmosLogStatus.PROCESSING_lower, "updated_at": get_current_datetime_cst()})
            self.cdb.update_document(document=doc, container=self.cdb.ai_jobs_container)

            # Step 2: Process the invoice detail asynchronously
            asyncio.run(self.ie.process_single_invoice_detail(request_details, ivce_dtl))

            # Step 3: On success, mark the document as 'done'
            doc.update({"status": CosmosLogStatus.DONE_lower, "updated_at": get_current_datetime_cst()})
            self.cdb.update_document(document=doc, container=self.cdb.ai_jobs_container)

            self.req_tracker.increment_processed(req_id=request_details["id"])
            return doc.get("request_details", {}).get("request_id")

        except Exception as e:
            # Step 4: On failure, log the error and mark the document as 'error'
            logger.error(f"Error in worker handle(): {str(e)}", exc_info=True)
            doc.update(
                {
                    "status": CosmosLogStatus.ERROR_lower,
                    "message": f"Error in worker handle(): {str(e)}",
                    "updated_at": get_current_datetime_cst(),
                }
            )
            self.cdb.update_document(document=doc, container=self.cdb.ai_jobs_container)


class RequestsTracker:
    """Tracks per-request processing progress and status internally."""

    class _PriorityRequests:
        def __init__(self, config: Config, cdb_utils: CDB_Utils) -> None:
            # def __init__(self, config: Config) -> None:
            self.config = config
            self.cdb_utils = cdb_utils
            self.refresh_config()

        def refresh_config(self):
            """Pull in the use_priority_queue flag and the prioritized_request_ids dictionary from config
            and set all as not queued"""
            self.use_priority_queue = self.config.use_priority_queue
            self.priority_queue = self.config.prioritized_request_ids
            self._build_df_priority_queue()

        def _build_df_priority_queue(self):
            """Build out the dataframe from the dictionary from CONFIG, set all as NOT_QUEUED by default"""
            self.df_priority_queue = pd.DataFrame(list(self.priority_queue.items()), columns=["request_id", "priority"])
            self.df_priority_queue["status"] = PriorityRequestStatus.NOT_QUEUED
            statuses = self.cdb_utils.get_request_id_statuses(self.df_priority_queue["request_id"].tolist())
            status_dict = {doc["id"]: doc["status"] for doc in statuses}
            # Update status from api request cosmos statuses, otherwise assume unqueued
            self.df_priority_queue["status"] = self.df_priority_queue.apply(
                lambda row: status_mapper.get(status_dict.get(row["request_id"]), PriorityRequestStatus.NOT_QUEUED), axis=1
            )

        def get_request_ids(self) -> Iterator[str]:
            """Iterator that returns the request ids in priority order"""
            if not self.use_priority_queue:
                return iter([])
            prioritized = self.df_priority_queue[self.df_priority_queue["status"].isin([PriorityRequestStatus.QUEUED])]
            if not prioritized.empty:
                request_id = prioritized.sort_values(by="priority", ascending=False).iloc[0]["request_id"]
                yield request_id

        def mark_completed(self, req_id: str):
            self.df_priority_queue.loc[self.df_priority_queue["request_id"] == req_id, "status"] = PriorityRequestStatus.COMPLETED

        def set_status(self, req_id: str, status: PriorityRequestStatus):
            self.df_priority_queue.loc[self.df_priority_queue["request_id"] == req_id, "status"] = status

    class _RequestDetailTracker:
        """Internal tracker for individual request details (not exposed externally)."""

        def __init__(self, api_req_uuid: str, req_id: str, count: int):
            self.api_req_uuid = api_req_uuid
            self.req_id = req_id
            self.total_details_count = count
            self.processed_count = 0
            self.status = CosmosLogStatus.PROCESSING_lower
            self.has_api_status_updated = False

        def increment_processed(self, step: int = 1):
            self.processed_count += step
            if self.processed_count >= self.total_details_count:
                self.status = CosmosLogStatus.DONE_lower

        def mark_failed(self):
            self.status = CosmosLogStatus.ERROR_lower

        def set_api_status_updated(self, flag: bool):
            self.has_api_status_updated = flag

        def to_dict(self):
            return {
                "total_details_count": self.total_details_count,
                "processed_count": self.processed_count,
                "status": self.status,
                "has_api_status_updated": self.has_api_status_updated,
            }

    def __init__(self, cdb_utils, config):
        self.cdb_utils = cdb_utils
        self._trackers_dict: dict[str, RequestsTracker._RequestDetailTracker] = {}
        self.config = config
        self._priority_requests = RequestsTracker._PriorityRequests(config, self.cdb_utils)

    # ---------- Public API ---------- #

    def add_request(self, api_req_uuid: str, req_id: str, count: int):
        """Create a new request detail tracker and update with processed jobs count"""
        # Create detail tracker object
        req_tracker = self._RequestDetailTracker(api_req_uuid=api_req_uuid, req_id=req_id, count=count)
        self._trackers_dict[req_id] = req_tracker

        # Get processed jobs from queue
        processed_count = self.cdb_utils.get_processed_jobs_count(req_id=req_id)

        # Update processed count in object
        if processed_count > 0:
            req_tracker.increment_processed(step=processed_count)

        self.cdb_utils.update_api_request_status(api_id=api_req_uuid, status=CosmosLogStatus.PROCESSING_lower)

    def increment_processed(self, req_id: str, step: int = 1):
        """Increase processed count for given request."""
        if tracker := self._trackers_dict.get(req_id):
            tracker.increment_processed(step)

            if tracker.status == CosmosLogStatus.DONE_lower:
                result = self.cdb_utils.update_api_request_status(api_id=tracker.api_req_uuid, status=CosmosLogStatus.DONE_lower)
                self._priority_requests.mark_completed(req_id)

                if result[Constants.STATUS_lower] == Constants.SUCCESS_lower:
                    tracker.has_api_status_updated = True

    def mark_failed(self, req_id: str):
        """Mark a request as failed."""
        if tracker := self._trackers_dict.get(req_id):
            tracker.mark_failed()
        self.cdb_utils.update_request_status(self.request_id, self.status)

    def set_api_status_updated(self, req_id: str, flag: bool = True):
        """Set the has_api_status_updated flag."""
        if tracker := self._trackers_dict.get(req_id):
            tracker.set_api_status_updated(flag)

    def get_status(self, req_id: str) -> Optional[CosmosLogStatus]:
        """Get the current status of a request."""
        if tracker := self._trackers_dict.get(req_id):
            return tracker.status
        return None

    def get_processed_count(self, req_id: str) -> Optional[int]:
        """Get how many details have been processed."""
        if tracker := self._trackers_dict.get(req_id):
            return tracker.processed_count
        return None

    def get_total_count(self, req_id: str) -> Optional[int]:
        """Get total details count."""
        if tracker := self._trackers_dict.get(req_id):
            return tracker.total_details_count
        return None

    def has_api_status_updated(self, req_id: str) -> Optional[bool]:
        """Check if API status update flag is set."""
        if tracker := self._trackers_dict.get(req_id):
            return tracker.has_api_status_updated
        return None

    def get_request_ids(self) -> Iterator[str]:
        """Return an iterator of all the request ids that have a priority in priority order"""
        for request_id in self._priority_requests.get_request_ids():
            yield request_id

    def refresh_config(self):
        """Refresh priorities from the app config"""
        self._priority_requests.refresh_config()

    def set_request_id_status(self, request_id: str, status: PriorityRequestStatus):
        self._priority_requests.set_status(request_id, status)

    def summary(self):
        """Return current state for all requests (for logging/debug)."""
        return {rid: t.to_dict() for rid, t in self._trackers_dict.items()}
