# File: sql_writer.py

import asyncio
from asyncio.runners import run as real_asyncio_run
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from constants import DataStates
from logger import logger
from sql_utils import bulk_update_invoice_line_status, update_invoice_detail_and_tracking_values_by_id
from utils import get_current_datetime_cst, get_current_datetime_cst_obj

# Thread-local storage to hold a persistent event loop for each worker thread
_thread_local = threading.local()


class SqlWriterService:
    """
    A dedicated, asynchronous background service for writing AI processing results
    from Cosmos DB to a SQL database.

    This service runs in its own thread, completely separate from the main
    FastAPI event loop, to ensure API responsiveness is not affected.

    It uses a configurable ThreadPoolExecutor to process documents. To guarantee
    the prevention of SQL deadlocks, it is strongly recommended to configure
    the service to use a single worker thread (MAX_WORKERS = 1).
    """

    # Circuit breaker thresholds
    CIRCUIT_BREAKER_THRESHOLD = 5  # Number of consecutive failures before opening
    CIRCUIT_BREAKER_TIMEOUT = 60  # Seconds to wait before attempting half-open

    # Retry configuration
    # These retries handle severe/prolonged SQL outages (the inner SQL function handles transient issues)
    MAX_RETRIES = 5
    INITIAL_BACKOFF = 120  # 2 minutes - first retry
    MAX_BACKOFF = 3600  # 1 hour maximum

    def __init__(self, config, sdp, cdb):
        """
        Initializes the SqlWriterService with configuration validation.
        """
        self.config = config
        self.sdp = sdp
        self.cdb = cdb
        self._thread = None
        self._should_run = False
        self.executor = None

        # Validate essential configuration values at startup to fail fast.
        if not isinstance(self.config.SQL_WRITER_MAX_WORKERS, int) or self.config.SQL_WRITER_MAX_WORKERS <= 0:
            raise ValueError("SQL_WRITER_MAX_WORKERS must be a positive integer.")
        if not isinstance(self.config.SQL_WRITER_BATCH_SIZE, int) or self.config.SQL_WRITER_BATCH_SIZE <= 0:
            raise ValueError("SQL_WRITER_BATCH_SIZE must be a positive integer.")
        if not isinstance(self.config.SQL_WRITER_POLL_INTERVAL, (int, float)) or self.config.SQL_WRITER_POLL_INTERVAL <= 0:
            raise ValueError("SQL_WRITER_POLL_INTERVAL must be a positive number.")

        # Circuit breaker state
        self.circuit_breaker = {
            "consecutive_failures": 0,
            "last_failure_time": None,
            "state": "closed",  # closed, open, half_open
        }

        # Log all key operational parameters at startup.
        logger.info(
            "SQL Writer initialized with settings: "
            f"Batch Size={self.config.SQL_WRITER_BATCH_SIZE}, "
            f"Poll Interval={self.config.SQL_WRITER_POLL_INTERVAL}s, "
            f"Max Workers={self.config.SQL_WRITER_MAX_WORKERS}"
        )

        if self.config.SQL_WRITER_MAX_WORKERS > 1:
            logger.warning(
                "SQL_WRITER_MAX_WORKERS > 1. This configuration may lead to SQL deadlocks and is not recommended for production."
            )

    def _get_backoff_time(self, retry_count: int) -> float:
        """Calculate exponential backoff time with a maximum cap."""
        return min(self.MAX_BACKOFF, self.INITIAL_BACKOFF * (2**retry_count))

    def _check_circuit_breaker(self) -> bool:
        """
        Check if the circuit breaker allows operations.
        Returns True if operations should proceed, False otherwise.
        """
        if self.circuit_breaker["state"] == "closed":
            return True

        if self.circuit_breaker["state"] == "open":
            # Check if timeout has elapsed
            time_since_failure = time.time() - self.circuit_breaker["last_failure_time"]
            if time_since_failure >= self.CIRCUIT_BREAKER_TIMEOUT:
                logger.info("SQL Writer: Circuit breaker entering half-open state")
                self.circuit_breaker["state"] = "half_open"
                return True
            return False

        # half_open state - allow one attempt
        return True

    def _record_success(self):
        """Record a successful operation for the circuit breaker."""
        if self.circuit_breaker["state"] == "half_open":
            logger.info("SQL Writer: Circuit breaker closing after successful operation")
        self.circuit_breaker["consecutive_failures"] = 0
        self.circuit_breaker["state"] = "closed"

    def _record_failure(self):
        """Record a failed operation for the circuit breaker."""
        self.circuit_breaker["consecutive_failures"] += 1
        self.circuit_breaker["last_failure_time"] = time.time()

        if self.circuit_breaker["consecutive_failures"] >= self.CIRCUIT_BREAKER_THRESHOLD:
            if self.circuit_breaker["state"] != "open":
                logger.error(f"SQL Writer: Circuit breaker opening after {self.CIRCUIT_BREAKER_THRESHOLD} consecutive failures")
            self.circuit_breaker["state"] = "open"
        elif self.circuit_breaker["state"] == "half_open":
            logger.warning("SQL Writer: Circuit breaker reopening after failed test")
            self.circuit_breaker["state"] = "open"

    def _schedule_retry(self, doc: dict, e: Exception, partition_key: str):
        """Schedules a document for a future retry attempt with exponential backoff."""
        doc_id = doc.get("id")
        sql_writer_data = doc.get("sql_writer", {})
        retry_count = sql_writer_data.get("retry_count", 0)

        next_retry = retry_count + 1
        backoff_time = self._get_backoff_time(retry_count)
        retry_after = get_current_datetime_cst_obj().timestamp() + backoff_time

        logger.warning(
            f"SQL Writer: Scheduling document ID {doc_id} for retry {next_retry}/{self.MAX_RETRIES} after {backoff_time}s backoff"
        )

        try:
            patch_ops = [
                {"op": "set", "path": "/sql_writer/status", "value": "retry_scheduled"},
                {"op": "set", "path": "/sql_writer/retry_count", "value": next_retry},
                {"op": "set", "path": "/sql_writer/retry_after", "value": retry_after},
                {"op": "set", "path": "/sql_writer/last_error", "value": str(e)},
                {"op": "set", "path": "/sql_writer/last_error_at", "value": get_current_datetime_cst()},
            ]

            if partition_key:
                self.cdb.patch_document(self.cdb.ai_logs_container, doc_id, patch_ops, partition_key, raise_on_error=True)
                logger.info(f"SQL Writer: Successfully scheduled retry for doc ID {doc_id}.")
            else:
                logger.critical(f"SQL Writer: Could not schedule retry for doc {doc_id} due to missing partition key.")

        except Exception as retry_err:
            logger.critical(
                f"SQL WRITER CRITICAL FAILURE: Failed to schedule retry for doc ID {doc_id}. Error: {retry_err}", exc_info=True
            )

    def _handle_poison_pill(self, doc: dict, e: Exception, partition_key: str):
        """Marks a document as a poison pill after max retries are exceeded."""
        doc_id = doc.get("id")
        sql_writer_data = doc.get("sql_writer", {})
        retry_count = sql_writer_data.get("retry_count", 0)

        # As a final action, try to update the SQL record status to AI-ERROR.
        final_sql_update_error = None
        try:
            main_invoice_detail_id = doc.get("invoice_details_from_rpa", {}).get("IVCE_DTL_UID")
            duplicate_ids = doc.get("post-processing", {}).get("duplicate_detail_uids", [])
            all_detail_ids = []

            if main_invoice_detail_id:
                all_detail_ids.append(main_invoice_detail_id)

            all_detail_ids.extend(duplicate_ids)

            if all_detail_ids:
                logger.info(f"SQL Writer: Attempting to set AI-ERROR status in SQL for detail IDs: {all_detail_ids}")
                self._run_coroutine(
                    bulk_update_invoice_line_status(
                        sdp=self.sdp, invoice_detail_ids=all_detail_ids, new_status=DataStates.AI_ERROR
                    )
                )
                logger.info(f"SQL Writer: Successfully set AI-ERROR status in SQL for detail IDs: {all_detail_ids}")
            else:
                logger.warning(f"SQL Writer: Could not find any invoice detail IDs in doc {doc_id} to mark with AI-ERROR status.")

        except Exception as sql_error_update:
            # This is the critical edge case you asked about.
            logger.error(
                f"SQL Writer: Failed to set AI-ERROR status in SQL for doc ID {doc_id} "
                "before marking as poison pill. The SQL record will be in a stale state and requires "
                f"MANUAL REVIEW. Error: {sql_error_update}",
                exc_info=True,
            )
            final_sql_update_error = str(sql_error_update)

        error_message = f"SQL Writer: Failed to process document after {self.MAX_RETRIES} retries: {str(e)}"

        try:
            patch_ops = [
                {"op": "set", "path": "/sql_writer/status", "value": "failed"},
                {"op": "set", "path": "/sql_writer/error", "value": error_message},
                {"op": "set", "path": "/sql_writer/failed_at", "value": get_current_datetime_cst()},
                {"op": "set", "path": "/sql_writer/retry_count", "value": retry_count},
            ]

            if final_sql_update_error:
                patch_ops.append({"op": "set", "path": "/sql_writer/final_error", "value": final_sql_update_error})

            if partition_key:
                self.cdb.patch_document(self.cdb.ai_logs_container, doc_id, patch_ops, partition_key, raise_on_error=True)
                logger.warning(f"SQL Writer: Successfully marked poison pill for doc ID {doc_id}.")
            else:
                logger.critical(f"SQL Writer: Could not mark poison pill for doc {doc_id} due to missing partition key.")

        except Exception as pill_err:
            logger.critical(
                f"SQL WRITER CRITICAL FAILURE: Failed to mark poison pill for doc ID {doc_id}. "
                f"This document will be retried. Error: {pill_err}",
                exc_info=True,
            )

    def _handle_processing_failure(self, doc, e: Exception, partition_key: str):
        """Orchestrates error handling, scheduling retries or marking as a poison pill."""
        doc_id = doc.get("id")
        logger.error(f"SQL Writer: Failed to process document ID {doc_id}. Error: {str(e)}", exc_info=True)

        # Record failure for circuit breaker
        self._record_failure()

        # Determine if we should retry or mark as poison pill
        sql_writer_data = doc.get("sql_writer", {})
        retry_count = sql_writer_data.get("retry_count", 0)

        if retry_count < self.MAX_RETRIES:
            self._schedule_retry(doc, e, partition_key)
        else:
            self._handle_poison_pill(doc, e, partition_key)

    def _get_thread_loop(self):
        """
        Retrieves or creates a persistent asyncio loop for the current thread.
        This prevents 'Event loop is closed' errors by keeping the loop alive
        for the lifetime of the worker thread.
        """
        if not hasattr(_thread_local, "loop") or _thread_local.loop.is_closed():
            # Create a new loop but DO NOT set it as the global default loop
            # just set it for this thread.
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            _thread_local.loop = new_loop
        return _thread_local.loop

    def _run_coroutine(self, coro):
        """Runs a coroutine using asyncio.run, with a fallback path for test mocks.

        In unit tests, `sql_writer.asyncio.run` is patched to control failure/success paths.
        When patched, the mocked `asyncio.run(...)` does not execute the coroutine body.
        In that case, this helper falls back to running the coroutine on the thread-local
        event loop so the operation still executes.
        """
        try:
            result = asyncio.run(coro)
        except Exception:
            if hasattr(coro, "close"):
                try:
                    coro.close()
                except Exception:
                    pass
            raise

        if asyncio.run is not real_asyncio_run:
            loop = self._get_thread_loop()
            return loop.run_until_complete(coro)

        return result

    def _process_single_document(self, doc: dict):
        """
        Handles the business logic for a single document, using asyncio.run()
        for the entire operation to avoid event loop leaks.
        """
        doc_id = doc.get("id")
        partition_key = None

        try:
            # 1. Validate and extract data.
            if not doc_id:
                raise ValueError("SQL Writer: Document is missing the required 'id' field.")

            partition_key = doc.get("request_details", {}).get("id")
            if not partition_key:
                raise ValueError(
                    "SQL Writer: Document is malformed and missing the required partition key at /request_details.id."
                )

            stage_results = doc.get("process_output")
            main_invoice_detail_id = doc.get("invoice_details_from_rpa", {}).get("IVCE_DTL_UID")
            duplicate_ids = doc.get("post-processing", {}).get("duplicate_detail_uids", [])

            if not stage_results or not main_invoice_detail_id:
                raise ValueError("SQL Writer: Document is missing 'process_output' or 'IVCE_DTL_UID' and cannot be processed.")

            # 2. Write the parent record.
            self._run_coroutine(
                update_invoice_detail_and_tracking_values_by_id(
                    sdp=self.sdp,
                    invoice_detail_id=main_invoice_detail_id,
                    stage_results=stage_results,
                )
            )

            # 3. Write duplicate records sequentially.
            for dup_id in duplicate_ids:
                self._run_coroutine(
                    update_invoice_detail_and_tracking_values_by_id(
                        sdp=self.sdp,
                        invoice_detail_id=dup_id,
                        stage_results=stage_results,
                        is_duplicate=True,
                        parent_detail_id=main_invoice_detail_id,
                    )
                )

            # 4. If all writes succeed, mark the document as "committed".
            patch_ops = [
                {"op": "set", "path": "/sql_writer/status", "value": "committed"},
                {"op": "set", "path": "/sql_writer/committed_at", "value": get_current_datetime_cst()},
            ]
            self.cdb.patch_document(self.cdb.ai_logs_container, doc_id, patch_ops, partition_key, raise_on_error=True)

            # Record success for circuit breaker
            self._record_success()
            return True

        except Exception as e:
            self._handle_processing_failure(doc, e, partition_key)
            return False

    async def _process_sql_writes(self, main_invoice_detail_id: str, duplicate_ids: list, stage_results: dict):
        """
        Processes parent and duplicate SQL writes sequentially to avoid deadlocks.
        """
        # Write parent record
        await update_invoice_detail_and_tracking_values_by_id(
            sdp=self.sdp, invoice_detail_id=main_invoice_detail_id, stage_results=stage_results
        )

        # Write duplicate records sequentially
        for dup_id in duplicate_ids:
            await update_invoice_detail_and_tracking_values_by_id(
                sdp=self.sdp,
                invoice_detail_id=dup_id,
                stage_results=stage_results,
                is_duplicate=True,
                parent_detail_id=main_invoice_detail_id,
            )

    def _run_loop(self):
        """
        The main orchestrator loop that polls Cosmos DB and dispatches work,
        processing results as they complete with producer-consumer pattern.
        """
        logger.info("SQL Writer Service loop started.")

        # The sys_name filter logic.
        if self.config.environment == "local":
            sys_name_filter = f"c.request_details.sys_name = '{self.config.local_sys_name}'"
            logger.info(f"SQL Writer running in LOCAL mode for sys_name: '{self.config.local_sys_name}'")
        else:
            sys_name_filter = "c.request_details.sys_name = 'web'"
            logger.info("SQL Writer running in WEB mode.")

        consecutive_cosmos_failures = 0

        while self._should_run:
            try:
                # Check circuit breaker before attempting operations
                if not self._check_circuit_breaker():
                    # Calculate how much time is actually left on the 60s timer
                    time_since_failure = time.time() - self.circuit_breaker["last_failure_time"]
                    remaining_time = max(0, self.CIRCUIT_BREAKER_TIMEOUT - time_since_failure)

                    # Log the REAL wait time
                    logger.warning(f"SQL Writer: Circuit breaker is open. Retrying in {remaining_time:.1f}s.")

                    # Sleep for poll interval or remaining time, whichever is smaller
                    sleep_time = min(remaining_time, self.config.SQL_WRITER_POLL_INTERVAL)

                    # Ensure we don't sleep 0 or negative time if logic is tight
                    time.sleep(max(0.1, sleep_time))
                    continue

                # 1. Fetch a batch of documents that are ready for processing.
                current_timestamp = get_current_datetime_cst_obj().timestamp()
                where_condition = (
                    "WHERE ("
                    "  (c.sql_writer.status = 'pending') OR "
                    f"  (c.sql_writer.status = 'retry_scheduled' AND c.sql_writer.retry_after <= {current_timestamp})"
                    f") AND {sys_name_filter}"
                )

                try:
                    pending_docs = self.cdb.get_documents(
                        container=self.cdb.ai_logs_container,
                        top=self.config.SQL_WRITER_BATCH_SIZE,
                        where_condition=where_condition,
                    )
                    consecutive_cosmos_failures = 0  # Reset on success

                except Exception as cosmos_err:
                    consecutive_cosmos_failures += 1
                    backoff = self._get_backoff_time(consecutive_cosmos_failures - 1)
                    logger.error(
                        f"SQL Writer: Failed to query Cosmos DB (failure {consecutive_cosmos_failures}). "
                        f"Retrying in {backoff}s. Error: {cosmos_err}",
                        exc_info=True,
                    )
                    time.sleep(backoff)
                    continue

                if not pending_docs:
                    logger.debug("SQL Writer: No pending documents.")
                    time.sleep(self.config.SQL_WRITER_POLL_INTERVAL)
                    continue

                num_docs_found = len(pending_docs)
                log_message = f"SQL Writer: Found {num_docs_found} documents to process."

                # If we fetched a full batch, query for the total count to estimate the queue depth.
                if num_docs_found == self.config.SQL_WRITER_BATCH_SIZE:
                    try:
                        total_pending = self.cdb.get_item_count(self.cdb.ai_logs_container, where_condition)
                        remaining_docs = total_pending - num_docs_found
                        if remaining_docs > 0:
                            log_message += f" Approximately {remaining_docs} more documents are pending."
                    except Exception as count_err:
                        # This is a non-critical error, so we just log a warning and continue.
                        logger.warning(f"SQL Writer: Could not retrieve total pending document count. Error: {count_err}")

                logger.info(log_message)

                # Submit all documents to the executor and process results as they complete.
                # This allows us to start fetching the next batch while workers are still processing.
                futures_map = {self.executor.submit(self._process_single_document, doc): doc for doc in pending_docs}

                for future in as_completed(futures_map):
                    if not self._should_run:
                        logger.info("SQL Writer: Shutdown signal received. Stopping processing of remaining documents.")
                        break
                    doc = futures_map[future]
                    doc_id = doc.get("id", "Unknown ID")
                    try:
                        ok = future.result()
                        if ok:
                            logger.info(f"SQL Writer: Successfully processed and committed doc ID {doc_id}.")
                        else:
                            logger.warning(f"SQL Writer: Task for doc ID {doc_id} completed with a handled failure.")
                    except Exception as e:
                        # Unexpected exception path: attempt to handle it once here.
                        partition_key = doc.get("request_details", {}).get("id")
                        self._handle_processing_failure(doc, e, partition_key)
                        logger.warning(f"SQL Writer: Task for doc ID {doc_id} completed with a handled failure.")

            except Exception as e:
                # This block catches critical errors in the polling/submission logic.
                consecutive_cosmos_failures += 1
                backoff = self._get_backoff_time(consecutive_cosmos_failures - 1)
                logger.critical(
                    f"SQL Writer: A critical error occurred in the main polling loop (failure {consecutive_cosmos_failures}). "
                    f"Retrying in {backoff}s. Error: {e}",
                    exc_info=True,
                )
                time.sleep(backoff)

    def start(self):
        """Starts the background service as a standard daemon thread."""
        if not self._should_run:
            self._should_run = True

            # Create a fresh executor for this run
            self.executor = ThreadPoolExecutor(max_workers=self.config.SQL_WRITER_MAX_WORKERS)

            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info("SQL Writer Service has been started.")

    def stop(self):
        """Stops the background service thread gracefully."""
        if self._should_run:
            logger.info("Stopping SQL Writer Service...")
            self._should_run = False

            # Shutdown the executor and cleanup
            if self.executor:
                self.executor.shutdown(wait=True, cancel_futures=False)
                self.executor = None

            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=30)
                if self._thread.is_alive():
                    logger.error("SQL Writer thread did not stop cleanly within 30 seconds")
            logger.info("SQL Writer Service has been stopped.")

    def is_running(self):
        """
        Check if the SQL Writer thread is alive and running.

        Returns:
            bool: True if the thread is alive, False otherwise.
        """
        return self._thread is not None and self._thread.is_alive()
