"""
Indexer script for populating Azure AI Search index with data from Azure SQL database.
This script uses multi-threading for better performance.
"""

import argparse
import asyncio
import logging
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from azure.core.exceptions import HttpResponseError
from azure.search.documents.indexes.models import SearchableField, SearchField, SearchFieldDataType, SimpleField

from azure_search_utils import AzureSearchUtils
from config import Config
from llm import LLM
from sdp import SDP
from sql_utils import get_all_index_data
from utils import remove_separators

# Logging configuration
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.propagate = False


class ProgressTracker:
    """Helper class for tracking and reporting progress"""

    def __init__(self, total_count, report_frequency=0.05, description="Progress"):
        """
        Initialize progress tracker

        Args:
            total_count: Total number of records to process
            report_frequency: Report after this fraction of progress (0.05 = every 5%)
            description: Description for progress reports
        """
        self.total_count = total_count
        self.current_count = 0
        self.start_time = time.time()
        self.last_report_count = 0
        self.description = description

        # Calculate threshold for progress reporting (max 5% of total or 5000 records)
        self.report_threshold = max(1, min(int(total_count * report_frequency), 5000))

    def update(self, increment=1):
        """
        Update progress and report if threshold reached

        Args:
            increment: Number of records processed in this update

        Returns:
            bool: True if progress report was generated
        """
        self.current_count += increment

        # Check if we should report progress
        if self.total_count > 0 and self.current_count - self.last_report_count >= self.report_threshold:
            self._report_progress()
            self.last_report_count = self.current_count
            return True
        return False

    def _report_progress(self):
        """Report current progress with rate and time estimates"""
        elapsed_time = time.time() - self.start_time
        progress_percent = (self.current_count / self.total_count) * 100
        records_per_second = self.current_count / elapsed_time if elapsed_time > 0 else 0

        # Estimate time remaining
        time_remaining_str = ""
        if records_per_second > 0:
            estimated_total_time = self.total_count / records_per_second
            estimated_remaining = estimated_total_time - elapsed_time
            time_remaining_str = f", est. {estimated_remaining:.1f}s remaining"

        logger.info(
            f"{self.description}: {self.current_count}/{self.total_count} records "
            f"({progress_percent:.1f}%), {records_per_second:.1f} records/sec{time_remaining_str}"
        )

    def final_report(self):
        """Generate final progress report"""
        elapsed_time_sec = time.time() - self.start_time
        elapsed_time_min = elapsed_time_sec / 60
        records_per_second = self.current_count / elapsed_time_sec if elapsed_time_sec > 0 else 0

        logger.info(
            f"{self.description} completed. Processed {self.current_count}/{self.total_count} records "
            f"({(self.current_count / self.total_count) * 100:.1f}%) in {elapsed_time_min:.1f}m "
            f"({records_per_second:.1f} records/sec)"
        )


class Indexer:
    """Main indexer class to manage data synchronization between SQL and Search index"""

    def __init__(self, config: Config, rebuild: bool = False, max_records: Optional[int] = None):
        """
        Initialize the indexer using parameters from the Config object.

        Args:
            config: Configuration object loaded from YAML or similar.
            rebuild: Whether to rebuild the index.
            max_records: Maximum number of records to process.
        """
        self.config = config
        self.rebuild = rebuild
        self.max_records = max_records

        # --- Read indexer settings from config ---
        indexer_cfg = getattr(config, "indexer_settings", {})  # Get the sub-dictionary

        self.max_quota_retries = indexer_cfg.get("max_quota_retries", 5)
        self.quota_retry_delay_minutes = indexer_cfg.get("quota_retry_delay_minutes", 10)
        self.batch_size = indexer_cfg.get("batch_size", 400)  # Default if not in config
        self.max_queue_size = indexer_cfg.get("max_queue_size", 4000)  # Default based on batch_size if missing
        self.min_sql_records = indexer_cfg.get("min_sql_records", 100000)  # Lower default safety
        # --- End reading settings ---

        # Initialize main components
        self.sdp = SDP(config)
        self.search_utils = AzureSearchUtils(config)
        self.llm = LLM(config)

        logger.info(f"Initializing Indexer. Rebuild: {rebuild}, Max Records: {max_records}")
        logger.info(f" Batch Size: {self.batch_size}, Max Queue Size: {self.max_queue_size}")
        logger.info(f" Quota Retries: Max={self.max_quota_retries}, Delay={self.quota_retry_delay_minutes} min")
        logger.info(f" Min SQL Records Required: {self.min_sql_records}")
        logger.info(f" SDP Connection: Server='{self.config.SDP_SERVER}', Database='{self.config.SDP_DATABASE}'")
        logger.info(f" Search Index: Endpoint='{self.search_utils.endpoint}', Index='{self.search_utils.index_name}'")

        # Data queues for inter-thread communication - use maxsize read from config
        self.sql_data_queue = queue.Queue(maxsize=self.max_queue_size)
        self.index_data_queue = queue.Queue(maxsize=self.max_queue_size)
        self.embedding_queue = queue.Queue(maxsize=self.max_queue_size)
        self.saving_queue = queue.Queue(maxsize=self.max_queue_size)

        # Control flags
        self.sql_data_complete = threading.Event()
        self.index_data_complete = threading.Event()
        self.processing_complete = threading.Event()
        self.embedding_complete = threading.Event()

        # Metrics
        self.records_processed = 0
        self.records_created = 0
        self.records_updated = 0
        self.records_deleted = 0
        self.embeddings_calculated = 0

        self.metrics_lock = threading.Lock()

        # Status/Error flags
        self.insufficient_sql_data = False
        self.sql_connection_error = False
        self.index_connection_error = False
        self.terminate_event = threading.Event()

    def _terminate_process(self):
        """Signal all threads to terminate due to critical error"""
        logger.error("Critical error detected. Terminating all processing.")
        self.terminate_event.set()
        # Set all completion events to avoid deadlocks
        self.sql_data_complete.set()
        self.index_data_complete.set()
        self.processing_complete.set()
        self.embedding_complete.set()
        # Drain the queues to prevent "Missing embedding" errors
        self._drain_queues()

    def _drain_queues(self):
        """Empty all queues to prevent downstream processing"""
        logger.info("Draining queues to prevent further processing...")
        # Empty all queues
        try:
            while not self.sql_data_queue.empty():
                self.sql_data_queue.get_nowait()
            while not self.index_data_queue.empty():
                self.index_data_queue.get_nowait()
            while not self.embedding_queue.empty():
                self.embedding_queue.get_nowait()
            while not self.saving_queue.empty():
                self.saving_queue.get_nowait()
            logger.info("All queues have been drained.")
        except Exception as e:
            logger.error(f"Error draining queues: {str(e)}")

    def prepare_index(self):
        """Prepare the search index (create or rebuild if needed)"""
        logger.info("Checking search index status...")

        if self.rebuild and self.search_utils.index_exists():
            logger.info("Deleting existing index due to rebuild flag...")
            self.search_utils.delete_index()

        if not self.search_utils.index_exists():
            logger.info("Creating search index...")
            self.create_search_index()
            logger.info("Search index created successfully.")
        else:
            logger.info("Search index already exists. Proceeding with data synchronization.")

    def create_search_index(self):
        """Create search index with predefined schema and configuration"""
        try:
            # Define fields for the index
            fields = [
                # Key field (exact match for unique identifier)
                SimpleField(name="DescriptionID", type=SearchFieldDataType.String, key=True, filterable=True, sortable=True),
                # MfrPartNum original value
                SearchableField(
                    name="MfrPartNum",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    searchable=True,
                    analyzer_name="custom_keyword_analyzer",
                ),
                # MfrPartNum with -/. chars removed
                SearchableField(
                    name="MfrPartNumExact",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    searchable=True,
                    analyzer_name="custom_keyword_analyzer",
                ),
                # Prefix match field with -/. chars removed
                SearchableField(
                    name="MfrPartNumPrefix",
                    type=SearchFieldDataType.String,
                    searchable=True,
                    filterable=False,
                    index_analyzer_name="ngram_front_analyzer",
                    search_analyzer_name="custom_keyword_analyzer",
                ),
                SearchableField(name="UPC", type=SearchFieldDataType.String, filterable=True, searchable=True),
                SearchableField(name="AKPartNum", type=SearchFieldDataType.String, filterable=True, searchable=True),
                # Vector field configuration for semantic search
                SearchField(
                    name="ItemDescription_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    hidden=False,
                    vector_search_dimensions=self.config.AOAI_EMBEDDING_DIMENSIONS,
                    vector_search_profile_name="vector-profile",
                ),
                # Other fields
                SimpleField(name="ItemID", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                SimpleField(name="MfrName", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="UNSPSC", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="ItemSourceName", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="ItemDescription", type=SearchFieldDataType.String, searchable=True),
                SimpleField(name="DescSourceName", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="ItemLastModified", type=SearchFieldDataType.DateTimeOffset),
                SimpleField(name="DescLastModified", type=SearchFieldDataType.DateTimeOffset),
            ]

            # Define vector search configuration - recommended settings for text-embedding-3-large (1536D)
            vector_search = self.search_utils.create_vector_search_config(
                profile_name="vector-profile",
                algorithm_name="vector-config",
                m=8,
                ef_construction=600,
                ef_search=800,
                metric="cosine",
            )

            # Define semantic search configuration
            semantic_search = self.search_utils.create_semantic_search_config(
                config_name="semantic-config", content_field_names=["ItemDescription"], title_field_name=None
            )

            # Define custom analyzers and token filters
            custom_analyzers = [
                {
                    "name": "ngram_front_analyzer",
                    "@odata.type": "#Microsoft.Azure.Search.CustomAnalyzer",
                    "tokenizer": "keyword_v2",
                    "tokenFilters": ["lowercase", "front_edgeNGram"],  # Order matters
                },
                {
                    "name": "custom_keyword_analyzer",
                    "@odata.type": "#Microsoft.Azure.Search.CustomAnalyzer",
                    "tokenizer": "keyword_v2",
                    "tokenFilters": ["lowercase"],
                },
            ]

            token_filters = [
                {
                    "name": "front_edgeNGram",
                    "@odata.type": "#Microsoft.Azure.Search.EdgeNGramTokenFilterV2",
                    "minGram": 6,
                    "maxGram": 30,
                    "side": "front",
                }
            ]

            # Create the index with all configurations
            result = self.search_utils.create_index(
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search,
                analyzers=custom_analyzers,
                token_filters=token_filters,
            )

            if not result:
                raise Exception("Failed to create search index")

        except Exception as e:
            logger.error(f"Error creating search index: {str(e)}")
            raise

    async def _process_sql_data_async(self):
        """Asynchronous helper to fetch and process SQL data."""
        records_count = 0
        total_count = 0
        data_generator = None

        try:
            # Get the async generator and total count
            data_generator, total_count = await get_all_index_data(self.sdp, batch_size=self.batch_size)
            logger.info(f"Found {total_count} records in SQL database.")

            # --- Check for minimum records ---
            if total_count < self.min_sql_records:
                logger.warning(
                    f"SQL database returned {total_count} records, which is less than "
                    f"the required minimum of {self.min_sql_records}. "
                    "Stopping indexer to protect search index data."
                )
                self.insufficient_sql_data = True  # Use this flag to indicate the condition
                self._terminate_process()  # Signal termination to other threads
                return records_count, total_count  # Return early

            progress_tracker = ProgressTracker(total_count, description="SQL data processing")

            # Iterate asynchronously over the generator
            async for batch_df in data_generator:
                # Check for termination signal
                if self.terminate_event.is_set():
                    logger.info("Termination signal received, stopping SQL data processing.")
                    break

                # Apply max_records limit if specified
                records_remaining_in_batch = len(batch_df)
                if self.max_records and records_count + records_remaining_in_batch > self.max_records:
                    # Calculate how many records to take from this batch
                    take_count = self.max_records - records_count
                    if take_count <= 0:  # Should not happen if checked before loop, but safety first
                        break
                    batch_df = batch_df.iloc[:take_count]
                    logger.info(f"Truncating batch to respect max_records limit ({self.max_records}).")

                # Process each record in the (potentially truncated) batch
                for _, record in batch_df.iterrows():
                    # Check for termination before potentially blocking put operation
                    if self.terminate_event.is_set():
                        logger.info("Termination signal received during batch processing, stopping.")
                        break  # Break from inner loop

                    # Add individual record to queue - this might block if queue is full
                    # Ensure sql_data_queue is thread-safe (like queue.Queue)
                    self.sql_data_queue.put(record)
                    records_count += 1
                    progress_tracker.update()

                    # Check if we've reached the max_records limit *after* adding the record
                    if self.max_records and records_count >= self.max_records:
                        logger.info(f"Reached max records limit ({self.max_records}). Stopping SQL data reader.")
                        break  # Break from inner loop

                # Check if termination or max_records limit was hit in the inner loop
                if self.terminate_event.is_set() or (self.max_records and records_count >= self.max_records):
                    break  # Break from outer async for loop

            # Final progress report (only if processing started)
            if records_count > 0:
                progress_tracker.final_report()
            elif total_count > 0:  # If we had records but processed none (e.g. immediate termination)
                logger.info("SQL data reader stopped before processing any records.")

            return records_count, total_count  # Return counts

        except Exception as e:
            logger.error(f"Error during asynchronous SQL data processing: {str(e)}", exc_info=True)
            # Re-raise the exception so it's caught by the outer synchronous method
            raise
        finally:
            # Properly close the async generator if it exists
            if data_generator is not None:
                await data_generator.aclose()

    def sql_data_reader(self):
        """Thread for reading data from SQL database"""
        logger.info("Starting SQL data reader thread...")
        loop = None
        try:
            # Create and set asyncio event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run the asynchronous processing function until it completes
            processed_count, total_found = loop.run_until_complete(self._process_sql_data_async())

            # Check if processing happened or if it exited early due to no data
            if not self.insufficient_sql_data:
                if processed_count > 0:
                    logger.info(f"SQL data reader finished processing {processed_count}/{total_found} records.")
                elif total_found > 0:
                    logger.info("SQL data reader finished. No records were processed (likely due to termination or limits).")
                # If total_count was 0, the warning is logged inside _process_sql_data_async

        except Exception as e:
            # Log the error caught from the async function or loop setup
            logger.error(f"Error in SQL data reader thread: {str(e)}", exc_info=True)
            # Set a flag to indicate SQL error
            self.sql_connection_error = True
            # Signal all threads to terminate
            self._terminate_process()
        finally:
            # Close the event loop for this thread
            if loop and not loop.is_closed():
                loop.close()
                logger.info("SQL data reader thread event loop closed.")
            # Signal that SQL data reading (or attempt) is complete
            self.sql_data_complete.set()
            logger.info("SQL data reader thread finished.")

    def index_data_reader(self):
        """Thread for reading data from Search index with composite key filtering"""
        logger.info("Starting index data reader thread...")

        try:
            # Simple wait to allow index creation to complete if needed
            time.sleep(1)

            if not self.search_utils.index_exists():
                logger.info("Index doesn't exist yet. Index data reader will exit.")
                self.index_data_complete.set()
                return

            # Define fields to select from the index
            select_fields = ["DescriptionID", "ItemID", "ItemLastModified", "DescLastModified"]

            # Track last seen keys
            last_item_id = None
            last_desc_id = None
            filter_expression = None

            records_count = 0

            while not self.terminate_event.is_set():
                # Create filter expression for next batch
                if last_item_id is not None:
                    # This composite filter ensures we get all remaining DescriptionIDs for the last ItemID
                    # and then continue with the next ItemIDs
                    filter_expression = (
                        f"(ItemID eq {last_item_id} and DescriptionID gt '{last_desc_id}') or (ItemID gt {last_item_id})"
                    )

                # Get batch from the index
                batch_df = self.search_utils.search(
                    query_text="*",
                    select=select_fields,
                    filter_expression=filter_expression,
                    top=self.batch_size,
                    orderby=["ItemID asc", "DescriptionID asc"],
                )

                # If batch is empty or None, we're done
                if batch_df is None or batch_df.empty:
                    break

                # Update tracking variables for next iteration
                last_item_id = batch_df["ItemID"].iloc[-1]
                last_desc_id = batch_df["DescriptionID"].iloc[-1]

                # Process each record in the batch
                for _, record in batch_df.iterrows():
                    # Check for termination before potentially blocking put operation
                    if self.terminate_event.is_set():
                        break

                    # Add individual record to queue - will block if queue is full
                    self.index_data_queue.put(record)
                    records_count += 1

                    # Check if we've reached the max_records limit
                    if self.max_records and records_count >= self.max_records:
                        logger.info(f"Reached max records limit ({self.max_records}). Stopping index data reader.")
                        break

                # Check if we've reached the max_records limit
                if self.max_records and records_count >= self.max_records:
                    break

            if not self.terminate_event.is_set():
                logger.info(f"Index data reader completed. Found {records_count} records in index.")
            else:
                logger.info("Index data reader terminated due to critical error.")

        except Exception as e:
            logger.error(f"Error in index data reader: {str(e)}")
            # Set a flag to indicate Index error
            self.index_connection_error = True
            # Signal all threads to terminate
            self._terminate_process()
        finally:
            # Signal that all index data has been read
            self.index_data_complete.set()
            logger.info("Index data reader thread finished.")

    def data_processor(self):
        """Thread for processing and comparing data from SQL and index"""
        logger.info("Starting data processor thread...")

        # Stats for debugging
        sql_records_processed = 0
        index_records_processed = 0
        iteration_count = 0

        # Track current ItemID being processed to handle potential gaps
        current_item_id = None

        # Working variables for comparison
        sql_record = None
        index_record = None

        try:
            while not self.terminate_event.is_set():
                iteration_count += 1

                # Check if we should exit the loop
                if (
                    self.sql_data_complete.is_set()
                    and self.index_data_complete.is_set()
                    and self.sql_data_queue.empty()
                    and self.index_data_queue.empty()
                    and sql_record is None
                    and index_record is None
                ):
                    break

                # Get SQL data if needed
                if sql_record is None and not (self.sql_data_complete.is_set() and self.sql_data_queue.empty()):
                    try:
                        sql_record = self.sql_data_queue.get(timeout=1)
                        sql_records_processed += 1
                    except queue.Empty:
                        if not self.sql_data_complete.is_set():
                            logger.debug("Waiting for SQL data...")
                            time.sleep(0.1)  # Wait for data
                            continue

                # Get index data if needed
                if index_record is None and not (self.index_data_complete.is_set() and self.index_data_queue.empty()):
                    try:
                        index_record = self.index_data_queue.get(timeout=1)
                        index_records_processed += 1
                    except queue.Empty:
                        if not self.index_data_complete.is_set():
                            logger.debug("Waiting for index data...")
                            time.sleep(0.1)  # Wait for data
                            continue

                # If we have no data to process at this point, continue to next iteration
                if sql_record is None and index_record is None:
                    time.sleep(0.1)
                    continue

                # Process records based on comparison
                if sql_record is None and index_record is not None:
                    # No more SQL records, but we have index records - delete the index record
                    self._queue_delete_record(index_record)
                    index_record = None

                elif sql_record is not None and index_record is None:
                    # No more index records, but we have SQL records - create the SQL record
                    self._queue_create_record(sql_record)
                    sql_record = None

                else:
                    # We have both SQL and index records - compare them
                    sql_item_id = int(sql_record["ItemID"])
                    index_item_id = int(index_record["ItemID"])

                    sql_desc_id = int(sql_record["DescriptionID"])
                    index_desc_id = int(index_record["DescriptionID"])

                    # Set current_item_id if not set
                    if current_item_id is None:
                        current_item_id = min(sql_item_id, index_item_id)

                    # Process records based on ItemID and DescriptionID comparison
                    if sql_item_id < index_item_id:
                        # SQL record's ItemID is lower - create it
                        self._queue_create_record(sql_record)
                        sql_record = None
                        current_item_id = sql_item_id

                    elif sql_item_id > index_item_id:
                        # Index record's ItemID is lower - delete it
                        self._queue_delete_record(index_record)
                        index_record = None
                        current_item_id = index_item_id

                    else:
                        # ItemIDs match, now compare DescriptionIDs
                        if sql_desc_id < index_desc_id:
                            # SQL record's DescriptionID is lower - create it
                            self._queue_create_record(sql_record)
                            sql_record = None

                        elif sql_desc_id > index_desc_id:
                            # Index record's DescriptionID is lower - delete it
                            self._queue_delete_record(index_record)
                            index_record = None

                        else:
                            # Both ItemID and DescriptionID match - check if update needed
                            self._compare_and_update_record(sql_record, index_record)
                            sql_record = None
                            index_record = None

                # Update processed count
                self.records_processed += 1

                # Check for termination before potentially blocking operations
                if self.terminate_event.is_set():
                    break

            if not self.terminate_event.is_set():
                logger.info(f"Data processor completed. Processed {self.records_processed} records.")
                logger.info(f"Total SQL records: {sql_records_processed}, Total index records: {index_records_processed}")
                logger.info(
                    f"Records to create: {self.records_created}, update: {self.records_updated}, delete: {self.records_deleted}"
                )

        except Exception as e:
            logger.error(f"Error in data processor: {str(e)}")
            logger.exception(e)  # Log the full stack trace
        finally:
            # Signal that all processing is complete
            self.processing_complete.set()

    def _compare_and_update_record(self, sql_record, index_record):
        """Compare SQL and index records and queue updates if needed"""
        needs_embedding = False
        needs_update = False

        # Convert to string format for comparison if needed
        sql_item_modified = pd.to_datetime(sql_record["ItemLastModified"], utc=True).round("ms")
        index_item_modified = pd.to_datetime(index_record["ItemLastModified"], utc=True).round("ms")

        sql_desc_modified = pd.to_datetime(sql_record["DescLastModified"], utc=True).round("ms")
        index_desc_modified = pd.to_datetime(index_record["DescLastModified"], utc=True).round("ms")

        # Create document for potential update
        doc = {"DescriptionID": str(sql_record["DescriptionID"]), "@search.action": "merge"}

        # Check if item-level data needs updating
        if sql_item_modified > index_item_modified:
            doc.update(
                {
                    "ItemID": int(sql_record["ItemID"]),
                    "MfrPartNum": str(sql_record["MfrPartNum"]) if sql_record["MfrPartNum"] is not None else None,
                    "MfrPartNumExact": (
                        str(remove_separators(sql_record["MfrPartNum"])) if sql_record["MfrPartNum"] is not None else None
                    ),
                    "MfrPartNumPrefix": (
                        str(remove_separators(sql_record["MfrPartNum"])) if sql_record["MfrPartNum"] is not None else None
                    ),
                    "MfrName": str(sql_record["MfrName"]) if sql_record["MfrName"] is not None else None,
                    "UPC": str(sql_record["UPC"]) if sql_record["UPC"] is not None else None,
                    "UNSPSC": str(sql_record["UNSPSC"]) if sql_record["UNSPSC"] is not None else None,
                    "AKPartNum": str(sql_record["AKPartNum"]) if sql_record["AKPartNum"] is not None else None,
                    "ItemSourceName": str(sql_record["ItemSourceName"]) if sql_record["ItemSourceName"] is not None else None,
                    "ItemLastModified": self._to_azure_datetime(sql_record["ItemLastModified"]),
                }
            )
            needs_update = True

        # Check if description-level data needs updating
        if sql_desc_modified > index_desc_modified:
            doc.update(
                {
                    "ItemDescription": str(sql_record["ItemDescription"]) if sql_record["ItemDescription"] is not None else None,
                    "DescSourceName": str(sql_record["DescSourceName"]) if sql_record["DescSourceName"] is not None else None,
                    "DescLastModified": self._to_azure_datetime(sql_record["DescLastModified"]),
                }
            )
            needs_update = True
            needs_embedding = True

        # Queue for appropriate processing - will block if queues are full
        if needs_update and not self.terminate_event.is_set():
            if needs_embedding:
                self.embedding_queue.put(doc)
            else:
                self.saving_queue.put(doc)

    def _queue_create_record(self, sql_record):
        """Prepare a record for creation in the index"""
        # Create a new document with properly converted values
        doc = {
            "DescriptionID": str(sql_record["DescriptionID"]),
            "ItemID": int(sql_record["ItemID"]),
            "MfrPartNum": str(sql_record["MfrPartNum"]) if sql_record["MfrPartNum"] is not None else None,
            "MfrPartNumExact": str(remove_separators(sql_record["MfrPartNum"])) if sql_record["MfrPartNum"] is not None else None,
            "MfrPartNumPrefix": (
                str(remove_separators(sql_record["MfrPartNum"])) if sql_record["MfrPartNum"] is not None else None
            ),
            "MfrName": str(sql_record["MfrName"]) if sql_record["MfrName"] is not None else None,
            "UPC": str(sql_record["UPC"]) if sql_record["UPC"] is not None else None,
            "UNSPSC": str(sql_record["UNSPSC"]) if sql_record["UNSPSC"] is not None else None,
            "AKPartNum": str(sql_record["AKPartNum"]) if sql_record["AKPartNum"] is not None else None,
            "ItemSourceName": str(sql_record["ItemSourceName"]) if sql_record["ItemSourceName"] is not None else None,
            "ItemDescription": str(sql_record["ItemDescription"]) if sql_record["ItemDescription"] is not None else None,
            "DescSourceName": str(sql_record["DescSourceName"]) if sql_record["DescSourceName"] is not None else None,
            "ItemLastModified": self._to_azure_datetime(sql_record["ItemLastModified"]),
            "DescLastModified": self._to_azure_datetime(sql_record["DescLastModified"]),
        }

        # Queue for embedding calculation - will block if queue is full
        if not self.terminate_event.is_set():
            self.embedding_queue.put(doc)
            # self.records_created += 1

    def _to_azure_datetime(self, value):
        """
        Converts a datetime value to an ISO 8601 string with timezone info (UTC)
        for compatibility with Azure Search Edm.DateTimeOffset.

        Accepts:
            - datetime (naive or aware)
            - string (returns as-is)
            - None (returns None)

        Returns:
            ISO 8601 string with 'Z' suffix for UTC or the original string/None.
        """
        if value is None:
            return None

        if isinstance(value, datetime):
            # Make timezone-aware if it's naive
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            else:
                value = value.astimezone(timezone.utc)
            return value.isoformat().replace("+00:00", "Z")

        # If it's already a string, return as-is
        return value

    def _queue_delete_record(self, index_record):
        """Prepare a record for deletion from the index"""
        # Create a deletion document
        doc = {"DescriptionID": index_record["DescriptionID"], "@search.action": "delete"}

        # Queue for saving (deletion) - will block if queue is full
        if not self.terminate_event.is_set():
            self.saving_queue.put(doc)
            # self.records_deleted += 1

    def embedding_calculator(self):
        """Thread for calculating embeddings for Item Descriptions"""
        logger.info("Starting embedding calculator thread...")

        try:
            batch = []

            while not self.terminate_event.is_set() and not (self.processing_complete.is_set() and self.embedding_queue.empty()):
                try:
                    # Get a document from the queue
                    doc = self.embedding_queue.get(timeout=1)
                    logger.debug(f"Got document for embedding: {doc['DescriptionID']}")
                    batch.append(doc)

                    if len(batch) >= self.batch_size or (self.processing_complete.is_set() and self.embedding_queue.empty()):
                        if batch and not self.terminate_event.is_set():
                            logger.debug(f"Processing batch of {len(batch)} documents for embedding")
                            self._process_embedding_batch(batch)
                            batch = []
                except queue.Empty:
                    # If queue is empty but not yet complete, wait for more data
                    if not self.processing_complete.is_set() and not self.terminate_event.is_set():
                        time.sleep(0.1)

            # Process any remaining documents if not terminating
            if batch and not self.terminate_event.is_set():
                logger.debug(f"Processing final batch of {len(batch)} documents for embedding")
                self._process_embedding_batch(batch)

            logger.info(f"Embedding calculator completed. Generated embeddings for {self.embeddings_calculated} documents.")

        except Exception as e:
            logger.error(f"Error in embedding calculator: {str(e)}")
            logger.exception(e)  # Log the full stack trace
        finally:
            # Signal that all embeddings are complete
            self.embedding_complete.set()

    def _process_embedding_batch(self, batch):
        """Process a batch of documents for embedding calculation"""
        if not batch or self.terminate_event.is_set():
            return

        try:
            # Extract text for embedding
            texts = [doc.get("ItemDescription", "") for doc in batch]
            logger.debug(f"Calculating embeddings for {len(texts)} documents")

            # Check termination state before long-running operation
            if self.terminate_event.is_set():
                logger.info("Termination event detected before embedding calculation, skipping batch")
                return

            # Calculate embeddings
            embeddings = self.llm.get_embeddings(texts)

            # Check termination again after potentially long-running operation
            if self.terminate_event.is_set():
                logger.info("Termination event detected after embedding calculation, not processing results")
                return

            if len(embeddings) != len(texts):
                logger.error(f"Got {len(embeddings)} embeddings for {len(texts)} texts")

            # Add embeddings back to documents
            for i, doc in enumerate(batch):
                if i < len(embeddings) and not self.terminate_event.is_set():
                    doc["ItemDescription_vector"] = embeddings[i]
                    # Put documents in saving queue immediately after adding embedding
                    # This will block if the queue is full
                    self.saving_queue.put(doc)
                    self.embeddings_calculated += 1
                elif not self.terminate_event.is_set():
                    logger.error(f"Missing embedding for document {i}")

            logger.debug(f"Generated embeddings for {len(batch)} documents.")

        except Exception as e:
            if not self.terminate_event.is_set():
                logger.error(f"Error calculating embeddings: {str(e)}")
                logger.exception(e)  # Log the full stack trace

    def data_saver(self):
        logger.info("Starting data saver thread...")

        try:
            batch = []
            docs_received = 0

            while not self.terminate_event.is_set() and not (
                self.processing_complete.is_set() and self.embedding_complete.is_set() and self.saving_queue.empty()
            ):
                try:
                    # Get a document from the queue with shorter timeout
                    doc = self.saving_queue.get(timeout=0.5)
                    docs_received += 1
                    logger.debug(f"Got document for saving: {doc.get('DescriptionID', 'unknown')}")
                    batch.append(doc)

                    # Process in batches or when queue is empty
                    if (
                        batch
                        and not self.terminate_event.is_set()
                        and (
                            len(batch) >= self.batch_size
                            or (
                                self.processing_complete.is_set()
                                and self.embedding_complete.is_set()
                                and self.saving_queue.empty()
                            )
                        )
                    ):
                        logger.debug(f"Saving batch of {len(batch)} documents")

                        self._save_batch(batch)
                        batch = []

                except queue.Empty:
                    # If queue is empty but not yet complete, wait for more data
                    if (
                        not (self.processing_complete.is_set() and self.embedding_complete.is_set())
                        and not self.terminate_event.is_set()
                    ):
                        time.sleep(0.1)

            # Process any remaining documents if not terminating
            if batch and not self.terminate_event.is_set():
                logger.debug(f"Saving final batch of {len(batch)} documents")
                # Call the modified _save_batch which now includes retry logic
                self._save_batch(batch)

            logger.info(f"Data saver completed processing loop. Received {docs_received} documents.")

        except Exception as e:
            # Catch potential exceptions from _save_batch if they trigger termination
            if not self.terminate_event.is_set():  # Avoid double logging if terminated within _save_batch
                logger.error(f"Unhandled error in data saver main loop: {str(e)}")
                logger.exception(e)  # Log full stack trace
                self._terminate_process()  # Ensure termination on unexpected errors here too
        finally:
            logger.info("Data saver thread finished.")

    def _is_quota_error(self, error: Exception) -> bool:
        """Checks if the exception is a known Azure Search quota error."""
        return isinstance(error, HttpResponseError) and "quota has been exceeded" in str(error)

    def _save_batch(self, batch: List[Dict[str, Any]]):
        """Save a batch of documents to the index with optimized serialization,
        quota retry logic, and accurate metric tracking based on provided info."""
        if not batch:
            return

        key_field = "DescriptionID"

        batch_id = id(batch)
        logger.debug(
            f"[DEBUG_RETRY] _save_batch called for batch ID {batch_id} with {len(batch)} documents. Key field: '{key_field}'. "
            "Initializing retries."
        )

        try:
            # --- Existing data preparation ---
            # This part remains the same - preparing delete_docs, regular_docs, all_docs
            delete_docs = [doc for doc in batch if doc.get("@search.action") == "delete"]
            regular_docs = [doc for doc in batch if doc.get("@search.action") != "delete"]

            for doc in regular_docs:
                for key in [k for k in doc if "vector" in k and doc[k] is not None]:
                    if hasattr(doc[key], "tolist"):
                        doc[key] = doc[key].tolist()
                    elif isinstance(doc[key], list):
                        doc[key] = [float(v) for v in doc[key]]
                for key in [k for k in doc if isinstance(doc[k], np.ndarray) or np.issubdtype(type(doc[k]), np.number)]:
                    if isinstance(doc[key], np.ndarray):
                        doc[key] = doc[key].tolist() if not pd.isna(doc[key]).all() else None
                    elif np.issubdtype(type(doc[key]), np.floating):
                        doc[key] = float(doc[key])
                    elif np.issubdtype(type(doc[key]), np.integer):
                        doc[key] = int(doc[key])
                for key in list(doc.keys()):
                    if isinstance(doc[key], (np.ndarray, list)):
                        if pd.isna(doc[key]).all():
                            doc[key] = None
                    elif pd.isna(doc[key]):
                        doc[key] = None

            all_docs = delete_docs + regular_docs
            # --- End of existing data preparation ---

            # --- Retry Logic ---
            retries = 0
            upload_result_wrapper = None  # Initialize result to handle potential early exit
            while True:
                try:
                    logger.debug(
                        f"[DEBUG_RETRY] Batch ID {batch_id}: Attempting upload, retry {retries + 1}/{self.max_quota_retries + 1} "
                        f"(0-indexed retries = {retries})"
                    )

                    # Call the provided upload_documents method
                    upload_result_wrapper = self.search_utils.upload_documents(all_docs)

                    logger.debug(f"[DEBUG_RETRY] Batch ID {batch_id}: Upload API call succeeded after {retries} retries.")
                    break  # Exit retry loop on success

                except HttpResponseError as http_error:
                    # Quota error handling remains the same
                    if self._is_quota_error(http_error):
                        retries += 1
                        if retries > self.max_quota_retries:
                            logger.error(
                                f"Azure Search quota error persisted after {self.max_quota_retries} retries for "
                                f"batch ID {batch_id}. Failing batch."
                            )
                            raise http_error
                        else:
                            delay_seconds = self.quota_retry_delay_minutes * 60
                            logger.warning(
                                f"Azure Search quota exceeded for batch ID {batch_id}. Waiting {self.quota_retry_delay_minutes} "
                                f"minutes before retry {retries}/{self.max_quota_retries}..."
                            )
                            sleep_start = time.time()
                            while time.time() - sleep_start < delay_seconds:
                                if self.terminate_event.is_set():
                                    logger.warning(
                                        f"Batch ID {batch_id}: Termination signal received during retry delay. Aborting retry."
                                    )
                                    raise http_error
                                time.sleep(1)
                    else:  # Non-quota error
                        logger.error(f"Batch ID {batch_id}: Non-quota HTTP error during document upload: {http_error}")
                        raise http_error
                # Other exceptions (non-HttpResponseError) during upload will also break the loop and be caught below.

            # --- Result processing and METRIC UPDATE (only runs if upload succeeds) ---

            # Check if the wrapper dictionary was returned and has the 'details' key
            if upload_result_wrapper is None or "details" not in upload_result_wrapper:
                logger.error(
                    f"Batch ID {batch_id}: Invalid result structure received from upload_documents. Expected dict with 'details'."
                    f" Got: {type(upload_result_wrapper)}. Cannot update metrics."
                )
                # This indicates a problem potentially in the search_utils.upload_documents wrapper itself or the SDK.
                # Terminate because we cannot reliably track progress.
                raise ValueError("Invalid result structure received from upload_documents wrapper.")

            # Extract the list of individual status objects from the 'details' key
            statuses = upload_result_wrapper["details"]

            # Use a set for efficient lookup of successful keys
            # Assumes each status object 'item' has boolean 'succeeded' and string 'key' attributes
            successful_keys = set()
            failed_items_details = []
            if isinstance(statuses, list):  # Ensure statuses is actually a list
                for item in statuses:
                    # Check attributes exist before accessing to be robust
                    if hasattr(item, "succeeded") and item.succeeded and hasattr(item, "key"):
                        successful_keys.add(item.key)
                    elif hasattr(item, "succeeded") and not item.succeeded:
                        # Collect details for logging failures
                        failed_key = getattr(item, "key", "Unknown Key")
                        status_code = getattr(item, "status_code", "N/A")
                        error_msg = getattr(item, "error_message", "Unknown Error")
                        failed_items_details.append({"key": failed_key, "statusCode": status_code, "errorMessage": error_msg})
            else:
                logger.error(
                    f"Batch ID {batch_id}: Expected 'details' in result to be a list, but got {type(statuses)}. "
                    "Cannot process results or update metrics."
                )
                raise TypeError("Invalid type for 'details' in upload_documents result.")

            # Log detailed failures if any
            if failed_items_details:
                logger.warning(
                    f"Batch ID {batch_id}: Failed to save {len(failed_items_details)} out of {len(all_docs)} documents "
                    "in the batch."
                )
                for item in failed_items_details[:10]:  # Log details for first few failures
                    logger.warning(
                        f"  - Failed Item Key: {item.get('key')}, Status Code: {item.get('statusCode')}, "
                        f"Error: {item.get('errorMessage')}"
                    )
                if len(failed_items_details) > 10:
                    logger.warning(f"  - ... (omitting details for {len(failed_items_details) - 10} more failures)")

            # --- Calculate batch-specific counts based on successful keys and original actions ---
            batch_created = 0
            batch_updated = 0
            batch_deleted = 0

            for doc in all_docs:
                doc_key = doc.get(key_field)  # Use the confirmed 'DescriptionID'
                if doc_key in successful_keys:
                    action = doc.get("@search.action")  # Check the action in the original doc

                    if action == "delete":
                        batch_deleted += 1
                    elif action == "merge" or action == "mergeOrUpload":
                        batch_updated += 1
                    elif action == "upload":  # Explicit upload
                        batch_created += 1
                    elif action is None:  # Action field omitted, default is upload
                        batch_created += 1
                    # else: Unknown action type specified - ignore for metrics or log warning?
                    # logger.warning(
                    #     f"Batch ID {batch_id}: Document Key {doc_key} succeeded with unknown action '{action}'. "
                    #     f"Not counted in metrics."
                    # )

            logger.debug(
                f"Batch ID {batch_id}: Metrics counts: Created={batch_created}, Updated={batch_updated}, Deleted={batch_deleted}"
            )

            # --- Thread-safe update of instance metrics ---
            if batch_created > 0 or batch_updated > 0 or batch_deleted > 0:
                with self.metrics_lock:
                    self.records_created += batch_created
                    self.records_updated += batch_updated
                    self.records_deleted += batch_deleted
                logger.debug(
                    f"Batch ID {batch_id}: Cumulative Metrics Updated: Created={self.records_created}, "
                    f"Updated={self.records_updated}, Deleted={self.records_deleted}"
                )
            else:
                # Log if the batch succeeded but no specific actions were counted (e.g., only failures in batch)
                if not failed_items_details and len(all_docs) > 0:
                    logger.debug(
                        f"Batch ID {batch_id}: Batch processed successfully but no specific create/update/delete actions were "
                        "counted (check document actions?)."
                    )
                elif len(all_docs) > 0:  # Only log this if there were docs to process
                    logger.debug(f"Batch ID {batch_id}: No successful operations were matched in this batch to update metrics.")

            # --- End Metric Update ---

        except Exception as e:
            # Outer exception handling remains largely the same
            logger.error(f"Error processing or saving batch ID {batch_id}: {str(e)}")

            # Avoid double stack trace logging for final quota error
            if not self._is_quota_error(e):
                logger.exception(e)

            # Always terminate on any exception escaping the main try block
            logger.error("Critical error: Document save exception or persistent quota issue. Terminating all processing.")
            self._terminate_process()

    def queue_monitor(self):
        """Thread for monitoring queue sizes and performance metrics"""
        logger.info("Starting queue monitor thread...")

        monitor_interval = 30  # seconds between reports
        check_interval = 1  # seconds between completion checks during sleep

        try:
            while not self.terminate_event.is_set():
                # Log queue sizes
                logger.info(
                    f"[QUEUE SIZES] SQL Data: {self.sql_data_queue.qsize()}; "
                    f"Index Data: {self.index_data_queue.qsize()}; "
                    f"Embedding: {self.embedding_queue.qsize()}; "
                    f"Saving: {self.saving_queue.qsize()}"
                )

                # Log processing stats
                logger.info(
                    f"[PROGRESS] Processed: {self.records_processed}, "
                    f"Created: {self.records_created}, "
                    f"Updated: {self.records_updated}, "
                    f"Deleted: {self.records_deleted}, "
                    f"Embeddings: {self.embeddings_calculated}"
                )

                # Break sleep into smaller intervals and check completion status during each interval
                completed = False
                for _ in range(int(monitor_interval / check_interval)):
                    # Check if all processing is complete
                    if (
                        self.sql_data_complete.is_set()
                        and self.index_data_complete.is_set()
                        and self.processing_complete.is_set()
                        and self.embedding_complete.is_set()
                        and self.saving_queue.empty()
                    ):
                        logger.info("All processing complete. Stopping queue monitor.")
                        completed = True
                        break

                    # Short sleep that can be interrupted
                    time.sleep(check_interval)

                    # Also check termination event during sleep
                    if self.terminate_event.is_set():
                        break

                if completed:
                    break

        except Exception as e:
            logger.error(f"Error in queue monitor: {str(e)}")

    def run(self) -> bool:
        """
        Run the indexer, starting all processing threads and logging final stats
        regardless of success or failure using a finally block.
        """
        logger.info("Starting indexer run...")
        start_time = time.time()
        # Determine final status; default to failure unless explicitly set to success.
        final_success_status = False
        run_exception = None  # To store exception info if caught

        try:
            # Initialize flags at the start of the run
            self.insufficient_sql_data = False
            self.sql_connection_error = False
            self.index_connection_error = False

            # Prepare index (create or rebuild)
            self.prepare_index()

            # Create threads
            threads = [
                threading.Thread(target=self.sql_data_reader, name="SqlDataReader"),
                threading.Thread(target=self.index_data_reader, name="IndexDataReader"),
                threading.Thread(target=self.data_processor, name="DataProcessor"),
                threading.Thread(target=self.embedding_calculator, name="EmbeddingCalculator"),
                threading.Thread(target=self.data_saver, name="DataSaver"),
                threading.Thread(target=self.queue_monitor, name="QueueMonitor"),
            ]

            # Start threads
            logger.info("Starting worker threads...")
            for thread in threads:
                thread.start()
                logger.debug(f"Started thread: {thread.name}")  # Optional: debug level

            # Wait for threads to complete
            logger.info("Waiting for all worker threads to complete...")
            for thread in threads:
                # Consider adding a timeout to join if threads might hang indefinitely
                thread.join()
                logger.debug(f"Thread completed: {thread.name}")  # Optional: debug level

            logger.info("All worker threads have completed or been signaled for termination.")

            # --- Determine Final Status based on flags set by threads ---
            # Checks are performed AFTER threads have finished.
            if self.insufficient_sql_data:
                logger.warning(
                    "Indexer run determined outcome: SQL query returned fewer records than the configured minimum of "
                    f"({self.min_sql_records})."
                )
                final_success_status = False
            elif self.sql_connection_error:
                logger.error("Indexer run determined outcome: Terminated due to SQL connection error.")
                final_success_status = False
            elif self.index_connection_error:
                logger.error("Indexer run determined outcome: Terminated due to Azure AI Search connection error.")
                final_success_status = False
            elif self.terminate_event.is_set():
                # This flag is the main indicator of critical errors within threads (like data_saver failure)
                logger.error("Indexer run determined outcome: Terminated due to critical error during processing.")
                final_success_status = False
            else:
                # If no flags indicate failure, the run was successful.
                logger.info("Indexer run determined outcome: Completed successfully.")
                final_success_status = True

        except Exception as e:
            # Catch unexpected errors during setup, thread management, etc.
            run_exception = e  # Store exception
            logger.error(f"Unhandled exception during indexer run orchestration: {e}", exc_info=True)
            # Ensure termination is signaled if an exception occurred here
            if hasattr(self, "_terminate_process") and callable(self._terminate_process):
                if not self.terminate_event.is_set():
                    self._terminate_process()  # Ensure other threads know about the failure
            else:  # Fallback if _terminate_process is missing
                if hasattr(self, "terminate_event"):
                    self.terminate_event.set()
            final_success_status = False  # Explicitly set failure status

        finally:
            # --- CALCULATE AND FORMAT TOTAL TIME ---
            total_duration_seconds = time.time() - start_time
            hours, rem = divmod(total_duration_seconds, 3600)
            minutes, seconds = divmod(rem, 60)
            duration_str = f"{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s"

            # --- Log Final Stats (Always Runs) ---
            log_level = logging.INFO if final_success_status else logging.WARNING
            outcome_message = "Completed Successfully" if final_success_status else "Terminated Due to Error(s)"
            if run_exception:
                outcome_message += f" (Exception: {run_exception})"

            logger.log(log_level, "--- Final Indexer Run Statistics ---")
            logger.log(log_level, f"Outcome: {outcome_message}")
            logger.log(log_level, f"Total processing time: {duration_str}")
            # Use getattr for safety in case attributes weren't initialized
            logger.log(log_level, f"Total records processed (from source): {getattr(self, 'records_processed', 'N/A')}")
            logger.log(log_level, f"Records created in index: {getattr(self, 'records_created', 'N/A')}")
            logger.log(log_level, f"Records updated in index: {getattr(self, 'records_updated', 'N/A')}")
            logger.log(log_level, f"Records deleted from index: {getattr(self, 'records_deleted', 'N/A')}")
            logger.log(log_level, f"Embeddings calculated: {getattr(self, 'embeddings_calculated', 'N/A')}")
            logger.log(log_level, "--- End Final Statistics ---")

        # The return value determined by the try/except block logic
        return final_success_status


def run_indexing(config_obj: Config, rebuild: bool = False, max_records: int = None) -> bool:
    """
    Core logic to create/update the Azure AI Search index.
    This function is called programmatically by the WebJob's run.py or by the main_cli function.
    Returns True for success, False for failure.
    """
    try:
        logger.info(f"Starting index update. Rebuild: {rebuild}, Max Records: {max_records}")
        indexer_instance = Indexer(config_obj, rebuild=rebuild, max_records=max_records)
        success = indexer_instance.run()
        logger.info(f"Index update finished. Success: {success}")
        return success
    except Exception:
        logger.exception("Unhandled exception during index update")  # logger.exception includes stack trace
        return False


def main_cli():
    """Main entry point for command-line script execution"""
    parser = argparse.ArgumentParser(description="Azure AI Search Indexer CLI")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the index")
    parser.add_argument("--max", type=int, help="Maximum number of records to process")
    args = parser.parse_args()

    logger.info("Indexer script started via CLI.")
    try:
        app_config = Config()  # Initialize configuration
        success = run_indexing(config_obj=app_config, rebuild=args.rebuild, max_records=args.max)

        logger.info(f"Indexer script finished via CLI. Success: {success}")
        return 0 if success else 1

    except Exception:
        logger.exception("Unhandled critical exception in CLI")
        return 1


if __name__ == "__main__":
    sys.exit(main_cli())
