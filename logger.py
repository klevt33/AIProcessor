"""
## Overview
This module provides functionality for initializing, configuring, and managing a logger instance.
It includes support for dynamic request IDs using `contextvars` and custom log formatting.
The logger is designed to handle application-level logging while suppressing verbose logs from
external libraries such as Azure SDK and SQLAlchemy.
"""

import logging
import os
import re
import sys
from contextvars import ContextVar
from typing import Iterable, Optional

from opencensus.ext.azure.log_exporter import AzureLogHandler

from constants import Constants, LogLevels

# logger = None

# Default logger configuration
DEFAULT_LOGGER_NAME = "AI API"  # Name of the logger
DEFAULT_LOG_LEVEL = LogLevels.INFO  # Default logging level

# Context variables for dynamic IDs
request_id_var: ContextVar[str] = ContextVar("request_id", default=Constants.EMPTY_STRING)
job_id_var: ContextVar[str] = ContextVar("job_id", default=Constants.EMPTY_STRING)

# Default format strings
DEFAULT_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
API_FMT = "%(asctime)s - [Request ID: %(request_id)s] - %(levelname)s - %(message)s"
JOB_FMT = "%(asctime)s - [JobName: %(name)s ID: %(job_id)s] - %(levelname)s - %(message)s"

# # --- TEMPORARY DEBUGGING CONFIGURATION FOR LOCAL FILE LOGGING ---
# # Set to True to enable logging to a local file for debugging.
# # Set to False or remove this section and the corresponding code block
# # in initialize_logger() to disable.
# ENABLE_LOCAL_FILE_LOGGING_FOR_DEBUG = True
# DEBUG_LOG_FILE_NAME = "app_debug.log"  # Name of the local log file
# # --- END TEMPORARY DEBUGGING CONFIGURATION ---

# ---- Configure your skips here ----
SKIP_EXACT: set[str] = {"Polling Cosmos DB...", "There are NO pending jobs."}

SKIP_SUBSTR: tuple[str, ...] = ("Polling Cosmos DB", "NO pending jobs.")  # substring  # example

SKIP_REGEX: tuple[str, ...] = (r"\bno pending jobs\b",)


class SkipLogsFilter(logging.Filter):
    """
    Flexible filter:
      - exact/substring/regex matching
    Attach this filter only to the handlers you want to silence.
    Reject any record whose logger name == "DBPoller", or Reject any record whose message matches a skip list.
    """

    def __init__(
        self,
        name: str = "",
        skip_exact: Optional[Iterable[str]] = None,
        skip_substr: Optional[Iterable[str]] = None,
        skip_regex: Optional[Iterable[str]] = None,
    ):
        super().__init__(name)
        self.skip_exact = set(skip_exact) if skip_exact else set(SKIP_EXACT)
        self.skip_substr = tuple(skip_substr) if skip_substr else tuple(SKIP_SUBSTR)
        self.skip_regex = tuple(re.compile(p, re.IGNORECASE) for p in (skip_regex or SKIP_REGEX))

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        msg = record.getMessage() if isinstance(record.msg, str) else str(record.msg)

        # # 1) drop if the logger’s name is exactly "DBPoller"
        # if record.name == "DBPoller":
        #     return False

        # 1) exact
        if msg in self.skip_exact:
            return False

        # 2) substring
        for s in self.skip_substr:
            if s and s in msg:
                return False

        # 3) regex
        for rx in self.skip_regex:
            if rx.search(msg):
                return False

        return True


class DatadogLogsFilter(logging.Filter):
    """
    This is filter would add a tag to each record whether to index in datadog or not.
    """

    def __init__(self):
        self.skip_exact = set(SKIP_EXACT)

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        msg = record.getMessage() if isinstance(record.msg, str) else str(record.msg)

        if msg in self.skip_exact:
            record.dd_excluded = True
        else:
            record.dd_excluded = False

        return True


class FlushStreamHandler(logging.StreamHandler):
    """Custom stream handler that ensures logs are flushed immediately."""

    def emit(self, record):
        super().emit(record)
        self.flush()


class JobIdFormatter(logging.Formatter):
    """Formatter that injects job_id into log records."""

    def format(self, record):  # noqa
        record.job_id = job_id_var.get() or "-"
        return super().format(record)


class SimpleLoggerFactory:
    """Factory for creating loggers with a single fixed formatter."""

    def __init__(self, fmt: str, level: str = DEFAULT_LOG_LEVEL):
        self.formatter = logging.Formatter(fmt)
        self.level = level

    def _configure_external_loggers(self):
        # Suppress verbose Azure and SQLAlchemy logs
        logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
        logging.getLogger("azure").setLevel(logging.WARNING)

        for name in [
            "sqlalchemy",
            "sqlalchemy.engine",
            "sqlalchemy.engine.base.Engine",
            "sqlalchemy.pool",
            "sqlalchemy.dialects",
            "sqlalchemy.orm",
            "sqlalchemy.sql",
            "sqlalchemy.schema",
            "sqlalchemy.future",
            "sqlalchemy.ext",
            "sqlalchemy.util",
        ]:
            logging.getLogger(name).setLevel(logging.WARNING)

    def _attach_azure_handler(self, logger: logging.Logger, conn_str: str):
        azure_handler = AzureLogHandler(connection_string=conn_str)
        azure_handler.addFilter(SkipLogsFilter(skip_substr=(), skip_regex=()))
        azure_handler.setFormatter(self.formatter)
        logger.addHandler(azure_handler)
        logger.info("AzureLogHandler attached; sending logs to Application Insights.")

    def create(
        self, name: str, level: str | None = None, azure_conn_str: str | None = None, handler: logging.Handler | None = None
    ) -> logging.Logger:
        """Create a logger with the configured formatter."""
        self._configure_external_loggers()
        ln = level or self.level
        logger = logging.getLogger(name)
        logger.setLevel(ln)

        # Stream (stdout) handler
        strh = handler or FlushStreamHandler(sys.stdout)
        strh.addFilter(DatadogLogsFilter())
        strh.setFormatter(self.formatter)
        logger.addHandler(strh)

        # # --- TEMPORARY DEBUGGING: ADD LOCAL FILE LOGGING ---
        # if ENABLE_LOCAL_FILE_LOGGING_FOR_DEBUG:
        #     try:
        #         # Create a file handler for local disk logging
        #         # This handler will write logs to the file specified by DEBUG_LOG_FILE_NAME
        #         file_debug_handler = logging.FileHandler(DEBUG_LOG_FILE_NAME)

        #         # Use the same formatter as other handlers for consistency
        #         file_debug_handler.setFormatter(self.formatter)  # Use the factory's formatter instance
        #         logger.addHandler(file_debug_handler)  # Add the configured handler to the logger

        #         # Log a message indicating that file logging is active.
        #         # This message will go to all handlers, including the new file handler and stdout.
        #         logger.info(f"Local debug file logging is active. Logs are being saved to: {DEBUG_LOG_FILE_NAME}")
        #     except Exception as e:
        #         # If file handler creation fails, print an error to stderr.
        #         # This uses print() directly to avoid issues if the logger itself is misconfigured.
        #         print(f"ERROR: Failed to initialize local debug file logger for '{DEBUG_LOG_FILE_NAME}': {e}", file=sys.stderr)
        # # --- END TEMPORARY DEBUGGING: ADD LOCAL FILE LOGGING ---

        # Datadog JSON file handler (singleton) under /home/LogFiles
        # try:
        #     if os.getenv("DD_ENV", Environments.LOCAL) != Environments.LOCAL:
        #         try:
        #             ddh = get_datadog_file_handler()
        #             if not any(getattr(h, "_id", "") == "datadog_json_file_handler" for h in logger.handlers):
        #                 ddh.setLevel(logger.level)
        #                 # attach YOUR filter here (affects only Datadog shipping)
        #                 ddh.addFilter(SkipLogsFilter())
        #                 logger.addHandler(ddh)
        #             logger.info("Datadog file handler attached.")
        #         except Exception as e:
        #             logger.warning(f"Failed to attach Datadog file handler: {e}")

        #     else:
        #         logger.debug("No Datadog agent found; skipping DatadogLogHandler.")

        # except Exception as e:
        #     logger.warning(f"Failed to attach Datadog file handler: {e}")

        # Azure handler
        conn = azure_conn_str or os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        if conn:
            self._attach_azure_handler(logger, conn)
        else:
            logger.debug("No Azure connection string; skipping AzureLogHandler.")

        logger.propagate = False
        return logger


def get_api_logger(name: str = "API", level: str | None = None, azure_conn_str: str | None = None) -> logging.Logger:
    """Logger that includes request_id in each line."""
    factory = SimpleLoggerFactory(fmt=API_FMT, level=level or DEFAULT_LOG_LEVEL)
    return factory.create(name=name, level=level, azure_conn_str=azure_conn_str)


def get_job_logger(name: str = "JOB", level: str | None = None, azure_conn_str: str | None = None) -> logging.Logger:
    """Logger that includes job_id in each line."""
    factory = SimpleLoggerFactory(fmt=JOB_FMT, level=level or DEFAULT_LOG_LEVEL)
    factory.formatter = JobIdFormatter(JOB_FMT)
    return factory.create(name=name, level=level, azure_conn_str=azure_conn_str)


def get_default_logger(name: str = "AI API", level: str | None = None, azure_conn_str: str | None = None) -> logging.Logger:
    """Standard logger without IDs."""
    factory = SimpleLoggerFactory(fmt=DEFAULT_FMT, level=level or DEFAULT_LOG_LEVEL)
    return factory.create(name=name, level=level, azure_conn_str=azure_conn_str)


def set_log_level(logger, level):
    logger.setLevel(level)


APP_INSIGHTS_CONN_STRING = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
LOG_LEVEL = os.getenv("LOG_LEVEL")
if APP_INSIGHTS_CONN_STRING:
    if LOG_LEVEL:
        logger = get_default_logger(level=LOG_LEVEL.upper(), azure_conn_str=APP_INSIGHTS_CONN_STRING)
    else:
        logger = get_default_logger(azure_conn_str=APP_INSIGHTS_CONN_STRING)
else:
    logger = get_default_logger(level=LogLevels.DEBUG)

# Example usage
# logger.info("This is a info log")
# logger.debug("This is a debug log")
# logger.warning("This is a warning log")
# logger.error("This is an error log")
# logger.critical("This is an critical log")
