"""
## Overview
This file defines custom exception classes.
"""


class InvoiceProcessingError(Exception):
    """
    Custom exception for errors encountered during invoice processing.

    Attributes:
        message (str): A descriptive error message explaining the issue.
        error_code (int): An optional error code representing the type of error.
        original_exception (Exception, optional): The original exception that was caught and wrapped.
        status_code (int, optional): HTTP status code from an API error, if relevant.
        params_at_error (dict, optional): Parameters that were being processed when the error occurred.
    """

    def __init__(
        self,
        message: str,
        error_code: int | None = None,  # Changed default to None for clarity
        original_exception: Exception = None,
        status_code: int | None = None,
        params_at_error: dict | None = None,
    ):  # New optional parameters

        super().__init__(message, error_code, original_exception, status_code, params_at_error)
        self.message = message  # Redundant if super().__init__(message) stores it in args[0], but explicit is fine.
        self.error_code = error_code
        self.original_exception = original_exception
        self.status_code = status_code
        self.params_at_error = params_at_error

    def __str__(self):
        # Customize string representation if needed, e.g., to include error_code or status_code
        parts = [self.message]
        if self.error_code is not None:
            parts.append(f"[InternalErrorCode: {self.error_code}]")
        if self.status_code is not None:
            parts.append(f"[HTTPStatus: {self.status_code}]")
        if self.original_exception is not None:
            parts.append(f"[OriginalException: {type(self.original_exception).__name__}]")
        return " ".join(parts)


class InvalidJsonResponseError(Exception):
    """
    Custom exception for errors related to JSON extraction or validation.

    Attributes:
        message (str): A descriptive error message explaining the issue.
        thread_id (Optional[int]): An optional thread ID for context, useful in multi-threaded applications.
        response (Optional[Any]): The problematic response object, if available, for debugging purposes.
    """

    # Added optional attributes for more context if needed later
    def __init__(self, message, thread_id=None, response=None):
        self.message = message
        self.thread_id = thread_id
        self.response = response  # Store the problematic response if useful
        super().__init__(message, thread_id, response)


class TruncatedJsonError(InvoiceProcessingError):
    """Indicates a JSON string appears to be incomplete or truncated."""


class MissingRequiredFieldError(InvoiceProcessingError):
    """Indicates a response is missing a mandatory field required by the schema."""
