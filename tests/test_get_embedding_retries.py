# test_llm.py
import json
import logging
import sys
import unittest
from unittest.mock import MagicMock

# --- Global variable for logger name ---
LLM_LOGGER_NAME = "AI API"

# --- Global tenacity.wait_fixed patching ---
import tenacity

original_tenacity_wait_fixed = tenacity.wait_fixed

mock_wait_callable_tier1 = MagicMock(return_value=0, name="mock_wait_callable_tier1")
mock_wait_callable_tier2 = MagicMock(return_value=0, name="mock_wait_callable_tier2")

DECORATOR_TRANSIENT_WAIT_SECONDS = 10
DECORATOR_OTHER_WAIT_SECONDS = 2
DECORATOR_TRANSIENT_MAX_ATTEMPTS = 10
DECORATOR_OTHER_MAX_ATTEMPTS = 5


def mock_wait_fixed_side_effect_global(wait_seconds_arg):
    if wait_seconds_arg == DECORATOR_TRANSIENT_WAIT_SECONDS:
        return mock_wait_callable_tier1
    elif wait_seconds_arg == DECORATOR_OTHER_WAIT_SECONDS:
        return mock_wait_callable_tier2
    else:
        return MagicMock(return_value=0)


tenacity.wait_fixed = MagicMock(side_effect=mock_wait_fixed_side_effect_global, name="globally_patched_tenacity_wait_fixed")
# --- End of tenacity.wait_fixed global patching ---

from openai import APIConnectionError, RateLimitError, RateLimitError as OpenAIRateLimitError

# --- Import SUT and dependencies ---
from llm import InvoiceProcessingError, retry_with_tiered_strategy

globally_patched_wait_fixed_mock = tenacity.wait_fixed
mock_target_function = MagicMock(name="mock_target_function")


@retry_with_tiered_strategy
def decorated_test_function(*args, **kwargs):
    return mock_target_function(*args, **kwargs)


# --- Helper functions (unchanged) ---
def create_mock_openai_response(status_code=500, headers=None, content_bytes=b"{}"):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.headers = headers if headers is not None else {"x-request-id": "mock-req-id"}
    mock_resp.content = content_bytes

    def json_func():
        try:
            return json.loads(content_bytes.decode())
        except Exception:
            return {"error": "mock error body"}

    mock_resp.json = json_func
    mock_resp.text = content_bytes.decode(errors="replace")
    mock_resp.request = MagicMock()
    mock_resp.url = "http://mock.test/api"
    return mock_resp


def create_mock_api_request():
    mock_req = MagicMock()
    mock_req.method = "POST"
    mock_req.url = "http://mock.test/api"
    mock_req.headers = {}
    mock_req.content = b""
    return mock_req


class TestRetryDecorator(unittest.TestCase):
    _original_handlers_store = {}  # Use a different name to avoid conflict with logging.handlers
    _our_test_handler = None
    _original_propagation_store = {}
    _original_level_store = {}

    @classmethod
    def setUpClass(cls):
        logger_instance = logging.getLogger(LLM_LOGGER_NAME)

        # Store original state
        cls._original_handlers_store[LLM_LOGGER_NAME] = list(logger_instance.handlers)
        cls._original_propagation_store[LLM_LOGGER_NAME] = logger_instance.propagate
        cls._original_level_store[LLM_LOGGER_NAME] = logger_instance.level

        # Clear existing handlers from the target logger
        for handler in cls._original_handlers_store[LLM_LOGGER_NAME]:
            logger_instance.removeHandler(handler)

        # Add our specific test handler
        cls._our_test_handler = logging.StreamHandler(sys.stdout)
        cls._our_test_handler.setLevel(logging.INFO)  # Ensure our handler catches INFO
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - DECORATOR_LOG_OUTPUT: %(message)s")
        cls._our_test_handler.setFormatter(formatter)
        logger_instance.addHandler(cls._our_test_handler)

        logger_instance.setLevel(logging.INFO)  # Ensure logger itself processes INFO
        logger_instance.propagate = False  # Stop propagation

        print(
            f"DEBUG setUpClass: Logger '{LLM_LOGGER_NAME}' configured. Level: {logger_instance.getEffectiveLevel()}, "
            f"Handlers: {logger_instance.handlers}, Propagate: {logger_instance.propagate}"
        )

        cls.initial_wait_fixed_call_count = globally_patched_wait_fixed_mock.call_count

    @classmethod
    def tearDownClass(cls):
        tenacity.wait_fixed = original_tenacity_wait_fixed

        logger_instance = logging.getLogger(LLM_LOGGER_NAME)
        if cls._our_test_handler and cls._our_test_handler in logger_instance.handlers:
            logger_instance.removeHandler(cls._our_test_handler)

        if LLM_LOGGER_NAME in cls._original_handlers_store:
            for handler_to_restore in cls._original_handlers_store[LLM_LOGGER_NAME]:
                if handler_to_restore not in logger_instance.handlers:  # Avoid adding duplicates if logic error
                    logger_instance.addHandler(handler_to_restore)

        if LLM_LOGGER_NAME in cls._original_propagation_store:
            logger_instance.propagate = cls._original_propagation_store[LLM_LOGGER_NAME]
        if LLM_LOGGER_NAME in cls._original_level_store:
            logger_instance.setLevel(cls._original_level_store[LLM_LOGGER_NAME])
        print(f"DEBUG tearDownClass: Logger '{LLM_LOGGER_NAME}' restored.")

    def setUp(self):
        mock_target_function.reset_mock()
        mock_target_function.side_effect = None
        mock_target_function.return_value = MagicMock()
        mock_wait_callable_tier1.reset_mock()
        mock_wait_callable_tier1.return_value = 0
        mock_wait_callable_tier2.reset_mock()
        mock_wait_callable_tier2.return_value = 0

    # No @patch('llm.logger.info') on test methods anymore
    def test_tier1_transient_error_retries_and_fails(self):
        self.assertEqual(self.initial_wait_fixed_call_count, 4)

        # --- Add this debug block ---
        current_logger = logging.getLogger(LLM_LOGGER_NAME)
        print(f"\nDEBUG IN TEST ({self.id()}): Logger '{current_logger.name}' status:")
        print(f"  Level: {current_logger.level} ({logging.getLevelName(current_logger.level)})")
        print(
            f"  Effective Level: {current_logger.getEffectiveLevel()} "
            f"({logging.getLevelName(current_logger.getEffectiveLevel())})"
        )
        print(f"  Disabled: {current_logger.disabled}")
        print(f"  Propagate: {current_logger.propagate}")
        print(f"  Handlers count: {len(current_logger.handlers)}")
        for i, h_obj in enumerate(current_logger.handlers):  # Renamed h to h_obj
            print(f"    Handler {i}: {h_obj}")
            print(f"      Level: {h_obj.level} ({logging.getLevelName(h_obj.level)})")
            if hasattr(h_obj, "formatter") and h_obj.formatter:
                print(f"      Formatter: {h_obj.formatter._fmt}")
            if hasattr(h_obj, "filters") and h_obj.filters:
                print(f"      Filters: {h_obj.filters}")
        # --- End of debug block ---

        mock_response = create_mock_openai_response(status_code=429)
        error_message_content = "Rate limit for test_tier1_fails"
        transient_error_instance = OpenAIRateLimitError(
            message=error_message_content, response=mock_response, body={"detail": error_message_content}
        )
        mock_target_function.side_effect = [transient_error_instance] * DECORATOR_TRANSIENT_MAX_ATTEMPTS

        # We can still assert log calls if we patch logger.info for this specific call block
        # but for now, let's rely on console output.
        with self.assertRaises(RateLimitError):
            decorated_test_function("test_input")

        self.assertEqual(mock_target_function.call_count, DECORATOR_TRANSIENT_MAX_ATTEMPTS)
        if DECORATOR_TRANSIENT_MAX_ATTEMPTS > 0:
            self.assertEqual(mock_wait_callable_tier1.call_count, DECORATOR_TRANSIENT_MAX_ATTEMPTS)
            mock_wait_callable_tier2.assert_not_called()
        self.assertEqual(globally_patched_wait_fixed_mock.call_count, self.initial_wait_fixed_call_count)
        # We cannot assert mock_logger_info.call_count without patching it.

    def test_tier2_other_error_retries_and_fails(self):
        self.assertEqual(self.initial_wait_fixed_call_count, 4)
        error_message_content = "ValueError for test_tier2_fails"
        other_error = ValueError(error_message_content)
        mock_target_function.side_effect = [other_error] * DECORATOR_OTHER_MAX_ATTEMPTS
        with self.assertRaises(ValueError):
            decorated_test_function("test_input")
        self.assertEqual(mock_target_function.call_count, DECORATOR_OTHER_MAX_ATTEMPTS)
        if DECORATOR_OTHER_MAX_ATTEMPTS > 0:
            self.assertEqual(mock_wait_callable_tier2.call_count, DECORATOR_OTHER_MAX_ATTEMPTS)
            mock_wait_callable_tier1.assert_not_called()
        self.assertEqual(globally_patched_wait_fixed_mock.call_count, self.initial_wait_fixed_call_count)

    def test_invoice_processing_error_no_retry(self):
        self.assertEqual(self.initial_wait_fixed_call_count, 4)
        custom_error = InvoiceProcessingError("Custom processing error")
        mock_target_function.side_effect = custom_error
        with self.assertRaises(InvoiceProcessingError):
            decorated_test_function("test_input")
        self.assertEqual(mock_target_function.call_count, 1)
        mock_wait_callable_tier1.assert_not_called()
        mock_wait_callable_tier2.assert_not_called()
        self.assertEqual(globally_patched_wait_fixed_mock.call_count, self.initial_wait_fixed_call_count)

    def test_success_on_first_attempt(self):
        self.assertEqual(self.initial_wait_fixed_call_count, 4)
        expected_result = "success"
        mock_target_function.return_value = expected_result
        mock_target_function.side_effect = None
        result = decorated_test_function("test_input")
        self.assertEqual(result, expected_result)
        self.assertEqual(mock_target_function.call_count, 1)
        mock_wait_callable_tier1.assert_not_called()
        mock_wait_callable_tier2.assert_not_called()
        self.assertEqual(globally_patched_wait_fixed_mock.call_count, self.initial_wait_fixed_call_count)

    def test_tier1_succeeds_after_retries(self):
        self.assertEqual(self.initial_wait_fixed_call_count, 4)
        mock_request = create_mock_api_request()
        error_message_content = "APIConnectionError for tier1_succeeds"
        transient_error = APIConnectionError(message=error_message_content, request=mock_request)
        expected_result = "success_after_tier1_retry"
        mock_target_function.side_effect = [transient_error, transient_error, expected_result]
        result = decorated_test_function("test_input")
        self.assertEqual(result, expected_result)
        self.assertEqual(mock_target_function.call_count, 3)
        self.assertEqual(mock_wait_callable_tier1.call_count, 2)
        mock_wait_callable_tier2.assert_not_called()
        self.assertEqual(globally_patched_wait_fixed_mock.call_count, self.initial_wait_fixed_call_count)

    def test_tier2_succeeds_after_retries(self):
        self.assertEqual(self.initial_wait_fixed_call_count, 4)
        error_message_content = "TypeError for tier2_succeeds"
        other_error = TypeError(error_message_content)
        expected_result = "success_after_tier2_retry"
        mock_target_function.side_effect = [other_error, expected_result]
        result = decorated_test_function("test_input")
        self.assertEqual(result, expected_result)
        self.assertEqual(mock_target_function.call_count, 2)
        self.assertEqual(mock_wait_callable_tier2.call_count, 1)
        mock_wait_callable_tier1.assert_not_called()
        self.assertEqual(globally_patched_wait_fixed_mock.call_count, self.initial_wait_fixed_call_count)

    def test_transient_error_bypasses_tier2_and_hits__tier1(self):
        self.assertEqual(self.initial_wait_fixed_call_count, 4)
        mock_response = create_mock_openai_response(status_code=429)
        error_message_content = "RateLimit for bypass_tier2"
        transient_error_instance = OpenAIRateLimitError(message=error_message_content, response=mock_response, body=None)
        expected_result = "success_after_tier1_retry_bypassing_tier2"
        mock_target_function.side_effect = [transient_error_instance, expected_result]
        result = decorated_test_function("test_input")
        self.assertEqual(result, expected_result)
        self.assertEqual(mock_target_function.call_count, 2)
        self.assertEqual(mock_wait_callable_tier1.call_count, 1)
        mock_wait_callable_tier2.assert_not_called()
        self.assertEqual(globally_patched_wait_fixed_mock.call_count, self.initial_wait_fixed_call_count)


if __name__ == "__main__":
    unittest.main(verbosity=2)  # Added verbosity to see test names
