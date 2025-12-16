# In tests/test_llm.py

import os
from unittest.mock import MagicMock

import pytest

from ai_stages import MatchContextType, ValidationResult

# --- Imports from your project ---
# Adjust these imports based on your exact project structure
from config import Config  # <-- Import the application's own Config class
from exceptions import InvoiceProcessingError
from llm import LLM, LLMClientType
from prompts import Prompts

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio


# --- Fixture for UNIT TESTS (remains for future use) ---
@pytest.fixture
def mock_config():
    """Provides a MagicMock configuration object for fast, isolated unit tests."""
    # This fixture is not used by the integration test but is good practice to keep.
    config = MagicMock()
    # Define mock attributes as needed for unit tests
    # ...
    return config


# --- CORRECTED Fixture for INTEGRATION TESTS ---
@pytest.fixture(scope="module")
def real_config():
    """
    Provides a REAL configuration object by initializing it the same way the main application does.
    """
    # This is the correct way: Leverage your application's own config loader.
    return Config()


# --- INTEGRATION TEST (Using the Correct Fixture) ---
integration = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Integration tests are skipped. Set RUN_INTEGRATION_TESTS=1 to run them.",
)


@integration
async def test_get_structured_response_integration_with_azure(real_config):  # <-- Now correctly uses the real_config
    """
    Tests the get_structured_response method by making a REAL API call to Azure.
    """
    # --- Arrange ---
    # The LLM instance is now created with a properly initialized Config object.
    llm_instance = LLM(config=real_config)

    test_prompt = Prompts.get_context_validator_prompt(
        invoice_text="DEWALT N460582 REPLACEMENT BLADE FOR DCE151B CABLE STRIPPER",
        mfr="DEWALT",
        pn="DCE151B",
        db_description="DEWALT 20V MAX XR CABLE STRIPPER KIT",
    )

    # --- Act ---
    try:
        result = await llm_instance.get_structured_response(
            prompt=test_prompt, output_model=ValidationResult, client_type=LLMClientType.CONTEXT_VALIDATOR
        )
    except InvoiceProcessingError as e:
        pytest.fail(f"Integration test failed with an InvoiceProcessingError: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected exception occurred during the integration test: {e}")

    # --- Assert ---
    print(f"\n--- Integration Test Result ---\n{result}\n-----------------------------")
    assert isinstance(result, ValidationResult)
    assert result.is_direct_match is False
    assert result.context_type == MatchContextType.REPLACEMENT_PART
    assert isinstance(result.reason, str) and len(result.reason) > 10
