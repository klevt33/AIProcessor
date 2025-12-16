# In tests/test_validator_stage.py

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# --- Imports from your project ---
from ai_stages import AIStages, MatchContextType, StageDetails, ValidationResult
from constants import Constants, Logs

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio


# --- FINAL CORRECTED FIXTURE ---
@pytest.fixture
def mock_ai_stages_instance():
    """
    Provides a properly mocked instance of the AIStages class for testing.
    This fixture patches all heavy dependencies (LLM, AzureSearchUtils, etc.)
    at the source where AIStages imports them, preventing their real constructors
    from running and causing validation errors during test setup.
    """
    # Use patch to replace all external dependencies initialized by AIStages
    with patch("ai_stages.LLM"), patch("ai_stages.AzureSearchUtils"), patch("ai_stages.ClassifierUtils"):

        # 1. Create mock objects for the AIStages constructor arguments
        mock_config = MagicMock()
        mock_category_utils = MagicMock()

        # 2. Instantiate the real AIStages class.
        #    Its __init__ will now use our mocked classes instead of the real ones.
        ai_stages_instance = AIStages(config=mock_config, category_utils=mock_category_utils)

        # The instance's attributes (e.g., ai_stages_instance.llms) are now MagicMock instances
        # created by the patchers. We can now yield the fully prepared instance.
        yield ai_stages_instance


# --- Fixtures for test data (unchanged) ---
@pytest.fixture
def base_ivce_dtl():
    ivce_dtl = MagicMock()
    ivce_dtl.ITM_LDSC = "DEWALT N460582 REPLACEMENT BLADE FOR DCE151B CABLE STRIPPER"
    return ivce_dtl


@pytest.fixture
def successful_previous_stage():
    details = StageDetails(stage_number=5, sub_stage_code="3.0", stage="COMPLETE_MATCH", sub_stage="COMPLETE_MATCH")
    details.details = {
        Logs.MFR_NAME: "DEWALT",
        Logs.PRT_NUM: "DCE151B",
        "matched_description": "DEWALT 20V MAX XR CABLE STRIPPER KIT",
    }
    return details


# --- Test Cases (Now targeting the corrected fixture) ---


async def test_validate_context_direct_match(mock_ai_stages_instance, base_ivce_dtl, successful_previous_stage):
    # Arrange
    expected_result = ValidationResult(is_direct_match=True, context_type=MatchContextType.DIRECT_MATCH, reason="...")
    # Configure the get_structured_response method on the mocked llms attribute
    mock_ai_stages_instance.llms.get_structured_response = AsyncMock(return_value=expected_result)

    # Act
    stage_details, _ = await mock_ai_stages_instance.validate_context(
        sdp=None,
        ai_engine_cache=None,
        ivce_dtl=base_ivce_dtl,
        previous_stage_details=successful_previous_stage,
        stage_number=8,
        sub_stage_code="6.0",
    )

    # Assert
    assert stage_details.status == Constants.SUCCESS_lower
    assert stage_details.details["is_direct_match"] is True
    mock_ai_stages_instance.llms.get_structured_response.assert_awaited_once()


async def test_validate_context_replacement_part(mock_ai_stages_instance, base_ivce_dtl, successful_previous_stage):
    # Arrange
    expected_result = ValidationResult(is_direct_match=False, context_type=MatchContextType.REPLACEMENT_PART, reason="...")
    mock_ai_stages_instance.llms.get_structured_response = AsyncMock(return_value=expected_result)

    # Act
    stage_details, _ = await mock_ai_stages_instance.validate_context(
        sdp=None,
        ai_engine_cache=None,
        ivce_dtl=base_ivce_dtl,
        previous_stage_details=successful_previous_stage,
        stage_number=8,
        sub_stage_code="6.0",
    )

    # Assert
    assert stage_details.status == Constants.SUCCESS_lower
    assert stage_details.details["is_direct_match"] is False


async def test_validate_context_missing_input(mock_ai_stages_instance, base_ivce_dtl, successful_previous_stage):
    # Arrange
    del successful_previous_stage.details["matched_description"]
    mock_llm = mock_ai_stages_instance.llms  # Get a reference to the mock

    # Act
    stage_details, _ = await mock_ai_stages_instance.validate_context(
        sdp=None,
        ai_engine_cache=None,
        ivce_dtl=base_ivce_dtl,
        previous_stage_details=successful_previous_stage,
        stage_number=8,
        sub_stage_code="6.0",
    )

    # Assert
    assert stage_details.status == Constants.ERROR_lower
    assert "Missing required inputs" in stage_details.details[Constants.MESSAGE]
    mock_llm.get_structured_response.assert_not_called()


async def test_validate_context_llm_api_error(mock_ai_stages_instance, base_ivce_dtl, successful_previous_stage):
    # Arrange
    mock_ai_stages_instance.llms.get_structured_response = AsyncMock(side_effect=Exception("Simulated API Error"))

    # Act
    stage_details, _ = await mock_ai_stages_instance.validate_context(
        sdp=None,
        ai_engine_cache=None,
        ivce_dtl=base_ivce_dtl,
        previous_stage_details=successful_previous_stage,
        stage_number=8,
        sub_stage_code="6.0",
    )

    # Assert
    assert stage_details.status == Constants.ERROR_lower
    assert "Unexpected error" in stage_details.details[Constants.MESSAGE]
