# In tests/test_ai_engine.py

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# --- Imports from your project ---
from ai_engine import AIEngine, StageResults  # <-- Add StageResults import
from ai_stages import StageDetails
from constants import Constants, Logs, StageNames

pytestmark = pytest.mark.asyncio


def create_mock_ivce_dtl():
    ivce_dtl = MagicMock()
    ivce_dtl.ITM_LDSC = "Test description string"
    ivce_dtl.fields = [Logs.MFR_NAME, Logs.PRT_NUM, Logs.UNSPSC]
    ivce_dtl.fields_to_consolidate = [("manufacturer_name",), ("part_number",), ("unspsc",)]
    return ivce_dtl


@pytest.fixture
def engine_setup():
    with patch("ai_engine.AIStages"), patch("ai_engine.StageUtils") as MockStageUtils:
        mock_stage_utils_instance = MockStageUtils.return_value
        mock_stage_utils_instance.consolidate_null_field.return_value = (
            {Logs.UNSPSC: Constants.UNDEFINED},
            StageDetails(stage_number=0, sub_stage_code="0.0", stage="dummy", sub_stage="dummy"),
        )

        engine = AIEngine(config=MagicMock(), sdp=MagicMock())
        engine.check_if_stage_allowed = MagicMock(return_value=True)
        # We replace the real StageUtils instance with our fully configured mock
        engine.stage_utils = mock_stage_utils_instance
        yield engine


async def test_scenario_1_validation_success_stops_pipeline(engine_setup):
    engine = engine_setup
    engine.stage_utils.check_if_next_stage_required.side_effect = [True, False]
    engine._consolidate_and_recheck_thresholds = AsyncMock(return_value=False)
    cm_details = StageDetails(
        stage_number=5,
        stage=StageNames.COMPLETE_MATCH,
        status=Constants.SUCCESS_lower,
        sub_stage_code="3.0",
        sub_stage="COMPLETE_MATCH",
    )
    validator_details = StageDetails(
        stage_number=8,
        stage=StageNames.CONTEXT_VALIDATOR,
        status=Constants.SUCCESS_lower,
        sub_stage_code="6.0",
        sub_stage="CONTEXT_VALIDATOR",
        is_validation_stage=True,
    )
    validator_details.is_final_success = True

    async def run_stage_side_effect(stage_config, ivce_dtl):
        if stage_config.stage_name == StageNames.COMPLETE_MATCH:
            return cm_details
        return StageDetails(
            stage_number=4,
            stage=StageNames.SEMANTIC_SEARCH,
            status=Constants.SUCCESS_lower,
            sub_stage_code="2.0",
            sub_stage="SEMANTIC_SEARCH",
        )

    engine.run_stage = AsyncMock(side_effect=run_stage_side_effect)
    engine.run_validation_stage = AsyncMock(return_value=validator_details)
    await engine.run_extraction_engine(ivce_dtl=create_mock_ivce_dtl(), lemmatizer=MagicMock())
    assert engine.run_stage.call_count == 2
    engine.run_validation_stage.assert_awaited_once()
    assert engine.stage_results.final_results[Logs.IVCE_LINE_STATUS] == "RC-AI"


async def test_scenario_2_validation_failure_continues_pipeline(engine_setup):
    engine = engine_setup
    engine.stage_utils.check_if_next_stage_required.side_effect = [True, False, False]
    engine._consolidate_and_recheck_thresholds = AsyncMock(return_value=False)
    cm_details = StageDetails(
        stage_number=5,
        stage=StageNames.COMPLETE_MATCH,
        status=Constants.SUCCESS_lower,
        sub_stage_code="3.0",
        sub_stage="COMPLETE_MATCH",
    )
    validator_details = StageDetails(
        stage_number=8,
        stage=StageNames.CONTEXT_VALIDATOR,
        status=Constants.SUCCESS_lower,
        sub_stage_code="6.0",
        sub_stage="CONTEXT_VALIDATOR",
        is_validation_stage=True,
    )
    validator_details.is_final_success = False
    default_details = StageDetails(
        stage_number=99, stage="Default", status=Constants.SUCCESS_lower, sub_stage_code="9.9", sub_stage="Default"
    )

    async def run_stage_side_effect(stage_config, ivce_dtl):
        if stage_config.stage_name == StageNames.COMPLETE_MATCH:
            return cm_details
        return default_details

    engine.run_stage = AsyncMock(side_effect=run_stage_side_effect)
    engine.run_validation_stage = AsyncMock(return_value=validator_details)
    await engine.run_extraction_engine(ivce_dtl=create_mock_ivce_dtl(), lemmatizer=MagicMock())
    assert engine.run_stage.call_count == 3
    engine.run_validation_stage.assert_awaited_once()
    assert engine.stage_results.final_results[Logs.IVCE_LINE_STATUS] == "RC-AI"


async def test_scenario_4_consolidation_triggers_stop(engine_setup):
    engine = engine_setup
    engine.stage_utils.check_if_next_stage_required.return_value = True
    engine._consolidate_and_recheck_thresholds = AsyncMock(side_effect=[False, True])
    default_details = StageDetails(
        stage_number=99, stage="Default", status=Constants.SUCCESS_lower, sub_stage_code="9.9", sub_stage="Default"
    )
    engine.run_stage = AsyncMock(return_value=default_details)
    await engine.run_extraction_engine(ivce_dtl=create_mock_ivce_dtl(), lemmatizer=MagicMock())
    assert engine.run_stage.call_count == 2
    assert engine._consolidate_and_recheck_thresholds.call_count == 2
    assert engine.stage_results.final_results[Logs.IVCE_LINE_STATUS] == "RC-AI"


async def test_scenario_5_fallthrough_triggers_best_result_selection(engine_setup):
    engine = engine_setup
    engine.stage_utils.check_if_next_stage_required.return_value = True
    engine._consolidate_and_recheck_thresholds = AsyncMock(return_value=False)
    default_details = StageDetails(
        stage_number=99, stage="Default", status=Constants.SUCCESS_lower, sub_stage_code="9.9", sub_stage="Default"
    )
    engine.run_stage = AsyncMock(return_value=default_details)
    await engine.run_extraction_engine(ivce_dtl=create_mock_ivce_dtl(), lemmatizer=MagicMock())
    assert engine.run_stage.call_count == 4
    assert engine._consolidate_and_recheck_thresholds.call_count == 4


async def test_scenario_6_best_result_ignores_invalidated_stage(engine_setup):
    engine = engine_setup
    # CORRECTED: Manually initialize stage_results before using it.
    engine.stage_results = StageResults()
    cm_details = StageDetails(
        stage_number=5,
        stage=StageNames.COMPLETE_MATCH,
        status=Constants.SUCCESS_lower,
        sub_stage_code="3.0",
        sub_stage="COMPLETE_MATCH",
    )
    cm_details.details = {"confidence_score": {"manufacturer_name": 100, "part_number": 100, "unspsc": 100}}
    validator_details = StageDetails(
        stage_number=8,
        stage=StageNames.CONTEXT_VALIDATOR,
        status=Constants.SUCCESS_lower,
        sub_stage_code="6.0",
        sub_stage="CONTEXT_VALIDATOR",
        is_validation_stage=True,
    )
    validator_details.is_final_success = False
    llm_details = StageDetails(
        stage_number=6,
        stage=StageNames.FINETUNED_LLM,
        status=Constants.SUCCESS_lower,
        sub_stage_code="4.0",
        sub_stage="FINETUNED_LLM",
    )
    llm_details.details = {"confidence_score": {"manufacturer_name": 70, "part_number": 70, "unspsc": 70}}
    engine.stage_results.add_stage_result(cm_details)
    engine.stage_results.add_stage_result(validator_details)
    engine.stage_results.add_stage_result(llm_details)
    await engine._find_best_result_after_fallthrough(ivce_dtl=create_mock_ivce_dtl())
    assert engine.stage_results.final_stage_key == 6


async def test_scenario_7_best_result_prefers_completeness_over_confidence(engine_setup):
    engine = engine_setup
    # CORRECTED: Manually initialize stage_results before using it.
    engine.stage_results = StageResults()
    stage_a_details = StageDetails(
        stage_number=5, stage="Stage_A", status=Constants.SUCCESS_lower, sub_stage_code="3.0", sub_stage="A"
    )
    stage_a_details.details = {"confidence_score": {"manufacturer_name": 100}}
    stage_b_details = StageDetails(
        stage_number=6, stage="Stage_B", status=Constants.SUCCESS_lower, sub_stage_code="4.0", sub_stage="B"
    )
    stage_b_details.details = {"confidence_score": {"manufacturer_name": 40, "part_number": 40, "unspsc": 40}}
    engine.stage_results.add_stage_result(stage_a_details)
    engine.stage_results.add_stage_result(stage_b_details)
    await engine._find_best_result_after_fallthrough(ivce_dtl=create_mock_ivce_dtl())
    assert engine.stage_results.final_stage_key == 6
