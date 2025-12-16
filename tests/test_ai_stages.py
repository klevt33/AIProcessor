from unittest.mock import MagicMock, patch

import pytest

from ai_stages import AIStages
from constants import Constants, DescriptionCategories, Logs


@pytest.fixture
def ai_stages():
    mock_config = MagicMock()
    mock_config.azure_clients.text_analytics_client = MagicMock()

    mock_category_utils = MagicMock()

    # Patch internal classes
    with patch("ai_stages.Agents"), patch("ai_stages.LLM"), patch("ai_stages.AzureSearchUtils"), patch(
        "ai_stages.ClassifierUtils"
    ):
        ai_stages = AIStages(config=mock_config, category_utils=mock_category_utils)

    return ai_stages


@pytest.mark.asyncio
async def test_fetch_classification_returns_bad_on_string_that_cleans_to_empty(ai_stages: AIStages):
    # 1. Create a fake ivce_dtl object with ITM_LDSC
    ivce_dtl = MagicMock()
    ivce_dtl.ITM_LDSC = "* \n*"

    # 2. Create a mock lemmatizer
    lemmatizer = MagicMock()
    lemmatizer.lemmatize.return_value = [""]

    stage_details = await ai_stages.fetch_classification(lemmatizer, ivce_dtl)
    assert stage_details.status == Constants.SUCCESS_lower
    assert stage_details.details[Logs.CATEGORY] == DescriptionCategories.BAD
    assert stage_details.details[Logs.CONFIDENCE] == 100
