import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from jobs.triggered.FinetuneLLM.run import main


class TestMainFunction(unittest.IsolatedAsyncioTestCase):

    @patch("jobs.triggered.FinetuneLLM.run.context")
    @patch("jobs.triggered.FinetuneLLM.run.handle_ft_job_completion", new_callable=AsyncMock)
    @patch("jobs.triggered.FinetuneLLM.run.reset_training_data", new_callable=AsyncMock)
    @patch("jobs.triggered.FinetuneLLM.run.update_finetune_job_id")
    @patch("jobs.triggered.FinetuneLLM.run.update_finetune_job_progress")
    @patch("jobs.triggered.FinetuneLLM.run.sys.exit")
    async def test_main_successful_job(
        self, mock_exit, mock_update_progress, mock_update_job_id, mock_reset_data, mock_handle_completion, mock_context
    ):
        # Mock job state
        mock_state = MagicMock()
        mock_state.load = AsyncMock()
        mock_state.in_progress = True
        mock_state.job_id = "job-123"
        mock_state.is_successful.return_value = True
        mock_state.is_unsuccessful.return_value = False
        mock_state.is_running.return_value = False
        mock_context.ft_job_state = mock_state

        await main()

        mock_state.load.assert_awaited_once()
        mock_handle_completion.assert_awaited_once_with("job-123")
        mock_update_job_id.assert_called_once_with("")
        mock_update_progress.assert_called_once_with(False)
        mock_exit.assert_not_called()

    @patch("jobs.triggered.FinetuneLLM.run.context")
    @patch("jobs.triggered.FinetuneLLM.run.update_finetune_job_id")
    @patch("jobs.triggered.FinetuneLLM.run.update_finetune_job_progress")
    @patch("jobs.triggered.FinetuneLLM.run.sys.exit")
    async def test_main_starts_new_job(self, mock_exit, mock_update_progress, mock_update_job_id, mock_context):
        # Mock job state
        mock_state = MagicMock()
        mock_state.load = AsyncMock()
        mock_state.in_progress = False
        mock_context.ft_job_state = mock_state

        # Mock ft_utils.finetune
        mock_ft_utils = MagicMock()
        mock_ft_utils.finetune = AsyncMock(return_value="new-job-456")
        mock_context.ft_utils = mock_ft_utils

        await main()

        mock_state.load.assert_awaited_once()
        mock_ft_utils.finetune.assert_awaited_once()
        mock_update_job_id.assert_called_once_with("new-job-456")
        mock_update_progress.assert_called_once_with(True)
        mock_exit.assert_not_called()
