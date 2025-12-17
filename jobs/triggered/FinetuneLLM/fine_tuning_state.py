import logging
from typing import Optional

from fine_tuning_utils import FineTuningUtils
from typing_extensions import Literal

from config import Config


class FinetuneJobState:

    def __init__(self, config: Config, logger: logging.Logger, ft_utils: FineTuningUtils, env: str):
        self.in_progress: bool = False
        self.job_id = None
        self.job_status: Optional[Literal["validating_files", "queued", "running", "succeeded", "failed", "cancelled"]] = None
        self.config = config
        self.logger = logger
        self.ft_utils = ft_utils
        self.env = env

    async def load(self):
        """
        Updates the in_progress, job_status, and job_id values.
        Must be run before retrieving the values or calling other methods.

        Returns:
            None
        """
        self.in_progress = self._is_finetune_job_in_progress()
        self.logger.info(f"Fine-tuning in progress: {self.in_progress}")
        if self.in_progress:
            self.job_id = self._get_finetune_job_id()
            self.job_status = self.ft_utils.get_fine_tune_job_status(self.job_id)
            self.logger.info(f"Fine-tuning job status: {self.job_status}")

    def is_successful(self) -> bool:
        return self.job_status == "succeeded"

    def is_unsuccessful(self) -> bool:
        """
        Determines whether the fine-tuning job has failed or been cancelled.

        Returns:
            bool: True if the job status is either 'failed' or 'cancelled', indicating
                  that the fine-tuning process did not complete successfully.
                  False otherwise.
        """
        return self.job_status in ["failed", "cancelled"]

    def is_running(self) -> bool:
        """
        Determines whether the fine-tuning job is still in progress.

        A job is considered still running if its status is not one of the terminal states:
        'succeeded', 'failed', or 'cancelled'.

        Returns:
            bool: True if the job is ongoing and has not yet completed or failed.
                  False if the job has reached a terminal state.
        """
        return self.job_status not in ["failed", "cancelled", "succeeded"]

    def _is_finetune_job_in_progress(self) -> bool:
        """
        Checks if there is an already finetuning job running.
        It will utilize the Azure config service to read the status.
        This will return True, when the previous fine-tuning job is still in progress.
        Use this method to check, before triggering the finetuning operation.

        Returns:
            is_finetuning_in_progress (bool): Returns True / False
        """
        from constants import AzureAppConfig, DataTypes, Environments

        app_config_label = Environments.DEV if self.env != Environments.PROD else Environments.PROD
        is_finetuning_in_progress = self.config.get_azure_app_config_value(
            AzureAppConfig.IS_LLM_FINE_TUNING_IN_PROGRESS, app_config_label, value_type=DataTypes.BOOLEAN, client_type="AI"
        )
        self.logger.debug(f"Value received for FT progress - {is_finetuning_in_progress}")
        return is_finetuning_in_progress

    def _get_finetune_job_id(self) -> Optional[str]:
        """
        Retrieves the current finetune job ID from Azure App Configuration.

        This function reads the job ID stored in Azure App Config using the appropriate
        environment label. Use this method to track or verify the current job ID.

        Returns:
            job_id (Optional[str]): The finetune job ID, or None if not found.
        """
        from constants import AzureAppConfig, DataTypes, Environments

        app_config_label = Environments.DEV if self.env != Environments.PROD else Environments.PROD
        job_id = self.config.get_azure_app_config_value(
            AzureAppConfig.FINETUNE_JOB_ID, app_config_label, value_type=DataTypes.STRING, client_type="AI"
        )

        self.logger.debug(f"Retrieved finetune job ID from Azure App Config: {job_id}")
        return job_id
