import json
import os
import re
import time
from datetime import datetime
from http import HTTPStatus
from io import BytesIO, IOBase, StringIO
from logging import Logger
from typing import Any, Optional, Union

import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobType
from data_loader import DataLoader
from openai import AzureOpenAI
from openai.types.fine_tuning import job_create_params
from typing_extensions import Literal

from config import Config
from constants import TrainingDataVersions


class FineTuningUtils:

    def __init__(self, config: Config, logger: Logger, data_loader: DataLoader):
        self.config = config
        self.logger = logger
        self.data_loader = data_loader
        self.deployment_name = None
        self.training_file_blob_name = None
        self.fine_tuning_job_id = None

        self.fine_tuning_client = AzureOpenAI(
            api_version=config.AOAI_FINETUNED_LLM_API_VERSION,
            azure_endpoint=config.AOAI_FINETUNED_LLM_API_BASE_URL,  # Use the base endpoint URL
            api_key=config.AOAI_FINETUNED_LLM_OPENAI_API_KEY,
        )

        self.azure_blob_storage_client = config.azure_clients.azure_blob_storage_client

        from constants import TrainingDataVersions

        self.TRAINING_DATA_VER_COL_NAME = "TRNG_DAT_VRSN_NUM"

        # The Training data version to be used for the rows for which currently fine-tuning is in progress.
        self.TRAINING_DATA_VER_IN_PROGRESS = TrainingDataVersions.FT_PROGRESS

        # polling interval to check the training file upload status
        self.POLL_INTERVAL_FILE_UPLOAD = 5.0  # 5 seconds

        # Timeout waiting for the training file upload to be completed.
        self.TIMEOUT_WAITING_FOR_FILE_UPLOAD = 30 * 60  # 30 minutes

        # polling interval to check for fine-tuning job status.
        self.POLL_INTERVAL_FINE_TUNE_JOB = 30 * 60  # 30 minutes

        # Timeout waiting for the fine-tuning job to be completed.
        self.TIMEOUT_WAITING_FOR_FINE_TUNE_JOB = 72 * 60 * 60  # 72 hours.

        # indicates whether training data rows are updated to new Training version.
        self.training_version_updated = False

        # indicates whether training data file is uploaded to AZ Storage
        self.training_file_uploaded = False

        self.training_file_id = None

    @staticmethod
    def format_gpt4o(record):
        """
        Convert to GPT-4o training format
        """

        def prepare_user_prompt(description):
            prompt_template = (
                "Format the response strictly as a JSON object. No additional text, explanations, or disclaimers - "
                "only return JSON in this structure:\n"
                "```json\n"
                "{\n"
                '   "ManufacturerName": "string",\n'
                '   "PartNumber": "string",\n'
                '   "UNSPSC": "string",\n'
                "}\n"
                "```\n"
                "If any attribute is unavailable, return an empty string for that attribute."
            )

            user_prompt = f"""
                Input Description:
                {description}
                {prompt_template}
            """
            return user_prompt.strip()

        def prepare_assistant_prompt(mfr_nm, mfr_pn, unspsc):
            assistant_prompt = f"""
                ```json
                {{
                    "ManufacturerName": "{mfr_nm}",
                    "PartNumber": "{mfr_pn}",
                    "UNSPSC": "{unspsc}"
                }}
                ```
            """
            return assistant_prompt.strip()

        json_line = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant in finding and structuring information about electrical parts.",
                },
                {"role": "user", "content": prepare_user_prompt(record["IVCE_PRDT_LDSC"])},
                {
                    "role": "assistant",
                    "content": prepare_assistant_prompt(record["MFR_NM"], record["MFR_PRT_NUM"], record["UNSPSC_CD"]),
                },
            ]
        }
        return json_line

    def create_training_file(self, data: Union[str, StringIO, BytesIO]) -> str:
        """
        Upload training data to Azure OpenAI for fine-tuning. This will be done by
        creating a file resource on Azure OpenAI. On successful file upload, it returns
        file id which needs to be used for creating finetuning job.

        The input data to be uploaded can be given as file or data buffer.

        Parameters:
            data (str or StringIO/BytesIO): Path to a file or in-memory buffer.

        Returns:
            str: The training file ID on successful upload.

        Raises:
            Exception: If uploading fails or data is invalid.
        """
        try:
            if isinstance(data, str):
                if os.path.exists(data):
                    self.logger.info(f"Using file {data} for fine-tuning")
                    with open(data, "rb") as f:
                        response = self.fine_tuning_client.files.create(file=(os.path.basename(data), f), purpose="fine-tune")
                else:
                    raise FileNotFoundError(f"Training file '{data}' does not exist.")
            elif hasattr(data, "getvalue") and data.getvalue().strip():
                self.logger.info("Using in-memory data for fine-tuning")
                response = self.fine_tuning_client.files.create(file=data, purpose="fine-tune")
            else:
                raise ValueError("Provided data is invalid or empty.")

            training_file_id = response.id
            self.logger.info(f"Created training file with ID: {training_file_id}")
            return training_file_id

        except Exception as e:
            self.logger.error(f"Failed to create training file: {e}")
            raise

    def wait_for_file_import(self, file_id: str) -> None:
        """
        Waits until the specified Azure OpenAI file is processed or fails/times out.
        For the given training file id, this method polls periodically the status of the file.
        Status 'processed' means success.

        Parameters:
            file_id (str): ID of the file to monitor.

        Raises:
            Exception: If the file fails to process or a timeout/other error occurs.
        """
        try:
            initial_status = self.fine_tuning_client.files.retrieve(file_id)
            self.logger.info(f"[wait_for_file_import] Initial file status for {file_id}: {initial_status.status}")

            final_status = self.fine_tuning_client.files.wait_for_processing(
                file_id, poll_interval=self.POLL_INTERVAL_FILE_UPLOAD, max_wait_seconds=self.TIMEOUT_WAITING_FOR_FILE_UPLOAD
            )

            if final_status.status != "processed":
                msg = f"[wait_for_file_import] File {file_id} failed to process. Final status: {final_status.status}"
                self.logger.error(msg)
                raise Exception(msg)

            self.logger.info(f"[wait_for_file_import] File {file_id} successfully processed and ready for fine-tuning.")

        except RuntimeError as e:
            msg = f"[wait_for_file_import] Timeout waiting for file {file_id} to be processed: {e}"
            self.logger.error(msg)
            raise

        except Exception as e:
            msg = f"[wait_for_file_import] Error while waiting for file {file_id} to be processed: {e}"
            self.logger.error(msg)
            raise

    def create_finetune_llm_job(
        self,
        training_file_id: str,
        base_llm_model_name: str,
        num_epochs: int = 1,
        seed: int = 42,
        batch_size: int = 32,
        learning_rate_multiplier: float = 0.2,
    ) -> str:
        """
        Initiates fine tuning of the given base LLM with the given training file.
        The training file needs to be the file id created as part of uploading the file
        to Open AI (create_training_file).
        Before initiating the fine tuning, make sure the training file is uploaded and processed.
        (that is create_training_file , wait_for_file_import are called and are successful)

        Parameters:
            training_file_id (str): File ID of the training dataset.
            base_llm_model_name (str): Name of the base model to fine-tune.
            num_epochs (int): Number of training epochs. Default is 1.
            seed (int): Seed for reproducibility. Default is 42.
            batch_size (int, optional): Batch size (if custom). Default is OpenAI default.
            learning_rate_multiplier (float, optional): Custom LR multiplier. Default is OpenAI default.

        Returns:
            fine_tuning_job_id (str): the fine tuning job id indicating the triggered job.

        Raises:
            Exception: If fine-tuning job creation fails.
        """
        # trigger fine tuning
        try:
            # Construct hyperparameters
            hyperparams: job_create_params.Hyperparameters = {
                "n_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate_multiplier": learning_rate_multiplier,
            }

            response = self.fine_tuning_client.fine_tuning.jobs.create(
                training_file=training_file_id, model=base_llm_model_name, hyperparameters=hyperparams, seed=seed
            )

            job_id = response.id
            self.logger.info(f"Triggered fine-tuning job successfully: {job_id}")
            return job_id

        except Exception as e:
            self.logger.error(f"Failed to trigger fine-tuning job: {e}")
            raise

    def get_fine_tune_job_status(
        self, fine_tuning_job_id: str
    ) -> Literal["validating_files", "queued", "running", "succeeded", "failed", "cancelled"]:
        """
        Retrieves the current status of a fine-tuning job using its unique identifier.

        Args:
            fine_tuning_job_id (str): The unique identifier of the fine-tuning job
                whose status is to be retrieved.

        Returns:
            Literal["validating_files", "queued", "running", "succeeded", "failed", "cancelled"]:
                The current status of the fine-tuning job. Possible values include:
                - "validating_files": The job is validating input files.
                - "queued": The job is waiting in the queue.
                - "running": The job is currently in progress.
                - "succeeded": The job completed successfully.
                - "failed": The job encountered an error and did not complete.
                - "cancelled": The job was cancelled before completion.
        """
        response = self.fine_tuning_client.fine_tuning.jobs.retrieve(fine_tuning_job_id)
        return response.status

    def get_fine_tune_model_name(self, fine_tuning_job_id: str):
        """
        Retrieves the name of the fine-tuned model associated with a given fine-tuning job ID.

        Args:
            fine_tuning_job_id (str): The unique identifier of the fine-tuning job
                for which the fine-tuned model name is to be retrieved.

        Returns:
            str: The name or ID of the fine-tuned model if the job has completed successfully.
                Returns `None` if the model has not yet been created or the job failed.
        """
        response = self.fine_tuning_client.fine_tuning.jobs.retrieve(fine_tuning_job_id)
        return response.fine_tuned_model

    def deploy_fine_tuned_model(
        self,
        fine_tuned_model_name: str,
        deploy_model_name_suffix: str,
        resource_group: str,
        resource_name: str,
        azure_api_version: str,
        deployment_version: str | None = None,
        sku_name: str = "Standard",
        retries: int = 1,
        tpm_capacity: int = 20,
        wait_for_completion: bool = True,
        poll_interval: float = 90.0,
        max_wait_seconds: int = 3600,
    ) -> str:
        """
        Deploys the fine-tuned model to Azure and optionally waits until deployment completes.

        Parameters:
            fine_tuned_model_name (str): Name of the fine-tuned model.
            deploy_model_name_suffix (str): Suffix to append to deployment name.
            resource_group (str): The azure resource group of the given deployment.
            resource_name (str): Azure OpenAI resource name.
            azure_api_version (str): API version to use.
            deployment_version (str, optional): Custom model version, default to timestamp or "2025-2".
            sku_name (str): Azure SKU type (default: "standard").
            retries (int): Number of retries to try to deploy (default 1).
            tpm_capacity (int): SKU capacity - 1 = 1,000 TPM (default: 20).
            wait_for_completion (bool): Whether to poll until deployment completes.
            poll_interval (float): Time (in seconds) between status checks.
            max_wait_seconds (int): Max time to wait (in seconds).

        Returns:
            str: Deployment name.

        Raises:
            Exception: If deployment fails or times out.
        """

        self.logger.info("Deploying fine-tuned model...")

        # Current versioning : <finetuned-model>-<res_name>-<date>-<deploy_model_name_suffix>
        def get_deployment_name(model_name: str, res_name: str = "spend-report-finetuned") -> str:
            # ignore the FT job id from the fine-tuned model name.
            temp = model_name.split(".")
            if len(temp) < 2:
                dep_name_prefix = model_name
            else:
                dep_name_prefix = temp[0]

            created_date = time.strftime("%m-%d-%y")
            return f"{dep_name_prefix}-{res_name}-{created_date}-{deploy_model_name_suffix}"

        retry_count = 0

        self.deployment_name = get_deployment_name(fine_tuned_model_name)

        # Get Azure AD token
        try:
            credential = DefaultAzureCredential()
            token = credential.get_token("https://management.azure.com/.default").token
        except Exception:
            self.logger.error("[deploy_fine_tuned_model] Failed to acquire Azure AD token.")
            raise

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        payload = {
            "sku": {"name": sku_name, "capacity": tpm_capacity},
            "properties": {"model": {"name": fine_tuned_model_name, "format": "OpenAI", "version": deployment_version or "1"}},
        }

        deployment_url = (
            f"https://management.azure.com/subscriptions/{self.config.SUBSCRIPTION_ID}/resourceGroups/"
            f"{resource_group}/providers/Microsoft.CognitiveServices/accounts/{resource_name}/"
            f"deployments/{self.deployment_name}?api-version={azure_api_version}"
        )

        self.logger.info(f"[deploy_fine_tuned_model] Deploying model to URL: {deployment_url}")
        self.logger.info(f"[deploy_fine_tuned_model] Deployment payload: {json.dumps(payload)}")

        while retry_count <= retries:
            try:
                response = requests.put(deployment_url, headers=headers, data=json.dumps(payload))

                if not response.ok:
                    error = response.json().get("error", {})
                    self.logger.error(
                        f"[deploy_fine_tuned_model] Deployment failed: {error.get('code')} - {error.get('message')}"
                    )
                    self.logger.error(f"Model deployment failed: {error.get('message', 'Unknown error')}")
                    retry_count += 1
                    continue
            except Exception:
                self.logger.error("[deploy_fine_tuned_model] Deployment failed.")
                retry_count += 1
                continue

            self.logger.info(
                f"[deploy_fine_tuned_model] Deployment request accepted. Model: {fine_tuned_model_name}, "
                f"Deployment: {self.deployment_name}"
            )

            # Wait for deployment to finish
            if wait_for_completion:
                status_url = deployment_url  # same URL used for GET
                self.logger.info("[deploy_fine_tuned_model] Waiting for deployment completion...")

                start_time = time.perf_counter()
                while time.perf_counter() - start_time < max_wait_seconds:
                    status_response = requests.get(status_url, headers=headers)
                    if not status_response.ok:
                        self.logger.warning(
                            f"[deploy_fine_tuned_model] Failed to fetch deployment status. Retrying in {poll_interval}s."
                        )
                        time.sleep(poll_interval)
                        continue

                    status_json = status_response.json()
                    state = status_json.get("properties", {}).get("provisioningState", "").lower()

                    if state in {"succeeded", "failed"}:
                        self.logger.info(f"[deploy_fine_tuned_model] Deployment status: {state}")
                        if state == "failed":
                            error_msg = status_json.get("properties", {}).get("statusMessage", "Unknown deployment error")
                            self.logger.error(f"[deploy_fine_tuned_model] Deployment error: {error_msg}")
                            retry_count += 1
                            continue
                        break
                    else:
                        self.logger.debug(
                            f"[deploy_fine_tuned_model] Deployment in progress (state: {state}). "
                            f"Re-checking status in {poll_interval}s..."
                        )
                        time.sleep(poll_interval)
                else:
                    self.logger.error(
                        f"Timed out waiting for deployment '{self.deployment_name}' to complete, Check status in AZ Console"
                    )
                    continue

                elapsed = time.perf_counter() - start_time
                self.logger.info(f"[deploy_fine_tuned_model] Deployment completed in {elapsed:.2f} seconds.")
            else:
                self.logger.info("[deploy_fine_tuned_model] Not waiting for deployment to complete, check status in AZ Console")

            return self.deployment_name
        raise Exception("Model deployment failed")

    def get_model_from_deployment(self, resource_group: str, resource_name: str, api_version: str, deployment_name: str) -> tuple:
        """
        Returns the fine tuned model name and the version for a given deployment name.

        Parameters:
            resource_group (str): The azure resource group of the given deployment.
            resource_name (str): The azure resource name of the given deployment.
            api_version (str): API version string for the API call.
            deployment_name (str): the deployment name for which model to be fetched.

        Returns:
            (model, version) (tuple): Tuple of the finetuned model name and version.
        """
        try:
            credential = DefaultAzureCredential()
            token = credential.get_token("https://management.azure.com/.default").token

            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

            deployment_url = (
                f"https://management.azure.com/subscriptions/{self.config.SUBSCRIPTION_ID}/resourceGroups/"
                f"{resource_group}/providers/Microsoft.CognitiveServices/accounts/{resource_name}"
                f"/deployments/{deployment_name}?api-version={api_version}"
            )
            self.logger.info(f"Using URL : {deployment_url} to get the model name")
            response = requests.get(deployment_url, headers=headers)
            if response.ok:
                deployment_details = response.json()
                finetuned_model_name = deployment_details["properties"]["model"]["name"]
                finetuned_model_version = deployment_details["properties"]["model"]["version"]
                self.logger.info(f"Current finetuned model : {finetuned_model_name}, model version : {finetuned_model_version}")
                return finetuned_model_name, finetuned_model_version
            else:
                raise Exception(f"Got error when querying model from deployment : {response.json()}")
        except Exception as e:
            self.logger.error(f"Got error when querying model from deployment: {str(e)}", exc_info=True)
            raise e

    def remove_finetuned_llm_deployment(
        self,
        resource_group: str,
        resource_name: str,
        subscription_id: str,
        azure_api_version: str,
        deployment_name: str,
        delete_only_dev: bool = True,
        wait_for_delete_complete: bool = True,
        poll_interval: float = 10.0,
        max_wait_seconds: int = 3 * 60,
    ):
        """
        Deletes the given Azure Open AI deployment from the Azure resource.

        Parameters:
            resource_group (str):            The azure Open AI resource group for this deployment.
            resource_name (str):             The azure open AI resource name for this deployment.
            subscription_id (str):           Azure account subscription ID.
            azure_api_version (str):         API version for the REST API.
            deployment_name (str):           The deployment to be deleted ( its name)
            delete_only_dev (bool):          If True only development deployment will be deleted.
            wait_for_delete_complete (bool): If true, method will wait and poll until
                                             deployment is deleted (timeout after max_wait_seconds seconds)
            poll_interval (int):             interval to pause between checking for
                                             the deployment deletion completion.
            max_wait_seconds (int):          Total time in seconds to wait until
                                             the deployment is completed.
        """
        self.logger.info(f"Removing deployment: {deployment_name}")
        if delete_only_dev:
            current_env = self.config.environment
            if "-prod" in deployment_name:
                self.logger.info(f"Not removing deployments in {current_env}")
                return

        # Get Azure AD token
        try:
            credential = DefaultAzureCredential()
            token = credential.get_token("https://management.azure.com/.default").token
        except Exception:
            self.logger.error("[deploy_fine_tuned_model] Failed to acquire Azure AD token.")
            raise

        headers = {"Authorization": f"Bearer {token}"}

        deployment_url = (
            f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/"
            f"{resource_group}/providers/Microsoft.CognitiveServices/accounts/{resource_name}/"
            f"deployments/{deployment_name}?api-version={azure_api_version}"
        )

        self.logger.info(f"Deleting deployment using URL: {deployment_url}")

        response = requests.delete(deployment_url, headers=headers)

        if not response.ok:
            error = response.json().get("error", {})
            self.logger.error(f"deployment {deployment_name} deletion failed: {error.get('code')} - {error.get('message')}")
            raise Exception(f"Model deployment failed: {error.get('message', 'Unknown error')}")

        if response.status_code == HTTPStatus.ACCEPTED or response.status_code == HTTPStatus.NO_CONTENT:
            # Wait for deployment deletion to complete
            if wait_for_delete_complete:
                status_url = deployment_url  # same URL used for GET
                self.logger.info(f"delete request success with response {response.status_code}, waiting for delete to complete")

                start_time = time.perf_counter()
                while time.perf_counter() - start_time < max_wait_seconds:
                    status_response = requests.get(status_url, headers=headers)
                    if not status_response.ok:
                        if status_response.status_code == HTTPStatus.NOT_FOUND:
                            # deployment no more exists, deletion completed.
                            elapsed = time.perf_counter() - start_time
                            self.logger.info(f"Deployment {deployment_name} deletion completed in {elapsed:.2f} seconds.")
                            break
                        else:
                            # error in waiting for delete, exit.
                            self.logger.error(f"Deployment {deployment_name} deletion resulted in {response.status_code}")
                            raise Exception(f"Model deployment failed: deletion resulted in {response.status_code}")
                    else:
                        # deployment still exists, check on next interval.
                        self.logger.info(f"Deployment {deployment_name} still exists. Re-checking in {poll_interval}s...")
                        time.sleep(poll_interval)
                else:
                    self.logger.info(
                        f"Timed out waiting for deleting deployment {deployment_name},"
                        " check in Azure AI Foundry console for status"
                    )
            else:
                self.logger.info(
                    f"delete request for deployment {deployment_name} triggered (not waiting for completion),"
                    f"response={response.status_code}, check in AZ console for status"
                )
        else:
            # response many not have error.
            self.logger.error(f"deployment {deployment_name} deletion resulted in {response.status_code}")
            raise Exception(f"Model deployment failed: deletion resulted in {response.status_code}")

    @staticmethod
    def get_all_deployments(resource_group: str, resource_name: str, subscription_id: str, azure_api_version: str):
        """
        Get all deployments in Azure CognitiveServices via REST API.

        Args:
            resource_group:
            resource_name:
            subscription_id:
            azure_api_version:

        Returns:
            All LLM deployments or None if error occurred.

        """
        credential = DefaultAzureCredential()
        token = credential.get_token("https://management.azure.com/.default").token

        headers = {"Authorization": f"Bearer {token}"}

        list_url = (
            f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/"
            f"{resource_group}/providers/Microsoft.CognitiveServices/accounts/{resource_name}/"
            f"deployments?api-version={azure_api_version}"
        )

        response = requests.get(list_url, headers=headers)

        if response.status_code == 200:
            deployments = response.json().get("value", [])
            return deployments
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    def get_all_deployment_names(
        self, resource_group: str, resource_name: str, subscription_id: str, azure_api_version: str
    ) -> list[str]:
        deployments = self.get_all_deployments(resource_group, resource_name, subscription_id, azure_api_version)
        return [deployment["name"] for deployment in deployments]

    @staticmethod
    def get_deployment_date(deployment_name: str) -> str | None:
        pattern = r"(\d+-\d+-\d+)-(dev|prod)$"
        match = re.search(pattern, deployment_name)
        if match:
            return match.group(1)  # First capture group: the date-like string
        return None

    @staticmethod
    def get_all_old_deployment_names(deployment_names: list[str], n: int = 2) -> list[str]:
        """
        Retrieve all the old deployment names to delete. We want to keep the n (1) newest deployments and get rid of
        everything else. We need to get rid of old deployments to stay within the 250k TPM quota.

        Args:
            deployment_names: deployment names from Azure CognitiveServices API
            n: number of deployments to keep (not old)

        Returns:
            All deployment names to remove

        """
        # Get list of unique deployment dates
        deployment_dates = list(
            {
                date
                for deployment_name in deployment_names
                if (date := FineTuningUtils.get_deployment_date(deployment_name)) is not None
            }
        )
        # Sort by date
        deployment_dates = sorted(deployment_dates, key=lambda d: datetime.strptime(d, "%m-%d-%y"))
        # Remove n oldest dates
        old_deployment_dates = deployment_dates[:-n]

        old_deployment_names = [
            deployment_name
            for deployment_name in deployment_names
            if FineTuningUtils.get_deployment_date(deployment_name) in old_deployment_dates
        ]

        return old_deployment_names

    def delete_training_file(self, training_file_id: str) -> None:
        """
        Deletes the training file that is uploaded and used for fine tuning LLM.

        Parameters:
            training_file_id (str): The training file id that is generated when uploading the file.

        Raises:
            Exception: If deletion fails.
        """
        try:
            response = self.fine_tuning_client.files.delete(file_id=training_file_id)

            if response.deleted:
                self.logger.info(f"[delete_training_file] Successfully deleted training file: {training_file_id}")
            else:
                self.logger.warning(f"[delete_training_file] File {training_file_id} was not deleted (no confirmation).")

        except Exception as e:
            self.logger.error(f"[delete_training_file] Error deleting file {training_file_id}: {e}")
            raise

    def cancel_fine_tuning_job(
        self,
        fine_tuning_job_id: Optional[str] = None,
        cancel_all: bool = False,
        cancelable_states=None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        """
        Cancels one or more fine-tuning jobs if they are in cancelable states.

        Parameters:
            fine_tuning_job_id (str, optional): Job ID to cancel. Required if cancel_all is False.
            cancel_all (bool): If True, attempts to cancel all jobs in cancelable states.
            cancelable_states (List[str]): List of statuses that allow cancellation.
            max_retries (int): Number of times to retry in case of transient failure.
            retry_delay (float): Delay in seconds between retries.

        Raises:
            Exception: If any error occurs while attempting cancellation.
        """
        if cancelable_states is None:
            cancelable_states = ["queued", "pending"]
        try:
            if cancel_all:
                jobs = self.fine_tuning_client.fine_tuning.jobs.list()
                jobs_to_cancel = [job for job in jobs if job.status in cancelable_states]
                if not jobs_to_cancel:
                    self.logger.info("[cancel_fine_tuning_job] No active jobs found in cancelable states.")
                    return
            else:
                if not fine_tuning_job_id:
                    raise ValueError("fine_tuning_job_id is required unless cancel_all=True")
                job = self.fine_tuning_client.fine_tuning.jobs.retrieve(fine_tuning_job_id)
                jobs_to_cancel = [job] if job.status in cancelable_states else []

                if not jobs_to_cancel:
                    self.logger.warning(
                        f"[cancel_fine_tuning_job] Job {fine_tuning_job_id} is in '{job.status}' state and cannot be cancelled."
                    )
                    return

            for job in jobs_to_cancel:
                attempt = 0
                while attempt < max_retries:
                    try:
                        response = self.fine_tuning_client.fine_tuning.jobs.cancel(job.id)
                        if response.status == "cancelled":
                            self.logger.info(f"[cancel_fine_tuning_job] Job {job.id} successfully cancelled.")
                        else:
                            self.logger.warning(
                                f"[cancel_fine_tuning_job] Job {job.id} cancel requested, "
                                f"but current status is '{response.status}'"
                            )
                        break
                    except Exception as ex:
                        attempt += 1
                        self.logger.warning(
                            f"[cancel_fine_tuning_job] Retry {attempt}/{max_retries} failed to cancel job {job.id}: {ex}"
                        )
                        if attempt >= max_retries:
                            self.logger.error(
                                f"[cancel_fine_tuning_job] Failed to cancel job {job.id} after {max_retries} attempts."
                            )
                        else:
                            time.sleep(retry_delay)

        except Exception as e:
            self.logger.error(f"[cancel_fine_tuning_job] Error cancelling fine-tuning job(s): {e}")
            raise

    def upload_file_to_blob_storage(
        self,
        container_name: str,
        blob_name: str,
        data: Union[str, IOBase, bytes],
        max_stream_length_size: int = 512 * 1024 * 1024,
        user_metadata: dict[str, str] | None = None,
        overwrite: bool = True,
    ) -> dict[str, Any] | None:
        """
        Uploads the given data to the Azure bolb storage container by creating
        as new blob object.
        For large files, if need to be uploaded in chunks, use 'max_block_chunk_size'
        to specify the chunk size.

        Parameters:
            container_name (str):            The blob storage container name
            blob_name (str):                 The blob name to be used to create,
                                             this will be usually the file name being used.
            data (Union[str, IOBase, bytes]): The data that needs to be stored, can be given
                                             as file name which exists locally or a data buffer.
            max_stream_length_size (int):    The maximum size you want azure to read from the
                                             given stream at a time to upload to Blob.
            user_metadata (dict[str, str]):  Use to provide store any metadata to the blob.
                                             Use this to store the training data version.
            overwrite (bool):                Will overwrite the blob with same name if True,
                                             if False, write will  throw error or creates a
                                             new version (if versioning is enabled for AZ account)

        Returns:
            blob_params dict[str, str]: Some parameters of the created blob object.
        """
        try:
            blob_client = self.azure_blob_storage_client.get_blob_client(container=container_name, blob=blob_name)
            blob_output_params: dict[str, Any]

            if isinstance(data, str):
                with open(data, "rb") as file:
                    blob_output_params = blob_client.upload_blob(
                        file,
                        blob_type=BlobType.BLOCKBLOB,
                        length=max_stream_length_size,
                        metadata=user_metadata,
                        overwrite=overwrite,
                    )

            elif isinstance(data, IOBase):  # BytesIO or file-like object
                blob_output_params = blob_client.upload_blob(
                    data, blob_type=BlobType.BLOCKBLOB, length=max_stream_length_size, metadata=user_metadata, overwrite=overwrite
                )

            elif isinstance(data, bytes):  # bytes directly
                blob_output_params = blob_client.upload_blob(
                    data, blob_type=BlobType.BLOCKBLOB, length=max_stream_length_size, metadata=user_metadata, overwrite=overwrite
                )
            else:
                raise ValueError("Unsupported data type for upload_blob")

            return blob_output_params
        except Exception as e:
            self.logger.info(f"Failed to upload data to blob {blob_name} on container {container_name} - {e}")
            return None

    def delete_file_from_blob_storage(self, container_name: str, blob_name: str, delete_snapshots: str = "include") -> None:
        """
        Deletes the given file from the Azure blob storage.

        Parameters:
            container_name: The blob storage container name.
            blob_name (str): The blob name to delete. ( The name given when creating)
            delete_snapshots (str): Flag to request to delete all snapshots along with the blob.
                Azure storage supports creating read-only snapshots of the blob at any given time.
                'include' : delete the blob along with all the snapshots.
                'only' : delete only the snapshots.
        """
        try:
            blob_client = self.azure_blob_storage_client.get_blob_client(container=container_name, blob=blob_name)

            blob_client.delete_blob(delete_snapshots=delete_snapshots)
        except Exception as e:
            self.logger.info(f"Failed to upload data to blob {blob_name} on container {container_name} - {e}")

    async def finetune(self) -> Optional[str]:
        """
        1)  Generating the required training data.
        2)  Uploading the training data file (or in-memory buffer) to the OPEN AI for fine tuning.
        3)  Wait for the file upload to complete.
        4)  Store the training data into azure blob storage.
        5)  Update the training version in training table stating in-progress.
        6)  Start the fine tuning job using the file id that is created as part of step #2

        Returns:
            Optional[str]: The job id of the new fine-tuning job.
        """
        self.fine_tuning_job_id = None
        self.training_file_id = None
        training_data = None

        try:
            # Step 1: Generate training data
            training_data = await self.data_loader.get_training_data_for_fine_tuning(formatter_fn=self.format_gpt4o)

            if not training_data:
                self.logger.info("Insufficient training data training data for fine-tuning.")
                return None

            # Step 2: Create training file from data and upload to OpenAI
            self.training_file_id = self.create_training_file(data=training_data["data"])

            # Step 3: Wait for finished upload to OpenAI
            self.wait_for_file_import(file_id=self.training_file_id)  # throws exception if it times out or fails.

            if self.config.TRAINING_DATA_TRANSFER_TYPE == "file":
                self.training_file_blob_name = training_data["data"]
            else:
                self.training_file_blob_name = training_data["data"].name

            # Step 4: upload the training file to storage container.
            self.upload_file_to_blob_storage(
                self.config.AZ_AKSSTAI_FINE_TUNED_LLM_CONTAINER_NAME,
                blob_name=self.training_file_blob_name,
                data=training_data["data"],
            )
            self.logger.info(f"Training file uploaded to storage : {self.training_file_blob_name}")

            self.training_file_uploaded = True

            # Step 5: Update the training tables rows training version to in-progress
            await self.data_loader.update_training_version(
                ref_row_ids=training_data["training_table_ids"],
                training_version_name=self.TRAINING_DATA_VER_IN_PROGRESS,
                bulk_update=True,
                batch_size=2000,
            )
            self.training_version_updated = True

            self.logger.info("Triggering fine tuning job.")
            # Step 6: trigger fine tuning job
            current_model_name, current_model_version = self.get_model_from_deployment(
                self.config.AOAI_RG,
                self.config.AOAI_FINETUNED_LLM_RESOURCE_NAME,
                self.config.AOAI_FINETUNED_LLM_DEPLOYMENT_API_VERSION,
                self.config.AOAI_FINETUNED_LLM_API_DEPLOYMENT,
            )

            self.logger.info(
                f"Using model - {current_model_name} as base model for fine tuning, with version - {current_model_version}"
            )

            self.fine_tuning_job_id = self.create_finetune_llm_job(
                training_file_id=self.training_file_id, base_llm_model_name=current_model_name
            )

            return self.fine_tuning_job_id
        except Exception as e:
            self.logger.error(f"An error occurred on fine-tuning job {e}", exc_info=True)

            # cleanup
            if self.fine_tuning_job_id:
                self.cancel_fine_tuning_job(self.fine_tuning_job_id)

            if self.training_file_id:
                self.delete_training_file(self.training_file_id)

            if training_data and self.training_version_updated:
                # move the training version to back to 'NEW', as finetuning is cancelled.
                await self.data_loader.update_training_version(
                    ref_row_ids=training_data["training_table_ids"],
                    training_version_name=TrainingDataVersions.NEW,
                    bulk_update=True,
                    batch_size=2000,
                )

            if self.training_file_uploaded:
                self.delete_file_from_blob_storage(
                    self.config.AZ_AKSSTAI_FINE_TUNED_LLM_CONTAINER_NAME, self.training_file_blob_name
                )

            raise e
