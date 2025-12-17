import asyncio
import json
import logging
from datetime import datetime
from urllib.parse import urlparse

import requests
from classifier_constants import API_ENDPOINT_URL, API_KEY, API_VERSION
from tenacity import retry, stop_after_attempt, wait_exponential


class AzureAPI:
    def __init__(self, logger: logging.Logger, project_name: str, api_version: str = API_VERSION) -> None:
        self.logger = logger
        self.api_version = api_version
        self.project_name = project_name

    @staticmethod
    def in_progress(status: str) -> bool:
        """Returns whether the status of an azure job is in progress (running or not started)"""
        return status in ["running", "notStarted"]

    @staticmethod
    def is_unsuccessful(status: str) -> bool:
        """Returns whether the status of an azure job is not successful and not in progress"""
        return not any((AzureAPI.in_progress(status), AzureAPI.is_successful(status)))

    @staticmethod
    def is_successful(status: str) -> bool:
        """Returns whether the status of an azure job is successful or not"""
        return status == "succeeded"

    @staticmethod
    def _get_job_id(headers: dict) -> str:
        """Extract the job id from the headers

        Args:
            headers (dict): Headers of the response from an API call

        Returns:
            str: job id
        """
        operation_url = headers["operation-location"]
        path = urlparse(operation_url).path
        return path.split("/")[-1].split("?")[0]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=5, max=30))
    def import_project_job(self, labels_file: dict) -> str:
        """Import project job to azure language studio

        Args:
            labels_file (dict): labels_file for the project

        Raises:
            e: Error from the requests post call
            Exception: Did not receive the expected 202 status code from the API

        Returns:
            str: job ID for the in progress import project job
        """
        self.logger.info(f"Importing project job for {self.project_name}")
        url = (
            f"{API_ENDPOINT_URL}language/authoring/analyze-text/projects/{self.project_name}/:import?"
            f"api-version={self.api_version}"
        )
        headers = {"Ocp-Apim-Subscription-Key": API_KEY, "Content-Type": "application/json"}
        try:
            response = requests.post(url, headers=headers, json=labels_file)
        except Exception as e:
            self.logger.exception("Error importing project job.")
            raise e
        if response.status_code != 202:
            self.logger.error("Unable to correctly import project job.")
            raise Exception("Unable to correctly import project job.")
        self.import_job_id = self._get_job_id(response.headers)
        self.logger.info(f"Import project job started. Job ID: {self.import_job_id}")
        return self.import_job_id

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=5, max=30))
    def get_import_job_status(self, job_id: str | None = None) -> str:
        """Get the status of the import job

        Args:
            job_id (str | None, optional): job ID to check the status of. Defaults to None.

        Raises:
            e: Error from the requests post call
            Exception: Did not receive the expected 200 status code from the API

        Returns:
            str: Status for the import job
        """
        if not job_id:
            job_id = self.import_job_id
        self.logger.info("Getting import job status.")
        url = (
            f"{API_ENDPOINT_URL}language/authoring/analyze-text/projects/{self.project_name}/import/jobs/{job_id}"
            f"?api-version={self.api_version}"
        )
        headers = {"Ocp-Apim-Subscription-Key": API_KEY}
        try:
            response = requests.get(url, headers=headers)
        except Exception as e:
            self.logger.exception("Error getting import project job status.")
            raise e
        if response.status_code != 200:
            self.logger.error("Unable to correctly get import project job status.")
            raise Exception("Unable to correctly get import project job status.")
        status = json.loads(response.content.decode("utf-8"))["status"]
        self.logger.info(f"Import job status: {status} for project: {self.project_name} for job id: {job_id}")
        return status

    async def wait_import_job(self, job_id: str | None = None):
        """Wait for the import job to complete. Will return when completed.

        Args:
            job_id (str | None, optional): Import job id to wait on. Defaults to None.

        Raises:
            RuntimeError: Unacceptable job status, not "running" and not "succeeded"
            TimeoutError: Waited longer than the maximum of time allotted
        """
        self.logger.info(f"Waiting for import job to complete for project {self.project_name}")
        max_wait = 60
        current_wait = 0
        status = self.get_import_job_status(job_id)
        while not self.is_successful(status):
            if not self.in_progress(status):
                self.logger.error(f"Import job status is in an unacceptable state. Status: {status}")
                raise RuntimeError("Import job status is in an unacceptable state.")
            if current_wait >= max_wait:
                raise TimeoutError("Waited too long for import job to complete.")
            await asyncio.sleep(5)
            current_wait += 5
            status = self.get_import_job_status(job_id)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=5, max=30))
    def start_training(self, model_name: str, model_version: str = "latest"):
        """Start training for azure language studio

        Args:
            model_name (str): Model name to train
            model_version (str): model version to use from azure language studio

        Raises:
            e: Error from the requests post call
            Exception: Did not receive the expected 202 status code from the API

        Returns:
            str: job ID for the in progress training job
        """
        self.logger.info(f"Starting training for {self.project_name}")
        url = (
            f"{API_ENDPOINT_URL}language/authoring/analyze-text/projects/{self.project_name}/:train"
            f"?api-version={self.api_version}"
        )
        headers = {"Ocp-Apim-Subscription-Key": API_KEY, "Content-Type": "application/json"}
        body = {
            "modelLabel": model_name,
            "trainingConfigVersion": model_version,
            "evaluationOptions": {"kind": "percentage", "trainingSplitPercentage": 80, "testingSplitPercentage": 20},
        }
        try:
            response = requests.post(url, headers=headers, json=body)
        except Exception as e:
            self.logger.exception("Error starting training.")
            raise e
        if response.status_code != 202:
            self.logger.error("Unable to start training.")
            raise Exception("Unable to start training.")
        self.training_job_id = self._get_job_id(response.headers)
        self.logger.info(f"Training job started. Job ID: {self.training_job_id}")
        return self.training_job_id

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=5, max=30))
    def get_training_status(self, job_id: str | None = None) -> str:
        """Get the status of the training job

        Args:
            job_id (str | None, optional): job ID to check the status of. Defaults to None.

        Raises:
            e: Error from the requests post call
            Exception: Did not receive the expected 200 status code from the API

        Returns:
            str: Status for the training job
        """
        if not job_id:
            job_id = self.training_job_id
        self.logger.info("Getting training job status.")
        url = (
            f"{API_ENDPOINT_URL}language/authoring/analyze-text/projects/{self.project_name}/train/jobs/{job_id}?"
            f"api-version={self.api_version}"
        )
        headers = {"Ocp-Apim-Subscription-Key": API_KEY}
        try:
            response = requests.get(url, headers=headers)
        except Exception as e:
            self.logger.exception("Error getting training job status.")
            raise e
        if response.status_code != 200:
            self.logger.error("Unable to correctly get training job status.")
            raise Exception("Unable to correctly get training job status.")
        status = json.loads(response.content.decode("utf-8"))["status"]
        self.logger.info(f"Training job status: {status} for project: {self.project_name} for job id: {job_id}")
        return status

    async def wait_training(self, job_id: str | None = None):
        """Wait for the training job to complete. Will return when completed.

        Args:
            job_id (str | None, optional): Training job id to wait on. Defaults to None.

        Raises:
            RuntimeError: Unacceptable job status, not "running" and not "succeeded" and not "notStarted"
            TimeoutError: Waited longer than the maximum of time allotted
        """
        self.logger.info(f"Waiting for training job to complete for project {self.project_name}")
        # TODO: update max wait to be reasonable even for large jobs, or maybe split into two sessions like finetuneLLM
        max_wait = 60 * 100
        current_wait = 0
        status = self.get_training_status(job_id)
        while not self.is_successful(status):
            if not self.in_progress(status):
                raise RuntimeError(f"Training status is in an unacceptable state. Status: {status}")
            if current_wait >= max_wait:
                raise TimeoutError("Waited too long for training job to complete.")
            await asyncio.sleep(5)
            current_wait += 5
            status = self.get_training_status(job_id)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=5, max=30))
    def get_model_evaluation(self, model_name: str) -> dict:
        """Retrieve the evaluation metrics for the given model.

        Args:
            model_name (str): Name of the model to get metrics for

        Raises:
            e: Error from the requests post call
            Exception: Did not receive the expected 200 status code from the API

        Returns:
            dict: Evaluation metrics data
        """
        self.logger.info("Getting model evaluation.")
        url = (
            f"{API_ENDPOINT_URL}language/authoring/analyze-text/projects/{self.project_name}/models/{model_name}/evaluation"
            f"/summary-result?api-version={self.api_version}"
        )
        headers = {"Ocp-Apim-Subscription-Key": API_KEY}
        try:
            response = requests.get(url, headers=headers)
        except Exception as e:
            self.logger.exception("Error getting model evaluation.")
            raise e
        if response.status_code != 200:
            self.logger.error("Unable to correctly get model evaluation.")
            raise Exception("Unable to correctly get model evaluation.")
        return json.loads(response.content.decode("utf-8"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=5, max=30))
    def deploy_model(self, deployment_name: str, model_name: str) -> str:
        """Deploy model for azure language studio

        Args:
            deployment_name (str): Name of deployment
            model_name (str): Name of the model to deploy

        Raises:
            e: Error from the requests post call
            Exception: Did not receive the expected 202 status code from the API

        Returns:
            str: job ID for the in progress deployment job
        """
        self.logger.info(f"Starting deployment for project: {self.project_name}, model: {model_name}")
        url = (
            f"{API_ENDPOINT_URL}language/authoring/analyze-text/projects/{self.project_name}/deployments/{deployment_name}"
            f"?api-version={self.api_version}"
        )
        headers = {"Ocp-Apim-Subscription-Key": API_KEY, "Content-Type": "application/json"}
        body = {"trainedModelLabel": model_name}
        try:
            response = requests.put(url, headers=headers, json=body)
        except Exception as e:
            self.logger.exception("Error starting deployment.")
            raise e
        if response.status_code != 202:
            self.logger.error("Unable to start deployment.")
            raise Exception("Unable to start deployment.")
        self.deployment_job_id = self._get_job_id(response.headers)
        self.logger.info(f"Deployment job started. Job ID: {self.deployment_job_id}")
        return self.deployment_job_id

    def get_deployment_status(self, deployment_name: str, job_id: str | None = None) -> str:
        """Get the status of the deployment

        Args:
            deployment_name (str): name of the deployment
            job_id (str | None, optional): job ID to check the status of. Defaults to None.

        Raises:
            e: Error from the requests post call
            Exception: Did not receive the expected 200 status code from the API

        Returns:
            str: Status for the deployment
        """
        if not job_id:
            job_id = self.deployment_job_id
        self.logger.info("Getting deployment job status.")
        url = (
            f"{API_ENDPOINT_URL}language/authoring/analyze-text/projects/{self.project_name}/deployments/"
            f"{deployment_name}/jobs/{job_id}?api-version={self.api_version}"
        )
        headers = {"Ocp-Apim-Subscription-Key": API_KEY}
        try:
            response = requests.get(url, headers=headers)
        except Exception as e:
            self.logger.exception("Error getting deployment job status.")
            raise e
        if response.status_code != 200:
            self.logger.error("Unable to correctly get deployment job status.")
            raise Exception("Unable to correctly get deployment job status.")
        status = json.loads(response.content.decode("utf-8"))["status"]
        self.logger.info(
            f"Deployment job status: {status} for project: {self.project_name} for deployment: {deployment_name} for job id:"
            f" {job_id}"
        )
        return status

    async def wait_deployment(self, deployment_name, job_id: str | None = None):
        """Wait for the deployment job to complete. Will return when completed.

        Args:
            job_id (str | None, optional): Deployment job id to wait on. Defaults to None.

        Raises:
            RuntimeError: Unacceptable job status, not "running" and not "succeeded"
            TimeoutError: Waited longer than the maximum of time allotted
        """
        self.logger.info(f"Waiting for training job to complete for project {self.project_name}")
        max_wait = 60 * 100
        current_wait = 0
        status = self.get_deployment_status(deployment_name, job_id)
        while not self.is_successful(status):
            if not self.in_progress(status):
                raise RuntimeError(f"Deployment status is in an unacceptable state. Status: {status}")
            if current_wait >= max_wait:
                raise TimeoutError("Waited too long for training job to complete.")
            await asyncio.sleep(5)
            current_wait += 5
            status = self.get_deployment_status(deployment_name, job_id)

    def _get_list_projects(self):
        url = f"{API_ENDPOINT_URL}language/authoring/analyze-text/projects?api-version={self.api_version}"
        headers = {"Ocp-Apim-Subscription-Key": API_KEY}
        try:
            response = requests.get(url, headers=headers)
        except Exception as e:
            self.logger.exception("Error getting deployment job status.")
            raise e
        if response.status_code != 200:
            self.logger.error("Unable to correctly get deployment job status.")
            raise Exception("Unable to correctly get deployment job status.")

        projects = json.loads(response.content.decode("utf-8"))["value"]
        return projects

    def get_latest_project_name(self, classifier: str) -> str:
        """For the given classifier, return the latest project data by parsing the date from the project name field,
        otherwise look at created date

        Args:
            classifier (str): Latest classifier project name to retrieve

        Raises:
            Exception: Unable to find projects for that classifier

        Returns:
            dict: project data of the match
        """
        return self.get_oldest_project_name(classifier, num=1)

    def get_oldest_project_name(self, classifier: str, num: int = 3) -> str:
        projects = [x["projectName"] for x in self._get_list_projects()]
        projects = [x for x in projects if x.startswith(f"{classifier}-classifier-")]

        if len(projects) < num:
            self.logger.info(f"Don't have enough projects to get to {num}")
            return None

        def get_date(data):
            date_str = data.removeprefix(f"{classifier}-classifier-")

            for fmt in ("%Y-%m-%d", "%m-%d-%y"):
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Cannot get date for {data}")

        # Sort by date to get the most recent
        projects = sorted(projects, key=get_date, reverse=True)
        return projects[num - 1]

    def delete_project(self, project_name: str):
        url = f"{API_ENDPOINT_URL}language/authoring/analyze-text/projects/{project_name}?api-version={self.api_version}"
        headers = {"Ocp-Apim-Subscription-Key": API_KEY}
        try:
            response = requests.delete(url, headers=headers)
        except Exception as e:
            self.logger.exception("Error getting deployment job status.")
            raise e
        if response.status_code != 202:
            self.logger.error("Unable to correctly get deployment job status.")
            raise Exception("Unable to correctly get deployment job status.")
