import json
import re
from typing import Any

from classifier_config import ClassifierConfig
from classifier_constants import LABELS_FILE_NAME
from training_data import TrainingData


class StorageClient:
    def __init__(self, classifier_config: ClassifierConfig):
        self.config = classifier_config.config
        self.classifier_config = classifier_config

    def save_training_data(self, training_data: TrainingData, training_folder: str):
        """Save the given training data to the storage container

        Args:
            training_data (TrainingData): training data to save
        """
        blob_client = self.config.azure_clients.azure_blob_storage_client.get_blob_client(
            container=self.classifier_config.container_name, blob=training_data.get_blob_location(training_folder)
        )
        blob_client.upload_blob(training_data.invoice_description, overwrite=True)

    def save_labels_file(self, labels_file_data: str):
        """Save the given labels file to the storage container

        Args:
            labels_file_data (str): labels file to save
        """
        blob_client = self.config.azure_clients.azure_blob_storage_client.get_blob_client(
            container=self.classifier_config.container_name, blob=f"{LABELS_FILE_NAME}"
        )
        blob_client.upload_blob(labels_file_data, overwrite=True)

    def _get_eval_file_name(self, project_name) -> str:
        return f"{project_name}.json"

    def save_evaluation(self, evaluation: dict[str, Any], project_name: str):
        json_data = json.dumps(evaluation, indent=4)

        container_name = self.classifier_config.container_name
        blob_name = self._get_eval_file_name(project_name)
        blob_client = self.config.azure_clients.azure_blob_storage_client.get_blob_client(
            container=container_name, blob=blob_name
        )
        blob_client.upload_blob(json_data, overwrite=True)

    def get_evaluation(self, project_name: str) -> dict[str, Any]:
        container_name = self.classifier_config.container_name
        blob_name = self._get_eval_file_name(project_name)
        blob_client = self.config.azure_clients.azure_blob_storage_client.get_blob_client(
            container=container_name, blob=blob_name
        )
        blob_content = blob_client.download_blob().readall()
        return json.loads(blob_content)

    def get_current_model_eval(self, project_name) -> dict[str, Any]:
        return self.get_evaluation(project_name)

    def delete_training_data(self, project_name):
        regex = r".*-classifier-(\d{4}-\d{2}-\d{2})"
        date_str = re.search(regex, project_name).group(1)

        container_client = self.config.azure_clients.azure_blob_storage_client.get_container_client(
            container=self.classifier_config.container_name
        )
        for blob in container_client.list_blobs(name_starts_with=f"training-data-{date_str}"):
            container_client.delete_blob(blob.name)
