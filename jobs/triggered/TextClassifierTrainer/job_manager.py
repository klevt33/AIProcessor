import json
from logging import Logger

from app_config import AppConfig
from azure_api import AzureAPI
from classifier_config import ClassifierConfig
from classifier_constants import TrainingDataLabels
from classifier_util import get_current_date_str, training_data_to_labels_file
from data_manager import DataManager, TrainingData
from evaluator import Evaluator
from job_config import JobConfig
from storage_client import StorageClient

from config import Config


class JobManager:
    def __init__(self, job_config: JobConfig, config: Config, app_config: AppConfig, logger: Logger) -> None:
        self.job_config = job_config
        self.config = config
        self.logger = logger
        self.app_config = app_config
        self.data_manager = DataManager(config, logger)

    def get_in_progress_job_count(self) -> int:
        count = 0
        azure_api = AzureAPI(self.logger, "TEMP")
        for classifier in self.job_config.classifiers:
            training_job_id = self.app_config.get_job_id(classifier)
            if training_job_id:
                PROJECT_NAME = azure_api.get_latest_project_name(classifier)
                azure_api.project_name = PROJECT_NAME
                status = azure_api.get_training_status(training_job_id)
                if azure_api.in_progress(status):
                    count += 1
        return count

    async def evaluate_in_progress_jobs(self) -> None:
        azure_api = AzureAPI(self.logger, "TEMP")
        for classifier in self.job_config.classifiers:
            try:
                training_job_id = self.app_config.get_job_id(classifier)
                if training_job_id:
                    PROJECT_NAME = azure_api.get_latest_project_name(classifier)
                    azure_api.project_name = PROJECT_NAME
                    status = azure_api.get_training_status(training_job_id)
                    if azure_api.in_progress(status):
                        self.logger.info(f"Classifier ({classifier}) is still in progress.")
                    elif azure_api.is_unsuccessful(status):
                        self.logger.error(f"Classifier ({classifier}) was unsuccessful.")
                    else:
                        classifier_config = ClassifierConfig(classifier, self.job_config, self.config)
                        storage_client = StorageClient(classifier_config)
                        self.logger.info(f"Classifier ({classifier}) was successful.")
                        evaluation = azure_api.get_model_evaluation(classifier_config.model_name)
                        self.logger.info(f"Classifier {classifier} eval: {evaluation}")
                        storage_client.save_evaluation(evaluation, PROJECT_NAME)
                        evaluator = Evaluator(classifier_config)
                        if evaluator.meets_threshold(evaluation):
                            self.logger.info(f"Classifier {classifier} met the threshold.")
                            project = self.app_config.get_project(classifier)
                            if project:
                                current_evaluation = storage_client.get_current_model_eval(project)
                            else:
                                # Assume beats model since there is no previous model
                                current_evaluation = None
                            if not current_evaluation or evaluator.beats_current_model(evaluation, current_evaluation):
                                self.logger.info(f"Classifier {classifier} beat the current model.")
                                azure_api.deploy_model(classifier_config.dev_deployment_name, classifier_config)
                                await azure_api.wait_deployment(classifier_config.dev_deployment_name)
                                azure_api.deploy_model(classifier_config.prod_deployment_name, classifier_config)
                                await azure_api.wait_deployment(classifier_config.prod_deployment_name)
                                self.app_config.set_project(classifier, PROJECT_NAME)
                                oldest_project_name = azure_api.get_oldest_project_name(classifier)
                                if oldest_project_name:
                                    azure_api.delete_project(oldest_project_name)
                                    storage_client.delete_training_data(oldest_project_name)
                            else:
                                azure_api.delete_project(PROJECT_NAME)
                                storage_client.delete_training_data(PROJECT_NAME)
                                self.logger.warning(f"Classifier: ({classifier}) failed to beat the current model evaluation.")
                        else:
                            self.logger.warning(f"Classifier ({classifier}) failed to meet the threshold.")
                        self.app_config.set_job_id(classifier, "")
            except Exception as e:
                self.logger.error(f"Unable to evaluate classifier: {classifier}", e)
        self.app_config.upload_changes()

    async def start_training_job(self, classifier: str, all_training_data: list[TrainingData]):
        classifier_config = ClassifierConfig(classifier, self.job_config, self.config)
        CLASSES = classifier_config.classes
        CONTAINER_NAME = classifier_config.container_name
        MODEL_NAME = classifier_config.model_name
        storage_client = StorageClient(classifier_config)

        # Get new training data
        # TODO: test
        training_to_upload = [
            d for d in all_training_data if getattr(d, classifier_config.col_name) in classifier_config.db_classes
        ]
        training_to_upload = self.data_manager.clean_data(training_to_upload)
        training_to_upload = self.data_manager.remove_duplicates(training_to_upload)
        training_to_upload = self.data_manager.balance_training_data(training_to_upload, classifier_config)
        self.logger.info(f"Training dataset size: {len(training_to_upload)}")

        new_training_data_count = sum(1 for d in training_to_upload if d.training_version[classifier] == TrainingDataLabels.NEW)
        if new_training_data_count < classifier_config.new_data_threshold:
            self.logger.info("Skipping classifier since it doesn't meet threshold of new documents.")
            return

        def map_data(datum: TrainingData, classifier_config=classifier_config) -> TrainingData:
            """Map data from classifier config yaml file, for example map MATERIAL to NOT_LOT for lot classifier"""
            class_col_name = classifier_config.col_name
            datum.classification = classifier_config.classes_dict[getattr(datum, class_col_name)]
            return datum

        applicable_training_data = tuple(map_data(d) for d in training_to_upload)
        training_folder = f"training-data-{get_current_date_str()}"
        self.logger.info("Uploading data")
        for training_datum in applicable_training_data:
            storage_client.save_training_data(training_datum, training_folder)

        PROJECT_NAME = f"{classifier}-classifier-{get_current_date_str()}"
        azure_api = AzureAPI(self.logger, PROJECT_NAME)
        labels_file_dict = training_data_to_labels_file(
            classifier, CONTAINER_NAME, PROJECT_NAME, applicable_training_data, CLASSES, training_folder
        )
        storage_client.save_labels_file(json.dumps(labels_file_dict, indent=4))
        azure_api.import_project_job(labels_file_dict)
        await azure_api.wait_import_job()
        training_job_id = azure_api.start_training(MODEL_NAME)
        for d in applicable_training_data:
            d.training_version[classifier] = TrainingDataLabels.USED.value
        await self.data_manager.update_training_data_version(applicable_training_data)
        self.app_config.set_job_id(classifier, training_job_id)
