import json
from collections import defaultdict
from logging import Logger

from classifier_constants import APP_CONF_CLASSIFIER_TRAINING

from config import Config
from constants import Environments


class AppConfig:
    def __init__(self, config: Config, logger: Logger, env: str) -> None:
        self.config = config
        self.logger = logger
        self.env = env

        self.app_config_label = Environments.DEV if self.env != Environments.PROD else Environments.PROD
        self._config_dict = defaultdict(
            dict,
            json.loads(
                self.config.get_azure_app_config_value(APP_CONF_CLASSIFIER_TRAINING, self.app_config_label, client_type="AI")
            ),
        )
        self.project_names = {}

    def upload_changes(self):
        self.config.set_azure_app_config_value(
            APP_CONF_CLASSIFIER_TRAINING,
            self.app_config_label,
            json.dumps(self._config_dict, indent=4),
            content_type="application/json",
            client_type="AI",
        )
        for k, v in self.project_names:
            self.config.set_azure_app_config_value(k, self.app_config_label, v, content_type="application/text", client_type="AI")

    def get_job_id(self, classifier: str) -> str | None:
        """
        Retrieves the current classifier's job ID from Azure App Configuration.

        Returns:
            job_id (Optional[str]): The classifier's job ID.
        """
        return self._config_dict[classifier].get("job_id", None)

    def set_job_id(self, classifier: str, job_id: str):
        self._config_dict[classifier]["job_id"] = job_id

    def get_project(self, classifier: str) -> str | None:
        return self._config_dict[classifier].get("project", None)

    def set_project(self, classifier: str, project: str):
        self._config_dict[classifier]["project"] = project
        self.project_names[f"LANGUAGE_STUDIO:{classifier.upper()}:PROJECT_NAME"] = project
