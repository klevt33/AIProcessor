from classifier_constants import get_container_name, get_model_name, get_project_name
from job_config import JobConfig

from config import Config
from utils import load_yaml


class ClassifierConfig:
    def __init__(self, classification: str, job_config: JobConfig, config: Config) -> None:
        yaml = load_yaml(f"{classification}_config.yaml")
        self.classifier = classification
        self.job_config = job_config
        self.config = config
        self.container_name: str = yaml.get("CONTAINER_NAME", get_container_name(classification))
        self.project_name: str = yaml.get("PROJECT_NAME", get_project_name(classification))
        self.model_name: str = yaml.get("MODEL_NAME", get_model_name(classification))
        self.dev_deployment_name: str = yaml.get("DEV_DEPLOYMENT_NAME", job_config.DEV_DEPLOYMENT_NAME)
        self.prod_deployment_name: str = yaml.get("PROD_DEPLOYMENT_NAME", job_config.PROD_DEPLOYMENT_NAME)
        self.threshold: float = yaml.get("THRESHOLD", job_config.THRESHOLD)
        self.new_data_threshold: float = yaml.get("NEW_DATA_THRESHOLD", job_config.NEW_DATA_THRESHOLD)
        self.col_name = yaml.get("CLASS_COL_NAME", "classification")

        classes_dict: list[str] | dict[str, str] = yaml["CLASSES"]
        if isinstance(classes_dict, list):
            classes_dict = {x: x for x in classes_dict}
        else:
            classes_dict = classes_dict
        self.classes_dict: dict[str, str] = classes_dict
        self.db_classes = self.classes_dict.keys()
        self.classes = list(set(self.classes_dict.values()))
