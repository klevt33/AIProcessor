from enum import StrEnum

LABELS_FILE_NAME = "labelsFile.json"
API_VERSION = "2022-05-01"
JOB_CONFIG_FILE_NAME = "job_config.yaml"
DEV_DEPLOYMENT_NAME = "dev"
PROD_DEPLOYMENT_NAME = "dev"
API_ENDPOINT_URL = "https://aks-language-studio.cognitiveservices.azure.com/"
API_KEY = os.environ["TEXT_ANALYTICS_KEY"]
APP_CONF_CLASSIFIER_TRAINING = "WEBJOB:CLASSIFIER_TRAINING"
ALL_CLASSIFIERS = ["description", "lot", "rental"]


class TrainingDataLabels(StrEnum):
    USED = "USED"
    NEW = "NEW"


def get_container_name(classifier: str) -> str:
    return f"spendreport-automated-{classifier}-classifier"


def get_project_name(classifier: str) -> str:
    return f"{classifier}-classifier-automated"


def get_model_name(classifier: str) -> str:
    return f"{classifier}-classifier"
