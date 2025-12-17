from classifier_constants import DEV_DEPLOYMENT_NAME, JOB_CONFIG_FILE_NAME, PROD_DEPLOYMENT_NAME

from utils import load_yaml


class JobConfig:
    def __init__(self) -> None:
        yaml = load_yaml(JOB_CONFIG_FILE_NAME)
        self.classifiers: list[str] = yaml["CLASSIFIERS"]
        self.RELOAD_ALL_TRAINING_DATA = yaml["RELOAD_ALL_TRAINING_DATA"]
        self.DEV_DEPLOYMENT_NAME = yaml.get("DEV_DEPLOYMENT_NAME", DEV_DEPLOYMENT_NAME)
        self.PROD_DEPLOYMENT_NAME = yaml.get("PROD_DEPLOYMENT_NAME", PROD_DEPLOYMENT_NAME)
        self.THRESHOLD = yaml.get("THRESHOLD", 1)
        self.NEW_DATA_THRESHOLD = yaml.get("NEW_DATA_THRESHOLD", 1000)
        self.MAX_CLASSIFIERS_RUNNING = yaml.get("MAX_CLASSIFIERS_RUNNING", 1)
