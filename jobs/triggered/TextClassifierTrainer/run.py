import asyncio
import logging
import os
import sys

WEBJOB_ID = "5"
WEBJOB_NAME = "TextClassifierTrainer"
WEBJOB_VERSION = "1.0.0"  # Prod_Major.Prod_Minor.Dev_Deploy

VAR_NAME = "WEB_APP_ENV"
AZURE_WWWROOT = "/home/site/wwwroot"
APP_ROOT = ""
ENV = ""


async def main(config, logger: logging.Logger):
    from app_config import AppConfig
    from classifier_sql import get_all_classifier_training_data
    from job_config import JobConfig
    from job_manager import JobManager
    from training_data import TrainingData

    from sdp import SDP

    job_config = JobConfig()
    app_config = AppConfig(config, logger, ENV)
    job_manager = JobManager(job_config, config, app_config, logger)
    sdp = SDP(config)

    await job_manager.evaluate_in_progress_jobs()

    in_progress_job_count = job_manager.get_in_progress_job_count()
    if in_progress_job_count >= job_config.MAX_CLASSIFIERS_RUNNING:
        logger.info("Already have max number of jobs running.")
        return

    all_training_data = [TrainingData(x) for x in (await get_all_classifier_training_data(sdp)).to_dict(orient="records")]

    for classifier in job_config.classifiers:
        if in_progress_job_count >= job_config.MAX_CLASSIFIERS_RUNNING:
            logger.info("Already have max number of jobs running.")
            return
        logger.info(f"Working on classifier: {classifier}")
        try:
            await job_manager.start_training_job(classifier, all_training_data)
            in_progress_job_count += 1
        except Exception:
            logger.error(f"Failed to work with classifier: {classifier}", exc_info=True)

    # Update all app config changes
    app_config.upload_changes()


def setup_env():
    """
    Add project root to sys.path
    """
    global APP_ROOT

    if VAR_NAME in os.environ:
        environment = os.environ[VAR_NAME]
    else:
        environment = "local"
    print("ENV Running: ", environment)

    global ENV
    ENV = environment

    if environment == "local":
        # Project root is three levels up.
        file_path = os.path.abspath(__file__)
        root = os.path.abspath(os.path.join(file_path, "..", "..", "..", ".."))
        APP_ROOT = root
        if os.path.isdir(root) and root not in sys.path:
            sys.path.insert(0, root)
            print(f"[INFO] Added to sys.path: {root}")

    else:
        # ─── Locate the folder that contains config.py ───────────────────────────────
        candidates = [os.path.join(os.environ.get("HOME", "/home"), "site", "wwwroot"), "/app", os.getcwd()]

        for root in candidates:
            if os.path.isfile(os.path.join(root, "config.py")):
                sys.path.insert(0, root)
                APP_ROOT = root
                print(f"[INFO] Added to sys.path: {root}")
                break
        else:
            raise ImportError("Cannot find config.py in any of the known locations: " + ", ".join(candidates))


def setup_job_config():
    from pathlib import Path

    from constants import LocalFiles
    from utils import load_yaml

    jobs_config_file = Path(__file__).resolve().parent / LocalFiles.JOB_CONFIG_FILE
    job_config_yaml = load_yaml(path=str(jobs_config_file))
    print(f"Job config yaml: {job_config_yaml}")

    from config import Config

    config = Config(app_root=APP_ROOT)  # Load application configuration
    return config


if __name__ == "__main__":
    # Load the app files
    setup_env()

    config = setup_job_config()

    # set logger
    from logger import get_job_logger, job_id_var

    print("APP VERSION Running: ", config.app_version, "WEBJOB VERSION Running: ", WEBJOB_VERSION)

    logger = get_job_logger(name=WEBJOB_NAME, azure_conn_str=config.APP_INSIGHTS_CONN_STRING)  # Initialize logger
    job_id_var.set(WEBJOB_ID)  # Store the request ID in the context variable

    asyncio.run(main(config, logger))
