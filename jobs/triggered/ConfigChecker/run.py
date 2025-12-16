import os
import sys

WEBJOB_ID = "1"
WEBJOB_NAME = "ConfigChecker"
WEBJOB_VERSION = "1.0.0"  # Prod_Major.Prod_Minor.Dev_Deploy

VAR_NAME = "WEB_APP_ENV"
AZURE_WWWROOT = "/home/site/wwwroot"
APP_ROOT = ""


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

    if environment == "local":
        # Project root is three levels up.
        file_path = os.path.abspath(__file__)
        root = os.path.abspath(os.path.join(file_path, "..", "..", "..", ".."))
        if os.path.isdir(root) and root not in sys.path:
            sys.path.insert(0, root)
            APP_ROOT = root
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


def display_config():
    from config import Config

    try:
        config = Config(app_root=APP_ROOT)
        values = config.__dict__
        logger.info(f"Config values are: {values}")
        return True

    except Exception as e:
        logger.exception(f"Error occurred in execution. Error: {str(e)}")
        return False


def main():
    logger.info("ConfigChecker WebJob started.")

    try:
        success = display_config()

        if success:
            logger.info("ConfigChecker WebJob completed successfully.")
            # For triggered WebJobs, Azure checks the exit code. 0 for success.
            # sys.exit(0) # Not strictly necessary as Python exits with 0 by default on normal completion
        else:
            logger.error("ConfigChecker WebJob encountered errors.")
            sys.exit(1)  # Non-zero exit code to indicate failure to Azure

    except Exception:
        logger.exception("An unhandled exception occurred in the ConfigChecker WebJob main function")
        sys.exit(2)  # Different non-zero exit code for job infrastructure failure


if __name__ == "__main__":
    # Load the app files
    setup_env()

    # set logger
    from config import Config
    from logger import get_job_logger, job_id_var

    CONFIG = Config(app_root=APP_ROOT)  # Load application configuration

    logger = get_job_logger(name=WEBJOB_NAME, azure_conn_str=CONFIG.APP_INSIGHTS_CONN_STRING)  # Initialize logger
    job_id_var.set(WEBJOB_ID)  # Store the request ID in the context variable

    logger.info(f"APP VERSION Running: {CONFIG.app_version}")
    logger.info(f"WEBJOB VERSION Running: {WEBJOB_VERSION}")

    main()
