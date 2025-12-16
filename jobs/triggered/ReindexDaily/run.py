# /jobs/triggered/ReindexDaily/run.py
import os
import sys

WEBJOB_ID = "1"
WEBJOB_NAME = "ReindexDaily"
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


# def setup_logger():
#     # --- Logging Configuration for WebJob ---
#     # Logs go to /home/LogFiles/Application/Python/ (for stdout/stderr in App Service)
#     # and optionally to a persistent file in /home/logs if you add a FileHandler
#     LOG_DIR_PERSISTENT = "/home/logs/ReindexDaily"  # /home is persistent in App Service
#     os.makedirs(LOG_DIR_PERSISTENT, exist_ok=True)
#     LOG_FILE_PERSISTENT = os.path.join(LOG_DIR_PERSISTENT, "reindex_job.log")

#     # Configure root logger. This will affect loggers in imported modules too (e.g., indexer.semantic_search_indexer)
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
#         handlers=[
#             logging.FileHandler(LOG_FILE_PERSISTENT), # Log to a persistent file
#             logging.StreamHandler(sys.stdout)         # Log to stdout (captured by Azure WebJobs dashboard)
#         ]
#     )
#     logger = logging.getLogger(__name__) # Logger for this run.py script
#     return logger


def main():
    from indexer.semantic_search_indexer import run_indexing

    logger.info("ReindexDaily WebJob started.")
    try:
        app_config = Config(app_root=APP_ROOT)  # Initialize configuration from environment variables

        # For the scheduled job, run with default behavior (no rebuild, process all)
        logger.info("Calling indexing logic with default parameters (no rebuild, no max records limit).")
        success = run_indexing(config_obj=app_config, rebuild=False, max_records=None)

        if success:
            logger.info("ReindexDaily WebJob completed successfully.")
            # For triggered WebJobs, Azure checks the exit code. 0 for success.
            # sys.exit(0) # Not strictly necessary as Python exits with 0 by default on normal completion
        else:
            logger.error("ReindexDaily WebJob encountered errors during indexing.")
            sys.exit(1)  # Non-zero exit code to indicate failure to Azure

    except Exception:
        logger.exception("An unhandled exception occurred in the ReindexDaily WebJob main function")
        sys.exit(2)  # Different non-zero exit code for job infrastructure failure


if __name__ == "__main__":
    # Load the app files
    setup_env()

    from config import Config
    from logger import get_job_logger, job_id_var

    CONFIG = Config(app_root=APP_ROOT)  # Load application configuration
    print("APP VERSION Running: ", CONFIG.app_version, "WEBJOB VERSION Running: ", WEBJOB_VERSION)

    logger = get_job_logger(name=WEBJOB_NAME, azure_conn_str=CONFIG.APP_INSIGHTS_CONN_STRING)  # Initialize logger
    job_id_var.set(WEBJOB_ID)  # Store the request ID in the context variable

    main()
