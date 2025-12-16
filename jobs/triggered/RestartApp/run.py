import json
import os
import sys
from time import sleep

import requests
from azure.identity import DefaultAzureCredential

WEBJOB_ID = "1"
WEBJOB_NAME = "RestartApp"
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


def restart_webapp(subscription_id: str, resource_group: str, app_name: str) -> bool:
    """
    Restarts an Azure Web App using the Azure Management REST API.

    Args:
        subscription_id (str): Azure subscription ID.
        resource_group (str): Azure resource group name.
        app_name (str): Name of the App Service to restart.

    Returns:
        bool: True if restart request was accepted, False otherwise.
    """
    try:
        # Get Azure AD token
        credential = DefaultAzureCredential()
        token = credential.get_token("https://management.azure.com/.default").token

        # Restart endpoint
        url = (
            f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/"
            f"{resource_group}/providers/Microsoft.Web/sites/{app_name}/restart?api-version=2022-03-01"
        )

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        logger.info(f"Triggering restart. URL: {url}")
        sleep(5)  # allow 5 sec to upload all logs to upload to analytics space
        response = requests.post(url, headers=headers)

        if response.status_code in (200, 202):
            logger.info(f"Restart triggered successfully for app '{app_name}'.")
            return True
        else:
            logger.info(f"Failed to restart app '{app_name}'. Status: {response.status_code}, Response: {response.text}")
            return False

    except Exception as e:
        logger.exception(f"Exception while restarting app: {e}")
        return False


def read_args():
    payload = None
    if len(sys.argv) > 1:
        try:
            # The first argument (after script name) is the JSON string
            payload = json.loads(sys.argv[1])
            logger.info(f"Received payload: {payload}")

            # Use values
            # my_key = payload.get("key")
            # env = payload.get("env")

            # print("Key:", my_key)
            # print("Env:", env)

        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from arguments.", exc_info=True)
    else:
        logger.info("No arguments received.")
    return payload


def main():
    logger.info("RestartApp WebJob started.")

    try:
        logger.info("Reading arhuments if any...")
        read_args()
        # success = trigger_restart(config_obj=CONFIG, payload=payload)
        success = restart_webapp(app_name=CONFIG.WA_APP_NAME, resource_group=CONFIG.WA_RG, subscription_id=CONFIG.SUBSCRIPTION_ID)

        if success:
            logger.info("RestartAPP WebJob completed successfully.")
            # For triggered WebJobs, Azure checks the exit code. 0 for success.
            # sys.exit(0) # Not strictly necessary as Python exits with 0 by default on normal completion
        else:
            logger.error("RestartAPP WebJob encountered errors.")
            sys.exit(1)  # Non-zero exit code to indicate failure to Azure

    except Exception:
        logger.exception("An unhandled exception occurred in the RestartAPP WebJob main function")
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
