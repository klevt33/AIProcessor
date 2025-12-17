import asyncio
import os
import sys
import uuid

import requests
from requests import Response

WEBJOB_ID = "1"
WEBJOB_NAME = "FinetuneLLM"
WEBJOB_VERSION = "1.0.0"  # Prod_Major.Prod_Minor.Dev_Deploy

VAR_NAME = "WEB_APP_ENV"
AZURE_WWWROOT = "/home/site/wwwroot"
APP_ROOT = ""
ENV = ""


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


def update_finetune_job_id(job_id: str):
    """
    Update the finetuning job id config value to the given id.

    Parameters:
        job_id: the finetuning job id.
    """
    from constants import AzureAppConfig, Environments

    app_config_label = Environments.DEV if ENV != Environments.PROD else Environments.PROD

    context.logger.info(f"Updating finetune job id: {job_id}")

    context.config.set_azure_app_config_value(AzureAppConfig.FINETUNE_JOB_ID, app_config_label, job_id, client_type="AI")


def update_finetune_job_progress(progress_val: bool):
    """
    Update the finetuning progress config value to the given status.

    Parameters:
        progress_val (bool): True/False
    """
    from constants import AzureAppConfig, Environments

    app_config_label = Environments.DEV if ENV != Environments.PROD else Environments.PROD

    context.logger.info(f"Updating finetune job progress: {progress_val}")

    context.config.set_azure_app_config_value(
        AzureAppConfig.IS_LLM_FINE_TUNING_IN_PROGRESS, app_config_label, progress_val, client_type="AI"
    )


def update_llm_deployment_url(llm_deployment_url: str):
    """
    Update the LLm deployment URL in the azure config.
    Call this method after the finetuning LLm is completed and ready to
    use the new deployed LLM.

    Parameters:
        llm_deployment_url (str): the LLm deployment URl to be used.
    """
    from constants import AzureAppConfig, Environments

    app_config_label = Environments.DEV if ENV != Environments.PROD else Environments.PROD
    context.logger.info(
        f"Updating the finetuned llm deployment config to {AzureAppConfig.AOAI_FINETUNED_LLM_API_DEPLOYMENT} "
        f"for {app_config_label} label"
    )
    context.config.set_azure_app_config_value(
        AzureAppConfig.AOAI_FINETUNED_LLM_API_DEPLOYMENT, app_config_label, llm_deployment_url, client_type="AI"
    )
    context.config.refresh_from_azure_app_config()


async def reload_web_app_config():
    """
    This method send APi request to Web App to reload the app configuration.
    Use this method after setting any app configuration to that the new
    configuration will take into effect.
    """
    from finetune_constants import RELOAD_APP_CONFIG_ENDPOINT

    from utils import get_webapp_access_token

    context.logger.info("Reloading web app config")
    access_token = get_webapp_access_token(context.config)
    app_base_url = getattr(context.config, "WEB_APP_BASE_URL", None)
    await call_web_app_api(app_base_url, RELOAD_APP_CONFIG_ENDPOINT, access_token)


async def call_web_app_api(app_base_url, api_endpoint, access_token, retry_count=3) -> Response:
    """
    This method executes the REST API call on the given end point for the given Azure Web app.

    Parameters:
        app_base_url (str): The base URL for the API endpoint. Should not end with a trailing slash.
        api_endpoint (str): The relative path of the API endpoint. Should begin with a leading slash.
        access_token (str): The access token to be used for authentication.
        retry_count (int): Numbers of times to retry the API on failure.

    Returns:
        Response: The response object from the REST API call.

    Raises:
        Exception: If the REST API call fails after all retries, or if App configuration errors occur.
    """
    try:
        api_url = f"{app_base_url}{api_endpoint}"

        api_headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}

        api_payload = {"id": str(uuid.uuid4())}

        api_response = None
        for attempt in range(retry_count):
            api_response = requests.post(api_url, headers=api_headers, json=api_payload)
            if api_response.status_code == 200:
                context.logger.info("Successfully reloaded the Azure APP Configuration!")
                return api_response

            if attempt < (retry_count - 1):
                logger.error(
                    f"Error in making API {api_endpoint} call, error : {api_response} "
                    f"retrying for {retry_count - attempt - 1} more times"
                )
        else:
            # failure and all retries done.
            if api_response:
                raise Exception(
                    f"Error in calling API to reload the app configuration - Status code: {api_response.status_code} - "
                    f"Content: {api_response.content.decode('utf-8', errors='replace')}"
                )
            else:
                raise Exception("Error in calling API to reload the app configuration")

    except Exception as e:
        logger.error(f"Unable to reload App configuration , error - {str(e)}", exc_info=True)
        raise e


def test_reloaded_config(deployment_name: str):
    """Creates a new temp config and ensures it pulls in the new expected deployment name

    Args:
        deployment_name (str): new expected deployment name

    Raises:
        Exception:
    """
    # Test to make sure config actually is reloaded
    from config import Config

    temp_config = Config(app_root=APP_ROOT)
    if temp_config.AOAI_FINETUNED_LLM_API_DEPLOYMENT != deployment_name:
        raise Exception("Error updating deployment name")


async def handle_ft_job_completion(job_id: str):
    """
    Handles the post-processing workflow after a fine-tuning (FT) job has completed.

    This function performs the following steps:
    - Deploys the fine-tuned model to 'dev' environment.
    - Evaluates the deployed model's performance against threshold minimums.
    - Compares the new model's accuracy against the current production model.
    - If accuracy improves:
        - Remove all but the most recent 2 deployments to make room for TPM quota
        - Deploy fine-tuned model to 'prod' environment.
        - Updates the deployment configuration to use the new model.
        - Reloads the web application configuration.
        - Updates training metadata and versioning.
    - If accuracy does not improve or is below threshold:
        - Removes the new model deployment.
        - Reverts training metadata and versioning.

    Args:
        job_id (str): The identifier of the completed fine-tuning job.

    Returns:
        None
    """
    from fine_tuning_eval import FineTunedLLMEvaluator
    from matcher import Matcher

    fine_tune_model_name = context.ft_utils.get_fine_tune_model_name(job_id)

    resource_group = context.config.AOAI_RG
    resource_name = context.config.AOAI_FINETUNED_LLM_RESOURCE_NAME
    subscription_id = context.ft_utils.config.SUBSCRIPTION_ID
    azure_api_version = context.config.AOAI_FINETUNED_LLM_DEPLOYMENT_API_VERSION

    # Deploy model to just dev for evaluation
    try:
        deployment_name = context.ft_utils.deploy_fine_tuned_model(
            fine_tuned_model_name=fine_tune_model_name,
            deploy_model_name_suffix="dev",
            resource_group=context.config.AOAI_RG,
            resource_name=context.config.AOAI_FINETUNED_LLM_RESOURCE_NAME,
            azure_api_version=context.config.AOAI_FINETUNED_LLM_DEPLOYMENT_API_VERSION,
            tpm_capacity=getattr(context.config, "TOKEN_COUNT", None),
        )
        context.ft_utils.deployment_name = deployment_name
    except Exception as e:
        logger.error("Unable to deploy fine-tuned model to dev", exc_info=True)
        raise e
    # Evaluate model
    context.logger.info(f"Finetuning LLM {fine_tune_model_name}, Deployment completed : {context.ft_utils.deployment_name}")

    ft_eval = FineTunedLLMEvaluator(
        config=context.config, logger=logger, matcher=Matcher(context.config), finetuned_llm=context.ft_utils.deployment_name
    )

    try:
        eval_llm_results_df = await ft_eval.evaluate_llm()
    except Exception as e:
        logger.error("Error while evaluating LLM", exc_info=True)
        raise e

    # Calculate accuracy of new model
    llm_accuracy_res = ft_eval.get_accuracy_from_eval_results(eval_llm_results_df)

    # context.logger.info(f"Finetuning LLM {fine_tune_model_name} Evaluation completed \n Accuracy  : {llm_accuracy_res}")

    # Calculate if accuracy hits minimum threshold
    accuracy_accepted = ft_eval.benchmark_result_with_threshold(llm_accuracy_res)

    accuracy_improved = False
    # If accuracy hits minimum threshold
    if accuracy_accepted:
        context.logger.info("Accuracy hit the minimum threshold")
        # Compare with current LLM results.
        current_llm_accuracy = ft_eval.get_accuracy_result_for_llm(ft_eval.config.AOAI_FINETUNED_LLM_API_DEPLOYMENT)

        accuracy_improved = ft_eval.compare_llm_accuracy_result(current_llm_accuracy, llm_accuracy_res)

        # If accuracy is better than current LLM results
        if accuracy_improved:
            context.logger.info("Accuracy is better than the current LLM results")
            # Remove old deployments to make sure there is room for TPM quota, keep only the latest 2
            deployment_names = context.ft_utils.get_all_deployment_names(
                resource_group=resource_group,
                resource_name=resource_name,
                subscription_id=subscription_id,
                azure_api_version=azure_api_version,
            )
            old_deployment_names = context.ft_utils.get_all_old_deployment_names(deployment_names, n=2)
            context.logger.info("Removing old deployments")
            try:
                for deployment_name in old_deployment_names:
                    if not context.config.DELETE_OLD_DEPLOYMENTS_PROD and "prod" in deployment_name:
                        continue
                    if not context.config.DELETE_OLD_DEPLOYMENTS_DEV and "dev" in deployment_name:
                        continue
                    context.ft_utils.remove_finetuned_llm_deployment(
                        resource_group, resource_name, subscription_id, azure_api_version, deployment_name, delete_only_dev=False
                    )
            except Exception as e:
                logger.error("failed to remove old fine-tuned LLM deployments")
                raise e

            # Deploy to prod
            context.logger.info("Deploying to prod")
            try:
                context.ft_utils.deploy_fine_tuned_model(
                    fine_tuned_model_name=fine_tune_model_name,
                    deploy_model_name_suffix="prod",
                    resource_group=context.config.AOAI_RG,
                    resource_name=context.config.AOAI_FINETUNED_LLM_RESOURCE_NAME,
                    azure_api_version=context.config.AOAI_FINETUNED_LLM_DEPLOYMENT_API_VERSION,
                    tpm_capacity=getattr(context.config, "TOKEN_COUNT", None),
                )
            except Exception as e:
                logger.error("Failed to deploy fine-tuned model to prod")
                raise e

            context.logger.info(
                f"Finetuning LLM {fine_tune_model_name} Accuracy Improved. Updating current deployment with new model"
            )
            update_llm_deployment_url(context.ft_utils.deployment_name)
            await reload_web_app_config()
            test_reloaded_config(context.ft_utils.deployment_name)

            training_data = await context.data_loader.get_in_progress_training_data()
            next_training_data_ver = await context.data_loader.get_next_training_version()

            await context.data_loader.update_training_version(
                ref_row_ids=training_data["training_table_ids"],
                training_version_name=next_training_data_ver,
                bulk_update=True,
                batch_size=2000,
            )

    context.logger.info(f"Evaluation done : accuracy_improved:{accuracy_improved}, accuracy_accepted:{accuracy_accepted}")

    # If model accuracy is unacceptable
    if (not accuracy_accepted) or (not accuracy_improved):
        context.logger.info(
            f"Finetuning LLM {fine_tune_model_name} Accuracy Not improved. NOT using the new model for deployment."
        )

        # Remove model deployment
        context.ft_utils.remove_finetuned_llm_deployment(
            resource_group=context.ft_utils.config.AOAI_RG,
            resource_name=context.ft_utils.config.AOAI_FINETUNED_LLM_RESOURCE_NAME,
            subscription_id=context.ft_utils.config.SUBSCRIPTION_ID,
            azure_api_version=context.ft_utils.config.AOAI_FINETUNED_LLM_DEPLOYMENT_API_VERSION,
            deployment_name=context.ft_utils.deployment_name,
            delete_only_dev=False,
            wait_for_delete_complete=True,
        )
        await reset_training_data()
    else:
        pass


async def reset_training_data():
    """
    Resets the status of in-progress training data by moving its version back to 'NEW'.

    This function performs the following steps:
    1. Retrieves training data currently marked as 'in progress'.
    2. Updates the training version of the retrieved data to 'NEW',
       effectively resetting its status.

    The update is performed in batches of 20,000 records without bulk update mode.

    Returns:
        None
    """
    from constants import TrainingDataVersions

    context.logger.info("Resetting training data")

    training_data = await context.data_loader.get_in_progress_training_data()
    # move the training version to back to 'NEW'
    await context.data_loader.update_training_version(
        ref_row_ids=training_data["training_table_ids"],
        training_version_name=TrainingDataVersions.NEW,
        bulk_update=True,
        batch_size=2000,
    )


async def main():
    """
    Entry point for the FinetuneLLM WebJob.

    This function orchestrates the fine-tuning workflow for a language model. It performs the following:

    - Checks if a fine-tuning job is already in progress.
        - If so, retrieves the job ID and checks its status.
            - If the job succeeded, it triggers post-processing via `handle_ft_job_completion`.
            - If the job failed or was cancelled, it resets the training state and marks the job as not in progress.
            - If the job is still running, it exits early to avoid duplication.
        - If no job is in progress, it starts a new fine-tuning job and marks it as in progress.

    Exception handling is included to log and exit gracefully in case of unexpected errors.

    Returns:
        None
    """
    from data_loader import DataLoader
    from fine_tuning_state import FinetuneJobState
    from fine_tuning_utils import FineTuningUtils

    # Setup context
    context.data_loader = DataLoader(config=context.config, logger=logger)
    context.ft_utils = FineTuningUtils(config=context.config, logger=logger, data_loader=context.data_loader)
    context.ft_job_state = FinetuneJobState(config=context.config, logger=logger, ft_utils=context.ft_utils, env=ENV)

    try:
        ft_job_state = context.ft_job_state
        await ft_job_state.load()

        # If fine-tune is already in progress
        if ft_job_state.in_progress:
            # If the job was successful
            if ft_job_state.is_successful():
                context.logger.info("Job was successful.")
                await handle_ft_job_completion(ft_job_state.job_id)
            # If the job was unsuccessful
            elif ft_job_state.is_unsuccessful():
                logger.warning("Finetuning was unsuccessful.")
                await reset_training_data()
            # If job status is still running
            elif ft_job_state.is_running():
                context.logger.info("Job is still running.")
                # Exit and let fine-tune job finish
                return
            update_finetune_job_id("")
            update_finetune_job_progress(False)
        else:
            # Start a new fine-tune job
            job_id = await context.ft_utils.finetune()
            update_finetune_job_id(job_id)
            update_finetune_job_progress(True)
            return

    except Exception:
        logger.exception("An unhandled exception occurred in the FinetuneLLM WebJob main function")
        sys.exit(2)  # Different non-zero exit code for job infrastructure failure


def setup_job_config():
    from pathlib import Path

    from constants import LocalFiles
    from utils import load_yaml

    jobs_config_file = Path(__file__).resolve().parent / LocalFiles.JOB_CONFIG_FILE
    job_config_yaml = load_yaml(path=str(jobs_config_file))
    print(f"Job config yaml: {job_config_yaml}")

    from app_context import context

    from config import Config

    config = Config(app_root=APP_ROOT)  # Load application configuration
    context.config = config
    context.config._locked = False
    context.config.MIN_DESCRIPTIONS_FOR_TRAINING = job_config_yaml["MIN_DESCRIPTIONS_FOR_TRAINING"]
    context.config.MAX_TRAINING_FILE_SIZE = job_config_yaml["MAX_TRAINING_FILE_SIZE"]
    context.config.TRAINING_DATA_TRANSFER_TYPE = job_config_yaml["TRAINING_DATA_TRANSFER_TYPE"]
    context.config.DATA_VERSION_TO_TRAIN = job_config_yaml["DATA_VERSION_TO_TRAIN"]
    context.config.TOKEN_COUNT = job_config_yaml["TOKEN_COUNT"]
    context.config.WEB_APP_BASE_URL = job_config_yaml["WEB_APP_BASE_URL"]
    context.config.CONFIDENCE_THRESHOLDS = job_config_yaml["CONFIDENCE_THRESHOLDS"]
    context.config.MFR_NAME_CONFIDENCE_THRESHOLD = job_config_yaml["CONFIDENCE_THRESHOLDS"]["MFR_NAME"]
    context.config.MFR_PN_CONFIDENCE_THRESHOLD = job_config_yaml["CONFIDENCE_THRESHOLDS"]["MFR_PN"]
    context.config.MFR_UNSPSC_CODE_CONFIDENCE_THRESHOLD = job_config_yaml["CONFIDENCE_THRESHOLDS"]["UNSPSC_CODE"]
    context.config.ACCURACY_THRESHOLDS = job_config_yaml["ACCURACY_THRESHOLDS"]
    context.config.MFR_NAME_SKIP_VALUES = job_config_yaml["MFR_NAME_SKIP_VALUES"]
    context.config.TEST_COLS_TO_KEEP = job_config_yaml["TEST_COLS_TO_KEEP"]
    context.config.FIELD_NAMES = job_config_yaml["FIELD_NAMES"]
    context.config.TEST_DATA_FILE = job_config_yaml["TEST_DATA_FILE"]
    context.config.TEST_RESULTS_FOLDER = job_config_yaml["TEST_RESULTS_FOLDER"]
    context.config.DELETE_OLD_PROD_DEPLOYMENTS = job_config_yaml["DELETE_OLD_DEPLOYMENTS"]["PROD"]
    context.config.DELETE_OLD_DEV_DEPLOYMENTS = job_config_yaml["DELETE_OLD_DEPLOYMENTS"]["DEV"]

    context.config._locked = True


if __name__ == "__main__":
    # Load the app files
    setup_env()

    setup_job_config()

    from app_context import context

    # set logger
    from logger import get_job_logger, job_id_var

    print("APP VERSION Running: ", context.config.app_version, "WEBJOB VERSION Running: ", WEBJOB_VERSION)

    logger = get_job_logger(name=WEBJOB_NAME, azure_conn_str=context.config.APP_INSIGHTS_CONN_STRING)  # Initialize logger
    context.logger = logger
    job_id_var.set(WEBJOB_ID)  # Store the request ID in the context variable

    asyncio.run(main())
