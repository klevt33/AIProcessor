import asyncio
import os
import sys

WEBJOB_ID = "1"
WEBJOB_NAME = "ReReviewRCAI"
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


async def update_rc_manual_status(
    sdp,
    num_material_records: int,
    num_non_material_records: int,
    query_for_status: str,
    manual_status_str: str,
    invoice_unique_id_col: str,
):
    """
    Fetch the material and Non-Material records for the RC-AI status.
    Randomly pick the required peprecentage of material and non material records.
    Update the status to DS1-AI for those columns.

    Parameters:
        sdp (SDP): The sdp database object for the DB operations.
        num_material_records (int): The number of material records to be considered for update.
        num_non_material_records (int): The number of non-material records to be considered for update.
        query_for_status (str): The IVCE_LNE_STAT status value of the records to be considered. (should be RC-AI).
        query_for_status (str): The IVCE_LNE_STAT status value to be updated for the sampled records. (should be DS1-AI).
        invoice_unique_id_col (str): The Unique ID to be used to indetify the records in IVCE_DTL table for update.
    """
    import sql_utils

    try:
        # get both queries data in DF
        material_df = await sql_utils.get_material_records(sdp, query_for_status)
        non_material_df = await sql_utils.get_non_material_records(sdp, query_for_status)

        logger.info(f"Total {len(material_df) + len(non_material_df)} {query_for_status} records available.")
        logger.info(f"Materials: {len(material_df)}, Non-Materials: {len(non_material_df)}")

        if len(material_df) > num_material_records:
            # select random sample for the required percentages.
            material_df = material_df.sample(
                n=min(num_material_records, len(material_df)), random_state=None
            )  # make state None to randomize for each day of Web job run.
        elif len(material_df) == 0:
            logger.debug("No material records available to be updated")

        if len(non_material_df) > num_non_material_records:
            non_material_df = non_material_df.sample(n=min(num_non_material_records, len(non_material_df)), random_state=None)
        elif len(non_material_df) == 0:
            logger.debug("No Non material records available to be updated")

        # update the status.
        ref_uuid_list = material_df[invoice_unique_id_col].tolist() + non_material_df[invoice_unique_id_col].tolist()

        if len(ref_uuid_list) > 0:
            logger.info(f"Total {len(ref_uuid_list)} records updated to {manual_status_str}.")
            logger.info(f"Materials: {len(material_df)}, Non-Materials: {len(non_material_df)}")
            await sql_utils.bulk_update_rc_status_with_temptable(sdp, ref_uuid_list, manual_status_str)
        else:
            logger.debug(f"No Records to be updated to {manual_status_str} status")
    except Exception as e:
        logger.error(
            f"Error in running the {WEBJOB_NAME} to update the {query_for_status} records to {manual_status_str}, "
            f"Exiting - {str(e)}",
            exc_info=True,
        )
        raise e


async def main(job_config):
    from sdp import SDP

    sdp = SDP(config=CONFIG)
    try:
        await update_rc_manual_status(
            sdp=sdp,
            num_material_records=job_config["NUM_MATERIAL_RECORDS"],
            num_non_material_records=job_config["NUM_NON_MATERIAL_RECORDS"],
            query_for_status=job_config["RC_STATUS_FOR_QUERY"],
            manual_status_str=job_config["RC_STATUS_TO_UPDATE"],
            invoice_unique_id_col=job_config["IVCE_DETAIL_UNIQUE_ID"],
        )
        sys.exit(0)
    except Exception:
        sys.exit(1)  # error already logged.


def setup_job_config():
    from pathlib import Path

    from constants import LocalFiles
    from utils import load_yaml

    jobs_config_file = Path(__file__).resolve().parent / LocalFiles.JOB_CONFIG_FILE
    job_config = load_yaml(path=jobs_config_file)
    return job_config


if __name__ == "__main__":
    # Load the app files
    setup_env()

    # set logger
    from config import Config
    from logger import get_job_logger, job_id_var

    CONFIG = Config(app_root=APP_ROOT)  # Load application configuration
    print("APP VERSION Running: ", CONFIG.app_version, "WEBJOB VERSION Running: ", WEBJOB_VERSION)

    logger = get_job_logger(name=WEBJOB_NAME, azure_conn_str=CONFIG.APP_INSIGHTS_CONN_STRING)  # Initialize logger
    job_id_var.set(WEBJOB_ID)  # Store the request ID in the context variable

    job_config = setup_job_config()
    asyncio.run(main(job_config))
