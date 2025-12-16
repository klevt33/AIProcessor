import asyncio
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

WEBJOB_ID = "1"
WEBJOB_NAME = "ResetStalledJobs"
WEBJOB_VERSION = "1.0.0"  # Prod_Major.Prod_Minor.Dev_Deploy

VAR_NAME = "WEB_APP_ENV"
AZURE_WWWROOT = "/home/site/wwwroot"
APP_ROOT = ""
TIME_STALLED_THRESHOLD = 30  # should be in minutes

MAX_TO_RESTART = 500


class MissingDetailException(Exception):
    pass


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


async def retreive_stalled_jobs(number_of_hours=24, line_statuses=("AI", "AI_ERROR")) -> list[int]:
    from sdp import SDP

    sdp = SDP(config=CONFIG)
    query = f"""
        SELECT
            IVCE_DTL_UID, REC_UPDD_DTTM, IVCE_LNE_STAT
        FROM
            RPAO.IVCE_DTL
        WHERE
            REC_UPDD_DTTM <= DATEADD(HOUR, -{number_of_hours}, GETDATE())  -- X hours ago
            AND IVCE_LNE_STAT IN ('{"', '".join(line_statuses)}')
        ORDER BY
            REC_UPDD_DTTM DESC;
    """
    df = await sdp.fetch_data(query)
    return df["IVCE_DTL_UID"].tolist()


async def reset_job_id(cdb, ivce_dtl_id: str):
    where_cond = f" WHERE c.id like '%~{ivce_dtl_id}' and c.request_details.sys_name = 'web' "
    order_by_cond = " ORDER BY c.created_at DESC "
    col_filter = " c.id "
    job = cdb.get_documents(
        cdb.ai_jobs_container, column_filter=col_filter, where_condition=where_cond, order_condition=order_by_cond
    )
    if job:
        ivce_dtl_id = job[0]["id"]

        # Build patch ops dynamically
        patch_ops = [{"op": "replace", "path": "/status", "value": "pending"}]

        now = datetime.now(ZoneInfo("America/Chicago"))
        patch_ops.append({"op": "set", "path": "/updated_at", "value": str(now)})

        cdb.patch_document(cdb.ai_jobs_container, ivce_dtl_id, patch_ops, partition_key=ivce_dtl_id)
    else:
        raise MissingDetailException("Not found in cosmos.")


async def main():
    logger.info("ResetAIJobs WebJob started.")
    stalled_dtl_ids = await retreive_stalled_jobs(number_of_hours=CONFIG.DELAY_IN_HOURS, line_statuses=CONFIG.LINE_STATUSES)
    logger.info(f"Found {len(stalled_dtl_ids)} jobs to retry")
    cdb = CDB(config=CONFIG)
    missing_dtl_ids = []
    successful_dtl_ids = []
    for dtl_id in stalled_dtl_ids:
        if len(successful_dtl_ids) >= MAX_TO_RESTART:
            break
        try:
            await reset_job_id(cdb, dtl_id)
            successful_dtl_ids.append(dtl_id)
        except MissingDetailException as e:
            logger.debug(f"❌ Failed {dtl_id}: {e}")
            missing_dtl_ids.append(dtl_id)
    logger.info(f"Failed detail ids: {missing_dtl_ids}")
    logger.info(f"Retried detail ids: {successful_dtl_ids}")


def setup_job_config(config):
    from pathlib import Path

    from constants import LocalFiles
    from utils import load_yaml

    jobs_config_file = Path(__file__).resolve().parent / LocalFiles.JOB_CONFIG_FILE
    job_config_yaml = load_yaml(path=str(jobs_config_file))
    print(f"Job config yaml: {job_config_yaml}")

    config._locked = False
    config.LINE_STATUSES = job_config_yaml["LINE_STATUSES"]
    config.DELAY_IN_HOURS = job_config_yaml["DELAY_IN_HOURS"]

    config._locked = True


if __name__ == "__main__":
    # Load the app files
    setup_env()

    # set logger
    from config import Config
    from logger import get_job_logger, job_id_var

    CONFIG = Config(app_root=APP_ROOT)  # Load application configuration
    setup_job_config(CONFIG)
    from cdb import CDB

    print("APP VERSION Running: ", CONFIG.app_version, "WEBJOB VERSION Running: ", WEBJOB_VERSION)

    logger = get_job_logger(name=WEBJOB_NAME, azure_conn_str=CONFIG.APP_INSIGHTS_CONN_STRING)  # Initialize logger
    job_id_var.set(WEBJOB_ID)  # Store the request ID in the context variable

    asyncio.run(main())
