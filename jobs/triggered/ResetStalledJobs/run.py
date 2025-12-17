import asyncio
import os
import sys
from copy import copy
from datetime import timedelta

WEBJOB_ID = "1"
WEBJOB_NAME = "ResetStalledJobs"
WEBJOB_VERSION = "1.0.0"  # Prod_Major.Prod_Minor.Dev_Deploy

VAR_NAME = "WEB_APP_ENV"
AZURE_WWWROOT = "/home/site/wwwroot"
APP_ROOT = ""
TIME_STALLED_THRESHOLD = 30  # should be in minutes


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


async def handle_stalled_jobs(config_obj):
    try:
        from cdb import CDB
        from constants import DataStates
        from sdp import SDP
        from sql_utils import bulk_update_invoice_line_status
        from utils import get_current_datetime_cst

        now = get_current_datetime_cst()
        # Example: Skip run if current time is 12:00 AM
        if now.hour < 1 or now.hour > 23:
            print("Skipping midnight run due to maintenance window.")
            return

        cdb = CDB(config=config_obj)
        sdp = SDP(config=config_obj)

        global TIME_STALLED_THRESHOLD
        if not isinstance(TIME_STALLED_THRESHOLD, int):
            TIME_STALLED_THRESHOLD = 30

        threshold = get_current_datetime_cst() - timedelta(minutes=TIME_STALLED_THRESHOLD)
        threshold_iso = threshold.isoformat()

        # column_filter = " c.id, c.status, c.created_at, c.updated_at"
        column_filter = " * "
        where_condition = f" WHERE c.status IN ('queued', 'processing') AND c.updated_at < '{threshold_iso}'"
        order_condition = " ORDER BY c.created_at ASC"

        total_stalled_jobs = cdb.get_documents_count(container=cdb.ai_jobs_container, where_condition=where_condition)
        logger.info(f"Found {total_stalled_jobs} stalled jobs where 'updated_at' < '{threshold_iso}'")

        remaining_stalled_jobs = copy(total_stalled_jobs)

        while remaining_stalled_jobs > 0:
            stalled_jobs = cdb.get_documents(
                top=50,
                column_filter=column_filter,
                container=cdb.ai_jobs_container,
                where_condition=where_condition,
                order_condition=order_condition,
            )

            # ids = [item["id"] for item in stalled_jobs]
            # print(f"Found {len(ids)} stalled jobs: {ids}")

            # Set each one back to pending
            for doc in stalled_jobs:
                # Remove system fields before upserting
                for key in ["_etag", "_rid", "_self", "_attachments", "_ts"]:
                    doc.pop(key, None)

                reset_log = {"stuck_status": doc["status"], "reset_at": doc["updated_at"]}
                doc["updated_at"] = get_current_datetime_cst()
                if "reset_details" in doc:
                    doc["reset_details"].update({str(len(doc["reset_details"]) + 1): reset_log})
                else:
                    doc["reset_details"] = {"1": reset_log}
                doc["status"] = "pending"

                # Drop existing log id to avoid duplicate id error
                log_id = cdb.convert_job_id_to_log_id(doc["id"])

                try:
                    cdb.ai_logs_container.delete_item(item=log_id, partition_key=log_id)
                except Exception as e:
                    logger.debug(f"Error occurred while deleting log record with Job {doc["id"]} and Log {log_id}: {str(e)}")

                try:
                    cdb.update_document(container=cdb.ai_jobs_container, document=doc)
                except Exception as e:
                    logger.debug(f"Error occurred while updating record with Job {doc["id"]} and Log {log_id}: {str(e)}")

            # Prepare detail IDs list
            details_ids = [sj["id"].split("~")[-1] for sj in stalled_jobs]

            # Bulk set them to 'AI' in IVCE_DTL table
            await bulk_update_invoice_line_status(sdp=sdp, invoice_detail_ids=details_ids, new_status=DataStates.AI)

            remaining_stalled_jobs -= len(stalled_jobs)
            logger.info(f"Requeued {len(stalled_jobs)} jobs. {remaining_stalled_jobs}/{total_stalled_jobs} jobs left.")
        else:
            logger.info("No stalled jobs present to requeue.")

        return True

    except Exception:
        logger.exception("Unhandled exception during handling stalled jobs")  # logger.exception includes stack trace
        return False


async def main():
    logger.info("ResetStalledJobs WebJob started.")

    try:
        logger.info("Calling stalled jobs handler.")
        success = await handle_stalled_jobs(config_obj=CONFIG)

        if success:
            logger.info("ResetStalledJobs WebJob completed successfully.")
            # For triggered WebJobs, Azure checks the exit code. 0 for success.
            # sys.exit(0) # Not strictly necessary as Python exits with 0 by default on normal completion
        else:
            logger.error("ResetStalledJobs WebJob encountered errors.")
            sys.exit(1)  # Non-zero exit code to indicate failure to Azure

    except Exception:
        logger.exception("An unhandled exception occurred in the ReindexDaily WebJob main function")
        sys.exit(2)  # Different non-zero exit code for job infrastructure failure


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

    asyncio.run(main())
