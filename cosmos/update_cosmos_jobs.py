"""
Reads all records from ai_process_logs container where status = "error".

For each error log id, converts the log id format (with -) into the corresponding job id format (with ~).

Example:

Log ID → a98acf77-a23a-4074-96c7-4682a8875ed7-296107

Job ID → a98acf77-a23a-4074-96c7-4682a8875ed7~296107

Updates the corresponding record in ai_jobs container to set its status = "pending".
"""

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from azure.cosmos import CosmosClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Cosmos connection details ---
COSMOS_ENDPOINT = "https://aks-cdb-prod.documents.azure.com:443/"
COSMOS_KEY = "<REDACTED>"
DATABASE_NAME = "spend_report_prod"
LOGS_CONTAINER_NAME = "ai_process_logs"
JOBS_CONTAINER_NAME = "ai_jobs"

DELETE_LOGS = True  # ✅ Set to True if you want to delete existing logs


def log_id_to_job_id(log_id: str) -> str:
    """Convert log id (with '-') into job id (with '~')."""
    prefix, suffix = log_id.rsplit("-", 1)
    return f"{prefix}~{suffix}"


def hold_items_in_queue():
    client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    db = client.get_database_client(DATABASE_NAME)
    jobs_container = db.get_container_client(JOBS_CONTAINER_NAME)

    query = (
        # "SELECT c.id FROM c WHERE c.status in ('pending', 'queued', 'processing') AND c.created_at > '2025-10-26T13:00:00-05:00'"
        "SELECT c.id FROM c WHERE c.status in ('hold') AND c.created_at > '2025-10-26T13:00:00-05:00'"
    )
    query = "SELECT * FROM c WHERE c.status = 'error' and c.created_at >'2025-10-16 10:40:11.00-05:00'"
    pending_jobs = jobs_container.query_items(query=query, enable_cross_partition_query=True)
    pending_jobs = list(pending_jobs)
    print(f"Pending items count {len(pending_jobs)}")

    count = 0
    for job in pending_jobs:
        if count > 0 and count % 200 == 0:
            print(f"Changed status of {count} jobs done!")
            # break

        job_id = job["id"]

        try:
            job_doc = jobs_container.read_item(item=job_id, partition_key=job_id)

            # Build patch ops dynamically
            patch_ops = [{"op": "replace", "path": "/status", "value": "pending"}]

            now = datetime.now(ZoneInfo("America/Chicago"))
            patch_ops.append({"op": "set", "path": "/updated_at", "value": str(now)})
            # patch_ops.append({"op": "replace", "path": "/created_at", "value": str(now)})

            if "message" in job_doc:
                patch_ops.append({"op": "remove", "path": "/message"})
            if "reset_details" in job_doc:
                patch_ops.append({"op": "remove", "path": "/reset_details"})

            jobs_container.patch_item(
                item=job_id, partition_key=job_id, patch_operations=patch_ops  # adjust if your PK is not /id
            )
            # print(f"Updated {job_id} → hold")

            count += 1
        except Exception as e:
            print(f"❌ Failed {job_id}: {e}")

    print(f"✅ Finished reprocessing {count} jobs")


def reprocess_failed_jobs():
    client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    db = client.get_database_client(DATABASE_NAME)
    logs_container = db.get_container_client(LOGS_CONTAINER_NAME)
    jobs_container = db.get_container_client(JOBS_CONTAINER_NAME)

    # query = "SELECT * FROM c WHERE c.response_details.status = 'error' and \
    #     c.request_details.start_time >'2025-10-26 13:00:00.00-05:00'"
    # query = "SELECT c.id FROM c WHERE c.response_details.status = 'error'"
    # query = (
    #     "SELECT c.id FROM c WHERE c.process_output.invoice_line_status='AI_ERROR'         and c.request_details.start_time"
    #     " >'2025-10-26 13:00:00.00-05:00'"
    # )
    query = (
        r"SELECT c.id FROM c where c.response_details.status='error' and c.request_details.start_time >'2025-10-16"
        r" 10:40:11.460915-05:00' and c.response_details.message like '%deadlocked%' order by c.request_details.start_time desc"
    )
    # query = "select value c.id from c where c.status in ('pending') and c.created_at>'2025-10-26 13:00:00.00-05:00'"
    error_logs = logs_container.query_items(query=query, enable_cross_partition_query=True)

    count = 0
    for log in error_logs:
        if count == 500:
            print("Reset of 500 jobs done!")
            break

        log_id = log["id"]
        job_id = log_id_to_job_id(log_id)

        try:
            job_doc = jobs_container.read_item(item=job_id, partition_key=job_id)

            # Build patch ops dynamically
            patch_ops = [{"op": "replace", "path": "/status", "value": "pending"}]

            now = datetime.now(ZoneInfo("America/Chicago"))
            patch_ops.append({"op": "replace", "path": "/updated_at", "value": str(now)})
            patch_ops.append({"op": "replace", "path": "/created_at", "value": str(now)})

            if "message" in job_doc:
                patch_ops.append({"op": "remove", "path": "/message"})
            if "reset_details" in job_doc:
                patch_ops.append({"op": "remove", "path": "/reset_details"})

            jobs_container.patch_item(
                item=job_id, partition_key=job_id, patch_operations=patch_ops  # adjust if your PK is not /id
            )
            print(f"Updated {job_id} → pending")

            # Step 3: Delete log only if DELETE_LOGS flag is True
            if DELETE_LOGS:
                logs_container.delete_item(item=log_id, partition_key=job_id.split("~")[0])
                print(f"Job {job_id} → pending | Log {log_id} deleted")

            count += 1
        except Exception as e:
            print(f"❌ Failed {job_id}: {e}")

    print(f"✅ Finished reprocessing {count} jobs")


if __name__ == "__main__":
    reprocess_failed_jobs()

    # hold_items_in_queue()
