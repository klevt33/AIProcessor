# File: validate_cosmos_consistency.py

import logging
import time

from cdb import CDB
from config import Config

# --- Setup basic logging for our script ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # --- TEST SETUP ---
    TEST_DOCUMENT_ID = "bug-repro-doc-001"
    TEST_PARTITION_KEY = "bug-repro-pk-001"
    TEST_SYS_NAME = "kirill"  # Use the sys_name from your config/test document

    try:
        config = Config()
        cdb = CDB(config)

        # 1. SETUP: Ensure the document is in the 'pending' state before we start.
        logger.info(f"SETUP: Resetting document '{TEST_DOCUMENT_ID}' to 'pending' status...")
        setup_patch_ops = [{"op": "set", "path": "/sql_writer/status", "value": "pending"}]
        cdb.patch_document(
            cdb.ai_logs_container,
            TEST_DOCUMENT_ID,
            setup_patch_ops,  # Use the positional argument 'operations'
            partition_key=TEST_PARTITION_KEY,
            raise_on_error=True,
        )
        logger.info("SETUP: Document reset successfully.")

        # Give a moment for the setup write to settle, just in case.
        time.sleep(1)

        # 2. ACT 1: Update the status to 'retry_scheduled'.
        logger.info("ACTION: Patching status to 'retry_scheduled'...")
        update_patch_ops = [{"op": "set", "path": "/sql_writer/status", "value": "retry_scheduled"}]
        cdb.patch_document(
            cdb.ai_logs_container,
            TEST_DOCUMENT_ID,
            update_patch_ops,  # Use the positional argument 'operations'
            partition_key=TEST_PARTITION_KEY,
            raise_on_error=True,
        )
        logger.info("ACTION: Patch successful.")

        # 3. ACT 2: Wait for 5 full seconds.
        logger.info("ACTION: Waiting for 5 seconds...")
        time.sleep(5)
        logger.info("ACTION: Wait complete.")

        # 4. ACT 3: Query for 'pending' documents, exactly like the SqlWriterService does.
        logger.info("VERIFICATION: Querying for documents with status = 'pending'...")
        where_condition = f"WHERE c.sql_writer.status = 'pending' AND c.request_details.sys_name = '{TEST_SYS_NAME}'"

        found_docs = cdb.get_documents(container=cdb.ai_logs_container, where_condition=where_condition)

        # 5. VERIFICATION: Analyze the result.
        print("\n" + "=" * 50)
        print("--- TEST RESULTS ---")
        print("=" * 50)

        if not found_docs:
            print("\n--> RESULT: The query returned ZERO documents.")
            print("--> CONCLUSION: The Stale Read theory is DISPROVEN. Cosmos DB is consistent.")
            print("--> NEXT STEP: The root cause is NOT database lag.")
        else:
            print(f"\n--> RESULT: The query returned {len(found_docs)} document(s).")
            print(f"--> Document found: {[doc['id'] for doc in found_docs]}")
            print("--> CONCLUSION: The Stale Read theory is CONFIRMED. Cosmos DB returned stale data after 5 seconds.")
            print("--> NEXT STEP: This is unexpected database behavior.")

    except Exception as e:
        logger.error(f"An error occurred during the test: {e}", exc_info=True)
