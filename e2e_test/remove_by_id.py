import asyncio
import logging
import sys

from config_e2e import DB_TABLE_NAME, EXCEL_FILE_PATH, EXCEL_TAB_NAME, PRIMARY_KEY_FIELD, get_ids_from_excel

from config import Config
from sdp import SDP

# --- Configuration ---
BATCH_SIZE = 500  # Adjust as needed
DB_PLACEHOLDER = "?"
# -------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main():
    # Read IDs from Excel
    try:
        id_list = get_ids_from_excel(EXCEL_FILE_PATH, EXCEL_TAB_NAME, PRIMARY_KEY_FIELD)
        if not id_list:
            logger.warning("No IDs found in the Excel file. Exiting.")
            return
        logger.info(f"Loaded {len(id_list)} IDs from '{EXCEL_FILE_PATH}' Sheet '{EXCEL_TAB_NAME}', Column '{PRIMARY_KEY_FIELD}'.")
    except FileNotFoundError:
        logger.error(f"Excel file not found: {EXCEL_FILE_PATH}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading IDs from Excel: {e}")
        sys.exit(1)

    # --- User Confirmation ---
    print("-" * 30)
    print(f"WARNING: This script will attempt to DELETE {len(id_list)} records")
    print(f"from the table '{DB_TABLE_NAME}'")
    print(f"based on IDs in the '{PRIMARY_KEY_FIELD}' column.")
    print("-" * 30)
    confirm = input("Are you sure you want to proceed? (y/n): ")
    if confirm.strip().lower() != "y":
        logger.info("Operation cancelled by user.")
        sys.exit(0)
    # -----------------------

    sdp = None  # Initialize sdp
    total_deleted = 0
    processed_batches = 0
    failed_batches = 0

    try:
        config = Config()
        # Consider if SDP needs context management or explicit closing
        sdp = SDP(config)
        logger.info(f"Attempting deletions in batches of {BATCH_SIZE}...")

        for i in range(0, len(id_list), BATCH_SIZE):
            batch_ids = id_list[i : i + BATCH_SIZE]
            if not batch_ids:  # Should not happen with range step, but safe check
                continue

            processed_batches += 1
            batch_start_num = i + 1
            batch_end_num = i + len(batch_ids)

            # --- Use Parameterized Query ---
            try:
                # 1. Generate placeholders dynamically
                placeholders = ",".join([DB_PLACEHOLDER] * len(batch_ids))

                # 2. Create the query template
                delete_query = f"DELETE FROM {DB_TABLE_NAME} WHERE {PRIMARY_KEY_FIELD} IN ({placeholders})"

                # 3. Execute with parameters (pass batch_ids as a tuple or list)
                await sdp.update_data(delete_query, tuple(batch_ids))

                logger.info(
                    f"Successfully processed batch {processed_batches} (IDs {batch_start_num}-{batch_end_num}). Deleted"
                    f" {len(batch_ids)} potential records."
                )
                # Note: Standard DELETE doesn't usually return the count reliably without specific clauses.
                # We assume the batch size represents the attempted deletions.
                total_deleted += len(batch_ids)

            except Exception as e:
                failed_batches += 1
                logger.error(f"Failed to process batch {processed_batches} (IDs {batch_start_num}-{batch_end_num}): {e}")
                raise
            # -------------------------------

        logger.info("=" * 30)
        logger.info("Deletion process finished.")
        logger.info(f"Total batches processed: {processed_batches}")
        logger.info(f"Successful batches: {processed_batches - failed_batches}")
        logger.info(f"Failed batches: {failed_batches}")
        logger.info(f"Total records potentially deleted (based on batch sizes): {total_deleted}")
        logger.info("=" * 30)

    except Exception as e:
        # Catch unexpected errors during setup or overall loop
        logger.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
