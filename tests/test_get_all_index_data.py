import asyncio

from config import Config
from sdp import SDP
from sql_utils import get_all_index_data


async def test_get_all_index_data():
    # Initialize SDP
    config = Config()
    sdp = SDP(config)

    # Set batch size
    batch_size = 1000

    print("Starting data retrieval test...")

    # Call the function
    data_generator, total_records = await get_all_index_data(sdp, batch_size=batch_size)

    print(f"Total records before deduplication: {total_records}")

    # Process batches
    batch_count = 0
    total_deduplicated_records = 0

    for batch_index, batch_df in enumerate(data_generator):
        batch_count += 1
        records_in_batch = len(batch_df)
        total_deduplicated_records += records_in_batch

        # Print information about each batch
        print(f"Batch {batch_index + 1}: {records_in_batch} records (offset: {batch_index * batch_size})")

        # Print sample records from first and second batches
        if batch_index == 0:
            print("\nSample records from first batch:")
            print(batch_df.head(3).to_string())
        elif batch_index == 1:
            print("\nSample records from second batch:")
            print(batch_df.head(3).to_string())

    # Print summary
    print("\nSummary:")
    print(f"Total batches processed: {batch_count}")
    print(f"Total records before deduplication: {total_records}")
    print(f"Total deduplicated records: {total_deduplicated_records}")
    print(f"Duplicate records removed: {total_records - total_deduplicated_records}")
    print(f"Deduplication rate: {((total_records - total_deduplicated_records) / total_records * 100):.2f}%")


# Run the test
if __name__ == "__main__":
    asyncio.run(test_get_all_index_data())
