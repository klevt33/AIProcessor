import asyncio

import pandas as pd

from config import Config
from sdp import SDP
from sql_utils import get_parts_data


async def test_get_parts_data():
    # Initialize configuration and SDP
    config = Config()
    sdp = SDP(config)

    # List of existing part numbers for testing
    existing_part_numbers = [
        "UA6HNB",
        "UA9CHB",
        "TZL1NL9610000LMFSTMV0LT40K80CR1WH",
        "31-200",
        "8075",
        "STTTB1-G",
        "WMSL-A-Z",
        "ZHUABLK1100FT",
        "WF4LED30K40K50K90CR1MBM6",
        "4Q-50T",
    ]

    # Add some non-existing part numbers for negative testing
    all_part_numbers = existing_part_numbers + ["NONEXIST123", "FAKE456", "NOTREAL789"]

    # Test with active_only=False (default)
    print("\n=== Testing with active_only=False ===")
    df_all = await get_parts_data(sdp, all_part_numbers)
    print(f"Total records returned: {len(df_all)}")
    if not df_all.empty:
        print("\nSample data (first 5 rows):")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print(df_all.head(5))

        # Print counts by part number
        print("\nRecords by part number:")
        part_counts = df_all["MfrPartNum"].value_counts()
        for part, count in part_counts.items():
            print(f"{part}: {count} records")
    else:
        print("No records found!")

    # Test with active_only=True
    print("\n=== Testing with active_only=True ===")
    df_active = await get_parts_data(sdp, all_part_numbers, active_only=True)
    print(f"Total active records returned: {len(df_active)}")
    if not df_active.empty:
        print("\nSample data (first 5 rows):")
        print(df_active.head(5))

        # Print counts by part number
        print("\nActive records by part number:")
        part_counts = df_active["MfrPartNum"].value_counts()
        for part, count in part_counts.items():
            print(f"{part}: {count} records")
    else:
        print("No active records found!")

    # Test with a single part number
    single_part = "UA6HNB"
    print(f"\n=== Testing with single part number: {single_part} ===")
    df_single = await get_parts_data(sdp, [single_part])
    print(f"Records for {single_part}: {len(df_single)}")
    if not df_single.empty:
        print("\nFull data for single part:")
        print(df_single)
    else:
        print(f"No records found for {single_part}!")

    # Test with only non-existing part numbers
    non_existing = ["NONEXIST123", "FAKE456", "NOTREAL789"]
    print("\n=== Testing with only non-existing part numbers ===")
    df_non_exist = await get_parts_data(sdp, non_existing)
    print(f"Records for non-existing parts: {len(df_non_exist)}")
    if not df_non_exist.empty:
        print("\nData for non-existing parts (should be empty):")
        print(df_non_exist)
    else:
        print("No records found for non-existing parts (expected result)")


async def run_tests():
    await test_get_parts_data()


def main():
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()
