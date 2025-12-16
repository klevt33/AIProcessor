# test_read_manufacturer_data_unittest.py

import asyncio
import threading
import time
import unittest
from unittest.mock import AsyncMock, patch

import pandas as pd

# To make this test file self-contained, we'll define the necessary components here.
# In your actual project, these would be imported from their respective modules.


# --- Component 1: The Custom Dictionary ---
class SpecialCharIgnoringDict(dict):
    _chars_to_remove = "-/\\_. "
    _translation_table = str.maketrans("", "", _chars_to_remove)

    @staticmethod
    def _normalize_key(key):
        if not isinstance(key, str):
            return key
        return key.upper().translate(SpecialCharIgnoringDict._translation_table)

    def __init__(self, *args, **kwargs):
        super().__init__()
        initial_data = dict(*args, **kwargs)
        for key, value in initial_data.items():
            self[key] = value

    def __setitem__(self, key, value):
        super().__setitem__(self._normalize_key(key), value)

    def __getitem__(self, key):
        return super().__getitem__(self._normalize_key(key))

    def __contains__(self, key):
        return super().__contains__(self._normalize_key(key))

    def copy(self):
        return type(self)(self)


# --- Component 2: The Function to be Tested (a simplified skeleton) ---
class SDP:
    pass


_manufacturer_data_cache = {"data": None, "expiry_time": 0, "ttl_seconds": 3600}
_manufacturer_data_populate_lock = threading.Lock()


async def get_manufacturer_data(sdp: SDP) -> pd.DataFrame:
    """A placeholder for the real database function. This will be mocked."""
    raise NotImplementedError("This function should be mocked in tests.")


async def read_manufacturer_data(sdp: SDP) -> SpecialCharIgnoringDict:
    """The actual function we are testing."""
    current_time = time.monotonic()
    if _manufacturer_data_cache["data"] is not None and current_time < _manufacturer_data_cache["expiry_time"]:
        return _manufacturer_data_cache["data"]

    if _manufacturer_data_populate_lock.acquire(blocking=True, timeout=1.0):
        try:
            current_time = time.monotonic()
            if _manufacturer_data_cache["data"] is not None and current_time < _manufacturer_data_cache["expiry_time"]:
                return _manufacturer_data_cache["data"]

            result_df = await get_manufacturer_data(sdp)
            result_df = result_df.dropna(subset=["UncleanName", "CleanName"]).copy()

            def normalize(name):
                return name.strip().upper()

            result_df["UncleanName"] = result_df["UncleanName"].map(normalize)
            result_df["CleanName"] = result_df["CleanName"].map(normalize)
            result_df = result_df[(result_df["CleanName"] != "") & (result_df["UncleanName"] != "")]
            raw_dict = dict(zip(result_df["UncleanName"], result_df["CleanName"]))
            raw_dict.update(dict(zip(result_df["CleanName"], result_df["CleanName"])))
            manufacturer_dict = SpecialCharIgnoringDict(raw_dict)
            _manufacturer_data_cache["data"] = manufacturer_dict
            _manufacturer_data_cache["expiry_time"] = current_time + _manufacturer_data_cache["ttl_seconds"]
            return manufacturer_dict
        finally:
            _manufacturer_data_populate_lock.release()
    return SpecialCharIgnoringDict()


# --- The Test Suite using unittest ---

# Mock data remains the same
MOCK_DB_DATA = pd.DataFrame(
    {
        "UncleanName": ["ABC Company", "XYZ Corp.", "Mega Corp", "DropThis", "Filtered Out"],
        "CleanName": ["ABC INC", "XYZ CORP", "MEGA CORP", None, ""],
    }
)


class TestReadManufacturerData(unittest.TestCase):

    def setUp(self):
        """
        This method is called before each test. It's the unittest equivalent
        of a pytest fixture for setup.
        """
        _manufacturer_data_cache["data"] = None
        _manufacturer_data_cache["expiry_time"] = 0
        if _manufacturer_data_populate_lock.locked():
            _manufacturer_data_populate_lock.release()

    @patch("__main__.get_manufacturer_data", new_callable=AsyncMock)
    def test_full_logic(self, mock_get_db_data):
        """
        Tests the full logic using unittest and asyncio.run().
        """

        # We define an async inner function to run our test logic
        async def run_test():
            # Arrange: Configure the mock
            mock_get_db_data.return_value = MOCK_DB_DATA.copy()

            # --- Act 1 & Assert 1: First call to populate cache ---
            manufacturer_map = await read_manufacturer_data(sdp=None)

            # Assertions using unittest methods
            self.assertIsInstance(manufacturer_map, SpecialCharIgnoringDict)
            self.assertEqual(manufacturer_map["ABC-Company"], "ABC INC")
            self.assertEqual(manufacturer_map["xyz / corp"], "XYZ CORP")
            self.assertEqual(manufacturer_map["MEGA CORP"], "MEGA CORP")

            # Test that invalid data was filtered
            with self.assertRaises(KeyError):
                _ = manufacturer_map["DropThis"]
            with self.assertRaises(KeyError):
                _ = manufacturer_map["Filtered Out"]

            # --- Act 2 & Assert 2: Second call to test caching ---
            map_from_cache = await read_manufacturer_data(sdp=None)

            # Assert that the mock was only called once
            mock_get_db_data.assert_called_once()

            # Assert the cached object is the same instance
            self.assertIs(map_from_cache, manufacturer_map)

        # Run the async test logic using asyncio.run()
        asyncio.run(run_test())


# This makes the script runnable from the command line
if __name__ == "__main__":
    unittest.main()
