# test_cache.py

import asyncio

# import logging # Keep if you log things directly in this script using the shared logger
import time

# --- Import necessary classes for initialization ---
try:
    from config import Config
    from sdp import SDP
except ImportError as e:
    print(f"Error importing Config or SDP: {e}")
    print("Please ensure config.py and sdp.py are in the Python path.")
    exit(1)

# --- Import your decorated function ---
try:
    from matching_utils import read_manufacturer_data
except ImportError as e:
    print(f"Error importing read_manufacturer_data from matching_utils: {e}")
    print("Please ensure matching_utils.py is in the Python path.")
    exit(1)

# --- Import the SHARED logger ---
try:
    from logger import logger, logging  # Use the same logger as matching_utils
except ImportError as e:
    print(f"Error importing logger from logger.py: {e}")
    print("Please ensure logger.py exists and is in the Python path.")
    exit(1)

# --- Configure SHARED Logger Level for the test ---
# Ensure the shared logger will output DEBUG messages for this test run.
# NOTE: Ideally, your logger.py might already configure this based on
# an environment variable or config file. If not, you can set it here.

logger.setLevel(logging.DEBUG)

# Optional: If the shared logger doesn't have a handler configured
# (e.g., in logger.py), you might need to add one here for testing.
# If logger.py *already* adds handlers, DO NOT add another one here
# unless you specifically want duplicate output.
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False  # Prevent potential duplicate logs if root also gets configured elsewhere


async def run_cache_test():
    """Calls the cached function multiple times to observe caching."""
    logger.info("--- Cache Test Starting ---")  # Now using the shared logger

    # --- Initialize SDP ---
    logger.info("Initializing Config and SDP...")
    try:
        config = Config()
        sdp = SDP(config)
        logger.info("SDP initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Config or SDP: {e}", exc_info=True)
        logger.error("Cannot proceed with the test without SDP.")
        return

    # --- First Call ---
    logger.info("Making first call to read_manufacturer_data...")
    start_time = time.monotonic()
    result1 = await read_manufacturer_data(sdp)
    end_time = time.monotonic()
    # Now the debug log from read_manufacturer_data should appear before this line
    logger.info(f"First call finished in {end_time - start_time:.4f} seconds.")
    if result1:
        logger.info(f"First call returned data with {len(result1)} items.")
    else:
        logger.warning("First call returned None or empty.")

    # ... (rest of the calls and verification remain the same, they will use the shared logger implicitly) ...

    await asyncio.sleep(0.1)

    # --- Second Call ---
    logger.info("Making second call to read_manufacturer_data (expecting cache hit)...")
    start_time = time.monotonic()
    result2 = await read_manufacturer_data(sdp)
    end_time = time.monotonic()
    logger.info(f"Second call finished in {end_time - start_time:.4f} seconds.")  # Expect very fast time
    # DEBUG log from read_manufacturer_data should NOT appear here

    await asyncio.sleep(0.1)

    # --- Third Call ---
    logger.info("Making third call to read_manufacturer_data (expecting cache hit)...")
    start_time = time.monotonic()
    result3 = await read_manufacturer_data(sdp)
    end_time = time.monotonic()
    logger.info(f"Third call finished in {end_time - start_time:.4f} seconds.")  # Expect very fast time
    # DEBUG log from read_manufacturer_data should NOT appear here

    # --- Verification ---
    logger.info("--- Verification ---")
    logger.info(f"Result 1 and Result 2 are the same object: {result1 is result2}")
    logger.info(f"Result 1 and Result 3 are the same object: {result1 is result3}")
    logger.info(f"Result 1 and Result 2 have equal content: {result1 == result2}")
    logger.info(f"Result 1 and Result 3 have equal content: {result1 == result3}")

    logger.info("NOTE: To test TTL expiry, wait >1 hour before running again...")
    logger.info("--- Cache Test Finished ---")


if __name__ == "__main__":
    asyncio.run(run_cache_test())
