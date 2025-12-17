import asyncio
import io
import json
import re
import time
from io import BytesIO
from logging import Logger
from typing import Callable

from pandas import DataFrame, Series

from config import Config
from constants import TrainingDataVersions
from utils import get_current_datetime_cst


class DataLoader:

    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger

        from sdp import SDP

        self.sdp = SDP(config=config)

    async def get_training_data_for_fine_tuning(
        self, formatter_fn: Callable[[Series], dict[str, list[dict[str, str]]]]
    ) -> dict[str, list | str | BytesIO | None]:
        """
        Fetches and formats training data for fine-tuning, using either file or buffer mode.

        Returns:
            dict: {
                'data': file name or BytesIO buffer,
                'training_table_ids': list of processed row IDs
            } or None if insufficient data.
        """
        try:
            self.logger.info("Getting training data for fine-tuning")
            min_length: int = self.config.MIN_DESCRIPTIONS_FOR_TRAINING
            max_size_in_mb: int = self.config.MAX_TRAINING_FILE_SIZE
            data_version: str = self.config.DATA_VERSION_TO_TRAIN
            transfer_type: str = self.config.TRAINING_DATA_TRANSFER_TYPE.lower()

            from sql_utils import get_training_data

            training_data_df: DataFrame = await get_training_data(sdp=self.sdp, training_data_version=data_version)

            if training_data_df is None or training_data_df.empty:
                self.logger.info(f"No records found for training version '{data_version}'.")
                raise Exception(f"No records found for training version '{data_version}'.")

            total_records = len(training_data_df)
            self.logger.info(f"Fetched {total_records} training records for version '{data_version}'.")

            if total_records < min_length:
                self.logger.info(f"Only {total_records} records available; minimum required is {min_length}. Aborting.")
                raise Exception(f"Only {total_records} records available; minimum required is {min_length}. Aborting.")

            file_name = f"training_data_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

            # Format all rows
            all_encoded_lines = []
            all_ids = []

            # Shuffle data, that way it won't retry the same data every day if it fails previous day but hit max data
            training_data_df = training_data_df.sample(frac=1).reset_index(drop=True)

            for _, row in training_data_df.iterrows():
                line = f"{json.dumps(formatter_fn(row))}\n".encode("utf-8")
                all_encoded_lines.append(line)
                all_ids.append(row["IVCE_XCTN_LLM_TRNL_PRDT_REF_UID"])

            from utils import find_max_fitting_index

            max_size_in_bytes = int(max_size_in_mb) * 1024 * 1024
            max_index = find_max_fitting_index(all_encoded_lines, max_size_in_bytes)
            included_lines = all_encoded_lines[:max_index]
            included_ids = all_ids[:max_index]
            total_bytes = sum(len(line) for line in included_lines)

            # Write to output
            training_data_info: dict[str, list | str | BytesIO | None] = {"data": None, "training_table_ids": included_ids}

            if transfer_type == "file":
                with open(file_name, "wb") as f:
                    f.writelines(included_lines)
                training_data_info["data"] = file_name

            elif transfer_type == "io_buffer":
                buffer = io.BytesIO()
                buffer.writelines(included_lines)
                buffer.seek(0)
                buffer.name = file_name
                training_data_info["data"] = buffer

            else:
                raise ValueError(f"Unsupported transfer type: {transfer_type}")

            self.logger.info(f"Prepared training data with {len(included_ids)} records ({total_bytes} bytes).")
            return training_data_info

        except Exception as e:
            self.logger.error(f"Error while preparing training data: {str(e)}", exc_info=True)
            raise e

    async def get_in_progress_training_data(self):
        """
        Retrieves all in progress training records.

        Returns:
            dict: A dictionary containing:
                - 'training_table_ids' (list): A list of processed row IDs from the training

        """
        from sql_utils import get_training_data

        self.logger.info("Fetching in progress training data")
        training_data_df = await get_training_data(sdp=self.sdp, training_data_version=TrainingDataVersions.FT_PROGRESS)
        training_data_df = await get_training_data(sdp=self.sdp, training_data_version=TrainingDataVersions.FT_PROGRESS)

        training_data = {"training_table_ids": training_data_df["IVCE_XCTN_LLM_TRNL_PRDT_REF_UID"].tolist()}

        return training_data

    async def update_training_version(
        self,
        ref_row_ids: list[str],
        training_version_name: str,
        bulk_update: bool = True,
        batch_size: int = 2000,
        wait_interval_per_batch: float = 5.0,
    ):
        """
        Updates the training version for a list of reference row IDs, using either a bulk temp table
        approach or batched updates.

        Parameters:
            ref_row_ids (list[str]): List of UIDs to update.
            training_version_name (str): Training version string to apply.
            bulk_update (bool): If True, use temp-table-based bulk update. Else fallback to batch loop.
            batch_size (int): Max rows per batch (for non-bulk mode).
            wait_interval_per_batch (float): Delay between batch executions (seconds).
        """
        try:
            total_rows = len(ref_row_ids)
            if total_rows == 0:
                self.logger.warning("No reference row IDs provided for update.")
                return

            self.logger.info(f"Updating rows (count: {len(ref_row_ids)}) with version {training_version_name}")

            if bulk_update:
                from sql_utils import bulk_update_training_version_with_temptable

                await bulk_update_training_version_with_temptable(
                    sdp=self.sdp, ref_uuid_list=ref_row_ids, version_name=training_version_name
                )
                return

            # Validate batch size
            batch_size = max(1, min(batch_size, 2000))
            total_batches, leftover = divmod(total_rows, batch_size)
            total_batches += 1 if leftover else 0

            # Estimate and log ETA
            estimated_time_sec = total_batches * (1.5 + wait_interval_per_batch)
            eta = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + estimated_time_sec))
            self.logger.info(f"Updating {total_rows} rows in {total_batches} batches. Estimated completion: {eta}")

            total_start = time.perf_counter()
            from sql_utils import bulk_update_training_version

            for i in range(0, total_rows, batch_size):
                batch_ids = ref_row_ids[i : i + batch_size]

                start = time.perf_counter()
                await bulk_update_training_version(sdp=self.sdp, ref_uuid_list=batch_ids, version_name=training_version_name)
                elapsed = time.perf_counter() - start
                self.logger.info(
                    f"Updated {len(batch_ids)} records in {elapsed:.2f}s. Sleeping for {wait_interval_per_batch}s..."
                )
                await asyncio.sleep(wait_interval_per_batch)

            total_elapsed = time.perf_counter() - total_start
            self.logger.info(f"All batches completed in {total_elapsed:.2f} seconds.")

        except Exception as e:
            self.logger.error(f"Error updating training version: {e}", exc_info=True)
            raise

    async def get_next_training_version(self) -> str:
        """
        Computes the next training data version string based on existing version entries
        in the database. Format: <YYYY>-<n>, where n increments within a year.

        Returns:
            str: The next training data version, e.g., '2025-4' or '2026-1'
        """
        from sql_utils import get_finetuned_llm_training_versions

        df, col_name = await get_finetuned_llm_training_versions(sdp=self.sdp)
        if df.empty:
            return f"{get_current_datetime_cst().year}-1"

        current_year = get_current_datetime_cst().year
        version_pattern = re.compile(r"^(\d{4})-(\d+)$")

        # Extract and parse valid version strings
        parsed_versions: list[tuple[int, int]] = (
            df[col_name]
            .dropna()
            .astype(str)
            .map(version_pattern.match)
            .dropna()
            .map(lambda m: (int(m.group(1)), int(m.group(2))))
        )

        # Group versions by year
        version_map: dict[int, list[int]] = {}
        for year, num in parsed_versions:
            version_map.setdefault(year, []).append(num)

        next_number = max(version_map[current_year]) + 1 if current_year in version_map else 1

        return f"{current_year}-{next_number}"
