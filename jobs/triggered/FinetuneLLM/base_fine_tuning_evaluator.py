import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from io import BytesIO
from logging import Logger

import finetune_constants
import pandas as pd
from app_context import context
from matcher import Matcher

from config import Config


class BaseFineTuningEvaluator(ABC):
    def __init__(self, config: Config, logger: Logger, matcher: Matcher, finetuned_llm: str | None = None):
        """
        Constructor to initiate the evaluation configuration.

        Parameters:
            config: The config file object.
            logger: the logger object to be used.
            finetuned_llm: LLM to be used for evaluation, if not provided, the
                        one that is configured for current env will be used.
        """
        self.finetuned_model = finetuned_llm
        self.config = config
        self.logger = logger
        self.matcher = matcher

        self.test_data_file_name = getattr(context.config, "TEST_DATA_FILE", None)
        # Folder to store the evaluation results.
        self.test_results_folder = getattr(context.config, "TEST_RESULTS_FOLDER", None)
        self.eval_result_file_name = self.test_results_folder + self._get_test_result_file_name()

        self.field_names = getattr(context.config, "FIELD_NAMES", [])
        self.input_field = getattr(context.config, "INPUT_FIELD", None)
        self.test_cols_to_keep = getattr(context.config, "TEST_COLS_TO_KEEP", [])

        # threshold to compare with the finetuned llm accuracy.
        # If the accuracy of new finetuned llm dont meet this,
        # it will be be not deployed and used.
        self.acc_thresholds = getattr(context.config, "ACCURACY_THRESHOLDS", {})
        CONFIDENCE_THRESHOLDS = getattr(context.config, "CONFIDENCE_THRESHOLDS", {})
        self.confidence_threshold_dict = {self._get_match_name(key): val for key, val in CONFIDENCE_THRESHOLDS.items()}
        self.confidence_col_name_dict = {
            self._get_match_name(field): self._get_conf_score_name(field) for field in self.field_names
        }

    def _get_test_result_file_name(self):
        file_name = f"eval_result_{self.finetuned_model}.xlsx"
        return file_name

    async def evaluate_llm(self):
        """
        Completes the total LLM evaluation.

        Step 1: Loads the test data to be used.
        Step 2: Run LLM stage inference for each test label data.
        Step 3: Evaluate the results and calculate accuracy.
        Step 4: Save the evaluation results.

        Returns the evaluation metrics data, which contains the
        accuracy of results compared to the input test data.

        Sample:
        Prediction_Type	Predictions	 Correct    Wrong  .... Accuracy
        -------------------------------------------------------------
        MFR_MATCH           5              4        1          80.0%
        MFR_PN_MATCH        5              2        3          40.0%
        UNSPSC_MATCH        5              3        2          60.0%
        """

        self.logger.info("Evaluating LLM")

        # Load the test data.
        test_data_df = self._load_labelled_test_data(
            self.config.AZ_AKSSTAI_FINE_TUNED_LLM_CONTAINER_NAME, self.test_data_file_name
        )

        if test_data_df is None or test_data_df.empty:
            self.logger.error("Unable to get labelled test data.")
            raise Exception("Unable to get labelled test data.")

        # Run the llm for the test labeled data
        evaluation_results_df = await self._trigger_llm_evaluation(test_data_df)

        # calculate the accuracy and other metrics.
        eval_metrics_df = self._calculate_eval_metrics(evaluation_results_df)

        # save the results and the metrics.
        self._save_evaluation_test_results(
            evaluation_results_df,
            eval_metrics_df,
            self.eval_result_file_name,
            self.config.AZ_AKSSTAI_FINE_TUNED_LLM_CONTAINER_NAME,
        )

        self.llm_accuracy_result = self._get_metrics_to_save(eval_metrics_df)
        self._save_fine_tuned_llm_accuracy_result(self.llm_accuracy_result)

        return eval_metrics_df

    def _get_metrics_to_save(self, eval_results_df: pd.DataFrame) -> dict:
        results = defaultdict(lambda: defaultdict(dict))
        results["finetuned_llm"] = self.finetuned_model

        for field in self.field_names:
            item = eval_results_df[eval_results_df["Prediction_Type"] == self._get_match_name(field)]

            def get_value_from_item(key, item=item) -> str:
                return item[key].iloc[0]

            results[field] = {
                "metrics": {
                    "accuracy": get_value_from_item("Accuracy"),
                    "precision": get_value_from_item("Precision"),
                    "recall": get_value_from_item("Recall"),
                    "f1_score": get_value_from_item("F1-Score"),
                },
                "counts": {
                    "total": get_value_from_item("Total_Predictions").item(),
                    "correct": get_value_from_item("Total_Correct").item(),
                    "wrong": get_value_from_item("Total_Wrong").item(),
                    "skipped": get_value_from_item("Total_Skipped").item(),
                    "false_positives": get_value_from_item("False_Positive").item(),
                    "false_negatives": get_value_from_item("False_Negative").item(),
                    "true_positives": get_value_from_item("True_Positive").item(),
                    "true_negatives": get_value_from_item("True_Negative").item(),
                },
                "threshold": {
                    "confidence": self.confidence_threshold_dict[self._get_match_name(field)],
                    "accuracy": self.acc_thresholds[field],
                },
            }

        for field in ("UNSPSC_CODE_MATCH_2", "UNSPSC_CODE_MATCH_4", "UNSPSC_CODE_MATCH_8"):
            item = eval_results_df[eval_results_df["Prediction_Type"] == self._get_match_name(field)]

            def get_value_from_item(key, item=item) -> str:
                return item[key].iloc[0]

            results[field] = {
                "metrics": {
                    "accuracy": get_value_from_item("Accuracy"),
                    "precision": get_value_from_item("Precision"),
                    "recall": get_value_from_item("Recall"),
                    "f1_score": get_value_from_item("F1-Score"),
                },
                "counts": {
                    "total": get_value_from_item("Total_Predictions").item(),
                    "correct": get_value_from_item("Total_Correct").item(),
                    "wrong": get_value_from_item("Total_Wrong").item(),
                    "skipped": get_value_from_item("Total_Skipped").item(),
                    "false_positives": get_value_from_item("False_Positive").item(),
                    "false_negatives": get_value_from_item("False_Negative").item(),
                    "true_positives": get_value_from_item("True_Positive").item(),
                    "true_negatives": get_value_from_item("True_Negative").item(),
                },
                "threshold": {
                    "confidence": self.confidence_threshold_dict[self._get_match_name("UNSPSC_CODE")],
                    "accuracy": self.acc_thresholds["UNSPSC_CODE"],
                },
            }

        return results

    def _load_labelled_test_data(self, container_name: str | None = None, blob_name: str | None = None) -> pd.DataFrame | None:
        """
        Reads the given test data from storage/local file and returns the
        data frame created from the file.

        The file in the storage is expected to be in csv or excel file.

        Parameters:
            container_name: The Az storage container name, needs to be provided
                             if using AZ storage test file.
            blob_name: The blob or the file name of the labelled test data.
                       needs to be provided if using AZ storage test file.
        Returns:
            test_data_df: The data frame containing the labelled test data.
                          Or returns None if error.
        """
        try:
            blob_client = self.config.azure_clients.azure_blob_storage_client.get_blob_client(
                container=container_name, blob=blob_name
            )
            data_out_stream = blob_client.download_blob()
            test_data_df = pd.read_excel(BytesIO(data_out_stream.readall()), engine="openpyxl", dtype={"UNSPSC_CODE_LABEL": str})
            self.logger.info(f"Loaded test data of {len(test_data_df)} records from {blob_name}")
            return test_data_df.copy()
        except Exception as ex:
            self.logger.error(f"Error while loading the test data : {str(ex)}", exc_info=True)
        return None

    async def _trigger_llm_evaluation(self, test_data_df: pd.DataFrame, num_rows: int = 0) -> pd.DataFrame:
        """
        Completes the LLM inference for all the test data and returns
        the data in a DataFrame.

        Parameters:
            test_data_df: test data
            num_rows: Consider only the first X rows for evaluation.
        """
        if num_rows != 0:
            test_data_df = test_data_df.head(num_rows)
        tasks = [self._evaluate_test_data_entry(row) for _, row in test_data_df.iterrows()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        evaluation_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Error in evaluating : {str(result)}", exc_info=True)
            else:
                evaluation_results.append(result)

        evaluation_results_df = pd.DataFrame(evaluation_results)
        return evaluation_results_df

    def _get_confidence_score(self, row, match_type):
        return row[self._get_conf_score_name(match_type)]

    def _post_process_calculate(self, df: pd.DataFrame, res_data: dict) -> dict:
        return res_data

    def _preprocess_calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Can override this method to do some preprocessing before calculating metrics,
        like skipping certain fields based on conditions

        Args:
            df (pd.DataFrame): dataframe to calculate metrics on

        Returns:
            pd.DataFrame: preprocessed dataframe
        """
        return df

    def _calculate_eval_metrics(self, test_result_df_orig: pd.DataFrame) -> pd.DataFrame:
        """
        For the given LLM evaluation restuls, calculate the
        accuracy and other metrics and return it as Data Frame.

        Parameters:
            test_result_df_orig: The evaluation results data.

        Returns:
            data frame: The evaluation metrics as data frame.
        """
        result_data: dict[str, list] = {
            "Prediction_Type": [],
            "Total_Predictions": [],
            "Total_Correct": [],
            "Total_Wrong": [],
            "Total_Skipped": [],
            "True_Positive": [],
            "True_Negative": [],
            "False_Positive": [],
            "False_Negative": [],
            "Recall": [],
            "Precision": [],
            "F1-Score": [],
            "Accuracy": [],
        }

        test_result_df_orig = self._preprocess_calculate(test_result_df_orig)

        self.logger.debug(f"Using {test_result_df_orig} for calculating the metrics")
        for field_name in self.field_names:
            match_type = self._get_match_name(field_name)
            test_result_df = test_result_df_orig.copy(deep=True)
            threshold = self.confidence_threshold_dict[match_type]
            labels_predicted = pd.DataFrame()

            skip_count = len(test_result_df[~test_result_df[match_type].isin(["yes", "no"])])
            test_result_df = test_result_df[test_result_df[match_type].isin(["yes", "no"])]
            test_result_df["MEETS_THRESHOLD"] = test_result_df.apply(
                lambda row, field_name=field_name, threshold=threshold: row[self._get_conf_score_name(field_name)] >= threshold,
                axis=1,
            )
            labels_predicted["TYPE_MATCH_BIN"] = test_result_df[match_type].map({"yes": 1, "no": 0})

            def classify_result(row, match_type=match_type):
                if row["MEETS_THRESHOLD"] and row[match_type] == "yes":
                    return finetune_constants.TRUE_POSITIVE
                elif not row["MEETS_THRESHOLD"] and row[match_type] == "no":
                    return finetune_constants.TRUE_NEGATIVE
                elif row["MEETS_THRESHOLD"] and row[match_type] == "no":
                    return finetune_constants.FALSE_POSITIVE
                elif not row["MEETS_THRESHOLD"] and row[match_type] == "yes":
                    return finetune_constants.FALSE_NEGATIVE

            test_result_df["PREDICTION_TYPE"] = test_result_df.apply(classify_result, axis=1)
            pred_counts = test_result_df["PREDICTION_TYPE"].value_counts()
            tp = pred_counts.get(finetune_constants.TRUE_POSITIVE, 0)
            tn = pred_counts.get(finetune_constants.TRUE_NEGATIVE, 0)
            fp = pred_counts.get(finetune_constants.FALSE_POSITIVE, 0)
            fn = pred_counts.get(finetune_constants.FALSE_NEGATIVE, 0)

            accuracy = round(((tp + tn) / (tp + tn + fp + fn)) * 100, 2)
            recall = round((tp / (tp + fn)) * 100, 2) if (tp + fn) != 0 else 0.0
            precision = round((tp / (tp + fp)) * 100, 2) if (tp + fp) != 0 else 0.0
            f1_score = round((2 * precision * recall) / (precision + recall), 2) if (precision + recall) != 0 else 0.0

            result_data["Prediction_Type"].append(match_type)

            total_predictions = len(labels_predicted)
            total_correct = labels_predicted["TYPE_MATCH_BIN"].sum()
            total_wrong = total_predictions - total_correct

            result_data["Total_Predictions"].append(total_predictions)
            result_data["Total_Correct"].append(total_correct)
            result_data["Total_Wrong"].append(total_wrong)
            result_data["Total_Skipped"].append(skip_count)
            result_data["True_Positive"].append(tp)
            result_data["True_Negative"].append(tn)
            result_data["False_Positive"].append(fp)
            result_data["False_Negative"].append(fn)

            result_data["Recall"].append(f"{recall}%")
            result_data["Precision"].append(f"{precision}%")
            result_data["F1-Score"].append(f"{f1_score}%")
            result_data["Accuracy"].append(f"{accuracy}%")

        result_data = self._post_process_calculate(test_result_df_orig, result_data)

        eval_metrics_df = pd.DataFrame(result_data)
        return eval_metrics_df

    def _save_evaluation_test_results(
        self,
        evaluated_results_df: pd.DataFrame,
        metrics_results_df: pd.DataFrame,
        blob_name: str,
        container_name: str | None = None,
    ):
        """
        Store the evaluation test results and metrics in the output file.

        Parameters:
            evaluated_results_df: Data frame of the test results data.
            metrics_results_df: Data frame of the accuracy metrics data.
            container_name: The AZ storage container name to use if using AZ storage.
                            If not provided, will be read from config.
            blob_name: The file name to be used, if not provided will be generated
                        based on current time as version.
        """
        try:
            # Create in-memory Excel file
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                workbook = writer.book

                # Define formats
                header_format = workbook.add_format({"bold": True, "align": "center", "valign": "vcenter"})
                center_format = workbook.add_format({"align": "center", "valign": "vcenter"})

                # Function to write and format each sheet
                def write_formatted_sheet(df, sheet_name):
                    df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1, header=False)
                    worksheet = writer.sheets[sheet_name]

                    # Set column widths and formats
                    for col_idx, col_name in enumerate(df.columns):
                        max_width = max(df[col_name].astype(str).map(len).max(), len(col_name)) + 2
                        worksheet.set_column(col_idx, col_idx, max_width, center_format)
                        worksheet.write(0, col_idx, col_name, header_format)

                # Write both sheets
                write_formatted_sheet(evaluated_results_df, "Evaluation_Results")
                write_formatted_sheet(metrics_results_df, "Evaluation_Metrics")

            # Upload to Azure Blob
            output.seek(0)

            if not container_name:
                container_name = self.config.AZ_AKSSTAI_FINE_TUNED_LLM_CONTAINER_NAME
            blob_client = self.config.azure_clients.azure_blob_storage_client.get_blob_client(
                container=container_name, blob=blob_name
            )
            blob_client.upload_blob(output, overwrite=True)

        except Exception as ex:
            self.logger.error("Error uploading results to AZ Storage")
            raise ex

    def get_accuracy_from_eval_results(self, eval_results_df: pd.DataFrame) -> dict:
        """
        For the given fine tuned llm evaluation results,
        return the accuracy of MFR Name, product number and unspsc code.
        """
        results = defaultdict(lambda: defaultdict(dict))

        results["finetuned_llm"] = self.finetuned_model
        for field in self.field_names:
            accuracy = eval_results_df[eval_results_df["Prediction_Type"] == self._get_match_name(field)]["Accuracy"].iloc[0]
            results["accuracy"][field] = float(accuracy[:-1])  # remove '%' at end to make it easier for comparisons.

        return results

    def _save_fine_tuned_llm_accuracy_result(
        self, accuracy_result: dict, blob_name: str | None = None, container_name: str | None = None
    ):
        """
        Store the accuracy details from the evaluation into AZ store file.
        If blob_name not provided the  output file name will be in the format of:
        accuracy_result_{self.finetuned_model}.json

        Parameters:
            accuracy_result: Dict containing the results for fields such as MFR name, PN and UNSPSC code.
            container_name: The AZ storage container name to use if using AZ storage.
                            If not provided, will be read from config.
            blob_name: The file name/ blob name to be used , if not provided will be generated
                        based on current time as version.
        """
        try:
            json_data = json.dumps(accuracy_result, indent=4)

            if not container_name:
                container_name = self.config.AZ_AKSSTAI_FINE_TUNED_LLM_CONTAINER_NAME
            if not blob_name:
                blob_name = self._get_accuracy_result_file_name()
            blob_client = self.config.azure_clients.azure_blob_storage_client.get_blob_client(
                container=container_name, blob=blob_name
            )
            blob_client.upload_blob(json_data, overwrite=True)

        except Exception as ex:
            self.logger.error("Error uploading accuracy results to AZ Storage")
            raise ex

    @abstractmethod
    async def _query_llm(self, description: str):
        pass

    async def _evaluate_test_data_entry(self, row: pd.Series) -> dict:
        """
        Run the LLM inference for each test data entry provided as input.
        Captures the LLm stage response and updates the result that the
        response item matching the expected data label.

        Parameters:
            row: each test data entry from the test data frame.

        Returns:
            dict: the response details for each MFR name, PN and UNSPSC code.
        """
        start = time.perf_counter()
        result = await self._query_llm(row[self.input_field])
        end = time.perf_counter()
        duration = round(end - start, 2)

        return_data = {"ITEM_PROC_TIME": duration}

        for col in self.test_cols_to_keep:
            return_data[col] = row[col]

        for field in self.field_names:
            return_data[f"{field}_LLM"] = result.get(field)
            return_data[self._get_match_name(field)] = await self.matcher.is_match(
                result.get(field), row[self._get_label_name(field)], field
            )
            return_data[self._get_conf_score_name(field)] = result.get(self._get_conf_score_name(field), {}).get(field)

        return return_data

    def _get_accuracy_result_file_name(self):
        file_name = f"{self.test_results_folder}accuracy_result_{self.finetuned_model}.json"
        return file_name

    @staticmethod
    def _get_label_name(string: str):
        return f"{string}_LABEL"

    @staticmethod
    def _get_conf_score_name(string: str):
        return f"{string}_CONF_SCORE"

    @staticmethod
    def _get_match_name(string: str):
        return f"{string}_MATCH"

    def benchmark_result_with_threshold(self, results: dict) -> bool:
        """
        Compare the given Finetuned LLm results with the threshold,
        return true if the accuracy is above the threshold.

        Parameters:
            results: The finetuned llm accuracy test results.
            thresholds: the threshold accuracy to compare to.
            check_only_items: Accuracy benchmarked only against these
                              (example:any or all of MFR_NAME,  MFR_PN, UNSPSC_CODE)

        Sample result:
        {
            "finetuned_llm": "<llm model name>",
            "accuracy" : {
                "MFR_NAME": 80,
                "MFR_PN": 90,
                "UNSPSC_CODE": 78
            }
        }

        Sample threshold:
        {
            "accuracy" : {
                "MFR_NAME": 80,
                "MFR_PN": 90,
                "UNSPSC_CODE": 78
            }
        }

        Returns:
            bool: True if the accuracy is above threshold for all the given check_only_items
        """
        result_accepted = False

        for item in self.field_names:
            llm_accuracy = results["accuracy"][item]
            threshold = self.acc_thresholds[item]
            self.logger.debug(f"{item} : LLM Accuracy : {llm_accuracy},  Threshold : {threshold}")
            if llm_accuracy < threshold:
                self.logger.info("Failed benchmark due to threshold.")
                break
        else:
            result_accepted = True
        return result_accepted

    def get_accuracy_result_for_llm(self, finetuned_llm_model_name: str, container_name: str | None = None) -> dict:
        """
        For the finetune model, return the accuracy result for
        the evaluation done with the test data.
        The accuracy results for finetuned models will be stored in
        AZ store with file : accuracy_result_<model_name>.json

        Parameters:
            finetuned_llm_model_name: model name to fetch the accuracy results.
            container_name: the AZ container name to look for accuracy result files.
        """
        if not container_name:
            container_name = self.config.AZ_AKSSTAI_FINE_TUNED_LLM_CONTAINER_NAME

        container_client = self.config.azure_clients.azure_blob_storage_client.get_container_client(container=container_name)

        # Find matching blob
        matching_blob = None
        llm_res_file_name = f"{self.test_results_folder}accuracy_result_{finetuned_llm_model_name}.json"

        for blob in container_client.list_blobs():
            if llm_res_file_name == blob.name:
                matching_blob = blob.name
                break

        if not matching_blob:
            raise FileNotFoundError(f"No accuracy result file for model '{finetuned_llm_model_name}'")

        # Download and load as dict
        blob_client = container_client.get_blob_client(matching_blob)
        blob_content = blob_client.download_blob().readall()
        self.logger.debug(f"LLM accuracy results content : {json.loads(blob_content)}")
        return json.loads(blob_content)

    def compare_llm_accuracy_result(
        self, current_deployed_accuracy: dict, new_finetuned_accuracy: dict, check_only_items=None
    ) -> bool:
        """
        Compare the given Finetuned LLm results with the threshold,
        return true if the accuracy is above the threshold.

        Parameters:
            current_deployed_accuracy: the accuracy results of currently used model.
            new_finetuned_accuracy: The finetuned llm accuracy test results.
            check_only_items: Accuracy benchmarked only against these
                              (example : any or all of MFR_NAME,  MFR_PN, UNSPSC_CODE)
        Returns:
            bool: True if the accuracy is better than the current given LLM.
        """
        accuracy_improved = False
        if not check_only_items:
            check_only_items = self.field_names

        for item in check_only_items:
            current_llm_accuracy = float(current_deployed_accuracy[item]["metrics"]["accuracy"][:-1])
            new_llm_accuracy = new_finetuned_accuracy["accuracy"][item]
            self.logger.info(f"{item} : New LLM Accuracy : {new_llm_accuracy},  Current LLM Accuracy : {current_llm_accuracy}")
            if new_llm_accuracy <= current_llm_accuracy:
                break
        else:
            accuracy_improved = True
        return accuracy_improved
