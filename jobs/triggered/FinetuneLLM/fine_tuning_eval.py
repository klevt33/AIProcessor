import time
from logging import Logger

import finetune_constants
import pandas as pd
from app_context import context
from base_fine_tuning_evaluator import BaseFineTuningEvaluator
from matcher import Matcher

from ai_engine import AIEngineCache
from ai_utils import get_clean_mfr_name
from config import Config

logger = context.logger


class FineTunedLLMEvaluator(BaseFineTuningEvaluator):
    """
    Class provides methods to evaluate any given fine tuned LLM.

    The LLM to be evaluated will be taken from the current configuration
    provided to this class.

    The test data needs to be provided or can be genertaed using
    the 'Invoice_test_data_for_AI' sheet which is being used for
    manual evaluation.
    """

    def __init__(self, config: Config, logger: Logger, matcher: Matcher, finetuned_llm: str | None = None):
        """
        Constructor to initiate the evaluation configuration.

        Parameters:
            config: The config file object.
            logger: the logger object to be used.
            finetuned_llm: LLM to be used for evaluation, if not provided, the
                        one that is configured for current env will be used.
        """
        super().__init__(config, logger, matcher, finetuned_llm)

        from ai_engine import AIEngine
        from sdp import SDP

        self.sdp = SDP(config=config)
        self.ai_engine = AIEngine(config=config, sdp=self.sdp)

    # Custom because of how the LLM is called with AI cache and all, also with custom mappings from LLM to MFR_NAME_LLM and stuff
    async def _query_llm(self, description: str):
        """
        Query the LLM with the given description text. Prompt will
        be created by the AI engine LLM stage API.

        Parameters:
            description: (str): The invoide description for each test label data.

        Returns:
            response: the stage details response including llm response, confidence score.
        """
        start = time.perf_counter()

        # aiengine uses regex and it may fail for non-string input.
        if not isinstance(description, str):
            description = str(description)

        from lemmatizer import get_lemmatizer
        from utils import clean_text_for_llm

        cleaned_description = clean_text_for_llm(get_lemmatizer(), description)

        from constants import Logs
        from invoice_extraction import InvoiceDetail

        ivce_dtl = InvoiceDetail({"is_special_case": False, "fields": [Logs.MFR_NAME, Logs.PRT_NUM, Logs.UNSPSC]})

        ai_cache = AIEngineCache()
        ai_cache.cleaned_description_for_llm = cleaned_description
        finetuned_stage_details = (await self.ai_engine.ai_stages.extract_from_finetuned_llm(self.sdp, ai_cache, ivce_dtl))[0]
        self.logger.debug(f"End Inference : Description: {description} , elapsed {time.perf_counter() - start:.2f}s")
        details = finetuned_stage_details.details
        # Map to the expected names
        details["MFR_NAME_LLM"] = details.get(finetune_constants.STAGE_RES_MFR_NAME)
        details["MFR_NAME_CONF_SCORE"] = details.get(finetune_constants.STAGE_RES_CONF_SCORE, {}).get(
            finetune_constants.STAGE_RES_MFR_NAME
        )
        details["MFR_PN_LLM"] = details.get(finetune_constants.STAGE_RES_PN)
        details["MFR_PN_CONF_SCORE"] = details.get(finetune_constants.STAGE_RES_CONF_SCORE, {}).get(
            finetune_constants.STAGE_RES_PN
        )
        details["UNSPSC_LLM"] = details.get(finetune_constants.STAGE_RES_UNSPSC)
        details["UNSPSC_CONF_SCORE"] = details.get(finetune_constants.STAGE_RES_CONF_SCORE, {}).get(
            finetune_constants.STAGE_RES_UNSPSC
        )
        return details

    async def _is_mfr_generic(self, mfr_actual: str, mfr_llm: str) -> bool:
        """Checks to see if the either string is generic

        Args:
            mfr_actual (str): Source truth of MFR
            mfr_llm (str): LLM guess of MFR

        Returns:
            bool: whether mfr_actual or mfr_llm are generic
        """
        if "GENERIC" in mfr_actual.upper() or "GENERIC" in mfr_llm.upper():
            return True
        mfr_llm = (await get_clean_mfr_name(self.sdp, mfr_llm))[0]
        return "GENERIC" in mfr_llm.upper()

    # Custom because - Skip part number if mfr is generic type
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
        result = await self._query_llm(row["ITM_LDSC"])
        end = time.perf_counter()
        duration = round(end - start, 2)

        mfr_is_generic = await self._is_mfr_generic(
            str(row["MFR_NAME_LABEL"]), str(result.get(finetune_constants.STAGE_RES_MFR_NAME))
        )

        # Flatten the result and add time
        return {
            "Label_ID": row["Label_ID"],
            "IVCE_DTL_UID": row["IVCE_DTL_UID"],
            "ITM_LDSC": row["ITM_LDSC"],
            "CLASSIFICATION": row["CLASSIFICATION"],
            "MFR_NAME_LABEL": row["MFR_NAME_LABEL"],
            "MFR_NAME_LLM": result.get(finetune_constants.STAGE_RES_MFR_NAME),
            "MFR_NAME_MATCH": await self.matcher.is_match(
                result.get(finetune_constants.STAGE_RES_MFR_NAME), row["MFR_NAME_LABEL"], "MFR_NAME"
            ),
            "MFR_NAME_CONF_SCORE": (
                result.get(finetune_constants.STAGE_RES_CONF_SCORE, {}).get(finetune_constants.STAGE_RES_MFR_NAME)
            ),
            "MFR_PN_LABEL": row["MFR_PN_LABEL"],
            "MFR_PN_LLM": result.get(finetune_constants.STAGE_RES_PN),
            # Skip part number if mfr is generic type
            "MFR_PN_MATCH": (
                "skip"
                if mfr_is_generic
                else await self.matcher.is_match(result.get(finetune_constants.STAGE_RES_PN), row["MFR_PN_LABEL"], "MFR_PN")
            ),
            "MFR_PN_CONF_SCORE": result.get(finetune_constants.STAGE_RES_CONF_SCORE, {}).get(finetune_constants.STAGE_RES_PN),
            "UNSPSC_CODE_LABEL": row["UNSPSC_CODE_LABEL"],
            "UNSPSC_CODE_LLM": result.get(finetune_constants.STAGE_RES_UNSPSC),
            "UNSPSC_CODE_MATCH": await self.matcher.is_match(
                result.get(finetune_constants.STAGE_RES_UNSPSC), row["UNSPSC_CODE_LABEL"], "UNSPSC_CODE"
            ),
            "UNSPSC_CODE_CONF_SCORE": (
                result.get(finetune_constants.STAGE_RES_CONF_SCORE, {}).get(finetune_constants.STAGE_RES_UNSPSC)
            ),
            "UNSPSC_CODE_MATCH_2_MATCH": (
                (
                    await self.matcher.check_unspsc_2(
                        str(row["UNSPSC_CODE_LABEL"]), str(result.get(finetune_constants.STAGE_RES_UNSPSC))
                    )
                ).value
            ),
            "UNSPSC_CODE_MATCH_4_MATCH": (
                (
                    await self.matcher.check_unspsc_4(
                        str(row["UNSPSC_CODE_LABEL"]), str(result.get(finetune_constants.STAGE_RES_UNSPSC))
                    )
                ).value
            ),
            "UNSPSC_CODE_MATCH_8_MATCH": (
                (
                    await self.matcher.check_unspsc_8(
                        str(row["UNSPSC_CODE_LABEL"]), str(result.get(finetune_constants.STAGE_RES_UNSPSC))
                    )
                ).value
            ),
            "ITEM_PROC_TIME": duration,
        }

    def _get_confidence_score(self, row, match_type):
        return row[self.confidence_col_name_dict[match_type]]

    @staticmethod
    def _skip_values(
        df: pd.DataFrame, llm_col_name: str, label_col_name: str, match_col_name: str, skip_values: list[str]
    ) -> pd.DataFrame:
        """If llm_col_name or label_col_name are in the list of skip_values, then update match_col name to be "skip"

        Args:
            df (DataFrame):
            llm_col_name (str):
            label_col_name (str):
            match_col_name (str):
            skip_values (list[str]):

        Returns:
            DataFrame:
        """
        condition = df[llm_col_name].isin(skip_values) | df[label_col_name].isin(skip_values)
        df.loc[condition, match_col_name] = "skip"
        return df

    def _preprocess_calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        return FineTunedLLMEvaluator._skip_values(
            df,
            "MFR_NAME_LLM",
            "MFR_NAME_LABEL",
            self._get_match_name("MFR_NAME"),
            getattr(context.config, "MFR_NAME_SKIP_VALUES", []),
        )

    def _post_process_calculate(self, df: pd.DataFrame, res_data: dict) -> dict:
        field_name = "UNSPSC_CODE"
        match_type = self._get_match_name(field_name)
        threshold = self.confidence_threshold_dict[match_type]
        labels_predicted = pd.DataFrame()

        for i in ("2", "4", "8"):
            test_result_df = df.copy(deep=True)
            match_type = f"UNSPSC_CODE_MATCH_{i}_MATCH"
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

            res_data["Prediction_Type"].append(match_type)

            total_predictions = len(labels_predicted)
            total_correct = labels_predicted["TYPE_MATCH_BIN"].sum()
            total_wrong = total_predictions - total_correct

            res_data["Total_Predictions"].append(total_predictions)
            res_data["Total_Correct"].append(total_correct)
            res_data["Total_Wrong"].append(total_wrong)
            res_data["Total_Skipped"].append(skip_count)
            res_data["True_Positive"].append(tp)
            res_data["True_Negative"].append(tn)
            res_data["False_Positive"].append(fp)
            res_data["False_Negative"].append(fn)

            res_data["Recall"].append(f"{recall}%")
            res_data["Precision"].append(f"{precision}%")
            res_data["F1-Score"].append(f"{f1_score}%")
            res_data["Accuracy"].append(f"{accuracy}%")
        return res_data
