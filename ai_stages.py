import json
from copy import copy
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from agents import Agents
from ai_utils import (
    calculate_confidence_for_finetuned_llm,
    calculate_confidence_for_web_search_results_with_ranking,
    check_if_rpa_can_boost_confidence,
    execute_hardened_llm_request,
    get_higher_confidence_web_search_result,
    run_agent_and_get_json_with_retry,
    validate_search_results_schema,
)
from azure_search_utils import AzureSearchUtils
from classifier_utils import ClassifierUtils
from constants import (
    Constants,
    DescriptionCategories,
    LocalFiles,
    Logs,
    MfrRelationshipType,
    SpecialCases,
    StageNames,
    SubStageNames,
)
from exceptions import InvalidJsonResponseError, InvoiceProcessingError
from llm import LLM, LLMClientType
from logger import logger
from matching_scores import best_match_score, process_manufacturer_dict
from matching_utils import analyze_description, mfr_eq_type
from prompts import Prompts
from semantic_matching import semantic_match_by_description
from utils import (
    clean_dataframe,
    clean_description,
    clean_dictionary,
    clean_text_for_classification,
    extract_json,
    get_alphanumeric,
    get_current_datetime_cst,
    get_stage_and_sub_stage_names_from_log_stage,
    have_min_length,
    is_not_empty,
    is_not_null,
    load_yaml,
)


class MatchContextType(str, Enum):
    """
    Enumeration for the contextual relationship between an invoice line and a matched product.
    """

    DIRECT_MATCH = "DIRECT_MATCH"
    REPLACEMENT_PART = "REPLACEMENT_PART"
    ACCESSORY_PART = "ACCESSORY_PART"
    LOT_OR_KIT = "LOT_OR_KIT"  # For items that contain the matched part plus others
    UNRELATED = "UNRELATED"


class ValidationResult(BaseModel):
    """
    Defines the structured output for the LLM Context Validator.
    This schema will be enforced by the Azure OpenAI service to guarantee a valid JSON response.
    """

    context_type: MatchContextType = Field(
        description="The classification of the relationship between the invoice text and the matched product."
    )
    reason: str = Field(description="A brief, clear explanation in 1-2 sentences that justifies the chosen context_type.")


class StageDetails:
    # --- Class Defaults ---
    # These provide safe default values for code access (dot notation)
    # but do NOT appear in the instance dictionary (JSON) unless explicitly set.
    is_validation_stage = False

    def __init__(
        self,
        stage_number: int,
        sub_stage_code: str,
        stage: str,
        sub_stage: str,
        status=None,
        details=None,
        is_validation_stage: bool = False,
    ):
        if details is None:
            details = {}
        self.stage_number = stage_number
        self.sub_stage_code = sub_stage_code
        self.stage_name = stage
        self.sub_stage_name = sub_stage
        self.status = status
        self.details = details
        # 1. Validation Flag
        # Only add to instance (and JSON) if True.
        # If False, code falls back to class default (False), and JSON remains clean.
        if is_validation_stage:
            self.is_validation_stage = True

            # 2. Final Success Flag
            # Only relevant for validators.
            self.is_final_success: bool = True


class AIStages:

    def __init__(self, config, category_utils):
        self.config = config
        self.category_utils = category_utils
        self.agents = Agents(config=config)
        self.llms = LLM(config=config)
        self.text_analytics_client = config.azure_clients.text_analytics_client
        self.search_utils = AzureSearchUtils(config)
        self.classifier_utils = ClassifierUtils(config=config, category_utils=category_utils)

    async def check_if_substage_required(self, ivce_dtl, stage_name: StageNames, sub_stage_name: SubStageNames) -> bool:
        """
        Determine whether the specified sub-stage is still active under a given stage
        in the invoice's stage mapping.

        A sub-stage is considered “required” if:
        1. `stage_name` exists as a key in `ivce_dtl.stage_mapping`, and
        2. `sub_stage_name` appears in the list of sub-stages for that stage.

        Args:
            ivce_dtl (YourInvoiceDetailClass):
                An object with a `.stage_mapping` attribute, which should be
                a dict mapping StageNames to lists of SubStageNames.
            stage_name (StageNames):
                The stage under which to look.
            sub_stage_name (SubStageNames):
                The specific sub-stage to check for presence.

        Returns:
            bool:
                True if `sub_stage_name` is present under `stage_name`;
                False otherwise.
        """
        mapping = getattr(ivce_dtl, "stage_mapping", {})
        return sub_stage_name in mapping.get(stage_name, [])

    async def fetch_classification(self, lemmatizer, ivce_dtl):
        """
        Classifies the description into a high-level category

        Args:
            - lemmatizer: Lemmatizer object to lemmatize words
            - ivce_dtl: invoice details object

        Returns:
            - stage_details (StageDetails)
            The StageDetails object contains the following attributes:
                - stage_number: 1
                - sub_stage_code: "1.0"
                - stage_name: Constants.CLASSIFICATION
                - sub_stage_name: Constants.DESCRIPTION_CLASSIFIER
                - status
                - error_message
                - details
                    - end_time
                    - category
                    - category_id
                    - confidence_score
                    - description
        """

        stage_details = StageDetails(
            stage_number=1, sub_stage_code="1.0", stage=StageNames.CLASSIFICATION, sub_stage=SubStageNames.DESCRIPTION_CLASSIFIER
        )

        try:
            description = ivce_dtl.ITM_LDSC

            cleaned_description = clean_text_for_classification(lemmatizer, description)

            # Case 1: Empty/short description
            if cleaned_description is None or not have_min_length(cleaned_description, length=self.config.min_description_len):
                category = DescriptionCategories.BAD
                category_id = self.category_utils.get_category_id(category_name=category)
                stage_details.status = Constants.SUCCESS_lower
                stage_details.details = {
                    Logs.IS_VERIFIED: False,
                    Logs.IS_MFR_CLEAN: True,
                    Logs.END_TIME: get_current_datetime_cst(),
                    Logs.CATEGORY: category,
                    Logs.CATEGORY_ID: category_id,
                    Logs.CONFIDENCE: 100,
                    Logs.DESCRIPTION: cleaned_description,
                }
                return stage_details

            cleaned_upper_description = cleaned_description.upper()

            # Case 2: Adjustments
            if cleaned_upper_description == Constants.ADJUSTMENTS:
                category = DescriptionCategories.AP_ADJUSTMENT
                category_id = self.category_utils.get_category_id(category_name=category)
                stage_details.status = Constants.SUCCESS_lower
                stage_details.details = {
                    Logs.IS_VERIFIED: False,
                    Logs.IS_MFR_CLEAN: True,
                    Logs.END_TIME: get_current_datetime_cst(),
                    Logs.CATEGORY: category,
                    Logs.CATEGORY_ID: category_id,
                    Logs.CONFIDENCE: 100,
                    Logs.DESCRIPTION: cleaned_upper_description,
                }
                return stage_details

            # Case 3: Normal classification → delegate to _classify
            return await self.classifier_utils.classify_using_azure_custom_language_model(
                cleaned_description=cleaned_description,
                stage_details=stage_details,
                project_name=self.config.LS_DESCRIPTION_PROJECT_NAME,
                deployment_name=self.config.LS_DESCRIPTION_DEPLOYMENT_NAME,
                category_id_fn=lambda cat, _: self.category_utils.get_category_id(category_name=cat),
            )

        except Exception as e:
            logger.error(f"CLASSIFICATION: Unexpected error during processing: {str(e)}", exc_info=True)
            stage_details.status = Constants.ERROR_lower
            stage_details.details = {
                Logs.END_TIME: get_current_datetime_cst(),
                Logs.MESSAGE: f"Unexpected error: {str(e)}",
                Logs.DESCRIPTION: getattr(ivce_dtl, "ITM_LDSC", None),
            }
            return stage_details

    async def fetch_lot_classification(self, cleaned_description, parent_category_id):
        """
        Classifies if the description is LOT or Not

        Args:
            - cleaned_description: Description cleaned for classification
            - parent_category_name: Name of parent class

        Returns:
            - stage_details (StageDetails)
            The StageDetails object contains the following attributes:
                - stage_number: 2
                - sub_stage_code: "1.1"
                - stage_name: Constants.CLASSIFICATION
                - sub_stage_name: Constants.LOT_CLASSIFIER
                - status
                - error_message
                - details
                    - end_time
                    - category
                    - category_id
                    - confidence_score
                    - description
        """
        stageDetails = StageDetails(
            stage_number=2, sub_stage_code="1.1", stage=StageNames.CLASSIFICATION, sub_stage=SubStageNames.LOT_CLASSIFIER
        )

        return await self.classifier_utils.classify_using_azure_custom_language_model(
            cleaned_description=cleaned_description,
            stage_details=stageDetails,
            project_name=self.config.LS_LOT_PROJECT_NAME,
            deployment_name=self.config.LS_LOT_DEPLOYMENT_NAME,
            category_id_fn=lambda category_name, parent_id: self.category_utils.get_category_id(
                category_name=category_name, parent_category_id=parent_id
            ),
            parent_category_id=parent_category_id,
        )

    async def fetch_rental_classification(self, cleaned_description):
        """
        Classifies if the description is RENTAL or Not

        Args:
            - cleaned_description: Description cleaned for classification

        Returns:
            - stage_details (StageDetails)
            The StageDetails object contains the following attributes:
                - stage_number: 3
                - sub_stage_code: "1.2"
                - stage_name: Constants.CLASSIFICATION
                - sub_stage_name: Constants.RENTAL_CLASSIFIER
                - status
                - error_message
                - details
                    - end_time
                    - category (RENTAL or NON_RENTAL)
                    - category_id (Y or N)
                    - confidence_score
                    - description
        """
        stageDetails = StageDetails(
            stage_number=3, sub_stage_code="1.2", stage=StageNames.CLASSIFICATION, sub_stage=SubStageNames.RENTAL_CLASSIFIER
        )

        return await self.classifier_utils.classify_using_azure_custom_language_model(
            cleaned_description=cleaned_description,
            stage_details=stageDetails,
            project_name=self.config.LS_RENTAL_PROJECT_NAME,
            deployment_name=self.config.LS_RENTAL_DEPLOYMENT_NAME,
            category_id_fn=lambda cat, _: Constants.Y if cat == Constants.RENTAL else Constants.N,
        )

    # --- Helper Functions for Complete Matching ---
    def _prepare_log_context(self, ivce_dtl) -> str:
        """Creates a consistent logging context string."""
        ivce_dtl_uid = str(ivce_dtl.IVCE_DTL_UID) if hasattr(ivce_dtl, "IVCE_DTL_UID") else ""
        return f"({ivce_dtl_uid})" if ivce_dtl_uid else ""

    def _process_match_dataframe(self, match_df: pd.DataFrame) -> Dict[str, Any]:
        """Cleans a match DataFrame and converts it to a clean dictionary."""
        cleaned_df = clean_dataframe(match_df)
        if cleaned_df.empty:
            return {}
        match_dict = cleaned_df.iloc[0].to_dict()
        return clean_dictionary(match_dict)

    def _populate_success_details(self, match_result: Dict[str, Any], description: str) -> Dict[str, Any]:
        """Populates the details dictionary for a successful match."""
        details = {
            Logs.END_TIME: get_current_datetime_cst(),
            Logs.MFR_NAME: match_result.get("CleanName"),
            Logs.PRT_NUM: match_result.get("MfrPartNum"),
            Logs.AKS_PRT_NUM: match_result.get("AKPartNum"),
            Logs.UPC: match_result.get("UPC"),
            Logs.UNSPSC: match_result.get("UNSPSC"),
            Logs.ITEM_SOURCE_NAME: match_result.get("ItemSourceName"),
            Logs.DESC_SOURCE_NAME: match_result.get("DescSourceName"),
            Logs.DESCRIPTION: description,
            Logs.CONFIDENCE: {
                Logs.MFR_NAME: match_result.get("ManufacturerNameConfidenceScore"),
                Logs.PRT_NUM: match_result.get("PartNumberConfidenceScore"),
                Logs.UNSPSC: match_result.get("UNSPSCConfidenceScore"),
            },
        }
        return clean_dictionary(details)

    def _handle_processing_error(
        self,
        stage_details: "StageDetails",
        e: Exception,
        description: str,
        log_context: str,
        stage_log_prefix: str,
        message_prefix: str = "An unexpected error occurred",
    ):
        """Centralized error handling to populate stage_details."""
        error_message = f"{message_prefix}: {str(e)}"

        # Use the new parameter instead of a hardcoded string
        logger.error(f"{stage_log_prefix}{log_context}: {error_message}", exc_info=True)

        stage_details.status = Constants.ERROR_lower
        stage_details.details = {
            Logs.END_TIME: get_current_datetime_cst(),
            Constants.MESSAGE: error_message,
            Logs.DESCRIPTION: description,
        }

    async def check_complete_match(self, sdp, ai_engine_cache, ivce_dtl, stage_number, sub_stage_code):
        """
        Process invoice item description to find matching parts and manufacturers.

        This function implements the complete match stage of invoice processing:
        1. Extracts the item description from invoice detail.
        2. Analyzes the description to find matching parts and manufacturers.
        3. Determines the best match using confidence scores.
        4. Falls back to manufacturer-only match if no part match is found.

        Args:
            sdp: Database connection object used for querying data during the matching process.
            ivce_dtl: Invoice detail object containing item description (ITM_LDSC).
            ai_engine_cache.description_embedding: A numpy array containing the embedding vector of the item description
                                previously generated by check_semantic_match.

        Returns:
            - stage_details (StageDetails): An object encapsulating the results of the matching process.
            The StageDetails object contains the following attributes:
                - stage number: 5
                - sub_stage_code: "3.0"
                - stage_name (str): The name of the processing stage (Constants.COMPLETE_MATCH).
                - sub_stage_name (str): The name of the sub-stage (Constants.COMPLETE_MATCH).
                - status (str): The status of the processing (Constants.SUCCESS_lower or Constants.ERROR_lower).
                - match_result (Dict or None): A dictionary containing the best matching item,
                or None if no match was found. The dictionary includes only non-empty values for:
                    - ItemID: Unique identifier for the matched item.
                    - DescriptionID: Identifier for the matched description.
                    - MfrPartNum: Manufacturer part number of the matched item.
                    - MfrName: Name of the manufacturer.
                    - CleanName: Cleaned version of the item description.
                    - UncleanName: Original uncleaned version of the item description.
                    - MfrNameMatchType: The type of manufacturer name match.
                    - PartNumberConfidenceScore: Confidence score for the part number match (float).
                    - ManufacturerNameConfidenceScore: Confidence score for the manufacturer name match (float).
                    - UNSPSCConfidenceScore: Confidence score for the UNSPSC code match (float).
                    - DescriptionSimilarity: Cosine similarity score between the invoice item description and the matched
                    item description (float).
                - matched_mfrs (Dict[str, str]): Dictionary of manufacturers found in the description,
                mapping from clean manufacturer names to their variants found in the text.
                - details (Dict): A dictionary containing additional details about the processing.

        Raises:
            InvoiceProcessingError: Raised if there is an error during the processing of the invoice item
            description, such as issues with database queries or analysis logic.

            Notes:
                - If the item description is empty or invalid, the function logs a warning and returns early
                with `match_result` set to None.
                - This function now accepts a description_embedding parameter that may have been generated by
                the check_semantic_match function.
                - All empty, null, NaN values are removed from the returned dictionaries.
        """
        stage_details = StageDetails(
            stage_number=stage_number,
            sub_stage_code=sub_stage_code,
            stage=StageNames.COMPLETE_MATCH,
            sub_stage=SubStageNames.COMPLETE_MATCH,
        )
        log_context = self._prepare_log_context(ivce_dtl)
        log_prefix = "EXACT MATCHING"
        raw_description = ivce_dtl.ITM_LDSC
        cleaned_description = clean_description(raw_description) if raw_description else ""

        try:
            # Step 1: Validate cleaned description
            if not cleaned_description:
                logger.warning(f"EXACT MATCHING{log_context}: Invoice item description is empty or becomes empty after cleaning")
                stage_details.status = Constants.ERROR_lower
                stage_details.details = {Constants.MESSAGE: "Invoice item description is empty or invalid after cleaning"}
                return stage_details, ai_engine_cache

            logger.debug(
                f"EXACT MATCHING{log_context}: Processing invoice item: {raw_description} (cleaned: {cleaned_description})"
            )

            # Step 2: Analyze description to find potential matches
            results_df, mfrs_dict, full_mfr_data, _, uipnc_list = await analyze_description(
                sdp,
                self.llms,
                cleaned_description,
                self.config.exact_match_source,
                self.search_utils,
                ai_engine_cache.description_embedding,
            )
            stage_details.matched_mfrs = clean_dictionary(mfrs_dict)
            ai_engine_cache.manufacturer_aliases = stage_details.matched_mfrs
            logger.debug(
                f"EXACT MATCHING{log_context}: Found {len(results_df)} potential parts and {len(mfrs_dict)} manufacturers."
            )

            match_result = None
            is_verified = False
            is_mfr_clean = False
            is_mfr_direct = False

            # Step 3: Determine best match from parts or fallback to manufacturer
            matched_db_description = None  # Variable to hold the description
            if not results_df.empty:
                best_match_df = best_match_score(results_df, mfrs_dict, cleaned_description, full_mfr_data, uipnc_list)
                if best_match_df is not None and not best_match_df.empty:
                    logger.debug(f"EXACT MATCHING{log_context}: Best match found from part data.")
                    is_verified = bool(best_match_df.iloc[0]["IsVerified"])
                    matched_db_description = best_match_df.iloc[0].get("ItemDescription", None)
                    match_result = self._process_match_dataframe(best_match_df)

            if match_result is None and mfrs_dict:
                logger.debug(f"EXACT MATCHING{log_context}: No part match found, attempting manufacturer-only match.")
                mfr_match_df = process_manufacturer_dict(mfrs_dict, cleaned_description)
                if mfr_match_df is not None:
                    match_result = self._process_match_dataframe(mfr_match_df)

            # Step 4: Finalize stage details based on outcome
            if match_result:
                is_mfr_clean = bool(match_result.get("CleanName"))
                is_mfr_direct = (
                    match_result.get("ManufacturerMatchRelationship", MfrRelationshipType.NOT_EQUIVALENT.name)
                    == MfrRelationshipType.DIRECT.name
                )
                stage_details.match_result = match_result
                stage_details.details = self._populate_success_details(match_result, cleaned_description)
                stage_details.status = Constants.SUCCESS_lower

                # Explicitly add the matched DB description to the details for the validator stage.
                # Use the safe variable to add the description. It will be None if no part match was found.
                stage_details.details["matched_description"] = matched_db_description

                log_message = json.dumps(match_result, default=str)
                logger.debug(f"EXACT MATCHING{log_context}: Final match result: {log_message}")
            else:
                logger.debug(f"EXACT MATCHING{log_context}: No match found for invoice item.")
                stage_details.match_result = {}
                stage_details.details = {Constants.MESSAGE: "No match found for invoice item"}
                stage_details.status = Constants.SUCCESS_lower

            # Add all flags to the details dictionary to ensure they are always present
            stage_details.details[Logs.IS_VERIFIED] = is_verified
            stage_details.details[Logs.IS_MFR_CLEAN] = is_mfr_clean
            stage_details.details[Logs.IS_MFR_DIRECT] = is_mfr_direct

            stage_details.details[Logs.DESCRIPTION] = cleaned_description
            stage_details.details[Logs.END_TIME] = get_current_datetime_cst()
            return stage_details, ai_engine_cache

        except InvoiceProcessingError as ipe:
            # Re-raising known, critical errors is often a good pattern
            self._handle_processing_error(
                stage_details,
                ipe,
                cleaned_description,
                log_context,
                stage_log_prefix=log_prefix,
                message_prefix="Failed to analyze description",
            )
            raise  # Re-raise the exception after logging details

        except Exception as e:
            # Catch any other unexpected errors
            self._handle_processing_error(stage_details, e, cleaned_description, log_context, stage_log_prefix=log_prefix)
            return stage_details, ai_engine_cache

    async def validate_context(self, sdp, ai_engine_cache, ivce_dtl, previous_stage_details, stage_number, sub_stage_code):
        """
        Validates the contextual relevance of a match from a previous stage (e.g., Complete Match)
        using a specialized LLM call that returns a structured Pydantic object.
        """
        stage_details = StageDetails(
            stage_number=stage_number,
            sub_stage_code=sub_stage_code,
            stage=StageNames.CONTEXT_VALIDATOR,
            sub_stage=SubStageNames.CONTEXT_VALIDATOR,
            is_validation_stage=True,
        )

        log_context = self._prepare_log_context(ivce_dtl)

        try:
            # 1. Extract required data from the previous stage's results
            original_invoice_text = ivce_dtl.ITM_LDSC
            prev_details = previous_stage_details.details

            matched_mfr = prev_details.get(Logs.MFR_NAME)
            matched_pn = prev_details.get(Logs.PRT_NUM)
            matched_description = prev_details.get("matched_description")

            # 2. Input validation (Defensive)
            if not (original_invoice_text and matched_mfr and matched_pn and matched_description):
                missing_fields = []
                if not original_invoice_text:
                    missing_fields.append("original_invoice_text")
                if not matched_mfr:
                    missing_fields.append("matched_mfr")
                if not matched_pn:
                    missing_fields.append("matched_pn")
                if not matched_description:
                    missing_fields.append("matched_description")

                logger.error(f"CONTEXT_VALIDATOR{log_context}: Missing required inputs: {missing_fields}")
                raise ValueError(f"Missing required inputs for context validation: {missing_fields}")

            # 3. Construct the prompt for the LLM
            prompt = Prompts.get_context_validator_prompt(
                invoice_text=original_invoice_text, mfr=matched_mfr, pn=matched_pn, db_description=matched_description
            )

            # 4. Call the LLM to get a structured, validated response
            validation_result: ValidationResult = await self.llms.get_structured_response(
                prompt=prompt, output_model=ValidationResult, client_type=LLMClientType.CONTEXT_VALIDATOR
            )

            # 5. Apply deterministic logic to the LLM's classification.
            is_direct_match_flag = validation_result.context_type == MatchContextType.DIRECT_MATCH

            # 6. Populate the details for this stage's log
            stage_details.status = Constants.SUCCESS_lower
            stage_details.details = {
                Logs.END_TIME: get_current_datetime_cst(),
                "is_direct_match": is_direct_match_flag,
                "context_type": validation_result.context_type.value,
                "reason": validation_result.reason,
                # Include original inputs for better auditability in the logs
                "input_invoice_text": original_invoice_text,
                "input_matched_description": matched_description,
                "validated_stage_number": previous_stage_details.stage_number,
                "validated_stage_name": previous_stage_details.stage_name,
            }
            stage_details.is_final_success = is_direct_match_flag
            logger.debug(f"CONTEXT_VALIDATOR{log_context}: Result: {validation_result.model_dump_json(indent=2)}")

            # Tag the primary stage as officially invalidated if the check failed.
            if not stage_details.is_final_success:
                if previous_stage_details.details.get(Logs.IS_VERIFIED) is True:
                    previous_stage_details.details[Logs.IS_VERIFIED] = False
                previous_stage_details.is_invalidated = True

        except Exception as e:
            logger.error(f"CONTEXT_VALIDATOR{log_context}: Error during processing: {str(e)}", exc_info=True)
            stage_details.status = Constants.ERROR_lower
            stage_details.details = {
                Logs.END_TIME: get_current_datetime_cst(),
                Constants.MESSAGE: f"Unexpected error during context validation: {str(e)}",
                "is_direct_match": False,  # Fail safely
            }
            stage_details.is_final_success = False

        return stage_details, ai_engine_cache

    # --- Helper Functions for Semantic Matching ---
    async def _generate_description_embedding(self, cleaned_description: str, log_context: str) -> np.ndarray:
        """Generates an embedding from the description, raising an error on failure."""
        logger.debug(f"SEMANTIC MATCHING{log_context}: Generating embedding from description.")
        try:
            embedding = self.llms.get_embeddings([cleaned_description])[0]
            logger.debug(f"SEMANTIC MATCHING{log_context}: Successfully generated embedding.")
            return embedding
        except Exception as e:
            # Wrap the exception to provide context before re-raising
            raise InvoiceProcessingError(f"Failed to generate embedding: {str(e)}") from e

    def _process_semantic_match_result(
        self, stage_details: "StageDetails", match_result: Dict[str, Any], cleaned_description: str, log_context: str
    ):
        """Populates stage_details for a successful semantic match."""
        cleaned_result = clean_dictionary(match_result)
        confidence_dict = clean_dictionary(
            {
                Logs.MFR_NAME: cleaned_result.get("ManufacturerNameConfidenceScore"),
                Logs.UNSPSC: cleaned_result.get("UNSPSCConfidenceScore"),
            }
        )

        stage_details.details = clean_dictionary(
            {
                Logs.IS_MFR_CLEAN: match_result.get("IsMfrClean", False),
                Logs.MFR_NAME: cleaned_result.get("MfrName"),
                Logs.UNSPSC: cleaned_result.get("UNSPSC"),
                Logs.DESCRIPTION: cleaned_description,
                Logs.CONFIDENCE: confidence_dict,
            }
        )

        stage_details.status = Constants.SUCCESS_lower
        log_message = json.dumps(cleaned_result, default=str)
        logger.debug(f"SEMANTIC MATCHING{log_context}: Match result: {log_message}")

    def _handle_no_semantic_match(self, stage_details: "StageDetails", cleaned_description: str, log_context: str):
        """Populates stage_details when no semantic match is found."""
        logger.debug(f"SEMANTIC MATCHING{log_context}: No semantic match found.")
        stage_details.status = Constants.SUCCESS_lower
        stage_details.details = {
            Logs.IS_MFR_CLEAN: False,
            Constants.MESSAGE: "No semantic match found",
            Logs.DESCRIPTION: cleaned_description,
        }

    async def check_semantic_match(self, sdp, ai_engine_cache, ivce_dtl, stage_number, sub_stage_code):
        """
        Process semantic matching using description embedding or by generating one from invoice description.

        This function implements the semantic match stage of invoice processing:
        1. Generates an embedding using the invoice item description.
        2. Performs semantic search to find matching items based on embedding similarity.
        3. Extracts manufacturer and UNSPSC information from the semantic search results.

        Args:
            sdp: Database connection object used for querying data during the matching process.
            ivce_dtl: Invoice detail object containing item description (ITM_LDSC).
            kwargs: not required for this function

        Returns:
            A tuple containing:
            - stage_details (StageDetails): An object encapsulating the results of the semantic matching process.
            The StageDetails object contains the following attributes:
                - stage number: 4
                - sub_stage_code: "2.0"
                - stage_name (str): The name of the processing stage (Constants.SEMANTIC_SEARCH).
                - sub_stage_name (str): The name of the sub-stage (Constants.SEMANTIC_SEARCH).
                - status (str): The status of the processing (Constants.SUCCESS_lower or Constants.ERROR_lower).
                - details (Dict): A dictionary containing additional details about the processing:
                If successful, contains only non-empty values for:
                    - Logs.MFR_NAME: Manufacturer name
                    - Logs.UNSPSC: UNSPSC code
                    - Logs.DESCRIPTION: Cleaned description
                    - Logs.CONFIDENCE: Dictionary containing confidence scores for manufacturer and UNSPSC
                If error occurs, contains:
                    - Constants.MESSAGE: Error message details
                    - Logs.DESCRIPTION: Cleaned description (if available)
            - description_embedding (numpy.ndarray or None): A Numpy array containing the embedding vector
            of the item description, which can be used in subsequent processing stages. If no embedding
            is generated (e.g., due to an empty description or processing errors), this will be None.

        Raises:
            InvoiceProcessingError: Raised if there is an error during the semantic matching process,
            such as issues with the search service or analysis logic.

        Notes:
            - If the item description is empty or invalid, the function logs a warning and returns early
            with description_embedding set to None.
            - All empty, null, NaN values are removed from the returned dictionaries.
        """
        stage_details = StageDetails(
            stage_number=stage_number,
            sub_stage_code=sub_stage_code,
            stage=StageNames.SEMANTIC_SEARCH,
            sub_stage=SubStageNames.SEMANTIC_SEARCH,
        )
        log_context = self._prepare_log_context(ivce_dtl)
        log_prefix = "SEMANTIC MATCHING"
        raw_description = ivce_dtl.ITM_LDSC
        cleaned_description = clean_description(raw_description) if raw_description else ""

        try:
            # Step 1: Validate description and generate embedding
            if not cleaned_description:
                logger.warning(f"SEMANTIC MATCHING{log_context}: Invoice item description is empty, cannot proceed.")
                stage_details.status = Constants.ERROR_lower
                stage_details.details = {Constants.MESSAGE: "Invoice item description is empty"}
                return stage_details, ai_engine_cache

            description_embedding = await self._generate_description_embedding(cleaned_description, log_context)
            ai_engine_cache.description_embedding = description_embedding

            # Step 2: Perform semantic search using the embedding
            logger.debug(f"SEMANTIC MATCHING{log_context}: Performing search with generated embedding.")

            # UPDATED: Unpack the tuple (result_data, raw_items)
            match_result, top_n_items = await semantic_match_by_description(
                description=description_embedding, azure_search_utils=self.search_utils, sdp=sdp
            )

            # UPDATED: Persist raw items to cache for the LLM
            if top_n_items:
                ai_engine_cache.semantic_search_results = top_n_items
                logger.debug(f"SEMANTIC MATCHING{log_context}: Cached {len(top_n_items)} raw items for LLM context.")

            # Step 3: Process the result (or lack thereof)
            if match_result:
                self._process_semantic_match_result(stage_details, match_result, cleaned_description, log_context)
            else:
                self._handle_no_semantic_match(stage_details, cleaned_description, log_context)

            stage_details.details[Logs.IS_VERIFIED] = False
            stage_details.details[Logs.END_TIME] = get_current_datetime_cst()
            return stage_details, ai_engine_cache

        except InvoiceProcessingError as ipe:
            # Catches specific, known processing errors (like embedding failure)
            self._handle_processing_error(
                stage_details,
                ipe,
                cleaned_description,
                log_context,
                stage_log_prefix=log_prefix,
                message_prefix="A processing error occurred",
            )
            return stage_details, ai_engine_cache

        except Exception as e:
            # Catches any other unexpected errors
            self._handle_processing_error(stage_details, e, cleaned_description, log_context, stage_log_prefix=log_prefix)
            return stage_details, ai_engine_cache

    async def extract_from_finetuned_llm(self, sdp, ai_engine_cache, ivce_dtl, stage_number, sub_stage_code):
        """
        Extracts the results fine-tuned LLM

        Args:
            - sdp: SDP object for DB connections
            - ai_engine_cache.cleaned_description_for_llm: Invoice item description cleaned for LLM

        Returns:
            - stage_details (StageDetails)
            The StageDetails object contains the following attributes:
                - stage_number: 6
                - - sub_stage_code: "4.0"
                - stage_name: Constants.FINETUNED_LLM
                - sub_stage_name: Constants.FINETUNED_LLM
                - status: success or error
                - is_mfr_clean_flag: Has the clean Manufacturer Name mapping found?
                - details
                    - manufacturer_name, unclean_manufacturer_name, part_number, unspsc, confidences, description
                    - confidences has three key-value pairs. Keys: [manufacturer_name, part_number, unspsc]
        """
        stage_details = StageDetails(
            stage_number=stage_number,
            sub_stage_code=sub_stage_code,
            stage=StageNames.FINETUNED_LLM,
            sub_stage=SubStageNames.FINETUNED_LLM,
        )

        log_context = self._prepare_log_context(ivce_dtl)
        original_description = ai_engine_cache.cleaned_description_for_llm

        try:
            # 1. Select Prompt Template & LLM Model
            # We explicitly use the raw 'aoai_gpt4o_finetuned' client here because we require
            # raw 'logprobs' for confidence scoring.
            llm_to_use = self.llms.aoai_gpt4o_finetuned

            if ivce_dtl.is_special_case and ivce_dtl.special_case_type == SpecialCases.CASE_1:
                prompt = Prompts.get_fine_tuned_llm_prompt_for_unspsc()
            else:
                prompt = Prompts.get_fine_tuned_llm_prompt()

            # Template expects {description} which will hold the fully constructed context+input
            prompt_template = PromptTemplate(input_variables=["description"], template=prompt)
            chain = prompt_template | llm_to_use

            # 2. Execute Hardened Request
            # Passes raw data; executor handles dynamic prompt construction, retries, and validation.
            response, results_json, trace_metadata = await execute_hardened_llm_request(
                llm_instance=self.llms,
                chain=chain,
                target_description=original_description,
                semantic_search_results=ai_engine_cache.semantic_search_results,
                manufacturer_aliases=ai_engine_cache.manufacturer_aliases,
            )

            # 3. Confidence Calculation
            details, is_mfr_clean_flag = await calculate_confidence_for_finetuned_llm(
                sdp=sdp, response=response, results_json=results_json, fields=ivce_dtl.fields
            )

            details[Logs.DESCRIPTION] = original_description
            details[Logs.END_TIME] = get_current_datetime_cst()

            stage_details.status = Constants.SUCCESS_lower
            if is_mfr_clean_flag is not None:
                details[Logs.IS_MFR_CLEAN] = is_mfr_clean_flag

            details[Logs.IS_VERIFIED] = False
            stage_details.details = details

            # 4. Attach Trace Metadata (Sibling to details)
            stage_details.rag_trace = trace_metadata

            logger.debug(f"FINETUNED LLM{log_context} stage output-\n{stage_details.details}")

        except ValueError as ve:
            # Clean logging for validation failures
            logger.warning(f"FINETUNED LLM{log_context}: Validation failed after retries. Reason: {str(ve)}")
            stage_details.status = Constants.ERROR_lower
            stage_details.details = {
                Logs.IS_MFR_CLEAN: False,
                Logs.END_TIME: get_current_datetime_cst(),
                Constants.MESSAGE: f"Validation failed: {str(ve)}",
                Logs.DESCRIPTION: original_description,
            }

        except Exception as e:
            # Full logging for unexpected crashes
            logger.error(f"FINETUNED LLM{log_context}: Unexpected error during processing: {str(e)}", exc_info=True)
            stage_details.status = Constants.ERROR_lower
            stage_details.details = {
                Logs.IS_MFR_CLEAN: False,
                Logs.END_TIME: get_current_datetime_cst(),
                Constants.MESSAGE: f"Unexpected error: {str(e)}",
                Logs.DESCRIPTION: original_description,
            }

        return stage_details, ai_engine_cache

    async def extract_with_ai_agents_from_websearch(self, sdp, ai_engine_cache, ivce_dtl, stage_number, sub_stage_code):
        """
        Extracts the results using web search agent using priority websites and general web

        Args:
            - sdp: SDP object for DB connections
            - ai_engine_cache.cleaned_description_for_llm: Invoice item description

        Returns:
            - stage_details (StageDetails)
            The StageDetails object contains the following attributes:
                - stage_number: 7
                - sub_stage_code: "5.0"
                - stage_name: Constants.EXTRACTION_WITH_LLM_AND_WEBSEARCH
                - sub_stage_name: Constants.AZURE_AI_AGENT_WITH_BING_SEARCH
                - status: success or error
                - is_mfr_clean_flag: Has the clean Manufacturer Name mapping found?
                - details
                    - manufacturer_name, unclean_manufacturer_name, part_number, unspsc, confidences, description, web_search_url
                    - confidences has three key-value pairs. Keys: [manufacturer_name, part_number, unspsc]
                - web_results_ranking_with_confidences: Json
        """
        description = ai_engine_cache.cleaned_description_for_llm
        logger.debug(f"AGENT WEB_SEARCH: Starting stage for description: '{description[:50]}...'")
        stage_details = StageDetails(
            stage_number=stage_number,
            sub_stage_code=sub_stage_code,
            stage=StageNames.EXTRACTION_WITH_LLM_AND_WEBSEARCH,
            sub_stage=SubStageNames.AZURE_AI_AGENT_WITH_BING_SEARCH,
        )
        search_thread_id = None
        rank_thread_id = None
        df = pd.DataFrame()

        try:
            # --- Step 1: Web Search Agent Execution ---
            ai_websearch_agent = await self.agents.get_agent_with_bing_search_async(
                self.config.AZ_AGENT_GBS_AGENT_DEPLOYMENT,
                Prompts.get_web_search_system_prompt(),
                self.config.AZ_AGENT_GBS_API_DEPLOYMENT,
            )

            # Select the appropriate prompt based on whether this is a special case (e.g., UNSPSC only)
            if ivce_dtl.is_special_case and ivce_dtl.special_case_type == SpecialCases.CASE_1:
                search_prompt = Prompts.get_web_search_prompt_for_unspsc(config=self.config, description=description)
            else:
                search_prompt = Prompts.get_web_search_prompt(config=self.config, description=description)

            # The run_agent_and_get_json_with_retry helper function handles SDK-level retries and also
            # retries on certain data validation failures (e.g., missing 'Source' key).
            # We wrap this call in a try/except block to interpret its final outcome.
            try:
                web_search_results_json, _, search_thread_id, _ = await run_agent_and_get_json_with_retry(
                    agents=self.agents,
                    prompt=search_prompt,
                    agent=ai_websearch_agent,
                    agent_type="web search",
                    validator=validate_search_results_schema,
                )
                df = pd.DataFrame(
                    [web_search_results_json] if isinstance(web_search_results_json, dict) else web_search_results_json
                )

                # --- Step 2: Clean and Validate Search Results ---
                if not df.empty and "Source" in df.columns:
                    df.columns = [str(col).strip() for col in df.columns]
                    # Clean the Source column and drop any rows that ultimately failed to return a valid URL.
                    df["Source"] = df["Source"].fillna(Constants.EMPTY_STRING).astype(str).str.strip()
                    df["Source"] = df["Source"].replace(
                        [Constants.EMPTY_STRING, "None", "null", "", Constants.MISSING_SOURCE], pd.NA, regex=False
                    )
                    df.dropna(subset=["Source"], inplace=True)

                    if not df.empty:
                        relevant_cols = [col for col in df.columns if col != "ID"]
                        if relevant_cols:
                            df = df.drop_duplicates(subset=relevant_cols, keep="first")
                        df.reset_index(drop=True, inplace=True)
                    logger.debug(
                        f"AGENT WEB_SEARCH: Web search DataFrame cleaned. Final shape: {df.shape}. Thread: {search_thread_id}"
                    )

            # This 'except' block is crucial for interpreting the final failure from the helper function.
            except InvalidJsonResponseError as json_err:
                raw_response = str(json_err.response or "").strip()
                parsed_for_empty_check = None

                # Perform a more lenient secondary check. The main parser has already failed.
                # We are now only trying to determine if the response is a valid "no results" case.
                try:
                    # extract_json can handle markdown fences and other text around the JSON.
                    parsed_for_empty_check = extract_json(raw_response)
                except Exception:
                    # If even the lenient parser fails, it's definitely not a simple empty JSON.
                    pass

                # Case 1: The lenient parser found an empty list/dict, OR the response is purely conversational.
                # This is a SUCCESS case.
                if parsed_for_empty_check in ([], {}) or ("[" not in raw_response and "{" not in raw_response):
                    logger.warning(
                        "AGENT WEB_SEARCH: Agent returned a valid 'no results' response. "
                        f"Response: '{raw_response}'. Thread: {json_err.thread_id}"
                    )
                    df = pd.DataFrame()  # Ensure dataframe is empty for subsequent checks.
                    stage_details.status = Constants.SUCCESS_lower

                    message = ""
                    if parsed_for_empty_check is not None:
                        message = "Agent returned a valid but empty JSON (no results found)."
                    else:
                        message = f"Agent conversationally reported no results: {raw_response}"

                    stage_details.details = {
                        "search_thread": json_err.thread_id,
                        Logs.IS_MFR_CLEAN: False,
                        Logs.END_TIME: get_current_datetime_cst(),
                        Constants.MESSAGE: message,
                        Logs.DESCRIPTION: description,
                    }
                # Case 2: The lenient parser failed OR found non-empty data. The response is genuinely malformed.
                # This is a HARD FAILURE.
                else:
                    logger.error(
                        "AGENT WEB_SEARCH: Process for search agent failed with malformed JSON that was not a valid empty"
                        f" response. Thread: {json_err.thread_id}"
                    )
                    stage_details.status = Constants.ERROR_lower
                    stage_details.details = {
                        "search_thread": json_err.thread_id,
                        Logs.IS_MFR_CLEAN: False,
                        Logs.END_TIME: get_current_datetime_cst(),
                        Constants.MESSAGE: f"Invalid JSON response from agent: {str(json_err)} Agent response: {raw_response}",
                        Logs.DESCRIPTION: description,
                    }

            # --- Step 3: Handle No Results & Proceed to Ranking ---

            # If the dataframe is empty after cleaning (and no error occurred), this is a successful "no results" outcome.
            if df.empty and stage_details.status != Constants.ERROR_lower:
                logger.debug(
                    "AGENT WEB_SEARCH: DataFrame is empty. No valid results found. Populating success details. "
                    f"Thread: {search_thread_id}"
                )
                # Ensure details are populated if they haven't been already by the exception handler.
                if not stage_details.details:
                    stage_details.status = Constants.SUCCESS_lower
                    stage_details.details = {
                        Logs.SEARCH_THREAD_ID: search_thread_id,
                        Logs.IS_MFR_CLEAN: False,
                        Logs.END_TIME: get_current_datetime_cst(),
                        Constants.MESSAGE: "No valid results found in web search after cleaning.",
                        Logs.DESCRIPTION: description,
                    }

            # Proceed to the ranking step only if we have valid results and no prior error has occurred.
            elif not df.empty and stage_details.status != Constants.ERROR_lower:
                try:
                    # --- Step 3a. Ranking Agent Call ---
                    logger.debug(
                        f"AGENT WEB_SEARCH: Preparing for ranking agent call ({len(df)} results). Thread: {search_thread_id}"
                    )
                    web_search_results_json_str = df.to_json(orient="records", indent=2)

                    ai_ranking_agent = await self.agents.get_agent_async(
                        self.config.AZ_AGENT_AGENT_DEPLOYMENT,
                        Prompts.get_web_results_ranking_system_prompt(),
                        self.config.AZ_AGENT_API_DEPLOYMENT,
                    )

                    if ivce_dtl.is_special_case and ivce_dtl.special_case_type == SpecialCases.CASE_1:
                        ranking_prompt = Prompts.get_web_results_ranking_prompt_for_unspsc(
                            description=description, results_json=web_search_results_json_str
                        )
                    else:
                        ranking_prompt = Prompts.get_web_results_ranking_prompt(
                            description=description, results_json=web_search_results_json_str
                        )

                    # The helper function is called again for the ranking agent.
                    web_results_ranking_json, _, rank_thread_id, _ = await run_agent_and_get_json_with_retry(
                        agents=self.agents,
                        prompt=ranking_prompt,
                        agent=ai_ranking_agent,
                        agent_type="ranking",
                        validator=validate_search_results_schema,
                    )

                    # --- Step 4. Confidence Calculation & Best Result Selection ---
                    df_with_confidences = await calculate_confidence_for_web_search_results_with_ranking(
                        sdp=sdp, web_results_ranking_json=web_results_ranking_json, ivce_dtl=ivce_dtl
                    )
                    logger.debug(
                        f"AGENT WEB_SEARCH: Confidence calculation completed. Result is None: {df_with_confidences is None}. "
                        f"Thread: {rank_thread_id}"
                    )

                    # Handle case where confidence calculation yields no usable results.
                    if df_with_confidences is None:
                        logger.debug(
                            "AGENT WEB_SEARCH: Confidence calculation returned None. Populating details. "
                            f"Thread: {rank_thread_id}"
                        )
                        stage_details.status = Constants.SUCCESS_lower
                        stage_details.details = {
                            Logs.SEARCH_THREAD_ID: search_thread_id,
                            Logs.IS_MFR_CLEAN: False,
                            Logs.RANK_THREAD_ID: rank_thread_id,
                            Logs.END_TIME: get_current_datetime_cst(),
                            Constants.MESSAGE: "No results after confidence calculation and ranking.",
                            Logs.DESCRIPTION: description,
                        }
                    else:
                        # Select the single best result from the scored and ranked list.
                        df_with_confidences["original_desc"] = description
                        best_result_df, is_mfr_clean_flag = await get_higher_confidence_web_search_result(
                            sdp=sdp, df=df_with_confidences, fields=ivce_dtl.fields
                        )
                        logger.debug(
                            "AGENT WEB_SEARCH: Best result selection completed. Found: "
                            f"{best_result_df is not None and not best_result_df.empty}. CleanFlag: {is_mfr_clean_flag}. "
                            f"Thread: {rank_thread_id}"
                        )

                        # --- Step 5. Prepare Final Success Response ---
                        logger.debug(f"AGENT WEB_SEARCH: Final success. Populating stage details. Thread: {rank_thread_id}")
                        stage_details.status = Constants.SUCCESS_lower
                        best_result_dict = best_result_df if isinstance(best_result_df, dict) else best_result_df.to_dict()
                        details = {
                            Logs.END_TIME: get_current_datetime_cst(),
                            Logs.DESCRIPTION: description,
                            # Direct access is safe here because rows without a valid source were dropped earlier.
                            Logs.WEB_SEARCH_URL: best_result_dict["Source"],
                            Logs.SEARCH_THREAD_ID: search_thread_id,
                            Logs.RANK_THREAD_ID: rank_thread_id,
                            Logs.IS_MFR_CLEAN: is_mfr_clean_flag if is_mfr_clean_flag is not None else False,
                            Logs.CONFIDENCE: {},
                        }

                        if Logs.MFR_NAME in ivce_dtl.fields:
                            details.update(
                                {
                                    Logs.MFR_NAME: best_result_dict["ManufacturerName"],
                                    Logs.UNCLN_MFR_NAME: best_result_dict["UncleanManufacturerName"],
                                }
                            )
                            details[Logs.CONFIDENCE].update({Logs.MFR_NAME: best_result_dict["MfrScore"]})

                        if Logs.PRT_NUM in ivce_dtl.fields:
                            details.update({Logs.PRT_NUM: best_result_dict["PartNumber"]})
                            details[Logs.CONFIDENCE].update({Logs.PRT_NUM: best_result_dict["PartNumberScore"]})

                        if Logs.UNSPSC in ivce_dtl.fields:
                            details.update({Logs.UNSPSC: best_result_dict["UNSPSC"]})
                            details[Logs.CONFIDENCE].update({Logs.UNSPSC: best_result_dict["UnspscScore"]})

                        if "UPC" in best_result_dict:
                            details[Logs.UPC] = best_result_dict["UPC"]

                        details[Logs.IS_VERIFIED] = False
                        stage_details.details = details
                        stage_details.best_result = best_result_dict
                        stage_details.web_results_ranking_with_confidences = df_with_confidences.to_dict(orient="records")
                        logger.debug(f"AZURE_AI_AGENT_WITH_BING_SEARCH -\n{stage_details.details}")

                # This 'except' block catches the final failure from the RANKING agent after all its retries.
                except InvalidJsonResponseError as json_err:
                    logger.debug(
                        "AGENT WEB_SEARCH: Process for ranking agent failed. Populating error details. "
                        f"Thread: {json_err.thread_id}"
                    )
                    stage_details.status = Constants.ERROR_lower
                    stage_details.details = {
                        Logs.SEARCH_THREAD_ID: search_thread_id,
                        Logs.RANK_THREAD_ID: json_err.thread_id,
                        Logs.IS_MFR_CLEAN: False,
                        Logs.END_TIME: get_current_datetime_cst(),
                        Constants.MESSAGE: f"Ranking agent failed: {str(json_err)} Agent response: {str(json_err.response)}",
                        Logs.DESCRIPTION: description,
                    }

        except Exception as e:
            # --- Final Catch-All for Unexpected Errors ---
            logger.error(f"AGENT WEB_SEARCH: Caught unexpected error in main try block: {str(e)}", exc_info=True)
            stage_details.status = Constants.ERROR_lower
            stage_details.details = {
                Logs.END_TIME: get_current_datetime_cst(),
                Constants.MESSAGE: f"Unexpected error: {str(e)}",
                Logs.DESCRIPTION: description,
                Logs.IS_MFR_CLEAN: False,
            }

        logger.debug(f"AGENT WEB_SEARCH: Returning final stage_details. Status: {stage_details.status}")
        return stage_details, ai_engine_cache


class StageUtils:

    def __init__(self):
        # Load thresholds file to check if next stage required each time
        self.thresholds = load_yaml(LocalFiles.THRESHOLDS_FILE)

        # Load confidences file to boost with RPA values
        self.confidences = load_yaml(LocalFiles.CONFIDENCES_FILE)

    # def are_both_fields_matching(self, *, field1, field2, field_key, ai_engine_cache):
    #     if field_key == Logs.MFR_NAME:
    #         relationship_type = mfr_eq_type(
    #             clean_name_1=field1, clean_name_2=field2, manufacturer_data_dict=ai_engine_cache.manufacturer_data_dict
    #         )
    #         return False if relationship_type == MfrRelationshipType.NOT_EQUIVALENT else True
    #     else:
    #         return True if get_alphanumeric(field1) == get_alphanumeric(field2) else False

    def are_both_fields_matching(self, *, field1, field2, field_key, ai_engine_cache):
        """
        Compares two field values for agreement. This version is hardened to handle None.
        """
        # Handle the special case for manufacturer name comparison first.
        if field_key == Logs.MFR_NAME:
            relationship_type = mfr_eq_type(
                clean_name_1=field1, clean_name_2=field2, manufacturer_data_dict=ai_engine_cache.manufacturer_data_dict
            )
            return bool(relationship_type)

        # For all other fields (like part_number, unspsc), apply the None-aware logic.
        if field1 is None and field2 is None:
            # If both are None, they are considered "matching" in this context.
            return True
        if field1 is None or field2 is None:
            # If one is None and the other is a string, they do not match.
            return False

        # If both are non-None strings, proceed with the original alphanumeric comparison.
        return get_alphanumeric(field1) == get_alphanumeric(field2)

    def consolidate_null_field(self, *, stage_results, stage_details, field_key, conf_key, stage_key, conf_stage_key):
        """
        Generic method to consolidate a field's value from previous stages to avoid writing null value.
        It fetches the best value from previous stages with higher confidence.

        Args:
            - stage_details: StageDetails object of the last stage run
            - field_key: Key to access/update the field value (e.g., Manufacturer Name, UNSPSC)
            - conf_key: Key to update the field's confidence score
            - stage_key: Key to update the field's stage
            - conf_key: Key to update the field's confidence stage

        Returns:
            - Tuple of (final value, confidence score, stage name, confidence stage name where it was found)
        """
        try:
            consolidated = {}
            final_value = Constants.UNDEFINED
            final_stage = Constants.UNDEFINED  # f"{stage_details.stage_name} - {stage_details.sub_stage_name}"
            final_conf = 0
            final_conf_stage = copy(final_stage)

            for number, st_details in stage_results.results.items():
                if 1 < number < stage_details.stage_number:
                    details = st_details.get("details", {})

                    if field_key in details and details[Logs.CONFIDENCE][field_key] > final_conf:

                        final_value = details[field_key]
                        final_conf = details[Logs.CONFIDENCE][field_key]
                        final_stage = f"{st_details['stage_name']} - {st_details['sub_stage_name']}"
                        final_conf_stage = copy(final_stage)

            consolidated[field_key] = final_value
            consolidated[stage_key] = final_stage
            consolidated[conf_key] = final_conf
            consolidated[conf_stage_key] = final_conf_stage
            if final_value != Constants.UNDEFINED:
                stage_details.details[field_key] = final_value
                stage_details.details[Logs.CONFIDENCE][field_key] = final_conf

        except Exception as e:
            logger.error(f"Error occurred in consolidate_null_field() {str(e)}", exc_info=True)
        return consolidated, stage_details

    # def consolidate_all_fields_confidences(self, *, stage_results, stage_details, fields, ivce_dtl, ai_engine_cache):
    #     consolidated = {}
    #     try:
    #         if stage_details is None:
    #             return stage_details, consolidated

    #         # Consolidate previous stages' confidence values
    #         for field_key, conf_key, conf_stage_key in fields:
    #             # If field is not present in current stage, then you can't compare with previous stages
    #             if field_key not in stage_details.details:
    #                 continue

    #             conf_value, conf_stage, has_rpa_boosted_conf = self.consolidate_field_confidence(
    #                 stage_results=stage_results,
    #                 stage_details=stage_details,
    #                 field_key=field_key,
    #                 ivce_dtl=ivce_dtl,
    #                 ai_engine_cache=ai_engine_cache,
    #                 consider_rpa=True,
    #             )
    #             # stage_details.details[Logs.CONFIDENCE][field_key] = conf_value
    #             consolidated["has_rpa_boosted_conf"] = has_rpa_boosted_conf
    #             consolidated[conf_key] = conf_value
    #             consolidated[conf_stage_key] = conf_stage
    #     except Exception as e:
    #         logger.error(f"Error occurred in consolidate_all_fields_confidences() {str(e)}", exc_info=True)

    #     return stage_details, consolidated

    def consolidate_all_fields_confidences(self, *, stage_results, stage_details, fields, ivce_dtl, ai_engine_cache):
        """
        Orchestrates the consolidation of confidence scores and data enrichment.
        Refactored to separate concerns into specific helper methods.
        """
        if stage_details is None or not stage_details.details:
            return stage_details, {}

        # 1. Initialization
        current_details = stage_details.details
        consolidated = self._initialize_consolidated_dict(current_details, fields, stage_details)

        # --- Propagate IS_VERIFIED (Reverted to Simple Propagation) ---
        if Logs.IS_VERIFIED in current_details:
            consolidated[Logs.IS_VERIFIED] = current_details[Logs.IS_VERIFIED]

        # 2. RPA Boosting
        self._apply_rpa_boosting(current_details, consolidated, fields, ivce_dtl)

        # UNSPSC Specifics: Get Threshold for CURRENT stage
        current_unspsc_threshold = self._get_field_threshold(stage_details, Logs.UNSPSC)

        # 3. Iterate through prior stages
        for prior_stage_number, prior_result in stage_results.results.items():
            if prior_stage_number >= stage_details.stage_number or not prior_result.get("details"):
                continue

            prior_details = prior_result["details"]
            stage_provenance = f"{prior_result['stage_name']} - {prior_result['sub_stage_name']}"

            # Pass A: Confidence Boosting
            self._apply_confidence_boosting(
                current_details,
                prior_details,
                consolidated,
                fields,
                stage_provenance,
                stage_details.stage_name,
                ai_engine_cache,
                ivce_dtl,
            )

            # Pass B: Strict Data Enrichment
            if self._check_strict_agreement(current_details, prior_details, ai_engine_cache):
                self._apply_data_enrichment(
                    current_details, prior_details, consolidated, stage_provenance, stage_details.stage_name, fields, ivce_dtl
                )

            # Pass C: Heuristic UNSPSC Enrichment
            # Logic: If the prior stage inherited this UNSPSC (it has a provenance key),
            # we ignore it. We only evaluate values against the thresholds of their ORIGINAL source stage,
            # which we have already processed (or will process) in this loop.
            if Logs.STAGE_UNSPSC in prior_details:
                continue

            # If we are here, the UNSPSC is native to the prior stage.
            # Get Threshold for the PRIOR stage
            try:
                prior_stage_name = prior_result.get("stage_name")
                prior_sub_stage_name = prior_result.get("sub_stage_name")
                prior_unspsc_threshold = (
                    self.thresholds.get(prior_stage_name, {}).get(prior_sub_stage_name, {}).get(Logs.UNSPSC, 0)
                )
            except AttributeError:
                prior_unspsc_threshold = 0

            # Apply Logic: Must meet BOTH current and prior thresholds
            self._apply_heuristic_unspsc_enrichment(
                current_details,
                prior_details,
                consolidated,
                stage_provenance,
                stage_details.stage_name,
                current_unspsc_threshold,
                prior_unspsc_threshold,
                fields,
                ivce_dtl,
            )

        # 4. Fallback UNSPSC Enrichment
        self._apply_fallback_unspsc_enrichment(stage_results, current_details, consolidated, stage_details, fields)

        return stage_details, consolidated

    # --- Helper Methods ---

    def _get_field_threshold(self, stage_details, field_key):
        """Safely retrieves the threshold for a specific field in the current stage."""
        try:
            return self.thresholds.get(stage_details.stage_name, {}).get(stage_details.sub_stage_name, {}).get(field_key, 0)
        except Exception:
            return 0

    def _apply_heuristic_unspsc_enrichment(
        self,
        current_details,
        prior_details,
        consolidated,
        stage_provenance,
        current_stage_name,
        current_threshold,
        prior_threshold,
        fields,
        ivce_dtl,
    ):
        """
        Pass 3: Fills UNSPSC if missing.
        Strict Condition: Value must meet thresholds of BOTH the Current Stage AND the Prior (Source) Stage.
        """
        field_key = Logs.UNSPSC

        # Only proceed if Current is Missing AND Prior has it
        curr_val = current_details.get(field_key)
        prior_val = prior_details.get(field_key)

        if self._is_valid_value(curr_val):
            return  # Already has a value

        if self._is_valid_value(prior_val):
            # Type check to handle Classification stage (float confidence)
            prior_conf_obj = prior_details.get(Logs.CONFIDENCE, {})
            if not isinstance(prior_conf_obj, dict):
                return

            prior_conf = prior_conf_obj.get(field_key, 0)

            # Double Threshold Verification
            if prior_conf >= current_threshold and prior_conf >= prior_threshold:
                uid = getattr(ivce_dtl, "IVCE_DTL_UID", "UNKNOWN_UID")
                logger.debug(
                    f"[{current_stage_name}]({uid}) Heuristic Enrichment for '{field_key}' from {stage_provenance}:"
                    f" {prior_val} (Conf: {prior_conf} meets Current:{current_threshold} & Prior:{prior_threshold})"
                )

                # Copy Data
                self._copy_enriched_field(
                    current_details, consolidated, field_key, prior_val, prior_conf, stage_provenance, fields
                )

    def _apply_fallback_unspsc_enrichment(self, stage_results, current_details, consolidated, stage_details, fields):
        """
        Pass 4: Last resort. If UNSPSC is still missing, pick the one with the Highest Confidence from any prior stage.
        """
        field_key = Logs.UNSPSC
        if self._is_valid_value(current_details.get(field_key)):
            return

        best_val = None
        best_conf = -1
        best_provenance = None

        for prior_num, prior_result in stage_results.results.items():
            if prior_num >= stage_details.stage_number or not prior_result.get("details"):
                continue

            p_details = prior_result["details"]
            p_val = p_details.get(field_key)

            # Type check to handle Classification stage (float confidence)
            p_conf_obj = p_details.get(Logs.CONFIDENCE, {})
            if not isinstance(p_conf_obj, dict):
                continue

            p_conf = p_conf_obj.get(field_key, 0)

            if self._is_valid_value(p_val) and p_conf > best_conf:
                best_val = p_val
                best_conf = p_conf
                best_provenance = f"{prior_result['stage_name']} - {prior_result['sub_stage_name']}"

        if best_val:
            logger.debug(
                f"[{stage_details.stage_name}] Fallback Enrichment for '{field_key}' from {best_provenance}: {best_val} (Max"
                f" Conf: {best_conf})"
            )
            self._copy_enriched_field(current_details, consolidated, field_key, best_val, best_conf, best_provenance, fields)

    def _copy_enriched_field(self, current_details, consolidated, field_key, value, confidence, provenance, fields):
        """Shared logic to commit enriched data to details and consolidated results."""
        # 1. Update consolidated output
        consolidated[field_key] = value

        # 2. Persist to current stage details (Value)
        current_details[field_key] = value

        # 3. Handle Confidence
        if Logs.CONFIDENCE not in current_details:
            current_details[Logs.CONFIDENCE] = {}
        current_details[Logs.CONFIDENCE][field_key] = confidence

        # Update specific tracking keys
        for f_key, c_key, cs_key in fields:
            if f_key == field_key:
                consolidated[c_key] = confidence
                consolidated[cs_key] = provenance

        # 4. Update Field Provenance
        stage_key_name = Logs.MAPPING.get(field_key, [None, f"{field_key}_stage"])[1]
        consolidated[stage_key_name] = provenance
        current_details[stage_key_name] = provenance

    def _initialize_consolidated_dict(self, current_details, fields, stage_details):
        """Sets up the initial consolidated dictionary with values from the current stage."""
        consolidated = {"has_rpa_boosted_conf": False}
        provenance = f"{stage_details.stage_name} - {stage_details.sub_stage_name}"

        for field_key, conf_key, conf_stage_key in fields:
            if field_key in current_details:
                consolidated[conf_key] = current_details.get(Logs.CONFIDENCE, {}).get(field_key)
                consolidated[conf_stage_key] = provenance
        return consolidated

    def _is_valid_value(self, value):
        """Checks if a value is non-null and non-empty. Guards against None types."""
        if value is None:
            return False
        return is_not_null(value) and is_not_empty(value)

    def _apply_confidence_boosting(
        self,
        current_details,
        prior_details,
        consolidated,
        fields,
        stage_provenance,
        current_stage_name,
        ai_engine_cache,
        ivce_dtl,
    ):
        """
        Pass 1: Boosts confidence if a specific field exists in both stages and matches.
        Updates both the consolidated dictionary (for final output) and current_details (for threshold checks).
        """
        uid = getattr(ivce_dtl, "IVCE_DTL_UID", "UNKNOWN_UID")

        for field_key, conf_key, conf_stage_key in fields:
            prior_val = prior_details.get(field_key)
            curr_val = current_details.get(field_key)

            # Condition 1: Field must be present and non-empty in BOTH stages
            if self._is_valid_value(prior_val) and self._is_valid_value(curr_val):

                # Condition 2: Values must match
                if self.are_both_fields_matching(
                    field1=prior_val, field2=curr_val, field_key=field_key, ai_engine_cache=ai_engine_cache
                ):

                    prior_conf = prior_details.get(Logs.CONFIDENCE, {}).get(field_key, 0)
                    current_conf = consolidated.get(conf_key, 0)

                    # Condition 3: Prior confidence is higher than current (already RPA-boosted) confidence
                    if prior_conf > current_conf:
                        logger.debug(
                            f"[{current_stage_name}]({uid}) Boosting '{field_key}' confidence from {current_conf} to"
                            f" {prior_conf} based on match with {stage_provenance}."
                        )

                        # 1. Update consolidated output
                        consolidated[conf_key] = prior_conf
                        consolidated[conf_stage_key] = stage_provenance

                        # 2. Update current stage details
                        # This is critical so the 'Check' step in the Orchestrator sees the new score.
                        if Logs.CONFIDENCE not in current_details:
                            current_details[Logs.CONFIDENCE] = {}
                        current_details[Logs.CONFIDENCE][field_key] = prior_conf

    def _check_strict_agreement(self, current_details, prior_details, ai_engine_cache):
        """
        Checks if both stages refer to the same entity (Strict Agreement).
        Requires BOTH Manufacturer and Part Number to be present and matching.
        """
        required_fields = [Logs.MFR_NAME, Logs.PRT_NUM]

        for field in required_fields:
            val1 = current_details.get(field)
            val2 = prior_details.get(field)

            # Check presence
            if not (self._is_valid_value(val1) and self._is_valid_value(val2)):
                return False

            # Check match
            if not self.are_both_fields_matching(field1=val1, field2=val2, field_key=field, ai_engine_cache=ai_engine_cache):
                return False

        return True

    def _apply_data_enrichment(
        self, current_details, prior_details, consolidated, stage_provenance, current_stage_name, fields, ivce_dtl
    ):
        """
        Pass 2: Copies missing fields from prior stage to current stage.
        Persists data to current_details to ensure it survives fallback logic.
        """
        enrichment_fields = [Logs.UNSPSC, Logs.UPC, Logs.AKS_PRT_NUM]
        uid = getattr(ivce_dtl, "IVCE_DTL_UID", "UNKNOWN_UID")

        for field_key in enrichment_fields:
            curr_val = current_details.get(field_key)
            prior_val = prior_details.get(field_key)

            # Condition: Missing in Current, Present in Prior
            if not self._is_valid_value(curr_val) and self._is_valid_value(prior_val):
                logger.debug(f"[{current_stage_name}]({uid}) Enriching with '{field_key}' from {stage_provenance}: {prior_val}")

                # 1. Update consolidated output
                consolidated[field_key] = prior_val

                # 2. Persist to current stage details (Critical for fallback persistence)
                current_details[field_key] = prior_val

                # 3. Handle Confidence for Enriched Fields (if applicable)
                if Logs.CONFIDENCE in prior_details and field_key in prior_details[Logs.CONFIDENCE]:
                    prior_conf = prior_details[Logs.CONFIDENCE][field_key]

                    if Logs.CONFIDENCE not in current_details:
                        current_details[Logs.CONFIDENCE] = {}
                    current_details[Logs.CONFIDENCE][field_key] = prior_conf

                    # Update specific confidence tracking keys (e.g., Logs.CONF_UNSPSC)
                    for f_key, c_key, cs_key in fields:
                        if f_key == field_key:
                            consolidated[c_key] = prior_conf
                            consolidated[cs_key] = stage_provenance

                # 4. Update Field Provenance
                stage_key_name = Logs.MAPPING.get(field_key, [None, f"{field_key}_stage"])[1]
                consolidated[stage_key_name] = stage_provenance

                # This ensures the provenance survives if this stage is picked via fallback.
                current_details[stage_key_name] = stage_provenance

    def _apply_rpa_boosting(self, current_details, consolidated, fields, ivce_dtl):
        """
        Pass 3: Applies business logic boost if AI extraction matches RPA data.
        """
        for field_key, conf_key, _ in fields:
            if conf_key in consolidated:
                if Logs.CONFIDENCE not in current_details:
                    current_details[Logs.CONFIDENCE] = {}

                ai_value = current_details.get(field_key)
                current_conf = consolidated[conf_key]

                if check_if_rpa_can_boost_confidence(ivce_dtl=ivce_dtl, field=field_key, ai_value=ai_value):
                    boosted_conf = current_conf + self.confidences[StageNames.RPA_PROCESS]["CONFIDENCE_BOOST"]
                    final_conf = min(boosted_conf, 100)
                    consolidated["has_rpa_boosted_conf"] = True
                else:
                    final_conf = current_conf

                consolidated[conf_key] = final_conf
                current_details[Logs.CONFIDENCE][field_key] = final_conf

    # def consolidate_field_confidence(
    #     self, *, stage_results, stage_details, field_key, ivce_dtl, ai_engine_cache, consider_rpa=False
    # ):
    #     """
    #     Generic method to consolidate a field's confidence value from previous stages.

    #     Args:
    #         - stage_details: StageDetails object of the last stage run
    #         - field_key: Key to access the field value (e.g., Manufacturer Name, UNSPSC)

    #     Returns:
    #         - Tuple of (final confidence score, stage name where it was found)
    #     """
    #     try:
    #         has_rpa_boosted_conf = False
    #         current_value = stage_details.details[field_key]
    #         current_conf = stage_details.details[Logs.CONFIDENCE][field_key]
    #         final_conf = copy(current_conf)
    #         final_conf_stage = f"{stage_details.stage_name} - {stage_details.sub_stage_name}"

    #         if not self.is_field_available(current_value):
    #             return final_conf, final_conf_stage, has_rpa_boosted_conf

    #         for number, st_details in stage_results.results.items():
    #             if 1 < number < stage_details.stage_number:
    #                 details = st_details.get("details", {})

    #                 if (
    #                     field_key in details
    #                     and self.are_both_fields_matching(
    #                         field1=details[field_key], field2=current_value, field_key=field_key,
    #                         ai_engine_cache=ai_engine_cache
    #                     )
    #                     and details[Logs.CONFIDENCE][field_key] > final_conf
    #                 ):

    #                     final_conf = details[Logs.CONFIDENCE][field_key]
    #                     final_conf_stage = f"{st_details['stage_name']} - {st_details['sub_stage_name']}"

    #         # Check if RPA values matches with stage prediction to boost its confidence
    #         if consider_rpa and check_if_rpa_can_boost_confidence(ivce_dtl=ivce_dtl, field=field_key, ai_value=current_value):
    #             final_conf += self.confidences[StageNames.RPA_PROCESS]["CONFIDENCE_BOOST"]
    #             # Cap the maximum confidence to 100
    #             if final_conf > 100:
    #                 final_conf = 100
    #             has_rpa_boosted_conf = True

    #     except Exception as e:
    #         logger.error(f"Error occurred in consolidate_field_confidence() {str(e)}", exc_info=True)
    #     return final_conf, final_conf_stage, has_rpa_boosted_conf

    def check_if_field_exists(self, field: str, details: Mapping[str, Any]) -> bool:
        """
        Return True iff the field key is present in the stage `details`.

        Args:
            field: Field name to check.
            details: Stage details mapping.

        Returns:
            True if `field` is a key in `details`; otherwise False.
        """
        return field in details

    def check_if_field_is_confident(self, field: str, confidences: Mapping[str, float], thresholds: Mapping[str, float]) -> bool:
        """
        Compare a field's confidence against the current stage thresholds.

        Args:
            field: Field name.
            confidences: Mapping of field -> confidence (current stage).
            thresholds: Mapping of field -> required minimum confidence (current stage).

        Returns:
            True if confidence >= threshold; otherwise False.
            Missing confidence or threshold is treated as not confident.
        """
        conf = confidences.get(field)
        thr = thresholds.get(field)
        if conf is None or thr is None:
            return False
        return conf >= thr

    def check_if_null_field_is_confident(
        self,
        field_key: str,
        stage_details_consolidated: Mapping[str, Any],
        thresholds: Mapping[str, Mapping[str, Mapping[str, float]]],
    ) -> bool:
        """
        Validate a NULL-consolidated field against the thresholds of the stage it came from.

        Looks up the field's consolidated confidence and its source stage (encoded in the
        consolidated payload), then uses that stage's thresholds for comparison.

        Args:
            field_key: Field name.
            stage_details_consolidated: Consolidated payload containing:
                - confidence under key `Logs.get_conf_key(field_key)`
                - source stage tag under key `Logs.get_stage_key(field_key)`
            thresholds: Global thresholds mapping:
                thresholds[stage_name][sub_stage_name][field_name] -> required min confidence

        Returns:
            True if the consolidated confidence meets the source stage's threshold; else False.
            Missing data (confidence, stage, or threshold) is treated as not confident.
        """
        conf_key = Logs.get_conf_key(field_key=field_key)
        stage_key = Logs.get_stage_key(field_key=field_key)

        confidence = stage_details_consolidated.get(conf_key)
        field_stage = stage_details_consolidated.get(stage_key)
        if confidence is None or field_stage is None:
            return False

        stage_name, sub_stage_name = get_stage_and_sub_stage_names_from_log_stage(field_stage)
        stage_thresholds = thresholds.get(stage_name, {}).get(sub_stage_name, {})
        thr = stage_thresholds.get(field_key)
        if thr is None:
            return False

        return confidence >= thr

    def check_if_next_stage_required(self, ivce_dtl, stage_details):
        try:
            if stage_details is None:
                return True

            stage_thresholds = self.thresholds[stage_details.stage_name][stage_details.sub_stage_name]
            if Logs.CONFIDENCE in stage_details.details:
                confidences = stage_details.details[Logs.CONFIDENCE]
                flags = [
                    self.check_if_field_exists(field, stage_details.details)
                    and self.check_if_field_is_confident(field, confidences, stage_thresholds)
                    for field in ivce_dtl.fields
                ]
                return not all(flags)  # Returns True if any one of flags is False
        except Exception as e:
            logger.error(f"Error occurred in check_if_next_stage_required() {str(e)}", exc_info=True)
        return True

    def check_if_next_stage_required_after_null_field_consolidation(
        self,
        null_fields: Sequence[str],
        ivce_dtl: Any,
        stage_details: Any,
        stage_details_consolidated: Optional[Mapping[str, Any]],
    ) -> bool:
        """
        Decide whether another extraction stage is required after consolidating NULL fields.

        Rules:
        - **All** fields in `ivce_dtl.fields` are evaluated.
        - If a field does **not exist** in `stage_details.details`, it is treated as **failed**.
        - For fields consolidated due to NULL, compare their consolidated confidence
            against the thresholds of the stage they came from.
        - For all other fields, compare against the thresholds of the current stage.
        - If **any** field fails its check, return True (next stage required).
        - **Fail-fast:** if at least one field needs current-stage confidences and
            those confidences are absent, immediately require next stage.

        Returns:
            True if a subsequent stage should run; False if all fields meet thresholds.
            Missing data (details, confidences, thresholds, consolidation info) is treated
            conservatively and triggers the next stage.
        """
        try:
            # Consolidation info missing => conservative
            if not stage_details_consolidated:
                return True

            current_thresholds: Mapping[str, float] = self.thresholds.get(stage_details.stage_name, {}).get(
                stage_details.sub_stage_name, {}
            )
            details: Mapping[str, Any] = getattr(stage_details, "details", {}) or {}
            confidences: Mapping[str, float] = details.get(Logs.CONFIDENCE, {}) or {}

            fields = list(getattr(ivce_dtl, "fields", []) or [])
            if not fields:
                return True  # no evaluable fields => conservative

            # ---- FAIL-FAST GUARD ----
            # If any field is NOT a null-consolidated field and we don't have current-stage
            # confidences, then we cannot validate it -> require next stage immediately.
            if any(f not in null_fields for f in fields) and (Logs.CONFIDENCE not in details or not confidences):
                return True
            # -------------------------

            def field_passes(field: str) -> bool:
                # If the field key itself isn't present in details, it's a failure.
                if not self.check_if_field_exists(field, details):
                    return False

                if field in null_fields:
                    return self.check_if_null_field_is_confident(
                        field_key=field, stage_details_consolidated=stage_details_consolidated, thresholds=self.thresholds
                    )
                return self.check_if_field_is_confident(field=field, confidences=confidences, thresholds=current_thresholds)

            all_pass = all(field_passes(f) for f in fields)
            return not all_pass

        except Exception as e:
            logger.error("Error in check_if_next_stage_required_after_null_field_consolidation(): %s", e, exc_info=True)
            return True

    def is_any_field_not_available(self, details, fields):
        """
        Checking if any of the fields required to consolidate are None or Null.
        If any is None/Null, we cannot stop here; we must proceed to the next stage.

        Args:
            details: The dictionary of results from the current stage.
            fields: A list of tuples, where the first element is the field key.
                    e.g. [(Logs.PRT_NUM, ...), (Logs.MFR_NAME, ...)]

        Returns:
            True: if any one required field is Missing, None, or Empty.
            False: if all required fields are present and valid.
        """
        for field_tuple in fields:
            # Handle tuple vs string
            field_key = field_tuple[0] if isinstance(field_tuple, (tuple, list)) else field_tuple

            # Get value (returns None if key missing)
            value = details.get(field_key)

            # Single check: Is it None, Empty, or Invalid?
            if self.is_field_missing(value):
                return True

        return False

    def is_field_missing(self, value):  # Was: is_field_available
        """Returns True if the value is None, empty, or invalid."""
        if value and is_not_null(value) and is_not_empty(value):
            return False  # It is NOT missing (it is valid)
        else:
            return True  # It IS missing
