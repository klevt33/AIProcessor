from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ai_stages import AIStages, StageDetails, StageUtils
from category_utils import CateogryUtils
from constants import (
    Constants,
    DataStates,
    DescriptionCategories,
    Fields,
    LocalFiles,
    Logs,
    SpecialCases,
    StageNames,
    SubStageNames,
)
from exceptions import InvoiceProcessingError
from lemmatizer import get_lemmatizer
from logger import logger
from matching_utils import read_manufacturer_data
from utils import (
    apply_rental_flag,
    clean_text_for_llm,
    get_current_datetime_cst,
    get_rental_flag,
    is_mfr_available_from_rpa,
    load_yaml,
)


@dataclass
class StageConfig:
    stage_number: int
    sub_stage_code: str
    stage_name: str
    fields_to_consolidate: list[tuple[str]]
    stage_fn: Callable[..., Any]
    validation_stage_name: Optional[str] = None  # Name of the stage that validates this one
    is_validation_only: bool = False  # Marks stages that shouldn't run in the main loop


@dataclass
class AIEngineCache:
    description_embedding: Optional[Any] = None
    cleaned_description: Optional[str] = None
    cleaned_description_for_llm: Optional[str] = None
    latest_stage_details: Optional[StageDetails] = None
    previous_stage_details: Optional[StageDetails] = None
    semantic_search_results: Optional[list] = None
    manufacturer_aliases: Optional[Dict[str, str]] = None


class AIEngine:

    def __init__(self, config, sdp):
        self.config = config
        self.sdp = sdp
        self.category_utils = CateogryUtils(sdp)
        self.ai_stages = AIStages(config=config, category_utils=self.category_utils)
        self.ai_engine_cache = AIEngineCache()
        self.stage_utils = StageUtils()
        self.special_cases = load_yaml(path=LocalFiles.SPECIAL_CASES_FILE)
        self.thresholds = load_yaml(LocalFiles.THRESHOLDS_FILE)
        self.categories = [
            DescriptionCategories.MATERIAL,
            DescriptionCategories.BAD,
            DescriptionCategories.TAX,
            DescriptionCategories.FEE,
            DescriptionCategories.FREIGHT,
            DescriptionCategories.LABOR,
            DescriptionCategories.DISCOUNTS,
            DescriptionCategories.AP_ADJUSTMENT,
            DescriptionCategories.LOT,
            DescriptionCategories.NOT_LOT,
            DescriptionCategories.GENERIC,
            DescriptionCategories.RENTAL,
            DescriptionCategories.NON_RENTAL,
        ]

        # Allowed status from AI process
        self.line_status = [DataStates.RC_AI, DataStates.DS1]

        self.prepare_extraction_engine()

    async def async_init(self):
        await self.category_utils.async_init()
        self.ai_engine_cache.manufacturer_data_dict = await read_manufacturer_data(sdp=self.sdp)

    def prepare_extraction_engine(self):
        self.consolidation_fields = {
            Fields.PART_NUMBER: (Logs.PRT_NUM, Logs.CONF_PRT_NUM, Logs.CONF_STAGE_PRT_NUM),
            Fields.MANUFACTURER_NAME: (Logs.MFR_NAME, Logs.CONF_MFR_NAME, Logs.CONF_STAGE_MFR_NAME),
            Fields.UNSPSC_CODE: (Logs.UNSPSC, Logs.CONF_UNSPSC, Logs.CONF_STAGE_UNSPSC),
        }
        self.extraction_engine = [
            StageConfig(
                stage_number=4,
                sub_stage_code="2.0",
                stage_name=StageNames.SEMANTIC_SEARCH,
                stage_fn=self.ai_stages.check_semantic_match,
                fields_to_consolidate=[
                    self.consolidation_fields[Fields.MANUFACTURER_NAME],
                    self.consolidation_fields[Fields.UNSPSC_CODE],
                ],
            ),
            StageConfig(
                stage_number=5,
                sub_stage_code="3.0",
                stage_name=StageNames.COMPLETE_MATCH,
                stage_fn=self.ai_stages.check_complete_match,
                fields_to_consolidate=[
                    self.consolidation_fields[Fields.PART_NUMBER],
                    self.consolidation_fields[Fields.MANUFACTURER_NAME],
                    self.consolidation_fields[Fields.UNSPSC_CODE],
                ],
                validation_stage_name=StageNames.CONTEXT_VALIDATOR,
            ),
            StageConfig(
                stage_number=8,
                sub_stage_code="6.0",
                stage_name=StageNames.CONTEXT_VALIDATOR,
                stage_fn=self.ai_stages.validate_context,
                fields_to_consolidate=None,  # The validator judges, it doesn't produce consolidatable fields.
                validation_stage_name=None,  # A validator does not have its own validator.
                is_validation_only=True,  # Declare this as a validation-only stage
            ),
            StageConfig(
                stage_number=6,
                sub_stage_code="4.0",
                stage_name=StageNames.FINETUNED_LLM,
                stage_fn=self.ai_stages.extract_from_finetuned_llm,
                fields_to_consolidate=[
                    self.consolidation_fields[Fields.PART_NUMBER],
                    self.consolidation_fields[Fields.MANUFACTURER_NAME],
                    self.consolidation_fields[Fields.UNSPSC_CODE],
                ],
            ),
            StageConfig(
                stage_number=7,
                sub_stage_code="5.0",
                stage_name=StageNames.EXTRACTION_WITH_LLM_AND_WEBSEARCH,
                stage_fn=self.ai_stages.extract_with_ai_agents_from_websearch,
                fields_to_consolidate=[
                    self.consolidation_fields[Fields.PART_NUMBER],
                    self.consolidation_fields[Fields.MANUFACTURER_NAME],
                    self.consolidation_fields[Fields.UNSPSC_CODE],
                ],
            ),
        ]

    def _get_stage_config_by_name(self, stage_name: str) -> Optional[StageConfig]:
        """Finds a StageConfig from the extraction_engine list by its name."""
        for config in self.extraction_engine:
            if config.stage_name == stage_name:
                return config
        return None

    def check_if_stage_allowed(self, ivce_dtl, stage_name):
        """
        Determine whether the specified stage is still active in the invoice's stage mapping.

        A stage is considered “required” if it exists as a key in
        `ivce_dtl.stage_mapping` and its list of sub-stages is non-empty.

        Args:
            ivce_dtl (YourInvoiceDetailClass):
                An object with a `.stage_mapping` attribute, which should be
                a dict mapping StageNames to lists of SubStageNames.
            stage_name (StageNames):
                The stage to check for presence in `ivce_dtl.stage_mapping`.

        Returns:
            bool:
                True if `stage_name` is in `ivce_dtl.stage_mapping` and its
                associated list has at least one sub-stage; False otherwise.
        """
        mapping = getattr(ivce_dtl, "stage_mapping", {})
        return stage_name in mapping and bool(mapping[stage_name])

    def check_if_valid_category(self, stage_details, category):
        if category not in self.categories:
            logger.error("Classified category not found in the list.")
            self.stage_results.final_stage_key = stage_details.stage_number
            self.stage_results.status = Constants.ERROR_lower
            self.stage_results.message = "Classified category not found in the list."
            raise InvoiceProcessingError(message="Classified category not found in the list.")

    def set_ignore_stage_flag(self, stage_details, flag: bool = True):
        """
        Sets a flag to stage results of a stage using stage_number. If the flag is True, stage is ignored during
        finalizing pipeline results. If the flag is False, stage is considered during finalizing pipeline results.
        Args:
            stage_details: Object
            flag: To identify consideration
        """
        self.stage_results.results[stage_details.stage_number]["ignore_stage"] = flag

    async def process_description(self, ivce_dtl):
        """
        Process single description and writes the output into SDP

        Args:
            - request: Request object from API call
            - ivce_dtl: Invoice detail row object

        Returns:
            stage_results: StageResults object containing data from all stages (for Cosmos DB)
        """

        # Re-initialize each time for new invoice processing
        lemmatizer = get_lemmatizer()
        self.stage_results = StageResults()
        self.stage_results.final_results = {}
        self.stage_results.first_stage_key = 0
        ivce_dtl.is_rental_from_rpa = get_rental_flag(ivce_dtl.RNTL_IND)
        self.stage_results.is_rental = apply_rental_flag(rental_flag=Constants.N, rpa_rental_flag=ivce_dtl.is_rental_from_rpa)
        await self.set_invoice_line_status(status=DataStates.DS1)

        try:
            if self.check_if_stage_allowed(ivce_dtl=ivce_dtl, stage_name=StageNames.CLASSIFICATION):
                # classify description
                classifier_stage_details = await self.ai_stages.fetch_classification(lemmatizer, ivce_dtl)
                self.stage_results.add_stage_result(stage_details=classifier_stage_details)

                if classifier_stage_details.status == Constants.ERROR_lower:
                    self.stage_results.final_stage_key = classifier_stage_details.stage_number
                    return self.stage_results

                else:
                    category = classifier_stage_details.details[Logs.CATEGORY]
                    confidence_score = classifier_stage_details.details[Logs.CONFIDENCE]
                    cleaned_description = classifier_stage_details.details[Logs.DESCRIPTION]

                    self.set_ignore_stage_flag(stage_details=classifier_stage_details, flag=False)
                    self.check_if_valid_category(stage_details=classifier_stage_details, category=category)

                    if category == DescriptionCategories.MATERIAL:
                        # LOTs are Non-rental always, unless RPA says so
                        lot_flag = await self.check_if_lot(ivce_dtl, lemmatizer)
                        if not lot_flag:
                            # Check if it is RENTAL, only if is not a LOT
                            await self.determine_rental_indicator(cleaned_description=cleaned_description)

                            # Run engine only if it is not LOT or None.
                            # If it is LOT, engine would have already been run before coming here.
                            await self.run_extraction_engine(ivce_dtl, lemmatizer)

                    else:
                        self.stage_results.final_stage_key = classifier_stage_details.stage_number
                        stage_threshold = self.thresholds[StageNames.CLASSIFICATION][SubStageNames.DESCRIPTION_CLASSIFIER]

                        if category in [
                            DescriptionCategories.FEE,
                            DescriptionCategories.FREIGHT,
                            DescriptionCategories.TAX,
                            DescriptionCategories.DISCOUNTS,
                        ]:
                            # Only these classes can be RENTALs
                            await self.determine_rental_indicator(cleaned_description=cleaned_description)

                        if confidence_score < stage_threshold:
                            await self.set_invoice_line_status(status=DataStates.DS1)

                        elif category == DescriptionCategories.BAD:
                            lot_flag = await self.check_if_lot(ivce_dtl, lemmatizer)

                            if not lot_flag:
                                # Check only if not LOT
                                await self.determine_rental_indicator(cleaned_description=cleaned_description)

                                await self.set_invoice_line_status(status=DataStates.DS1)

                        else:
                            await self.set_invoice_line_status(status=DataStates.RC_AI)

                        return self.stage_results

            else:
                logger.warning("Running extraction engine directly without classification...")

                if ivce_dtl.stage_count == 0:
                    logger.info("process_description(): No stages are available in mapping to run extraction engine.")
                    return self.stage_results

                await self.run_extraction_engine(ivce_dtl, lemmatizer)

        except Exception as e:
            logger.error(f"Error in process_description(): {str(e)}", exc_info=True)

        return self.stage_results

    async def check_if_lot(self, ivce_dtl, lemmatizer):
        """
        This method runs LOT classifier if the description has 'LOT' characters in it.
        If it is classified as LOT, check CLN_MFR_AI_NM written by RPA from PO mapping.
        If it is Null or UNDEFINED, then runs the AI engine directly.

        NOTE: Every LOT will contain `LOT` characters in the description. So, sending only MATERIAL/BAD descriptions
        which have 'LOT' characters in it to LOT classifier.
        """
        classifier_stage_details_dict = self.stage_results.get_stage_result(stage_number=1)
        cleaned_description = classifier_stage_details_dict[Logs.DETAILS][Logs.DESCRIPTION]
        parent_category = classifier_stage_details_dict[Logs.DETAILS]["category"]

        if Constants.LOT_lower in cleaned_description:

            if parent_category == DescriptionCategories.BAD:
                parent_id = self.category_utils.convert_bad_parent_to_compatible_parent(
                    additional_parent_id=self.category_utils.get_category_id(category_name=DescriptionCategories.MATERIAL)
                )
            else:
                parent_id = classifier_stage_details_dict[Logs.DETAILS]["category_id"]

            lot_classifier_stage_details = await self.ai_stages.fetch_lot_classification(
                cleaned_description=cleaned_description, parent_category_id=parent_id
            )
            self.stage_results.add_stage_result(stage_details=lot_classifier_stage_details)

            if lot_classifier_stage_details.status == Constants.ERROR_lower:
                self.stage_results.final_stage_key = lot_classifier_stage_details.stage_number
                return None

            else:
                category = lot_classifier_stage_details.details[Logs.CATEGORY]
                lot_classifier_stage_details.details[Logs.CONFIDENCE]

                self.check_if_valid_category(stage_details=lot_classifier_stage_details, category=category)

                if category == DescriptionCategories.LOT:
                    self.set_ignore_stage_flag(stage_details=lot_classifier_stage_details, flag=False)

                    # convert this to special case, to handle future steps (fields and stages)
                    await self.special_case_fn(case_name=SpecialCases.CASE_2, ivce_dtl=ivce_dtl)

                    if (
                        is_mfr_available_from_rpa(mfr_nm=ivce_dtl.CLN_MFR_AI_NM)
                        and ivce_dtl.CLN_MFR_AI_NM != DescriptionCategories.LOT
                    ):
                        message = f"RPA has already found {ivce_dtl.CLN_MFR_AI_NM} as MFR name from PO."
                        self.stage_results.results[lot_classifier_stage_details.stage_number][Logs.DETAILS][
                            Logs.MESSAGE
                        ] = message
                        self.stage_results.final_stage_key = lot_classifier_stage_details.stage_number
                        await self.set_invoice_line_status(status=DataStates.RC_AI)
                        self.stage_results.final_results.update(
                            {
                                Logs.MFR_NAME: ivce_dtl.CLN_MFR_AI_NM,
                                Logs.CONF_MFR_NAME: 100,
                                Logs.STAGE_MFR_NAME: "RPA",
                                Logs.CONF_STAGE_MFR_NAME: "RPA",
                            }
                        )
                        return True

                    #     Export to trainable table – REC_ACTV_IND = ‘N’ - Invoice pipeline
                    await self.run_extraction_engine(ivce_dtl, lemmatizer)
                    return True
                else:
                    self.set_ignore_stage_flag(stage_details=lot_classifier_stage_details)
                    return False
        logger.debug("There are no 'LOT' characters in description. Skipping LOT classifier...")
        return None

    async def determine_rental_indicator(self, cleaned_description):
        # If it is rental from RPA, No need to run classifier again.
        if self.stage_results.is_rental == Constants.Y:
            logger.debug("RPA marked as Rental so skipping rental classifier.")
            stage_details = StageDetails(
                stage_number=3,
                sub_stage_code="1.2",
                stage=StageNames.CLASSIFICATION,
                sub_stage=SubStageNames.RENTAL_CLASSIFIER,
                status=Constants.SUCCESS_lower,
            )
            stage_details.details = {
                Logs.END_TIME: get_current_datetime_cst(),
                Constants.MESSAGE: "RPA marked as Rental so skipping rental classifier.",
            }
            self.stage_results.add_stage_result(stage_details=stage_details)
            return

        if cleaned_description == Constants.EMPTY_STRING:
            logger.debug("Empty cleaned string for description so setting rental to N.")
            stage_details = StageDetails(
                stage_number=3,
                sub_stage_code="1.2",
                stage=StageNames.CLASSIFICATION,
                sub_stage=SubStageNames.RENTAL_CLASSIFIER,
                status=Constants.SUCCESS_lower,
            )
            stage_details.details = {
                Logs.END_TIME: get_current_datetime_cst(),
                Constants.MESSAGE: "Empty cleaned string for description so setting rental to N",
            }
            self.stage_results.add_stage_result(stage_details=stage_details)
            self.stage_results.is_rental = Constants.N
            return

        rental_classifier_stage_details = await self.ai_stages.fetch_rental_classification(
            cleaned_description=cleaned_description
        )
        self.stage_results.add_stage_result(stage_details=rental_classifier_stage_details)

        if rental_classifier_stage_details.status == Constants.ERROR_lower:
            self.stage_results.final_stage_key = rental_classifier_stage_details.stage_number
            return None

        else:
            category = rental_classifier_stage_details.details[Logs.CATEGORY]
            rental_classifier_stage_details.details[Logs.CONFIDENCE]

            self.check_if_valid_category(stage_details=rental_classifier_stage_details, category=category)
            # self.set_ignore_stage_flag(stage_details=rental_classifier_stage_details, flag=False)
            self.set_ignore_stage_flag(stage_details=rental_classifier_stage_details)

            if category == DescriptionCategories.RENTAL:
                self.stage_results.is_rental = Constants.Y
            else:
                self.stage_results.is_rental = Constants.N

    async def set_invoice_line_status(self, status):
        if status not in self.line_status:
            error_msg = f"Invalid invoice line status '{status}'"
            logger.error(error_msg, exc_info=True)
            raise InvoiceProcessingError(message=error_msg)

        self.stage_results.final_results.update({Logs.IVCE_LINE_STATUS: status})

    async def _choose_final_stage(self):
        """
        Determine which extraction stage to treat as final and persist its results.

        This helper inspects the `latest_stage_details` in `self.ai_engine_cache`:
        - If a confidence score (`Logs.CONFIDENCE`) is present, it means
            the web-search (final) stage produced results, so we choose that as final.
        - Otherwise, we fall back to the previous stage (usually the fine-tuned LLM).

        Side Effects
        ------------
        - Updates `self.stage_results.final_stage_key` to the chosen stage number.
        - Calls `await self.prepare_data_to_write_into_sdp()` to persist interim results.
        - Sets the invoice-line status to `DataStates.DS1`.

        Returns
        -------
        StageDetails
            A deep copy of the chosen `StageDetails` object for downstream use.
        """
        # Web-search produced confidence → use latest stage
        if Logs.CONFIDENCE in self.ai_engine_cache.latest_stage_details.details:
            self.stage_results.final_stage_key = self.ai_engine_cache.latest_stage_details.stage_number
            # await self.prepare_data_to_write_into_sdp()
            final_stage_details = deepcopy(self.ai_engine_cache.latest_stage_details)
        else:
            # No web-search results → use previous (LLM) stage
            self.stage_results.final_stage_key = self.ai_engine_cache.previous_stage_details.stage_number
            # await self.prepare_data_to_write_into_sdp()
            final_stage_details = deepcopy(self.ai_engine_cache.previous_stage_details)

        # Always mark as DS1 when selecting a final extraction output
        await self.set_invoice_line_status(status=DataStates.DS1)

        # Return a snapshot of the chosen stage details
        return final_stage_details

    async def _consolidate_unspsc_if_needed(self, final_stage_details, ivce_dtl):
        """
        Ensure the final UNSPSC code is populated by consolidating non-null values from prior stages.

        If the chosen final stage's details have no UNSPSC value (missing or empty),
        this method invokes `self.stage_utils.consolidate_null_field(...)` to pull the
        best available UNSPSC from earlier stage results.  It then updates
        `self.stage_results.final_results` and sets the invoice-line status to DS1 or RC_AI
        based on whether further processing would have been required.

        Parameters
        ----------
        final_stage_details : StageDetails
            The details object from the chosen final extraction stage.
        ivce_dtl : InvoiceDetail
            The invoice-line detail instance—used by `check_if_next_stage_required` to
            determine the appropriate status after consolidation.

        Raises
        ------
        Exception
            Any error during null-field consolidation is logged via `logger.error` and re-raised.
        """
        try:
            field_key = Logs.UNSPSC
            # Only act if UNSPSC is missing or blank
            if (
                field_key not in final_stage_details.details
                or final_stage_details.details[field_key].strip() == Constants.EMPTY_STRING
            ):

                # Pull the best non-null UNSPSC from earlier stages
                final_stage_consolidated, final_stage_details_consolidated = self.stage_utils.consolidate_null_field(
                    stage_results=self.stage_results,
                    stage_details=final_stage_details,
                    field_key=field_key,
                    conf_key=Logs.CONF_UNSPSC,
                    stage_key=Logs.STAGE_UNSPSC,
                    conf_stage_key=Logs.CONF_STAGE_UNSPSC,
                )

                if final_stage_consolidated[field_key] == Constants.UNDEFINED:
                    return

                # Update the final-results map
                self.stage_results.final_results.update(final_stage_consolidated)

                # Set status based on whether additional processing would be triggered
                if self.stage_utils.check_if_next_stage_required_after_null_field_consolidation(
                    null_fields=[field_key],
                    stage_details=final_stage_details_consolidated,
                    stage_details_consolidated=final_stage_consolidated,
                    ivce_dtl=ivce_dtl,
                ):
                    await self.set_invoice_line_status(status=DataStates.DS1)
                else:
                    await self.set_invoice_line_status(status=DataStates.RC_AI)

        except Exception as e:
            logger.error(f"_consolidate_unspsc_if_needed() - UNSPSC Null value consolidation: {str(e)}", exc_info=True)
            raise e

    # async def run_stage(self, stage_config, ivce_dtl):
    #     """
    #     Execute a single extraction stage and determine if processing should continue.

    #     This method wraps the invocation of one stage in the pipeline. It performs the following steps:
    #     1. Checks if the stage is allowed via `self.check_if_stage_allowed`.
    #     2. Calls the stage function (`stage_config.stage_fn`), which returns a new
    #         `StageDetails` and updates `self.ai_engine_cache`.
    #     3. Records the returned `StageDetails` in `self.stage_results` and in
    #         `self.ai_engine_cache.latest_stage_details`.
    #     4. Uses `self.stage_utils.check_if_next_stage_required` to decide whether to
    #         continue.
    #         - If continuation is allowed, consolidates confidence values across fields
    #         via `self.stage_utils.consolidate_all_fields_confidences`; stores the
    #         consolidated map in `self.ai_engine_cache.latest_stage_consolidated`.
    #         - If consolidation then signals “stop,” writes interim results to the SDP,
    #         updates status to `RC_AI`, and merges consolidated fields into
    #         `self.stage_results.final_results`.
    #         - If the initial check signals “stop,” finalizes immediately without
    #         consolidation.
    #     5. Returns a boolean `next_stage_required`:
    #         - `True` if the pipeline should proceed to the next stage.
    #         - `False` if extraction should end here (after any required finalization).

    #     Parameters
    #     ----------
    #     stage_config : StageConfig
    #         Configuration for this stage, including:
    #         - `stage_name` (StageNames): the enum name of the stage.
    #         - `stage_fn` (Callable): async function to execute.
    #         - `fields_to_consolidate` (List[Tuple[…] or None): which fields to merge confidences for.
    #     ivce_dtl : InvoiceDetail
    #         The invoice-line detail instance being processed; passed through to
    #         `check_if_stage_allowed` and `check_if_next_stage_required`.

    #     Returns
    #     -------
    #     bool
    #         `True` if the next stage in the pipeline should run; `False` if processing
    #         terminates here (and results have been finalized).

    #     Raises
    #     ------
    #     Exception
    #         Any exception from the stage function, consolidation, or persistence is logged
    #         with `logger.error` and re-raised.
    #     """
    #     next_stage_required = False
    #     try:
    #         # Call stage
    #         stage_details = None
    #         stage_consolidated = None

    #         # Only run if allowed
    #         if self.check_if_stage_allowed(ivce_dtl=ivce_dtl, stage_name=stage_config.stage_name):
    #             # Execute the stage
    #             stage_details, self.ai_engine_cache = await stage_config.stage_fn(self.sdp, self.ai_engine_cache, ivce_dtl)
    #             self.ai_engine_cache.latest_stage_details = copy(stage_details)
    #             self.stage_results.add_stage_result(stage_details=stage_details)

    #             # Decide whether to continue
    #             if self.stage_utils.check_if_next_stage_required(stage_details=stage_details, ivce_dtl=ivce_dtl):
    #                 # Check if all fields are there
    #                 if not self.stage_utils.is_any_field_not_available(
    #                     details=stage_details.details, fields=stage_config.fields_to_consolidate
    #                 ):
    #                     # Consolidate confidences if configured
    #                     stage_details, stage_consolidated = self.stage_utils.consolidate_all_fields_confidences(
    #                         stage_results=self.stage_results,
    #                         stage_details=stage_details,
    #                         fields=stage_config.fields_to_consolidate,
    #                         ivce_dtl=ivce_dtl,
    #                         ai_engine_cache=self.ai_engine_cache,
    #                     )
    #                     self.ai_engine_cache.latest_stage_consolidated = copy(stage_consolidated)

    #                     # If consolidation says stop, finalize here
    #                     if not self.stage_utils.check_if_next_stage_required(stage_details=stage_details, ivce_dtl=ivce_dtl):
    #                         # Prepare final data and update with consolidated results
    #                         self.stage_results.final_stage_key = stage_details.stage_number
    #                         # await self.prepare_data_to_write_into_sdp()
    #                         await self.set_invoice_line_status(status=DataStates.RC_AI)
    #                         self.stage_results.final_results.update(stage_consolidated)
    #                         return next_stage_required

    #             else:
    #                 # Immediate stop without consolidation
    #                 self.stage_results.final_stage_key = stage_details.stage_number
    #                 # await self.prepare_data_to_write_into_sdp()
    #                 await self.set_invoice_line_status(status=DataStates.RC_AI)
    #                 return next_stage_required

    #         # If we get here, stage completed and next is allowed
    #         next_stage_required = True

    #     except Exception as e:
    #         logger.error(f"run_extraction_engine() - {stage_config.stage_name}: {str(e)}", exc_info=True)
    #         raise e

    #     return next_stage_required

    # async def run_extraction_engine(self, ivce_dtl, lemmatizer):
    #     """
    #     Execute the end-to-end invoice-line extraction pipeline.

    #     This method steps through each configured stage in `self.extraction_engine`,
    #     carrying forward intermediate results in `self.ai_engine_cache`.  Stages may
    #     include semantic search, exact match, fine-tuned LLM extraction, and web-search
    #     fallback.  If any stage indicates “stop,” the loop exits early; otherwise it runs
    #     all stages in sequence.

    #     After all stages:
    #     1. Selects the final stage details via `self._choose_final_stage()`, which
    #         compares the last two stage outputs (e.g. to see whether web‐search produced
    #         a confidence score).
    #     2. Persists the final results and invoice-line status to the SDP.
    #     3. Ensures the invoice-line is marked DS1.
    #     4. If the UNSPSC code is still blank, calls
    #         `self._consolidate_unspsc_if_needed(final_stage_details, ivce_dtl)` to
    #         pull the best non-null UNSPSC from any prior stage, updating status as needed.

    #     All intermediate data (description embedding, cleaned text, previous/latest
    #     StageDetails) live in `self.ai_engine_cache`.

    #     Parameters
    #     ----------
    #     ivce_dtl : InvoiceDetail
    #         The invoice-line detail instance to process.
    #     lemmatizer : Callable[[str], str]
    #         A function that normalizes and lemmatizes the raw item description.

    #     Raises
    #     ------
    #     Exception
    #         Any unexpected error from `self.run_stage`, `_choose_final_stage`, or
    #         `_consolidate_unspsc_if_needed` is logged and re-raised.
    #     """
    #     # If number of stages run so far matches with the number of sub stages in mapping, then nothing proceed further here.
    #     if len(self.stage_results.results) == ivce_dtl.sub_stage_count:
    #         logger.debug("run_extraction_engine(): No stages are available in mapping to run in extraction engine.")
    #         return

    #     self.ai_engine_cache.description_embedding = None
    #     self.ai_engine_cache.cleaned_description_for_llm = clean_text_for_llm(lemmatizer, ivce_dtl.ITM_LDSC)

    #     next_stage_required = True
    #     self.ai_engine_cache.previous_stage_details = None
    #     self.ai_engine_cache.latest_stage_details = None

    #     # Run each stage in turn until one signals “stop”
    #     for sc in self.extraction_engine:
    #         if not next_stage_required:
    #             break
    #         self.ai_engine_cache.previous_stage_details = deepcopy(self.ai_engine_cache.latest_stage_details)
    #         next_stage_required = await self.run_stage(stage_config=sc, ivce_dtl=ivce_dtl)

    #     # If we completed all stages, finalize and handle UNSPSC (only if required)
    #     if next_stage_required:
    #         final_stage_details = await self._choose_final_stage()
    #         if Logs.UNSPSC in ivce_dtl.fields:
    #             await self._consolidate_unspsc_if_needed(final_stage_details, ivce_dtl)

    async def _consolidate_and_recheck_thresholds(self, stage_details, stage_config, ivce_dtl):
        """
        Performs confidence score boosting using only non-invalidated prior stages,
        then re-evaluates thresholds and returns the final decision and data.

        Returns a tuple: (bool, Optional[dict])
        - (True, consolidated_data) if the pipeline should stop.
        - (False, None) if the pipeline should continue.
        """
        sanitized_stage_results = StageResults()
        sanitized_stage_results.results = {
            num: res for num, res in self.stage_results.results.items() if not res.get("is_invalidated")
        }

        # logger.debug(
        #     f"[{stage_details.stage_name}] Performing confidence boosting with "
        #     f"{len(sanitized_stage_results.results)} trusted prior stages."
        # )

        # This utility returns the details object with boosted scores and the consolidated dictionary.
        stage_details_after_consolidation, stage_consolidated_data = self.stage_utils.consolidate_all_fields_confidences(
            stage_results=sanitized_stage_results,
            stage_details=stage_details,
            fields=stage_config.fields_to_consolidate,
            ivce_dtl=ivce_dtl,
            ai_engine_cache=self.ai_engine_cache,
        )

        # Perform the threshold check on the newly boosted scores AND check for missing fields.
        # check_if_next_stage_required handles the "is field available" check internally.
        if not self.stage_utils.check_if_next_stage_required(stage_details=stage_details_after_consolidation, ivce_dtl=ivce_dtl):
            # Thresholds are now met. Return a "stop" signal and the rich data.
            return True, stage_consolidated_data
        else:
            # Thresholds are still not met. Return a "continue" signal.
            return False, None

    async def _find_best_result_after_fallthrough(self, ivce_dtl):
        """
        Selects the best result when no stage met thresholds. It first tries to find a
        winner by score, then defaults to the last-run primary stage as a fallback.
        """
        best_score = -1
        best_stage_details_dict = None
        required_fields = getattr(ivce_dtl, "fields", [])

        # --- Step 1: Run the scoring contest ---
        if required_fields:
            invalidated_stage_numbers = set()
            stages_with_validators = {
                cfg.stage_name: cfg.validation_stage_name for cfg in self.extraction_engine if cfg.validation_stage_name
            }
            failed_validators = {
                res["stage_name"]
                for res in self.stage_results.results.values()
                if res.get("is_validation_stage") and not res.get("is_final_success")
            }

            for primary_name, validator_name in stages_with_validators.items():
                if validator_name in failed_validators:
                    for res in self.stage_results.results.values():
                        if res.get("stage_name") == primary_name:
                            invalidated_stage_numbers.add(res["stage_number"])
                            break

            divisor = len(required_fields)
            for stage_number, stage_result in self.stage_results.results.items():
                if (
                    stage_number in invalidated_stage_numbers
                    or stage_result.get("is_validation_stage")
                    or stage_result.get("status") != Constants.SUCCESS_lower
                ):
                    continue

                details = stage_result.get("details", {})
                confidences = details.get(Logs.CONFIDENCE)

                # Type Check for Confidence (essential for stability)
                if not isinstance(confidences, dict):
                    continue

                score_sum = sum(confidences.get(field, 0) for field in required_fields)
                current_score = score_sum / divisor

                if current_score > best_score:
                    best_score = current_score
                    best_stage_details_dict = stage_result

        # --- Step 2: Finalize based on the outcome ---
        if best_stage_details_dict:
            # A winner was found via scoring.
            self.stage_results.final_stage_key = best_stage_details_dict["stage_number"]

            # Use standard update (pipelines.py handles formatting)
            self.stage_results.final_results.update(best_stage_details_dict["details"])

            self.stage_results.final_results[Logs.CONFIDENCE] = best_stage_details_dict["details"].get(Logs.CONFIDENCE)

        else:
            # No winner from scoring. Apply the "last stage" default logic.
            last_stage_run = self.ai_engine_cache.latest_stage_details
            if last_stage_run:
                if last_stage_run.is_validation_stage:
                    final_stage_source = self.ai_engine_cache.previous_stage_details
                else:
                    final_stage_source = last_stage_run

                if final_stage_source:
                    self.stage_results.final_stage_key = final_stage_source.stage_number
                    # REVERTED: Use standard update
                    self.stage_results.final_results.update(final_stage_source.details)

        # In all fall-through cases, the final status is DS1.
        await self.set_invoice_line_status(status=DataStates.DS1)

    async def run_stage(self, stage_config: StageConfig, ivce_dtl):
        """
        Executes a single primary stage, adds its results, and returns the details.
        This method no longer decides whether to stop the pipeline.
        """
        stage_details = None
        if self.check_if_stage_allowed(ivce_dtl=ivce_dtl, stage_name=stage_config.stage_name):
            try:
                # Pass the stage identity to the stage function
                stage_details, self.ai_engine_cache = await stage_config.stage_fn(
                    self.sdp,
                    self.ai_engine_cache,
                    ivce_dtl,
                    stage_number=stage_config.stage_number,
                    sub_stage_code=stage_config.sub_stage_code,
                )
                self.stage_results.add_stage_result(stage_details=stage_details)
            except Exception as e:
                logger.error(f"run_extraction_engine() - {stage_config.stage_name}: {str(e)}", exc_info=True)
                if not stage_details:
                    stage_details = StageDetails(
                        stage_number=stage_config.stage_number,
                        sub_stage_code=stage_config.sub_stage_code,
                        stage=stage_config.stage_name,
                        sub_stage="Error",
                        is_validation_stage=stage_config.is_validation_only,
                    )
                stage_details.status = Constants.ERROR_lower
                stage_details.details = {"message": str(e)}

        return stage_details

    async def run_validation_stage(self, validator_config: StageConfig, ivce_dtl, primary_stage_details):
        """Executes a validation stage, passing it the results of the primary stage."""
        validator_details = None
        if self.check_if_stage_allowed(ivce_dtl=ivce_dtl, stage_name=validator_config.stage_name):
            try:
                # Pass the stage identity to the validation function
                validator_details, self.ai_engine_cache = await validator_config.stage_fn(
                    self.sdp,
                    self.ai_engine_cache,
                    ivce_dtl,
                    primary_stage_details,
                    stage_number=validator_config.stage_number,
                    sub_stage_code=validator_config.sub_stage_code,
                )
                self.stage_results.add_stage_result(stage_details=validator_details)
            except Exception as e:
                logger.error(f"run_extraction_engine() - {validator_config.stage_name}: {str(e)}", exc_info=True)
                if not validator_details:
                    validator_details = StageDetails(
                        stage_number=validator_config.stage_number,
                        sub_stage_code=validator_config.sub_stage_code,
                        stage=validator_config.stage_name,
                        sub_stage="Error",
                        is_validation_stage=validator_config.is_validation_only,
                    )
                validator_details.status = Constants.ERROR_lower
                validator_details.details = {"message": str(e)}
                validator_details.is_final_success = False

        return validator_details

    async def run_extraction_engine(self, ivce_dtl, lemmatizer):
        """
        Executes the end-to-end invoice-line extraction pipeline, following a clean
        (Enrich + Boost) -> Check flow for each stage.
        """
        self.ai_engine_cache.cleaned_description_for_llm = clean_text_for_llm(lemmatizer, ivce_dtl.ITM_LDSC)
        self.ai_engine_cache.latest_stage_details = None
        self.ai_engine_cache.previous_stage_details = None

        loop_was_broken = False
        for stage_config in self.extraction_engine:
            if stage_config.is_validation_only:
                continue

            primary_stage_details = await self.run_stage(stage_config=stage_config, ivce_dtl=ivce_dtl)
            if not primary_stage_details or primary_stage_details.status == Constants.ERROR_lower:
                continue
            self.ai_engine_cache.latest_stage_details = primary_stage_details
            self.stage_results.add_stage_result(primary_stage_details)

            # --- The single, unified consolidation and check step ---
            should_stop, consolidated_data = await self._consolidate_and_recheck_thresholds(
                primary_stage_details, stage_config, ivce_dtl
            )

            if should_stop:
                # --- THRESHOLDS MET ---
                if not stage_config.validation_stage_name:
                    # Finalize using the rich, consolidated data.
                    self.stage_results.final_results.update(consolidated_data)
                    loop_was_broken = await self._finalize_and_break(primary_stage_details, ivce_dtl)
                    break

                # --- VALIDATOR LOGIC ---
                self.ai_engine_cache.previous_stage_details = primary_stage_details
                validator_config = self._get_stage_config_by_name(stage_config.validation_stage_name)

                # If a validator is configured AND allowed for this specific pipeline (YAML), run it.
                should_run_validator = validator_config and self.check_if_stage_allowed(
                    ivce_dtl=ivce_dtl, stage_name=validator_config.stage_name
                )

                if should_run_validator:
                    # Case A: Validator is active. It acts as a Gatekeeper.
                    validator_details = await self.run_validation_stage(validator_config, ivce_dtl, primary_stage_details)
                    self.ai_engine_cache.latest_stage_details = validator_details

                    if validator_details and validator_details.is_final_success:
                        # Validation Passed -> Stop and use Primary Stage Data
                        data_source_stage = self.ai_engine_cache.previous_stage_details

                        # Note: We use the consolidated data (which might contain enriched fields)
                        self.stage_results.final_results.update(consolidated_data)
                        loop_was_broken = await self._finalize_and_break(data_source_stage, ivce_dtl)
                        break
                    else:
                        # Validation Failed -> Continue Loop to find a better match
                        continue
                else:
                    # Case B: No Validator active (or skipped via Config).
                    # The Primary Stage met thresholds and is trusted by default.
                    self.stage_results.final_results.update(consolidated_data)
                    loop_was_broken = await self._finalize_and_break(primary_stage_details, ivce_dtl)
                    break

        if not loop_was_broken:
            await self._find_best_result_after_fallthrough(ivce_dtl)

    async def _finalize_and_break(self, data_source_stage, ivce_dtl):
        """Helper to set the final stage key, handle final consolidation, and signal the loop to break."""
        self.stage_results.final_stage_key = data_source_stage.stage_number
        await self.set_invoice_line_status(status=DataStates.RC_AI)

        if not data_source_stage.is_validation_stage and Logs.UNSPSC in ivce_dtl.fields:
            await self._consolidate_unspsc_if_needed(data_source_stage, ivce_dtl)

        return True  # Signal to the caller that the loop should break.


class StageResults:

    def __init__(self):
        self.results = {}  # Dictionary to hold stage-wise results
        self.final_results = {}  # Dictionary to hold the final consolidated output
        self.first_stage_key = 0
        self.first_sub_stage_key = None
        self.is_rental = None

    def add_stage_result(self, stage_details):
        """
        Store results for a specific stage.

        Args:
            stage_name (str): Name of the processing stage.
            data (Any): Processed data (can be text, JSON, embeddings, etc.).
            status (str): Processing status ('success', 'failed', 'pending').
            error (Optional[str]): Error message if the stage failed.
        """
        if self.first_stage_key == 0:
            self.first_stage_key = stage_details.stage_number
            self.first_sub_stage_key = stage_details.sub_stage_code

        # self.results[stage_details.sub_stage_code] = stage_details.__dict__
        self.results[stage_details.stage_number] = stage_details.__dict__

    def get_stage_result(self, stage_number: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve results of a specific processing stage.

        Args:
            stage_name (int): The stage whose result is needed.

        Returns:
            dict: Data and metadata of the requested stage, or None if not found.
        """
        return self.results.get(stage_number)

    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve results from all stages.

        Returns:
            dict: All stored results.
        """
        return self.results

    def clear_results(self):
        """Clear all stored stage results."""
        self.results.clear()
