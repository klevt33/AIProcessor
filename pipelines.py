from copy import copy, deepcopy
from typing import Any

from ai_engine import AIEngine, StageResults
from constants import Constants, DataStates, DescriptionCategories, Logs, SpecialCases, StageNames, SubStageNames
from logger import logger
from sql_utils import count_number_in_trainable_table, get_invoice_duplicate_detail_ids
from utils import get_current_datetime_cst, get_spl_cases, have_min_length, is_valid_aks_part_number, is_valid_upc_number


class Pipelines:

    def __init__(self, config, sdp):
        self.config = config
        self.sdp = sdp
        self.fields = [Logs.MFR_NAME, Logs.PRT_NUM, Logs.UNSPSC]
        self.additional_fields = [Logs.AKS_PRT_NUM, Logs.UPC, Logs.DESCRIPTION]

        self.special_cases = get_spl_cases(config.app_root)
        self.stage_mapping = self.get_default_pipeline_stage_mapping()

    def get_default_pipeline_stage_mapping(self):
        """
        This returns the default or normal case of stage mapping to be considered.

        Returns:
            dict[str: list[str]]: mapping stage and corresponding sub stages
        """
        default_case = self.special_cases[SpecialCases.CASE_0]
        return default_case["STAGES"]

    def calculate_stage_counts(self, ivce_dtl):
        """
        It calculates stage level count and sub stage level count for checking later
        to decide whether to run extraction engine or not

        Args:
            ivce_dtl (YourInvoiceDetailClass):
                Invoice detail object that includes at least:
                - `.stage_mapping` (dict): current mapping of stages to sub-stages.
        """
        ivce_dtl.stage_count = len(ivce_dtl.stage_mapping)
        ivce_dtl.sub_stage_count = sum(len(v) for v in self.stage_mapping.values())

    async def check_min_length_condition(self, request_details: dict, ivce_dtl):
        """
        Ensure item descriptions meet the minimum length requirement, and if not,
        mark the invoice for classification and enqueue the description classifier.

        This method checks whether `ivce_dtl.ITM_LDSC` (the item description)
        satisfies the minimum length defined in `self.config.min_description_len`.
        If the description is too short, it:
        1. Sets `request_details["classify"] = True` to trigger classification.
        2. Adds the DESCRIPTION_CLASSIFIER sub-stage under the CLASSIFICATION
            stage in `ivce_dtl.stage_mapping` via `apply_additions`.

        Args:
            request_details (dict):
                Mutable dict of processing flags and metadata for this invoice.
            ivce_dtl (YourInvoiceDetailClass):
                Invoice detail object that includes at least:
                - `.ITM_LDSC` (str): the item description text.
                - `.stage_mapping` (dict): current mapping of stages to sub-stages.

        Returns:
            tuple:
                - The (possibly modified) `request_details` dict.
                - The `ivce_dtl` object, with:
                    * `request_details["classify"]` set to True if description
                    is below minimum length.
                    * `.stage_mapping` updated to include
                    DESCRIPTION_CLASSIFIER under CLASSIFICATION when needed.
        """
        if not have_min_length(ivce_dtl.ITM_LDSC, length=self.config.min_description_len):
            request_details["classify"] = True
            ivce_dtl.stage_mapping = await self.apply_additions(
                stages=ivce_dtl.stage_mapping, add_stages={StageNames.CLASSIFICATION: SubStageNames.DESCRIPTION_CLASSIFIER}
            )

        return request_details, ivce_dtl

    async def apply_additions(self, stages: dict[str, list[str]], add_stages: dict[str, str | list[str]]) -> dict[str, list[str]]:
        """
        Add specified sub-stages into a mapping of stages, returning a new dict.

        For each entry in `add_stages`, if the stage exists it will append the
        given sub-stage(s) (avoiding duplicates). If the stage does not exist,
        it will be created with the provided sub-stages.

        Args:
            stages (dict[str, list[str]]):
                Original mapping of stage names to lists of sub-stage names.
            add_stages (dict[str, str | list[str]]):
                Mapping of stage names to one or more sub-stage names to add.

        Returns:
            dict[str, list[str]]:
                A brand-new dictionary containing the original stages plus any
                newly added sub-stages.
        """
        # shallow‐copy lists so we don’t mutate the input
        final_stages = {stage: subs.copy() for stage, subs in stages.items()}

        for stage, subs in add_stages.items():
            # normalize to list
            to_add = subs if isinstance(subs, (list, tuple)) else [subs]

            if stage not in final_stages:
                final_stages[stage] = []

            for sub in to_add:
                if sub not in final_stages[stage]:
                    final_stages[stage].append(sub)
                else:
                    logger.debug(f"Sub-stage {sub!r} already present under stage {stage!r}")

        return final_stages

    async def apply_skips(self, stages: dict[str, list[str]], skip_stages: dict[str, str]) -> dict[str, list[str]]:
        """
        Remove specified sub-stages from a mapping of stages, returning a new filtered dict.

        For each entry in `skip_stages`, if the stage exists and the given sub-stage
        is present, that sub-stage will be removed. Any stage that ends up with no
        sub-stages is omitted entirely. Invalid stage or sub-stage names are logged
        at WARNING level.

        Args:
            stages (dict[str, list[str]]):
                Original mapping of stage names to lists of sub-stage names.
            skip_stages (dict[str, str]):
                Mapping of stage names to the single sub-stage name to remove.

        Returns:
            dict[str, list[str]]:
                A brand-new dictionary containing only the stages and sub-stages
                remaining after the specified skips. The input `stages` is never mutated.

        Logs:
            WARNING if a `skip_stages` key refers to a non-existent stage or
            if its value refers to a sub-stage not present in the original.
        """

        # 1) Clone everything (deep enough to avoid mutating the input)
        final_stages = {stage: subs.copy() for stage, subs in stages.items()}

        # 2) Validate skip entries
        for stage, subs_to_skip in skip_stages.items():
            if stage not in stages:
                logger.warning(f"Invalid stage to skip: {stage!r}")
            elif len(subs_to_skip) > 0:
                for sub_to_skip in subs_to_skip:
                    if sub_to_skip not in stages[stage]:
                        logger.warning(f"Invalid sub-stage to skip: {sub_to_skip!r} for stage {stage!r}")

        # 3) Remove the requested sub-stages
        for stage, subs_to_skip in skip_stages.items():
            # delete the entire stage if length of sub stages to skip is 0
            if len(subs_to_skip) == 0:
                del final_stages[stage]

            # only act on ones that passed validation
            elif stage in final_stages:
                for sub_to_skip in subs_to_skip:
                    if sub_to_skip in final_stages[stage]:
                        final_stages[stage].remove(sub_to_skip)
                        # if that was the last (or only one) substage, drop the entire stage
                        if not final_stages[stage]:
                            del final_stages[stage]

        return final_stages

    async def check_special_cases(self, pre_process_details: dict, ivce_dtl):
        """
        Inspect an invoice's detail record for any configured “special cases” and
        update both the request flags and the invoice detail accordingly.

        Args:
            pre_process_details (dict):
                pre-processing details that affect the actual pipeline.
            ivce_dtl (InvoiceDetail):

        Returns:
            tuple:
                - The (possibly modified) `request_details` dict.
                - The `ivce_dtl` object, with:
                    * `.is_special_case` = True if matched
                    * `.special_case` = the matched spec dict
                    * `.stage_mapping` potentially pruned via `apply_skips`
        """
        ivce_params = ivce_dtl.__dict__
        ivce_dtl.is_special_case = False

        for name, sc in self.special_cases.items():
            input_col = sc.get("INPUT_COLUMN", None)
            input_values = sc.get("INPUT_VALUES", None)
            fields = sc.get("OUTPUT_FIELDS", self.fields)

            if input_col is not None and input_values is not None and ivce_params[input_col] in input_values:
                pre_process_details["classify"] = False
                pre_process_details["special_case_type"] = name
                # request_details["special_case"] = ivce_params[input_col]
                pre_process_details["special_case"] = sc["NAME"]
                pre_process_details["fields_to_extract"] = fields

                await self.apply_special_case(sc=sc, case_name=name, ivce_dtl=ivce_dtl)

                # Write case specific code if any required
                # if name == SpecialCases.CASE_1:
                #     pass
                # if name == SpecialCases.CASE_2:
                #     pass
                break
        return ivce_dtl

    async def apply_special_case(self, case_name: str, ivce_dtl: Any, sc: dict | None = None):
        logger.info(f"Applying special case '{case_name}'.")
        if sc is None:
            sc = self.special_cases[case_name]

        skip_stages = sc.get("SKIP_STAGES", {})
        fields = sc.get("OUTPUT_FIELDS", self.fields)

        ivce_dtl.fields = fields
        ivce_dtl.is_special_case = True
        ivce_dtl.special_case = sc
        ivce_dtl.special_case_type = case_name

        if len(skip_stages) > 0:
            ivce_dtl.stage_mapping = await self.apply_skips(stages=deepcopy(ivce_dtl.stage_mapping), skip_stages=skip_stages)
        logger.debug(f"Applied special case '{case_name}'.")

        # Update stage counts after applying special case each time
        self.calculate_stage_counts(ivce_dtl)

    async def is_akp_or_upc_exists(self, ivce_dtl, pre_process_details):
        aks_pn = ivce_dtl.AKS_PRT_NUM
        upc_cd = ivce_dtl.UPC_CD

        if (
            is_valid_aks_part_number(akp=aks_pn)
            and await count_number_in_trainable_table(sdp=self.sdp, number=aks_pn, field=Constants.AKP) > 0
        ):
            pre_process_details[Constants.MESSAGE] = "ASK number is already present. Exiting the AI process..."
            logger.info(f"ASK number {aks_pn} is already present. Exiting the AI process...")
            return True

        if (
            is_valid_upc_number(upc=upc_cd)
            and await count_number_in_trainable_table(sdp=self.sdp, number=upc_cd, field=Constants.UPC) > 0
        ):
            pre_process_details[Constants.MESSAGE] = "UPC code is already present. Exiting the AI process..."
            logger.info(f"UPC code {upc_cd} is already present. Exiting the AI process...")
            return True

        return False

    async def process(self, request_details, ivce_dtl):
        """
        Orchestrate a single invoice through special-case handling and the main pipeline.

        1. Initializes `ivce_dtl.stage_mapping` as a copy of the default stages.
        2. Calls `check_special_cases` to detect and mark any configured special case.
        3. If a special case is detected:
            - Routes CASE_1 invoices into the generic pipeline via `run_generic_pipeline`.
        Otherwise:
            - Runs the full, default pipeline via `run_pipeline`.

        Args:
            request_details (dict):
                Mutable dict of processing flags and metadata for this invoice.
            ivce_dtl (InvoiceDetail):

        Returns:
            stage_results
        """
        pipeline_results = PipelineResults()

        # Update invoice details object with required info
        ivce_dtl.fields = self.fields
        ivce_dtl.stage_mapping = copy(self.stage_mapping)
        self.calculate_stage_counts(ivce_dtl)

        # Update pre-process details with required info
        pipeline_results.pre_process_details[Constants.ITM_AI_LDSC] = ivce_dtl.ITM_LDSC

        # Exit the process if ASK or UPC already exists.
        if await self.is_akp_or_upc_exists(ivce_dtl=ivce_dtl, pre_process_details=pipeline_results.pre_process_details):
            pipeline_results.stage_results.final_results = {Logs.IVCE_LINE_STATUS: DataStates.RC_RPA}
            pipeline_results.stage_results.status = Constants.SUCCESS_lower
            return pipeline_results

        # Check specila cases
        ivce_dtl = await self.check_special_cases(pipeline_results.pre_process_details, ivce_dtl)

        # Get the duplicate IDs before processing, to avoid IVCE_LNE_STAT change issue
        dup_ids = await get_invoice_duplicate_detail_ids(
            sdp=self.sdp, invoice_detail_id=ivce_dtl.IVCE_DTL_UID, invoice_num=ivce_dtl.IVCE_NUM
        )

        pipeline_results.stage_results = await self.run_pipeline(pipeline_results, request_details, ivce_dtl)

        # Update post-process details if any
        if pipeline_results.stage_results.status == Constants.SUCCESS_lower:
            self.prepare_duplicate_ids_for_writer(pipeline_results=pipeline_results, dup_ids=dup_ids)
            # await self.process_duplicate_ids(pipeline_results=pipeline_results, ivce_dtl=ivce_dtl, dup_ids=dup_ids)

            # if "duplicate_id_errors" in pipeline_results.post_process_details:
            #     pipeline_results.stage_results.status = Constants.ERROR_lower
            #     pipeline_results.stage_results.message = "Error occurred during updating duplicate detail IDs."

        return pipeline_results

    async def overwrite_verification_indicator_if_requreid(self, pipeline_results, stage_results):
        """
        Set the data verification indicator to 'N' if invoice line status is decided as DS1.
        It means even though data is verified, we are not trusting because it doesn't cross the threshold.
        Args:
            pipeline_results: Object containing all pipeline details
            stage_results: Object containing all stages results and final results
        """
        # HIST: Comment this and uncomment below commented one
        # stage_results.final_results[Logs.IS_VERIFIED] = Constants.N
        # pipeline_results.post_process_details.update({Logs.IS_VERIFIED: Constants.N})

        if stage_results.final_results[Logs.IVCE_LINE_STATUS] == DataStates.DS1:
            stage_results.final_results[Logs.IS_VERIFIED] = Constants.N
            pipeline_results.post_process_details.update({Logs.IS_VERIFIED: Constants.N})

        elif stage_results.final_results.get(Logs.CATEGORY, None) in (
            DescriptionCategories.LOT,
            DescriptionCategories.GENERIC,
            None,
        ):
            stage_results.final_results[Logs.IS_VERIFIED] = Constants.N
            pipeline_results.post_process_details.update({Logs.IS_VERIFIED: Constants.N})

    # async def process_duplicate_ids(self, pipeline_results, ivce_dtl, dup_ids):
    #     """
    #     Update the post process results. It contains details that are being written to SDP which are
    #     different from process output (final_results)
    #     Args:
    #         pipeline_results: Object containing all pipeline details
    #         ivce_dtl: Invoice details object
    #         dup_ids: list of duplicate IDs
    #     """
    #     pipeline_results.post_process_details.update({"duplicate_detail_uids": dup_ids})

    #     if len(dup_ids) == 0:
    #         logger.info(f"There are no duplicate detail UIDs for invoice detail UID {ivce_dtl.IVCE_DTL_UID}")

    #     for uid in dup_ids:
    #         try:
    #             # Update invoice details table and Upsert into invoice tracking table
    #             await update_invoice_detail_and_tracking_values_by_id(
    #                 sdp=self.sdp,
    #                 invoice_detail_id=uid,
    #                 stage_results=pipeline_results.stage_results,
    #                 is_duplicate=True,
    #                 parent_detail_id=ivce_dtl.IVCE_DTL_UID,
    #             )
    #             logger.debug(f"Updated extracted details in IVCE_DTL and IVCE_TRKG_MSTR for duplicate invoice detail UID {uid}")
    #             success_details = {
    #                 "status": Constants.SUCCESS_lower,
    #                 "invoice_detail_uid": uid,
    #                 "message": f"Updated detail ID {uid} successfully.",
    #             }

    #             duplicate_id_details = pipeline_results.post_process_details.get("duplicate_id_details", [])
    #             duplicate_id_details.append(success_details)
    #             pipeline_results.post_process_details["duplicate_id_details"] = duplicate_id_details

    #         except Exception as e:
    #             logger.error(f"Error occurred while updaitng SDP {str(e)}", exc_info=True)
    #             error_details = {
    #                 "status": Constants.ERROR_lower,
    #                 "invoice_detail_uid": uid,
    #                 "message": f"Error occurred while updaitng SDP: {str(e)}",
    #             }

    #             duplicate_id_errors = pipeline_results.post_process_details.get("duplicate_id_errors", [])
    #             duplicate_id_errors.append(error_details)
    #             pipeline_results.post_process_details["duplicate_id_errors"] = duplicate_id_errors

    def prepare_duplicate_ids_for_writer(self, pipeline_results, dup_ids):
        """
        Records the list of duplicate detail UIDs in the post_process_details
        so it can be saved to the Cosmos DB document for the background writer to handle.
        """
        pipeline_results.post_process_details.update({"duplicate_detail_uids": dup_ids})

        if len(dup_ids) > 0:
            logger.info(
                f"Identified {len(dup_ids)} duplicate IDs for invoice detail UID"
                f" {pipeline_results.stage_results.final_results.get('IVCE_DTL_UID')}. They will be processed by the SQL Writer"
                " Service."
            )

    async def run_pipeline(self, pipeline_results, request_details, ivce_dtl):
        request_details, ivce_dtl = await self.check_min_length_condition(request_details, ivce_dtl)

        ai_engine = AIEngine(config=self.config, sdp=self.sdp)
        await ai_engine.async_init()
        ai_engine.special_case_fn = self.apply_special_case
        stage_results = await ai_engine.process_description(ivce_dtl=ivce_dtl)

        # Update stage_results to write into SDP
        await self.prepare_data_to_write_into_sdp(
            pipeline_results=pipeline_results, stage_results=stage_results, ivce_dtl=ivce_dtl
        )

        # Overwrite if required
        await self.overwrite_verification_indicator_if_requreid(pipeline_results, stage_results)

        # Trim the fields as per SDP column size
        self.trim_field_values_as_per_sdp(stage_results=stage_results)

        # try:
        #     # Update invoice details table and Upsert into invoice tracking table
        #     await update_invoice_detail_and_tracking_values_by_id(
        #         sdp=self.sdp, invoice_detail_id=ivce_dtl.IVCE_DTL_UID, stage_results=stage_results
        #     )
        #     logger.debug(
        #         f"Updated extracted details in IVCE_DTL and IVCE_TRKG_MSTR for invoice detail UID {ivce_dtl.IVCE_DTL_UID}"
        #     )

        # except Exception as e:
        #     logger.error(f"Error occurred while updaitng SDP {str(e)}", exc_info=True)
        #     stage_results.status = Constants.ERROR_lower
        #     stage_results.message = f"Error occurred while updaitng SDP: {str(e)}"

        return stage_results

    def run_generic_pipeline(self):
        pass

    def populate_default_final_results(self, is_error=False) -> dict:
        default_final_results = {}

        if not is_error:
            default_final_results.update({Logs.IVCE_LINE_STATUS: DataStates.DS1, Logs.IS_VERIFIED: Constants.N})
        else:
            default_final_results.update({Logs.IVCE_LINE_STATUS: DataStates.AI_ERROR})

        default_final_results.update({Logs.IS_VERIFIED: Constants.N})
        default_final_results.update({Logs.END_TIME: get_current_datetime_cst()})
        return default_final_results

    def trim_field_values_as_per_sdp(self, stage_results):
        try:
            final_fields = stage_results.final_results
            if Logs.DESCRIPTION in final_fields and len(final_fields[Logs.DESCRIPTION]) > 2000:
                final_fields[Logs.DESCRIPTION] = final_fields[Logs.DESCRIPTION][:2000]
            if Logs.PRT_NUM in final_fields and len(final_fields[Logs.PRT_NUM]) > 50:
                final_fields[Logs.PRT_NUM] = final_fields[Logs.PRT_NUM][:50]
            if Logs.MFR_NAME in final_fields and len(final_fields[Logs.MFR_NAME]) > 50:
                final_fields[Logs.MFR_NAME] = final_fields[Logs.MFR_NAME][:50]
            if Logs.UPC in final_fields and len(final_fields[Logs.UPC]) > 50:
                final_fields[Logs.UPC] = final_fields[Logs.UPC][:50]
            if Logs.UNSPSC in final_fields and len(final_fields[Logs.UNSPSC]) > 20:
                final_fields[Logs.UNSPSC] = final_fields[Logs.UNSPSC][:20]
            if Logs.WEB_SEARCH_URL in final_fields and len(final_fields[Logs.WEB_SEARCH_URL]) > 2000:
                final_fields[Logs.WEB_SEARCH_URL] = final_fields[Logs.WEB_SEARCH_URL][:2000]

            stage_results.final_results.update(final_fields)
        except Exception as e:
            logger.error(f"Error occurred in trim_field_values_as_per_sdp(). {str(e)}", exc_info=True)

    async def prepare_data_to_write_into_sdp(self, pipeline_results, stage_results, ivce_dtl):
        """
        Prepares the final fields, stages, confidences, and confidence stages
        consolidated to write into SDP and CosmosDB.
        Ensures all expected keys are always present.
        """

        def get_field(field: str, results: dict) -> str:
            try:
                value = results.get(field, Constants.EMPTY_STRING)
                return value if value is not None and value != Constants.EMPTY_STRING else Constants.UNDEFINED
            except Exception as err:
                logger.error(f"Error fetching field {field}: {err}", exc_info=True)
                return Constants.UNDEFINED

        def get_conf(field: str, results: dict) -> float:
            try:
                return results.get(Logs.CONFIDENCE, {}).get(field, 0.0)
            except Exception as err:
                logger.error(f"Error fetching field {field}: {err}", exc_info=True)
                return 0.0

        def get_stage(key):
            stage_results = results.get(key, {})
            stage_details = stage_results.get("details", {})
            stage_name = stage_results.get("stage_name", Constants.UNDEFINED)
            sub_stage_name = stage_results.get("sub_stage_name", Constants.UNDEFINED)
            return stage_name, sub_stage_name, stage_details, stage_results

        def get_category_fields(stage_details, stage=Constants.UNDEFINED):
            return {
                Logs.CATEGORY: stage_details.get(Logs.CATEGORY, Constants.UNDEFINED),
                Logs.CATEGORY_ID: stage_details.get(Logs.CATEGORY_ID, Constants.UNDEFINED),
                Logs.CONF_CATEGORY: stage_details.get(Logs.CONFIDENCE, 0.0),
                Logs.STAGE_CATEGORY: stage,
            }

        def get_other_fields():
            return {
                Logs.END_TIME: get_current_datetime_cst(),
                Logs.WEB_SEARCH_URL: get_field(Logs.WEB_SEARCH_URL, final_stage_details),
            }

        def get_ldsc_fields():
            return {Logs.DESCRIPTION: get_field(Logs.DESCRIPTION, final_stage_details), Logs.STAGE_DESCRIPTION: final_stage}

        def get_verification_fields():
            is_verified = get_field(Logs.IS_VERIFIED, final_stage_details)
            if isinstance(is_verified, bool) and is_verified:
                is_verified = Constants.Y
            else:
                is_verified = Constants.N
            return {Logs.IS_VERIFIED: is_verified}

        def get_mfr_fields(ivce_dtl):
            field_mfr_nm = get_field(Logs.MFR_NAME, final_stage_details)
            field_is_mfr_clean = get_field(Logs.IS_MFR_CLEAN, final_stage_details)
            if field_mfr_nm == Constants.UNDEFINED and getattr(ivce_dtl, "special_case_type", None) == SpecialCases.CASE_2:
                field_mfr_nm = DescriptionCategories.LOT
                field_is_mfr_clean = True
            return {
                Logs.MFR_NAME: field_mfr_nm,
                Logs.STAGE_MFR_NAME: final_stage if field_mfr_nm != Constants.UNDEFINED else Constants.UNDEFINED,
                Logs.CONF_MFR_NAME: get_conf(Logs.MFR_NAME, final_stage_details),
                Logs.CONF_STAGE_MFR_NAME: final_stage,
                Logs.IS_MFR_CLEAN: field_is_mfr_clean,
            }

        def get_pn_fields():
            field_pn = get_field(Logs.PRT_NUM, final_stage_details)
            return {
                Logs.PRT_NUM: field_pn,
                Logs.STAGE_PRT_NUM: final_stage if field_pn != Constants.UNDEFINED else Constants.UNDEFINED,
                Logs.CONF_PRT_NUM: get_conf(Logs.PRT_NUM, final_stage_details),
                Logs.CONF_STAGE_PRT_NUM: final_stage,
            }

        def get_unspsc_fields():
            field_unspsc = get_field(Logs.UNSPSC, final_stage_details)
            return {
                Logs.UNSPSC: field_unspsc,
                Logs.STAGE_UNSPSC: final_stage if field_unspsc != Constants.UNDEFINED else Constants.UNDEFINED,
                Logs.CONF_UNSPSC: get_conf(Logs.UNSPSC, final_stage_details),
                Logs.CONF_STAGE_UNSPSC: final_stage,
            }

        def get_upc_fields():
            field_upc = get_field(Logs.UPC, final_stage_details)
            return {Logs.UPC: field_upc, Logs.STAGE_UPC: final_stage if field_upc != Constants.UNDEFINED else Constants.UNDEFINED}

        def get_akp_fields():
            field_akp = get_field(Logs.AKS_PRT_NUM, final_stage_details)
            return {
                Logs.AKS_PRT_NUM: field_akp,
                Logs.STAGE_AKS_PRT_NUM: final_stage if field_akp != Constants.UNDEFINED else Constants.UNDEFINED,
            }

        def sort_final_results():
            """
            Sort keys for CosmosDB storage consistency
            """
            stage_results.final_results = dict(sorted(stage_results.final_results.items()))

        def update_classification_error(stage_details):
            stage_results.status = Constants.ERROR_lower  # Set status to error
            stage_results.message = stage_details.get("message", "No error message - final stage results preparation")
            logger.warning("Error in classification stage. Populating defaults.")

            sort_final_results()

        try:
            fields = ivce_dtl.fields
            # Default empty structure
            final_results = deepcopy(getattr(stage_results, "final_results", {}))
            results = getattr(stage_results, "results", {})
            # print(results)
            if not results:
                if ivce_dtl.stage_count == 0:
                    stage_results.status = Constants.SUCCESS_lower  # Set status to success
                    stage_results.final_results = self.populate_default_final_results()
                    logger.info("No stage results found. Populating defaults.")
                else:
                    stage_results.status = Constants.ERROR_lower  # Set status to error
                    stage_results.final_results = self.populate_default_final_results(is_error=True)
                    logger.warning("No stage results found. Populating defaults.")

                stage_results.message = "No stage results found. Populating defaults."
                sort_final_results()
                return stage_results  # Exit early since results are missing

            # Extract first stage info
            first_stage_name, first_sub_stage_name, first_stage_details, first_stage_results = get_stage(
                stage_results.first_stage_key
            )
            " - ".join([first_stage_name, first_sub_stage_name])

            # Extract final stage info
            final_stage_name, final_sub_stage_name, final_stage_details, final_stage_results = get_stage(
                stage_results.final_stage_key
            )
            final_stage = " - ".join([final_stage_name, final_sub_stage_name])

            # Category info
            " - ".join([StageNames.CLASSIFICATION, SubStageNames.DESCRIPTION_CLASSIFIER])
            stage_results.final_results = self.populate_default_final_results()
            stage_results.final_results.update({Constants.IS_RENTAL: stage_results.is_rental})

            # Verification fields
            verification_fields = get_verification_fields()
            stage_results.final_results.update(verification_fields)

            # LDSC fields
            ldsc_fields = get_ldsc_fields()
            stage_results.final_results.update(ldsc_fields)

            if first_stage_name == StageNames.CLASSIFICATION:
                if first_stage_results.get("status", Constants.ERROR_lower) == Constants.ERROR_lower:
                    update_classification_error(first_stage_details)
                    return stage_results  # Exit early since results are missing

                # Find max stage number of classification sub stages.
                # For simplicity "first_stage_last_substage_key" is written fsls
                fsls_key = max(
                    [
                        k
                        for k, v in results.items()
                        if v.get("stage_name") == StageNames.CLASSIFICATION and not v.get("ignore_stage")
                    ]
                )
                fsls_stage_name, fsls_sub_stage_name, fsls_stage_details, fsls_stage_results = get_stage(fsls_key)
                fsls_stage = " - ".join([fsls_stage_name, fsls_sub_stage_name])

                if fsls_stage_results.get("status", Constants.ERROR_lower) == Constants.ERROR_lower:
                    update_classification_error(fsls_stage_details)
                    return stage_results  # Exit early since results are missing

                category_info = get_category_fields(fsls_stage_details, fsls_stage)
                stage_results.final_results.update(category_info)

            if final_stage_name == StageNames.CLASSIFICATION:
                stage_results.status = Constants.SUCCESS_lower  # Mark as completed
                stage_results.final_results.update(
                    {
                        # write cat conf in mfr conf if it is last stage
                        Logs.CONF_MFR_NAME: stage_results.final_results[Logs.CONF_CATEGORY],
                        # Logs.IVCE_LINE_STATUS: DataStates.DS1,
                        Logs.IS_MFR_CLEAN: get_field(Logs.IS_MFR_CLEAN, final_stage_details),
                    }
                )
                stage_results.final_results.update(final_results)
                sort_final_results()
                return stage_results

            # ASK PN fields - OR condition added temporarily
            if Logs.AKS_PRT_NUM in fields or Logs.AKS_PRT_NUM in final_stage_details:
                akp_fields = get_akp_fields()
                stage_results.final_results.update(akp_fields)

            # UPC fields - OR condition added temporarily
            if Logs.UPC in fields or Logs.UPC in final_stage_details:
                upc_fields = get_upc_fields()
                stage_results.final_results.update(upc_fields)

            # PN fields
            if Logs.PRT_NUM in fields:
                pn_fields = get_pn_fields()
                stage_results.final_results.update(pn_fields)

            # MFR fields
            if Logs.MFR_NAME in fields:
                # Pass ivce_dtl, not pipeline_results.pre_processing_details as it is not updated if item identified as LOT by AI
                # and converted case to special case during process
                mfr_fields = get_mfr_fields(ivce_dtl)
                stage_results.final_results.update(mfr_fields)

            # UNSPSC fields
            if Logs.UNSPSC in fields:
                unspsc_fields = get_unspsc_fields()
                stage_results.final_results.update(unspsc_fields)

            # Other fields
            other_fields = get_other_fields()
            stage_results.final_results.update(other_fields)

            # Overwrite if there are any consolidated results - line status, etc
            stage_results.final_results.update(final_results)

            # Re-apply verification fields to ensure they are not overwritten
            stage_results.final_results.update(verification_fields)

            # Explicitly add the parent invoice detail UID to the final results payload.
            # This ensures it's available for logging and makes the data contract more complete.
            stage_results.final_results[Logs.IVCE_DTL_UID] = ivce_dtl.IVCE_DTL_UID

            sort_final_results()

            if final_stage_results.get("status", Constants.ERROR_lower) == Constants.ERROR_lower:
                stage_results.status = Constants.ERROR_lower  # Set status to error
                stage_results.message = final_stage_details.get("message", "No error message - final stage results preparation")
                stage_results.final_results.update({Logs.IVCE_LINE_STATUS: DataStates.AI_ERROR})
            else:
                stage_results.status = Constants.SUCCESS_lower  # Mark as completed
                if stage_results.final_results.get(Logs.IS_MFR_CLEAN, None) is False:
                    stage_results.final_results.update({Logs.IVCE_LINE_STATUS: DataStates.DS1})

        except Exception as e:
            logger.exception(f"Error while preparing final results. {str(e)}")
            # If an error occurs, still ensure all expected fields exist
            stage_results.status = Constants.ERROR_lower
            stage_results.message = f"Error while preparing final results. Info: {str(e)}"
            stage_results.final_results = self.populate_default_final_results()
            stage_results.final_results.update({Logs.IVCE_LINE_STATUS: DataStates.AI_ERROR})

        return stage_results


class PipelineResults:

    def __init__(self):
        # Holds preprocess info
        self.pre_process_details = {}

        # Holds stage results
        self.stage_results = StageResults()

        # Holds postprocess info
        self.post_process_details = {}
