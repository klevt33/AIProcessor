from constants import Constants, Logs
from exceptions import InvoiceProcessingError
from logger import logger
from utils import get_current_datetime_cst


class ClassifierUtils:

    def __init__(self, config, category_utils):
        self.config = config
        self.category_utils = category_utils
        self.text_analytics_client = config.azure_clients.text_analytics_client

    async def classify_using_azure_custom_language_model(
        self,
        cleaned_description: str,
        stage_details,
        project_name: str,
        deployment_name: str,
        category_id_fn: callable,
        parent_category_id: str = None,
    ):
        """Generic classifier wrapper for Azure Language Studio classification."""

        try:
            poller = self.text_analytics_client.begin_single_label_classify(
                documents=[cleaned_description], project_name=project_name, deployment_name=deployment_name
            )

            document_results = list(poller.result())
            classification_result = document_results[0] if document_results else None

            if classification_result is None or classification_result.is_error:
                code = getattr(classification_result.error, "code", "UNKNOWN")
                msg = getattr(classification_result.error, "message", "No message")
                message = f"Classification failed (stage={stage_details.sub_stage_name}) with code '{code}' and message '{msg}'"
                stage_details.status = Constants.ERROR_lower
                stage_details.details = {
                    Logs.END_TIME: get_current_datetime_cst(),
                    Constants.MESSAGE: message,
                    Logs.DESCRIPTION: cleaned_description,
                }
                raise InvoiceProcessingError(message=message)

            if classification_result.kind == Constants.CUSTOM_DOCUMENT_CLASSIFICATION:
                classification = classification_result.classifications[0]
                logger.info(
                    f"[{stage_details.sub_stage_name}] Classified as '{classification.category}' "
                    f"with confidence {classification.confidence_score}"
                )

                category_id = category_id_fn(classification.category, parent_category_id)

                stage_details.status = Constants.SUCCESS_lower
                stage_details.details = {
                    Logs.IS_VERIFIED: False,
                    Logs.IS_MFR_CLEAN: True,
                    Logs.END_TIME: get_current_datetime_cst(),
                    Logs.CATEGORY: classification.category,
                    Logs.CATEGORY_ID: category_id,
                    Logs.CONFIDENCE: classification.confidence_score * 100,
                    Logs.DESCRIPTION: cleaned_description,
                }

        except Exception as e:
            logger.error(f"{stage_details.sub_stage_name}: Unexpected error during processing: {str(e)}", exc_info=True)
            stage_details.status = Constants.ERROR_lower
            stage_details.details = {
                Logs.END_TIME: get_current_datetime_cst(),
                Constants.MESSAGE: f"Unexpected error: {str(e)}",
                Logs.DESCRIPTION: cleaned_description,
            }

        return stage_details
