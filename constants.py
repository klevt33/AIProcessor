"""
## Overview
This module defines a collection of constants, enumerations, and classes for managing configuration keys, logging statuses,
stage names, sub-stage names, description categories, environments, local file paths, log levels, database objects,
data states, and manufacturer name match types. These are primarily used in applications related to AI, NLP, and data processing.
"""

import os
from enum import Enum


# Constants class to store various configuration keys and predefined values
class Constants:
    """
    A collection of constants used across the application for configuration and predefined values.

    Attributes:
    -----------
    KEY_VAULT : str
        Key vault configuration key.
    LOGGING_KEY : str
        Key for logging configuration.
    """

    # Config file keys start
    KEY_VAULT = "KEY_VAULT"

    AKP = "AKP"
    UPC = "UPC"

    SPENDREPORT_WA = "SPENDREPORT_WA"

    LANGUAGE_STUDIO = "LANGUAGE_STUDIO"
    DESCRIPTION_CLASSIFIER = "DESCRIPTION_CLASSIFIER"
    LOT_CLASSIFIER = "LOT_CLASSIFIER"
    RENTAL_CLASSIFIER = "RENTAL_CLASSIFIER"
    PROJECT_NAME = "PROJECT_NAME"
    DEPLOYMENT_NAME = "DEPLOYMENT_NAME"

    OPENAI_DEPLOYMENT = "OPENAI_DEPLOYMENT"
    BASE_LLM = "BASE_LLM"
    FINETUNED_LLM = "FINETUNED_LLM"
    EMBEDDING_MODEL = "EMBEDDING_MODEL"
    CONTEXT_VALIDATOR = "CONTEXT_VALIDATOR"
    AZURE_AI_SEARCH = "AZURE_AI_SEARCH"
    AZURE_SEARCH = "AZURE_SEARCH"

    AI_AGENT = "AI_AGENT"
    AZURE_AI_AGENT = "AZURE_AI_AGENT"
    AZURE_AI_AGENT_WITH_BING_SEARCH = "AZURE_AI_AGENT_WITH_BING_SEARCH"

    COSMOS_DB = "COSMOS_DB"

    SDP = "SDP"
    # Config file keys end

    SQL = "SQL"
    CSV = "CSV"
    LLM = "LLM"
    AGENT = "AGENT"
    CUSTOM_DOCUMENT_CLASSIFICATION = "CustomDocumentClassification"

    READ = "READ"
    WRITE = "WRITE"
    UNKNOWN = "Unknown"
    STATUS_lower = "status"
    SUCCESS_lower = "success"
    IGNORE_lower = "ignore"
    ERROR_lower = "error"
    MESSAGE = "message"
    ERROR_MESSAGE = "error_message"

    LOT_lower = "lot"
    ADJUSTMENTS = "ADJUSTMENTS"

    MFR_MATCH = "MFR_MATCH"
    PART_NUM_LENGTH = "PART_NUM_LENGTH"
    PART_NUM_MATCH = "PART_NUM_MATCH"
    DESCRIPTION_SIMILARITY = "DESCRIPTION_SIMILARITY"
    DATA_SOURCES = "DATA_SOURCES"

    # confidences.yaml keys start
    WEIGHTAGE = "WEIGHTAGE"
    EMPTY_STRING = ""
    NULL_STRING = "NULL"
    NOT_DETECTED = "Not Detected"
    NO_MATCH = "No Match"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"
    EXACT = "Exact"
    LIKELY = "Likely"
    POSSIBLE = "Possible"
    MISMATCH = "Mismatch"
    STRONG = "Strong"
    PARTIAL = "Partial"
    # confidences.yaml keys end

    ERROR = "ERROR"
    UNDEFINED = "UNDEFINED"
    COMPLETED_lower = "completed"
    FAILED_lower = "failed"
    INCOMPLETE_lower = "incomplete"
    MISSING_SOURCE = "Missing Source URL"

    MAX_WORKERS = "MAX_WORKERS"
    BUFFER_SIZE = "BUFFER_SIZE"
    REFILL_THRESH = "REFILL_THRESH"
    POLL_INTERVAL = "POLL_INTERVAL"
    SHOULD_PAUSE_PROCESSING = "SHOULD_PAUSE_PROCESSING"

    RENTAL = "RENTAL"
    IS_RENTAL = "is_rental"
    Y = "Y"
    N = "N"

    CASES = "CASES"
    CASE = "CASE"

    STORAGE_ACCOUNT = "STORAGE_ACCOUNT"
    BLOB_CONTAINERS = "BLOB_CONTAINERS"
    ACCOUNT = "ACCOUNT"
    API_DEPLOYMENT = "API_DEPLOYMENT"

    ITM_AI_LDSC = "ITM_AI_LDSC"


class LLMResponseMethods:
    JSON_MODE = "json_mode"
    JSON_SCHEMA = "json_schema"
    FUNCTION_CALLING = "function_calling"


class Logs:
    START_TIME = "start_time"
    END_TIME = "end_time"

    ID = "id"
    API_REQUEST_UUID = "api_request_uuid"
    PATH = "path"
    APP_VERSION = "app_version"
    REQUEST = "request"
    REQUEST_DETAILS = "request_details"
    REQUEST_ID = "request_id"
    RECEIVED_AT = "received_at"
    CREATED_AT = "created_at"
    STARTED_AT = "started_at"
    COMPLETED_AT = "completed_at"
    TOTAL_IDS_IN_REQUEST = "total_ids_in_req"
    CLASSIFY = "classify"
    SYS_NAME = "sys_name"

    STATUS = "status"

    CATEGORY = "category"
    AKS_PRT_NUM = "aks_part_number"
    UPC = "upc"
    MFR_NAME = "manufacturer_name"
    UNCLN_MFR_NAME = "unclean_manufacturer_name"
    PRT_NUM = "part_number"
    UNSPSC = "unspsc"
    ITEM_SOURCE_NAME = "item_source_name"
    DESC_SOURCE_NAME = "desc_source_name"

    DESCRIPTION = "description"
    IVCE_LINE_STATUS = "invoice_line_status"
    WEB_SEARCH_URL = "web_search_url"
    SEARCH_THREAD_ID = "search_thread"
    RANK_THREAD_ID = "rank_thread"
    IS_MFR_CLEAN = "is_mfr_clean_flag"
    IS_MFR_DIRECT = "is_mfr_direct_flag"
    IS_VERIFIED = "is_verified_flag"
    CATEGORY_ID = "category_id"
    PRNT_DTL_ID = "PRNT_IVCE_DTL_ID"
    IVCE_DTL_UID = "IVCE_DTL_UID"

    CONFIDENCE = "confidence_score"
    CONF_CATEGORY = "category_conf"
    CONF_MFR_NAME = "manufacturer_name_conf"
    CONF_PRT_NUM = "part_number_conf"
    CONF_UNSPSC = "unspsc_conf"

    CONF_STAGE_MFR_NAME = "manufacturer_name_conf_stage"
    CONF_STAGE_PRT_NUM = "part_number_conf_stage"
    CONF_STAGE_UNSPSC = "unspsc_conf_stage"

    STAGE_CATEGORY = "category_stage"
    STAGE_MFR_NAME = "manufacturer_name_stage"
    STAGE_PRT_NUM = "part_number_stage"
    STAGE_UNSPSC = "unspsc_stage"
    STAGE_AKS_PRT_NUM = "aks_part_number_stage"
    STAGE_UPC = "upc_stage"
    STAGE_DESCRIPTION = "description_stage"

    DETAILS = "details"
    MESSAGE = "message"

    MAPPING = {
        PRT_NUM: [CONF_PRT_NUM, STAGE_PRT_NUM, CONF_STAGE_PRT_NUM],
        MFR_NAME: [CONF_MFR_NAME, STAGE_MFR_NAME, CONF_STAGE_MFR_NAME],
        UNSPSC: [CONF_UNSPSC, STAGE_UNSPSC, CONF_STAGE_UNSPSC],
    }

    @classmethod
    def get_conf_key(cls, field_key):
        return cls.MAPPING[field_key][0]

    @classmethod
    def get_stage_key(cls, field_key):
        return cls.MAPPING[field_key][1]

    @classmethod
    def get_conf_stage_key(cls, field_key):
        return cls.MAPPING[field_key][2]


class AzureAppConfig:
    MAIN_APP_RESTART_TIME = "MAIN:APP_RESTART_TIME"
    MAIN_MIN_DESCRIPTION_LENGTH = "MAIN:MIN_DESCRIPTION_LENGTH"
    MAIN_WORKER_SETTINGS = "MAIN:WORKER_SETTINGS"
    MAIN_AGENT_RETRY_SETTINGS = "MAIN:AGENT_RETRY_SETTINGS"
    MAIN_INDEXER_SETTINGS = "MAIN:INDEXER_SETTINGS"
    SQL_WRITER_SETTINGS = "MAIN:SQL_WRITER_SETTINGS"
    MAIN_USE_PRIORITIZED_REQUEST_IDS = "MAIN:USE_PRIORITIZED_REQUEST_IDS"
    MAIN_PRIORITIZED_REQUEST_IDS = "MAIN:PRIORITIZED_REQUEST_IDS"
    # Finetuned LLM
    AOAI_FINETUNED_LLM_API_DEPLOYMENT = "OPENAI_DEPLOYMENT:FINETUNED_LLM:API_DEPLOYMENT"
    # Classifiers
    LS_DESCRIPTION_PROJECT_NAME = "LANGUAGE_STUDIO:DESCRIPTION:PROJECT_NAME"
    LS_LOT_PROJECT_NAME = "LANGUAGE_STUDIO:LOT:PROJECT_NAME"
    LS_RENTAL_PROJECT_NAME = "LANGUAGE_STUDIO:RENTAL:PROJECT_NAME"
    LS_DESCRIPTION_DEPLOYMENT_NAME = "LANGUAGE_STUDIO:DESCRIPTION:DEPLOYMENT_NAME"
    LS_LOT_DEPLOYMENT_NAME = "LANGUAGE_STUDIO:LOT:DEPLOYMENT_NAME"
    LS_RENTAL_DEPLOYMENT_NAME = "LANGUAGE_STUDIO:RENTAL:DEPLOYMENT_NAME"

    # WEBJOBS --------
    # Finetuned LLM Webjob
    IS_LLM_FINE_TUNING_IN_PROGRESS = "WEBJOB:FINETUNE_LLM:FT_PROGRESS"
    # Finetuned LLM Webjob ID
    FINETUNE_JOB_ID = "WEBJOB:FINETUNE_LLM:FT_LLM_JOB_ID"


class Fields:
    PART_NUMBER = "PART_NUMBER"
    MANUFACTURER_NAME = "MANUFACTURER_NAME"
    UNSPSC_CODE = "UNSPSC_CODE"


class TrainingDataVersions:
    NEW = "NEW"
    FT_PROGRESS = "FT_PROGRESS"


class SpecialCases:
    CASE_0 = "CASE_0"
    CASE_1 = "CASE_1"
    CASE_2 = "CASE_2"


# Class to define statuses for Cosmos DB logs
class CosmosLogStatus:
    """
    Represents various statuses for Cosmos DB logs.

    Attributes:
    -----------
    DONE_lower : str
        Status indicating the process is completed.
    PROCESSING_lower : str
        Status indicating the process is ongoing.
    ERROR_lower : str
        Status indicating an error occurred.
    QUEUED_lower : str
        Status indicating the process is queued for processing.
    PENDING_lower : str
        Status indicating the process is pending.
    """

    DONE_lower = "done"
    PROCESSING_lower = "processing"
    ERROR_lower = "error"
    QUEUED_lower = "queued"
    PENDING_lower = "pending"


class StageNames:
    """
    Represents the names of different stages in the processing pipeline.

    Attributes:
    -----------
    RPA_EXTRACTION : str
        Previous stage to AI
    CLASSIFICATION : str
        Stage for classification tasks.
    SEMANTIC_SEARCH : str
        Stage for semantic search tasks.
    COMPLETE_MATCH : str
        Stage for complete match tasks.
    FINETUNED_LLM : str
        Stage for fine-tuned LLM tasks.
    EXTRACTION_WITH_LLM_AND_WEBSEARCH : str
        Stage for LLM web-search tasks.
    """

    RPA_PROCESS = "RPA_PROCESS"
    CLASSIFICATION = "CLASSIFICATION"
    COMPLETE_MATCH = "COMPLETE_MATCH"
    CONTEXT_VALIDATOR = "CONTEXT_VALIDATOR"
    SEMANTIC_SEARCH = "SEMANTIC_SEARCH"
    FINETUNED_LLM = "FINETUNED_LLM"
    EXTRACTION_WITH_LLM_AND_WEBSEARCH = "EXTRACTION_WITH_LLM_AND_WEBSEARCH"


class SubStageNames:
    """
    Represents the names of sub-stages within the main stages.

    Attributes:
    -----------
    CLASSIFIERS : str
        Sub-stage for classifiers.
    FINE_TUNED_LLM : str
        Sub-stage for fine-tuned LLM.
    AZURE_AI_AGENT_WITH_BING_SEARCH : str
        Sub-stage for LLM web-search.
    """

    PO_MAPPING = "PO_MAPPING"
    DESCRIPTION_CLASSIFIER = "DESCRIPTION_CLASSIFIER"
    LOT_CLASSIFIER = "LOT_CLASSIFIER"
    RENTAL_CLASSIFIER = "RENTAL_CLASSIFIER"
    COMPLETE_MATCH = "COMPLETE_MATCH"
    CONTEXT_VALIDATOR = "CONTEXT_VALIDATOR"
    SEMANTIC_SEARCH = "SEMANTIC_SEARCH"
    FINETUNED_LLM = "FINETUNED_LLM"
    AZURE_AI_AGENT_WITH_BING_SEARCH = "AZURE_AI_AGENT_WITH_BING_SEARCH"


# Class to define categories for descriptions
class DescriptionCategories:
    """
    Represents categories for descriptions.

    Attributes:
    -----------
    BAD | UNCLASSIFIED : str
        Category for bad descriptions.
    TAX : str
        Category for tax-related descriptions.
    FEE : str
        Category for fee-related descriptions.
    LABOR : str
        Category for labor-related descriptions.
    DISCOUNTS : str
        Category for discounts/promotions/rebates related descriptions.
    FREIGHT : str
        Category for freight-related descriptions.
    MATERIAL : str
        Category for material-related descriptions.
    AP_ADJUSTMENT : str
        Category for adjustments-related descriptions.
    LOT : str
        Sub-category of material-related descriptions representing lot of items together.
    RENTAL : str
        Category to identify an item as rental item.
    """

    BAD = "BAD"
    TAX = "TAX"
    FEE = "FEE"
    LABOR = "LABOR"
    DISCOUNTS = "DISCOUNTS"
    FREIGHT = "FREIGHT"
    MATERIAL = "MATERIAL"
    AP_ADJUSTMENT = "AP ADJUSTMENT"
    UNCLASSIFIED = "UNCLASSIFIED"
    LOT = "LOT"
    NOT_LOT = "NOT_LOT"
    GENERIC = "GENERIC"
    RENTAL = "RENTAL"
    NON_RENTAL = "NON_RENTAL"


# Class to specify environment types
class Environments:
    """
    Represents the types of environments.

    Attributes:
    -----------
    LOCAL : str
        Local environment.
    WEB : str
        Web environment.  (dev and prod)
    """

    LOCAL = "local"
    WEB = "web"
    DEV = "dev"
    PROD = "prod"


class DataTypes(str, Enum):
    STRING = "str"
    BOOLEAN = "bool"
    INTEGER = "int"
    FLOAT = "float"
    JSON = "json"


# Class to store file paths for local configuration files
class LocalFiles:
    """
    Represents file paths for local configuration files.

    Attributes:
    -----------
    CONFIG_FILE : str
        Path to the configuration file.
    THRESHOLD_FILE : str
        Path to the threshold file.
    CONFIDENCES_FILE : str
        Path to the confidences file.
    SPECIAL_CASE_FILE : str
        Path to the special case file.
    """

    CONFIG_FILE = "config.yaml"
    CONFIDENCES_FILE = "confidences.yaml"
    THRESHOLDS_FILE = "thresholds.yaml"
    SPECIAL_CASES_FILE = "special_cases.yaml"
    JOB_CONFIG_FILE = "job_config.yaml"

    @classmethod
    def prepend_app_root(cls, app_root: str) -> None:
        """
        Prepend `app_root` to every class-level string attribute
        that isn't already an absolute path.
        """
        for name, val in vars(cls).items():
            # only target uppercase attributes that are strings
            if name.isupper() and isinstance(val, str) and not os.path.isabs(val):
                setattr(cls, name, os.path.join(app_root, val))


# Class to define logging levels
class LogLevels:
    """
    Represents logging levels.

    Attributes:
    -----------
    DEBUG : str
        Debug level logging.
    INFO : str
        Info level logging.
    WARNING : str
        Warning level logging.
    ERROR : str
        Error level logging.
    CRITICAL : str
        Critical level logging.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Class to specify database names, schemas, and table names
class DatabaseObjects:
    """
    Represents database objects such as names, schemas, and table names.

    Attributes:
    -----------
    CDB_DB : str
        Name of Cosmos DB
    CDB_CONTAINER_* : str
        Name of the cosmos db container
    SDP_DB_* : str
        Name of the database.
    SCHEMA_* : str
        Name of the schema.
    TBL_* : str
        Name of the table.
    """

    CDB_DB = "spend_report_dev"

    CDB_CONTAINER_AI_API_REQUESTS = "AI_API_REQUESTS"
    CDB_CONTAINER_AI_JOBS = "AI_JOBS"
    CDB_CONTAINER_AI_PROCESS_LOGS = "AI_PROCESS_LOGS"
    CDB_CONTAINER_MFR_DATA_CAHCE = "MFR_DATA_CAHCE"

    SDP_DB_SDPDWH = "SDPDWH"
    SDP_DB_AKSCAP = "AKSCAP"

    SCHEMA_RPAO = "RPAO"
    SCHEMA_AIML = "AIML"

    # TBL_INVC_DTL = 'RPAO.IVCE_DTL_VW'
    # TBL_IVCE_HDR = 'RPAO.IVCE_HDR_VW'
    # TBL_IVCE_TRKG_MSTR = 'RPAO.IVCE_TRKG_MSTR_VW'
    # TBL_MFR_DTL = 'RPAO.MFR_DTL_VW'
    # TBL_IVCE_XCTN_LLM_TRNL_MFR_REF = 'AIML.IVCE_XCTN_LLM_TRNL_MFR_REF_VW' # items
    # TBL_IVCE_XCTN_LLM_TRNL_PRDT_REF = 'AIML.IVCE_XCTN_LLM_TRNL_PRDT_REF_VW' # descriptions

    TBL_INVC_DTL = "RPAO.IVCE_DTL"
    TBL_IVCE_HDR = "RPAO.IVCE_HDR"
    TBL_IVCE_TRKG_MSTR = "RPAO.IVCE_TRKG_MSTR"
    TBL_MFR_DTL = "RPAO.MFR_DTL"
    TBL_IVCE_XCTN_CLSFR_CTGY_DTL = "AIML.IVCE_XCTN_CLSFR_CTGY_DTL"  # classifier mapping
    TBL_IVCE_XCTN_LLM_TRNL_MFR_REF = "AIML.IVCE_XCTN_LLM_TRNL_MFR_REF"  # items
    TBL_IVCE_XCTN_LLM_TRNL_PRDT_REF = "AIML.IVCE_XCTN_LLM_TRNL_PRDT_REF"  # descriptions
    TBL_IVCE_XCTN_CLSFR_TRNL_DESC_REF = "AIML.IVCE_XCTN_CLSFR_TRNL_DESC_REF"


class DatadogServices:
    """
    Service names to identify logs in Datadog
    """

    DEV_API = "ai-procurement-platform-dev_api"
    PROD_API = "ai-procurement-platform-prod_api"


class DataStates:
    """
    Represents various states of data processing.

    Attributes:
    -----------
    DS1 : str
        State 1 of data processing.
    AI_ERROR : str
        State indicating an AI error.
    AI_APPROVED : str
        State indicating AI approval.
    """

    RPA_USER = "RPA_USER"
    AI_USER = "AI_USER"
    RC_RPA = "RC-RPA"
    AI_API_ERROR = "AI-API-ERROR"
    AI = "AI"
    AI_ERROR = "AI_ERROR"
    RC_AI = "RC-AI"
    DS1 = "DS1"
    DS2 = "DS2"
    BR = "BR"
    RC_MANUAL = "RC-Manual"


# Enum to enumerate types of manufacturer name matches
class MfrNameMatchType(Enum):
    """
    Enumerates types of manufacturer name matches.

    Attributes:
    -----------
    SINGLE_MATCH : str
        Indicates a single match for manufacturer name.
    MULTIPLE_MATCHES : str
        Indicates multiple matches for manufacturer name.
    NO_VALID_MATCHES : str
        Indicates no valid matches for manufacturer name.
    NO_MANUFACTURERS_FOUND : str
        Indicates no matches for manufacturer name were found.
    """

    SINGLE_MATCH = "SINGLE_MATCH"
    MULTIPLE_MATCHES_ONE_VALID = "MULTIPLE_MATCHES_ONE_VALID"
    NO_VALID_MATCHES = "NO_VALID_MATCHES"
    NO_MANUFACTURERS_FOUND = "NO_MANUFACTURERS_FOUND"

    def __str__(self):
        return self.value


class MfrRelationshipType(Enum):
    """
    Enumerates types of relationships between two manufacturers based on their CleanNames.

    Attributes:
    -----------
    DIRECT : str
        Indicates the manufacturers have the same CleanName.
    PARENT : str
        Indicates the first manufacturer is the parent of the second.
    CHILD : str
        Indicates the first manufacturer is a child of the second.
    SIBLING : str
        Indicates the manufacturers share the same parent but are different.
    NOT_EQUIVALENT : str
        Indicates no recognized relationship exists between the manufacturers.
    """

    DIRECT = "DIRECT"
    PARENT = "PARENT"
    CHILD = "CHILD"
    SIBLING = "SIBLING"
    NOT_EQUIVALENT = "NOT_EQUIVALENT"

    def __str__(self):
        return self.value

    def __bool__(self):
        # Only NOT_EQUIVALENT evaluates to False, all others evaluate to True
        return self is not MfrRelationshipType.NOT_EQUIVALENT


class APIPaths:
    """
    It defines the API paths available in the application.
    """

    GET_WORKER_STATUS = "/v1/get_worker_status"
    PROCESS_ALL_INVOICES = "/v1/process_all_invoices"
    PROCESS_INVOICE = "/v1/process_invoice"
    PROCESS_INVOICE_DETAILS = "/v1/process_invoice_details"
    RELOAD_APP_CONFIG = "/v1/reload_app_config"
    GET_COSMOS_QUEUE_STATUS = "/v1/get_cosmos_queue_status"
    GET_CONFIG_SNAPSHOT = "/v1/get_config_snapshot"
