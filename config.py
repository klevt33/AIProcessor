#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
## Overview
The provided Python code defines a Config class that manages application settings, environment variables, and secrets.
It is designed to load configuration data from local YAML files, environment-specific variables, and Azure Key Vault secrets.
The class is particularly useful for applications that need to manage configurations across different environments
(e.g., local, development, production).
"""

import json
import os
from typing import Any

from azure.appconfiguration import ConfigurationSetting
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from az_clients import AzureClients
from constants import AzureAppConfig, Constants, DataTypes, Environments, LocalFiles, LogLevels
from utils import load_env_config, load_yaml


class Config:
    """
    A configuration class to manage application settings, environment variables, and secrets.

    This class initializes configuration settings from local files, environment variables, and Azure Key Vault secrets.

    Methods:
    --------
    __init__():
        Initializes the configuration by setting up local config, environment variables, and secrets.

    setup_config():
        Loads application configuration from a local YAML file.

    setup_env_variables():
        Loads environment-specific variables from a YAML file.

    setup_secrets():
        Retrieves secrets from Azure Key Vault for non-local environments.
    """

    def __init__(self, app_root=Constants.EMPTY_STRING) -> None:
        """
        Initializes the Config class by setting up configuration, environment variables, and secrets.
        """
        self._locked = False
        self.app_root = app_root
        self.setup_config()
        self.setup_env_variables()
        self.setup_secrets()
        if self.environment == Environments.LOCAL:
            self.setup_temp_env()

        # call after loading all env vars
        self.setup_az_clinets()
        self.refresh_from_azure_app_config()

        self._locked = True

    def __setattr__(self, key, value):
        if hasattr(self, "_locked") and key != "_locked" and self._locked:
            raise AttributeError("Config object is read-only!")
        super().__setattr__(key, value)

    def setup_config(self):
        """
        Loads application configuration from a local YAML file.

        This method reads the configuration file specified in `LocalFiles.CONFIG_FILE` and sets attributes
        """
        config_yaml = load_yaml(path=LocalFiles.CONFIG_FILE, app_root=self.app_root)
        self.app_version = config_yaml["APP_VERSION"]
        self.allowed_clients = config_yaml["ALLOWED_CLIENTS"]
        self.exact_match_source = config_yaml["EXACT_MATCH_SOURCE"]
        self.predefined_sites_filepath = config_yaml["PREDEFINED_SITES"]
        sql_writer_settings = config_yaml.get("SQL_WRITER_SETTINGS", {})
        self.SQL_WRITER_BATCH_SIZE = sql_writer_settings.get("batch_size", 25)
        self.SQL_WRITER_POLL_INTERVAL = sql_writer_settings.get("poll_interval_seconds", 10)
        self.SQL_WRITER_MAX_WORKERS = sql_writer_settings.get("max_workers", 1)
        # self.min_description_len = config_yaml["MIN_DESCRIPTION_LENGTH"]
        # self.max_workers = config_yaml["MAX_WORKERS"]
        # self.job_timeout_sec = config_yaml['JOB_TIMEOUT_SECONDS']
        # self.agent_retry_settings = config_yaml.get("AGENT_RETRY_SETTINGS", {})
        # self.indexer_settings = config_yaml.get("INDEXER_SETTINGS", {})
        self.environment = (
            Environments.LOCAL if config_yaml["ENVIRONMENT"] == Environments.LOCAL else os.getenv("WEB_APP_ENV", "dev").lower()
        )
        self.log_level = (
            LogLevels.DEBUG
            if config_yaml["ENVIRONMENT"] == Environments.LOCAL
            else os.getenv("LOG_LEVEL", LogLevels.INFO).upper()
        )

    def setup_env_variables(self):
        """
        Loads environment-specific variables from a YAML file.

        This method reads the environment configuration file based on the current environment and sets attributes.
        """
        env_yaml = load_env_config(self.environment, app_root=self.app_root)

        self.SUBSCRIPTION_ID = env_yaml["SUBSCRIPTION_ID"]

        # Key-Valut configs
        self.KV_RG_URL = env_yaml[Constants.KEY_VAULT]["RG_URL"]
        self.KV_AI_URL = env_yaml[Constants.KEY_VAULT]["AI_URL"]

        # API Webapp configs
        self.WA_TENANT_ID = env_yaml[Constants.SPENDREPORT_WA]["TENANT_ID"]
        self.WA_AUDIENCE = env_yaml[Constants.SPENDREPORT_WA]["AUDIENCE"]
        self.WA_ISSUER_V1 = env_yaml[Constants.SPENDREPORT_WA]["ISSUER_V1"]
        self.WA_ISSUER_V2 = env_yaml[Constants.SPENDREPORT_WA]["ISSUER_V2"]
        self.WA_JWKS_URL = env_yaml[Constants.SPENDREPORT_WA]["JWKS_URL"]
        self.WA_KUDU_USER_NAME = env_yaml[Constants.SPENDREPORT_WA]["KUDU_USER_NAME"]
        self.WA_KUDU_PASSWORD = env_yaml[Constants.SPENDREPORT_WA]["KUDU_PASSWORD"]
        self.WA_KUDU_BASE_URL = env_yaml[Constants.SPENDREPORT_WA]["KUDU_BASE_URL"]
        self.WA_APP_NAME = env_yaml[Constants.SPENDREPORT_WA]["APP_NAME"]
        self.WA_RG = env_yaml[Constants.SPENDREPORT_WA]["RESOURCE_GROUP"]
        self.WA_API_ACCESS_SCOPE = env_yaml[Constants.SPENDREPORT_WA]["API_ACCESS_SCOPE"]

        if self.environment == Environments.LOCAL:
            self.APP_INSIGHTS_CONN_STRING = env_yaml[Constants.SPENDREPORT_WA]["APPLICATIONINSIGHTS_CONNECTION_STRING"]
        else:
            self.APP_INSIGHTS_CONN_STRING = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", None)

        self.WA_ISSUER_V1 = self.WA_ISSUER_V1.replace("TENANT_ID", self.WA_TENANT_ID)
        self.WA_ISSUER_V2 = self.WA_ISSUER_V2.replace("TENANT_ID", self.WA_TENANT_ID)
        self.WA_JWKS_URL = self.WA_JWKS_URL.replace("TENANT_ID", self.WA_TENANT_ID)

        # Language Studio
        self.LS_API_ENDPOINT_URL = env_yaml[Constants.LANGUAGE_STUDIO]["API_ENDPOINT_URL"]
        self.LS_API_KEY = env_yaml[Constants.LANGUAGE_STUDIO]["API_KEY"]

        # OPEN AI configs
        openai_yaml = env_yaml[Constants.OPENAI_DEPLOYMENT]
        self.AOAI_RG = openai_yaml["RESOURCE_GROUP"]

        # # Base LLM
        self.AOAI_BASE_LLM_API_TYPE = openai_yaml[Constants.BASE_LLM]["API_TYPE"]
        self.AOAI_BASE_LLM_API_VERSION = openai_yaml[Constants.BASE_LLM]["API_VERSION"]
        self.AOAI_BASE_LLM_API_DEPLOYMENT = openai_yaml[Constants.BASE_LLM]["API_DEPLOYMENT"]
        self.AOAI_BASE_LLM_OPENAI_API_KEY = openai_yaml[Constants.BASE_LLM]["API_KEY"]
        self.AOAI_BASE_LLM_API_BASE_URL = openai_yaml[Constants.BASE_LLM]["API_BASE_URL"]
        self.AOAI_BASE_LLM_API_ENDPOINT_URL = openai_yaml[Constants.BASE_LLM]["API_ENDPOINT_URL"]
        self.AOAI_BASE_LLM_API_ENDPOINT_URL = self.AOAI_BASE_LLM_API_ENDPOINT_URL.replace(
            Constants.API_DEPLOYMENT, self.AOAI_BASE_LLM_API_DEPLOYMENT
        )

        # # Finetuned LLM
        self.AOAI_FINETUNED_LLM_RESOURCE_NAME = openai_yaml[Constants.FINETUNED_LLM]["RESOURCE_NAME"]
        self.AOAI_FINETUNED_LLM_DEPLOYMENT_API_VERSION = openai_yaml[Constants.FINETUNED_LLM]["DEPLOYMENT_API_VERSION"]
        self.AOAI_FINETUNED_LLM_API_TYPE = openai_yaml[Constants.FINETUNED_LLM]["API_TYPE"]
        self.AOAI_FINETUNED_LLM_API_VERSION = openai_yaml[Constants.FINETUNED_LLM]["API_VERSION"]

        self.AOAI_FINETUNED_LLM_OPENAI_API_KEY = openai_yaml[Constants.FINETUNED_LLM]["API_KEY"]
        self.AOAI_FINETUNED_LLM_API_BASE_URL = openai_yaml[Constants.FINETUNED_LLM]["API_BASE_URL"]
        self.AOAI_FINETUNED_LLM_API_ENDPOINT_URL = openai_yaml[Constants.FINETUNED_LLM]["API_ENDPOINT_URL"]

        context_validator_yaml = openai_yaml[Constants.CONTEXT_VALIDATOR]
        self.AOAI_CONTEXT_VALIDATOR_API_TYPE = context_validator_yaml["API_TYPE"]
        self.AOAI_CONTEXT_VALIDATOR_API_VERSION = context_validator_yaml["API_VERSION"]
        self.AOAI_CONTEXT_VALIDATOR_API_DEPLOYMENT = context_validator_yaml["API_DEPLOYMENT"]
        self.AOAI_CONTEXT_VALIDATOR_API_BASE_URL = context_validator_yaml["API_BASE_URL"]
        self.AOAI_CONTEXT_VALIDATOR_API_ENDPOINT_URL = context_validator_yaml["API_ENDPOINT_URL"]
        # Optional: Replace the placeholder in the endpoint URL if needed
        # if self.AOAI_CONTEXT_VALIDATOR_API_ENDPOINT_URL and self.AOAI_CONTEXT_VALIDATOR_API_DEPLOYMENT:
        #     self.AOAI_CONTEXT_VALIDATOR_API_ENDPOINT_URL = self.AOAI_CONTEXT_VALIDATOR_API_ENDPOINT_URL.replace(
        #         Constants.API_DEPLOYMENT, self.AOAI_CONTEXT_VALIDATOR_API_DEPLOYMENT
        #     )

        # Embedding Model Configuration
        self.AOAI_EMBEDDING_API_TYPE = openai_yaml[Constants.EMBEDDING_MODEL]["API_TYPE"]
        self.AOAI_EMBEDDING_API_VERSION = openai_yaml[Constants.EMBEDDING_MODEL]["API_VERSION"]
        self.AOAI_EMBEDDING_API_DEPLOYMENT = openai_yaml[Constants.EMBEDDING_MODEL]["API_DEPLOYMENT"]
        self.AOAI_EMBEDDING_API_MODEL = openai_yaml[Constants.EMBEDDING_MODEL]["API_MODEL"]
        self.AOAI_EMBEDDING_OPENAI_API_KEY = openai_yaml[Constants.EMBEDDING_MODEL]["API_KEY"]
        self.AOAI_EMBEDDING_API_BASE_URL = openai_yaml[Constants.EMBEDDING_MODEL]["API_BASE_URL"]
        self.AOAI_EMBEDDING_API_ENDPOINT_URL = openai_yaml[Constants.EMBEDDING_MODEL]["API_ENDPOINT_URL"]
        self.AOAI_EMBEDDING_DIMENSIONS = openai_yaml[Constants.EMBEDDING_MODEL]["EMBEDDING_DIMENSIONS"]
        self.AOAI_EMBEDDING_API_ENDPOINT_URL = self.AOAI_EMBEDDING_API_ENDPOINT_URL.replace(
            Constants.API_DEPLOYMENT, self.AOAI_EMBEDDING_API_DEPLOYMENT
        )

        # Azure AI Search Configuration
        ai_search_yaml = env_yaml[Constants.AZURE_AI_SEARCH]
        self.AZ_SEARCH_SERVICE_NAME = ai_search_yaml["SERVICE_NAME"]
        self.AZ_SEARCH_API_ENDPOINT_URL = ai_search_yaml["API_ENDPOINT_URL"]
        self.AZ_SEARCH_API_KEY = ai_search_yaml["API_KEY"]
        self.AZ_SEARCH_INDEX_NAME = ai_search_yaml["INDEX_NAME"]

        ai_agent_yaml = env_yaml[Constants.AI_AGENT]
        self.AZ_AGENT_PROJECT_CONNECTION_STRING = ai_agent_yaml["PROJECT_CONNECTION_STRING"]
        self.AZ_AGENT_API_TYPE = ai_agent_yaml["API_TYPE"]
        self.AZ_AGENT_API_BASE_URL = ai_agent_yaml["API_BASE_URL"]

        self.AZ_AGENT_API_VERSION = ai_agent_yaml[Constants.AZURE_AI_AGENT]["API_VERSION"]
        self.AZ_AGENT_API_DEPLOYMENT = ai_agent_yaml[Constants.AZURE_AI_AGENT]["API_DEPLOYMENT"]
        self.AZ_AGENT_OPENAI_API_KEY = ai_agent_yaml[Constants.AZURE_AI_AGENT]["API_KEY"]
        self.AZ_AGENT_API_ENDPOINT_URL = ai_agent_yaml[Constants.AZURE_AI_AGENT]["API_ENDPOINT_URL"]
        self.AZ_AGENT_AGENT_DEPLOYMENT = ai_agent_yaml[Constants.AZURE_AI_AGENT]["AGENT_DEPLOYMENT"]
        self.AZ_AGENT_API_ENDPOINT_URL = self.AZ_AGENT_API_ENDPOINT_URL.replace(
            Constants.API_DEPLOYMENT, self.AZ_AGENT_API_DEPLOYMENT
        )

        self.AZ_AGENT_GBS_BING_CONNECTION_NAME = ai_agent_yaml[Constants.AZURE_AI_AGENT_WITH_BING_SEARCH]["BING_CONNECTION_NAME"]
        self.AZ_AGENT_GBS_API_VERSION = ai_agent_yaml[Constants.AZURE_AI_AGENT_WITH_BING_SEARCH]["API_VERSION"]
        self.AZ_AGENT_GBS_API_DEPLOYMENT = ai_agent_yaml[Constants.AZURE_AI_AGENT_WITH_BING_SEARCH]["API_DEPLOYMENT"]
        self.AZ_AGENT_GBS_OPENAI_API_KEY = ai_agent_yaml[Constants.AZURE_AI_AGENT_WITH_BING_SEARCH]["API_KEY"]
        self.AZ_AGENT_GBS_API_ENDPOINT_URL = ai_agent_yaml[Constants.AZURE_AI_AGENT_WITH_BING_SEARCH]["API_ENDPOINT_URL"]
        self.AZ_AGENT_GBS_AGENT_DEPLOYMENT = ai_agent_yaml[Constants.AZURE_AI_AGENT_WITH_BING_SEARCH]["AGENT_DEPLOYMENT"]
        self.AZ_AGENT_GBS_API_ENDPOINT_URL = self.AZ_AGENT_GBS_API_ENDPOINT_URL.replace(
            Constants.API_DEPLOYMENT, self.AZ_AGENT_GBS_API_DEPLOYMENT
        )

        # CSOMOS DB configs
        self.COSMOS_DB_URI = env_yaml[Constants.COSMOS_DB]["BASE_URI"]
        self.COSMOS_DB_PRIMARY_KEY = env_yaml[Constants.COSMOS_DB]["PRIMARY_KEY"]
        self.COSMOS_DB_DATABASE_ID = env_yaml[Constants.COSMOS_DB]["DATABASE_ID"]
        self.COSMOS_DB_CONTAINER_ID_MAP = env_yaml[Constants.COSMOS_DB]["CONTAINER_ID"]

        # SDP configs
        self.SDP_DRIVER = env_yaml[Constants.SDP]["DRIVER"]
        sdp_server = env_yaml[Constants.SDP]["PRIVATE_SERVER_1433"]
        # sdp_server = env_yaml[Constants.SDP]['PRIVATE_SERVER_3342']
        self.SDP_SERVER = sdp_server if self.environment == Environments.LOCAL else os.getenv("SDP_SERVER", sdp_server)
        self.SDP_DATABASE = env_yaml[Constants.SDP]["DATABASE"]
        self.SDP_UID = env_yaml[Constants.SDP]["UID"]
        self.SDP_PWD = env_yaml[Constants.SDP]["PWD"]

        # Azure Storage configuration
        az_st_acc_yaml = env_yaml[Constants.STORAGE_ACCOUNT]
        az_aksstai_st_acc_yaml = az_st_acc_yaml["AKSSTAI"]
        self.AZ_AKSSTAI_ST_ACC_BLOB_API_BASE_URL = az_aksstai_st_acc_yaml[Constants.BLOB_CONTAINERS]["API_BASE_URL"]
        self.AZ_AKSSTAI_FINE_TUNED_LLM_CONTAINER_NAME = az_aksstai_st_acc_yaml[Constants.BLOB_CONTAINERS][
            "FINE_TUNED_LLM_CONTAINER_NAME"
        ]

    def setup_secrets(self):
        """
        Retrieves secrets from Azure Key Vault for non-local environments.

        This method authenticates using `DefaultAzureCredential` and retrieves secrets from Azure Key Vault.
        """

        def get_kv_client(url):
            # Authenticate
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=url, credential=credential)
            return client

        rg_client = get_kv_client(self.KV_RG_URL)
        ai_client = get_kv_client(self.KV_AI_URL)

        # Kudu
        self.WA_KUDU_PASSWORD = rg_client.get_secret("SPEND-REPORT-WA-KUDU-PASSWORD").value

        # App config
        self.RG_APP_CONFIG_CONN_STRING = rg_client.get_secret("SPEND-REPORT-WA-AZURE-APP-CONFIG-CONNECTION-STRING").value
        self.AI_APP_CONFIG_CONN_STRING = ai_client.get_secret("SPEND-REPORT-WA-AZURE-APP-CONFIG-CONNECTION-STRING").value
        self.WA_ACCESS_CLIENT_SECRET = rg_client.get_secret("SPEND-REPORT-WA-CLIENT-SECRET").value

        # OpenAI
        self.AOAI_BASE_LLM_OPENAI_API_KEY = rg_client.get_secret("SPEND-REPORT-WA-OPENAI-DEPLOYMENT-BASE-LLM-API-KEY").value
        # Uses the same API key as base_llm
        self.AOAI_CONTEXT_VALIDATOR_API_KEY = self.AOAI_BASE_LLM_OPENAI_API_KEY
        self.AOAI_FINETUNED_LLM_OPENAI_API_KEY = rg_client.get_secret(
            "SPEND-REPORT-WA-OPENAI-DEPLOYMENT-FINETUNED-LLM-API-KEY"
        ).value
        self.AOAI_EMBEDDING_OPENAI_API_KEY = rg_client.get_secret(
            "SPEND-REPORT-WA-OPENAI-DEPLOYMENT-EMBEDDING-MODEL-API-KEY"
        ).value
        self.AZ_SEARCH_API_KEY = rg_client.get_secret("SPEND-REPORT-WA-OPENAI-DEPLOYMENT-AZURE-SEARCH-API-KEY").value

        # AI Agent
        self.AZ_AGENT_OPENAI_API_KEY = rg_client.get_secret("SPEND-REPORT-WA-AI-AGENT-AZURE-AI-AGENT-API-KEY").value
        self.AZ_AGENT_GBS_OPENAI_API_KEY = rg_client.get_secret(
            "SPEND-REPORT-WA-AI-AGENT-AZURE-AI-AGENT-WITH-BING-SEARCH-API-KEY"
        ).value

        # Language Studio Key
        self.LS_API_KEY = rg_client.get_secret("SPEND-REPORT-WA-LANGUAGE-STUDIO-API-KEY").value

        # Cosmos DB
        self.COSMOS_DB_PRIMARY_KEY = rg_client.get_secret("SPEND-REPORT-WA-COSMOSDB-PRIMARY-KEY").value

        # SDP
        self.SDP_PWD = rg_client.get_secret("SPEND-REPORT-WA-SDP-SVC-PASSWORD").value

    def setup_temp_env(self):
        """
        Loads temporary environment variables from a YAML file.

        This method reads and sets attributes required for isolation of local development servers.
        """
        env_yaml = load_env_config("temp", app_root=self.app_root)

        self.local_sys_name = env_yaml["sys_name"]

    def setup_az_clinets(self):
        self.azure_clients = AzureClients(config=self)

    def refresh_from_azure_app_config(self):
        """
        Refreshes application configuration values from Azure App Configuration
        based on the current environment (DEV or PROD).

        This method sets the appropriate label (environment) and updates instance variables
        with the latest configuration values from Azure App Configuration.
        """
        if not self._locked:
            self.app_config_label = (
                Environments.DEV
                if self.environment == Environments.LOCAL or self.environment == Environments.DEV
                else Environments.PROD
            )

        # Set locked to False for updating config object values
        self._locked = False

        self.app_config_label = Environments.DEV if self.environment != Environments.PROD else Environments.PROD

        # Fetch the settings object for the SQL Writer
        sql_writer_settings = self.get_azure_app_config_value(
            AzureAppConfig.SQL_WRITER_SETTINGS, self.app_config_label, value_type=DataTypes.JSON
        )

        # Parse the dictionary and set the individual config attributes
        self.SQL_WRITER_BATCH_SIZE = sql_writer_settings.get("batch_size", 25)
        self.SQL_WRITER_POLL_INTERVAL = sql_writer_settings.get("poll_interval_seconds", 10)
        self.SQL_WRITER_MAX_WORKERS = sql_writer_settings.get("max_workers", 1)

        # General config
        self.min_description_len = self.get_azure_app_config_value(
            AzureAppConfig.MAIN_MIN_DESCRIPTION_LENGTH, self.app_config_label, value_type=DataTypes.INTEGER
        )

        self.worker_settings = self.get_azure_app_config_value(
            AzureAppConfig.MAIN_WORKER_SETTINGS, self.app_config_label, value_type=DataTypes.JSON
        )

        """
        Following are the default `agent_retry_settings`
        ------------------------------------------------
        -   max_attempts: 6                       # Total attempts (1 initial + N-1 retries)
        -   default_wait_seconds: 30              # Fallback wait time in seconds
        -   attempt_timeout_seconds: 360          # Timeout per individual agent call attempt in seconds
        -   first_attempt_congestion_multiplier_sec: 15
        -   first_attempt_max_congestion_delay_sec: 90
        -   first_attempt_random_delay_min_sec: 1
        -   first_attempt_random_delay_max_sec: 5
        -   Additional settings for the new (centralized) agents.py implementation (currently not used)
        -   coordinator_concurrency: 5            # Number of concurrent calls to the coordinator
        -   client_reinit_on_auth_failure: true   # Reinitialize client on auth failure
        -   client_reinit_debounce_seconds: 300   # Debounce time for reinitializing client (5 minutes)
        -   agent_setup_max_retries: 3            # Max retries for agent setup
        -   agent_setup_retry_base_delay: 2       # Base delay for agent setup retries in seconds
        """
        self.agent_retry_settings = self.get_azure_app_config_value(
            AzureAppConfig.MAIN_AGENT_RETRY_SETTINGS, self.app_config_label, value_type=DataTypes.JSON
        )

        """
        Following are the default `indexer_settings`
        ------------------------------------------------
            # Max number of retries for Azure Search quota errors (initial attempt + retries)
        -   max_quota_retries: 5
            # Delay in minutes between quota error retries
        -   quota_retry_delay_minutes: 10
            # Number of records processed together in memory and sent to Azure Search
        -   batch_size: 400
            # Maximum number of records allowed in each processing queue (e.g., sql_data_queue)
        -   max_queue_size: 4000
            # Minimum records required from SQL to proceed (prevents index wipeout on SQL issues)
        -   min_sql_records: 200000
        """
        self.indexer_settings = self.get_azure_app_config_value(
            AzureAppConfig.MAIN_INDEXER_SETTINGS, self.app_config_label, value_type=DataTypes.JSON
        )

        self.use_priority_queue = self.get_azure_app_config_value(
            AzureAppConfig.MAIN_USE_PRIORITIZED_REQUEST_IDS, self.app_config_label, value_type=DataTypes.BOOLEAN
        )

        # Prioritized request ids to process
        self.prioritized_request_ids = self.get_azure_app_config_value(
            AzureAppConfig.MAIN_PRIORITIZED_REQUEST_IDS, self.app_config_label, value_type=DataTypes.JSON
        )

        # Finetuned LLM details
        self.AOAI_FINETUNED_LLM_API_DEPLOYMENT = self.get_azure_app_config_value(
            AzureAppConfig.AOAI_FINETUNED_LLM_API_DEPLOYMENT, self.app_config_label, client_type="AI"
        )
        self.AOAI_FINETUNED_LLM_API_ENDPOINT_URL = self.AOAI_FINETUNED_LLM_API_ENDPOINT_URL.replace(
            Constants.API_DEPLOYMENT, self.AOAI_FINETUNED_LLM_API_DEPLOYMENT
        )

        # Description classifier details
        self.LS_DESCRIPTION_DEPLOYMENT_NAME = self.get_azure_app_config_value(
            AzureAppConfig.LS_DESCRIPTION_DEPLOYMENT_NAME, self.app_config_label, client_type="AI"
        )
        self.LS_DESCRIPTION_PROJECT_NAME = self.get_azure_app_config_value(
            AzureAppConfig.LS_DESCRIPTION_PROJECT_NAME, self.app_config_label, client_type="AI"
        )

        # LOT classifier details
        self.LS_LOT_DEPLOYMENT_NAME = self.get_azure_app_config_value(
            AzureAppConfig.LS_LOT_DEPLOYMENT_NAME, self.app_config_label, client_type="AI"
        )
        self.LS_LOT_PROJECT_NAME = self.get_azure_app_config_value(
            AzureAppConfig.LS_LOT_PROJECT_NAME, self.app_config_label, client_type="AI"
        )

        # RENTAL classifier details
        self.LS_RENTAL_DEPLOYMENT_NAME = self.get_azure_app_config_value(
            AzureAppConfig.LS_RENTAL_DEPLOYMENT_NAME, self.app_config_label, client_type="AI"
        )
        self.LS_RENTAL_PROJECT_NAME = self.get_azure_app_config_value(
            AzureAppConfig.LS_RENTAL_PROJECT_NAME, self.app_config_label, client_type="AI"
        )

        # Set locked to True for restricting modification of config object values
        self._locked = True

    def _fetch_app_config_client(self, client_type: str):
        if client_type == "AI":
            app_config_client = self.azure_clients.ai_app_config_client
        else:
            app_config_client = self.azure_clients.rg_app_config_client
        return app_config_client

    def get_azure_app_config_value(self, key: str, label: str, value_type: DataTypes = DataTypes.STRING, client_type="RG") -> Any:
        """
        Retrieves and casts a configuration value from Azure App Configuration.

        Args:
            key (str): The key to retrieve.
            label (str): The environment label (e.g., 'Production').
            value_type (DataTypes): The expected return type.

        Returns:
            Any: The value cast to the specified type.

        Raises:
            ValueError: If conversion fails or type is unsupported.
        """
        app_config_client = self._fetch_app_config_client(client_type=client_type)

        try:
            setting = app_config_client.get_configuration_setting(key=key, label=label)
        except Exception as e:
            print(f"Error occurred in get_azure_app_config_value(). key: {key}, label:{label}, value_type:{value_type}. {str(e)}")

        raw_value = setting.value.strip()
        if value_type == DataTypes.BOOLEAN:
            return raw_value.lower() in ["true", "1", "yes"]
        elif value_type == DataTypes.INTEGER:
            return int(raw_value)
        elif value_type == DataTypes.FLOAT:
            return float(raw_value)
        elif value_type == DataTypes.STRING:
            return raw_value
        elif value_type == DataTypes.JSON:
            return json.loads(raw_value)
        else:
            raise ValueError(f"Unsupported data type: {value_type}")

    def set_azure_app_config_value(self, key: str, label: str, value, content_type="application/text", client_type="RG") -> None:
        """
        Sets or updates a configuration value in Azure App Configuration.

        Args:
            key (str): The configuration key to set or update.
            label (str): The label (environment-specific) to scope the key.
            value (Any): The value to assign to the configuration key.

        Returns:
            None
        """
        app_config_client = self._fetch_app_config_client(client_type=client_type)

        app_config_client.set_configuration_setting(
            configuration_setting=ConfigurationSetting(key=key, label=label, value=value, content_type=content_type)
        )
