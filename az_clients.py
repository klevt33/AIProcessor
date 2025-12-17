from azure.ai.projects import AIProjectClient
from azure.ai.textanalytics import TextAnalyticsClient
from azure.appconfiguration import AzureAppConfigurationClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.mgmt.web import WebSiteManagementClient
from azure.search.documents.indexes import SearchIndexClient
from azure.storage.blob import BlobServiceClient


class AzureClients:

    def __init__(self, config):
        self.config = config

        self.setup_text_analytics_client()
        self.setup_azure_project_client()
        self.setup_azure_blob_storage_client()
        self.setup_azure_webapp_client()
        self.setup_search_index_client()
        # self.setup_language_studio_client()
        self.setup_azure_app_config_client()

    def setup_text_analytics_client(self):
        self.text_analytics_client = TextAnalyticsClient(
            endpoint=self.config.LS_API_ENDPOINT_URL, credential=AzureKeyCredential(self.config.LS_API_KEY)
        )

    def setup_azure_project_client(self):
        self.azure_project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(), conn_str=self.config.AZ_AGENT_PROJECT_CONNECTION_STRING
        )

    def setup_azure_webapp_client(self):
        self.azure_web_client = WebSiteManagementClient(DefaultAzureCredential(), self.config.SUBSCRIPTION_ID)
        self.azure_webapp = self.azure_web_client.web_apps.list_application_settings(self.config.WA_RG, self.config.WA_APP_NAME)

    def setup_azure_blob_storage_client(self):
        self.azure_blob_storage_client = BlobServiceClient(
            account_url=self.config.AZ_AKSSTAI_ST_ACC_BLOB_API_BASE_URL, credential=DefaultAzureCredential()
        )

    def setup_search_index_client(self):
        self.credential = AzureKeyCredential(self.config.AZ_SEARCH_API_KEY)
        self.search_index_client = SearchIndexClient(endpoint=self.config.AZ_SEARCH_API_ENDPOINT_URL, credential=self.credential)

    # def setup_language_studio_client(self):
    #     self.lang_studio_client = ConversationAuthoringClient(
    #         endpoint=self.config.LS_API_ENDPOINT_URL,
    #         credential=AzureKeyCredential(self.config.LS_API_KEY)
    #     )

    def setup_azure_app_config_client(self):
        self.rg_app_config_client = AzureAppConfigurationClient.from_connection_string(self.config.RG_APP_CONFIG_CONN_STRING)
        self.ai_app_config_client = AzureAppConfigurationClient.from_connection_string(self.config.AI_APP_CONFIG_CONN_STRING)
