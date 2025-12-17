"""
## Overview
This module provides functionality for interacting with Azure Cosmos DB. It includes methods for creating containers, adding
items, querying documents, and updating records. The module is designed to handle AI job documents and AI process logs, ensuring
robust and efficient database operations.
"""

import json
import math
from enum import Enum
from typing import List, Optional

from azure.cosmos import ContainerProxy, CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError

from constants import Constants, DatabaseObjects, Logs
from logger import logger


class CDB:
    """
    Handles interactions with Azure Cosmos DB, including creating containers, adding items,
    querying documents, and updating records.

    Args:
        config (Config): Configuration object containing Cosmos DB connection details.

    Attributes:
        ai_jobs_container (ContainerProxy): Container for storing AI job documents.
        ai_logs_container (ContainerProxy): Container for storing AI process logs.
    """

    def __init__(self, config):
        """
        Initializes the Cosmos DB client and creates containers if they do not exist.

        Args:
            config (Config): Configuration object containing Cosmos DB connection details.
        """
        self.config = config

        client = CosmosClient(url=config.COSMOS_DB_URI, credential=config.COSMOS_DB_PRIMARY_KEY)
        database = client.create_database_if_not_exists(config.COSMOS_DB_DATABASE_ID)

        # Container for AI API requests
        self.ai_api_requests_container = database.create_container_if_not_exists(
            id=config.COSMOS_DB_CONTAINER_ID_MAP[DatabaseObjects.CDB_CONTAINER_AI_API_REQUESTS],
            partition_key=PartitionKey(path="/id"),
        )

        # Container for AI job documents
        self.ai_jobs_container = database.create_container_if_not_exists(
            id=config.COSMOS_DB_CONTAINER_ID_MAP[DatabaseObjects.CDB_CONTAINER_AI_JOBS], partition_key=PartitionKey(path="/id")
        )

        # Container for AI process logs
        self.ai_logs_container = database.create_container_if_not_exists(
            id=config.COSMOS_DB_CONTAINER_ID_MAP[DatabaseObjects.CDB_CONTAINER_AI_PROCESS_LOGS],
            partition_key=PartitionKey(path="/request_details.id"),
        )

    def add_items(self, container, documents):
        """
        Adds multiple documents to a Cosmos DB container.

        Args:
            container (ContainerProxy): The Cosmos DB container to add items to.
            documents (List[dict]): List of documents to be added.

        Returns:
            Tuple[List[str], List[Tuple[str, str]]]: Lists of duplicate and failed document IDs.
        """
        duplicates = []
        failures = []
        for document in documents:
            result = self.add_item(container, document)

            if result["status"] == "duplicate":
                duplicates.append(result["id"])
            elif result["status"] == "error":
                failures.append((result["id"], result["message"]))

        return duplicates, failures

    # def add_item(self, container, document):
    #     """
    #     Adds multiple documents to a Cosmos DB container.

    #     Args:
    #         container (ContainerProxy): The Cosmos DB container to add items to.
    #         document (dict): document to be added.

    #     Returns:
    #         dict[str, Any]: result dict from cosmos transaction
    #     """
    #     result = {
    #         Constants.STATUS_lower: Constants.ERROR_lower,
    #         Constants.MESSAGE: None,
    #         "id": None
    #     }
    #     try:
    #         result = self.add_item(container=container, document=document)

    #     except ServiceRequestError as e:
    #         logger.error("ServiceRequestError for id=%s: %s", result["id"], e, exc_info=True)
    #         raise e

    #     except Exception as e:
    #         logger.error(f"Error occurred while uploading documents to cosmos db {str(e)}", result["id"], e, exc_info=True)
    #         raise e
    #     return result

    def get_documents(
        self,
        container: ContainerProxy,
        top: int | None = None,
        column_filter: Optional[str] = None,
        where_condition: Optional[str] = None,
        order_condition: Optional[str] = None,
    ) -> List[dict]:
        """
        Queries documents from a Cosmos DB container.

        Args:
            container (ContainerProxy): The Cosmos DB container to query.
            top (int, optional): Maximum number of documents to fetch.
            column_filter (str, optional): Columns to select in the query.
            where_condition (str, optional): WHERE clause for filtering documents.
            order_condition (str, optional): ORDER BY clause for sorting documents.

        Returns:
            List[dict]: A list of documents matching the query.
        """
        documents = []
        try:
            query = "SELECT "
            if top:
                query += f"TOP {top} "
            if column_filter:
                query += column_filter + " FROM c "
            else:
                query += "* FROM c "
            if where_condition:
                query += where_condition
            if order_condition:
                query += order_condition
            # logger.debug(f"\nQuery: {query}\n")
            documents = list(container.query_items(query, enable_cross_partition_query=True))

        except CosmosHttpResponseError as e:
            logger.error(f"Failed to fetch items: {e.message}. Query: {query}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to read documents from cosmos db: {str(e)}", exc_info=True)
        return documents

    def get_documents_count(self, container: ContainerProxy, where_condition: Optional[str] = None) -> int:
        """
        Queries the document count from a Cosmos DB container.

        Args:
            container (ContainerProxy): The Cosmos DB container to query.
            where_condition (str, optional): A WHERE clause (starting with 'WHERE') for filtering documents.

        Returns:
            int: The count of documents matching the query.
        """
        query = "SELECT VALUE COUNT(1) FROM c "
        if where_condition:
            query += f" {where_condition}"

        try:
            items = list(container.query_items(query=query, enable_cross_partition_query=True))
            count = items[0] if items else 0

        except CosmosHttpResponseError as e:
            logger.error(f"Failed to fetch items: {e.message}. Query: {query}", exc_info=True)
            count = 0
        except Exception as e:
            logger.error(f"Failed to read documents from Cosmos DB: {str(e)}. Query: {query}", exc_info=True)
            count = 0

        return count

    def update_document(self, container, document):
        """
        Updates a document in a Cosmos DB container.
        replace_item() → updates an existing document by id and partition key.
        upsert_item() → insert or replace automatically — this is what you want.

        Args:
            container (ContainerProxy): The Cosmos DB container to update the document in.
            document (dict): The document to be updated.

        Raises:
            CosmosHttpResponseError: If the update operation fails.
        """
        try:
            document = json.dumps(document, default=str, allow_nan=False)
            document = json.loads(document)

            response = container.upsert_item(document)
            # container.replace_item(item=document['id'], body=document)
            response = {Constants.STATUS_lower: Constants.SUCCESS_lower, Constants.MESSAGE: "Update document in cosmos db."}
        except CosmosHttpResponseError as e:
            logger.error(f"Failed to update item: {e.status_code} - {e.message}. Document: {document}", exc_info=True)
            response = {
                Constants.STATUS_lower: Constants.ERROR_lower,
                Constants.MESSAGE: f"Failed to update item: {e.status_code} - {e.message}",
            }
        except Exception as e:
            logger.error(f"Failed to update document in cosmos db: {e.status_code} - {e.message}", exc_info=True)
            response = {
                Constants.STATUS_lower: Constants.ERROR_lower,
                Constants.MESSAGE: f"Failed to update item: {e.status_code} - {e.message}",
            }
        return response

    def patch_document(self, container, doc_id, operations, partition_key=None, raise_on_error=False):
        """
        Patches a document in a Cosmos DB container with flexible error handling and a consistent return contract.

        - In API Mode (default, raise_on_error=False): Catches all exceptions and returns a status dictionary.
        - In Worker Mode (raise_on_error=True): Raises exceptions on any failure, including 'NotFound', to allow a background
        worker to handle retries or poison pills.
        """
        try:
            # Enforce the best practice that a partition key must be provided for a patch operation.
            if partition_key is None:
                raise ValueError("partition_key cannot be None for a patch operation.")

            logger.debug(f"Attempting to patch document id={doc_id}, partition_key={partition_key}, operations={operations}")

            # Execute the patch and get the updated document body in the response.
            response = container.patch_item(item=doc_id, partition_key=partition_key, patch_operations=operations)

            logger.info(f"Patched document id={doc_id} successfully.")

            # Return the success dictionary, using the project's constants.
            return {
                Constants.STATUS_lower: Constants.SUCCESS_lower,
                Constants.MESSAGE: "Field added/updated successfully.",
                "data": response,
            }

        except CosmosResourceNotFoundError as e:
            # A 'NotFound' error is a critical failure for our background worker,
            # but a standard error for the API.
            error_message = f"Document with id={doc_id} not found."
            logger.error(f"CosmosResourceNotFoundError patching id={doc_id} (pkey={partition_key}): {error_message}")

            if raise_on_error:
                # For the SqlWriterService, this is a fatal error for the document, so we raise.
                raise ValueError(error_message) from e
            else:
                # For the API, return a structured error dictionary using constants.
                return {Constants.STATUS_lower: Constants.ERROR_lower, Logs.ID: doc_id, Constants.MESSAGE: error_message}

        except Exception as e:
            # For any other unexpected error, decide whether to raise or return.
            logger.error(f"Unexpected error patching document id={doc_id}", exc_info=True)

            if raise_on_error:
                # The worker needs the exception to trigger its failure logic.
                raise
            else:
                # The API returns a structured error dictionary using constants.
                return {
                    Constants.STATUS_lower: Constants.ERROR_lower,
                    Logs.ID: doc_id,
                    Constants.MESSAGE: f"Failed to patch item: {e}",
                }

    def add_item(self, container, document):
        """
        Adds a single document to a Cosmos DB container.

        Args:
            container (ContainerProxy): The Cosmos DB container to add the document to.
            document (dict): The document to be added.

        Returns:
            dict: Status of the operation, including the document ID and any error messages.
        """

        def scrub_nans(obj):
            """
            Recursively replaces NaN values in the document with None.

            Args:
                obj: The object to scrub.

            Returns:
                The scrubbed object.
            """
            if isinstance(obj, float) and math.isnan(obj):
                return None
            elif isinstance(obj, dict):
                return {k: scrub_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [scrub_nans(v) for v in obj]
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj

        try:
            document = scrub_nans(document)
            document = json.dumps(document, default=str, allow_nan=False)
            document = json.loads(document)

            container.create_item(body=document)
            return {
                Constants.STATUS_lower: Constants.SUCCESS_lower,
                Constants.MESSAGE: "Item added to cosmos db successfully.",
                "id": document.get("id"),
            }

        except CosmosHttpResponseError as e:
            if e.status_code == 409:
                logger.error(f"Conflict: Item with id '{document['id']}' already exists.", exc_info=True)
                return {
                    Constants.STATUS_lower: "duplicate",
                    Constants.MESSAGE: f"Conflict: Item with id '{document['id']}' already exists.",
                    "id": document.get("id"),
                }
            else:
                logger.error(f"Failed to create item: {e.message}. Document: \n {json.dumps(document, indent=2)}", exc_info=True)
                return {
                    Constants.STATUS_lower: Constants.ERROR_lower,
                    "id": document.get("id"),
                    Constants.MESSAGE: f"Failed to create item: {e.message}.",
                }
        except Exception as e:
            logger.error(f"Failed to create item: {str(e)}. Document: \n {json.dumps(document, indent=2)}", exc_info=True)
            return {
                Constants.STATUS_lower: Constants.ERROR_lower,
                "id": document.get("id"),
                Constants.MESSAGE: f"Failed to create item: {str(e)}",
            }

    def convert_log_id_to_job_id(self, log_id: str) -> str:
        """
        Convert log id (with '-') into job id (with '~').
        """
        prefix, suffix = log_id.rsplit("-", 1)
        return f"{prefix}~{suffix}"

    def convert_job_id_to_log_id(self, log_id: str) -> str:
        """
        Convert job id (with '~') into log id (with '-').
        """
        prefix, suffix = log_id.rsplit("~", 1)
        return f"{prefix}-{suffix}"

    def get_item_count(self, container, where_condition: str) -> int:
        """
        Returns the count of documents in a container that match a WHERE condition.

        Args:
            container: The Cosmos DB container client.
            where_condition (str): The SQL WHERE clause to filter the count.

        Returns:
            int: The total number of matching documents.
        """
        query = f"SELECT VALUE COUNT(1) FROM c {where_condition}"
        try:
            # The result of a VALUE COUNT query is a list with a single number
            count = list(container.query_items(query=query, enable_cross_partition_query=True))[0]
            return count
        except Exception as e:
            logger.error(f"Cosmos DB: Failed to get item count with query '{query}'. Error: {e}", exc_info=True)
            raise  # Re-raise the exception to be handled by the caller
