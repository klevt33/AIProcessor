"""
Module: azure_search_utils.py

Purpose:
This module provides the `AzureSearchUtils` class, a comprehensive utility for interacting
with Azure AI Search. It aims to simplify common operations related to Azure AI Search,
including index lifecycle management (creation, deletion, existence checks),
uploading documents, and performing diverse search queries such as text-based,
vector-based, hybrid, and semantic searches.

High-Level Design:
- Configuration-Driven: The `AzureSearchUtils` class is initialized with a configuration
  object containing Azure AI Search service endpoint URL, API key, and the target index name.
- Client Management: It internally manages instances of `SearchIndexClient` (for index
  schema operations) and `SearchClient` (for document and query operations).
  The `SearchClient` is lazy-initialized to ensure it's only created if the specified
  index actually exists, preventing errors at startup.
- Index Operations: Provides methods to create, delete, and check for the existence
  of search indexes. Index creation supports fields, vector search configurations,
  semantic search configurations, custom analyzers, and token filters.
- Data Upload: Includes a method to upload documents in batches to the specified index.
- Advanced Search Capabilities:
    - The primary `search` method is highly flexible, supporting:
        - Text search (simple, full, semantic).
        - Vector search (via `VectorizedQuery`).
        - Hybrid search (combining text and vector).
        - OData filtering.
        - Field selection (`select`).
        - Searching specific fields (`search_fields`).
        - Pagination (both `top` for single page and fetching all results).
        - Sorting (`orderby`).
        - Total result count inclusion.
        - Semantic configurations.
    - Retry Mechanism: A private method `_execute_search_with_retry` implements
      robust retry logic with exponential backoff for search operations to handle
      transient network issues or temporary service unavailability.
- Helper Methods: Includes factory methods (`create_vector_search_config`,
  `create_semantic_search_config`) to simplify the creation of complex
  configuration objects for vector and semantic search.
- Use-Case Specific Method: `get_parts_data_from_index` demonstrates a practical
  application of the search capabilities, fetching specific data based on a list
  of manufacturer part numbers.
- Error Handling: Uses custom exceptions (`InvoiceProcessingError`) and logs errors
  and warnings, providing context for diagnostics.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ServiceRequestError, ServiceResponseError
from azure.search.documents import SearchClient, SearchItemPaged
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
)
from azure.search.documents.models import QueryType, VectorFilterMode, VectorizedQuery
from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_attempt, wait_exponential

from exceptions import InvoiceProcessingError
from logger import logger

# --- Helper Logic for Retries ---


def is_retryable_search_error(exception: BaseException) -> bool:
    """
    Determines if an exception raised during search should trigger a retry.
    Retries on network/connection errors, but fails fast on configuration errors.
    """
    error_msg = str(exception).lower()

    # Fail fast: Do not retry semantic configuration errors (user config error)
    if "semanticconfiguration" in error_msg or "semantic configuration" in error_msg:
        return False

    # Retry on:
    # 1. Standard Azure Service/HTTP errors
    # 2. ConnectionResetError/OSError (often wrapped or direct)
    # 3. General transient failures
    if isinstance(exception, (HttpResponseError, ServiceRequestError, ServiceResponseError, ConnectionError, OSError)):
        return True

    # Retry on generic Exception is usually too broad, but if you want maximum resilience
    # for 'Connection aborted' which might come as a generic Exception tuple:
    return True


class PreFetchedResults:
    """
    Helper class to mimic Azure's SearchItemPaged object.
    Used to return materialized data from the retry-protected method
    so the upstream 'search' method consumes it without triggering new network calls.
    """

    def __init__(self, data: List[Dict[str, Any]], count: Optional[int]):
        self._data = data
        self._count = count

    def __iter__(self):
        return iter(self._data)

    def get_count(self):
        return self._count


class AzureSearchUtils:
    """General utilities for working with Azure AI Search"""

    def __init__(self, config: Any):
        """
        Initialize Azure AI Search utilities using a configuration object.

        This constructor sets up the connection parameters for Azure AI Search,
        including the service endpoint, API key, and the name of the index
        to operate on. It initializes the `SearchIndexClient` for managing
        the index schema. The `SearchClient` for querying is initialized lazily.

        Args:
            config (Any): A configuration object expected to have attributes like
                          `AZ_SEARCH_API_ENDPOINT_URL` (str): The endpoint URL of the Azure AI Search service.
                          `AZ_SEARCH_API_KEY` (str): The API key for authenticating with the service.
                          `AZ_SEARCH_INDEX_NAME` (str): The name of the search index to be used.
        """
        self.config = config
        self.endpoint = config.AZ_SEARCH_API_ENDPOINT_URL
        self.api_key = config.AZ_SEARCH_API_KEY
        self.index_name = config.AZ_SEARCH_INDEX_NAME
        # self.index_name = "test-index"

        # Initialize clients
        self.credential = AzureKeyCredential(self.api_key)
        self.index_client = config.azure_clients.search_index_client

        # Search client will be initialized if/when index exists
        self._search_client = None

    @property
    def search_client(self) -> Optional[SearchClient]:
        """
        Provides lazy initialization for the SearchClient.

        The SearchClient is used for querying and uploading documents to the index.
        It is initialized only when first accessed and only if the target index
        is confirmed to exist. This prevents errors if the index is not yet created.

        Returns:
            Optional[SearchClient]: The initialized SearchClient if the index exists,
                                    otherwise None.
        """
        if self._search_client is None and self.index_exists():
            self._search_client = SearchClient(endpoint=self.endpoint, index_name=self.index_name, credential=self.credential)
        return self._search_client

    def index_exists(self) -> bool:
        """
        Checks if the configured search index currently exists in the Azure AI Search service.

        Returns:
            bool: True if the index exists, False otherwise (e.g., if it hasn't been
                  created or if there's an error accessing it).
        """
        try:
            self.index_client.get_index(self.index_name)
            return True
        except Exception:
            return False

    def delete_index(self) -> bool:
        """
        Deletes the configured search index from the Azure AI Search service if it exists.

        If the index is successfully deleted, the internal `_search_client` instance
        is reset to None, as it would no longer be valid.

        Returns:
            bool: True if the index was successfully deleted or if it didn't exist initially.
                  False if an error occurred during the deletion attempt.
        """
        try:
            if self.index_exists():
                self.index_client.delete_index(self.index_name)
                # Reset search client since index no longer exists
                self._search_client = None
                return True
            return False
        except Exception:
            return False

    def create_index(
        self,
        fields: List[Union[SimpleField, SearchableField]],
        vector_search: Optional[VectorSearch] = None,
        semantic_search: Optional[SemanticSearch] = None,
        analyzers: Optional[List[Dict[str, Any]]] = None,
        token_filters: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Creates a new search index with the specified schema and configurations.

        This method allows defining fields, vector search profiles, semantic search
        configurations, custom text analyzers, and token filters for the new index.
        If successful, it also initializes (or re-initializes) the internal
        `_search_client` to be ready for operations on the newly created index.

        Args:
            fields (List[Union[SimpleField, SearchableField]]): A list of field definitions
                for the index schema (e.g., SimpleField, SearchableField).
            vector_search (Optional[VectorSearch]): Configuration for vector search
                capabilities, including profiles and algorithms. Defaults to None.
            semantic_search (Optional[SemanticSearch]): Configuration for semantic
                search capabilities. Defaults to None.
            analyzers (Optional[List[Dict[str, Any]]]): A list of custom analyzer
                definitions. Defaults to None.
            token_filters (Optional[List[Dict[str, Any]]]): A list of custom token
                filter definitions. Defaults to None.

        Returns:
            bool: True if the index is created successfully, False if an error occurs.
        """
        try:
            # Build the index definition
            index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search, semantic_search=semantic_search)

            # Add custom analyzers if provided
            if analyzers:
                index.analyzers = analyzers

            # Add custom token filters if provided
            if token_filters:
                index.token_filters = token_filters

            # Create the index in Azure AI Search
            self.index_client.create_index(index)

            # Initialize the search client for querying the index
            self._search_client = SearchClient(endpoint=self.endpoint, index_name=self.index_name, credential=self.credential)

            return True
        except Exception:
            return False

    def create_vector_search_config(
        self,
        profile_name: str = "vector-profile",
        algorithm_name: str = "vector-config",
        m: int = 8,
        ef_construction: int = 600,
        ef_search: int = 800,
        metric: str = "cosine",
    ) -> VectorSearch:
        """
        Creates a VectorSearch configuration object for defining vector search behavior in an index.

        This helper method constructs a `VectorSearch` object with a HNSW (Hierarchical Navigable
        Small World) algorithm configuration, which is commonly used for efficient approximate
        nearest neighbor searches.

        Args:
            profile_name (str): Name for the vector search profile.
                                Defaults to "vector-profile".
            algorithm_name (str): Name for the HNSW algorithm configuration.
                                  Defaults to "vector-config".
            m (int): HNSW algorithm parameter M (number of bi-directional links created
                     for every new element). Defaults to 8.
            ef_construction (int): HNSW algorithm parameter efConstruction (size of the dynamic
                                   list for candidate neighbors during index construction).
                                   Defaults to 600.
            ef_search (int): HNSW algorithm parameter efSearch (size of the dynamic list for
                             candidate neighbors during search). Defaults to 800.
            metric (str): The similarity metric to use for vector comparison.
                          Common values are "cosine", "dotProduct", "euclidean".
                          Defaults to "cosine".

        Returns:
            VectorSearch: A `VectorSearch` configuration object.
        """
        return VectorSearch(
            profiles=[VectorSearchProfile(name=profile_name, algorithm_configuration_name=algorithm_name)],
            algorithms=[
                HnswAlgorithmConfiguration(
                    name=algorithm_name,
                    kind=VectorSearchAlgorithmKind.HNSW,
                    parameters=HnswParameters(m=m, ef_construction=ef_construction, ef_search=ef_search, metric=metric),
                )
            ],
        )

    def create_semantic_search_config(
        self,
        config_name: str = "semantic-config",
        content_field_names: Optional[List[str]] = None,
        title_field_name: Optional[str] = None,
    ) -> SemanticSearch:
        """
        Creates a SemanticSearch configuration object for enabling semantic ranking in an index.

        Semantic search enhances search relevance by understanding the intent and contextual
        meaning of search queries. This configuration specifies which fields should be
        prioritized for semantic processing.

        Args:
            config_name (str): Name for the semantic configuration.
                               Defaults to "semantic-config".
            content_field_names (Optional[List[str]]): A list of field names that contain
                the main content to be used for semantic ranking. Defaults to None.
            title_field_name (Optional[str]): The name of the field that represents
                the title. Defaults to None.

        Returns:
            SemanticSearch: A `SemanticSearch` configuration object.
        """
        content_fields = []
        if content_field_names:
            content_fields = [SemanticField(field_name=name) for name in content_field_names]

        title_field = SemanticField(field_name=title_field_name) if title_field_name else None

        return SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name=config_name,
                    prioritized_fields=SemanticPrioritizedFields(title_field=title_field, content_fields=content_fields),
                )
            ]
        )

    def upload_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Uploads a list of documents to the configured Azure AI Search index.

        Args:
            documents (List[Dict[str, Any]]): A list of documents to upload. Each document
                should be a dictionary where keys correspond to field names in the index.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the upload operation,
                            including "success_count", "total_count", and detailed "details"
                            for each document upload attempt.

        Raises:
            InvoiceProcessingError: If the search client is not initialized (e.g., if the
                                    index does not exist or was not properly created).
        """
        if not self.search_client:
            raise InvoiceProcessingError("Index does not exist. Create the index before uploading documents.")

        result = self.search_client.upload_documents(documents)
        success_count = sum(1 for r in result if r.succeeded)

        return {"success_count": success_count, "total_count": len(documents), "details": result}

    @retry(
        retry=retry_if_exception(is_retryable_search_error),
        wait=wait_exponential(multiplier=0.5, min=1, max=10),
        stop=stop_after_attempt(4),  # 1 initial + 3 retries
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _execute_search_with_retry(
        self, query_text: Optional[str], search_params: Dict[str, Any], retries: int
    ) -> Union[SearchItemPaged[Dict[str, Any]], PreFetchedResults]:
        """
        Executes a search query against the Azure AI Search index with robust retry logic.

        This method wraps the Azure SDK search call and forces the materialization
        (download) of results within the retry block. This ensures that transient
        network errors occurring during the data transfer phase (e.g., ConnectionResetError)
        are caught and retried automatically.

        Args:
            query_text (Optional[str]): The text to search for. Can be None for filter-only
                                        or vector-only queries.
            search_params (Dict[str, Any]): A dictionary of parameters to pass to the
                                            search client's `search` method.
            retries (int): Deprecated. The number of retries is now controlled by the
                           decorator configuration (1 initial attempt + 3 retries).
                           Kept for signature compatibility with the caller.

        Returns:
            Union[SearchItemPaged, PreFetchedResults]: An iterable object containing the search
                                                       results. It mimics the interface of
                                                       Azure's SearchItemPaged (supporting
                                                       iteration and .get_count()).

        Raises:
            InvoiceProcessingError: If a specific non-retryable configuration error occurs
                                    (e.g., missing semantic configuration).
            HttpResponseError: If the search fails after all retry attempts due to API errors.
            ConnectionError: If the search fails after all retry attempts due to network issues.
        """
        # 1. Initiate the search (this is usually lazy)
        results = self.search_client.search(search_text=query_text, **search_params)

        # 2. Force Materialization (Download data)
        # We iterate over the results here to force the network transfer.
        # If the connection drops here, Tenacity catches it.
        try:
            results_list = list(results)
        except Exception as e:
            # Check specifically for semantic config error to provide clean context
            # (Tenacity would retry this otherwise if we didn't filter it in is_retryable_search_error,
            # but catching here adds the InvoiceProcessingError context)
            if "semanticConfiguration" in str(e) or "semantic configuration" in str(e).lower():
                raise InvoiceProcessingError("Semantic search requires a valid configuration in the index.") from e
            raise e

        # 3. Retrieve Count if requested
        # Safe to do after materialization as SDK usually caches it or fetches it separately.
        count = None
        if search_params.get("include_total_count"):
            try:
                count = results.get_count()
            except Exception:
                # If count fetch fails but data succeeded, strictly speaking we could retry,
                # but it's rare to fail here if data transfer succeeded.
                pass

        # 4. Return wrapper that looks like the original iterator
        return PreFetchedResults(results_list, count)

    def search(
        self,
        query_text: Optional[str] = None,
        filter_expression: Optional[str] = None,
        vector_query: Optional[Dict[str, Any]] = None,
        select: Optional[List[str]] = None,
        search_fields: Optional[List[str]] = None,
        top: Optional[int] = None,
        orderby: Optional[Union[str, List[str]]] = None,
        query_type: QueryType = QueryType.SIMPLE,
        include_total_count: bool = False,
        vector_filter_mode: VectorFilterMode = VectorFilterMode.PRE_FILTER,
        semantic_configuration_name: Optional[str] = None,
        retries: int = 3,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, int]]:
        """
        Performs a search query on the Azure AI Search index.

        Supports various search types including text, vector, and hybrid (text + vector).
        Allows filtering, field selection, sorting, and semantic ranking.
        Handles pagination: if `top` is specified, a single page of results is returned;
        if `top` is None, all results are fetched page by page.
        Includes retry logic for transient API failures.

        Args:
            query_text (Optional[str]): The search query string. Can be None or "*" for
                                        filter-only or vector-only searches. Defaults to None.
            filter_expression (Optional[str]): OData filter expression (e.g., "category eq 'electronics'").
                                               Defaults to None.
            vector_query (Optional[Dict[str, Any]]): Dictionary defining vector query parameters.
                Expected keys: 'vector' (List[float]), 'fields' (str or List[str]),
                'k_nearest_neighbors' (int, optional), 'exhaustive' (bool, optional),
                'oversampling' (float, optional). Defaults to None.
            select (Optional[List[str]]): List of field names to retrieve in the results.
                                          Defaults to None (all fields).
            search_fields (Optional[List[str]]): List of field names to search over for text queries.
                                                 Defaults to None (searches all searchable fields).
            top (Optional[int]): Maximum number of results to return. If None, retrieves all
                                 results using pagination. Defaults to None.
            orderby (Optional[Union[str, List[str]]]): Sort order expression(s)
                (e.g., "last_modified desc"). Defaults to None.
            query_type (QueryType): Type of search query. Options are QueryType.SIMPLE,
                                    QueryType.FULL, QueryType.SEMANTIC. Defaults to QueryType.SIMPLE.
            include_total_count (bool): Whether to include the total count of matching
                                        documents. Defaults to False.
            vector_filter_mode (VectorFilterMode): Specifies when the vector filter is applied
                (VectorFilterMode.PRE_FILTER or VectorFilterMode.POST_FILTER).
                Defaults to VectorFilterMode.PRE_FILTER.
            semantic_configuration_name (Optional[str]): Name of the semantic configuration
                to use if `query_type` is QueryType.SEMANTIC. Defaults to None.
            retries (int): Number of times to retry the search API call if it fails.
                           Defaults to 1 (meaning 1 retry after the initial attempt).

        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, int]]:
                - If `include_total_count` is False: A pandas DataFrame containing the search results.
                - If `include_total_count` is True: A tuple where the first element is the
                  DataFrame of results, and the second is the total count (int) of matching documents.
                  Returns an empty DataFrame if no results are found.

        Raises:
            InvoiceProcessingError: If the search client is not initialized, or if `vector_query`
                                    is malformed, or for specific configuration errors.
            HttpResponseError: If the search fails after all retry attempts. (Propagated from _execute_search_with_retry)
        """
        if not self.search_client:
            raise InvoiceProcessingError("Search client not initialized")

        # Process vector queries
        vector_queries = []
        if vector_query:
            if "vector" not in vector_query or "fields" not in vector_query:
                raise InvoiceProcessingError("vector_query requires 'vector' and 'fields'")

            vector_queries.append(
                VectorizedQuery(
                    vector=vector_query["vector"],
                    fields=vector_query["fields"],
                    k_nearest_neighbors=vector_query.get("k_nearest_neighbors", 5),
                    exhaustive=vector_query.get("exhaustive", False),
                    oversampling=vector_query.get("oversampling", None),
                    # weight=vector_query.get('weight', 1.0) # Weight is part of HybridSearch, not VectorizedQuery directly in
                    # newer SDKs? Check SDK version. Assuming it's handled differently or not needed here.
                )
            )

        # Common search parameters setup
        base_search_params = {
            "filter": filter_expression,
            "vector_queries": vector_queries if vector_queries else None,
            "select": select,
            "search_fields": search_fields,
            "include_total_count": include_total_count,
            "query_type": query_type,
            "order_by": orderby,
        }

        # Add vector filter mode if vector queries are present
        if vector_queries and vector_filter_mode:
            base_search_params["vector_filter_mode"] = vector_filter_mode

        # Add semantic configuration if specified (query_type check is often handled by the SDK, but explicit check is fine)
        if semantic_configuration_name and query_type == QueryType.SEMANTIC:  # Assuming QueryType enum usage
            base_search_params["semantic_configuration_name"] = semantic_configuration_name
        elif semantic_configuration_name and query_type == "semantic":  # String comparison fallback
            base_search_params["semantic_configuration_name"] = semantic_configuration_name

        # If `top` is specified, perform a single search request
        if top is not None:
            search_params = base_search_params.copy()
            search_params["top"] = top

            # Execute search with retry logic
            results = self._execute_search_with_retry(query_text, search_params, retries)

            # Extract total count if requested
            total_count = results.get_count() if include_total_count else None

            df = (
                pd.DataFrame(
                    [
                        {
                            **{
                                k: v
                                for k, v in result.items()
                                if k not in ["@search.score", "@search.reranker_score", "@search.hybrid_score"]
                            },  # Avoid duplicating special fields
                            "@search.score": result.get("@search.score", 0.0),
                            "@search.reranker_score": result.get("@search.reranker_score", 0.0),
                            "@search.hybrid_score": result.get("@search.hybrid_score", 0.0),
                        }
                        for result in results
                    ]
                )
                if results
                else pd.DataFrame()
            )

            # Return DataFrame with total count if requested
            if include_total_count:
                return df, total_count
            return df

        # If `top` is None, retrieve all results with explicit pagination
        else:
            all_results_list = []
            page_size = 1000  # Max results per request per Azure Search docs (check current limit)
            skip = 0
            total_count = None
            page_num = 0  # For potential error logging context

            while True:
                page_num += 1
                search_params = base_search_params.copy()
                search_params["top"] = page_size
                search_params["skip"] = skip

                # Execute paged search with retry logic
                results_page = self._execute_search_with_retry(query_text, search_params, retries)

                # Get total count from first page if requested
                if include_total_count and total_count is None:
                    try:
                        total_count = results_page.get_count()
                    except Exception as e_count_page:
                        logger.error(
                            "SEARCH_CONSUME_ERROR: During results_page.get_count() "
                            f"(pagination path, page {page_num}): {e_count_page}",
                            exc_info=True,
                        )
                        raise

                # Convert current page to list to evaluate if it's empty
                current_page_list = []
                try:
                    current_page_list = list(results_page)  # Materialize the page
                except Exception as e_iter_page:
                    logger.error(
                        f"SEARCH_CONSUME_ERROR: During list(results_page) (pagination path, page {page_num}): {e_iter_page}",
                        exc_info=True,
                    )
                    raise

                if not current_page_list:
                    break  # No more results, exit loop

                all_results_list.extend(current_page_list)
                skip += len(current_page_list)  # More robust than skip += page_size if last page is smaller

                # Optional: Add a check to prevent infinite loops if skip doesn't advance
                if len(current_page_list) < page_size:
                    break  # Last page reached

            df = (
                pd.DataFrame(
                    [
                        {
                            **{
                                k: v
                                for k, v in result.items()
                                if k not in ["@search.score", "@search.reranker_score", "@search.hybrid_score"]
                            },
                            "@search.score": result.get("@search.score", 0.0),
                            "@search.reranker_score": result.get("@search.reranker_score", 0.0),
                            "@search.hybrid_score": result.get("@search.hybrid_score", 0.0),
                        }
                        for result in all_results_list
                    ]
                )
                if all_results_list
                else pd.DataFrame()
            )

            # Return DataFrame with total count if requested
            if include_total_count:
                # Ensure total_count is fetched if include_total_count=True but no results found
                if total_count is None and include_total_count and not all_results_list:
                    # Fetch count explicitly if pagination didn't run
                    count_params = base_search_params.copy()
                    count_params["top"] = 0  # Only need the count
                    count_params["include_total_count"] = True
                    count_results = self._execute_search_with_retry(
                        query_text=None, search_params=count_params, retries=retries  # No query text needed for count usually
                    )
                    try:
                        total_count = count_results.get_count()
                    except Exception as e_explicit_count:
                        logger.error(
                            f"SEARCH_CONSUME_ERROR: During count_results.get_count() (explicit count path): {e_explicit_count}",
                            exc_info=True,
                        )
                        raise

                return df, total_count if total_count is not None else 0
            return df

    async def get_parts_data_from_index(self, mfr_part_numbers: List[str]) -> pd.DataFrame:
        """
        Fetches parts data from the Azure AI Search index based on a list of manufacturer part numbers.

        This method constructs a filter query to match exact manufacturer part numbers against
        the 'MfrPartNumExact' field in the index. It retrieves specified fields including
        vector embeddings ('ItemDescription_vector').
        Input part numbers containing '-', '/', or '.' characters are currently filtered out.

        Args:
            mfr_part_numbers (List[str]): A list of Manufacturer Part Numbers (strings)
                                          to search for.

        Returns:
            pd.DataFrame: A DataFrame containing the parts data for the found manufacturer
                          part numbers. Columns include "ItemID", "MfrPartNum", "MfrName",
                          "UPC", "UNSPSC", "AKPartNum", "DescriptionID", "ItemDescription",
                          and "ItemDescription_vector". Returns an empty DataFrame if no
                          valid part numbers are provided or if no matching parts are found.

        Raises:
            This method may propagate exceptions from the underlying `self.search` call,
            such as `InvoiceProcessingError` or `HttpResponseError` if the search operation fails.
        """

        def get_empty_dataframe() -> pd.DataFrame:
            # Defines the schema for an empty DataFrame, consistent with expected output.
            columns = [
                "ItemID",
                "MfrPartNum",
                "MfrName",
                "UPC",
                "UNSPSC",
                "AKPartNum",
                "ItemSourceName",
                "DescriptionID",
                "ItemDescription",
                "DescSourceName",
                "ItemDescription_vector",
            ]
            return pd.DataFrame(columns=columns)

        # Handle empty or invalid mfr_part_numbers
        if not mfr_part_numbers or not isinstance(mfr_part_numbers, list):
            return get_empty_dataframe()

        # Filter out None or empty strings from the input list
        mfr_part_numbers = [pn.strip() for pn in mfr_part_numbers if pn and isinstance(pn, str)]

        # Filter out all values containing -/. chars
        mfr_part_numbers = [p for p in mfr_part_numbers if not any(c in p for c in "-/.")]

        # If no valid part numbers remain after filtering, return an empty DataFrame
        if not mfr_part_numbers:
            return get_empty_dataframe()

        # Build filter expression for multiple part numbers using OR conditions
        filter_conditions: List[str] = [f"MfrPartNumExact eq '{pn}'" for pn in mfr_part_numbers]
        filter_expression: str = " or ".join(filter_conditions)

        # Select all relevant fields for the output
        select_fields: List[str] = [
            "ItemID",
            "MfrPartNumExact",
            "MfrName",
            "UPC",
            "UNSPSC",
            "AKPartNum",
            "ItemSourceName",
            "DescriptionID",
            "ItemDescription",
            "DescSourceName",
            "ItemDescription_vector",
        ]

        # Execute search with filter expression
        # Let exceptions propagate to be caught by the calling function
        results: pd.DataFrame = self.search(
            query_text="*",  # Wildcard query to match everything, filtering is done by filter_expression
            filter_expression=filter_expression,
            select=select_fields,
        )

        # Rename MfrPartNumExact column to match the original function's output
        if not results.empty and "MfrPartNumExact" in results.columns:
            results = results.rename(columns={"MfrPartNumExact": "MfrPartNum"})

        return results
