"""
## Overview
This module provides functionality for interacting with Azure OpenAI services, including generating embeddings,
invoking language models, and calculating cosine similarity between text or embedding vectors.
It handles rate limiting and retries using the `tenacity` library to ensure robust operations.
"""

from enum import Enum
from functools import wraps
from typing import Type

import numpy as np
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
    APITimeoutError as OpenAPITimeoutError,
    InternalServerError as OpenAIInternalServerError,
    RateLimitError as OpenAIRateLimitError,
)
from pydantic import BaseModel
from scipy.spatial.distance import cosine
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

from ai_schemas import FinetunedLLMGeneralResponse, FinetunedLLMUNSPSCOnlyResponse
from constants import LLMResponseMethods
from exceptions import InvoiceProcessingError
from logger import logger

EMBEDDING_TRANSIENT_ERRORS = (OpenAIRateLimitError, OpenAIAPIConnectionError, OpenAPITimeoutError, OpenAIInternalServerError)


def retry_with_tiered_strategy(func):
    """
    A decorator that applies different retry strategies based on the type of exception:
    - For specified transient errors: retry up to 10 times with fixed 10-second delay.
    - For any other errors (excluding InvoiceProcessingError itself): retry up to 5 times with fixed 2-second delay.

    If all retries fail, the original exception is re-raised by tenacity, to be handled by the calling function.

    Args:
        func (Callable): The function to apply the retry strategy to.

    Returns:
        Callable: The wrapped function with retry strategies applied.
    """

    # --- Retry Policy Constants ---
    TRANSIENT_MAX_ATTEMPTS = 10
    TRANSIENT_WAIT_SECONDS = 10
    OTHER_MAX_ATTEMPTS = 5
    OTHER_WAIT_SECONDS = 2

    def _create_logging_callback(tier_name: str, policy_max_attempts: int):
        """Creates a `before_sleep` callback for tenacity that logs retry attempts."""

        def log_retry_attempt(rs: RetryCallState):
            exception = rs.outcome.exception()
            logger.info(
                f"Embeddings ({tier_name}): Attempt {rs.attempt_number}/{policy_max_attempts} failed. "
                f"Retrying {type(exception).__name__} (upcoming attempt {rs.attempt_number + 1}) "
                f"in {rs.next_action.sleep:.2f}s. Error: {str(exception)}"
            )

        return log_retry_attempt

    common_retry_kwargs = {"reraise": True}

    # Tier 1: Policy for specified transient errors
    transient_retry_decorator = retry(
        retry=retry_if_exception_type(EMBEDDING_TRANSIENT_ERRORS),
        wait=wait_fixed(TRANSIENT_WAIT_SECONDS),
        stop=stop_after_attempt(TRANSIENT_MAX_ATTEMPTS),
        before_sleep=_create_logging_callback(tier_name="Tier 1 - OpenAI Transient", policy_max_attempts=TRANSIENT_MAX_ATTEMPTS),
        **common_retry_kwargs,
    )

    # Tier 2: Policy for ALL OTHER errors not covered by the transient policy.
    # Excludes InvoiceProcessingError to prevent retrying an error already wrapped.
    other_errors_retry_decorator = retry(
        retry=retry_if_not_exception_type(EMBEDDING_TRANSIENT_ERRORS + (InvoiceProcessingError,)),
        wait=wait_fixed(OTHER_WAIT_SECONDS),
        stop=stop_after_attempt(OTHER_MAX_ATTEMPTS),
        before_sleep=_create_logging_callback(tier_name="Tier 2 - Other Errors", policy_max_attempts=OTHER_MAX_ATTEMPTS),
        **common_retry_kwargs,
    )

    # Apply tenacity decorators.
    # Decorators are applied from bottom up in terms of definition, but top down for execution (outermost first).
    # This order ensures that an exception flows from func -> other_errors_policy -> transient_errors_policy.
    @transient_retry_decorator  # Outer decorator, handles transient errors or re-raises from inner.
    @other_errors_retry_decorator  # Inner decorator, handles "other" errors first.
    @wraps(func)
    def new_wrapper(*args, **kwargs):
        """
        This thin wrapper simply calls the original function.
        Exceptions from 'func' will propagate to the tenacity @retry decorators.
        """
        return func(*args, **kwargs)

    return new_wrapper


class LLMClientType(Enum):
    """Enumeration for the available LLM clients."""

    CONTEXT_VALIDATOR = "CONTEXT_VALIDATOR"


class LLM:
    """
    A class to interact with Azure OpenAI services, including language models and embeddings.

    Args:
        config (Config): Configuration object containing Azure OpenAI connection details.

    Attributes:
        aoai_gpt4o_finetuned (AzureChatOpenAI): Client for interacting with a fine-tuned Azure OpenAI GPT-4 model.
        aoai_gpt4o (AzureChatOpenAI): Client for interacting with a base Azure OpenAI GPT-4 model.
        embeddings_client (AzureOpenAIEmbeddings): Client for generating embeddings using Azure OpenAI.
    """

    def __init__(self, config):
        """
        Initializes Azure OpenAI clients for language models and embeddings.

        Args:
            config (Config): Configuration object containing Azure OpenAI connection details.
        """
        self.config = config

        # Initialize fine-tuned GPT-4 model client
        self.aoai_gpt4o_finetuned = AzureChatOpenAI(
            azure_deployment=config.AOAI_FINETUNED_LLM_API_DEPLOYMENT,
            openai_api_key=config.AOAI_FINETUNED_LLM_OPENAI_API_KEY,
            azure_endpoint=config.AOAI_FINETUNED_LLM_API_ENDPOINT_URL,
            openai_api_version=config.AOAI_FINETUNED_LLM_API_VERSION,
            # response_format={},
            temperature=0.1,
            logprobs=True,
        )

        self.finetuned_llm_general = self.enforce_llm_with_structured_output(
            llm=self.aoai_gpt4o_finetuned, schema=FinetunedLLMGeneralResponse, retry_count=2
        )
        self.finetuned_llm_unspsc_only = self.enforce_llm_with_structured_output(
            llm=self.aoai_gpt4o_finetuned, schema=FinetunedLLMUNSPSCOnlyResponse, retry_count=2
        )

        # Initialize base GPT-4 model client
        self.aoai_gpt4o = AzureChatOpenAI(
            azure_deployment=config.AOAI_BASE_LLM_API_DEPLOYMENT,
            openai_api_key=config.AOAI_BASE_LLM_OPENAI_API_KEY,
            azure_endpoint=config.AOAI_BASE_LLM_API_ENDPOINT_URL,
            openai_api_version=config.AOAI_BASE_LLM_API_VERSION,
            temperature=0.1,
        )

        # Initialize embeddings client
        self.embeddings_client = AzureOpenAIEmbeddings(
            model=config.AOAI_EMBEDDING_API_MODEL,
            azure_deployment=config.AOAI_EMBEDDING_API_DEPLOYMENT,
            api_key=config.AOAI_EMBEDDING_OPENAI_API_KEY,
            azure_endpoint=config.AOAI_EMBEDDING_API_BASE_URL,
            api_version=config.AOAI_EMBEDDING_API_VERSION,
            dimensions=config.AOAI_EMBEDDING_DIMENSIONS,
        )

        # Initialize a new client for the Context Validator using its dedicated deployment
        self.aoai_context_validator = AzureChatOpenAI(
            azure_deployment=config.AOAI_CONTEXT_VALIDATOR_API_DEPLOYMENT,
            openai_api_key=config.AOAI_CONTEXT_VALIDATOR_API_KEY,
            azure_endpoint=config.AOAI_CONTEXT_VALIDATOR_API_BASE_URL,  # Use BASE_URL for LangChain
            openai_api_version=config.AOAI_CONTEXT_VALIDATOR_API_VERSION,
            temperature=0.0,
        )
        # Create a map to resolve client types to actual client instances
        self.client_map = {
            LLMClientType.CONTEXT_VALIDATOR: self.aoai_context_validator,
            # In the future, we can add other clients here:
            # LLMClientType.BASE_LLM: self.aoai_gpt4o,
        }

    def enforce_llm_with_structured_output(self, llm, schema, strict=True, method=LLMResponseMethods.JSON_SCHEMA, retry_count=0):
        llm = llm.with_structured_output(schema=schema, strict=strict, method=method)

        # if retry_count > 0:
        #     llm = self.enforce_llm_with_retries_on_error(llm=llm, schema=schema, retry_count=retry_count)
        return llm

    def enforce_llm_with_retries_on_error(self, llm, schema, retry_count=2):
        output_parser = PydanticOutputParser(pydantic_object=schema)
        # retry only twice if JSON is malformed
        retrying_llm = RetryWithErrorOutputParser.from_llm(llm=llm, parser=output_parser, max_retries=retry_count)
        return retrying_llm

    @retry(
        retry=retry_if_exception_type(
            (
                OpenAIRateLimitError,
                OpenAIAPIConnectionError,
                OpenAPITimeoutError,
                OpenAIInternalServerError,
                KeyError,  # Retrying KeyError as discussed
            )
        ),
        wait=wait_exponential(multiplier=2, min=5, max=30),
        stop=stop_after_attempt(6),  # Adjusted: 1 initial attempt + 5 retries (total 6 attempts)
        reraise=True,
        before_sleep=lambda rs: logger.info(
            f"LLM Response: Retrying {type(rs.outcome.exception()).__name__} "
            f"in {rs.next_action.sleep:.2f}s (Attempt"
            f" {rs.attempt_number + 1}/{LLM.get_llm_response.retry.stop.max_attempt_number}). "  # Show max attempts
            f"Error: {str(rs.outcome.exception())}"
        ),
    )
    async def get_llm_response(self, chain, params):
        """
        Get response using LangChain AzureOpenAI LLM with runnable chain and params as input.

        Args:
            chain (LangChain): LangChain runnable chain object.
            params (dict): Input parameters for the chain.

        Returns:
            dict: The response generated by the language model.

        Raises:
            InvoiceProcessingError: If the response generation fails.
        """
        try:
            # # FOR FUTURE REVIEW: the async method signature / sync call mismatch.
            # response_obj = chain.invoke(params, config={"return_raw": True})

            # # Parsed object (your Pydantic model)
            # parsed_obj = response_obj["parsed"]

            # # Raw ChatResult (with logprobs)
            # raw_result = response_obj["raw"]
            # return raw_result, parsed_obj

            # FOR FUTURE REVIEW: the async method signature / sync call mismatch.
            response_obj = chain.invoke(params)
            return response_obj

        except (
            OpenAIRateLimitError,
            OpenAIAPIConnectionError,
            OpenAPITimeoutError,
            OpenAIInternalServerError,
            KeyError,
        ) as retryable_error:
            # This block catches errors designated for retry by the decorator.

            if isinstance(retryable_error, KeyError) and str(retryable_error) == "'choices'":
                # Graceful warning for KeyError ('choices') before retry, no full stack trace here.
                logger.warning(
                    f"LLM Response: Encountered KeyError '{str(retryable_error)}' (likely malformed API response missing"
                    f" 'choices'). Params (preview): {str(params)[:200]}... Will attempt retry."
                )
            elif isinstance(
                retryable_error, (OpenAIRateLimitError, OpenAIAPIConnectionError, OpenAPITimeoutError, OpenAIInternalServerError)
            ):
                # Existing logging for other specific OpenAI transient errors
                request_id = getattr(retryable_error, "request_id", None)
                headers = getattr(retryable_error, "headers", {})
                azure_request_id = headers.get("x-request-id") or headers.get("x-ms-client-request-id")  # Adjust as needed

                logger.warning(
                    f"LLM Response: Transient OpenAI error: {type(retryable_error).__name__} - {str(retryable_error)}. "
                    f"SDK_RequestID: {request_id or 'N/A'}, AzureRequestID: {azure_request_id or 'N/A'}. "
                    f"Params (preview): {str(params)[:200]}... Will attempt retry."
                )
            else:  # Should not be reached if types in decorator match types here, but as a fallback
                logger.warning(
                    f"LLM Response: Encountered retryable error: {type(retryable_error).__name__} - {str(retryable_error)}. "
                    f"Params (preview): {str(params)[:200]}... Will attempt retry."
                )
            raise  # Re-raise for tenacity to handle the retry

        except Exception as e:
            # This block is for non-retryable errors, or if all retries are exhausted
            # and tenacity re-raises an error not caught above (though reraise=True makes it re-raise the original).
            error_type = type(e).__name__
            error_message = str(e)
            status_code = None
            response_text = None
            azure_request_id = None

            # Try to extract details if 'e' is an API error with a response attribute
            if hasattr(e, "response") and e.response is not None:
                status_code = getattr(e.response, "status_code", None)
                response_text = getattr(e.response, "text", None)
                headers = getattr(e.response, "headers", {})
                azure_request_id = headers.get("x-request-id") or headers.get("x-ms-client-request-id")
            elif hasattr(e, "headers"):  # Some OpenAI errors might have headers directly
                headers = getattr(e, "headers", {})
                azure_request_id = headers.get("x-request-id") or headers.get("x-ms-client-request-id")

            log_message_parts = [
                f"Error getting LLM response (non-retryable or retries exhausted). Type: {error_type}, Msg: '{error_message}'",
                f"Status: {status_code or 'N/A'}",
                f"AzureRequestID: {azure_request_id or 'N/A'}",
                f"Params (preview): {str(params)[:200]}...",
                f"API Resp (preview): {str(response_text)[:200] if response_text else 'N/A'}...",
            ]
            # Log with full stack trace for these truly unexpected/final errors
            logger.error(", ".join(log_message_parts), exc_info=True)

            raise InvoiceProcessingError(
                message=(
                    f"Failed LLM response: {error_type} - {error_message}{f' (Status: {status_code})' if status_code else ''}"
                ),
                original_exception=e,
                status_code=status_code,
                params_at_error=params,
            )

    @retry(
        retry=retry_if_exception_type(
            (OpenAIRateLimitError, OpenAIAPIConnectionError, OpenAPITimeoutError, OpenAIInternalServerError)
        ),
        wait=wait_exponential(multiplier=2, min=2, max=30),  # Wait 2s, 4s, 8s... up to 30s
        stop=stop_after_attempt(5),  # Stop after 5 attempts
        reraise=True,
        before_sleep=lambda rs: logger.warning(
            f"Structured Response: Retrying {type(rs.outcome.exception()).__name__} "
            f"in {rs.next_action.sleep:.2f}s (Attempt {rs.attempt_number}). "
            f"Error: {str(rs.outcome.exception())}"
        ),
    )
    async def get_structured_response(self, prompt: str, output_model: Type[BaseModel], client_type: LLMClientType) -> BaseModel:
        """
        Calls a specified Azure OpenAI model and forces its output to conform to a Pydantic schema.
        This is a generic method that can be used with any configured LLM client.

        Args:
            prompt (str): The user prompt to send to the model.
            output_model (Type[BaseModel]): The Pydantic model class defining the desired output schema.
            client_type (LLMClientType): The enum member specifying which client to use for the call.

        Returns:
            BaseModel: An instantiated Pydantic model with the data from the LLM's response.
        """
        # 1. Select the correct client based on the provided client_type
        llm_client = self.client_map.get(client_type)
        if not llm_client:
            # Immediate failure, no retry needed
            raise ValueError(f"Invalid client_type specified: {client_type}. No client configured.")

        try:
            # 2. Create the structured output chain using the selected client
            structured_llm_chain = llm_client.with_structured_output(schema=output_model, method="function_calling", strict=True)

            # 3. Invoke the chain
            # Note: Using invoke (sync) inside async def.
            response_model_instance = structured_llm_chain.invoke([HumanMessage(content=prompt)])

            return response_model_instance

        except (
            OpenAIRateLimitError,
            OpenAIAPIConnectionError,
            OpenAPITimeoutError,
            OpenAIInternalServerError,
        ) as transient_error:
            # Capture transient errors to log context, then re-raise so @retry can handle them
            logger.warning(
                f"Structured Response: Transient error for {output_model.__name__}: {str(transient_error)}. Triggering retry..."
            )
            raise

        except Exception as e:
            # Capture non-retryable errors, wrap them, and fail immediately
            error_message = (
                f"Failed to get structured response for schema {output_model.__name__} using client {client_type.name}: {str(e)}"
            )
            logger.error(error_message, exc_info=True)
            raise InvoiceProcessingError(message=error_message, original_exception=e)

    @retry_with_tiered_strategy
    def get_embeddings(self, text_list):
        """
        Generate embeddings for a list of text inputs using Azure OpenAI embeddings.
        Retry logic is handled by the @retry_with_tiered_strategy decorator.

        Args:
            text_list (List[str]): List of text strings to generate embeddings for.

        Returns:
            List[List[float]]: A list of embedding vectors.

        Raises:
            Various OpenAI exceptions (e.g., OpenAIRateLimitError) if retries fail.
        """
        # Pre-processing steps:
        if not text_list:
            logger.warning("get_embeddings called with an empty or None text_list.")
            return []
        if not isinstance(text_list, list):
            # This makes the method flexible for single string inputs.
            text_list = [text_list]

        processed_text_list = [str(text) if text is not None else "" for text in text_list]

        # Core API call.
        embeddings = self.embeddings_client.embed_documents(processed_text_list)
        return embeddings

    def cosine_similarity(self, input1, input2):
        """
        Calculate the cosine similarity between two inputs which can be either text strings or embedding vectors.

        Args:
            input1: Either a string or an embedding vector (list/array of floats).
            input2: Either a string or an embedding vector (list/array of floats).

        Returns:
            float: Cosine similarity between the two inputs (range from -1 to 1, where 1 means identical).

        Raises:
            InvoiceProcessingError: If cosine similarity calculation fails.
        """
        try:
            # Check if both inputs are strings (need embeddings)
            if isinstance(input1, str) and isinstance(input2, str):
                # Process both texts in a single API call
                embeddings = self.get_embeddings([input1, input2])
                embedding1, embedding2 = embeddings
            else:
                # Handle mixed case (one embedding, one text) or both embeddings
                embedding1 = input1
                embedding2 = input2

                # If input1 is a string, get its embedding
                if isinstance(input1, str):
                    embedding1 = self.get_embeddings(input1)[0]

                # If input2 is a string, get its embedding
                if isinstance(input2, str):
                    embedding2 = self.get_embeddings(input2)[0]

            # Ensure embeddings are numpy arrays for cosine calculation
            embedding1 = np.array(embedding1)
            embedding2 = np.array(embedding2)

            # Calculate cosine similarity
            # Note: `cosine` from scipy calculates distance, so we subtract it from 1 to get similarity
            similarity = 1 - cosine(embedding1, embedding2)

            return similarity

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            raise InvoiceProcessingError(f"Failed to calculate cosine similarity: {str(e)}")
