"""
Module: agents.py

Purpose:
This module provides the `Agents` class, responsible for interacting with Azure AI Project agents.
It is a core component for leveraging AI capabilities, presumably for tasks related to
invoice line item matching, such as data extraction, interpretation, and complex matching logic.
The module handles the lifecycle of AI agents, including their creation (with or without
Bing search integration for external data grounding) and execution of prompts.

High-Level Design:
- Agent Interaction: Utilizes the `azure-ai-projects` SDK to communicate with the
  Azure AI Project service.
- Agent Management: Provides methods to get existing agents or create new ones,
  configured with specific system prompts, models, and tools (like Bing Search).
- Asynchronous Operations: Employs `asyncio` for non-blocking agent calls,
  making it suitable for I/O-bound operations and improving overall application
  responsiveness. Blocking SDK calls are wrapped using `asyncio.to_thread`.
- Robust Retry Mechanism: Implements a sophisticated retry strategy for agent runs
  (`_execute_run_with_retry_async`) to handle transient issues like rate limits,
  temporary failures, and timeouts.
  - Dynamic Delays: Calculates wait times based on error responses (e.g., "try again in X seconds"
    from rate limit errors) and adds jitter.
  - Congestion Control: Features a mechanism to introduce a delay for the *first attempt*
    of an agent run if many other tasks are currently in a retry-wait state. This is managed
    by a global counter (`_SHARED_RETRY_WAITING_TASKS_COUNT`) and aims to prevent
    overwhelming the AI service during periods of high load or instability.
- Configuration-Driven: Relies on an external configuration object for Azure connection
  details, agent behavior (e.g., model names), and retry parameters.
- Error Handling: Includes comprehensive error handling and logging for issues
  encountered during agent creation, run execution, or result processing.

The `Agents` class encapsulates all this functionality, offering a streamlined interface
for other parts of the invoice processing solution to utilize AI agents.
"""

import asyncio
import random
import re
import threading
from typing import Any, Callable, Optional, Tuple

from azure.ai.projects.models import Agent as AzureAgent, BingGroundingTool, RunError, ThreadRun as AzureRun

from constants import Constants
from exceptions import InvoiceProcessingError
from logger import logger

# --- Global (Module-Level) State for Retry Count ---
_SHARED_RETRY_WAITING_TASKS_COUNT = 0
_SHARED_RETRY_COUNT_LOCK = threading.Lock()


# --- END: Global (Module-Level) State ---
class Agents:

    def __init__(self, config: Any):
        """
        Initializes the Agents client with configuration settings.

        This constructor sets up the Azure AI Project client, configures retry parameters
        for agent calls (maximum retries, wait times, timeouts), and initializes
        the Bing grounding tool if specified in the configuration. It also reads
        parameters for calculating delays on the first attempt of an agent run,
        based on current congestion.

        Args:
            config (Any): A configuration object containing settings such as
                          Azure connection strings (AZ_AGENT_PROJECT_CONNECTION_STRING),
                          Bing connection name (AZ_AGENT_GBS_BING_CONNECTION_NAME),
                          and 'agent_retry_settings' dictionary with keys like
                          'max_attempts', 'default_wait_seconds', 'attempt_timeout_seconds',
                          'first_attempt_congestion_multiplier_sec', and
                          'first_attempt_max_congestion_delay_sec'.
        """
        self.config = config
        retry_settings_dict = getattr(config, "agent_retry_settings", {})
        self.max_agent_retries = retry_settings_dict.get("max_attempts", 3)
        self.default_retry_wait_seconds = retry_settings_dict.get("default_wait_seconds", 60)
        self.agent_call_timeout_seconds = retry_settings_dict.get("attempt_timeout_seconds", 180)

        # --- START: New parameters for first attempt delay calculation ---
        self.first_attempt_congestion_multiplier_sec = retry_settings_dict.get("first_attempt_congestion_multiplier_sec", 10)
        self.first_attempt_max_congestion_delay_sec = retry_settings_dict.get("first_attempt_max_congestion_delay_sec", 60)
        # self.first_attempt_random_delay_min_sec = retry_settings_dict.get('first_attempt_random_delay_min_sec', 1)
        # self.first_attempt_random_delay_max_sec = retry_settings_dict.get('first_attempt_random_delay_max_sec', 5)
        # --- END: New parameters ---

        logger.debug(
            f"Agent Retry Settings: Max Attempts={self.max_agent_retries}, "
            f"Default Wait={self.default_retry_wait_seconds}s, "
            f"Attempt Timeout={self.agent_call_timeout_seconds}s, "
            f"First Attempt Congestion Multiplier={self.first_attempt_congestion_multiplier_sec}s, "
            f"First Attempt Max Congestion Delay={self.first_attempt_max_congestion_delay_sec}s"
            # f"First Attempt Random Delay=[{self.first_attempt_random_delay_min_sec}-{self.first_attempt_random_delay_max_sec}]s"
        )

        self.azure_project_client = config.azure_clients.azure_project_client
        self.azure_bing_connection = self.azure_project_client.connections.get(
            connection_name=config.AZ_AGENT_GBS_BING_CONNECTION_NAME
        )
        self.bing_tool = BingGroundingTool(connection_id=self.azure_bing_connection.id)

    # --- Generic retry wrapper for most SDK calls ---
    async def _execute_sdk_call_with_retry_async(
        self, sdk_call_func: Callable, *args: Any, operation_name: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """
        Executes a generic Azure SDK call with a simple asynchronous retry mechanism.

        Handles:
        - Retries up to `self.max_agent_retries` times.
        - Timeouts for each individual call attempt (`self.agent_call_timeout_seconds`).
        - Fixed sleep interval (`self.default_retry_wait_seconds`) between retries.
        - `asyncio.TimeoutError` is NOT retried and is re-raised immediately.
        - Does NOT use congestion delay or manage _SHARED_RETRY_WAITING_TASKS_COUNT.

        Args:
            sdk_call_func (Callable): The SDK function to call.
            *args: Positional arguments for the SDK call.
            operation_name (Optional[str]): A descriptive name for the operation for logging.
            **kwargs: Keyword arguments for the SDK call.

        Returns:
            Any: The result of the successful SDK call.

        Raises:
            asyncio.TimeoutError: If an attempt times out.
            Exception: If an unexpected error occurs and all retries are exhausted.
        """
        last_exception = None
        op_name = operation_name or getattr(sdk_call_func, "__name__", "Unnamed SDK Call")

        for attempt in range(self.max_agent_retries):
            logger.debug(f"SDK Call Attempt {attempt + 1}/{self.max_agent_retries} for '{op_name}'")
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(sdk_call_func, *args, **kwargs), timeout=self.agent_call_timeout_seconds
                )
                logger.debug(f"SDK call '{op_name}' successful on attempt {attempt + 1}")
                return result
            except asyncio.TimeoutError:
                logger.error(f"SDK call '{op_name}' failed due to timeout on attempt {attempt + 1}. Not retrying TimeoutError.")
                raise  # Re-raise immediately, do not retry
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"SDK call '{op_name}' attempt {attempt + 1} failed with {type(e).__name__}: {str(e)}. "
                    "Preparing for retry if attempts remain."
                )

                if attempt + 1 < self.max_agent_retries:
                    logger.debug(
                        f"Waiting {self.default_retry_wait_seconds:.2f} seconds before "
                        f"retry attempt {attempt + 2} for '{op_name}'..."
                    )
                    await asyncio.sleep(self.default_retry_wait_seconds + random.uniform(0, 5.0))
                else:
                    logger.error(
                        f"SDK call '{op_name}' failed after {self.max_agent_retries} attempts. Last exception: "
                        f"{type(last_exception).__name__ if last_exception else 'N/A'}."
                    )

        if last_exception:
            # Add context if it's a generic exception from the SDK
            raise InvoiceProcessingError(
                f"SDK call '{op_name}' failed after {self.max_agent_retries} retries: {last_exception}"
            ) from last_exception
        else:
            # Should not be reached if max_agent_retries > 0 and an exception always occurs on failure.
            raise InvoiceProcessingError(
                f"SDK call '{op_name}' failed definitively after {self.max_agent_retries} attempts without a "
                "specific final exception."
            )

    # --- Internal method containing the core SDK call and retry logic ---
    async def _execute_run_with_retry_async(self, thread_id: str, agent_id: str) -> AzureRun:
        """
        Executes an agent run with a comprehensive asynchronous retry mechanism.

        This internal method attempts to create and process an agent run. It handles:
        - Retries up to `self.max_agent_retries` times.
        - Timeouts for each individual agent call attempt (`self.agent_call_timeout_seconds`).
        - Dynamic sleep intervals between retries, including parsing "try again in X seconds"
          from rate limit error messages and adding jitter.
        - Congestion-based delay for the first attempt: Before the first attempt, it calculates
          a delay based on `_SHARED_RETRY_WAITING_TASKS_COUNT` to avoid overwhelming
          the service if many tasks are already retrying.
        - Increments/decrements a global counter (`_SHARED_RETRY_WAITING_TASKS_COUNT`)
          when a task enters a retry-wait state and after it completes the wait.
        - Logs detailed information about each attempt, status, and error.

        Args:
            thread_id (str): The ID of the Azure AI agent thread to use for the run.
            agent_id (str): The ID of the Azure AI agent to run.

        Returns:
            azure.ai.projects.models.Run: The Azure AI Run object representing the outcome
                                          of the agent execution. This object is returned
                                          even if the run ultimately failed after retries,
                                          allowing callers to inspect the final status and error.

        Raises:
            asyncio.TimeoutError: If an attempt times out and timeouts are non-retriable by design
                                  in this function (it breaks the loop on TimeoutError).
            Exception: If an unexpected, non-retriable error occurs, or if all retries are
                       exhausted and the last attempt resulted in an exception rather than a
                       Run object with a 'failed' status.
            InvoiceProcessingError: If the run definitively fails after all retries without a
                                    clear run object or specific exception being propagated from
                                    the last attempt.
        """
        global _SHARED_RETRY_WAITING_TASKS_COUNT

        last_run_object = None
        last_exception = None

        for attempt in range(self.max_agent_retries):
            if attempt == 0:
                current_retry_tasks = 0
                with _SHARED_RETRY_COUNT_LOCK:
                    current_retry_tasks = _SHARED_RETRY_WAITING_TASKS_COUNT

                # 1. Calculate a fixed delay based on the counter
                congestion_based_delay = min(
                    current_retry_tasks * self.first_attempt_congestion_multiplier_sec,
                    self.first_attempt_max_congestion_delay_sec,
                )

                # 2. Add a randomized delay from 1 to 5 seconds
                random_component_delay = random.uniform(0, 5.0)

                total_first_attempt_delay = congestion_based_delay + random_component_delay

                if total_first_attempt_delay > 0:
                    logger.debug(
                        f"First attempt for [Thread: {thread_id}] (retrying tasks: {current_retry_tasks}). "
                        f"Calculated delay: {congestion_based_delay}s (congestion) + {random_component_delay:.2f}s "
                        f"(random) = {total_first_attempt_delay:.2f}s."
                    )
                    await asyncio.sleep(total_first_attempt_delay)
                else:
                    logger.debug(
                        f"First attempt for [Thread: {thread_id}] (retrying tasks: {current_retry_tasks}). No delay applied."
                    )

            logger.debug(f"Agent Run Attempt {attempt + 1}/{self.max_agent_retries} [Thread: {thread_id}, Agent: {agent_id}]")
            wait_time_for_next_attempt = None

            try:
                run = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.azure_project_client.agents.create_and_process_run, thread_id=thread_id, agent_id=agent_id
                    ),
                    timeout=self.agent_call_timeout_seconds,
                )
                last_run_object = run
                if run.status == Constants.COMPLETED_lower:
                    logger.debug(f"Agent run successful on attempt {attempt + 1} [Thread: {thread_id}]")
                    return run
                elif run.status in (Constants.FAILED_lower, Constants.INCOMPLETE_lower):
                    wait_time_for_next_attempt = self.default_retry_wait_seconds + random.uniform(0, 5.0)
                    log_reason = "default"
                    run_details = run.last_error
                    if isinstance(run_details, RunError) and run_details.code == "rate_limit_exceeded":
                        error_message = run_details.message or ""
                        match = re.search(r"try again in (\d+)\s*seconds?", error_message, re.IGNORECASE)
                        if match:
                            try:
                                parsed_seconds = int(match.group(1))
                                distance_from_end = max(0, (self.max_agent_retries - 2) - attempt)
                                jitter_min = distance_from_end * 6 + 1
                                jitter_max = distance_from_end * 6 + 5
                                jitter = random.uniform(jitter_min, jitter_max)
                                wait_time_for_next_attempt = parsed_seconds + jitter
                                log_reason = (
                                    f"parsed ({parsed_seconds}s + {jitter:.2f}s jitter [{jitter_min}-{jitter_max}s range])"
                                )
                            except (ValueError, IndexError):
                                logger.warning(
                                    f"Rate limit: Failed parse wait seconds from '{error_message}'. Using default wait. "
                                    f"[Thread: {thread_id}]"
                                )
                                log_reason = "rate_limit_exceeded, parse failed"
                        else:
                            logger.warning(
                                f"Rate limit: Msg missing wait time '{error_message}'. Using default wait. [Thread: {thread_id}]"
                            )
                            log_reason = "rate_limit_exceeded, no time found"
                    else:
                        error_code = getattr(run_details, "code", "Unknown") if run_details else "Unknown"
                        logger.warning(
                            f"Run failed (code: '{error_code}', details: {run_details}). Using default wait. "
                            f"[Thread: {thread_id}]"
                        )
                        log_reason = f"failed code '{error_code}'"
                    logger.warning(f"Agent run attempt {attempt + 1} failed ({run.status}, {log_reason}). [Thread: {thread_id}]")
                else:
                    logger.warning(
                        f"Agent run attempt {attempt + 1} yielded status '{run.status}'. Not retrying by default. "
                        f"[Thread: {thread_id}]"
                    )
                    return run
            except asyncio.TimeoutError:
                last_exception = asyncio.TimeoutError(
                    f"Attempt {attempt + 1} timed out after {self.agent_call_timeout_seconds}s for SDK call."
                )
                logger.error(
                    f"Agent run failed due to timeout on attempt {attempt + 1}. Not retrying TimeoutError. [Thread: {thread_id}]"
                )
                last_run_object = None
                # TimeoutError is non-retriable - break out of retry loop
                break
            except Exception as e:
                last_exception = e
                logger.error(
                    f"Unexpected error during agent run execution attempt {attempt + 1}: {type(e).__name__} - {str(e)} "
                    f"[Thread: {thread_id}]",
                    exc_info=False,
                )
                last_run_object = None
                wait_time_for_next_attempt = self.default_retry_wait_seconds + random.uniform(0, 5.0)

            if wait_time_for_next_attempt is not None:
                if attempt + 1 < self.max_agent_retries:
                    logger.debug(
                        f"Waiting {wait_time_for_next_attempt:.2f} seconds before retry attempt {attempt + 2}... "
                        f"[Thread: {thread_id}]"
                    )

                    incremented_here = False
                    try:
                        with _SHARED_RETRY_COUNT_LOCK:
                            _SHARED_RETRY_WAITING_TASKS_COUNT += 1
                            incremented_here = True
                            logger.debug(
                                f"[Thread: {thread_id}] Incremented global _SHARED_RETRY_WAITING_TASKS_COUNT to "
                                f"{_SHARED_RETRY_WAITING_TASKS_COUNT} before sleep for attempt {attempt + 2}."
                            )

                        await asyncio.sleep(wait_time_for_next_attempt)
                    finally:
                        if incremented_here:
                            with _SHARED_RETRY_COUNT_LOCK:
                                _SHARED_RETRY_WAITING_TASKS_COUNT -= 1
                                if _SHARED_RETRY_WAITING_TASKS_COUNT < 0:
                                    logger.warning(
                                        f"[Thread: {thread_id}] Global _SHARED_RETRY_WAITING_TASKS_COUNT fell below zero "
                                        f"({_SHARED_RETRY_WAITING_TASKS_COUNT}), resetting to 0."
                                    )
                                    _SHARED_RETRY_WAITING_TASKS_COUNT = 0
                                logger.debug(
                                    f"[Thread: {thread_id}] Decremented global _SHARED_RETRY_WAITING_TASKS_COUNT to "
                                    f"{_SHARED_RETRY_WAITING_TASKS_COUNT} after sleep."
                                )
                else:
                    logger.error(
                        f"Agent run failed after {self.max_agent_retries} attempts (last status: "
                        f"{last_run_object.status if last_run_object else 'N/A'}, last exception: "
                        f"{type(last_exception).__name__ if last_exception else 'N/A'}). [Thread: {thread_id}]"
                    )
            else:
                logger.debug(f"Exiting retry loop after attempt {attempt + 1}. [Thread: {thread_id}]")
                break

        if last_run_object and last_run_object.status == Constants.COMPLETED_lower:
            return last_run_object
        elif last_exception:
            logger.error(f"Agent run ultimately failed due to exception: {last_exception}. [Thread: {thread_id}]")
            raise last_exception
        elif last_run_object:
            logger.error(
                f"Agent run ultimately failed after max retries. Returning last run object (Status: {last_run_object.status}). "
                f"[Thread: {thread_id}]"
            )
            return last_run_object
        else:
            logger.error(
                "Agent run ultimately failed without a final run object or specific exception information at this stage "
                f"(it was logged per attempt). [Thread: {thread_id}]"
            )
            raise InvoiceProcessingError(
                f"Agent run failed definitively after {self.max_agent_retries} attempts for thread {thread_id}."
            )

    # --- Making agent retrieval async ---
    async def _check_if_agent_exists_async(self, agent_name: str) -> Optional[AzureAgent]:
        """
        Asynchronously checks if an agent with the given name already exists.

        It lists all agents in the project and iterates through them to find a match
        by name. If found, it retrieves the full agent object.

        Args:
            agent_name (str): The name of the agent to check for.

        Returns:
            Optional[azure.ai.projects.models.Agent]: The agent object if found, otherwise None.
                                                       Returns None also if an error occurs during
                                                       the check.
        """
        ai_agent = None
        try:
            # Using the new simplified retry wrapper
            existing_agents_response = await self._execute_sdk_call_with_retry_async(
                self.azure_project_client.agents.list_agents, operation_name="list_agents"
            )
            # Assuming existing_agents_response is the direct response object with a 'data' attribute
            # Based on original code: for agent in existing_agents.data:
            for agent_in_list in existing_agents_response.data:
                if agent_in_list.name == agent_name:
                    agent_id = agent_in_list.id
                    ai_agent = await self._execute_sdk_call_with_retry_async(
                        self.azure_project_client.agents.get_agent,
                        agent_id,  # Positional argument for get_agent
                        operation_name="get_agent",
                    )
                    logger.debug(f"Reusing existing agent: {agent_id} (Name: {agent_name})")
                    break
        except asyncio.TimeoutError:
            logger.error(
                f"Timeout while checking if agent '{agent_name}' exists. This error is not retried further by this function."
            )
            raise  # Re-raise TimeoutError as it's non-retriable by the wrapper
        except Exception as e:
            # This catches errors from _execute_sdk_call_with_retry_async if all its retries fail,
            # or if the response structure from list_agents is unexpected.
            logger.error(f"Failed to check/get existing agent '{agent_name}' even after retries: {e}", exc_info=True)
            raise InvoiceProcessingError(f"Failed to check/get existing agent '{agent_name}' after retries: {e}") from e
        return ai_agent

    async def get_agent_async(self, agent_name: str, system_prompt: str, model_name: str) -> AzureAgent:
        """
        Asynchronously retrieves an existing agent by name or creates a new one if not found.

        This method first calls `_check_if_agent_exists_async`. If the agent doesn't exist,
        it proceeds to create a new standard agent with the specified name, system prompt,
        and model.

        Args:
            agent_name (str): The desired name for the agent.
            system_prompt (str): The system instructions/prompt for the agent.
            model_name (str): The model to be used by the agent (e.g., "gpt-4o").

        Returns:
            azure.ai.projects.models.Agent: The retrieved or newly created agent object.

        Raises:
            InvoiceProcessingError: If agent creation fails.
        """
        try:
            ai_agent = await self._check_if_agent_exists_async(agent_name)
            if ai_agent is None:
                logger.debug(f"Creating new agent: {agent_name}")
                # Using the new simplified retry wrapper
                ai_agent = await self._execute_sdk_call_with_retry_async(
                    self.azure_project_client.agents.create_agent,
                    model=model_name,
                    name=agent_name,
                    instructions=system_prompt,
                    headers={"x-ms-enable-preview": "true"},
                    operation_name="create_agent",
                )
                logger.debug(f"Created agent '{agent_name}', ID: {ai_agent.id}")
        except asyncio.TimeoutError:
            logger.error(f"Timeout during get_agent_async for '{agent_name}'.")
            raise
        except Exception as e:
            # Catches errors from _check_if_agent_exists_async or the create_agent call
            logger.error(f"Failed in get_agent_async for '{agent_name}' after all retries: {e}", exc_info=True)
            raise InvoiceProcessingError(f"Failed to get or create agent '{agent_name}' after retries: {e}") from e
        return ai_agent

    async def get_agent_with_bing_search_async(self, agent_name: str, system_prompt: str, model_name: str) -> AzureAgent:
        """
        Asynchronously retrieves an existing agent or creates a new one with Bing search enabled.

        Similar to `get_agent_async`, but the created agent will be configured with
        the Bing grounding tool, allowing it to perform web searches.

        Args:
            agent_name (str): The desired name for the agent.
            system_prompt (str): The system instructions/prompt for the agent.
            model_name (str): The model to be used by the agent (e.g., "gpt-4").

        Returns:
            azure.ai.projects.models.Agent: The retrieved or newly created agent object,
                                           configured with Bing search capabilities if created.

        Raises:
            InvoiceProcessingError: If agent creation fails.
        """
        try:
            ai_websearch_agent = await self._check_if_agent_exists_async(agent_name)
            if ai_websearch_agent is None:
                if not hasattr(self, "bing_tool"):  # Check if bing_tool was initialized
                    logger.error(f"Bing tool not initialized. Cannot create agent '{agent_name}' with Bing search.")
                    raise InvoiceProcessingError(
                        f"Bing tool not available for agent '{agent_name}'. Initialization might have failed."
                    )

                logger.debug(f"Creating new agent with Bing: {agent_name}")
                # Using the new simplified retry wrapper
                ai_websearch_agent = await self._execute_sdk_call_with_retry_async(
                    self.azure_project_client.agents.create_agent,
                    model=model_name,
                    name=agent_name,
                    instructions=system_prompt,
                    tools=self.bing_tool.definitions,
                    headers={"x-ms-enable-preview": "true"},
                    operation_name="create_agent_with_bing",
                )
                logger.debug(f"Created agent with Bing '{agent_name}', ID: {ai_websearch_agent.id}")
        except asyncio.TimeoutError:
            logger.error(f"Timeout during get_agent_with_bing_search_async for '{agent_name}'.")
            raise
        except Exception as e:
            logger.error(f"Failed in get_agent_with_bing_search_async for '{agent_name}' after all retries: {e}", exc_info=True)
            raise InvoiceProcessingError(f"Failed to get or create agent with Bing '{agent_name}' after retries: {e}") from e
        return ai_websearch_agent

    # --- Main run method is now async ---
    async def run(self, prompt: str, agent: AzureAgent) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Asynchronously runs a prompt against a specified agent, managing thread creation,
        message posting, and execution with retries.

        This method orchestrates the entire process of an agent interaction:
        1. Creates a new communication thread using the Azure AI SDK.
        2. Posts the user's prompt as a message within that thread.
        3. Invokes `_execute_run_with_retry_async` to run the agent against the thread,
           which handles the actual execution and retry logic.
        4. If the run completes successfully, it retrieves and parses messages from the
           thread to extract the assistant's response.
        5. Handles various run statuses (completed, failed, error) and logs appropriately.

        Args:
            prompt (str): The user prompt/query to send to the agent.
            agent (azure.ai.projects.models.Agent): The initialized Azure AI Agent object to use.

        Returns:
            Tuple[Optional[str], Optional[str], Optional[str]]: A tuple containing:
                - assistant_response (Optional[str]): The textual response from the assistant,
                  or None if the run failed, errored, or no response was found.
                - run_status (Optional[str]): The final status of the agent run
                  (e.g., "completed", "failed", "error"), or None if a critical
                  error occurred before the run status could be determined.
                - thread_id (Optional[str]): The ID of the Azure AI thread used for the
                  interaction, or None if thread creation failed.
        """
        thread_obj = None
        run_obj = None
        assistant_response = None
        run_status_val = None
        current_thread_id = "N/A"

        try:
            # 1. Create Thread
            thread_obj = await self._execute_sdk_call_with_retry_async(
                self.azure_project_client.agents.create_thread, operation_name="create_thread"
            )
            current_thread_id = thread_obj.id
            logger.debug(f"Created thread, ID: {current_thread_id}")

            # 2. Create Message
            message_obj = await self._execute_sdk_call_with_retry_async(
                self.azure_project_client.agents.create_message,
                thread_id=current_thread_id,
                role="user",
                content=prompt,
                operation_name="create_message",
            )
            logger.debug(f"Created message, ID: {message_obj.id} in thread {current_thread_id}")

            # 3. Execute Run (This handles the AI processing time)
            run_obj = await self._execute_run_with_retry_async(thread_id=current_thread_id, agent_id=agent.id)

            run_status_val = run_obj.status
            current_thread_id = run_obj.thread_id if hasattr(run_obj, "thread_id") and run_obj.thread_id else current_thread_id

            if run_status_val == Constants.COMPLETED_lower:
                # --- Race Condition Mitigation ---
                # Brief pause to ensure messages are indexed and available via the API
                # immediately after the run status flips to 'completed'.
                await asyncio.sleep(0.5)

                try:
                    # 4. List Messages
                    messages_response = await self._execute_sdk_call_with_retry_async(
                        self.azure_project_client.agents.list_messages,
                        thread_id=current_thread_id,
                        operation_name="list_messages",
                    )

                    messages_data = (
                        messages_response.data if hasattr(messages_response, "data") else messages_response.get("data", [])
                    )

                    # Sort by creation time to get the latest
                    sorted_messages = sorted(
                        messages_data,
                        key=lambda x: (x.get("created_at", 0) if isinstance(x, dict) else getattr(x, "created_at", 0)),
                    )

                    # Iterate backwards to find the latest assistant response efficiently
                    for msg_item in reversed(sorted_messages):
                        role = (
                            msg_item.get("role", "").lower()
                            if isinstance(msg_item, dict)
                            else getattr(msg_item, "role", "").lower()
                        )
                        if role == "assistant":
                            content_blocks = (
                                msg_item.get("content", []) if isinstance(msg_item, dict) else getattr(msg_item, "content", [])
                            )
                            if content_blocks and content_blocks[0].get("type") == "text":
                                text_content_dict = content_blocks[0].get("text", {})
                                assistant_response_val = text_content_dict.get("value")
                                if assistant_response_val is not None:
                                    assistant_response = assistant_response_val
                                    break

                    # Log warning if run completed but no response was found
                    if not assistant_response:
                        logger.warning(f"Run completed but no assistant text message found. [Thread: {current_thread_id}]")

                except asyncio.TimeoutError:
                    logger.error(f"Timeout listing messages after run completion. [Thread: {current_thread_id}]")
                    # We keep run_status_val as 'completed' for audit, but extraction will fail due to None response
                except Exception as msg_ex:
                    logger.error(
                        f"Failed to list messages after run completion: {msg_ex} [Thread: {current_thread_id}]", exc_info=True
                    )

            elif run_status_val == Constants.FAILED_lower:
                # --- CAPTURE AZURE ERROR DETAILS ---
                error_details_obj = run_obj.last_error
                error_code = getattr(error_details_obj, "code", "Unknown") if error_details_obj else "Unknown"
                error_msg = getattr(error_details_obj, "message", str(error_details_obj)) if error_details_obj else "No details"

                # Pass details back to caller
                assistant_response = f"Azure Run Failed. Code: {error_code}. Message: {error_msg}"

                logger.error(f"Run failed (Code: {error_code}) after retries: {error_details_obj} [Thread: {current_thread_id}]")

            else:
                assistant_response = f"Run ended with non-terminal status: {run_status_val}"
                logger.warning(f"Run finished with non-terminal status: {run_status_val} [Thread: {current_thread_id}]")

        # --- EXCEPTION HANDLING WITH RESPONSE CAPTURE ---
        except asyncio.TimeoutError as e:
            # Capture for Cosmos DB
            assistant_response = f"Timeout during agent run orchestration: {str(e)}"
            logger.error(f"Timeout during agent run orchestration: {str(e)} [Thread: {current_thread_id}]", exc_info=False)
            if not run_status_val:
                run_status_val = Constants.ERROR_lower

        except InvoiceProcessingError as ipe:
            # Capture for Cosmos DB
            assistant_response = f"InvoiceProcessingError: {str(ipe)}"
            logger.error(
                f"InvoiceProcessingError during agent run orchestration: {str(ipe)} [Thread: {current_thread_id}]",
                exc_info=True if "SDK call" in str(ipe) else False,
            )
            if not run_status_val:
                run_status_val = Constants.ERROR_lower

        except Exception as e:
            # Capture for Cosmos DB
            assistant_response = f"Generic exception: {type(e).__name__} - {str(e)}"
            logger.error(
                f"Generic exception during agent run orchestration: {type(e).__name__} - {str(e)} [Thread: {current_thread_id}]",
                exc_info=True,
            )
            if not run_status_val:
                run_status_val = Constants.ERROR_lower

        return (assistant_response, run_status_val, current_thread_id if current_thread_id != "N/A" else None)
