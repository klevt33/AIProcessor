from unittest.mock import MagicMock, call

import pytest

# It's good practice to be able to import the class you're testing
from sql_writer import SqlWriterService


# Let's create a fixture that will be used by all our tests
@pytest.fixture
def mock_sql_writer(mocker):
    """
    Creates an instance of SqlWriterService with mocked dependencies.
    """
    # 1. Create mock objects for all dependencies
    mock_config = MagicMock()
    # Set the config values our code relies on
    mock_config.SQL_WRITER_MAX_WORKERS = 1
    mock_config.SQL_WRITER_BATCH_SIZE = 10
    mock_config.SQL_WRITER_POLL_INTERVAL = 1

    mock_sdp = MagicMock()
    mock_cdb = MagicMock()

    # 2. Patch external functions that we want to control
    # This prevents our tests from actually sleeping or running async code
    mocker.patch("sql_writer.time.sleep")
    mocker.patch("sql_writer.asyncio.run")

    # 3. Create the service instance with our mocks
    service = SqlWriterService(config=mock_config, sdp=mock_sdp, cdb=mock_cdb)

    # 4. Return the service and the mocks so we can use them in our tests
    return service, mock_sdp, mock_cdb


# A sample document that mimics what Cosmos DB would return
# We can reuse this for multiple tests
SAMPLE_DOC = {
    "id": "doc-123",
    "request_details": {"id": "req-abc"},
    "invoice_details_from_rpa": {"IVCE_DTL_UID": 987},
    "process_output": {"status": "success", "message": "All good"},
    "post-processing": {"duplicate_detail_uids": [654, 321]},
}


def test_process_single_document_happy_path(mock_sql_writer, mocker):
    """
    Tests the successful processing of a single document with duplicates.
    """
    # Arrange: Get the service and mocks from our fixture
    service, mock_sdp, mock_cdb = mock_sql_writer

    # We need to mock the update_invoice_detail_and_tracking_values_by_id function
    # because it's imported into the sql_writer module.
    mock_update_sql = mocker.patch("sql_writer.update_invoice_detail_and_tracking_values_by_id")

    # Act: Call the method we want to test
    service._process_single_document(SAMPLE_DOC)

    # Assert: Verify that the correct actions were taken

    # 1. Assert that the SQL update function was called for the parent and all duplicates
    expected_sql_calls = [
        # Call for the parent record
        call(sdp=mock_sdp, invoice_detail_id=987, stage_results=SAMPLE_DOC["process_output"]),
        # Call for the first duplicate
        call(
            sdp=mock_sdp,
            invoice_detail_id=654,
            stage_results=SAMPLE_DOC["process_output"],
            is_duplicate=True,
            parent_detail_id=987,
        ),
        # Call for the second duplicate
        call(
            sdp=mock_sdp,
            invoice_detail_id=321,
            stage_results=SAMPLE_DOC["process_output"],
            is_duplicate=True,
            parent_detail_id=987,
        ),
    ]
    # `assert_has_calls` checks that these specific calls were made. The `any_order=True` is
    # important because the order of duplicate processing isn't guaranteed.
    mock_update_sql.assert_has_calls(expected_sql_calls, any_order=True)
    assert mock_update_sql.call_count == 3

    # 2. Assert that the Cosmos DB document was patched to "committed"
    mock_cdb.patch_document.assert_called_once_with(
        mock_cdb.ai_logs_container,
        "doc-123",
        [
            {"op": "set", "path": "/sql_writer/status", "value": "committed"},
            {"op": "set", "path": "/sql_writer/committed_at", "value": mocker.ANY},
        ],  # mocker.ANY ignores the timestamp
        "req-abc",
        raise_on_error=True,
    )

    # 3. Assert that the circuit breaker success method was called
    assert service.circuit_breaker["state"] == "closed"
    assert service.circuit_breaker["consecutive_failures"] == 0


def test_process_single_document_fails_once_then_succeeds(mock_sql_writer, mocker):
    """
    Tests that a document is correctly scheduled for retry on the first failure.
    """
    # Arrange
    service, mock_sdp, mock_cdb = mock_sql_writer
    service.MAX_RETRIES = 3

    # Instead of mocking the inner function, we mock the asyncio.run wrapper
    # This gives us direct control over the outcome of each async operation.
    mock_async_run = mocker.patch("sql_writer.asyncio.run")
    mock_async_run.side_effect = [
        Exception("Simulated SQL Deadlock"),  # 1st call fails (in the first attempt)
        None,  # 2nd call succeeds (parent record, second attempt)
        None,  # 3rd call succeeds (duplicate 1)
        None,  # 4th call succeeds (duplicate 2)
    ]

    # --- ACTION 1: First processing attempt (will fail immediately) ---
    service._process_single_document(SAMPLE_DOC)

    # --- ASSERT 1: Verify that a retry was scheduled ---

    # 1a. Check that asyncio.run was attempted exactly once before failing
    mock_async_run.assert_called_once()

    # 1b. Check that the document was patched to "retry_scheduled"
    mock_cdb.patch_document.assert_called_once()
    call_args, _ = mock_cdb.patch_document.call_args
    patched_ops = call_args[2]
    patched_values = {op["path"]: op["value"] for op in patched_ops}

    assert patched_values.get("/sql_writer/status") == "retry_scheduled"
    assert patched_values.get("/sql_writer/retry_count") == 1

    # 1c. Verify the circuit breaker recorded one failure
    assert service.circuit_breaker["consecutive_failures"] == 1

    # --- ARRANGE 2: Prepare for the second, successful attempt ---
    doc_for_retry = SAMPLE_DOC.copy()
    doc_for_retry["sql_writer"] = {"retry_count": 1}

    # --- ACTION 2: Second processing attempt (will succeed) ---
    service._process_single_document(doc_for_retry)

    # --- ASSERT 2: Verify it was committed successfully ---

    # 2a. Check that asyncio.run was called 3 more times (for parent + 2 duplicates)
    # Total call count is 1 (from fail) + 3 (from success) = 4
    assert mock_async_run.call_count == 4

    # 2b. Check that the document was patched to "committed"
    # The last call to patch_document should be the "committed" status
    assert mock_cdb.patch_document.call_count == 2
    final_call_args, _ = mock_cdb.patch_document.call_args
    final_patched_ops = final_call_args[2]
    final_patched_values = {op["path"]: op["value"] for op in final_patched_ops}

    assert final_patched_values.get("/sql_writer/status") == "committed"

    # 2c. Verify the circuit breaker was reset
    assert service.circuit_breaker["consecutive_failures"] == 0


def test_process_single_document_exceeds_max_retries_becomes_poison_pill(mock_sql_writer, mocker):
    """
    Tests that a document that repeatedly fails is marked as a poison pill,
    and the failure of the final SQL status update is logged.
    """
    # Arrange
    service, mock_sdp, mock_cdb = mock_sql_writer
    service.MAX_RETRIES = 2

    # Mock asyncio.run to always fail. This will cover both the main processing
    # and the final "bulk_update" call inside the poison pill handler.
    mock_async_run = mocker.patch("sql_writer.asyncio.run")
    mock_async_run.side_effect = Exception("Permanent SQL Connection Error")

    # We no longer need to mock bulk_update_invoice_line_status directly,
    # as mocking asyncio.run covers it.

    # --- ACTION: Simulate repeated failures leading to poison pill ---
    doc = SAMPLE_DOC.copy()
    for i in range(service.MAX_RETRIES + 1):
        doc["sql_writer"] = {"retry_count": i}
        service._process_single_document(doc)

    # --- ASSERT ---

    # 1. Verify asyncio.run was called 4 times:
    #    - 3 times for the regular processing attempts.
    #    - 1 time for the final "set AI-ERROR" attempt inside the poison pill handler.
    assert mock_async_run.call_count == 4

    # 2. Verify the document was patched to "failed" in Cosmos DB on the last attempt.
    assert mock_cdb.patch_document.call_count == 3
    final_call_args, _ = mock_cdb.patch_document.call_args
    final_patched_ops = final_call_args[2]
    final_patched_values = {op["path"]: op["value"] for op in final_patched_ops}

    assert final_patched_values.get("/sql_writer/status") == "failed"
    assert "/sql_writer/failed_at" in final_patched_values

    # 3. CRITICAL: Verify the error from the final SQL update attempt was logged.
    #    Because our mock causes the 4th call to fail, this field MUST be present.
    assert "/sql_writer/final_error" in final_patched_values
    assert final_patched_values["/sql_writer/final_error"] == "Permanent SQL Connection Error"
