import uuid

import pytest

from cdb import CDB
from config import Config

# Mark all tests in this file as 'integration' tests
# pytestmark = pytest.mark.integration


def test_get_item_count_with_real_db():
    """
    Tests the get_item_count method by connecting to a real Cosmos DB,
    creating test data, counting it, and cleaning up afterwards.
    """
    # Arrange (Part 1): Setup connection and test data
    try:
        config = Config()
        cdb = CDB(config)
    except Exception as e:
        pytest.fail(f"Failed to connect to Cosmos DB. Ensure config is correct. Error: {e}")

    test_run_id = str(uuid.uuid4())
    number_of_docs_to_create = 3
    # This list will now hold tuples of (document_id, partition_key) for cleanup
    created_items = []

    print(f"\nIntegration Test: Using test_run_id: {test_run_id}")

    try:
        # Arrange (Part 2): Insert test documents with the correct partition key
        for i in range(number_of_docs_to_create):
            # --- START: MODIFIED BLOCK ---
            doc_id = f"{test_run_id}-{i}"
            # This now matches the partition key schema of your application
            partition_key_value = f"req-{test_run_id}"

            item_body = {
                "id": doc_id,
                "test_run_id": test_run_id,  # Custom field for easy querying
                "request_details": {"id": partition_key_value},  # The ACTUAL partition key
                "description": f"Test document {i} for get_item_count",
            }
            # --- END: MODIFIED BLOCK ---

            cdb.ai_logs_container.create_item(body=item_body)
            # Store both the id and the partition key for proper deletion
            created_items.append({"id": doc_id, "pk": partition_key_value})

        print(f"Integration Test: Successfully created {len(created_items)} test documents.")

        # Act: Call the method we are testing
        where_condition = f"WHERE c.test_run_id = '{test_run_id}'"
        count = cdb.get_item_count(cdb.ai_logs_container, where_condition)

        # Assert: Check if the count is correct
        assert count == number_of_docs_to_create
        print(f"Integration Test: Assertion successful! Count returned was {count}.")

    finally:
        # Teardown: Clean up the documents we created using the correct partition key.
        if not created_items:
            print("Integration Test: No documents to clean up.")
        else:
            print(f"Integration Test: Cleaning up {len(created_items)} documents...")
            for item in created_items:
                try:
                    # --- MODIFIED LINE ---
                    # Provide the correct partition_key value for the delete operation
                    cdb.ai_logs_container.delete_item(item=item["id"], partition_key=item["pk"])
                except Exception as e:
                    print(
                        f"ERROR: Failed to clean up document with id '{item['id']}'. Manual cleanup may be required. Error: {e}"
                    )
            print("Integration Test: Cleanup complete.")
