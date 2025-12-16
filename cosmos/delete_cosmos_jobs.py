from azure.cosmos import CosmosClient

# --- Cosmos DB Config ---
COSMOS_ENDPOINT = "https://aks-cdb-dev.documents.azure.com:443/"
COSMOS_KEY = "<REDACTED>"
DATABASE_NAME = "spend_report_dev"
# LOGS_CONTAINER_NAME = "ai_process_logs"
CONTAINER_NAME = "ai_jobs"

# --- Allowed IDs ---
keep_prefixes = {
    "11040358-59de-4ef9-80ac-61e03afff64d",
    "1e3a58a9-c63f-4002-97f5-ff7fc9202058",
    "3f19ff3f-50ba-40d9-9ac4-3f6fa8c5e62c",
    "490fff0f-b871-4e6c-be1c-f478443a2db9",
    "4fa8051a-7911-447b-a27e-c6cf948901c8",
    "516e5f66-aa80-407d-8bd4-fbadc35011f9",
    "56ff5867-19b2-4f72-bef1-728989abaee8",
    "7c36e6d7-f773-476c-a770-078d61c39143",
    "92148b7b-0577-4c2f-9709-1c348740df64",
    "9c12fd7a-4b9b-4120-a816-ffc4ee7a8caa",
    "de34a3f9-7222-4f69-bed2-1a14193e110b",
    "e8d4ce20-6e69-42dd-9803-84a03444fe91",
}
# --- Connect to Cosmos ---
client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
db = client.get_database_client(DATABASE_NAME)
container = db.get_container_client(CONTAINER_NAME)

# --- Query all pending items ---
query = "SELECT c.id, c.partitionKey FROM c WHERE c.status = 'pending'"
items = list(container.query_items(query=query, enable_cross_partition_query=True))

delete_count = 0
skip_count = 0

for item in items:
    item_id = item["id"]
    prefix = item_id.split("~")[0]
    if prefix not in keep_prefixes:
        container.delete_item(item["id"], partition_key=item["id"])
        delete_count += 1
    else:
        skip_count += 1

print(f"✅ Deleted {delete_count} pending items not matching prefixes.")
print(f"⏭️ Kept {skip_count} items with allowed prefixes.")


# client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
# db = client.get_database_client(DATABASE_NAME)
# container = db.get_container_client(CONTAINER_NAME)

# allowed_prefixes = [
#     "11040358-59de-4ef9-80ac-61e03afff64d",
#     "1e3a58a9-c63f-4002-97f5-ff7fc9202058",
#     "3f19ff3f-50ba-40d9-9ac4-3f6fa8c5e62c",
#     "490fff0f-b871-4e6c-be1c-f478443a2db9",
#     "4fa8051a-7911-447b-a27e-c6cf948901c8",
#     "516e5f66-aa80-407d-8bd4-fbadc35011f9",
#     "56ff5867-19b2-4f72-bef1-728989abaee8",
#     "7c36e6d7-f773-476c-a770-078d61c39143",
#     "92148b7b-0577-4c2f-9709-1c348740df64",
#     "9c12fd7a-4b9b-4120-a816-ffc4ee7a8caa",
#     "de34a3f9-7222-4f69-bed2-1a14193e110b",
#     "e8d4ce20-6e69-42dd-9803-84a03444fe91",
# ]

# result = container.scripts.execute_stored_procedure(
#     sproc="bulkDeletePendingNotMatchingPrefixes",
#     params=[allowed_prefixes],
#     partition_key=None  # or set partition key if needed
# )

# print(f"✅ Deleted {result['deletedCount']} documents (server-side)")
