import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from config import Config

# Configuration
config = Config()
endpoint = config.AZ_SEARCH_API_ENDPOINT_URL  # e.g., https://your-service.search.windows.net
index_name = config.AZ_SEARCH_INDEX_NAME
api_key = config.AZ_SEARCH_API_KEY

# Create a test vector (1536 dimensions to match your schema)
test_vector = np.random.rand(config.AOAI_EMBEDDING_DIMENSIONS).astype(np.float32).tolist()

# Initialize the search client
credential = AzureKeyCredential(api_key)
search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

# Create a vectorized query - note the field name is a string, not a list
vector_query = VectorizedQuery(
    vector=test_vector, fields="ItemDescription_vector", k_nearest_neighbors=5  # This is correct as a list of strings
)

# Execute the search
try:
    results = search_client.search(
        search_text=None,  # No text search
        vector_queries=[vector_query],  # Note: This is a list of VectorizedQuery objects
        select=["DescriptionID", "MfrPartNumExact", "ItemDescription"],
        top=5,
    )

    # Process results
    print("Search results:")
    result_count = 0
    for result in results:
        result_count += 1
        print(f"Score: {result['@search.score']}")
        print(f"DescriptionID: {result['DescriptionID']}")
        print(f"MfrPartNumExact: {result['MfrPartNumExact']}")
        print(f"ItemDescription: {result['ItemDescription']}")
        print("---")

    print(f"Total results: {result_count}")

except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e)}")
