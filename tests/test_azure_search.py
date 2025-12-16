import asyncio

from azure_search_utils import AzureSearchUtils
from config import Config
from llm import LLM


async def test_search_scenarios():
    """
    Test script to demonstrate different search scenarios with Azure AI Search
    """
    # Initialize configuration and search utilities
    config = Config()
    search_utils = AzureSearchUtils(config)
    llm = LLM(config)

    print("===== Azure AI Search Testing =====")

    # Test 1: Exact matching on DescriptionID
    print("\n\n===== Test 1: Exact matching on DescriptionID =====")
    filter_expr = "DescriptionID eq '28696'"
    print(f"Query parameters: filter_expression='{filter_expr}', query_text='*'")

    # Using '*' as query_text to satisfy the API requirement
    results = search_utils.search(
        query_text="*",  # Wildcard query to match everything
        filter_expression=filter_expr,
        select=["DescriptionID", "MfrPartNumExact", "ItemDescription"],
        top=3,
    )

    print(f"Results ({len(results)} items):")
    print(results)

    # Test 2: Exact matching on MfrPartNumExact
    print("\n\n===== Test 2: Exact matching on MfrPartNumExact =====")
    query_text = "Q402018ABI"
    print(f"Query parameters: query_text='{query_text}', query_type='simple'")

    results = search_utils.search(query_text=query_text, select=["DescriptionID", "MfrPartNumExact", "ItemDescription"], top=3)

    print(f"Results ({len(results)} items):")
    print(results)

    # Test 3: Prefix search on MfrPartNumPrefix (searching for "Q40")
    print("\n\n===== Test 3: Prefix search on MfrPartNumPrefix =====")
    query_text = "Q40"
    print(f"Query parameters: query_text='{query_text}', search_fields=['MfrPartNumPrefix'], queryType='simple'")

    results = search_utils.search(
        query_text=query_text,
        search_fields=["MfrPartNumPrefix"],  # Explicitly specify to search only in this field
        select=["DescriptionID", "MfrPartNumExact", "MfrPartNumPrefix", "ItemDescription"],
        top=5,
    )

    print(f"Results ({len(results)} items):")
    print(results)

    # Test 4: Search on ItemDescription
    print("\n\n===== Test 4: Search on ItemDescription =====")
    query_text = "ENCL"
    print(f"Query parameters: query_text='{query_text}', queryType='full'")

    results = search_utils.search(
        query_text=query_text, query_type="full", select=["DescriptionID", "MfrPartNumExact", "ItemDescription"], top=5
    )

    print(f"Results ({len(results)} items):")
    print(results)

    # Test 5: Vector search on ItemDescription_vector
    print("\n\n===== Test 5: Vector search on ItemDescription_vector =====")
    # Generate embedding for a search phrase
    query_phrase = "panel for enclosure"
    embeddings = llm.get_embeddings([query_phrase])

    if embeddings and len(embeddings) > 0:
        # KEY FIX: Use a string for the fields parameter, not a list
        vector_query = {
            "vector": embeddings[0],
            "fields": "ItemDescription_vector",  # String, not list
            "k_nearest_neighbors": 5,
            "weight": 1.0,
        }

        print(f"Query parameters: vector_query for phrase '{query_phrase}'")

        results = search_utils.search(
            vector_query=vector_query, select=["DescriptionID", "MfrPartNumExact", "ItemDescription"], top=5
        )

        print(f"Results ({len(results)} items):")
        print(results)
    else:
        print("Failed to generate embeddings for vector search")

    # Test 6: Hybrid search (text + vector)
    print("\n\n===== Test 6: Hybrid search (text + vector) =====")
    query_text = "panel"
    query_phrase = "panel for enclosure"
    embeddings = llm.get_embeddings([query_phrase])

    if embeddings and len(embeddings) > 0:
        vector_query = {
            "vector": embeddings[0],
            "fields": "ItemDescription_vector",  # String, not list
            "k_nearest_neighbors": 5,
            "weight": 0.5,
        }

        print(f"Query parameters: query_text='{query_text}', vector_query for phrase '{query_phrase}'")

        results = search_utils.search(
            query_text=query_text,
            vector_query=vector_query,
            select=["DescriptionID", "MfrPartNumExact", "ItemDescription"],
            top=5,
            vector_filter_mode="preFilter",
        )

        print(f"Results ({len(results)} items):")
        print(results)
    else:
        print("Failed to generate embeddings for hybrid search")

    print("\n\n===== Test 7: Filtering by MfrPartNumExact =====")
    filter_expr = "MfrPartNumExact eq 'Q402018ABI'"
    print(f"Query parameters: filter_expression='{filter_expr}', query_text='*'")

    results = search_utils.search(
        query_text="*", filter_expression=filter_expr, select=["DescriptionID", "MfrPartNumExact", "ItemDescription"], top=3
    )

    print(f"Results ({len(results)} items):")
    print(results)


async def main():
    await test_search_scenarios()


if __name__ == "__main__":
    asyncio.run(main())
