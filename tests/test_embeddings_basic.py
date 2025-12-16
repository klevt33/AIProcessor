from openai import AzureOpenAI

# Define your Azure OpenAI service details
azure_endpoint = "https://aks-ais-rpa-ai-dev.openai.azure.com/"  # Use the base endpoint URL
model_name = "text-embedding-3-large"
deployment = "spend-report-embedding-dev"  # Ensure this matches the deployment name in Azure Portal

api_version = "2024-12-01-preview"  # Ensure this matches the supported API version
api_key = "<REDACTED>"

# Initialize the AzureOpenAI client
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=azure_endpoint,  # Use the base endpoint URL
    api_key=api_key,  # Pass the API key directly
)

# Generate embeddings for the input phrases
try:
    response = client.embeddings.create(input=["first phrase", "second phrase", "third phrase"], model=deployment)

    # Process and print the embeddings
    for item in response.data:
        length = len(item.embedding)
        print(
            f"data[{item.index}]: length={length}, "
            f"[{item.embedding[0]}, {item.embedding[1]}, "
            f"..., {item.embedding[length - 2]}, {item.embedding[length - 1]}]"
        )
    print(response.usage)
except Exception as e:
    print(f"An error occurred: {e}")
