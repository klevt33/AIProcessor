from config import Config  # Import Config from the correct module
from llm import LLM  # Import the LLM class from your module (adjust the import as needed)

# Initialize config using the provided method
config = Config()  # Create the config object using Config()
llm = LLM(config)  # Initialize the LLM class with the config

# Example list of texts for which to generate embeddings
text_list = ["This is the first sentence.", "Here is the second sentence.", "This is the third one."]

# Test the get_embeddings function
embeddings = llm.get_embeddings(text_list)

# Check if embeddings were successfully generated
if embeddings:
    for i, embedding in enumerate(embeddings):
        print(f"Embedding {i + 1}: Length = {len(embedding)}")
else:
    print("Failed to generate embeddings.")
