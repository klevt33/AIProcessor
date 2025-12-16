from config import Config
from llm import LLM
from utils import clean_description

# Initialize config and LLM objects
config = Config()
llm = LLM(config)

# Define two input texts
text1 = clean_description(
    "006E8P-31131-A3 373-COROS2-TBPAIL-06 6F TB FREEDM ONE INTERLOCKING OS2 SM 0.65/0.65/.5 DB/KM ARMORED PLENUM"
)
text2 = clean_description("FREEDM One Tight-Buffered, Interlocking Armored Cable, Plenum, 6 F, Single-mode (OS2)")

# Calculate cosine similarity
similarity_score = llm.cosine_similarity(text1, text2)

if similarity_score is not None:
    print(f"Cosine Similarity: {similarity_score:.4f}")
else:
    print("Failed to calculate cosine similarity.")
