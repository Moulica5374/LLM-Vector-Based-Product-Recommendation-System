from pinecone import Pinecone, ServerlessSpec
import os

# Configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")  # We'll set this in EC2
INDEX_NAME = "beauty-products"

print("Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

print(f"Creating index '{INDEX_NAME}'...")
pc.create_index(
    name=INDEX_NAME,
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

print("✓ Index created successfully!")
print(f"Index name: {INDEX_NAME}")
print(f"Dimensions: 1536")
print(f"Metric: cosine")