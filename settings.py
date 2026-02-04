import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "beauty-products")
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")

# AWS
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "recommendation-system-llms")

# Processing
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 200))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 10000))

# Paths
S3_RAW_PATH = f"s3://{S3_BUCKET}/raw/"
S3_PROCESSED_PATH = f"s3://{S3_BUCKET}/processed/"
S3_EMBEDDINGS_PATH = f"s3://{S3_BUCKET}/embeddings/"

