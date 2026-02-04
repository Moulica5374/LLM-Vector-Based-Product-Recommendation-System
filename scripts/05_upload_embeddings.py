import os
import pandas as pd
from pinecone import Pinecone
from tqdm import tqdm

S3_PATH = "s3://recommendation-system-llms/embeddings/products_embeddings.parquet"
INDEX_NAME = "beauty-products"
BATCH_SIZE = 200
CHUNK_SIZE = 10000  # Process 10k rows at a time

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

print("Starting chunked upload to Pinecone...")

# Read and process in chunks to avoid memory issues
parquet_file = pd.read_parquet(S3_PATH, engine='pyarrow')
total_rows = len(parquet_file)
print(f"Total vectors to upload: {total_rows}")

uploaded = 0

for chunk_start in range(0, total_rows, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, total_rows)
    print(f"\nProcessing chunk {chunk_start} to {chunk_end}...")
    
    # Get chunk
    df_chunk = parquet_file.iloc[chunk_start:chunk_end]
    
    # Prepare vectors for this chunk
    vectors = []
    for _, row in df_chunk.iterrows():
        vec = row.drop("parent_asin").tolist()
        vectors.append((str(row["parent_asin"]), vec))
    
    # Upload in batches
    for i in tqdm(range(0, len(vectors), BATCH_SIZE), desc=f"Uploading chunk"):
        batch = vectors[i:i+BATCH_SIZE]
        index.upsert(vectors=batch)
        uploaded += len(batch)
    
    print(f"Progress: {uploaded}/{total_rows} vectors uploaded")

print("\nDone uploading to Pinecone.")
print("Index stats:", index.describe_index_stats())