import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import gc

IN_PATH = "s3://recommendation-system-llms/processed_data/products_for_embedding.parquet"
OUT_PATH = "s3://recommendation-system-llms/embeddings/products_embeddings.parquet"
TEMP_DIR = "s3://recommendation-system-llms/embeddings/temp_chunks/"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

print("Loading data...")
df = pd.read_parquet(IN_PATH)
texts = df["text_for_embedding"].tolist()
parent_asins = df["parent_asin"].tolist()

BATCH_SIZE = 200
CHUNK_SIZE = 2000  # Write to S3 every 2000 embeddings to avoid memory issues

chunk_files = []
embeddings_buffer = []
asins_buffer = []
chunk_num = 0

print(f"Generating embeddings for {len(texts)} products in batches of {BATCH_SIZE}...")
print(f"Writing intermediate chunks every {CHUNK_SIZE} embeddings...")

for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i:i + BATCH_SIZE]
    batch_asins = parent_asins[i:i + BATCH_SIZE]
    
    # Get embeddings for this batch
    resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
    batch_embeddings = [item.embedding for item in resp.data]
    
    # Add to buffer
    embeddings_buffer.extend(batch_embeddings)
    asins_buffer.extend(batch_asins)
    
    # Write chunk when buffer is large enough or at the end
    if len(embeddings_buffer) >= CHUNK_SIZE or i + BATCH_SIZE >= len(texts):
        # Create dataframe for this chunk
        chunk_df = pd.DataFrame(embeddings_buffer)
        chunk_df["parent_asin"] = asins_buffer
        
        # Write chunk to temporary location
        chunk_path = f"{TEMP_DIR}chunk_{chunk_num:04d}.parquet"
        chunk_df.to_parquet(chunk_path, index=False)
        chunk_files.append(chunk_path)
        
        print(f"\nWrote chunk {chunk_num} with {len(embeddings_buffer)} embeddings to {chunk_path}")
        
        # Clear buffers and force garbage collection
        embeddings_buffer = []
        asins_buffer = []
        del chunk_df
        gc.collect()
        
        chunk_num += 1

print(f"\nCombining {len(chunk_files)} chunks into final file...")

# Read and combine all chunks
all_chunks = []
for chunk_file in tqdm(chunk_files):
    chunk = pd.read_parquet(chunk_file)
    all_chunks.append(chunk)
    del chunk
    gc.collect()

# Concatenate all chunks
final_df = pd.concat(all_chunks, ignore_index=True)
del all_chunks
gc.collect()

# Write final result
print(f"Writing final embeddings to {OUT_PATH}...")
final_df.to_parquet(OUT_PATH, index=False)

print(f"\nDone! Final shape: {final_df.shape}")
print(f"Output: {OUT_PATH}")

# Clean up temporary chunks
print("\nCleaning up temporary files...")
import s3fs
s3 = s3fs.S3FileSystem()
for chunk_file in chunk_files:
    try:
        s3.rm(chunk_file)
    except:
        pass

print("All done!")