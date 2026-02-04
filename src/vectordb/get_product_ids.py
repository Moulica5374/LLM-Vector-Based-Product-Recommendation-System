import pandas as pd
import random

S3_PATH = "s3://recommendation-system-llms/embeddings/products_embeddings.parquet"

print("Loading product data from S3...")
df = pd.read_parquet(S3_PATH)

print(f"\nTotal products: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Get random sample of product IDs
sample_size = 20
sample_ids = df['parent_asin'].sample(n=min(sample_size, len(df))).tolist()

print(f"\nRandom {len(sample_ids)} Product IDs:")
for i, pid in enumerate(sample_ids, 1):
    print(f"{i:2d}. {pid}")

print("\n💡 Copy any of these IDs to use in your query!")
print(f"\n Example usage:")
print(f'   target_product = "{sample_ids[0]}"')