from huggingface_hub import list_repo_files
import pandas as pd
import boto3
import io

BUCKET = "recommendation-system-llms"
S3_KEY = "raw_data/reviews_sample.parquet"
CATEGORY = "All_Beauty"
N = 20000

print("Listing files in the repository...")
files = list_repo_files(
    repo_id="McAuley-Lab/Amazon-Reviews-2023",
    repo_type="dataset"
)

# Filter for All_Beauty files
beauty_files = [f for f in files if "All_Beauty" in f and f.endswith('.parquet')]
print("Found files:")
for f in beauty_files[:10]:  # Show first 10
    print(f"  {f}")

if not beauty_files:
    print("\nNo All_Beauty files found. All available files:")
    for f in files[:20]:
        print(f"  {f}")
else:
    # Use the first parquet file found
    from huggingface_hub import hf_hub_download
    
    target_file = beauty_files[0]
    print(f"\nDownloading: {target_file}")
    
    file_path = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename=target_file,
        repo_type="dataset"
    )
    
    print("Reading parquet file...")
    df = pd.read_parquet(file_path)
    
    # Sample N rows
    if len(df) > N:
        df = df.head(N)
        print(f"Sampled first {N} rows out of {len(df)} total")
    
    print("Converting to parquet in memory...")
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    print("Uploading directly to S3...")
    s3 = boto3.client("s3")
    s3.upload_fileobj(buffer, BUCKET, S3_KEY)
    
    print(f"Done. Uploaded to s3://{BUCKET}/{S3_KEY}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")