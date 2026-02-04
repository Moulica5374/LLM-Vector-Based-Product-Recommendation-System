"""
Data Preprocessing Script
Loads raw product data, cleans it, and creates text for embeddings
"""

import pandas as pd
import numpy as np
import time
from typing import Any

# Configuration
S3_INPUT_PATH = "s3://recommendation-system-llms/raw_data/reviews_sample.parquet"
S3_OUTPUT_PATH = "s3://recommendation-system-llms/processed_data/products_for_embedding.parquet"

# Required columns for embeddings
REQUIRED_COLS = [
    "parent_asin",
    "title",
    "description",
    "features",
    "categories",
    "main_category",
    "average_rating"
]


def safe_str(x: Any) -> str:
    """
    Safely convert any value to string
    Handles None, NaN, numpy arrays, etc.
    
    Args:
        x: Value to convert
        
    Returns:
        String representation or empty string
    """
    if x is None:
        return ""
    
    if isinstance(x, float) and pd.isna(x):
        return ""
    
    if isinstance(x, np.ndarray):
        if len(x) == 0:
            return ""
        return ", ".join(map(str, x.tolist()))
    
    return str(x)


def load_data(s3_path: str) -> pd.DataFrame:
    """
    Load raw product data from S3
    
    Args:
        s3_path: S3 path to parquet file
        
    Returns:
        DataFrame with raw product data
    """
    print(f"Loading data from {s3_path}...")
    t0 = time.time()
    
    df = pd.read_parquet(s3_path)
    
    load_time = round(time.time() - t0, 2)
    print(f"✓ Loaded {df.shape[0]:,} products in {load_time} seconds")
    print(f"  Columns: {list(df.columns)}")
    
    return df


def explore_data(df: pd.DataFrame) -> None:
    """
    Print data exploration statistics
    
    Args:
        df: DataFrame to explore
    """
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    print(f"\nShape: {df.shape}")
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    }).sort_values('Percentage', ascending=False)
    print(missing_df[missing_df['Missing'] > 0])


def create_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert complex fields to text for embedding
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with text columns added
    """
    print("\n" + "="*60)
    print("CREATING TEXT COLUMNS")
    print("="*60)
    
    print("Converting features to text...")
    df["features_text"] = df["features"].apply(safe_str)
    
    print("Converting title to text...")
    df["title_text"] = df["title"].apply(safe_str)
    
    print("Converting description to text...")
    df["desc_text"] = df["description"].apply(safe_str)
    
    print("Converting category to text...")
    df["main_cat_text"] = df["main_category"].apply(safe_str)
    
    print("✓ Text columns created")
    
    return df


def create_embedding_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all text fields into single embedding text
    
    Args:
        df: DataFrame with text columns
        
    Returns:
        DataFrame with text_for_embedding column
    """
    print("\n" + "="*60)
    print("CREATING EMBEDDING TEXT")
    print("="*60)
    
    print("Combining text fields...")
    df["text_for_embedding"] = (
        "Title: " + df["title_text"] + ". "
        "Category: " + df["main_cat_text"] + ". "
        "Features: " + df["features_text"] + ". "
        "Description: " + df["desc_text"] + ". "
        "Average rating: " + df["average_rating"].astype(str)
    )
    
    print("✓ Embedding text created")
    
    # Show sample
    print("\nSample embedding text:")
    print("-" * 60)
    sample = df["text_for_embedding"].iloc[0]
    print(sample[:200] + "..." if len(sample) > 200 else sample)
    
    return df


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only required columns for output
    
    Args:
        df: Full DataFrame
        
    Returns:
        DataFrame with only required columns
    """
    output_cols = [
        "parent_asin",
        "text_for_embedding",
        "title",
        "main_category",
        "average_rating"
    ]
    
    print("\n" + "="*60)
    print("SELECTING OUTPUT COLUMNS")
    print("="*60)
    
    print(f"Selected columns: {output_cols}")
    output_df = df[output_cols].copy()
    
    print(f"✓ Output shape: {output_df.shape}")
    
    return output_df


def save_data(df: pd.DataFrame, s3_path: str) -> None:
    """
    Save processed data to S3
    
    Args:
        df: DataFrame to save
        s3_path: S3 destination path
    """
    print("\n" + "="*60)
    print("SAVING DATA")
    print("="*60)
    
    print(f"Saving to {s3_path}...")
    t0 = time.time()
    
    df.to_parquet(s3_path, index=False)
    
    save_time = round(time.time() - t0, 2)
    print(f"✓ Saved {len(df):,} records in {save_time} seconds")
    print(f"  Size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def main():
    """
    Main preprocessing pipeline
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load data
    df = load_data(S3_INPUT_PATH)
    
    # Explore data
    explore_data(df)
    
    # Create text columns
    df = create_text_columns(df)
    
    # Create embedding text
    df = create_embedding_text(df)
    
    # Select output columns
    output_df = select_output_columns(df)
    
    # Save processed data
    save_data(output_df, S3_OUTPUT_PATH)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE ✓")
    print("="*60)
    print(f"\nProcessed data saved to: {S3_OUTPUT_PATH}")
    print(f"Total records: {len(output_df):,}")
    print(f"Columns: {list(output_df.columns)}")


if __name__ == "__main__":
    main()