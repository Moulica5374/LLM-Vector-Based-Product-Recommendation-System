import pandas as pd
from pinecone import Pinecone
import os

print("Loading product data from S3...")

S3_PATH = "s3://recommendation-system-llms/processed_data/products_for_embedding.parquet"

# Load the data (assumes s3fs is already installed in the environment)
df = pd.read_parquet(S3_PATH)
print(f"Loaded {len(df)} products")
print(f"Columns: {df.columns.tolist()}\n")

# Keep only the columns we need
metadata_df = df[["parent_asin", "title", "main_category", "average_rating"]].copy()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("beauty-products")

def show_recommendations(product_asin, top_k=5):
    """
    Show recommendations with product details
    """
    query_product_df = metadata_df[metadata_df["parent_asin"] == product_asin]

    if query_product_df.empty:
        print(f"Product {product_asin} not found in data")
        return

    query_product = query_product_df.iloc[0]

    result = index.fetch(ids=[product_asin])

    if product_asin not in result["vectors"]:
        print(f"Product {product_asin} not found in vector database")
        return

    query_vector = result["vectors"][product_asin]["values"]
    similar = index.query(vector=query_vector, top_k=top_k + 1, include_values=False)

    print("\n" + "=" * 80)
    print("QUERY PRODUCT")
    print("=" * 80)
    print(f"Title: {query_product['title']}")
    print(f"Category: {query_product['main_category']}")
    print(f"Rating: {query_product['average_rating']}/5.0")
    print(f"ASIN: {product_asin}")

    print("\n" + "=" * 80)
    print("RECOMMENDED PRODUCTS")
    print("=" * 80)

    count = 0
    for match in similar["matches"]:
        if match["id"] == product_asin:
            continue

        rec_data = metadata_df[metadata_df["parent_asin"] == match["id"]]
        if rec_data.empty:
            continue

        rec_product = rec_data.iloc[0]
        count += 1

        print(f"\n{count}. {rec_product['title'][:85]}")
        print(f"   Category: {rec_product['main_category']}")
        print(f"   Rating: {rec_product['average_rating']}/5.0")
        print(f"   Similarity Score: {match['score']:.4f}")
        print(f"   ASIN: {match['id']}")

        if count >= top_k:
            break

    print("\n" + "=" * 80)


def get_user_history_recommendations(purchase_history, top_k=10):
    """
    Get recommendations based on multiple purchases
    """
    print("\n" + "=" * 80)
    print("PURCHASE HISTORY")
    print("=" * 80)

    for i, asin in enumerate(purchase_history, 1):
        product_df = metadata_df[metadata_df["parent_asin"] == asin]
        if not product_df.empty:
            product = product_df.iloc[0]
            print(f"{i}. {product['title'][:70]}")
            print(f"   Category: {product['main_category']} | Rating: {product['average_rating']}/5.0")

    result = index.fetch(ids=purchase_history)

    if not result["vectors"]:
        print("No valid products found in purchase history")
        return

    embeddings = [result["vectors"][pid]["values"] for pid in result["vectors"]]
    avg_embedding = [sum(x) / len(embeddings) for x in zip(*embeddings)]

    similar = index.query(vector=avg_embedding, top_k=top_k + len(purchase_history), include_values=False)

    print("\n" + "=" * 80)
    print("PERSONALIZED RECOMMENDATIONS")
    print("=" * 80)

    count = 0
    for match in similar["matches"]:
        if match["id"] in purchase_history:
            continue

        rec_data = metadata_df[metadata_df["parent_asin"] == match["id"]]
        if rec_data.empty:
            continue

        rec_product = rec_data.iloc[0]
        count += 1

        print(f"\n{count}. {rec_product['title'][:85]}")
        print(f"   Category: {rec_product['main_category']}")
        print(f"   Rating: {rec_product['average_rating']}/5.0")
        print(f"   Similarity Score: {match['score']:.4f}")
        print(f"   ASIN: {match['id']}")

        if count >= top_k:
            break

    print("\n" + "=" * 80)


def export_recommendations_to_csv(product_asin, output_file="recommendations.csv", top_k=10):
    """
    Export recommendations to CSV
    """
    result = index.fetch(ids=[product_asin])

    if product_asin not in result["vectors"]:
        print(f"Product {product_asin} not found")
        return None

    query_vector = result["vectors"][product_asin]["values"]
    similar = index.query(vector=query_vector, top_k=top_k + 1, include_values=False)

    results = []
    for match in similar["matches"]:
        if match["id"] == product_asin:
            continue

        product_data = metadata_df[metadata_df["parent_asin"] == match["id"]]
        if not product_data.empty:
            product = product_data.iloc[0]
            results.append({
                "Rank": len(results) + 1,
                "ASIN": match["id"],
                "Title": product["title"],
                "Category": product["main_category"],
                "Rating": product["average_rating"],
                "Similarity_Score": f"{match['score']:.4f}"
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Recommendations saved to {output_file}")
    return results_df


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EXAMPLE 1: SIMILAR PRODUCT RECOMMENDATIONS")
    print("=" * 80)

    sample_asin = metadata_df["parent_asin"].iloc[0]
    show_recommendations(sample_asin, top_k=5)

    print("\n" + "=" * 80)
    print("EXAMPLE 2: PERSONALIZED RECOMMENDATIONS")
    print("=" * 80)

    user_history = metadata_df["parent_asin"].sample(min(3, len(metadata_df))).tolist()
    get_user_history_recommendations(user_history, top_k=5)

    print("\n" + "=" * 80)
    print("EXAMPLE 3: EXPORT RECOMMENDATIONS TO CSV")
    print("=" * 80)

    export_recommendations_to_csv(sample_asin, "beauty_recommendations.csv", top_k=10)
