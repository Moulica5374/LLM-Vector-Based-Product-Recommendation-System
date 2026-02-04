# LLM & VECTOR EMBEDDINGS PRODUCT RECOMMENDATION SYSTEM

An LLM-powered semantic search and recommendation engine for beauty products using vector embeddings and similarity search.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20EC2-orange.svg)](https://aws.amazon.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings-green.svg)](https://openai.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-purple.svg)](https://pinecone.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io)

## Overview

A production-ready recommendation system that uses semantic embeddings to find similar beauty products. The system processes Amazon product data, generates vector embeddings using OpenAI's latest models, and stores them in Pinecone for fast similarity search.

**Key Capabilities:**
- Find products similar to a given item
- Generate personalized recommendations based on user purchase history
- Semantic search across 20,000+ beauty products
- Sub-100ms query latency
- Beautiful web interface for easy interaction

## Demo Video

> Add your video link here after recording

[![Demo Video](https://img.shields.io/badge/Watch-Demo-red?style=for-the-badge&logo=youtube)](https://youtube.com/your-video-link)

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Source** | HuggingFace Datasets | Amazon Reviews 2023 (Beauty category) |
| **Cloud Storage** | AWS S3 | Raw & processed data storage |
| **Compute** | AWS EC2 | Model deployment & API hosting |
| **Embeddings** | OpenAI `text-embedding-3-small` | 1536-dimensional semantic vectors |
| **Vector Database** | Pinecone (Serverless) | Similarity search & retrieval |
| **Web Interface** | Streamlit | Interactive UI for recommendations |
| **Processing** | Pandas + NumPy | Data transformation pipeline |
| **Visualization** | Plotly | Interactive charts and graphs |
| **Runtime** | Python 3.12 | Core development environment |

## Features

- Processes **20,000 beauty product records** from Amazon Reviews 2023
- Generates semantic embeddings using **OpenAI's latest model**
- **Batch processing** for optimized API usage (200 items/batch)
- **Serverless vector database** with automatic scaling
- **Cosine similarity** search for product recommendations
- Direct **S3 integration** for scalable data pipeline
- **Interactive web UI** with real-time recommendations
- **Multi-product personalized recommendations**
- **Category-based browsing** with analytics
- **CSV export** functionality

## Architecture
```
┌─────────────────┐
│  HuggingFace    │
│    Dataset      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    AWS S3       │
│  Raw Storage    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│  (Pandas)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  OpenAI API     │
│  Embeddings     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Pinecone      │
│  Vector Store   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit UI   │
│  + Query API    │
└─────────────────┘
```

## Getting Started

### Prerequisites
```bash
# Python 3.12+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:
```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Pinecone
PINECONE_API_KEY=pcsk-...

# AWS (optional if using AWS CLI configured)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

## Usage

### Data Pipeline

#### 1. Data Ingestion
```bash
python scripts/01_ingest_data.py
```
- Downloads 20K Amazon beauty products from [HuggingFace](https://github.com/hyp1231/AmazonReviews2023/blob/main/amazon-c4/README.md)
- Uploads raw data to S3: `s3://recommendation-system-llms/raw/`

#### 2. Data Preprocessing
```bash
python scripts/02_preprocess_data.py
```
- Cleans and formats product metadata
- Combines title, description, features into text for embedding
- Saves to: `s3://recommendation-system-llms/processed_data/`

#### 3. Generate Embeddings
```bash
python scripts/03_generate_embeddings.py
```
- Creates 1536-dimensional vectors using OpenAI API
- Batched processing (200 items/batch) for efficiency
- Saves to: `s3://recommendation-system-llms/embeddings/`
- **Processing time**: ~30 minutes for 20K products

#### 4. Initialize Pinecone Index
```bash
python scripts/04_setup_pinecone.py
```
- Creates `beauty-products` index
- Dimension: 1536
- Metric: Cosine similarity
- Region: us-east-1

#### 5. Upload to Vector DB
```bash
python scripts/05_upload_embeddings.py
```
- Loads embeddings into Pinecone
- Chunked uploads (10K rows/chunk, 200 vectors/batch)
- **Upload time**: ~5-10 minutes

---

## Web Interface

The system includes a beautiful Streamlit web interface for easy interaction.

### Features:
- **Product Search**: Search by title and get instant recommendations
- **Similar Products**: Find products similar to any item with similarity scores
- **Personalized Recommendations**: Add multiple products to cart for personalized suggestions
- **Category Browser**: Explore products by category with interactive charts
- **CSV Export**: Download recommendations for further analysis
- **Visual Analytics**: Interactive Plotly charts showing similarity distributions

### Launch the UI:
```bash
# Set environment variable
export PINECONE_API_KEY="your-key"

# Start the Streamlit app
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# Access in browser
http://your-ec2-public-ip:8501
```

### UI Screenshots:

#### Similar Product Recommendations
Search for a product and get instant similar recommendations with similarity scores

#### Personalized Multi-Product Recommendations
Add products to cart and get personalized suggestions based on combined preferences

#### Category Browser with Analytics
Explore products by category with interactive pie and bar charts

---

## Querying the System (Programmatic)

### Basic Similarity Search
```python
from pinecone import Pinecone
import os

# Initialize
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("beauty-products")

# Find similar products
def find_similar_products(product_asin, top_k=10):
    # Fetch product embedding
    result = index.fetch(ids=[product_asin])
    
    if product_asin not in result['vectors']:
        return None
    
    query_vector = result['vectors'][product_asin]['values']
    
    # Query for similar items
    results = index.query(
        vector=query_vector,
        top_k=top_k + 1,  # +1 to exclude the query product
        include_values=False
    )
    
    # Filter out the query product
    return [m for m in results['matches'] if m['id'] != product_asin][:top_k]

# Example usage
similar = find_similar_products("B00A1B2C3D", top_k=5)
for match in similar:
    print(f"{match['id']}: {match['score']:.4f}")
```

### User-Based Recommendations
```python
def get_user_recommendations(user_purchase_history, top_k=10):
    """Generate recommendations based on user's purchase history"""
    
    # Fetch all user product embeddings
    result = index.fetch(ids=user_purchase_history)
    
    if not result['vectors']:
        return []
    
    # Average the embeddings (simple collaborative filtering)
    embeddings = [result['vectors'][pid]['values'] for pid in result['vectors']]
    avg_embedding = [sum(x) / len(embeddings) for x in zip(*embeddings)]
    
    # Query for recommendations
    results = index.query(
        vector=avg_embedding,
        top_k=top_k + len(user_purchase_history),
        include_values=False
    )
    
    # Exclude products user already has
    recommendations = [
        m for m in results['matches'] 
        if m['id'] not in user_purchase_history
    ]
    
    return recommendations[:top_k]

# Example
user_history = ["B001234567", "B007654321", "B009876543"]
recs = get_user_recommendations(user_history, top_k=10)
```

### Semantic Search with Text
```python
from openai import OpenAI

def search_by_text(query_text, top_k=10):
    """Search products using natural language"""
    
    # Generate embedding for query text
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.embeddings.create(
        input=query_text,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    # Search in Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_values=False
    )
    
    return results['matches']

# Example
results = search_by_text("moisturizer for dry sensitive skin", top_k=5)
```

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Total Products** | 20,000 |
| **Vector Dimension** | 1,536 |
| **Storage Used** | 0.12 GB / 2 GB |
| **Query Latency (p50)** | ~50ms |
| **Query Latency (p99)** | ~200ms |
| **Similarity Metric** | Cosine |
| **Index Type** | Serverless (Auto-scaling) |
| **Embedding Cost** | ~$0.40 for 20K products |

## Use Cases

1. **E-commerce Recommendations**: "Customers who bought this also bought..."
2. **Product Discovery**: Help users find similar products they might like
3. **Personalized Shopping**: Recommendations based on browsing/purchase history
4. **Category Exploration**: Browse top-rated items by category
5. **Semantic Search**: Natural language product search

## Lessons Learned

### Memory Management
**Challenge**: Out of Memory (OOM) errors when processing large Parquet files

**Solutions Implemented:**
1. **Chunked Processing**
   - Process 10,000 rows at a time instead of loading entire dataset
   - Releases memory after each chunk
   
2. **Batch Processing**
   - OpenAI API: 200 items/batch
   - Pinecone uploads: 200 vectors/batch
   
3. **Streaming with PyArrow**
```python
   import pyarrow.parquet as pq
   
   parquet_file = pq.ParquetFile(S3_PATH)
   for batch in parquet_file.iter_batches(batch_size=10000):
       df_chunk = batch.to_pandas()
       # Process chunk
```

### Cost Optimization
- **OpenAI Embeddings**: $0.02 per 1M tokens
  - 20K products = $0.40
- **Pinecone Free Tier**: 2M write units, 0.12GB storage
  - Cost: $0 (within free tier)
- **Total Cost**: < $1 for entire project

### Rate Limiting
- Implemented exponential backoff for API calls
- Added progress tracking with `tqdm`
- Graceful error handling and retry logic

## Troubleshooting

### "Killed" Error During Upload
```bash
# Reduce chunk size
CHUNK_SIZE = 5000  # Instead of 10000
```

### API Rate Limits
```python
# Add delay between batches
import time
time.sleep(0.5)  # 500ms delay
```

### Connection Timeouts
```python
# Increase timeout in Pinecone client
index = pc.Index("beauty-products", pool_threads=30)
```

### Streamlit Port Access
```bash
# Open port 8501 in AWS Security Group
# EC2 → Security Groups → Edit Inbound Rules
# Add: Custom TCP, Port 8501, Source 0.0.0.0/0
```

## References

- [Amazon Reviews 2023 Dataset](https://github.com/hyp1231/AmazonReviews2023)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Vector Similarity Search Theory](https://www.pinecone.io/learn/vector-similarity/)

## Future Enhancements

- Add metadata filtering (category, price range, brand)
- Implement hybrid search (keyword + semantic)
- Build Streamlit UI for interactive queries - **Completed**
- Add A/B testing framework
- Integrate with real-time user behavior tracking
- Deploy as REST API with FastAPI
- Add monitoring and analytics dashboard
- Implement user authentication and saved preferences
- Add product images to UI
- Deploy to production with Docker + Kubernetes

## License

MIT License - see [LICENSE](LICENSE) file for details

## Author

Moulica Goli - [GitHub](https://github.com/moulica5374) | [LinkedIn](https://linkedin.com/in/moulicagoli)

---

**Star this repo if you find it helpful!**

## Acknowledgments

- Amazon Reviews 2023 dataset by McAuley Lab
- OpenAI for embeddings API
- Pinecone for vector database
- Streamlit for beautiful UI framework