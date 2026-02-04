# LLM & VECTOR EMBEDDINGS PRODUCT RECOMMENDATION SYSTEM

An LLM-powered semantic search and recommendation engine for beauty products using vector embeddings and similarity search.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20SageMaker-orange.svg)](https://aws.amazon.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings-green.svg)](https://openai.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-purple.svg)](https://pinecone.io)


## Tech Stack 
Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Source** | HuggingFace Datasets | Amazon Reviews 2023 (Beauty category) |
| **Cloud Storage** | AWS S3 | Raw & processed data storage |
| **LLM Platform** | AWS Bedrock | Cloud-based model orchestration |
| **Embeddings** | OpenAI `text-embedding-3-small` | 1536-dimensional semantic vectors |
| **Vector Database** | Pinecone (Serverless) | Similarity search & retrieval |
| **Processing** | Pandas + NumPy | Data transformation pipeline |
| **Runtime** | Python 3.12 | Core development environment |

## Features


-  Processes **20,000 beauty product records** from Amazon Reviews 2023
-  Generates semantic embeddings using **OpenAI's latest model**
-  **Batch processing** for optimized API usage (200 items/batch)
-  **Serverless vector database** with automatic scaling
-  **Cosine similarity** search for product recommendations
-  Direct **S3 integration** for scalable data pipeline



## Usage

### **Data Ingestion**
```bash
python scripts/01_ingest_data.py
```
Downloads 20K amazon beauty products from HuggingFace [https://github.com/hyp1231/AmazonReviews2023/blob/main/amazon-c4/README.md] → S3 

###  **Data Preprocessing**
```bash
python scripts/02_preprocess_data.py
```
Cleans and formats product data for embedding generation

###  **Generate Embeddings**
```bash
python scripts/03_generate_embeddings.py
```
Creates 1536-dim vectors using OpenAI API (batched for efficiency)

###  **Initialize Pinecone Index**
```bash
python scripts/04_setup_pinecone.py
```
Sets up vector database with cosine similarity metric

###  **Upload to Vector DB**
```bash
python scripts/05_upload_embeddings.py
```
Loads embeddings into Pinecone for similarity search

---


### Lessons Learned 

- While creating the embedding being very cautios about out of memory issues 
Implemented
    - Using Batch Size
    - Chunking

- While loading the embeddings into the vector database
  Implemented
    - Using Batch Size
    - Chunking
Other techniques 
- 
