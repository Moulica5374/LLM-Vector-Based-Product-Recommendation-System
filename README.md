# LLM & VECTOR EMBEDDINGS PRODUCT RECOMMENDATION SYSTEM

An LLM-powered semantic search and recommendation engine for beauty products using vector embeddings and similarity search.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20Bedrock-orange.svg)](https://aws.amazon.com)
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