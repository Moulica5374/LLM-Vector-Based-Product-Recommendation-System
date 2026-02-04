"""
Vector database module for Pinecone operations
"""

from .query import (
    get_pinecone_index,
    find_similar_products,
    get_user_recommendations,
    search_by_text,
    get_recommendations_with_metadata,
    get_user_recommendations_with_metadata
)

__all__ = [
    'get_pinecone_index',
    'find_similar_products',
    'get_user_recommendations',
    'search_by_text',
    'get_recommendations_with_metadata',
    'get_user_recommendations_with_metadata'
]