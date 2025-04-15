"""
Script to load documents with embeddings into the vector database
"""

import os
import sys
import json
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_db.qdrant_manager import QdrantManager

def load_documents_to_vector_db(input_path: str, collection_name: str = "tech_docs"):
    """
    Load documents with embeddings into the vector database
    
    Args:
        input_path (str): Path to documents with embeddings
        collection_name (str): Name of the collection
    """
    print(f"Loading documents from {input_path}")
    
    # Load documents
    with open(input_path, 'r') as f:
        documents = json.load(f)
        
    print(f"Loaded {len(documents)} documents with embeddings")
    
    # Initialize vector database
    vector_db = QdrantManager(collection_name)
    
    # Add documents to vector database
    vector_db.add_documents(documents)
    
    print(f"Added {len(documents)} documents to vector database collection '{collection_name}'")
    
    return True

if __name__ == "__main__":
    input_path = "data/processed/sample_docs_with_embeddings.json"
    collection_name = "tech_docs"
    
    load_documents_to_vector_db(input_path, collection_name)
