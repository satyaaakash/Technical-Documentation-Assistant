"""
Script to generate embeddings for processed documents
"""

import os
import sys
import json
from typing import List, Dict, Any
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings.embedding_manager import EmbeddingManager

def generate_embeddings_for_documents(input_path: str, output_path: str):
    """
    Generate embeddings for documents and save them
    
    Args:
        input_path (str): Path to input JSON documents
        output_path (str): Path to save documents with embeddings
    """
    print(f"Loading documents from {input_path}")
    
    # Load documents
    with open(input_path, 'r') as f:
        documents = json.load(f)
        
    print(f"Loaded {len(documents)} documents")
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager()
    
    # Extract text from documents
    texts = [doc['text'] for doc in documents]
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = embedding_manager.create_embedding(texts)
    
    # Add embeddings to documents
    for i, embedding in enumerate(embeddings):
        documents[i]['embedding'] = embedding
        
    # Save documents with embeddings
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(documents, f)
        
    print(f"Saved {len(documents)} documents with embeddings to {output_path}")
    
    return documents

if __name__ == "__main__":
    input_path = "data/processed/sample_docs.json"
    output_path = "data/processed/sample_docs_with_embeddings.json"
    
    generate_embeddings_for_documents(input_path, output_path)
