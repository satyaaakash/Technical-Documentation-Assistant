"""
Vector database manager for Qdrant
"""

import os
from typing import List, Dict, Any
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantManager:
    def __init__(self, collection_name: str, path: str = "./data/vector_db"):
        """
        Initialize Qdrant vector database manager
        
        Args:
            collection_name (str): Name of the collection to use
            path (str): Path to store the database
        """
        self.collection_name = collection_name
        os.makedirs(path, exist_ok=True)
        
        # Initialize local Qdrant instance
        self.client = QdrantClient(path=path)
        
        # Check if collection exists, create if not
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self._create_collection()
    
    def _create_collection(self, vector_size: int = 1024):
        """Create a new collection with the specified parameters"""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the vector database
        
        Args:
            documents (List[Dict]): List of documents with 'text', 'embedding', and 'metadata'
        """
        points = []
        for i, doc in enumerate(documents):
            points.append(
                PointStruct(
                    id=i + self._get_collection_size(),
                    vector=doc['embedding'],
                    payload={
                        'text': doc['text'],
                        **doc['metadata']
                    }
                )
            )
        
        # Upload in batches to avoid timeouts with large collections
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        print(f"Added {len(documents)} documents to collection {self.collection_name}")
    
    def _get_collection_size(self) -> int:
        """Get the current size of the collection"""
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.points_count
    
    def search(self, query_vector: List[float], limit: int = 5) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query_vector (List[float]): Embedding vector of the query
            limit (int): Maximum number of results to return
            
        Returns:
            List[Dict]: List of matching documents with similarity scores
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        return [
            {
                'text': hit.payload.get('text', ''),
                'metadata': {k: v for k, v in hit.payload.items() if k != 'text'},
                'score': hit.score
            }
            for hit in results
        ]
