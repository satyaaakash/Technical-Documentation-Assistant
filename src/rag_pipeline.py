"""
Script that implements the full RAG pipeline
"""

import os
import sys
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings.embedding_manager import EmbeddingManager
from src.vector_db.qdrant_manager import QdrantManager
from src.web_search import WebSearcher
from src.agent.simple_agent import SimpleAgent

class RAGPipeline:
    def __init__(self, collection_name: str = "tech_docs", use_llm: bool = False):
        """
        Initialize the RAG pipeline
        
        Args:
            collection_name (str): Name of the vector database collection
            use_llm (bool): Whether to use LLM for response generation
        """
        self.embedding_manager = EmbeddingManager()
        self.vector_db = QdrantManager(collection_name)
        self.web_searcher = WebSearcher()
        
        # Initialize LLM if required
        self.llm = None
        if use_llm:
            from src.llm.simplified_llm import SimpleLLM
            self.llm = SimpleLLM()
            
        # Initialize agent
        self.agent = SimpleAgent(
            vector_search_func=self.search_vector_db,
            web_search_func=self.search_web,
            llm_func=self.generate_llm_response if use_llm else None
        )
        
        print(f"RAG pipeline initialized (using LLM: {use_llm})")
        
    def search_vector_db(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search the vector database
        
        Args:
            query (str): Query text
            limit (int): Maximum number of results
            
        Returns:
            List[Dict]: List of matching documents
        """
        # Generate embedding for query
        query_embedding = self.embedding_manager.create_embedding(query)
        
        # Search vector database
        results = self.vector_db.search(query_embedding, limit)
        
        return results
    
    def search_web(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web
        
        Args:
            query (str): Query text
            limit (int): Maximum number of results
            
        Returns:
            List[Dict]: List of search results
        """
        return self.web_searcher.search(query, limit)
    
    def generate_llm_response(self, prompt: str) -> str:
        """
        Generate a response using the LLM
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Generated response
        """
        if self.llm is None:
            return "[LLM not initialized]"
            
        return self.llm.generate(prompt)
    
    def process_query(self, query: str) -> str:
        """
        Process a query through the pipeline
        
        Args:
            query (str): User query
            
        Returns:
            str: Response
        """
        return self.agent.process_query(query)

if __name__ == "__main__":
    # Create pipeline without LLM for testing
    pipeline = RAGPipeline(use_llm=False)
    
    # Test queries
    test_queries = [
        "What is PyTorch?",
        "How to install TensorFlow?",
        "What are transformers in machine learning?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = pipeline.process_query(query)
        print(f"Response: {response[:200]}...")  # Show first 200 chars
