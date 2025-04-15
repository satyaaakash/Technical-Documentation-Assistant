"""
Test the robust version of our system that works with or without GPU
"""
import os
import sys
from src.embeddings.embedding_manager import EmbeddingManager
from src.vector_db.qdrant_manager import QdrantManager
from src.web_search import WebSearcher
from src.fallback_agent import FallbackAgent

def main():
    """Main test function"""
    print("=== Technical Documentation Assistant - Robust Test ===")
    
    # Initialize components
    print("Initializing embedding manager...")
    embedding_manager = EmbeddingManager()
    
    print("Initializing vector database...")
    vector_db = QdrantManager("tech_docs")
    
    print("Initializing web searcher...")
    web_searcher = WebSearcher()
    
    # Try to detect GPU
    import torch
    has_gpu = torch.cuda.is_available()
    print(f"GPU available: {has_gpu}")
    
    # Initialize agent - first try with LLM, fall back if needed
    print("Initializing agent...")
    agent = FallbackAgent(
        vector_db_manager=vector_db,
        embedding_manager=embedding_manager,
        web_searcher=web_searcher,
        use_llm=has_gpu  # Only use LLM if GPU is available
    )
    
    # Test queries
    test_queries = [
        "What is PyTorch?",
        "How does TensorFlow compare to PyTorch?",
        "What are transformers in machine learning?",
        "How to implement a neural network in PyTorch?",
        "What is the difference between RNN and Transformer models?"
    ]
    
    # Process each query
    for query in test_queries:
        print(f"\n=== Processing query: {query} ===")
        response = agent.process_query(query)
        print(f"Response:\n{response}")
        print("=" * 50)

if __name__ == "__main__":
    main()
