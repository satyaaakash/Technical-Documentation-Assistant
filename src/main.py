"""
Main script for the Technical Documentation Assistant
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from src.document_processor import DocumentProcessor
from src.web_search import WebSearcher
from src.agent.simple_agent import SimpleAgent

def main():
    """Main function to test the assistant"""
    print("Technical Documentation Assistant - Test Mode")
    
    # Initialize document processor
    processor = DocumentProcessor()
    print("Document processor initialized")
    
    # Initialize web searcher
    web_searcher = WebSearcher()
    print("Web searcher initialized")
    
    # Initialize agent without LLM for now
    agent = SimpleAgent(
        vector_search_func=lambda query: [{"text": f"Mock result for: {query}", "metadata": {"source": "mock"}}],
        web_search_func=lambda query: web_searcher.search(query)
    )
    print("Agent initialized (without LLM)")
    
    # Test processing a GitHub README
    try:
        print("\nProcessing GitHub README...")
        docs = processor.process_github_readme("huggingface", "transformers")
        print(f"Processed {len(docs)} chunks from GitHub README")
        
        # Save sample
        sample_path = os.path.join("data", "processed", "sample_docs.json")
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        with open(sample_path, "w") as f:
            json.dump(docs[:3], f, indent=2)  # Save first 3 documents as sample
        print(f"Saved sample documents to {sample_path}")
    except Exception as e:
        print(f"Error processing GitHub README: {e}")
    
    # Test the agent with a mock query
    try:
        print("\nTesting agent...")
        response = agent.process_query("What is PyTorch?")
        print(f"Agent response: {response[:200]}...")  # Show first 200 chars
    except Exception as e:
        print(f"Error testing agent: {e}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()
