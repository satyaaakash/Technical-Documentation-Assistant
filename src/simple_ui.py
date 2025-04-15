"""
Simple web UI for the Technical Documentation Assistant
"""

import os
import sys
import gradio as gr

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings.embedding_manager import EmbeddingManager
from src.vector_db.qdrant_manager import QdrantManager
from src.web_search import WebSearcher
from src.fallback_agent import FallbackAgent

# Initialize components
print("Initializing Technical Documentation Assistant...")
embedding_manager = EmbeddingManager()
vector_db = QdrantManager("tech_docs")
web_searcher = WebSearcher()

# Try to detect GPU
try:
    import torch
    has_gpu = torch.cuda.is_available()
    print(f"GPU available: {has_gpu}")
except Exception:
    has_gpu = False
    print("GPU not available")

# Initialize agent
agent = FallbackAgent(
    vector_db_manager=vector_db,
    embedding_manager=embedding_manager,
    web_searcher=web_searcher,
    use_llm=has_gpu  # Only use LLM if GPU is available
)

def process_query(query):
    """Process a query and return the response"""
    if not query or query.strip() == "":
        return "Please enter a question."
    
    return agent.process_query(query)

# Create Gradio interface
demo = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(
        lines=3, 
        placeholder="Ask a question about PyTorch, TensorFlow, or transformers..."
    ),
    outputs=gr.Textbox(lines=10),
    title="Technical Documentation Assistant",
    description="Ask questions about programming libraries, frameworks, and technical concepts",
    examples=[
        "What is PyTorch?",
        "How to install TensorFlow?",
        "What are transformers in machine learning?",
        "How does PyTorch compare to TensorFlow?",
        "What is JAX?"
    ]
)

if __name__ == "__main__":
    # Get hostname for user information
    import socket
    hostname = socket.gethostname()
    
    print(f"=== Technical Documentation Assistant UI ===")
    print(f"Running on: {hostname}")
    print(f"To access remotely via SSH port forwarding:")
    print(f"ssh -N -L 7860:{hostname}:7860 username@hpg.rc.ufl.edu")
    print(f"Then open http://localhost:7860 in your browser")
    
    # Launch the UI
    demo.launch(share=False)
