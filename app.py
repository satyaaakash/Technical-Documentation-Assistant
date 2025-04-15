import streamlit as st
import os
import sys
import json

# Import your project components
from src.embeddings.embedding_manager import EmbeddingManager
from src.vector_db.qdrant_manager import QdrantManager
from src.web_search import WebSearcher

# Page configuration
st.set_page_config(
    page_title="Technical Documentation Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize components
@st.cache_resource
def load_components():
    embedding_manager = EmbeddingManager()
    vector_db = QdrantManager("tech_docs")
    web_searcher = WebSearcher()
    return embedding_manager, vector_db, web_searcher

# Load components with a loading indicator
with st.spinner("Loading components..."):
    embedding_manager, vector_db, web_searcher = load_components()

# Title and description
st.title("📚 Technical Documentation Assistant")
st.markdown("""
This tool helps you find information about programming libraries like PyTorch, TensorFlow, and Transformers.
Ask a technical question, and I'll search my knowledge base for relevant information.
""")

# Query input
query = st.text_input("Ask a technical question:", 
                      placeholder="Example: What is PyTorch? How does TensorFlow compare to PyTorch?")

if query:
    with st.spinner("Searching for information..."):
        # Get query embedding
        query_embedding = embedding_manager.create_embedding(query)
        
        # Search vector database
        vector_results = vector_db.search(query_embedding, limit=3)
        
        # If no results from vector DB, try web search
        if not vector_results:
            st.info("No information found in the knowledge base. Searching the web...")
            web_results = web_searcher.search(query, num_results=3)
            
            if web_results:
                st.subheader("Web Search Results")
                for i, result in enumerate(web_results):
                    with st.expander(f"Result {i+1}: {result.get('title', 'No title')}"):
                        st.markdown(f"**Source**: [{result.get('url', 'No URL')}]({result.get('url', 'No URL')})")
                        st.markdown(f"**Snippet**: {result.get('snippet', 'No snippet')}")
            else:
                st.error("No information found from any source.")
        else:
            st.subheader("Knowledge Base Results")
            for i, result in enumerate(vector_results):
                with st.expander(f"Result {i+1} (Relevance: {result.get('score', 0):.2f})"):
                    st.markdown(f"**Content**: {result.get('text', 'No content')}")
                    st.markdown(f"**Source**: {result.get('metadata', {}).get('source', 'Unknown')}")

# Sample questions
st.sidebar.header("Sample Questions")
sample_questions = [
    "What is PyTorch?",
    "How does TensorFlow compare to PyTorch?",
    "What are transformers in machine learning?",
    "How to implement a neural network in PyTorch?",
    "What is the difference between RNN and Transformer models?"
]

for question in sample_questions:
    if st.sidebar.button(question):
        st.session_state.query = question
        st.experimental_rerun()

# About section
st.sidebar.header("About")
st.sidebar.markdown("""
This Technical Documentation Assistant is a proof-of-concept demonstrating:
- Vector database search with semantic similarity
- Embedding-based retrieval
- Web search fallback

Created as a learning project for AI Engineering.
""")
