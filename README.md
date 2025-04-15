# Technical Documentation Assistant

This project demonstrates an AI-powered assistant for technical documentation using retrieval-augmented generation principles.

## Features

- **Vector Search**: Find documentation based on semantic similarity
- **Multiple Source Integration**: Combines information from PyTorch, TensorFlow, and Hugging Face Transformers
- **Web Search Fallback**: When information isn't found in the knowledge base, falls back to web search

## How to Use

Simply ask a technical question in the input box and get relevant information from the knowledge base or web search.

## Sample Questions

Try some of these questions:
- What is PyTorch?
- How does TensorFlow compare to PyTorch?
- What are transformers in machine learning?
- How to implement a neural network in PyTorch?
- What is the difference between RNN and Transformer models?

## Technologies Used

- Vector Embeddings (BAAI/bge-large-en-v1.5)
- Qdrant Vector Database
- Streamlit for UI
