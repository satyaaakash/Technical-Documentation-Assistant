"""
Process additional documentation sources
"""
import os
import sys
from src.document_processor import DocumentProcessor
from src.embeddings.embedding_manager import EmbeddingManager
from src.vector_db.qdrant_manager import QdrantManager

# Initialize components
processor = DocumentProcessor()
embedding_manager = EmbeddingManager()
vector_db = QdrantManager('tech_docs')

# Process documentation from scikit-learn
docs = processor.process_github_readme('scikit-learn', 'scikit-learn')
embeddings = embedding_manager.create_embedding([doc['text'] for doc in docs])
for i, doc in enumerate(docs):
    doc['embedding'] = embeddings[i]
vector_db.add_documents(docs)
print(f"Added {len(docs)} scikit-learn documents")

# Add more libraries here as needed
