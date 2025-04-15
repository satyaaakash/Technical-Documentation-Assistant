#!/bin/bash
#SBATCH --job-name=rag_pipeline
#SBATCH --output=logs/rag_pipeline_%j.out
#SBATCH --error=logs/rag_pipeline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --account=carpena

# Load required modules
module purge
module load python
module load cuda/12.2.2

# Activate virtual environment
source $SLURM_SUBMIT_DIR/venv/bin/activate

# Print environment info
echo "== Environment Information =="
echo "HOSTNAME: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Step 1: Process documentation
echo -e "\n== Step 1: Processing Documentation =="
python -c "
import os, sys
sys.path.append('$SLURM_SUBMIT_DIR')
from src.document_processor import DocumentProcessor

# Process PyTorch documentation
processor = DocumentProcessor()
docs = processor.process_github_readme('pytorch', 'pytorch')
processor.save_documents(docs, 'data/processed/pytorch_docs.json')

# Process Hugging Face Transformers documentation
docs = processor.process_github_readme('huggingface', 'transformers')
processor.save_documents(docs, 'data/processed/transformers_docs.json')

# Process TensorFlow documentation
docs = processor.process_github_readme('tensorflow', 'tensorflow')
processor.save_documents(docs, 'data/processed/tensorflow_docs.json')
"

# Step 2: Generate embeddings
echo -e "\n== Step 2: Generating Embeddings =="
python -c "
import os, sys
sys.path.append('$SLURM_SUBMIT_DIR')
from src.embeddings.embedding_manager import EmbeddingManager
import json

# Initialize embedding manager
embedding_manager = EmbeddingManager()

# Process each document file
for doc_file in ['pytorch_docs.json', 'transformers_docs.json', 'tensorflow_docs.json']:
    input_path = f'data/processed/{doc_file}'
    output_path = input_path.replace('.json', '_with_embeddings.json')
    
    print(f'Processing {input_path}')
    
    # Load documents
    with open(input_path, 'r') as f:
        documents = json.load(f)
    
    # Extract text
    texts = [doc['text'] for doc in documents]
    
    # Generate embeddings
    embeddings = embedding_manager.create_embedding(texts)
    
    # Add embeddings to documents
    for i, embedding in enumerate(embeddings):
        documents[i]['embedding'] = embedding
    
    # Save documents with embeddings
    with open(output_path, 'w') as f:
        json.dump(documents, f)
    
    print(f'Saved {len(documents)} documents with embeddings to {output_path}')
"

# Step 3: Load documents into vector database
echo -e "\n== Step 3: Loading Documents into Vector Database =="
python -c "
import os, sys
sys.path.append('$SLURM_SUBMIT_DIR')
from src.vector_db.qdrant_manager import QdrantManager
import json

# Initialize vector database
vector_db = QdrantManager('tech_docs')

# Load each document file
for doc_file in ['pytorch_docs_with_embeddings.json', 'transformers_docs_with_embeddings.json', 'tensorflow_docs_with_embeddings.json']:
    input_path = f'data/processed/{doc_file}'
    
    print(f'Loading {input_path}')
    
    # Load documents
    with open(input_path, 'r') as f:
        documents = json.load(f)
    
    # Add documents to vector database
    vector_db.add_documents(documents)
    
    print(f'Added {len(documents)} documents to vector database')
"

# Step 4: Test the RAG pipeline
echo -e "\n== Step 4: Testing RAG Pipeline =="
python -c "
import os, sys
sys.path.append('$SLURM_SUBMIT_DIR')
from src.rag_pipeline import RAGPipeline

# Create pipeline with LLM
pipeline = RAGPipeline(use_llm=True)

# Test queries
test_queries = [
    'What is PyTorch?',
    'How to install TensorFlow?',
    'What are transformers in machine learning?',
    'What is the difference between PyTorch and TensorFlow?'
]

for query in test_queries:
    print(f'\nQuery: {query}')
    response = pipeline.process_query(query)
    print(f'Response: {response}')
"

echo -e "\n== Pipeline Test Complete =="
