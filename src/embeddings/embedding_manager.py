"""
Manager for text embeddings
"""

from typing import List, Union
import torch
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize the embedding manager
        
        Args:
            model_name (str): Name of the embedding model to use
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        print(f"Embedding model loaded and moved to {device}")
    
    def create_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Create embeddings for text or list of texts
        
        Args:
            text (str or List[str]): Text or list of texts to embed
            
        Returns:
            List[float] or List[List[float]]: Embedding vector(s)
        """
        # Convert to list if single string
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        
        # Return single embedding or list of embeddings
        return embeddings[0].tolist() if is_single else [emb.tolist() for emb in embeddings]
