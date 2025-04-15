"""
Fallback agent that works without LLM when needed
"""

import os
from typing import Dict, List, Any

class FallbackAgent:
    """
    Agent that can work with or without LLM capabilities
    """
    
    def __init__(self, vector_db_manager, embedding_manager, web_searcher, use_llm=True):
        """
        Initialize the fallback agent
        
        Args:
            vector_db_manager: Vector database manager
            embedding_manager: Embedding manager
            web_searcher: Web search tool
            use_llm: Whether to use LLM
        """
        self.vector_db = vector_db_manager
        self.embedding_manager = embedding_manager
        self.web_searcher = web_searcher
        self.use_llm = use_llm
        
        # Try to initialize LLM if needed
        self.llm = None
        if use_llm:
            try:
                from src.llm.simplified_llm import SimpleLLM
                self.llm = SimpleLLM(model_name="gpt2")
                print("LLM initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize LLM: {e}")
                print("Falling back to non-LLM mode")
                self.use_llm = False
    
    def process_query(self, query: str) -> str:
        """
        Process a query and return a response
        
        Args:
            query: User query
            
        Returns:
            Response text
        """
        # Get embedding for the query
        query_embedding = self.embedding_manager.create_embedding(query)
        
        # Search vector database
        vector_results = self.vector_db.search(query_embedding, limit=5)
        
        # If we have vector results, use them
        if vector_results:
            if self.use_llm and self.llm:
                # Format for LLM
                context = self._format_vector_results(vector_results)
                prompt = f"User Query: {query}\n\n{context}\n\nBased on this information, answer the user's query:"
                response = self.llm.generate(prompt, max_length=500)
                return response
            else:
                # Format for direct display (no LLM)
                return self._format_vector_results_human(query, vector_results)
        
        # Fall back to web search
        web_results = self.web_searcher.search(query, num_results=3)
        
        if web_results:
            if self.use_llm and self.llm:
                # Format for LLM
                context = self._format_web_results(web_results)
                prompt = f"User Query: {query}\n\n{context}\n\nBased on these search results, answer the user's query:"
                response = self.llm.generate(prompt, max_length=500)
                return response
            else:
                # Format for direct display (no LLM)
                return self._format_web_results_human(query, web_results)
        
        # No results found
        return f"I couldn't find relevant information for: {query}"
    
    def _format_vector_results(self, results: List[Dict[str, Any]]) -> str:
        """Format vector results for LLM consumption"""
        formatted = "Knowledge Base Results:\n\n"
        for i, result in enumerate(results):
            formatted += f"Document {i+1}:\n"
            formatted += f"Text: {result.get('text', 'No content')}\n"
            formatted += f"Source: {result.get('metadata', {}).get('source', 'Unknown')}\n\n"
        return formatted
    
    def _format_web_results(self, results: List[Dict[str, Any]]) -> str:
        """Format web results for LLM consumption"""
        formatted = "Web Search Results:\n\n"
        for i, result in enumerate(results):
            formatted += f"Result {i+1}:\n"
            formatted += f"Title: {result.get('title', 'No title')}\n"
            formatted += f"URL: {result.get('url', 'No URL')}\n"
            formatted += f"Snippet: {result.get('snippet', 'No snippet')}\n\n"
        return formatted
    
    def _format_vector_results_human(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Format vector results for direct human consumption"""
        response = f"Results for query: '{query}'\n\n"
        response += f"Found {len(results)} relevant documents:\n\n"
        
        for i, result in enumerate(results):
            response += f"Document {i+1}:\n"
            response += f"Content: {result.get('text', 'No content')}\n"
            response += f"Source: {result.get('metadata', {}).get('source', 'Unknown')}\n"
            response += f"Relevance: {result.get('score', 0):.2f}\n\n"
        
        return response
    
    def _format_web_results_human(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Format web results for direct human consumption"""
        response = f"Web search results for query: '{query}'\n\n"
        
        for i, result in enumerate(results):
            response += f"Result {i+1}:\n"
            response += f"Title: {result.get('title', 'No title')}\n"
            response += f"URL: {result.get('url', 'No URL')}\n"
            response += f"Snippet: {result.get('snippet', 'No snippet')}\n\n"
        
        return response
