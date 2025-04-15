"""
Simple agent framework for coordinating tools
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple

class SimpleAgent:
    def __init__(
        self, 
        vector_search_func=None, 
        web_search_func=None,
        llm_func=None
    ):
        """
        Initialize the agent with tool functions
        
        Args:
            vector_search_func: Function to search vector database
            web_search_func: Function to search the web
            llm_func: Function to process with LLM
        """
        self.vector_search_func = vector_search_func
        self.web_search_func = web_search_func
        self.llm_func = llm_func
        
        # Default system prompt
        self.system_prompt = """
        You are a helpful Technical Documentation Assistant. Your goal is to provide
        accurate information by searching the knowledge base first, and only using
        web search if necessary. Always cite your sources.
        
        When answering:
        1. Be concise and accurate
        2. Include code examples where helpful
        3. Cite your sources using [Source: URL] format
        """
    
    def process_query(self, query: str) -> str:
        """
        Process a user query through the agent framework
        
        Args:
            query (str): User query
            
        Returns:
            str: Response to the query
        """
        # Step 1: Determine if we need vector search or web search
        search_decision_prompt = f"""
        User Query: {query}
        
        Determine which search method is most appropriate:
        - If this is about programming, software libraries, or technical documentation: Use VECTOR_SEARCH
        - If this is about current events, general information, or non-technical topics: Use WEB_SEARCH
        
        Output only one of: VECTOR_SEARCH or WEB_SEARCH
        """
        
        # If we have LLM function, use it to decide, otherwise default to vector search
        if self.llm_func:
            search_method = self.llm_func(search_decision_prompt).strip()
        else:
            # Default to vector search for development
            search_method = "VECTOR_SEARCH"
        
        # Step 2: Perform the appropriate search
        if search_method == "VECTOR_SEARCH" and self.vector_search_func:
            search_results = self.vector_search_func(query)
            source_type = "knowledge base"
        elif self.web_search_func:
            search_results = self.web_search_func(query)
            source_type = "web search"
        else:
            search_results = []
            source_type = "none"
        
        # Step 3: Generate a response using the LLM
        if not search_results:
            return f"I couldn't find any information related to your query in our {source_type}."
        
        # Format search results for the LLM
        formatted_results = self._format_search_results(search_results, source_type)
        
        response_prompt = f"""
        User Query: {query}
        
        {formatted_results}
        
        Based on the {source_type} results above, provide a helpful response to the user's query.
        Include relevant information and cite sources appropriately.
        """
        
        # Generate response with LLM if available, otherwise return raw results
        if self.llm_func:
            return self.llm_func(response_prompt)
        else:
            return f"Development mode: Query: {query}\nSearch method: {search_method}\nResults: {formatted_results}"
    
    def _format_search_results(self, results: List[Dict[str, Any]], source_type: str) -> str:
        """Format search results for the LLM prompt"""
        if source_type == "knowledge base":
            formatted = f"Knowledge Base Results ({len(results)} items):\n\n"
            for i, result in enumerate(results):
                formatted += f"Result {i+1}:\n"
                formatted += f"Text: {result.get('text', 'No text')}\n"
                formatted += f"Source: {result.get('metadata', {}).get('source', 'Unknown')}\n\n"
        else:  # web search
            formatted = f"Web Search Results ({len(results)} items):\n\n"
            for i, result in enumerate(results):
                formatted += f"Result {i+1}:\n"
                formatted += f"Title: {result.get('title', 'No title')}\n"
                formatted += f"URL: {result.get('url', 'No URL')}\n"
                formatted += f"Snippet: {result.get('snippet', 'No snippet')}\n\n"
                
        return formatted
