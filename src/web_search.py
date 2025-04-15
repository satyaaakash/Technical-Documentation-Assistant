"""
Module for performing web searches
"""

import requests
from typing import List, Dict, Any
import os
import json
from time import sleep
from bs4 import BeautifulSoup

class WebSearcher:
    def __init__(self, cache_dir: str = "data/search_cache"):
        """
        Initialize the web searcher
        
        Args:
            cache_dir (str): Directory to cache search results
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a web search
        
        Args:
            query (str): Search query
            num_results (int): Number of results to return
            
        Returns:
            List[Dict]: List of search results with metadata
        """
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"{query.replace(' ', '_')[:100]}.json")
        
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return json.load(f)
        
        # Perform search using SearX or a similar open search API
        # For demonstration, we'll use a mockup
        results = self._mock_search(query, num_results)
        
        # Cache results
        with open(cache_file, "w") as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def _mock_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """
        Mock search function for development (can be replaced with real search API)
        
        Args:
            query (str): Search query
            num_results (int): Number of results to return
            
        Returns:
            List[Dict]: List of mock search results
        """
        # In a real implementation, you would call an actual search API here
        # For now, we'll create a simulated response
        
        if "python" in query.lower():
            return [
                {
                    "title": "Python Documentation",
                    "url": "https://docs.python.org/3/",
                    "snippet": "Python 3 documentation. Welcome! This is the documentation for Python 3.10.0."
                },
                {
                    "title": "The Python Tutorial",
                    "url": "https://docs.python.org/3/tutorial/",
                    "snippet": "Python is an easy to learn, powerful programming language. It has efficient high-level data structures..."
                }
            ][:num_results]
        elif "pytorch" in query.lower() or "torch" in query.lower():
            return [
                {
                    "title": "PyTorch Documentation",
                    "url": "https://pytorch.org/docs/stable/",
                    "snippet": "PyTorch is an optimized tensor library for deep learning using GPUs and CPUs."
                },
                {
                    "title": "PyTorch Tutorials",
                    "url": "https://pytorch.org/tutorials/",
                    "snippet": "Learn how to use PyTorch through beginner-friendly tutorials."
                }
            ][:num_results]
        else:
            return [
                {
                    "title": f"Search result for {query} - 1",
                    "url": "https://example.com/result1",
                    "snippet": f"This is a sample search result for {query}."
                },
                {
                    "title": f"Search result for {query} - 2",
                    "url": "https://example.com/result2",
                    "snippet": f"Another sample search result for {query}."
                }
            ][:num_results]
    
    def fetch_webpage_content(self, url: str) -> str:
        """
        Fetch content from a webpage
        
        Args:
            url (str): URL of the webpage
            
        Returns:
            str: Text content of the webpage
        """
        try:
            response = requests.get(
                url, 
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            # Get text and clean it up
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            print(f"Error fetching webpage {url}: {e}")
            return ""
