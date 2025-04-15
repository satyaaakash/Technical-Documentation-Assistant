"""
Module for processing and ingesting documents
"""

import os
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import json

class DocumentProcessor:
    def __init__(self):
        """Initialize the document processor"""
        pass
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove code block markers but keep the content
        text = re.sub(r'```\w*\n', '', text)
        text = re.sub(r'```', '', text)
        return text
        
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text (str): Text to split
            chunk_size (int): Approximate characters per chunk
            overlap (int): Character overlap between chunks
            
        Returns:
            List[str]: List of text chunks
        """
        text = self._clean_text(text)
        
        # Split by paragraphs first
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If paragraph fits in current chunk, add it
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If paragraph is larger than chunk_size, split it
                if len(para) > chunk_size:
                    words = para.split()
                    current_chunk = ""
                    
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= chunk_size:
                            current_chunk += word + " "
                        else:
                            # Add overlap
                            overlap_point = max(0, len(current_chunk) - overlap)
                            chunks.append(current_chunk.strip())
                            current_chunk = current_chunk[overlap_point:] + word + " "
                else:
                    current_chunk = para + "\n\n"
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def process_github_readme(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        Process README from a GitHub repository
        
        Args:
            owner (str): GitHub repository owner
            repo (str): GitHub repository name
            
        Returns:
            List[Dict]: List of processed document chunks with metadata
        """
        # Construct the URL for the README
        api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        
        # Get the README content
        response = requests.get(api_url, headers={"Accept": "application/vnd.github.v3.raw"})
        
        if response.status_code != 200:
            print(f"Error fetching README: {response.status_code}")
            return []
            
        # Get the content
        content = response.text
        
        # Chunk the content
        chunks = self.chunk_text(content)
        
        # Create document objects
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "metadata": {
                    "source": f"https://github.com/{owner}/{repo}",
                    "document_type": "github_readme",
                    "repo_owner": owner,
                    "repo_name": repo,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            })
            
        return documents
    
    def process_webpage(self, url: str) -> List[Dict[str, Any]]:
        """
        Process content from a webpage
        
        Args:
            url (str): URL of the webpage
            
        Returns:
            List[Dict]: List of processed document chunks with metadata
        """
        # Get the webpage content
        try:
            response = requests.get(
                url, 
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Error fetching webpage {url}: {e}")
            return []
            
        # Parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get the text content
        # Focus on the main content sections that are likely to contain documentation
        content_sections = []
        
        # Try to find main content section
        main = soup.find("main")
        if main:
            content_sections.append(main.get_text())
        else:
            # Try to find article or div with content-like class names
            for tag in ["article", "div"]:
                for section in soup.find_all(tag, class_=lambda c: c and any(x in str(c).lower() for x in ["content", "main", "article", "documentation", "docs"])):
                    content_sections.append(section.get_text())
                    
        # If no content sections found, use the body
        if not content_sections:
            content_sections.append(soup.get_text())
            
        # Join all content
        content = "\n\n".join(content_sections)
        
        # Chunk the content
        chunks = self.chunk_text(content)
        
        # Create document objects
        documents = []
        title = soup.title.string if soup.title else url
        
        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "metadata": {
                    "source": url,
                    "title": title,
                    "document_type": "webpage",
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            })
            
        return documents
    
    def save_documents(self, documents: List[Dict[str, Any]], output_path: str):
        """
        Save processed documents to a JSON file
        
        Args:
            documents (List[Dict]): List of processed documents
            output_path (str): Path to save the documents
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save documents
        with open(output_path, "w") as f:
            json.dump(documents, f, indent=2)
            
        print(f"Saved {len(documents)} documents to {output_path}")
