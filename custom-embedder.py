# custom_embedder.py

import os
import requests
from typing import List
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv

load_dotenv()

class CustomAPIEmbeddings(Embeddings):
    """Custom embedding model using an external API."""
    
    def __init__(
        self, 
        api_url: str = None, 
        api_key: str = None, 
        model_name: str = "custom-embeddings",
        batch_size: int = 32
    ):
        """Initialize the custom embedding model.
        
        Args:
            api_url: URL of the embedding API
            api_key: API key for authentication
            model_name: Name of the embedding model
            batch_size: Batch size for embedding requests
        """
        self.api_url = api_url or os.getenv("EMBEDDING_API_URL")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY")
        self.model_name = model_name
        self.batch_size = batch_size
        
        if not self.api_url:
            raise ValueError("API URL must be provided either directly or via EMBEDDING_API_URL environment variable")
        
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json={"text": text, "model": self.model_name}
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("embedding", [])
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return a zero vector as fallback (adjust dimension as needed)
            return [0.0] * 1536
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_embeddings = [self._get_embedding(text) for text in batch]
            embeddings.extend(batch_embeddings)
            
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query."""
        return self._get_embedding(text)


# Example integration with our Text-to-SQL application
# Add this code to app.py to use a custom embedder

"""
from custom_embedder import CustomAPIEmbeddings

# In the TextToSQLApp __init__ method, replace OpenAIEmbeddings with:
self.embedding = CustomAPIEmbeddings(
    api_url=os.getenv("EMBEDDING_API_URL"),
    api_key=os.getenv("EMBEDDING_API_KEY"),
    model_name=os.getenv("EMBEDDING_MODEL_NAME", "custom-embeddings")
)
"""
