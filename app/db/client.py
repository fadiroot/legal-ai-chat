"""
Azure Cognitive Search client for vector search operations.
"""
import os
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from typing import Optional


class SearchClientManager:
    """Manages Azure Cognitive Search client connection."""
    
    def __init__(self):
        self.endpoint: Optional[str] = None
        self.api_key: Optional[str] = None
        self.index_name: Optional[str] = None
        self.credential: Optional[AzureKeyCredential] = None
        self.client: Optional[SearchClient] = None
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization - only called when client is actually needed."""
        if self._initialized:
            return
        
        self.endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
        self.api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
        self.index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME", "legal-documents-index")
        
        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Azure Cognitive Search credentials not found. "
                "Please set AZURE_AI_SEARCH_ENDPOINT and AZURE_AI_SEARCH_API_KEY in .env"
            )
        
        self.credential = AzureKeyCredential(self.api_key)
        self._initialized = True
    
    def get_client(self) -> SearchClient:
        """Get or create Azure Cognitive Search client."""
        self._initialize()
        
        if self.client is None:
            self.client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=self.credential
            )
        return self.client
    
    def close(self):
        """Close the search client connection."""
        if self.client:
            self.client.close()
            self.client = None


search_client_manager = SearchClientManager()
