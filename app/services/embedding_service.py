"""
Embedding service using Azure OpenAI.
"""
import os
from typing import List, Optional
from openai import AzureOpenAI


class EmbeddingService:
    """Service for generating embeddings using Azure OpenAI."""
    
    def __init__(self):
        self.aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.aoai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.aoai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        self.aoai_embed_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        if not self.aoai_embed_deployment:
            raise ValueError(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME is required. "
                "Please set it in your .env file with the name of your embedding deployment."
            )
        
        if not self.aoai_endpoint or not self.aoai_api_key:
            raise ValueError(
                "Azure OpenAI credentials not found. "
                "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env"
            )
        
        self._embedding_client: Optional[AzureOpenAI] = None
    
    def _get_embedding_client(self):
        """Get or create Azure OpenAI embedding client."""
        if self._embedding_client is None and self.aoai_endpoint and self.aoai_api_key:
            self._embedding_client = AzureOpenAI(
                api_key=self.aoai_api_key,
                api_version=self.aoai_api_version,
                azure_endpoint=self.aoai_endpoint
            )
        return self._embedding_client
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch."""
        client = self._get_embedding_client()
        response = client.embeddings.create(
            input=texts,
            model=self.aoai_embed_deployment
        )
        return [data.embedding for data in response.data]
    
    def list_available_deployments(self) -> List[str]:
        """List available deployments in Azure OpenAI resource for debugging."""
        client = self._get_embedding_client()
        if not client:
            return []
        
        try:
            models = client.models.list()
            return [model.id for model in models]
        except Exception:
            return []