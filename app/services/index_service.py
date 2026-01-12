"""
Service for creating and managing Azure Cognitive Search indexes.
"""
import os
from typing import Optional
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchAlgorithmKind,
    SearchField
)
from azure.core.credentials import AzureKeyCredential


class IndexService:
    """Service for managing Azure Cognitive Search indexes."""
    
    def __init__(self, index_name: Optional[str] = None):
        self.endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
        self.api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
        self.index_name = index_name or os.getenv("AZURE_AI_SEARCH_INDEX_NAME", "legal-documents-index")
        
        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Azure Cognitive Search credentials not found. "
                "Please set AZURE_AI_SEARCH_ENDPOINT and AZURE_AI_SEARCH_API_KEY in .env"
            )
        
        self.credential = AzureKeyCredential(self.api_key)
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=self.credential
        )
    
    def create_index(self, vector_dimension: int = 3072) -> bool:
        """
        Create the search index with the proper schema for vector search.
        
        Args:
            vector_dimension: Dimension of the embedding vectors (default: 3072 for text-embedding-3-large)
            
        Returns:
            True if successful
        """
        try:
            try:
                existing_index = self.index_client.get_index(self.index_name)
                return False
            except Exception:
                pass
            
            fields = [
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    searchable=False,
                    filterable=False,
                    sortable=False,
                    facetable=False
                ),
                SearchField(
                    name="content",
                    type=SearchFieldDataType.String,
                    searchable=True,
                    filterable=False,
                    sortable=False,
                    facetable=False,
                    analyzer_name="en.microsoft"
                ),
                SimpleField(
                    name="document_name",
                    type=SearchFieldDataType.String,
                    searchable=False,
                    filterable=True,
                    sortable=True,
                    facetable=True
                ),
                SimpleField(
                    name="page_number",
                    type=SearchFieldDataType.Int32,
                    searchable=False,
                    filterable=True,
                    sortable=True,
                    facetable=True
                ),
                SimpleField(
                    name="chunk_index",
                    type=SearchFieldDataType.Int32,
                    searchable=False,
                    filterable=True,
                    sortable=True,
                    facetable=False
                ),
                SimpleField(
                    name="token_count",
                    type=SearchFieldDataType.Int32,
                    searchable=False,
                    filterable=True,
                    sortable=True,
                    facetable=False
                ),
                SearchField(
                    name="contentVector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=vector_dimension,
                    vector_search_profile_name="vector-profile"
                )
            ]
            
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="vector-profile",
                        algorithm_configuration_name="hnsw-config"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="hnsw-config",
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters=HnswParameters(
                            m=4,
                            ef_construction=400,
                            ef_search=500,
                            metric="cosine"
                        )
                    )
                ]
            )
            
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            self.index_client.create_index(index)
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "already exists" in error_msg.lower():
                return False
            else:
                raise Exception(f"Failed to create index: {error_msg}")
    
    def delete_index(self) -> bool:
        """Delete the search index."""
        try:
            self.index_client.delete_index(self.index_name)
            return True
        except Exception as e:
            error_msg = str(e)
            if "not found" in error_msg.lower():
                return False
            else:
                raise Exception(f"Failed to delete index: {error_msg}")
    
    def index_exists(self) -> bool:
        """Check if the index exists."""
        try:
            self.index_client.get_index(self.index_name)
            return True
        except Exception:
            return False
    
    def get_index_info(self) -> dict:
        """Get information about the index."""
        try:
            index = self.index_client.get_index(self.index_name)
            return {
                "exists": True,
                "name": index.name,
                "fields": [
                    {
                        "name": f.name,
                        "type": str(f.type),
                        "key": getattr(f, 'key', False),
                        "searchable": getattr(f, 'searchable', False),
                        "filterable": getattr(f, 'filterable', False),
                        "vector_dimensions": getattr(f, 'vector_search_dimensions', None)
                    }
                    for f in index.fields
                ]
            }
        except Exception as e:
            return {
                "exists": False,
                "error": str(e)
            }
