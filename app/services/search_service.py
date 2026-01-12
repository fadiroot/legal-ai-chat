"""
Vector search service using Azure Cognitive Search.
"""
from typing import List, Dict, Optional
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from app.models.chunk import Chunk


class SearchService:
    """Service for vector search operations using Azure Cognitive Search."""
    
    def __init__(self, search_client: SearchClient):
        """
        Initialize search service.
        
        Args:
            search_client: Azure Cognitive Search client
        """
        self.client = search_client
        self.index_name = self.client._index_name
    
    def index_chunk(self, chunk: Chunk, embedding: List[float]) -> bool:
        """
        Index a single chunk with its embedding into Azure Cognitive Search.
        
        Args:
            chunk: Chunk object to index
            embedding: Embedding vector for the chunk
            
        Returns:
            True if successful
        """
        try:
            document = {
                "id": chunk.id,
                "content": chunk.content,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "contentVector": embedding,
            }
            
            self.client.upload_documents(documents=[document])
            return True
        except Exception as e:
            raise Exception(f"Error indexing chunk: {str(e)}")
    
    def index_chunks_batch(self, chunks: List[Chunk], embeddings: List[List[float]]) -> bool:
        """
        Index multiple chunks with their embeddings in batch.
        
        Args:
            chunks: List of Chunk objects
            embeddings: List of embedding vectors (must match chunks order)
            
        Returns:
            True if successful
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        try:
            documents = []
            for chunk, embedding in zip(chunks, embeddings):
                document = {
                    "id": chunk.id,
                    "content": chunk.content,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "contentVector": embedding,
                }
                documents.append(document)
            
            batch_size = 1000
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for batch_num, i in enumerate(range(0, len(documents), batch_size), 1):
                batch = documents[i:i + batch_size]
                result = self.client.upload_documents(documents=batch)
                
                if hasattr(result, 'results'):
                    errors = [r for r in result.results if not r.succeeded]
                    if errors:
                        error_messages = [f"Document {r.key}: {r.error_message}" for r in errors]
                        raise Exception(f"Failed to index {len(errors)} documents: {', '.join(error_messages[:3])}")
            
            return True
        except Exception as e:
            error_msg = str(e)
            if "SSL" in error_msg or "_ssl" in error_msg:
                raise Exception(
                    f"SSL connection error when connecting to Azure Cognitive Search. "
                    f"This usually means:\n"
                    f"1. The endpoint URL is incorrect or malformed\n"
                    f"2. Network/firewall is blocking the connection\n"
                    f"3. SSL certificate validation failed\n"
                    f"Original error: {error_msg}"
                )
            elif "401" in error_msg or "Unauthorized" in error_msg:
                raise Exception(
                    f"Authentication failed. Check your AZURE_AI_SEARCH_API_KEY. "
                    f"Original error: {error_msg}"
                )
            elif "404" in error_msg or "not found" in error_msg.lower():
                raise Exception(
                    f"Index not found. Make sure the index '{self.index_name}' exists in Azure Cognitive Search. "
                    f"Original error: {error_msg}"
                )
            else:
                raise Exception(f"Error indexing chunks batch: {error_msg}")
    
    def search_similar_chunks(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        filter_expression: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Embedding vector of the query
            top_k: Number of top results to return
            filter_expression: Optional OData filter expression
            
        Returns:
            List of dictionaries containing:
            - content: Chunk text
            - document_name: Source document name
            - page_number: Page number
            - chunk_index: Chunk index
            - score: Similarity score
        """
        try:
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k,
                fields="contentVector"
            )
            
            search_options = {
                "vector_queries": [vector_query],
                "select": ["id", "content", "document_name", "page_number", "chunk_index", "token_count"],
                "top": top_k
            }
            
            if filter_expression:
                search_options["filter"] = filter_expression
            
            results = self.client.search(
                search_text="",
                **search_options
            )
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.get("content", ""),
                    "document_name": result.get("document_name", ""),
                    "page_number": result.get("page_number", 0),
                    "chunk_index": result.get("chunk_index", 0),
                    "score": result.get("@search.score", 0.0),
                    "id": result.get("id", "")
                })
            
            return formatted_results
        except Exception as e:
            raise Exception(f"Error searching similar chunks: {str(e)}")
    
    def list_all_documents(self, top: int = 100) -> List[Dict]:
        """
        List all documents/chunks in the index.
        
        Args:
            top: Maximum number of results to return (default: 100)
            
        Returns:
            List of all indexed chunks
        """
        try:
            results = self.client.search(
                search_text="*",
                select=["id", "content", "document_name", "page_number", "chunk_index", "token_count"],
                top=top
            )
            
            documents = []
            for result in results:
                documents.append({
                    "id": result.get("id", ""),
                    "content": result.get("content", ""),
                    "document_name": result.get("document_name", ""),
                    "page_number": result.get("page_number", 0),
                    "chunk_index": result.get("chunk_index", 0),
                    "token_count": result.get("token_count", 0)
                })
            
            return documents
        except Exception as e:
            raise Exception(f"Error listing documents: {str(e)}")
    
    def get_documents_by_name(self, document_name: str) -> List[Dict]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_name: Name of the document to retrieve
            
        Returns:
            List of chunks for the specified document
        """
        try:
            filter_expression = f"document_name eq '{document_name}'"
            results = self.client.search(
                search_text="*",
                filter=filter_expression,
                select=["id", "content", "document_name", "page_number", "chunk_index", "token_count"],
                top=1000
            )
            
            chunks = []
            for result in results:
                chunks.append({
                    "id": result.get("id", ""),
                    "content": result.get("content", ""),
                    "document_name": result.get("document_name", ""),
                    "page_number": result.get("page_number", 0),
                    "chunk_index": result.get("chunk_index", 0),
                    "token_count": result.get("token_count", 0)
                })
            
            chunks.sort(key=lambda x: (x["page_number"], x["chunk_index"]))
            return chunks
        except Exception as e:
            raise Exception(f"Error getting documents by name: {str(e)}")
    
    def get_document_statistics(self) -> Dict:
        """
        Get statistics about indexed documents.
        
        Returns:
            Dictionary with statistics about the index
        """
        try:
            all_docs = self.list_all_documents(top=10000)
            
            doc_stats = {}
            total_chunks = len(all_docs)
            total_tokens = 0
            
            for doc in all_docs:
                doc_name = doc["document_name"]
                if doc_name not in doc_stats:
                    doc_stats[doc_name] = {
                        "document_name": doc_name,
                        "chunk_count": 0,
                        "pages": set(),
                        "total_tokens": 0
                    }
                
                doc_stats[doc_name]["chunk_count"] += 1
                doc_stats[doc_name]["pages"].add(doc["page_number"])
                doc_stats[doc_name]["total_tokens"] += doc.get("token_count", 0)
                total_tokens += doc.get("token_count", 0)
            
            for doc_name in doc_stats:
                doc_stats[doc_name]["pages"] = sorted(list(doc_stats[doc_name]["pages"]))
                doc_stats[doc_name]["page_count"] = len(doc_stats[doc_name]["pages"])
            
            return {
                "total_documents": len(doc_stats),
                "total_chunks": total_chunks,
                "total_tokens": total_tokens,
                "documents": list(doc_stats.values())
            }
        except Exception as e:
            raise Exception(f"Error getting document statistics: {str(e)}")
