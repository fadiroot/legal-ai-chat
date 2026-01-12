"""
API endpoints for document ingestion and processing.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
from azure.search.documents import SearchClient

from app.db.deps import get_search_client
from app.services.pdf_service import PDFService
from app.services.chunk_service import ChunkService
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService
from app.services.index_service import IndexService


router = APIRouter(prefix="/documents", tags=["documents"])


class CreateIndexRequest(BaseModel):
    """Request model for creating index."""
    vector_dimension: int = 3072
    index_name: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "vector_dimension": 3072,
                "index_name": "pdf-documents-index"
            }
        }


@router.post("/upload-multiple")
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    search_client: SearchClient = Depends(get_search_client)
):
    """
    Upload and process multiple PDF documents at once.
    
    Args:
        files: List of PDF files to upload and process
        
    Returns:
        Summary of all processed documents with individual results
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    total_pages = 0
    total_chunks = 0
    total_embeddings = 0
    
    for file_idx, file in enumerate(files, 1):
        try:
            if not file.filename.endswith('.pdf'):
                results.append({
                    "status": "error",
                    "document_name": file.filename,
                    "error": "Only PDF files are supported",
                    "file_index": file_idx
                })
                continue
            
            pdf_bytes = await file.read()
            document_name = file.filename
            
            pdf_service = PDFService()
            pages = pdf_service.extract_text_from_bytes(pdf_bytes, document_name)
            
            if not pages:
                results.append({
                    "status": "error",
                    "document_name": document_name,
                    "error": "No text extracted from PDF",
                    "file_index": file_idx
                })
                continue
            
            chunk_service = ChunkService()
            chunks = chunk_service.chunk_pages(pages, document_name)
            
            if not chunks:
                results.append({
                    "status": "error",
                    "document_name": document_name,
                    "error": "No chunks created from document",
                    "file_index": file_idx
                })
                continue
            
            embedding_service = EmbeddingService()
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = embedding_service.generate_embeddings(chunk_texts)
            
            valid_embeddings = 0
            for chunk, embedding in zip(chunks, embeddings):
                if embedding:
                    chunk.embedding = embedding
                    valid_embeddings += 1
            
            search_service = SearchService(search_client)
            search_service.index_chunks_batch(chunks, embeddings)
            
            total_pages += len(pages)
            total_chunks += len(chunks)
            total_embeddings += valid_embeddings
            
            results.append({
                "status": "success",
                "document_name": document_name,
                "pages_processed": len(pages),
                "chunks_created": len(chunks),
                "chunks_indexed": len(chunks),
                "embeddings_generated": valid_embeddings,
                "file_index": file_idx
            })
            
        except Exception as e:
            error_msg = str(e)
            results.append({
                "status": "error",
                "document_name": file.filename if hasattr(file, 'filename') else "unknown",
                "error": error_msg,
                "file_index": file_idx
            })
    
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    
    return {
        "status": "completed",
        "total_files": len(files),
        "successful": successful,
        "failed": failed,
        "total_pages_processed": total_pages,
        "total_chunks_created": total_chunks,
        "total_embeddings_generated": total_embeddings,
        "results": results
    }


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    search_client: SearchClient = Depends(get_search_client)
):
    """
    Upload and process a single PDF document.
    
    Returns:
        Summary of processed document
    """
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        pdf_bytes = await file.read()
        document_name = file.filename
        
        pdf_service = PDFService()
        pages = pdf_service.extract_text_from_bytes(pdf_bytes, document_name)
        
        if not pages:
            raise HTTPException(status_code=400, detail="No text extracted from PDF")
        
        chunk_service = ChunkService()
        chunks = chunk_service.chunk_pages(pages, document_name)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks created from document")
        
        embedding_service = EmbeddingService()
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = embedding_service.generate_embeddings(chunk_texts)
        
        valid_embeddings = 0
        for chunk, embedding in zip(chunks, embeddings):
            if embedding:
                chunk.embedding = embedding
                valid_embeddings += 1
        
        search_service = SearchService(search_client)
        search_service.index_chunks_batch(chunks, embeddings)
        
        return {
            "status": "success",
            "document_name": document_name,
            "pages_processed": len(pages),
            "chunks_created": len(chunks),
            "chunks_indexed": len(chunks),
            "embeddings_generated": valid_embeddings,
            "message": f"Successfully processed {document_name}"
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.post("/create-index")
async def create_index(request: CreateIndexRequest = CreateIndexRequest()):
    """
    Create the Azure Cognitive Search index with the proper schema for vector search.
    
    Args:
        request: Request with vector_dimension (default: 3072 for text-embedding-3-large)
        
    Returns:
        Status of index creation
    """
    try:
        index_service = IndexService(index_name=request.index_name)
        
        if index_service.index_exists():
            info = index_service.get_index_info()
            return {
                "status": "exists",
                "message": f"Index '{index_service.index_name}' already exists",
                "index_info": info,
                "suggestion": "Use a different index_name in the request, or change AZURE_AI_SEARCH_INDEX_NAME in .env file"
            }
        
        success = index_service.create_index(vector_dimension=request.vector_dimension)
        
        if success:
            info = index_service.get_index_info()
            return {
                "status": "created",
                "message": f"Index '{index_service.index_name}' created successfully",
                "index_info": info
            }
        else:
            return {
                "status": "failed",
                "message": f"Failed to create index '{index_service.index_name}'"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating index: {str(e)}")


@router.get("/index-info")
async def get_index_info():
    """
    Get information about the current search index.
    
    Returns:
        Information about the index including fields and configuration
    """
    try:
        index_service = IndexService()
        info = index_service.get_index_info()
        
        return {
            "status": "success",
            "index_name": index_service.index_name,
            "index_info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting index info: {str(e)}")


@router.delete("/delete-index")
async def delete_index():
    """
    Delete the Azure Cognitive Search index.
    
    ⚠️ WARNING: This will delete all indexed documents!
    
    Returns:
        Status of index deletion
    """
    try:
        index_service = IndexService()
        success = index_service.delete_index()
        
        if success:
            return {
                "status": "deleted",
                "message": f"Index '{index_service.index_name}' deleted successfully"
            }
        else:
            return {
                "status": "not_found",
                "message": f"Index '{index_service.index_name}' does not exist"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting index: {str(e)}")


@router.get("/list")
async def list_documents(
    document_name: Optional[str] = None,
    top: int = 100,
    search_client: SearchClient = Depends(get_search_client)
):
    """
    List all indexed documents/chunks.
    
    Args:
        document_name: Optional filter to get chunks for a specific document
        top: Maximum number of results to return (default: 100, max: 1000)
        
    Returns:
        List of indexed chunks
    """
    try:
        search_service = SearchService(search_client)
        
        if document_name:
            chunks = search_service.get_documents_by_name(document_name)
            return {
                "status": "success",
                "document_name": document_name,
                "chunk_count": len(chunks),
                "chunks": chunks
            }
        else:
            chunks = search_service.list_all_documents(top=min(top, 1000))
            return {
                "status": "success",
                "total_chunks": len(chunks),
                "chunks": chunks
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


