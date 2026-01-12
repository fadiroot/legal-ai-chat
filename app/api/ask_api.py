"""
API endpoints for Q&A functionality.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from azure.search.documents import SearchClient

from app.db.deps import get_search_client
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService
from app.services.answer_service import AnswerService


router = APIRouter(prefix="/ask", tags=["ask"])


class AskRequest(BaseModel):
    """Request model for asking questions."""
    question: str
    top_k: Optional[int] = 5
    filter_document: Optional[str] = None  # Optional filter by document name
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the key terms of the contract?"
            }
        }


class AnswerResponse(BaseModel):
    """Response model for Q&A answers."""
    answer: str
    sources: List[dict]
    question: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the key terms of the contract?",
                "answer": "Based on the document, the key terms include...",
                "sources": [
                    {
                        "document_name": "contract.pdf",
                        "page_number": 1,
                        "chunk_index": 0,
                        "score": 0.95
                    }
                ]
            }
        }


@router.post("", response_model=AnswerResponse)
async def ask_question(
    request: AskRequest,
    search_client: SearchClient = Depends(get_search_client)
):
    """
    Ask a question and get an answer based on indexed documents.
    
    Uses RAG (Retrieval-Augmented Generation) to:
    1. Search for relevant document chunks using vector similarity
    2. Generate a natural language answer using Azure OpenAI GPT-4
    
    Args:
        request: AskRequest with question and optional parameters
        
    Returns:
        AnswerResponse with LLM-generated answer and source citations
    """
    try:
        # Step 1: Generate embedding for the question
        embedding_service = EmbeddingService()
        embeddings = embedding_service.generate_embeddings([request.question])
        question_embedding = embeddings[0] if embeddings else None
        
        if not question_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate embedding for question")
        
        # Step 2: Search for similar chunks
        search_service = SearchService(search_client)
        
        # Build filter expression if document filter is provided
        filter_expression = None
        if request.filter_document:
            filter_expression = f"document_name eq '{request.filter_document}'"
        
        similar_chunks = search_service.search_similar_chunks(
            query_embedding=question_embedding,
            top_k=request.top_k,
            filter_expression=filter_expression
        )
        
        if not similar_chunks:
            return AnswerResponse(
                question=request.question,
                answer="No relevant information found in the indexed documents.",
                sources=[]
            )
        
        # Step 3: Generate answer using LLM (RAG v2)
        answer_service = AnswerService()
        
        # Detect language from question
        has_arabic = any('\u0600' <= char <= '\u06FF' for char in request.question)
        language = "ar" if has_arabic else "en"
        
        # Generate natural language answer from chunks
        answer = answer_service.generate_answer(
            question=request.question,
            chunks=similar_chunks,
            language=language
        )
        
        # Step 4: Format sources for response (without content)
        sources = [
            {
                "document_name": chunk["document_name"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "score": chunk["score"]
            }
            for chunk in similar_chunks
        ]
        
        return AnswerResponse(
            question=request.question,
            answer=answer,
            sources=sources
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ask-api"}
