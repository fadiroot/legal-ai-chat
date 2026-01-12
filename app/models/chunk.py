"""
Chunk model for storing document chunks with metadata.
"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class Chunk(BaseModel):
    """Represents a chunk of text from a document."""
    id: str
    content: str
    document_name: str
    page_number: int
    chunk_index: int
    token_count: int
    embedding: Optional[list[float]] = None
    created_at: Optional[datetime] = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chunk_123",
                "content": "This is a sample chunk of legal text...",
                "document_name": "contract.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "token_count": 350,
                "embedding": None,
                "created_at": "2024-01-01T00:00:00"
            }
        }


