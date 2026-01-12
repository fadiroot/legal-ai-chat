"""
FastAPI main application for Legal AI Chat system.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.api import ask_api, document_api


load_dotenv()

app = FastAPI(
    title="Legal AI Chat API",
    description="API for Legal AI Chat system with document ingestion and Q&A",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask_api.router)
app.include_router(document_api.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Legal AI Chat API",
        "version": "1.0.0",
        "endpoints": {
            "ask": "/ask",
            "documents": "/documents",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
