"""
FastAPI main application for Legal AI Chat system.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.api import ask_api, document_api, maf_api


load_dotenv()

app = FastAPI(
    title="Legal AI Chat API with Microsoft Agent Framework",
    description="API for Legal AI Chat system with document ingestion, Q&A, and MAF-powered multi-agent processing",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(ask_api.router)
app.include_router(document_api.router)
app.include_router(maf_api.router)  # New MAF endpoints


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Legal AI Chat API with Microsoft Agent Framework",
        "version": "2.0.0",
        "description": "Multi-agent, role-aware, legally safe AI assistant for Saudi regulations",
        "endpoints": {
            "legacy_ask": "/ask",
            "maf_ask": "/maf/ask",
            "maf_clarify": "/maf/clarify", 
            "maf_approvals": "/maf/approvals",
            "maf_health": "/maf/health",
            "maf_statistics": "/maf/statistics",
            "documents": "/documents",
            "docs": "/docs"
        },
        "features": [
            "Employee classification and role-aware responses",
            "Document-based answers with strict citation requirements",
            "Hallucination detection and compliance validation",
            "Human-in-the-loop approval for sensitive content",
            "Audit trail and performance monitoring",
            "Saudi legal and regulatory compliance"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
