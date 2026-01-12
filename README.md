# Legal AI Chat System

**Version:** 2.0.0 (Enhanced with RAG v2)  
**Status:** ✅ **PRODUCTION READY**

A production-ready FastAPI-based Legal AI Chat system that ingests PDF documents, processes them into searchable chunks, and provides intelligent Q&A functionality using Azure services with **RAG v2** (Retrieval-Augmented Generation).

---

## 🚀 Quick Start

### One-Command Setup

```bash
./setup.sh
```

This will create a virtual environment, install dependencies, create `.env` file if needed, and start the FastAPI server.

**Access the API:**
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Before First Run

1. **Create `.env` file** (if not auto-created):
   ```bash
   cp env_template.txt .env
   ```

2. **Update `.env`** with your Azure credentials:
   - `AZURE_PROJECT_ENDPOINT` and `AZURE_PROJECT_API_KEY`
   - `AZURE_AI_SEARCH_ENDPOINT` and `AZURE_AI_SEARCH_API_KEY`
   - `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`
   - `AZURE_OPENAI_DEPLOYMENT_NAME` (e.g., `gpt-4o`)
   - `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` (e.g., `text-embedding-3-large`)

3. **Create Azure Cognitive Search Index**:
   ```bash
   curl -X POST "http://localhost:8000/documents/create-index" \
     -H "Content-Type: application/json" \
     -d '{"vector_dimension": 3072}'
   ```

---

## 🎯 Features

- **Document Ingestion**: Extract text from PDF files using Azure Document Intelligence
- **Text Chunking**: Split documents into 300-500 token chunks for optimal search
- **Vector Embeddings**: Generate embeddings using Azure OpenAI
- **Vector Search**: Index and search chunks using Azure Cognitive Search
- **Intelligent Q&A**: Get natural language answers with RAG v2 (LLM-generated, not raw chunks)
- **Bilingual Support**: Arabic and English with automatic language detection
- **Batch Processing**: Upload multiple files or process entire folders

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              DOCUMENT PROCESSING PIPELINE                   │
│                                                              │
│  1. PDF Upload/Selection                                    │
│     ↓                                                        │
│  2. Azure Document Intelligence (Text Extraction)           │
│     ↓                                                        │
│  3. Chunking Service (300-500 tokens per chunk)             │
│     ↓                                                        │
│  4. Azure OpenAI (Embedding Generation)                     │
│     ↓                                                        │
│  5. Azure Cognitive Search (Vector Indexing)                │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Q&A PIPELINE (RAG v2)                      │
│                                                              │
│  1. User Question (Arabic/English)                          │
│     ↓                                                        │
│  2. Question Embedding (Azure OpenAI)                       │
│     ↓                                                        │
│  3. Vector Search (Azure Cognitive Search)                  │
│     ↓                                                        │
│  4. Top-K Chunks Retrieved                                  │
│     ↓                                                        │
│  5. LLM Answer Generation (Azure OpenAI GPT-4o)             │
│     ↓                                                        │
│  6. Natural Language Answer + Source Citations              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.12 or higher
- Azure account with access to:
  - Azure Document Intelligence (or Azure AI Foundry)
  - Azure OpenAI (or Azure AI Foundry)
  - Azure Cognitive Search

### Manual Installation

1. **Create a virtual environment:**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   - Copy `env_template.txt` to `.env`
   - Update `.env` with your Azure credentials

4. **Create Azure Cognitive Search Index:**
   ```bash
   curl -X POST "http://localhost:8000/documents/create-index" \
     -H "Content-Type: application/json" \
     -d '{"vector_dimension": 3072}'
   ```

5. **Start the server:**
   ```bash
   uvicorn app.main:app --reload
   ```

---

## 💻 Usage Examples

### Upload Single Document

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"
```

### Upload Multiple Documents

```bash
curl -X POST "http://localhost:8000/documents/upload-multiple" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "files=@doc3.pdf"
```

### Ask Questions (Arabic)

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي فئات الموظفين الخاضعين لنظام الأجور؟"}'
```

### Ask Questions (English)

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key provisions?"}'
```

### Batch Process Folder

```bash
python batch_process_documents.py ./documents
```

---

## 📊 API Endpoints

### Document Management
- `POST /documents/upload` - Upload single PDF
- `POST /documents/upload-multiple` - Upload multiple PDFs
- `POST /documents/process-local` - Process local file
- `GET /documents/list` - List indexed documents
- `GET /documents/statistics` - Get document stats
- `POST /documents/create-index` - Create search index
- `DELETE /documents/delete-index` - Delete index

### Q&A
- `POST /ask` - Ask questions (RAG v2)
- `GET /ask/health` - Health check

### General
- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc docs

---

## 🔐 Configuration

### Required Environment Variables

- `AZURE_PROJECT_ENDPOINT` - Azure AI Foundry project endpoint
- `AZURE_PROJECT_API_KEY` - Azure AI Foundry API key
- `AZURE_AI_SEARCH_ENDPOINT` - Azure Cognitive Search endpoint
- `AZURE_AI_SEARCH_API_KEY` - Azure Cognitive Search API key
- `AZURE_AI_SEARCH_INDEX_NAME` - Search index name (default: `legal-documents-index`)
- `AZURE_OPENAI_ENDPOINT` - Azure OpenAI endpoint
- `AZURE_OPENAI_API_KEY` - Azure OpenAI API key
- `AZURE_OPENAI_DEPLOYMENT_NAME` - GPT deployment (e.g., `gpt-4o`)
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` - Embedding deployment (e.g., `text-embedding-3-large`)

---

## 🛑 Troubleshooting

### Common Issues

1. **Azure credentials not found**: Ensure `.env` file exists with all required variables
2. **Index not found**: Create the Azure Cognitive Search index using `/documents/create-index`
3. **Embedding dimension mismatch**: Ensure embedding model matches index dimensions (3072 for text-embedding-3-large)
4. **PDF extraction fails**: Verify Azure Document Intelligence credentials
5. **Port already in use**: Change port in `setup.sh` or use `--port 8001` with uvicorn

---

## 📝 License

This project is a starter template for Legal AI Chat systems.
