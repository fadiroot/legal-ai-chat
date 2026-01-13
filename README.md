# Legal AI Chat API with Microsoft Agent Framework

A sophisticated multi-agent AI system for processing legal questions with role-based access control, document-based answers, and compliance validation. Built with FastAPI and Microsoft Agent Framework (MAF).

## 🚀 Features

- **Multi-Agent Workflow**: 6 specialized agents working in sequence using Microsoft Agent Framework
- **Role-Based Access Control**: Employee type and level classification with access validation
- **Document-Based Answers**: Strict citation requirements - all answers must come from indexed documents
- **Hallucination Detection**: Validation agent checks for unsupported claims and citation errors
- **Auto-Approval**: Direct results for low-risk questions (currently all requests are auto-approved)
- **Compliance Focus**: Saudi legal and regulatory compliance built-in
- **Audit Trail**: Complete execution trace for all agent interactions
- **Vector Search**: Azure AI Search with semantic similarity for document retrieval
- **Multi-Language Support**: Arabic and English question processing

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [MAF Workflow](#maf-workflow)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)

## 🏗️ Architecture

### System Overview

```
User Request → FastAPI → MAF Orchestrator → Agent Workflow → Response
```

### MAF Workflow (6 Agents)

1. **Classification Agent**: Extracts employee type and level from questions
2. **User Profiling Agent**: Validates access and normalizes classification
3. **Legal Retrieval Agent**: Role-aware document retrieval using Azure AI Search
4. **Legal Reasoning Agent**: Generates answers strictly from retrieved documents
5. **Validation Agent**: Detects hallucinations and ensures compliance
6. **Approval Agent**: Manages human-in-the-loop (currently auto-approves all)

### Technology Stack

- **Framework**: FastAPI
- **Agent Framework**: Microsoft Agent Framework (`agent-framework`)
- **Vector Search**: Azure AI Search
- **LLM**: Azure OpenAI (GPT-4, text-embedding-3-large)
- **Document Processing**: Azure Document Intelligence
- **State Management**: Pydantic models with JSON serialization

## 📦 Installation

### Prerequisites

- Python 3.12+ (or Python 3.10+)
- Azure account with:
  - Azure OpenAI (GPT-4 and text-embedding-3-large deployments)
  - Azure AI Search
  - Azure Document Intelligence

### Quick Setup

1. **Clone the repository**:
```bash
cd "document pip"
```

2. **Run setup script**:
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Create a virtual environment
- Install all dependencies
- Install `agent-framework --pre`
- Create `.env` file from template (if exists)
- Start the server

### Manual Setup

1. **Create virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install agent-framework --pre
```

3. **Configure environment variables**:
Create a `.env` file with:

```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large

# Azure AI Search
AZURE_AI_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_AI_SEARCH_API_KEY=your-search-key
AZURE_AI_SEARCH_INDEX_NAME=legal-documents-index

# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-doc-intel.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-doc-intel-key
```

4. **Start the server**:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## ⚙️ Configuration

### Agent Configuration

The orchestrator can be configured via the `get_orchestrator()` function in `app/api/maf_api.py`:

```python
config = {
    "max_retries": 3,
    "timeout_seconds": 300,
    "agents": {
        "classification": {
            "confidence_threshold": 0.7,
            "auto_clarify": True
        },
        "user_profiling": {
            "cache_enabled": True,
            "profile_expiry_hours": 1
        },
        "legal_retrieval": {
            "default_top_k": 5,
            "max_top_k": 20,
            "min_score_threshold": 0.7
        },
        "legal_reasoning": {
            "min_chunks_for_answer": 1,
            "max_chunks_to_process": 10
        },
        "validation": {
            "similarity_threshold": 0.3,
            "max_unsupported_claims": 2
        },
        "approval": {
            "auto_approval_enabled": True,
            "high_risk_requires_approval": True
        }
    }
}
```

## 📡 API Endpoints

### Root & Health

- `GET /` - API information and available endpoints
- `GET /health` - Health check

### MAF Endpoints (Recommended)

- `POST /maf/ask` - Ask a question using the full MAF pipeline
- `POST /maf/clarify` - Respond to clarification questions
- `GET /maf/approvals/pending` - Get pending approval requests
- `POST /maf/approvals/action` - Approve/reject requests
- `GET /maf/health` - MAF system health check
- `GET /maf/statistics` - Execution statistics
- `GET /maf/agents/info` - Agent information and workflow
- `GET /maf/monitor/agents` - Agent performance monitoring
- `GET /maf/monitor/pipeline` - Pipeline health monitoring

### Legacy Endpoints

- `POST /ask` - Simple Q&A (without MAF workflow)
- `POST /documents/upload` - Upload single PDF document
- `POST /documents/upload-multiple` - Upload multiple PDFs
- `POST /documents/create-index` - Create search index
- `GET /documents/index-info` - Get index information
- `GET /documents/list` - List indexed documents

### Interactive API Documentation

- `GET /docs` - Swagger UI documentation
- `GET /redoc` - ReDoc documentation

## 🔄 MAF Workflow

### How It Works

1. **Request Arrives**: User sends question to `/maf/ask`
2. **Orchestrator Initializes**: Creates/retrieves MAF orchestrator instance
3. **State Creation**: Initial `AgentState` created with question and session ID
4. **Workflow Execution**: State passed through 6 agents sequentially:

```
Initial State (JSON)
    ↓
Classification Agent → Extracts employee type/level
    ↓
User Profiling Agent → Validates access
    ↓
Legal Retrieval Agent → Searches documents
    ↓
Legal Reasoning Agent → Generates answer
    ↓
Validation Agent → Checks quality/compliance
    ↓
Approval Agent → Auto-approves
    ↓
Final State (JSON) → Converted to Response
```

5. **Response Generation**: Final state converted to `FinalResponse` with answer, sources, and metadata

### State Passing

- Each agent receives state as a JSON string
- Parses to `AgentState` Pydantic model
- Executes agent-specific logic
- Updates state with results
- Serializes back to JSON and sends to next agent

### Agent Details

#### 1. Classification Agent
- **Purpose**: Extract employee type and level from question
- **Output**: `classification_result` with employee type, level, category
- **Can Request Clarification**: If employee type/level is unclear

#### 2. User Profiling Agent
- **Purpose**: Validate access and normalize classification
- **Output**: `user_profile` with validated employee info
- **Features**: Access validation, user context management

#### 3. Legal Retrieval Agent
- **Purpose**: Role-aware document retrieval
- **Input**: Question + user profile
- **Output**: `retrieval_result` with relevant document chunks
- **Technology**: Azure AI Search vector similarity search

#### 4. Legal Reasoning Agent
- **Purpose**: Generate answer from documents only
- **Input**: Question + retrieved chunks
- **Output**: `reasoning_result` with answer and citations
- **Technology**: Azure OpenAI GPT-4

#### 5. Validation Agent
- **Purpose**: Detect hallucinations and verify citations
- **Input**: Answer + source documents
- **Output**: `validation_result` with confidence, issues, compliance status
- **Checks**: Unsupported claims, citation errors, compliance

#### 6. Approval Agent
- **Purpose**: Manage human-in-the-loop (currently auto-approves all)
- **Input**: Validation result
- **Output**: `approval_result` with approval status
- **Current Behavior**: Always auto-approves for direct results

## 💡 Usage Examples

### Ask a Question (MAF Pipeline)

```bash
curl -X POST "http://localhost:8000/maf/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "I am an Associate Engineer. How many years of experience do I need to reach the Professional Engineer level?",
    "session_id": "user-session-123"
  }'
```

**Response**:
```json
{
  "answer": "Based on the engineering regulations, Associate Engineers need 3 years of experience...",
  "sources": [
    {
      "document": "engineering_regulations.pdf",
      "article": "Article 15",
      "content": "Promotion requirements for Professional Engineer..."
    }
  ],
  "employee_type": "Engineer",
  "level": "Associate",
  "confidence": "high",
  "needs_human_review": false,
  "session_id": "user-session-123",
  "processing_time_ms": 2500,
  "agent_trace": [...]
}
```

### Upload Documents

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@engineering_regulations.pdf"
```

### Create Search Index

```bash
curl -X POST "http://localhost:8000/documents/create-index" \
  -H "Content-Type: application/json" \
  -d '{
    "vector_dimension": 3072
  }'
```

### Get System Statistics

```bash
curl "http://localhost:8000/maf/statistics"
```

## 📁 Project Structure

```
.
├── app/
│   ├── agents/              # MAF agents
│   │   ├── __init__.py
│   │   ├── base_agent.py     # Base agent class
│   │   ├── orchestrator.py   # MAF orchestrator
│   │   ├── schemas.py        # Pydantic models
│   │   ├── classification_agent.py
│   │   ├── user_profiling_agent.py
│   │   ├── legal_retrieval_agent.py
│   │   ├── legal_reasoning_agent.py
│   │   ├── validation_agent.py
│   │   └── approval_agent.py
│   ├── api/                  # API endpoints
│   │   ├── ask_api.py        # Legacy Q&A endpoint
│   │   ├── document_api.py   # Document upload/management
│   │   └── maf_api.py        # MAF endpoints
│   ├── services/             # Business logic services
│   │   ├── answer_service.py
│   │   ├── chunk_service.py
│   │   ├── embedding_service.py
│   │   ├── index_service.py
│   │   ├── pdf_service.py
│   │   └── search_service.py
│   ├── db/                   # Database dependencies
│   │   ├── client.py
│   │   └── deps.py
│   ├── models/               # Data models
│   │   └── chunk.py
│   └── main.py               # FastAPI application
├── requirements.txt          # Python dependencies
├── setup.sh                  # Setup script
└── README.md                 # This file
```

## 🔍 Key Concepts

### Agent State

The `AgentState` Pydantic model carries data through the workflow:
- `session_id`: Unique session identifier
- `user_question`: Original question
- `classification_result`: Employee type/level classification
- `user_profile`: Validated user profile
- `retrieval_result`: Retrieved document chunks
- `reasoning_result`: Generated answer with citations
- `validation_result`: Validation and compliance checks
- `approval_result`: Approval status
- `metadata`: Execution trace and additional data

### Workflow Execution

The workflow uses Microsoft Agent Framework's graph-based execution:
- Each agent is an executor function
- State is passed as JSON strings between agents
- The framework handles message routing and error propagation
- Execution is asynchronous and can be monitored

### Error Handling

- Each agent catches exceptions and continues workflow
- Errors are logged in `agent_trace`
- Failed agents mark state with error information
- Orchestrator generates error responses if workflow fails

## 🛠️ Development

### Running in Development Mode

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Debug Mode

Enable debug logging for agents:

```bash
curl -X POST "http://localhost:8000/maf/debug/enable" \
  -H "Content-Type: application/json" \
  -d '{
    "enable_debug": true
  }'
```

### Monitoring

- **Agent Performance**: `GET /maf/monitor/agents`
- **Pipeline Health**: `GET /maf/monitor/pipeline`
- **Session Activity**: `GET /maf/monitor/sessions`

## 📝 Notes

- **agent-framework**: Requires `--pre` flag during installation (pre-release package)
- **Auto-Approval**: Currently all requests are auto-approved for direct results
- **State Serialization**: State is passed as JSON strings between agents for compatibility with agent-framework
- **Vector Dimensions**: Default is 3072 for `text-embedding-3-large` model

## 🤝 Contributing

This is a production system. Please ensure:
- All tests pass
- Code follows existing patterns
- Documentation is updated
- Agent-framework compatibility is maintained

## 📄 License

[Add your license information here]

## 🔗 Related Documentation

- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Azure AI Search](https://learn.microsoft.com/azure/search/)
- [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/)
