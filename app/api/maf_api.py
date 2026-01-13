"""
MAF API endpoints for the Microsoft Agent Framework Legal AI Assistant.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from azure.search.documents import SearchClient
from datetime import datetime

from app.db.deps import get_search_client
from app.agents.orchestrator import MAFOrchestrator
from app.agents.schemas import FinalResponse, ConfidenceLevel


router = APIRouter(prefix="/maf", tags=["maf"])


class MAFAskRequest(BaseModel):
    """Request model for MAF-powered questions."""
    question: str = Field(..., description="User question", min_length=10, max_length=2000)
    session_id: Optional[str] = Field(None, description="Optional session identifier for conversation tracking")
    user_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional user context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "ما هي شروط ترقية المهندس المحترف إلى مهندس استشاري؟",
                "session_id": "user-session-123",
                "user_context": {
                    "department": "engineering",
                    "location": "riyadh"
                }
            }
        }


class MAFAskResponse(BaseModel):
    """Response model for MAF-powered answers."""
    answer: str
    sources: List[Dict[str, Any]]
    employee_type: str
    level: Optional[str]
    confidence: str
    needs_human_review: bool
    session_id: str
    processing_time_ms: int
    agent_trace: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "بناءً على لائحة المهندسين، يتطلب الترقية من مهندس محترف إلى استشاري...",
                "sources": [
                    {
                        "document": "engineering_regulations.pdf",
                        "article": "Article 15",
                        "content": "Requirements for consultant engineer promotion..."
                    }
                ],
                "employee_type": "Engineer",
                "level": "Professional",
                "confidence": "high",
                "needs_human_review": False,
                "session_id": "user-session-123",
                "processing_time_ms": 2500,
                "agent_trace": []
            }
        }


class ClarificationRequest(BaseModel):
    """Request model for clarification responses."""
    session_id: str = Field(..., description="Session identifier from original request")
    clarification_response: str = Field(..., description="User's response to clarification question")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user-session-123",
                "clarification_response": "أنا مهندس محترف في وزارة النقل"
            }
        }


class ApprovalRequest(BaseModel):
    """Request model for approval actions."""
    request_id: str = Field(..., description="Approval request identifier")
    action: str = Field(..., description="Action to take: 'approve' or 'reject'")
    reviewer_id: str = Field(..., description="Reviewer identifier")
    notes: Optional[str] = Field(None, description="Reviewer notes or rejection reason")


class ApprovalResponse(BaseModel):
    """Response model for approval actions."""
    success: bool
    message: str
    request_id: str


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: str
    agents: Dict[str, Dict[str, Any]]
    overall_status: str
    timestamp: str


class StatisticsResponse(BaseModel):
    """Response model for statistics."""
    orchestrator_stats: Dict[str, Any]
    agent_stats: Dict[str, Any]
    configuration: Dict[str, Any]


class DebugRequest(BaseModel):
    """Request model for debug mode."""
    session_id: Optional[str] = Field(None, description="Session ID to debug")
    agent_name: Optional[str] = Field(None, description="Specific agent to debug")
    enable_debug: bool = Field(True, description="Enable debug mode")


class AgentTraceResponse(BaseModel):
    """Response model for agent trace."""
    session_id: str
    agent_traces: List[Dict[str, Any]]
    total_agents: int
    execution_summary: Dict[str, Any]


# Global orchestrator instance (in production, use dependency injection)
_orchestrator: Optional[MAFOrchestrator] = None


def get_orchestrator(search_client: SearchClient = Depends(get_search_client)) -> MAFOrchestrator:
    """Get or create MAF orchestrator instance."""
    global _orchestrator
    
    if _orchestrator is None:
        try:
            # Initialize orchestrator with configuration
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
                        "auto_approval_enabled": True,  # Enable auto-approval for low-risk cases
                        "high_risk_requires_approval": True
                    }
                }
            }
            
            _orchestrator = MAFOrchestrator(search_client=search_client, config=config)
        except ImportError as e:
            # Re-raise with clearer message for API
            raise HTTPException(
                status_code=503,
                detail=f"MAF system requires agent-framework package. {str(e)}. Please install with: pip install agent-framework --pre"
            ) from e
    
    return _orchestrator


@router.post("/ask", response_model=MAFAskResponse)
async def ask_question_maf(
    request: MAFAskRequest,
    orchestrator: MAFOrchestrator = Depends(get_orchestrator)
):
    """
    Ask a question using the Microsoft Agent Framework pipeline.
    
    This endpoint processes questions through the complete MAF workflow:
    1. Classification Agent - Identifies employee type and level
    2. User Profiling Agent - Validates access and manages user context
    3. Legal Retrieval Agent - Performs role-aware document retrieval
    4. Legal Reasoning Agent - Synthesizes answers from documents only
    5. Validation Agent - Detects hallucinations and ensures compliance
    6. Approval Agent - Manages human-in-the-loop for sensitive content
    
    Args:
        request: MAFAskRequest with question and optional context
        
    Returns:
        MAFAskResponse with comprehensive answer and metadata
    """
    try:
        # Process question through MAF pipeline
        response = await orchestrator.process_question(
            question=request.question,
            session_id=request.session_id
        )
        
        # Convert to API response format
        return MAFAskResponse(
            answer=response.answer,
            sources=response.sources,
            employee_type=response.employee_type,
            level=response.level,
            confidence=response.confidence.value,
            needs_human_review=response.needs_human_review,
            session_id=response.session_id,
            processing_time_ms=response.processing_time_ms or 0,
            agent_trace=response.agent_trace
        )
        
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"MAF system requires agent-framework package. {str(e)}. Please install with: pip install agent-framework --pre"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"MAF processing failed: {str(e)}"
        )


@router.post("/clarify", response_model=MAFAskResponse)
async def handle_clarification(
    request: ClarificationRequest,
    orchestrator: MAFOrchestrator = Depends(get_orchestrator)
):
    """
    Handle user response to clarification questions.
    
    When the Classification Agent requires clarification about employee type
    or level, users can provide additional information through this endpoint.
    
    Args:
        request: ClarificationRequest with session ID and clarification response
        
    Returns:
        MAFAskResponse with updated processing results
    """
    try:
        # Process clarification through orchestrator
        response = await orchestrator.handle_clarification_response(
            session_id=request.session_id,
            clarification_response=request.clarification_response
        )
        
        return MAFAskResponse(
            answer=response.answer,
            sources=response.sources,
            employee_type=response.employee_type,
            level=response.level,
            confidence=response.confidence.value,
            needs_human_review=response.needs_human_review,
            session_id=response.session_id,
            processing_time_ms=response.processing_time_ms or 0,
            agent_trace=response.agent_trace
        )
        
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"MAF system requires agent-framework package. {str(e)}. Please install with: pip install agent-framework --pre"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Clarification processing failed: {str(e)}"
        )


@router.get("/approvals/pending")
async def get_pending_approvals(
    priority: Optional[str] = None,
    orchestrator: MAFOrchestrator = Depends(get_orchestrator)
):
    """
    Get pending approval requests.
    
    Returns a list of requests that require human approval, filtered by
    priority if specified.
    
    Args:
        priority: Optional priority filter (low, medium, high, urgent)
        
    Returns:
        List of pending approval requests
    """
    try:
        pending_requests = orchestrator.get_pending_approvals()
        
        # Filter by priority if specified
        if priority:
            pending_requests = [
                req for req in pending_requests 
                if req.get("priority") == priority
            ]
        
        return {
            "pending_requests": pending_requests,
            "total_count": len(pending_requests),
            "filtered_by_priority": priority
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve pending approvals: {str(e)}"
        )


@router.post("/approvals/action", response_model=ApprovalResponse)
async def handle_approval_action(
    request: ApprovalRequest,
    orchestrator: MAFOrchestrator = Depends(get_orchestrator)
):
    """
    Approve or reject a pending request.
    
    Allows authorized reviewers to approve or reject requests that are
    pending human review.
    
    Args:
        request: ApprovalRequest with action details
        
    Returns:
        ApprovalResponse with action result
    """
    try:
        if request.action.lower() == "approve":
            success = orchestrator.approve_request(
                request_id=request.request_id,
                reviewer_id=request.reviewer_id,
                reviewer_notes=request.notes
            )
            message = "Request approved successfully" if success else "Failed to approve request"
            
        elif request.action.lower() == "reject":
            success = orchestrator.reject_request(
                request_id=request.request_id,
                reviewer_id=request.reviewer_id,
                rejection_reason=request.notes or "No reason provided"
            )
            message = "Request rejected successfully" if success else "Failed to reject request"
            
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid action. Must be 'approve' or 'reject'"
            )
        
        return ApprovalResponse(
            success=success,
            message=message,
            request_id=request.request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Approval action failed: {str(e)}"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    orchestrator: MAFOrchestrator = Depends(get_orchestrator)
):
    """
    Perform health check on the MAF system.
    
    Checks the health of the orchestrator and all agents in the pipeline.
    
    Returns:
        HealthCheckResponse with system health status
    """
    try:
        health_status = await orchestrator.health_check()
        
        return HealthCheckResponse(
            status=health_status["orchestrator"],
            agents=health_status["agents"],
            overall_status=health_status["overall_status"],
            timestamp=health_status["timestamp"]
        )
        
    except Exception as e:
        return HealthCheckResponse(
            status="unhealthy",
            agents={},
            overall_status="error",
            timestamp=datetime.now().isoformat()
        )


@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(
    orchestrator: MAFOrchestrator = Depends(get_orchestrator)
):
    """
    Get MAF system statistics and performance metrics.
    
    Returns execution statistics, agent performance data, and configuration
    information for monitoring and optimization purposes.
    
    Returns:
        StatisticsResponse with system statistics
    """
    try:
        stats = orchestrator.get_execution_statistics()
        
        return StatisticsResponse(
            orchestrator_stats=stats["orchestrator_stats"],
            agent_stats=stats["agent_stats"],
            configuration=stats["configuration"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@router.get("/agents/info")
async def get_agent_info():
    """
    Get information about available agents and their capabilities.
    
    Returns:
        Dictionary with agent information and workflow description
    """
    return {
        "workflow": [
            {
                "step": 1,
                "agent": "ClassificationAgent",
                "purpose": "Extract employee type and job category from user questions",
                "mandatory": True,
                "can_require_clarification": True
            },
            {
                "step": 2,
                "agent": "UserProfilingAgent", 
                "purpose": "Validate access level and normalize classification",
                "mandatory": True,
                "can_require_clarification": False
            },
            {
                "step": 3,
                "agent": "LegalRetrievalAgent",
                "purpose": "Perform role-aware, category-aware retrieval using Azure AI Search",
                "mandatory": True,
                "can_require_clarification": False
            },
            {
                "step": 4,
                "agent": "LegalReasoningAgent",
                "purpose": "Synthesize answers strictly from retrieved documents",
                "mandatory": True,
                "can_require_clarification": False
            },
            {
                "step": 5,
                "agent": "ValidationAgent",
                "purpose": "Detect hallucinations, verify citations, ensure compliance",
                "mandatory": True,
                "can_require_clarification": False
            },
            {
                "step": 6,
                "agent": "ApprovalAgent",
                "purpose": "Manage human-in-the-loop for sensitive cases",
                "mandatory": False,
                "can_require_clarification": False
            }
        ],
        "employee_types": [
            {
                "type": "Engineer",
                "levels": ["Associate", "Professional", "Consultant"],
                "access_documents": ["engineering_regulations", "technical_standards", "promotion_guidelines"]
            },
            {
                "type": "Wage Band Employee", 
                "levels": ["Ordinary", "Skilled", "Assistant Technician"],
                "access_documents": ["wage_band_regulations", "benefits_guide", "leave_policies"]
            },
            {
                "type": "Civil Servant",
                "levels": [],
                "access_documents": ["executive_hr_regulations", "administrative_procedures", "general_policies"]
            },
            {
                "type": "External/Citizen",
                "levels": [],
                "access_documents": ["public_information", "general_guidelines"]
            }
        ],
        "compliance_features": [
            "No hallucinations allowed",
            "All answers must be cited from documents",
            "Role-based access control",
            "Human approval for sensitive content",
            "Audit trail for all interactions",
            "Saudi legal compliance focus"
        ]
    }


@router.get("/debug/trace/{session_id}", response_model=AgentTraceResponse)
async def get_agent_trace(
    session_id: str,
    orchestrator: MAFOrchestrator = Depends(get_orchestrator)
):
    """
    Get detailed agent execution trace for a specific session.
    
    This endpoint allows you to debug and monitor the complete execution
    flow of all agents for a given session.
    
    Args:
        session_id: Session identifier from a previous request
        
    Returns:
        AgentTraceResponse with complete execution trace
    """
    try:
        # In a real implementation, you would retrieve the trace from storage
        # For now, we'll return a message indicating how to use this
        
        # Get execution statistics which may contain trace info
        stats = orchestrator.get_execution_statistics()
        
        return AgentTraceResponse(
            session_id=session_id,
            agent_traces=[],
            total_agents=6,
            execution_summary={
                "message": "Agent traces are included in the agent_trace field of each /maf/ask response",
                "how_to_use": "Make a request to /maf/ask and check the 'agent_trace' field in the response",
                "trace_structure": {
                    "agent_name": "Name of the agent",
                    "agent_id": "Unique agent instance ID",
                    "event": "started|completed|error",
                    "timestamp": "ISO timestamp",
                    "metadata": "Additional execution metadata"
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve agent trace: {str(e)}"
        )


@router.post("/debug/enable")
async def enable_debug_mode(
    request: DebugRequest,
    orchestrator: MAFOrchestrator = Depends(get_orchestrator)
):
    """
    Enable debug mode for agents.
    
    When debug mode is enabled, agents will log detailed information
    about their execution, including input/output states and intermediate results.
    
    Args:
        request: DebugRequest with configuration
        
    Returns:
        Success message with debug configuration
    """
    try:
        # Update orchestrator config to enable debug mode
        if request.enable_debug:
            # Enable debug for all agents
            for agent_name, _ in orchestrator.agent_pipeline:
                if agent_name in orchestrator.config.get("agents", {}):
                    orchestrator.config["agents"][agent_name]["debug"] = True
                else:
                    if "agents" not in orchestrator.config:
                        orchestrator.config["agents"] = {}
                    if agent_name not in orchestrator.config["agents"]:
                        orchestrator.config["agents"][agent_name] = {}
                    orchestrator.config["agents"][agent_name]["debug"] = True
        
        return {
            "success": True,
            "message": f"Debug mode {'enabled' if request.enable_debug else 'disabled'}",
            "configuration": orchestrator.config.get("agents", {}),
            "note": "Debug logs will appear in the server console/logs"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to configure debug mode: {str(e)}"
        )


@router.get("/monitor/agents")
async def monitor_agents(
    orchestrator: MAFOrchestrator = Depends(get_orchestrator)
):
    """
    Monitor individual agent performance and status.
    
    Returns detailed information about each agent's execution statistics,
    including success rates, average execution times, and recent errors.
    
    Returns:
        Dictionary with agent monitoring data
    """
    try:
        stats = orchestrator.get_execution_statistics()
        agent_stats = stats.get("agent_stats", {})
        
        # Format agent monitoring data
        agent_monitoring = {}
        
        for agent_name, agent_data in agent_stats.items():
            agent_monitoring[agent_name] = {
                "status": "healthy",
                "total_executions": agent_data.get("total_executions", 0),
                "successful_executions": agent_data.get("successful_executions", 0),
                "failed_executions": agent_data.get("total_executions", 0) - agent_data.get("successful_executions", 0),
                "average_execution_time_ms": agent_data.get("average_execution_time", 0),
                "success_rate": (
                    agent_data.get("successful_executions", 0) / agent_data.get("total_executions", 1) * 100
                    if agent_data.get("total_executions", 0) > 0 else 0
                )
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "agents": agent_monitoring,
            "overall_health": "healthy" if all(
                agent["status"] == "healthy" for agent in agent_monitoring.values()
            ) else "degraded"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve agent monitoring data: {str(e)}"
        )


@router.get("/monitor/pipeline")
async def monitor_pipeline(
    orchestrator: MAFOrchestrator = Depends(get_orchestrator)
):
    """
    Monitor the complete pipeline execution flow.
    
    Returns information about pipeline health, execution flow,
    and any bottlenecks or issues in the agent pipeline.
    
    Returns:
        Dictionary with pipeline monitoring data
    """
    try:
        stats = orchestrator.get_execution_statistics()
        orchestrator_stats = stats.get("orchestrator_stats", {})
        
        # Calculate pipeline metrics
        total_executions = orchestrator_stats.get("total_executions", 0)
        successful_executions = orchestrator_stats.get("successful_executions", 0)
        failed_executions = orchestrator_stats.get("failed_executions", 0)
        avg_execution_time = orchestrator_stats.get("average_execution_time", 0)
        
        # Get agent performance
        agent_performance = stats.get("agent_stats", {})
        
        # Identify bottlenecks (agents with highest execution times)
        bottlenecks = sorted(
            [
                {
                    "agent": name,
                    "avg_time_ms": data.get("average_execution_time", 0),
                    "total_executions": data.get("total_executions", 0)
                }
                for name, data in agent_performance.items()
            ],
            key=lambda x: x["avg_time_ms"],
            reverse=True
        )[:3]  # Top 3 slowest agents
        
        return {
            "timestamp": datetime.now().isoformat(),
            "pipeline_health": {
                "status": "healthy" if failed_executions < total_executions * 0.1 else "degraded",
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "success_rate": (successful_executions / total_executions * 100) if total_executions > 0 else 0,
                "average_execution_time_seconds": avg_execution_time
            },
            "agent_performance": {
                name: {
                    "avg_time_ms": data.get("average_execution_time", 0),
                    "success_rate": (
                        data.get("successful_executions", 0) / data.get("total_executions", 1) * 100
                        if data.get("total_executions", 0) > 0 else 0
                    )
                }
                for name, data in agent_performance.items()
            },
            "bottlenecks": bottlenecks,
            "recommendations": [
                f"Agent '{b['agent']}' is the slowest with {b['avg_time_ms']:.2f}ms average execution time"
                for b in bottlenecks
            ] if bottlenecks else ["No significant bottlenecks detected"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve pipeline monitoring data: {str(e)}"
        )


@router.get("/monitor/sessions")
async def monitor_sessions(
    limit: int = 10,
    orchestrator: MAFOrchestrator = Depends(get_orchestrator)
):
    """
    Monitor recent session activity.
    
    Returns information about recent sessions, including execution times,
    success rates, and common issues.
    
    Args:
        limit: Maximum number of recent sessions to return
        
    Returns:
        Dictionary with session monitoring data
    """
    try:
        stats = orchestrator.get_execution_statistics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "recent_activity": {
                "total_sessions": stats.get("orchestrator_stats", {}).get("total_executions", 0),
                "successful_sessions": stats.get("orchestrator_stats", {}).get("successful_executions", 0),
                "failed_sessions": stats.get("orchestrator_stats", {}).get("failed_executions", 0)
            },
            "note": "Detailed session traces are available in the agent_trace field of each /maf/ask response",
            "how_to_debug": {
                "step_1": "Make a request to /maf/ask",
                "step_2": "Check the 'agent_trace' field in the response",
                "step_3": "Use /maf/debug/trace/{session_id} for detailed trace (if implemented)",
                "step_4": "Enable debug mode with /maf/debug/enable for detailed logging"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve session monitoring data: {str(e)}"
        )