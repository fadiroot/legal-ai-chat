"""
Shared schemas and data models for MAF agents.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


class EmployeeType(str, Enum):
    """Employee classification types."""
    ENGINEER = "Engineer"
    WAGE_BAND = "Wage Band Employee"
    CIVIL_SERVANT = "Civil Servant"
    EXTERNAL = "External"
    CITIZEN = "Citizen"


class EngineerLevel(str, Enum):
    """Engineer classification levels."""
    ASSOCIATE = "Associate"
    PROFESSIONAL = "Professional"
    CONSULTANT = "Consultant"


class WageBandLevel(str, Enum):
    """Wage Band Employee levels."""
    ORDINARY = "Ordinary"
    SKILLED = "Skilled"
    ASSISTANT_TECHNICIAN = "Assistant Technician"


class ConfidenceLevel(str, Enum):
    """Confidence levels for agent outputs."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AgentStatus(str, Enum):
    """Agent execution status."""
    SUCCESS = "success"
    CLARIFICATION_REQUIRED = "clarification_required"
    ERROR = "error"
    NEEDS_APPROVAL = "needs_approval"


class UserProfile(BaseModel):
    """User profile with employee classification."""
    employee_type: EmployeeType
    level: Optional[Union[EngineerLevel, WageBandLevel]] = None
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class DocumentChunk(BaseModel):
    """Document chunk with metadata."""
    content: str
    document_name: str
    page_number: int
    chunk_index: int
    score: float
    article: Optional[str] = None
    employee_type: Optional[List[EmployeeType]] = None
    level: Optional[List[Union[EngineerLevel, WageBandLevel]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClassificationResult(BaseModel):
    """Result from Classification Agent."""
    status: AgentStatus
    employee_type: Optional[EmployeeType] = None
    level: Optional[Union[EngineerLevel, WageBandLevel]] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    question_to_user: Optional[str] = None
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Result from Legal Retrieval Agent."""
    status: AgentStatus
    chunks: List[DocumentChunk] = Field(default_factory=list)
    total_results: int = 0
    search_metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


class ReasoningResult(BaseModel):
    """Result from Legal Reasoning Agent."""
    status: AgentStatus
    answer: Optional[str] = None
    sources_used: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    reasoning_steps: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None


class ValidationResult(BaseModel):
    """Result from Validation Agent."""
    validated: bool
    confidence: ConfidenceLevel
    needs_human_review: bool
    issues: List[str] = Field(default_factory=list)
    hallucination_detected: bool = False
    citation_errors: List[str] = Field(default_factory=list)
    compliance_flags: List[str] = Field(default_factory=list)


class ApprovalResult(BaseModel):
    """Result from Approval Agent."""
    approved: bool
    requires_human_approval: bool
    approval_reasons: List[str] = Field(default_factory=list)
    reviewer_notes: Optional[str] = None
    approval_timestamp: Optional[datetime] = None


class AgentState(BaseModel):
    """Shared state between agents."""
    session_id: str
    user_question: str
    user_profile: Optional[UserProfile] = None
    classification_result: Optional[ClassificationResult] = None
    retrieval_result: Optional[RetrievalResult] = None
    reasoning_result: Optional[ReasoningResult] = None
    validation_result: Optional[ValidationResult] = None
    approval_result: Optional[ApprovalResult] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FinalResponse(BaseModel):
    """Final response format for the legal AI assistant."""
    answer: str
    sources: List[Dict[str, Any]]
    employee_type: str
    level: Optional[str] = None
    confidence: ConfidenceLevel
    needs_human_review: bool
    session_id: str
    processing_time_ms: Optional[int] = None
    agent_trace: List[Dict[str, Any]] = Field(default_factory=list)


class AgentInput(BaseModel):
    """Base input for all agents."""
    state: AgentState
    agent_config: Dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    """Base output for all agents."""
    status: AgentStatus
    updated_state: AgentState
    execution_time_ms: int
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)