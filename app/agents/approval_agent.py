"""Approval Agent - Auto-approves all requests for direct results."""

from typing import Dict, Any, Optional, List
from enum import Enum
import uuid
from datetime import datetime

from .base_agent import BaseAgent
from .schemas import (
    AgentInput, AgentOutput, AgentState, AgentStatus,
    ApprovalResult
)


class ApprovalStatus(str, Enum):
    """Approval status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalPriority(str, Enum):
    """Approval priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ApprovalRequest(dict):
    """Approval request structure."""
    def __init__(self, **kwargs):
        super().__init__()
        self.update({
            "request_id": str(uuid.uuid4()),
            "session_id": kwargs.get("session_id"),
            "priority": kwargs.get("priority", ApprovalPriority.MEDIUM),
            "status": ApprovalStatus.PENDING,
            "question": kwargs.get("question"),
            "answer": kwargs.get("answer"),
            "user_profile": kwargs.get("user_profile"),
            "created_at": datetime.now(),
            "expires_at": kwargs.get("expires_at"),
            "reviewer_notes": None,
            "approved_by": None,
            "approved_at": None,
            "metadata": kwargs.get("metadata", {})
        })


class ApprovalAgent(BaseAgent):
    """Agent that auto-approves all requests for direct results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("ApprovalAgent", config)
        self._approval_queue: Dict[str, ApprovalRequest] = {}
        self._approval_history: List[ApprovalRequest] = []
    
    async def _execute_agent_logic(self, input_data: AgentInput) -> AgentOutput:
        """Execute approval logic - auto-approves all requests."""
        state = input_data.state
        if not self._validate_state(state):
            return AgentOutput(
                status=AgentStatus.ERROR,
                updated_state=state,
                execution_time_ms=0,
                error_message="Invalid state: missing session_id or user_question"
            )
        if not state.validation_result:
            return AgentOutput(
                status=AgentStatus.ERROR,
                updated_state=state,
                execution_time_ms=0,
                error_message="Validation required before approval"
            )
        try:
            approval_result = ApprovalResult(
                approved=True,
                requires_human_approval=False,
                approval_reasons=["Auto-approved: Direct result mode"]
            )
            state.approval_result = approval_result
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                updated_state=state,
                execution_time_ms=0
            )
        except Exception as e:
            approval_result = ApprovalResult(
                approved=False,
                requires_human_approval=True,
                approval_reasons=[f"Error: {str(e)}"]
            )
            state.approval_result = approval_result
            return AgentOutput(
                status=AgentStatus.ERROR,
                updated_state=state,
                execution_time_ms=0,
                error_message=f"Approval processing failed: {str(e)}"
            )
    
    def _cleanup_expired_requests(self) -> None:
        """Clean up expired approval requests."""
        current_time = datetime.now()
        expired_requests = [
            request_id for request_id, request in self._approval_queue.items()
            if request.get("expires_at") and current_time > request["expires_at"]
        ]
        for request_id in expired_requests:
            expired_request = self._approval_queue.pop(request_id)
            self._approval_history.append(expired_request)
    
    def approve_request(self, request_id: str, reviewer_id: str, reviewer_notes: Optional[str] = None) -> bool:
        """Approve a pending request."""
        if request_id not in self._approval_queue:
            return False
        request = self._approval_queue[request_id]
        if request["status"] != ApprovalStatus.PENDING:
            return False
        request["status"] = ApprovalStatus.APPROVED
        request["approved_by"] = reviewer_id
        request["approved_at"] = datetime.now()
        request["reviewer_notes"] = reviewer_notes
        return True
    
    def reject_request(self, request_id: str, reviewer_id: str, rejection_reason: str) -> bool:
        """Reject a pending request."""
        if request_id not in self._approval_queue:
            return False
        request = self._approval_queue[request_id]
        if request["status"] != ApprovalStatus.PENDING:
            return False
        request["status"] = ApprovalStatus.REJECTED
        request["approved_by"] = reviewer_id
        request["approved_at"] = datetime.now()
        request["reviewer_notes"] = rejection_reason
        return True
    
    def get_pending_requests(self, priority: Optional[ApprovalPriority] = None) -> List[ApprovalRequest]:
        """Get pending approval requests."""
        self._cleanup_expired_requests()
        pending_requests = [
            request for request in self._approval_queue.values()
            if request["status"] == ApprovalStatus.PENDING
        ]
        if priority:
            pending_requests = [
                request for request in pending_requests
                if request["priority"] == priority
            ]
        priority_order = {
            ApprovalPriority.URGENT: 0,
            ApprovalPriority.HIGH: 1,
            ApprovalPriority.MEDIUM: 2,
            ApprovalPriority.LOW: 3
        }
        pending_requests.sort(
            key=lambda x: (priority_order[x["priority"]], x["created_at"])
        )
        return pending_requests
    
    def get_approval_statistics(self) -> Dict[str, Any]:
        """Get approval statistics."""
        total_requests = len(self._approval_queue) + len(self._approval_history)
        pending_requests = len([r for r in self._approval_queue.values() if r["status"] == ApprovalStatus.PENDING])
        approved_requests = len([r for r in self._approval_queue.values() if r["status"] == ApprovalStatus.APPROVED])
        rejected_requests = len([r for r in self._approval_queue.values() if r["status"] == ApprovalStatus.REJECTED])
        expired_requests = len(self._approval_history)
        return {
            "total_requests": total_requests,
            "pending_requests": pending_requests,
            "approved_requests": approved_requests,
            "rejected_requests": rejected_requests,
            "expired_requests": expired_requests,
            "approval_rate": approved_requests / max(total_requests - pending_requests, 1),
            "queue_size": len(self._approval_queue)
        }