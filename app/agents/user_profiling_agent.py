"""
User Profiling Agent for access validation and state management.

This agent validates access levels, normalizes classification,
and persists user profile in memory/state.
"""

from typing import Dict, Any, Optional, Set
from datetime import datetime, timedelta
import hashlib
import json

from .base_agent import BaseAgent
from .schemas import (
    AgentInput, AgentOutput, AgentState, AgentStatus,
    UserProfile, EmployeeType, EngineerLevel, WageBandLevel,
    ClassificationResult
)


class UserProfilingAgent(BaseAgent):
    """
    Agent responsible for user profiling and access validation.
    
    Key responsibilities:
    1. Validate access level based on classification
    2. Normalize classification results
    3. Persist user profile in shared state
    4. Manage session-based user context
    5. Apply access control rules
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("UserProfilingAgent", config)
        
        # Access control rules
        self._init_access_rules()
        
        # In-memory user profile cache (in production, use Redis)
        self._profile_cache: Dict[str, UserProfile] = {}
        
    def _init_access_rules(self) -> None:
        """Initialize access control rules for different employee types."""
        
        # Define document access permissions by employee type and level
        self.access_rules = {
            EmployeeType.ENGINEER: {
                "allowed_document_types": [
                    "engineering_regulations",
                    "promotion_guidelines", 
                    "technical_standards",
                    "professional_development",
                    "general_hr_policies"
                ],
                "restricted_topics": [
                    "disciplinary_procedures",  # Requires manager level
                    "salary_structures"  # Requires HR access
                ],
                "levels": {
                    EngineerLevel.ASSOCIATE: {
                        "max_access_level": "basic",
                        "additional_restrictions": ["senior_policies"]
                    },
                    EngineerLevel.PROFESSIONAL: {
                        "max_access_level": "intermediate",
                        "additional_restrictions": []
                    },
                    EngineerLevel.CONSULTANT: {
                        "max_access_level": "advanced",
                        "additional_restrictions": []
                    }
                }
            },
            EmployeeType.WAGE_BAND: {
                "allowed_document_types": [
                    "wage_band_regulations",
                    "benefits_guide",
                    "leave_policies",
                    "general_hr_policies"
                ],
                "restricted_topics": [
                    "engineering_regulations",
                    "management_policies"
                ],
                "levels": {
                    WageBandLevel.ORDINARY: {
                        "max_access_level": "basic",
                        "additional_restrictions": ["technical_procedures"]
                    },
                    WageBandLevel.SKILLED: {
                        "max_access_level": "intermediate", 
                        "additional_restrictions": []
                    },
                    WageBandLevel.ASSISTANT_TECHNICIAN: {
                        "max_access_level": "intermediate",
                        "additional_restrictions": []
                    }
                }
            },
            EmployeeType.CIVIL_SERVANT: {
                "allowed_document_types": [
                    "executive_hr_regulations",
                    "administrative_procedures",
                    "general_policies",
                    "benefits_guide"
                ],
                "restricted_topics": [
                    "engineering_regulations",
                    "wage_band_specific"
                ],
                "levels": {}  # No sub-levels for civil servants
            },
            EmployeeType.EXTERNAL: {
                "allowed_document_types": [
                    "public_information",
                    "general_guidelines"
                ],
                "restricted_topics": [
                    "internal_policies",
                    "employee_specific",
                    "confidential"
                ],
                "levels": {}
            }
        }
        
        # High-risk topics that require human approval
        self.high_risk_topics = {
            "disciplinary_actions",
            "termination_procedures", 
            "legal_disputes",
            "salary_negotiations",
            "promotion_appeals",
            "grievance_procedures"
        }
    
    async def _execute_agent_logic(self, input_data: AgentInput) -> AgentOutput:
        """
        Execute user profiling logic.
        
        Args:
            input_data: Agent input with state and config
            
        Returns:
            AgentOutput with user profile
        """
        state = input_data.state
        
        if not self._validate_state(state):
            return AgentOutput(
                status=AgentStatus.ERROR,
                updated_state=state,
                execution_time_ms=0,
                error_message="Invalid state: missing session_id or user_question"
            )
        
        # Check if classification was successful
        if not state.classification_result or state.classification_result.status != AgentStatus.SUCCESS:
            return AgentOutput(
                status=AgentStatus.ERROR,
                updated_state=state,
                execution_time_ms=0,
                error_message="Classification required before user profiling"
            )
        
        # Step 1: Create or retrieve user profile
        user_profile = self._create_user_profile(state.classification_result, state.session_id)
        
        # Step 2: Validate access permissions
        access_validation = self._validate_access_permissions(user_profile, state.user_question)
        
        # Step 3: Check for high-risk topics
        risk_assessment = self._assess_question_risk(state.user_question, user_profile)
        
        # Step 4: Update state with profile and metadata
        state.user_profile = user_profile
        state.metadata.update({
            "access_validation": access_validation,
            "risk_assessment": risk_assessment,
            "profile_created_at": datetime.now().isoformat()
        })
        
        # Step 5: Cache profile for session
        self._cache_user_profile(state.session_id, user_profile)
        
        # Determine if we can proceed
        if not access_validation["has_access"]:
            return AgentOutput(
                status=AgentStatus.ERROR,
                updated_state=state,
                execution_time_ms=0,
                error_message=f"Access denied: {access_validation['reason']}"
            )
        
        if risk_assessment["requires_approval"]:
            state.metadata["requires_human_approval"] = True
        
        return AgentOutput(
            status=AgentStatus.SUCCESS,
            updated_state=state,
            execution_time_ms=0
        )
    
    def _create_user_profile(
        self, 
        classification: ClassificationResult, 
        session_id: str
    ) -> UserProfile:
        """
        Create user profile from classification result.
        
        Args:
            classification: Classification result
            session_id: Session identifier
            
        Returns:
            UserProfile object
        """
        # Check if we have a cached profile for this session
        cached_profile = self._profile_cache.get(session_id)
        if cached_profile and self._is_profile_valid(cached_profile):
            return cached_profile
        
        # Create new profile
        profile = UserProfile(
            employee_type=classification.employee_type,
            level=classification.level,
            confidence=classification.confidence,
            metadata={
                "session_id": session_id,
                "classification_method": classification.reasoning,
                "created_via": "classification_agent"
            }
        )
        
        return profile
    
    def _validate_access_permissions(
        self, 
        profile: UserProfile, 
        question: str
    ) -> Dict[str, Any]:
        """
        Validate user access permissions for the question.
        
        Args:
            profile: User profile
            question: User question
            
        Returns:
            Access validation result
        """
        employee_type = profile.employee_type
        level = profile.level
        
        # Get access rules for employee type
        if employee_type not in self.access_rules:
            return {
                "has_access": False,
                "reason": f"No access rules defined for {employee_type.value}",
                "allowed_topics": [],
                "restricted_topics": []
            }
        
        rules = self.access_rules[employee_type]
        
        # Check document type access
        allowed_docs = rules["allowed_document_types"]
        restricted_topics = rules["restricted_topics"]
        
        # Analyze question for restricted topics
        question_lower = question.lower()
        found_restrictions = []
        
        for restricted in restricted_topics:
            if restricted.lower().replace("_", " ") in question_lower:
                found_restrictions.append(restricted)
        
        # Check level-specific restrictions
        if level and "levels" in rules and level in rules["levels"]:
            level_rules = rules["levels"][level]
            additional_restrictions = level_rules.get("additional_restrictions", [])
            
            for restricted in additional_restrictions:
                if restricted.lower().replace("_", " ") in question_lower:
                    found_restrictions.append(restricted)
        
        # Determine access
        has_access = len(found_restrictions) == 0
        reason = ""
        
        if not has_access:
            reason = f"Question contains restricted topics: {', '.join(found_restrictions)}"
        
        return {
            "has_access": has_access,
            "reason": reason,
            "allowed_document_types": allowed_docs,
            "restricted_topics": restricted_topics,
            "found_restrictions": found_restrictions,
            "access_level": rules["levels"].get(level, {}).get("max_access_level", "basic") if level else "basic"
        }
    
    def _assess_question_risk(
        self, 
        question: str, 
        profile: UserProfile
    ) -> Dict[str, Any]:
        """
        Assess risk level of the question.
        
        Args:
            question: User question
            profile: User profile
            
        Returns:
            Risk assessment result
        """
        question_lower = question.lower()
        
        # Check for high-risk topics
        high_risk_found = []
        for risk_topic in self.high_risk_topics:
            if risk_topic.lower().replace("_", " ") in question_lower:
                high_risk_found.append(risk_topic)
        
        # Check for sensitive keywords
        sensitive_keywords = [
            "terminate", "fire", "dismiss", "disciplinary", "grievance",
            "appeal", "complaint", "legal action", "lawsuit", "violation",
            "إنهاء خدمة", "فصل", "تأديب", "شكوى", "تظلم", "مخالفة"
        ]
        
        sensitive_found = []
        for keyword in sensitive_keywords:
            if keyword.lower() in question_lower:
                sensitive_found.append(keyword)
        
        # Determine risk level
        risk_level = "low"
        requires_approval = False
        
        if high_risk_found or len(sensitive_found) >= 2:
            risk_level = "high"
            requires_approval = True
        elif sensitive_found:
            risk_level = "medium"
            # Medium risk may require approval for certain employee types
            if profile.employee_type in [EmployeeType.EXTERNAL, EmployeeType.WAGE_BAND]:
                requires_approval = True
        
        return {
            "risk_level": risk_level,
            "requires_approval": requires_approval,
            "high_risk_topics": high_risk_found,
            "sensitive_keywords": sensitive_found,
            "reasoning": f"Risk level: {risk_level} based on content analysis"
        }
    
    def _cache_user_profile(self, session_id: str, profile: UserProfile) -> None:
        """
        Cache user profile for session.
        
        Args:
            session_id: Session identifier
            profile: User profile to cache
        """
        # In production, this would use Redis with TTL
        self._profile_cache[session_id] = profile
        
        # Clean up old profiles (simple LRU-like cleanup)
        if len(self._profile_cache) > 1000:  # Max cache size
            # Remove oldest 20% of profiles
            sorted_profiles = sorted(
                self._profile_cache.items(),
                key=lambda x: x[1].created_at
            )
            
            to_remove = len(sorted_profiles) // 5  # Remove 20%
            for session_to_remove, _ in sorted_profiles[:to_remove]:
                del self._profile_cache[session_to_remove]
    
    def _is_profile_valid(self, profile: UserProfile) -> bool:
        """
        Check if cached profile is still valid.
        
        Args:
            profile: Cached user profile
            
        Returns:
            True if profile is valid, False otherwise
        """
        # Profile expires after 1 hour
        expiry_time = profile.created_at + timedelta(hours=1)
        return datetime.now() < expiry_time
    
    def get_cached_profile(self, session_id: str) -> Optional[UserProfile]:
        """
        Get cached user profile for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Cached UserProfile or None
        """
        profile = self._profile_cache.get(session_id)
        if profile and self._is_profile_valid(profile):
            return profile
        elif profile:
            # Remove expired profile
            del self._profile_cache[session_id]
        return None
    
    def clear_profile_cache(self, session_id: Optional[str] = None) -> None:
        """
        Clear profile cache.
        
        Args:
            session_id: Specific session to clear, or None to clear all
        """
        if session_id:
            self._profile_cache.pop(session_id, None)
        else:
            self._profile_cache.clear()