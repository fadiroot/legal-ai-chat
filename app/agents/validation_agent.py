"""
Validation/Compliance Agent for hallucination detection and compliance checking.

This agent detects hallucinations, verifies citations, ensures correct employee 
category usage, assigns confidence levels, and flags risky answers.
"""

from typing import Dict, Any, Optional, List, Set
import re
from difflib import SequenceMatcher
from openai import AsyncOpenAI
import os

from .base_agent import BaseAgent
from .schemas import (
    AgentInput, AgentOutput, AgentState, AgentStatus,
    ValidationResult, ConfidenceLevel, ReasoningResult,
    DocumentChunk, UserProfile, EmployeeType
)


class ValidationAgent(BaseAgent):
    """
    Agent responsible for validating answers and ensuring compliance.
    
    Key responsibilities:
    1. Detect hallucinations by comparing answer to source documents
    2. Verify citations and source accuracy
    3. Ensure correct employee category usage
    4. Assign confidence levels based on multiple factors
    5. Flag risky answers that need human review
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("ValidationAgent", config)
        from openai import AzureOpenAI
        # Use AzureOpenAI for Azure endpoints
        self.openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-15-preview"
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        
        # Initialize validation configuration
        self._init_validation_config()
    
    def _init_validation_config(self) -> None:
        """Initialize validation configuration and thresholds."""
        
        # Hallucination detection thresholds
        self.similarity_threshold = self._get_config_value("similarity_threshold", 0.3)
        self.max_unsupported_claims = self._get_config_value("max_unsupported_claims", 2)
        
        # Citation validation settings
        self.min_citation_similarity = self._get_config_value("min_citation_similarity", 0.5)
        self.require_article_references = self._get_config_value("require_article_references", True)
        
        # Risk assessment keywords
        self.high_risk_keywords = {
            "legal_advice": [
                "you should", "you must", "it is required", "mandatory",
                "يجب عليك", "مطلوب منك", "إلزامي", "ضروري"
            ],
            "definitive_statements": [
                "always", "never", "all", "none", "definitely", "certainly",
                "دائماً", "أبداً", "جميع", "لا شيء", "بالتأكيد", "حتماً"
            ],
            "external_references": [
                "according to law", "the regulation states", "legal requirement",
                "وفقاً للقانون", "ينص النظام", "متطلب قانوني"
            ],
            "sensitive_topics": [
                "termination", "dismissal", "disciplinary", "penalty", "fine",
                "إنهاء خدمة", "فصل", "تأديب", "عقوبة", "غرامة"
            ]
        }
        
        # Employee category validation rules
        self.category_specific_terms = {
            EmployeeType.ENGINEER: [
                "engineering", "technical", "professional", "consultant",
                "هندسة", "فني", "مهني", "استشاري"
            ],
            EmployeeType.WAGE_BAND: [
                "wage", "salary band", "skilled", "ordinary", "technician",
                "أجر", "بند", "ماهر", "عادي", "فني"
            ],
            EmployeeType.CIVIL_SERVANT: [
                "civil service", "administrative", "government", "executive",
                "خدمة مدنية", "إداري", "حكومي", "تنفيذي"
            ]
        }
        
        # Compliance flags
        self.compliance_patterns = {
            "unauthorized_advice": [
                r"I recommend", r"You should", r"My advice",
                r"أنصح", r"يجب أن", r"نصيحتي"
            ],
            "external_knowledge": [
                r"Generally", r"Usually", r"In most cases", r"Typically",
                r"عادة", r"في الغالب", r"في معظم الحالات"
            ],
            "speculation": [
                r"might be", r"could be", r"probably", r"likely",
                r"قد يكون", r"ربما", r"على الأرجح"
            ]
        }
    
    async def _execute_agent_logic(self, input_data: AgentInput) -> AgentOutput:
        """
        Execute validation logic.
        
        Args:
            input_data: Agent input with state and config
            
        Returns:
            AgentOutput with validation results
        """
        state = input_data.state
        
        if not self._validate_state(state):
            return AgentOutput(
                status=AgentStatus.ERROR,
                updated_state=state,
                execution_time_ms=0,
                error_message="Invalid state: missing session_id or user_question"
            )
        
        # Check if reasoning was successful
        if not state.reasoning_result or state.reasoning_result.status != AgentStatus.SUCCESS:
            return AgentOutput(
                status=AgentStatus.ERROR,
                updated_state=state,
                execution_time_ms=0,
                error_message="Successful reasoning required before validation"
            )
        
        try:
            reasoning_result = state.reasoning_result
            retrieval_result = state.retrieval_result
            user_profile = state.user_profile
            
            # Step 1: Detect hallucinations
            hallucination_check = self._detect_hallucinations(
                answer=reasoning_result.answer,
                source_chunks=retrieval_result.chunks if retrieval_result else [],
                sources_used=reasoning_result.sources_used
            )
            
            # Step 2: Verify citations
            citation_check = self._verify_citations(
                answer=reasoning_result.answer,
                sources_used=reasoning_result.sources_used,
                available_chunks=retrieval_result.chunks if retrieval_result else []
            )
            
            # Step 3: Check employee category compliance
            category_check = self._check_employee_category_compliance(
                answer=reasoning_result.answer,
                user_profile=user_profile
            )
            
            # Step 4: Assess answer risk
            risk_assessment = await self._assess_answer_risk(
                answer=reasoning_result.answer,
                question=state.user_question,
                user_profile=user_profile
            )
            
            # Step 5: Check compliance patterns
            compliance_check = self._check_compliance_patterns(reasoning_result.answer)
            
            # Step 6: Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                reasoning_confidence=reasoning_result.confidence,
                hallucination_score=hallucination_check["confidence"],
                citation_score=citation_check["accuracy"],
                category_compliance=category_check["compliant"],
                risk_level=risk_assessment["risk_level"]
            )
            
            # Step 7: Determine if human review is needed
            needs_human_review = self._determine_human_review_requirement(
                hallucination_check=hallucination_check,
                citation_check=citation_check,
                risk_assessment=risk_assessment,
                compliance_check=compliance_check
            )
            
            # Step 8: Compile all issues
            all_issues = []
            all_issues.extend(hallucination_check.get("issues", []))
            all_issues.extend(citation_check.get("issues", []))
            all_issues.extend(category_check.get("issues", []))
            all_issues.extend(compliance_check.get("issues", []))
            all_issues.extend(risk_assessment.get("issues", []))
            
            # Step 9: Create validation result
            validation_result = ValidationResult(
                validated=hallucination_check["valid"] and citation_check["valid"] and category_check["compliant"],
                confidence=overall_confidence,
                needs_human_review=needs_human_review,
                issues=all_issues,
                hallucination_detected=not hallucination_check["valid"],
                citation_errors=citation_check.get("errors", []),
                compliance_flags=compliance_check.get("flags", [])
            )
            
            # Update state
            state.validation_result = validation_result
            state.metadata.update({
                "validation_details": {
                    "hallucination_check": hallucination_check,
                    "citation_check": citation_check,
                    "category_check": category_check,
                    "risk_assessment": risk_assessment,
                    "compliance_check": compliance_check
                }
            })
            
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                updated_state=state,
                execution_time_ms=0
            )
            
        except Exception as e:
            error_message = f"Validation failed: {str(e)}"
            
            # Create minimal validation result for error case
            validation_result = ValidationResult(
                validated=False,
                confidence=ConfidenceLevel.LOW,
                needs_human_review=True,
                issues=[f"Validation error: {str(e)}"],
                hallucination_detected=True
            )
            
            state.validation_result = validation_result
            
            return AgentOutput(
                status=AgentStatus.ERROR,
                updated_state=state,
                execution_time_ms=0,
                error_message=error_message
            )
    
    def _detect_hallucinations(
        self,
        answer: str,
        source_chunks: List[DocumentChunk],
        sources_used: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect potential hallucinations in the answer.
        
        Args:
            answer: Generated answer
            source_chunks: Available source chunks
            sources_used: Sources claimed to be used
            
        Returns:
            Hallucination detection result
        """
        if not answer or "This information is not found" in answer:
            return {
                "valid": True,
                "confidence": 1.0,
                "issues": [],
                "method": "no_claims_to_verify"
            }
        
        # Extract factual claims from the answer
        claims = self._extract_factual_claims(answer)
        
        if not claims:
            return {
                "valid": True,
                "confidence": 0.8,
                "issues": [],
                "method": "no_factual_claims"
            }
        
        # Verify each claim against source documents
        unsupported_claims = []
        supported_claims = []
        
        for claim in claims:
            is_supported = self._verify_claim_against_sources(claim, source_chunks)
            
            if is_supported:
                supported_claims.append(claim)
            else:
                unsupported_claims.append(claim)
        
        # Calculate confidence based on support ratio
        total_claims = len(claims)
        supported_ratio = len(supported_claims) / total_claims if total_claims > 0 else 0
        
        # Determine validity
        is_valid = len(unsupported_claims) <= self.max_unsupported_claims
        
        issues = []
        if unsupported_claims:
            issues.append(f"Found {len(unsupported_claims)} unsupported claims")
            for claim in unsupported_claims[:3]:  # Show first 3 unsupported claims
                issues.append(f"Unsupported: {claim[:100]}...")
        
        return {
            "valid": is_valid,
            "confidence": supported_ratio,
            "issues": issues,
            "supported_claims": len(supported_claims),
            "unsupported_claims": len(unsupported_claims),
            "total_claims": total_claims,
            "method": "claim_verification"
        }
    
    def _extract_factual_claims(self, answer: str) -> List[str]:
        """
        Extract factual claims from the answer.
        
        Args:
            answer: Generated answer
            
        Returns:
            List of factual claims
        """
        # Split answer into sentences
        sentences = re.split(r'[.!?]+', answer)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            # Skip sentences that are clearly not factual claims
            skip_patterns = [
                r'^(This information|هذه المعلومة)',
                r'^(Based on|بناءً على)',
                r'^(Please|يرجى)',
                r'^(For more|لمزيد)',
                r'^(However|ومع ذلك)'
            ]
            
            should_skip = any(re.match(pattern, sentence, re.IGNORECASE) for pattern in skip_patterns)
            if should_skip:
                continue
            
            # Consider sentences with specific indicators as factual claims
            factual_indicators = [
                r'\d+',  # Contains numbers
                r'(Article|المادة)',  # References articles
                r'(must|يجب)',  # Contains obligations
                r'(required|مطلوب)',  # Contains requirements
                r'(according to|وفقاً لـ)'  # References sources
            ]
            
            has_factual_indicator = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in factual_indicators)
            if has_factual_indicator:
                claims.append(sentence)
        
        return claims
    
    def _verify_claim_against_sources(self, claim: str, source_chunks: List[DocumentChunk]) -> bool:
        """
        Verify if a claim is supported by source documents.
        
        Args:
            claim: Factual claim to verify
            source_chunks: Available source chunks
            
        Returns:
            True if claim is supported, False otherwise
        """
        if not source_chunks:
            return False
        
        # Calculate similarity with each source chunk
        max_similarity = 0.0
        
        for chunk in source_chunks:
            similarity = self._calculate_text_similarity(claim, chunk.content)
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity >= self.similarity_threshold
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize texts
        text1_norm = re.sub(r'\s+', ' ', text1.lower().strip())
        text2_norm = re.sub(r'\s+', ' ', text2.lower().strip())
        
        # Use sequence matcher for similarity
        similarity = SequenceMatcher(None, text1_norm, text2_norm).ratio()
        
        # Also check for keyword overlap
        words1 = set(text1_norm.split())
        words2 = set(text2_norm.split())
        
        if words1 and words2:
            keyword_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
            # Combine sequence similarity and keyword overlap
            similarity = (similarity + keyword_overlap) / 2
        
        return similarity
    
    def _verify_citations(
        self,
        answer: str,
        sources_used: List[Dict[str, Any]],
        available_chunks: List[DocumentChunk]
    ) -> Dict[str, Any]:
        """
        Verify citation accuracy.
        
        Args:
            answer: Generated answer
            sources_used: Sources claimed to be used
            available_chunks: Available source chunks
            
        Returns:
            Citation verification result
        """
        if not sources_used:
            return {
                "valid": len(available_chunks) == 0,  # Valid if no sources available
                "accuracy": 0.0 if available_chunks else 1.0,
                "issues": ["No sources cited"] if available_chunks else [],
                "errors": []
            }
        
        errors = []
        valid_citations = 0
        
        for source in sources_used:
            # Check if cited document exists in available chunks
            document_name = source.get("document", "")
            article = source.get("article", "")
            
            matching_chunks = [
                chunk for chunk in available_chunks
                if chunk.document_name == document_name
            ]
            
            if not matching_chunks:
                errors.append(f"Cited document '{document_name}' not found in sources")
                continue
            
            # If article is specified, verify it exists
            if article and article != "N/A":
                article_found = any(
                    chunk.article == article for chunk in matching_chunks
                )
                if not article_found:
                    errors.append(f"Article '{article}' not found in document '{document_name}'")
                    continue
            
            # Verify cited content matches source
            cited_content = source.get("content", "")
            if cited_content:
                content_verified = any(
                    self._calculate_text_similarity(cited_content, chunk.content) >= self.min_citation_similarity
                    for chunk in matching_chunks
                )
                if not content_verified:
                    errors.append(f"Cited content does not match source in '{document_name}'")
                    continue
            
            valid_citations += 1
        
        accuracy = valid_citations / len(sources_used) if sources_used else 0
        is_valid = len(errors) == 0
        
        issues = []
        if errors:
            issues.append(f"Found {len(errors)} citation errors")
        
        return {
            "valid": is_valid,
            "accuracy": accuracy,
            "issues": issues,
            "errors": errors,
            "valid_citations": valid_citations,
            "total_citations": len(sources_used)
        }
    
    def _check_employee_category_compliance(
        self,
        answer: str,
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """
        Check if answer is appropriate for employee category.
        
        Args:
            answer: Generated answer
            user_profile: User profile
            
        Returns:
            Category compliance result
        """
        if not user_profile or not user_profile.employee_type:
            return {
                "compliant": False,
                "issues": ["No user profile available for category check"]
            }
        
        employee_type = user_profile.employee_type
        answer_lower = answer.lower()
        
        issues = []
        
        # Check for inappropriate category-specific terms
        for other_type, terms in self.category_specific_terms.items():
            if other_type != employee_type:
                found_terms = [term for term in terms if term.lower() in answer_lower]
                if found_terms:
                    issues.append(
                        f"Answer contains terms specific to {other_type.value}: {', '.join(found_terms)}"
                    )
        
        # Check for generic advice that might not apply to this category
        generic_patterns = [
            r"all employees",
            r"every worker", 
            r"جميع الموظفين",
            r"كل العاملين"
        ]
        
        for pattern in generic_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                issues.append("Answer contains generic advice that may not apply to specific employee category")
                break
        
        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "employee_type": employee_type.value
        }
    
    async def _assess_answer_risk(
        self,
        answer: str,
        question: str,
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """
        Assess risk level of the answer.
        
        Args:
            answer: Generated answer
            question: Original question
            user_profile: User profile
            
        Returns:
            Risk assessment result
        """
        risk_factors = []
        risk_score = 0.0
        
        # Check for high-risk keywords
        for category, keywords in self.high_risk_keywords.items():
            found_keywords = [kw for kw in keywords if kw.lower() in answer.lower()]
            if found_keywords:
                risk_factors.append(f"{category}: {', '.join(found_keywords)}")
                risk_score += 0.2
        
        # Check for definitive legal statements
        definitive_patterns = [
            r"you must",
            r"it is required",
            r"the law states",
            r"يجب عليك",
            r"مطلوب",
            r"ينص القانون"
        ]
        
        for pattern in definitive_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                risk_factors.append(f"Definitive legal statement: {pattern}")
                risk_score += 0.3
                break
        
        # Check for sensitive topics based on user profile
        if user_profile.employee_type == EmployeeType.EXTERNAL:
            if any(term in answer.lower() for term in ["internal", "employee only", "confidential"]):
                risk_factors.append("Internal information provided to external user")
                risk_score += 0.4
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "issues": [f"Risk level: {risk_level}"] if risk_level != "low" else []
        }
    
    def _check_compliance_patterns(self, answer: str) -> Dict[str, Any]:
        """
        Check for compliance pattern violations.
        
        Args:
            answer: Generated answer
            
        Returns:
            Compliance check result
        """
        flags = []
        issues = []
        
        for violation_type, patterns in self.compliance_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, answer, re.IGNORECASE)
                if matches:
                    flags.append(f"{violation_type}: {', '.join(matches)}")
                    issues.append(f"Compliance violation: {violation_type}")
        
        return {
            "compliant": len(flags) == 0,
            "flags": flags,
            "issues": issues
        }
    
    def _calculate_overall_confidence(
        self,
        reasoning_confidence: ConfidenceLevel,
        hallucination_score: float,
        citation_score: float,
        category_compliance: bool,
        risk_level: str
    ) -> ConfidenceLevel:
        """
        Calculate overall confidence level.
        
        Args:
            reasoning_confidence: Confidence from reasoning agent
            hallucination_score: Hallucination detection score
            citation_score: Citation accuracy score
            category_compliance: Employee category compliance
            risk_level: Risk assessment level
            
        Returns:
            Overall confidence level
        """
        # Convert reasoning confidence to numeric
        reasoning_score = {
            ConfidenceLevel.HIGH: 1.0,
            ConfidenceLevel.MEDIUM: 0.7,
            ConfidenceLevel.LOW: 0.4
        }.get(reasoning_confidence, 0.4)
        
        # Calculate weighted score
        weights = {
            "reasoning": 0.3,
            "hallucination": 0.3,
            "citation": 0.2,
            "compliance": 0.1,
            "risk": 0.1
        }
        
        overall_score = (
            weights["reasoning"] * reasoning_score +
            weights["hallucination"] * hallucination_score +
            weights["citation"] * citation_score +
            weights["compliance"] * (1.0 if category_compliance else 0.0) +
            weights["risk"] * (1.0 if risk_level == "low" else 0.5 if risk_level == "medium" else 0.0)
        )
        
        # Convert back to confidence level
        if overall_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif overall_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _determine_human_review_requirement(
        self,
        hallucination_check: Dict[str, Any],
        citation_check: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        compliance_check: Dict[str, Any]
    ) -> bool:
        """
        Determine if human review is required.
        
        Args:
            hallucination_check: Hallucination detection result
            citation_check: Citation verification result
            risk_assessment: Risk assessment result
            compliance_check: Compliance check result
            
        Returns:
            True if human review is needed
        """
        # Require review if hallucinations detected
        if not hallucination_check.get("valid", True):
            return True
        
        # Require review if citation errors
        if not citation_check.get("valid", True):
            return True
        
        # Require review for high-risk answers
        if risk_assessment.get("risk_level") == "high":
            return True
        
        # Require review for compliance violations
        if not compliance_check.get("compliant", True):
            return True
        
        # Require review for medium risk with low confidence
        if (risk_assessment.get("risk_level") == "medium" and
            hallucination_check.get("confidence", 1.0) < 0.7):
            return True
        
        return False