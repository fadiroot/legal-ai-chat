"""
Classification Agent for employee type and level extraction.

This agent is the MANDATORY FIRST STEP in the MAF workflow.
It extracts employee type and job category from user questions using
entity extraction and semantic understanding.
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from openai import AsyncOpenAI
import os

from .base_agent import BaseAgent
from .schemas import (
    AgentInput, AgentOutput, AgentState, AgentStatus,
    ClassificationResult, EmployeeType, EngineerLevel, WageBandLevel
)


class ClassificationAgent(BaseAgent):
    """
    Agent responsible for classifying users based on their questions.
    
    Key responsibilities:
    1. Extract employee type and job category from user questions
    2. Use entity extraction + semantic understanding
    3. Decide if clarification is required
    4. NEVER proceed with ambiguous classifications
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("ClassificationAgent", config)
        from openai import AzureOpenAI
        # Use AzureOpenAI for Azure endpoints
        self.openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-15-preview"
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        
        # Classification patterns and keywords
        self._init_classification_patterns()
    
    def _init_classification_patterns(self) -> None:
        """Initialize classification patterns and keywords."""
        
        # Employee type keywords (Arabic and English)
        self.employee_type_keywords = {
            EmployeeType.ENGINEER: [
                # English
                "engineer", "engineering", "technical", "consultant", "professional",
                "associate engineer", "professional engineer", "consultant engineer",
                # Arabic - include both individual words and phrases
                "مهندس", "هندسة", "فني", "استشاري", "مهني",
                "مهندس مساعد", "مهندس محترف", "مهندس استشاري",
                "انا مهندس", "أنا مهندس", "انا مهندس مساعد", "أنا مهندس مساعد",
                "انا مهندس محترف", "أنا مهندس محترف", "انا مهندس استشاري", "أنا مهندس استشاري"
            ],
            EmployeeType.WAGE_BAND: [
                # English
                "wage band", "salary band", "ordinary", "skilled", "technician",
                "assistant technician", "wage employee",
                # Arabic
                "بند الأجور", "بند أجور", "عادي", "ماهر", "فني مساعد",
                "موظف أجور", "عامل"
            ],
            EmployeeType.CIVIL_SERVANT: [
                # English
                "civil servant", "government employee", "public servant",
                "administrative", "executive", "hr regulations",
                # Arabic
                "موظف عام", "موظف حكومي", "خدمة مدنية", "إداري",
                "لائحة الموارد البشرية", "اللائحة التنفيذية"
            ]
        }
        
        # Level keywords
        self.level_keywords = {
            # Engineer levels
            EngineerLevel.ASSOCIATE: [
                "associate", "مساعد", "junior", "entry level"
            ],
            EngineerLevel.PROFESSIONAL: [
                "professional", "محترف", "senior", "experienced"
            ],
            EngineerLevel.CONSULTANT: [
                "consultant", "استشاري", "expert", "lead"
            ],
            # Wage Band levels
            WageBandLevel.ORDINARY: [
                "ordinary", "عادي", "basic", "general"
            ],
            WageBandLevel.SKILLED: [
                "skilled", "ماهر", "experienced", "qualified"
            ],
            WageBandLevel.ASSISTANT_TECHNICIAN: [
                "assistant technician", "فني مساعد", "technical assistant"
            ]
        }
    
    async def _execute_agent_logic(self, input_data: AgentInput) -> AgentOutput:
        """
        Execute classification logic.
        
        Args:
            input_data: Agent input with state and config
            
        Returns:
            AgentOutput with classification result
        """
        state = input_data.state
        
        if not self._validate_state(state):
            return AgentOutput(
                status=AgentStatus.ERROR,
                updated_state=state,
                execution_time_ms=0,
                error_message="Invalid state: missing session_id or user_question"
            )
        
        # Step 1: Keyword-based classification
        keyword_result = self._classify_by_keywords(state.user_question)
        
        # Step 2: LLM-based semantic classification
        llm_result = await self._classify_by_llm(state.user_question)
        
        # Step 3: Combine and validate results
        final_result = self._combine_classification_results(
            keyword_result, llm_result, state.user_question
        )
        
        # Update state with classification result
        state.classification_result = final_result
        
        return AgentOutput(
            status=final_result.status,
            updated_state=state,
            execution_time_ms=0  # Will be set by base class
        )
    
    def _classify_by_keywords(self, question: str) -> ClassificationResult:
        """
        Classify using keyword matching.
        
        Args:
            question: User question
            
        Returns:
            ClassificationResult from keyword analysis
        """
        question_lower = question.lower().strip()
        
        # Find employee type matches with weighted scoring
        type_scores = {}
        for emp_type, keywords in self.employee_type_keywords.items():
            score = 0
            for keyword in keywords:
                keyword_lower = keyword.lower().strip()
                # Exact phrase match gets higher weight (especially for Arabic phrases)
                if keyword_lower in question_lower:
                    # Longer phrases get more weight
                    weight = len(keyword_lower.split()) * 2
                    score += weight
                # Also check if key words are present (for compound phrases)
                elif len(keyword_lower.split()) > 1:
                    # Check if all words in phrase are present
                    words = keyword_lower.split()
                    if all(word in question_lower for word in words):
                        score += len(words)  # Partial credit for word matches
            
            if score > 0:
                type_scores[emp_type] = score
        
        if not type_scores:
            return ClassificationResult(
                status=AgentStatus.CLARIFICATION_REQUIRED,
                confidence=0.0,
                reasoning="No employee type keywords found"
            )
        
        # Get best employee type match
        best_type = max(type_scores, key=type_scores.get)
        # More lenient confidence calculation
        type_confidence = min(type_scores[best_type] / 2.0, 1.0)  # Lower threshold
        
        # Find level matches for the identified type
        level = None
        level_confidence = 0.0
        
        if best_type == EmployeeType.ENGINEER:
            level_scores = {}
            for eng_level, keywords in self.level_keywords.items():
                if isinstance(eng_level, EngineerLevel):
                    score = 0
                    for keyword in keywords:
                        keyword_lower = keyword.lower().strip()
                        # Phrase match gets more weight
                        if keyword_lower in question_lower:
                            score += 2 if len(keyword_lower.split()) > 1 else 1
                        # Also check individual word match
                        elif keyword_lower in question_lower:
                            score += 1
                    
                    if score > 0:
                        level_scores[eng_level] = score
            
            if level_scores:
                level = max(level_scores, key=level_scores.get)
                level_confidence = min(level_scores[level] / 1.5, 1.0)  # More lenient
        
        elif best_type == EmployeeType.WAGE_BAND:
            level_scores = {}
            for wb_level, keywords in self.level_keywords.items():
                if isinstance(wb_level, WageBandLevel):
                    score = 0
                    for keyword in keywords:
                        keyword_lower = keyword.lower().strip()
                        if keyword_lower in question_lower:
                            score += 2 if len(keyword_lower.split()) > 1 else 1
                    
                    if score > 0:
                        level_scores[wb_level] = score
            
            if level_scores:
                level = max(level_scores, key=level_scores.get)
                level_confidence = min(level_scores[level] / 1.5, 1.0)  # More lenient
        
        # Calculate overall confidence - more lenient
        overall_confidence = (type_confidence + level_confidence) / 2.0 if level else type_confidence
        
        # Lower threshold from 0.7 to 0.5 for more lenient matching
        # If we found both type and level, be even more lenient
        threshold = 0.4 if level else 0.5
        
        return ClassificationResult(
            status=AgentStatus.SUCCESS if overall_confidence >= threshold else AgentStatus.CLARIFICATION_REQUIRED,
            employee_type=best_type,
            level=level,
            confidence=overall_confidence,
            reasoning=f"Keyword-based classification: {best_type.value}" + 
                     (f" - {level.value}" if level else "")
        )
    
    async def _classify_by_llm(self, question: str) -> ClassificationResult:
        """
        Classify using LLM semantic understanding.
        
        Args:
            question: User question
            
        Returns:
            ClassificationResult from LLM analysis
        """
        try:
            # Create classification prompt
            prompt = self._create_classification_prompt(question)
            
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse LLM response
            return self._parse_llm_response(response.choices[0].message.content)
            
        except Exception as e:
            return ClassificationResult(
                status=AgentStatus.ERROR,
                confidence=0.0,
                reasoning=f"LLM classification failed: {str(e)}"
            )
    
    def _create_classification_prompt(self, question: str) -> Dict[str, str]:
        """
        Create classification prompt for LLM.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with system and user prompts
        """
        system_prompt = """You are a Saudi legal AI assistant specializing in employee classification.

Your task is to classify the user based on their question into one of these categories:

EMPLOYEE TYPES:
1. Engineer (مهندس)
   - Associate (مساعد)
   - Professional (محترف) 
   - Consultant (استشاري)

2. Wage Band Employee (موظف بند الأجور)
   - Ordinary (عادي)
   - Skilled (ماهر)
   - Assistant Technician (فني مساعد)

3. Civil Servant (موظف عام)
   - General government employee under Executive HR Regulations

4. External/Citizen (خارجي/مواطن)
   - Not a government employee

RULES:
- If the classification is ambiguous, return "CLARIFICATION_REQUIRED"
- Be conservative - only classify if you're confident (>70%)
- Consider both explicit mentions and implicit context
- Saudi regulations differ by employee type, so accuracy is critical

Response format:
{
  "employee_type": "Engineer|Wage Band Employee|Civil Servant|External",
  "level": "Associate|Professional|Consultant|Ordinary|Skilled|Assistant Technician|null",
  "confidence": 0.0-1.0,
  "reasoning": "explanation",
  "needs_clarification": true/false
}"""

        user_prompt = f"""Classify this user question:

Question: {question}

Analyze the question and determine the employee type and level. Consider:
1. Explicit job titles or classifications mentioned
2. Context clues about their role or responsibilities  
3. Type of legal/regulatory question being asked
4. Language used (formal vs informal, technical terms)

Provide your classification in the specified JSON format."""

        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _parse_llm_response(self, response: str) -> ClassificationResult:
        """
        Parse LLM response into ClassificationResult.
        
        Args:
            response: LLM response text
            
        Returns:
            ClassificationResult from parsed response
        """
        try:
            # Extract JSON from response
            import json
            
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)
            
            # Map to our enums
            employee_type = None
            if parsed.get("employee_type"):
                type_mapping = {
                    "Engineer": EmployeeType.ENGINEER,
                    "Wage Band Employee": EmployeeType.WAGE_BAND,
                    "Civil Servant": EmployeeType.CIVIL_SERVANT,
                    "External": EmployeeType.EXTERNAL
                }
                employee_type = type_mapping.get(parsed["employee_type"])
            
            level = None
            if parsed.get("level") and parsed["level"] != "null":
                level_mapping = {
                    "Associate": EngineerLevel.ASSOCIATE,
                    "Professional": EngineerLevel.PROFESSIONAL,
                    "Consultant": EngineerLevel.CONSULTANT,
                    "Ordinary": WageBandLevel.ORDINARY,
                    "Skilled": WageBandLevel.SKILLED,
                    "Assistant Technician": WageBandLevel.ASSISTANT_TECHNICIAN
                }
                level = level_mapping.get(parsed["level"])
            
            confidence = float(parsed.get("confidence", 0.0))
            needs_clarification = parsed.get("needs_clarification", False)
            
            # Lower threshold to 0.5 for more lenient matching
            status = AgentStatus.CLARIFICATION_REQUIRED if needs_clarification or confidence < 0.5 else AgentStatus.SUCCESS
            
            return ClassificationResult(
                status=status,
                employee_type=employee_type,
                level=level,
                confidence=confidence,
                reasoning=parsed.get("reasoning", "LLM-based classification")
            )
            
        except Exception as e:
            return ClassificationResult(
                status=AgentStatus.ERROR,
                confidence=0.0,
                reasoning=f"Failed to parse LLM response: {str(e)}"
            )
    
    def _combine_classification_results(
        self, 
        keyword_result: ClassificationResult,
        llm_result: ClassificationResult,
        question: str
    ) -> ClassificationResult:
        """
        Combine keyword and LLM classification results.
        
        Args:
            keyword_result: Result from keyword classification
            llm_result: Result from LLM classification
            question: Original user question
            
        Returns:
            Combined ClassificationResult
        """
        # If either failed, use the other
        if keyword_result.status == AgentStatus.ERROR:
            return llm_result
        if llm_result.status == AgentStatus.ERROR:
            return keyword_result
        
        # Trust keyword matching if it found a clear match (even if LLM disagrees)
        # This handles cases like "أنا مهندس مساعد" where keyword matching is reliable
        if (keyword_result.status == AgentStatus.SUCCESS and 
            keyword_result.employee_type and 
            keyword_result.confidence >= 0.5):
            
            # If keyword found both type and level, trust it
            if keyword_result.level:
                return ClassificationResult(
                    status=AgentStatus.SUCCESS,
                    employee_type=keyword_result.employee_type,
                    level=keyword_result.level,
                    confidence=keyword_result.confidence,
                    reasoning=f"Keyword-based classification (clear match): {keyword_result.employee_type.value} - {keyword_result.level.value}"
                )
            # If keyword found type but not level, still trust it
            elif keyword_result.employee_type:
                return ClassificationResult(
                    status=AgentStatus.SUCCESS,
                    employee_type=keyword_result.employee_type,
                    level=None,
                    confidence=keyword_result.confidence,
                    reasoning=f"Keyword-based classification: {keyword_result.employee_type.value}"
                )
        
        # If both agree and have reasonable confidence, use the result
        if (keyword_result.employee_type == llm_result.employee_type and
            keyword_result.level == llm_result.level and
            keyword_result.confidence >= 0.5 and llm_result.confidence >= 0.5):
            
            combined_confidence = (keyword_result.confidence + llm_result.confidence) / 2.0
            
            return ClassificationResult(
                status=AgentStatus.SUCCESS,
                employee_type=keyword_result.employee_type,
                level=keyword_result.level,
                confidence=combined_confidence,
                reasoning=f"Keyword and LLM agreement: {keyword_result.employee_type.value}" +
                         (f" - {keyword_result.level.value}" if keyword_result.level else "")
            )
        
        # If LLM found something with good confidence, trust it
        if (llm_result.status == AgentStatus.SUCCESS and 
            llm_result.employee_type and 
            llm_result.confidence >= 0.6):
            return llm_result
        
        # If results disagree or confidence is low, require clarification
        return ClassificationResult(
            status=AgentStatus.CLARIFICATION_REQUIRED,
            confidence=0.0,
            question_to_user=self._generate_clarification_question(question),
            reasoning="Classification results disagree or have low confidence"
        )
    
    def _generate_clarification_question(self, original_question: str) -> str:
        """
        Generate clarification question for the user.
        
        Args:
            original_question: Original user question
            
        Returns:
            Clarification question string
        """
        # Detect language
        has_arabic = any('\u0600' <= char <= '\u06FF' for char in original_question)
        
        if has_arabic:
            return """لتقديم إجابة دقيقة، يرجى تحديد فئة وظيفتك:

1. مهندس (Associate/Professional/Consultant)
2. موظف بند الأجور (عادي/ماهر/فني مساعد)  
3. موظف عام (خدمة مدنية)
4. خارجي/مواطن

يرجى تحديد الفئة والمستوى إن أمكن."""
        else:
            return """To provide an accurate answer, please specify your employee category:

1. Engineer (Associate/Professional/Consultant)
2. Wage Band Employee (Ordinary/Skilled/Assistant Technician)
3. Civil Servant (General government employee)
4. External/Citizen

Please specify both the category and level if applicable."""