"""
Legal Reasoning Agent for document-based answer synthesis.

This agent reads retrieved chunks ONLY and synthesizes answers strictly 
from documents. NO searching, NO guessing, NO external legal knowledge.
"""

from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
import os
import json

from .base_agent import BaseAgent
from .schemas import (
    AgentInput, AgentOutput, AgentState, AgentStatus,
    ReasoningResult, DocumentChunk, UserProfile, ConfidenceLevel,
    EmployeeType
)


class LegalReasoningAgent(BaseAgent):
    """
    Agent responsible for synthesizing legal answers from retrieved documents.
    
    Key responsibilities:
    1. Read retrieved chunks ONLY
    2. Synthesize answers strictly from documents
    3. NO searching, NO guessing, NO external legal knowledge
    4. Provide clear citations and reasoning steps
    5. Handle missing information appropriately
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("LegalReasoningAgent", config)
        from openai import AzureOpenAI
        # Use AzureOpenAI for Azure endpoints
        self.openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-15-preview"
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        
        # Initialize reasoning configuration
        self._init_reasoning_config()
    
    def _init_reasoning_config(self) -> None:
        """Initialize reasoning configuration and templates."""
        
        # Answer quality thresholds
        self.min_chunks_for_answer = self._get_config_value("min_chunks_for_answer", 1)
        self.max_chunks_to_process = self._get_config_value("max_chunks_to_process", 10)
        self.min_content_length = self._get_config_value("min_content_length", 50)
        
        # Confidence calculation weights
        self.confidence_weights = {
            "chunk_relevance": 0.3,
            "content_completeness": 0.25,
            "citation_quality": 0.25,
            "answer_coherence": 0.2
        }
        
        # Response templates for different scenarios
        self.response_templates = {
            "no_information": {
                "en": "This information is not found in the provided documents. I can only provide answers based on the official documents available in the system.",
                "ar": "لا توجد هذه المعلومات في الوثائق المتاحة. يمكنني فقط تقديم إجابات بناءً على الوثائق الرسمية الموجودة في النظام."
            },
            "partial_information": {
                "en": "Based on the available documents, I can provide the following information. However, some details may not be complete:",
                "ar": "بناءً على الوثائق المتاحة، يمكنني تقديم المعلومات التالية. ومع ذلك، قد تكون بعض التفاصيل غير مكتملة:"
            },
            "insufficient_access": {
                "en": "The available documents do not contain sufficient information for your employee category. You may need to consult with HR or your supervisor for specific guidance.",
                "ar": "الوثائق المتاحة لا تحتوي على معلومات كافية لفئة موظفيك. قد تحتاج إلى استشارة الموارد البشرية أو مشرفك للحصول على إرشادات محددة."
            }
        }
    
    async def _execute_agent_logic(self, input_data: AgentInput) -> AgentOutput:
        """
        Execute legal reasoning logic.
        
        Args:
            input_data: Agent input with state and config
            
        Returns:
            AgentOutput with reasoning results
        """
        state = input_data.state
        
        if not self._validate_state(state):
            return AgentOutput(
                status=AgentStatus.ERROR,
                updated_state=state,
                execution_time_ms=0,
                error_message="Invalid state: missing session_id or user_question"
            )
        
        # Check if retrieval was successful
        if not state.retrieval_result or state.retrieval_result.status != AgentStatus.SUCCESS:
            return AgentOutput(
                status=AgentStatus.ERROR,
                updated_state=state,
                execution_time_ms=0,
                error_message="Successful retrieval required before reasoning"
            )
        
        try:
            # Step 1: Validate retrieved chunks
            chunks = state.retrieval_result.chunks
            validation_result = self._validate_chunks(chunks, state.user_profile)
            
            if not validation_result["is_valid"]:
                reasoning_result = ReasoningResult(
                    status=AgentStatus.SUCCESS,  # Not an error, just no information
                    answer=validation_result["message"],
                    confidence=ConfidenceLevel.LOW,
                    reasoning_steps=["No valid chunks found for reasoning"]
                )
                state.reasoning_result = reasoning_result
                
                return AgentOutput(
                    status=AgentStatus.SUCCESS,
                    updated_state=state,
                    execution_time_ms=0
                )
            
            # Step 2: Prepare chunks for reasoning
            processed_chunks = self._prepare_chunks_for_reasoning(chunks)
            
            # Step 3: Generate answer using LLM
            answer_result = await self._generate_answer_from_chunks(
                question=state.user_question,
                chunks=processed_chunks,
                user_profile=state.user_profile
            )
            
            # Step 4: Calculate confidence level
            confidence = self._calculate_confidence_level(
                chunks=processed_chunks,
                answer=answer_result["answer"],
                reasoning_steps=answer_result["reasoning_steps"]
            )
            
            # Step 5: Create reasoning result
            reasoning_result = ReasoningResult(
                status=AgentStatus.SUCCESS,
                answer=answer_result["answer"],
                sources_used=answer_result["sources_used"],
                confidence=confidence,
                reasoning_steps=answer_result["reasoning_steps"]
            )
            
            # Update state
            state.reasoning_result = reasoning_result
            
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                updated_state=state,
                execution_time_ms=0
            )
            
        except Exception as e:
            error_message = f"Reasoning failed: {str(e)}"
            
            reasoning_result = ReasoningResult(
                status=AgentStatus.ERROR,
                error_message=error_message
            )
            
            state.reasoning_result = reasoning_result
            
            return AgentOutput(
                status=AgentStatus.ERROR,
                updated_state=state,
                execution_time_ms=0,
                error_message=error_message
            )
    
    def _validate_chunks(
        self, 
        chunks: List[DocumentChunk], 
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """
        Validate retrieved chunks for reasoning.
        
        Args:
            chunks: Retrieved document chunks
            user_profile: User profile
            
        Returns:
            Validation result dictionary
        """
        if not chunks:
            language = "ar" if any('\u0600' <= char <= '\u06FF' for char in user_profile.metadata.get("original_question", "")) else "en"
            return {
                "is_valid": False,
                "message": self.response_templates["no_information"][language],
                "reason": "no_chunks"
            }
        
        # Check if chunks meet minimum requirements
        valid_chunks = [
            chunk for chunk in chunks 
            if len(chunk.content.strip()) >= self.min_content_length
        ]
        
        if len(valid_chunks) < self.min_chunks_for_answer:
            language = "ar" if any('\u0600' <= char <= '\u06FF' for char in user_profile.metadata.get("original_question", "")) else "en"
            return {
                "is_valid": False,
                "message": self.response_templates["insufficient_access"][language],
                "reason": "insufficient_chunks"
            }
        
        return {
            "is_valid": True,
            "valid_chunks": valid_chunks,
            "total_chunks": len(chunks)
        }
    
    def _prepare_chunks_for_reasoning(self, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """
        Prepare chunks for reasoning by cleaning and structuring content.
        
        Args:
            chunks: Raw document chunks
            
        Returns:
            List of processed chunks
        """
        processed_chunks = []
        
        # Limit number of chunks to process
        chunks_to_process = chunks[:self.max_chunks_to_process]
        
        for i, chunk in enumerate(chunks_to_process):
            processed_chunk = {
                "index": i,
                "content": chunk.content.strip(),
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "article": chunk.article,
                "score": chunk.score,
                "metadata": chunk.metadata
            }
            
            # Clean content
            processed_chunk["cleaned_content"] = self._clean_chunk_content(chunk.content)
            
            # Extract key information
            processed_chunk["key_info"] = self._extract_key_information(chunk.content)
            
            processed_chunks.append(processed_chunk)
        
        return processed_chunks
    
    def _clean_chunk_content(self, content: str) -> str:
        """
        Clean chunk content for better processing.
        
        Args:
            content: Raw chunk content
            
        Returns:
            Cleaned content
        """
        # Remove excessive whitespace
        cleaned = " ".join(content.split())
        
        # Remove common OCR artifacts
        cleaned = cleaned.replace("_", " ")
        cleaned = cleaned.replace("|", " ")
        
        # Remove repeated characters (common in OCR)
        import re
        cleaned = re.sub(r'(.)\1{3,}', r'\1', cleaned)
        
        return cleaned.strip()
    
    def _extract_key_information(self, content: str) -> Dict[str, Any]:
        """
        Extract key information from chunk content.
        
        Args:
            content: Chunk content
            
        Returns:
            Dictionary with extracted information
        """
        import re
        
        key_info = {
            "articles": [],
            "sections": [],
            "numbers": [],
            "dates": [],
            "references": []
        }
        
        # Extract article references
        article_patterns = [
            r'المادة\s*(\d+)',
            r'Article\s*(\d+)',
            r'البند\s*(\d+)',
            r'Section\s*(\d+)'
        ]
        
        for pattern in article_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            key_info["articles"].extend(matches)
        
        # Extract numbers (percentages, amounts, etc.)
        number_pattern = r'\d+(?:\.\d+)?%?'
        key_info["numbers"] = re.findall(number_pattern, content)
        
        # Extract dates
        date_patterns = [
            r'\d{4}/\d{1,2}/\d{1,2}',
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{4}-\d{1,2}-\d{1,2}'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, content)
            key_info["dates"].extend(matches)
        
        return key_info
    
    async def _generate_answer_from_chunks(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """
        Generate answer from chunks using LLM.
        
        Args:
            question: User question
            chunks: Processed chunks
            user_profile: User profile
            
        Returns:
            Dictionary with answer and metadata
        """
        # Detect language
        has_arabic = any('\u0600' <= char <= '\u06FF' for char in question)
        language = "ar" if has_arabic else "en"
        
        # Create reasoning prompt
        prompt = self._create_reasoning_prompt(question, chunks, user_profile, language)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            # Parse response
            return self._parse_reasoning_response(response.choices[0].message.content, chunks)
            
        except Exception as e:
            # Fallback to template-based response
            return {
                "answer": self.response_templates["no_information"][language],
                "sources_used": [],
                "reasoning_steps": [f"LLM reasoning failed: {str(e)}"]
            }
    
    def _create_reasoning_prompt(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        user_profile: UserProfile,
        language: str
    ) -> Dict[str, str]:
        """
        Create reasoning prompt for LLM.
        
        Args:
            question: User question
            chunks: Processed chunks
            user_profile: User profile
            language: Language (ar/en)
            
        Returns:
            Dictionary with system and user prompts
        """
        if language == "ar":
            system_prompt = f"""أنت مساعد قانوني متخصص في الأنظمة السعودية.

مهمتك: الإجابة على الأسئلة القانونية بناءً فقط على الوثائق المقدمة.

قواعد صارمة:
1. استخدم فقط المعلومات الموجودة في الوثائق المقدمة
2. لا تضيف معلومات من معرفتك العامة
3. إذا لم تجد المعلومة في الوثائق، قل: "هذه المعلومة غير موجودة في الوثائق المقدمة"
4. اذكر مصدر كل معلومة (اسم الوثيقة والمادة)
5. تأكد من أن الإجابة مناسبة لفئة الموظف: {user_profile.employee_type.value}

تنسيق الإجابة:
{{
  "answer": "الإجابة المفصلة بناءً على الوثائق",
  "sources": [
    {{"document": "اسم الوثيقة", "article": "رقم المادة", "content": "النص المرجعي"}}
  ],
  "reasoning_steps": ["خطوة 1", "خطوة 2"],
  "confidence": "high/medium/low"
}}"""
        else:
            system_prompt = f"""You are a Saudi legal AI assistant specialized in government regulations.

Your task: Answer legal questions based ONLY on the provided documents.

STRICT RULES:
1. Use ONLY information from the provided documents
2. Do NOT add information from your general knowledge
3. If information is not in the documents, say: "This information is not found in the provided documents"
4. Cite the source for every piece of information (document name and article)
5. Ensure the answer is appropriate for employee type: {user_profile.employee_type.value}

Response format:
{{
  "answer": "Detailed answer based on documents",
  "sources": [
    {{"document": "document name", "article": "article number", "content": "reference text"}}
  ],
  "reasoning_steps": ["step 1", "step 2"],
  "confidence": "high/medium/low"
}}"""
        
        # Prepare chunks text
        chunks_text = ""
        for i, chunk in enumerate(chunks):
            chunks_text += f"\n--- Document Chunk {i+1} ---\n"
            chunks_text += f"Document: {chunk['document_name']}\n"
            chunks_text += f"Page: {chunk['page_number']}\n"
            if chunk.get('article'):
                chunks_text += f"Article: {chunk['article']}\n"
            chunks_text += f"Content: {chunk['cleaned_content']}\n"
        
        user_prompt = f"""Question: {question}

Available Documents:
{chunks_text}

Please provide a comprehensive answer based ONLY on the information in these documents. Follow the JSON format specified in the system prompt."""
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _parse_reasoning_response(
        self, 
        response: str, 
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Parse LLM reasoning response.
        
        Args:
            response: LLM response
            chunks: Original chunks for reference
            
        Returns:
            Parsed response dictionary
        """
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)
            
            return {
                "answer": parsed.get("answer", "Unable to generate answer"),
                "sources_used": parsed.get("sources", []),
                "reasoning_steps": parsed.get("reasoning_steps", ["LLM reasoning"]),
                "llm_confidence": parsed.get("confidence", "medium")
            }
            
        except Exception as e:
            # Fallback parsing
            return {
                "answer": response if len(response) > 50 else "Unable to generate answer",
                "sources_used": [
                    {
                        "document": chunk["document_name"],
                        "article": chunk.get("article", "N/A"),
                        "content": chunk["content"][:100] + "..."
                    }
                    for chunk in chunks[:3]  # Include top 3 chunks as sources
                ],
                "reasoning_steps": [f"Fallback parsing due to: {str(e)}"]
            }
    
    def _calculate_confidence_level(
        self,
        chunks: List[Dict[str, Any]],
        answer: str,
        reasoning_steps: List[str]
    ) -> ConfidenceLevel:
        """
        Calculate confidence level for the answer.
        
        Args:
            chunks: Processed chunks
            answer: Generated answer
            reasoning_steps: Reasoning steps
            
        Returns:
            ConfidenceLevel enum
        """
        score = 0.0
        
        # Chunk relevance score
        if chunks:
            avg_score = sum(chunk["score"] for chunk in chunks) / len(chunks)
            score += self.confidence_weights["chunk_relevance"] * avg_score
        
        # Content completeness score
        if len(answer) > 200:  # Detailed answer
            score += self.confidence_weights["content_completeness"] * 1.0
        elif len(answer) > 100:  # Moderate answer
            score += self.confidence_weights["content_completeness"] * 0.7
        else:  # Short answer
            score += self.confidence_weights["content_completeness"] * 0.4
        
        # Citation quality score
        if "This information is not found" in answer:
            score += self.confidence_weights["citation_quality"] * 0.3
        elif len(chunks) >= 3:  # Multiple sources
            score += self.confidence_weights["citation_quality"] * 1.0
        elif len(chunks) >= 2:  # Some sources
            score += self.confidence_weights["citation_quality"] * 0.7
        else:  # Single source
            score += self.confidence_weights["citation_quality"] * 0.5
        
        # Answer coherence score (based on reasoning steps)
        if len(reasoning_steps) >= 3:
            score += self.confidence_weights["answer_coherence"] * 1.0
        elif len(reasoning_steps) >= 2:
            score += self.confidence_weights["answer_coherence"] * 0.7
        else:
            score += self.confidence_weights["answer_coherence"] * 0.4
        
        # Convert to confidence level
        if score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW