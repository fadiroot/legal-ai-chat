"""
Legal Retrieval Agent with role-aware Azure AI Search.

This agent performs role-aware, category-aware retrieval using Azure AI Search
with metadata filtering. It NEVER reasons or answers - only retrieves.
"""

from typing import Dict, Any, Optional, List
from azure.search.documents import SearchClient
from azure.core.exceptions import AzureError
import os

from .base_agent import BaseAgent
from .schemas import (
    AgentInput, AgentOutput, AgentState, AgentStatus,
    RetrievalResult, DocumentChunk, UserProfile,
    EmployeeType, EngineerLevel, WageBandLevel
)
from ..services.embedding_service import EmbeddingService


class LegalRetrievalAgent(BaseAgent):
    """
    Agent responsible for role-aware document retrieval.
    
    Key responsibilities:
    1. Perform role-aware, category-aware retrieval
    2. Call Azure AI Search with metadata filtering
    3. NEVER reason or answer - only retrieve
    4. Apply access control at search level
    5. Return structured document chunks
    """
    
    def __init__(self, search_client: SearchClient, config: Optional[Dict[str, Any]] = None):
        super().__init__("LegalRetrievalAgent", config)
        self.search_client = search_client
        self.embedding_service = EmbeddingService()
        
        # Initialize search configuration
        self._init_search_config()
    
    def _init_search_config(self) -> None:
        """Initialize search configuration and filters."""
        
        # Default search parameters
        self.default_top_k = self._get_config_value("default_top_k", 5)
        self.max_top_k = self._get_config_value("max_top_k", 20)
        # Lower threshold to 0.5 for more lenient matching (can be configured)
        self.min_score_threshold = self._get_config_value("min_score_threshold", 0.5)
        
        # Employee type to document type mapping
        self.employee_document_mapping = {
            EmployeeType.ENGINEER: [
                "engineering_regulations",
                "technical_standards", 
                "promotion_guidelines_engineers",
                "professional_development",
                "project_management_guidelines"
            ],
            EmployeeType.WAGE_BAND: [
                "wage_band_regulations",
                "salary_structures_wage_band",
                "benefits_wage_band",
                "leave_policies",
                "training_programs_wage_band"
            ],
            EmployeeType.CIVIL_SERVANT: [
                "executive_hr_regulations",
                "administrative_procedures",
                "civil_service_law",
                "general_policies",
                "benefits_civil_servants"
            ],
            EmployeeType.EXTERNAL: [
                "public_information",
                "general_guidelines",
                "citizen_services"
            ]
        }
        
        # Level-specific document access
        self.level_document_mapping = {
            # Engineer levels
            EngineerLevel.ASSOCIATE: {
                "allowed": ["basic_procedures", "entry_level_guidelines"],
                "restricted": ["senior_management", "strategic_planning"]
            },
            EngineerLevel.PROFESSIONAL: {
                "allowed": ["intermediate_procedures", "project_leadership"],
                "restricted": ["executive_decisions"]
            },
            EngineerLevel.CONSULTANT: {
                "allowed": ["advanced_procedures", "strategic_guidance", "executive_consultation"],
                "restricted": []
            },
            # Wage Band levels
            WageBandLevel.ORDINARY: {
                "allowed": ["basic_operations", "standard_procedures"],
                "restricted": ["technical_leadership", "specialized_procedures"]
            },
            WageBandLevel.SKILLED: {
                "allowed": ["skilled_operations", "technical_procedures"],
                "restricted": ["management_procedures"]
            },
            WageBandLevel.ASSISTANT_TECHNICIAN: {
                "allowed": ["technical_assistance", "specialized_support"],
                "restricted": ["independent_technical_decisions"]
            }
        }
    
    async def _execute_agent_logic(self, input_data: AgentInput) -> AgentOutput:
        """
        Execute legal retrieval logic.
        
        Args:
            input_data: Agent input with state and config
            
        Returns:
            AgentOutput with retrieval results
        """
        state = input_data.state
        
        if not self._validate_state(state):
            return AgentOutput(
                status=AgentStatus.ERROR,
                updated_state=state,
                execution_time_ms=0,
                error_message="Invalid state: missing session_id or user_question"
            )
        
        # Check if user profile exists
        if not state.user_profile:
            return AgentOutput(
                status=AgentStatus.ERROR,
                updated_state=state,
                execution_time_ms=0,
                error_message="User profile required before retrieval"
            )
        
        try:
            # Step 1: Generate question embedding
            embeddings = self.embedding_service.generate_embeddings([state.user_question])
            if not embeddings or not embeddings[0]:
                raise ValueError("Failed to generate embedding for question")
            
            question_embedding = embeddings[0]
            
            # Step 2: Build search filters based on user profile
            search_filters = self._build_search_filters(state.user_profile)
            
            # Step 3: Perform vector search with filters
            search_results = await self._perform_vector_search(
                question_embedding=question_embedding,
                filters=search_filters,
                user_profile=state.user_profile,
                question=state.user_question
            )
            
            # Step 4: Convert to DocumentChunk objects
            document_chunks = self._convert_to_document_chunks(search_results)
            
            # Step 5: Apply post-retrieval filtering
            filtered_chunks = self._apply_post_retrieval_filtering(
                document_chunks, state.user_profile
            )
            
            # Step 6: Create retrieval result
            # Check if any results were below threshold
            below_threshold_count = sum(
                1 for chunk in filtered_chunks 
                if chunk.metadata.get("below_threshold", False)
            )
            
            retrieval_result = RetrievalResult(
                status=AgentStatus.SUCCESS,
                chunks=filtered_chunks,
                total_results=len(filtered_chunks),
                search_metadata={
                    "original_results": len(search_results),
                    "filtered_results": len(filtered_chunks),
                    "search_filters": search_filters,
                    "min_score": min([chunk.score for chunk in filtered_chunks]) if filtered_chunks else 0,
                    "max_score": max([chunk.score for chunk in filtered_chunks]) if filtered_chunks else 0,
                    "below_threshold_count": below_threshold_count,
                    "threshold_warning": below_threshold_count > 0,
                    "min_score_threshold": self.min_score_threshold
                }
            )
            
            # Update state
            state.retrieval_result = retrieval_result
            
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                updated_state=state,
                execution_time_ms=0
            )
            
        except Exception as e:
            error_message = f"Retrieval failed: {str(e)}"
            
            # Log detailed error for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[LegalRetrievalAgent] Error details: {str(e)}", exc_info=True)
            
            # Check if it's a common issue
            error_str = str(e).lower()
            if "index" in error_str or "not found" in error_str:
                error_message = f"Search index not found or not accessible. Please ensure the index exists and contains documents. Original error: {str(e)}"
            elif "connection" in error_str or "timeout" in error_str:
                error_message = f"Azure AI Search connection failed. Please check your credentials and network connection. Original error: {str(e)}"
            elif "vector" in error_str or "embedding" in error_str:
                error_message = f"Vector search configuration error. Please check your index schema. Original error: {str(e)}"
            
            retrieval_result = RetrievalResult(
                status=AgentStatus.ERROR,
                error_message=error_message,
                chunks=[],
                total_results=0
            )
            
            state.retrieval_result = retrieval_result
            
            return AgentOutput(
                status=AgentStatus.ERROR,
                updated_state=state,
                execution_time_ms=0,
                error_message=error_message
            )
    
    def _build_search_filters(self, user_profile: UserProfile) -> Dict[str, Any]:
        """
        Build Azure AI Search filters based on user profile.
        
        Note: Currently, the index schema doesn't include document_type, access_level,
        or employee_types fields. Filters are disabled until these fields are added
        to the index schema.
        
        Args:
            user_profile: User profile with employee type and level
            
        Returns:
            Dictionary with search filters (currently returns None for filter_expression
            since required fields don't exist in index)
        """
        employee_type = user_profile.employee_type
        allowed_doc_types = self.employee_document_mapping.get(employee_type, [])
        
        # TODO: When index schema is updated to include these fields, uncomment filters:
        # filters = []
        # 
        # # Employee type filter
        # if allowed_doc_types:
        #     doc_type_filters = [f"document_type eq '{doc_type}'" for doc_type in allowed_doc_types]
        #     filters.append(f"({' or '.join(doc_type_filters)})")
        # 
        # # Level-based filtering
        # if user_profile.level and user_profile.level in self.level_document_mapping:
        #     level_config = self.level_document_mapping[user_profile.level]
        #     if level_config.get("allowed"):
        #         allowed_level_filters = [f"access_level eq '{level}'" for level in level_config["allowed"]]
        #         filters.append(f"({' or '.join(allowed_level_filters)})")
        #     if level_config.get("restricted"):
        #         restricted_filters = [f"access_level ne '{level}'" for level in level_config["restricted"]]
        #         filters.extend(restricted_filters)
        # 
        # # Employee type access filter
        # filters.append(f"(employee_types/any(t: t eq '{employee_type.value}') or employee_types/any(t: t eq 'All'))")
        # 
        # final_filter = " and ".join(filters) if filters else None
        
        # For now, return None to disable filters (fields don't exist in index)
        # Access control will be handled in post-retrieval filtering instead
        return {
            "filter_expression": None,  # Disabled until index schema is updated
            "employee_type": employee_type.value,
            "level": user_profile.level.value if user_profile.level else None,
            "allowed_document_types": allowed_doc_types,
            "note": "Filters disabled - index schema missing document_type, access_level, employee_types fields"
        }
    
    async def _perform_vector_search(
        self,
        question_embedding: List[float],
        filters: Dict[str, Any],
        user_profile: UserProfile,
        question: str
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search with Azure AI Search.
        
        Args:
            question_embedding: Question embedding vector
            filters: Search filters
            user_profile: User profile
            question: Original question
            
        Returns:
            List of search results
        """
        try:
            # Validate search client
            if not self.search_client:
                raise Exception("Search client not initialized. Please check Azure AI Search configuration.")
            
            # Configure search parameters
            top_k = min(self._get_config_value("top_k", self.default_top_k), self.max_top_k)
            
            # Validate embedding
            if not question_embedding or len(question_embedding) == 0:
                raise Exception("Invalid question embedding. Embedding generation may have failed.")
            
            # Import VectorizedQuery for proper vector search
            from azure.search.documents.models import VectorizedQuery
            
            # Create vector query with required 'kind' parameter
            # Note: VectorizedQuery automatically sets the 'kind' parameter
            vector_query = VectorizedQuery(
                vector=question_embedding,
                k_nearest_neighbors=top_k,
                fields="contentVector"  # Use camelCase as per Azure Search schema
            )
            
            # Only select fields that actually exist in the index schema
            # Current index fields: id, content, document_name, page_number, chunk_index, token_count, contentVector
            search_params = {
                "search_text": "",  # Empty string for pure vector search (not None)
                "vector_queries": [vector_query],
                "select": [
                    "id", "content", "document_name", "page_number", 
                    "chunk_index", "token_count"
                ],
                "top": top_k
            }
            
            # Add filter if exists
            if filters.get("filter_expression"):
                search_params["filter"] = filters["filter_expression"]
            
            # Execute search
            try:
                results = self.search_client.search(**search_params)
            except Exception as search_error:
                # Provide more specific error messages
                error_msg = str(search_error)
                if "index" in error_msg.lower():
                    raise Exception(f"Search index not found. Please create the index first using /documents/create-index. Details: {error_msg}")
                elif "unauthorized" in error_msg.lower() or "authentication" in error_msg.lower():
                    raise Exception(f"Azure AI Search authentication failed. Please check your API key. Details: {error_msg}")
                else:
                    raise Exception(f"Search execution failed: {error_msg}")
            
            # Convert to list and add scores (collect all results first since iterator can only be used once)
            all_results = []
            for result in results:
                # Calculate score (Azure AI Search returns @search.score)
                score = getattr(result, '@search.score', 0.0)
                
                all_results.append({
                    "id": result.get("id"),
                    "content": result.get("content", ""),
                    "document_name": result.get("document_name", ""),
                    "page_number": result.get("page_number", 0),
                    "chunk_index": result.get("chunk_index", 0),
                    "article": result.get("article"),
                    "document_type": result.get("document_type"),
                    "employee_types": result.get("employee_types", []),
                    "access_level": result.get("access_level"),
                    "metadata": result.get("metadata", {}),
                    "score": score
                })
            
            # If no results found, provide helpful message
            if len(all_results) == 0:
                raise Exception("No documents found in search index. Please upload documents using /documents/upload endpoint.")
            
            # Sort by score descending
            all_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Filter results above threshold
            search_results = [r for r in all_results if r["score"] >= self.min_score_threshold]
            
            # If no results meet threshold, return the best results anyway (with lower confidence flag)
            # This allows the pipeline to continue and provide a response, even if relevance is lower
            if len(search_results) == 0:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Found {len(all_results)} documents but none met the minimum score threshold ({self.min_score_threshold}). "
                    f"Returning top {top_k} results with lower confidence."
                )
                
                # Return top K results even if below threshold, but mark them
                search_results = all_results[:top_k]
                for result in search_results:
                    result["below_threshold"] = True  # Flag to indicate low relevance
            
            return search_results
            
        except AzureError as e:
            raise Exception(f"Azure AI Search service error: {str(e)}. Please check your Azure AI Search configuration.")
        except Exception as e:
            # Re-raise with context if it's already our formatted error
            if "Search index not found" in str(e) or "authentication failed" in str(e) or "No documents found" in str(e):
                raise
            raise Exception(f"Search execution error: {str(e)}")
    
    def _convert_to_document_chunks(self, search_results: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Convert search results to DocumentChunk objects.
        
        Args:
            search_results: Raw search results
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        for result in search_results:
            try:
                # Parse employee types (if field exists in index)
                employee_types = []
                if result.get("employee_types"):
                    for emp_type_str in result["employee_types"]:
                        try:
                            if emp_type_str != "All":
                                employee_types.append(EmployeeType(emp_type_str))
                        except ValueError:
                            continue  # Skip invalid employee types
                
                # Create DocumentChunk with only fields that exist in current index schema
                # Note: article, document_type, employee_types, access_level are optional
                # and may not exist in the index schema yet
                chunk = DocumentChunk(
                    content=result.get("content", ""),
                    document_name=result.get("document_name", ""),
                    page_number=result.get("page_number", 0),
                    chunk_index=result.get("chunk_index", 0),
                    score=result.get("score", 0.0),
                    article=result.get("article"),  # Optional field - may not exist
                    employee_type=employee_types if employee_types else None,
                    metadata={
                        "document_type": result.get("document_type"),  # Optional - may not exist
                        "access_level": result.get("access_level"),  # Optional - may not exist
                        "original_metadata": result.get("metadata", {}),  # Optional - may not exist
                        "search_id": result.get("id"),
                        "token_count": result.get("token_count", 0),  # Available in index
                        "below_threshold": result.get("below_threshold", False)  # Flag for low relevance results
                    }
                )
                
                chunks.append(chunk)
                
            except Exception as e:
                # Log error but continue with other results
                print(f"Error converting search result to DocumentChunk: {e}")
                continue
        
        return chunks
    
    def _apply_post_retrieval_filtering(
        self, 
        chunks: List[DocumentChunk], 
        user_profile: UserProfile
    ) -> List[DocumentChunk]:
        """
        Apply additional filtering after retrieval.
        
        Args:
            chunks: List of document chunks
            user_profile: User profile
            
        Returns:
            Filtered list of document chunks
        """
        filtered_chunks = []
        
        for chunk in chunks:
            # Check employee type access
            if chunk.employee_type:
                if user_profile.employee_type not in chunk.employee_type:
                    continue  # Skip if user type not in allowed types
            
            # Check access level restrictions
            access_level = chunk.metadata.get("access_level")
            if access_level and user_profile.level:
                level_config = self.level_document_mapping.get(user_profile.level, {})
                restricted_levels = level_config.get("restricted", [])
                
                if access_level in restricted_levels:
                    continue  # Skip restricted content
            
            # Additional content-based filtering could go here
            
            filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get search statistics and performance metrics.
        
        Returns:
            Dictionary with search statistics
        """
        # This would typically track metrics like:
        # - Average search time
        # - Cache hit rates  
        # - Most common queries
        # - Filter effectiveness
        
        return {
            "agent_name": self.agent_name,
            "total_searches": 0,  # Would be tracked in production
            "average_results": 0,
            "cache_enabled": False,  # Would implement caching in production
            "supported_employee_types": list(self.employee_document_mapping.keys()),
            "min_score_threshold": self.min_score_threshold
        }