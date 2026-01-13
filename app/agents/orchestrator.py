"""MAF Orchestrator using Microsoft Agent Framework 100%."""

import time
import uuid
import logging
import json
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime
from azure.search.documents import SearchClient

if TYPE_CHECKING:
    from agent_framework import WorkflowContext

from .schemas import (
    AgentState, AgentStatus, FinalResponse, ConfidenceLevel, AgentInput
)
from .classification_agent import ClassificationAgent
from .user_profiling_agent import UserProfilingAgent
from .legal_retrieval_agent import LegalRetrievalAgent
from .legal_reasoning_agent import LegalReasoningAgent
from .validation_agent import ValidationAgent
from .approval_agent import ApprovalAgent

logger = logging.getLogger(__name__)

AGENT_FRAMEWORK_AVAILABLE = None
_executor = None
_WorkflowBuilder = None
_WorkflowContext = None


def _import_agent_framework():
    """Lazy import of agent-framework - only when needed."""
    global AGENT_FRAMEWORK_AVAILABLE, _executor, _WorkflowBuilder, _WorkflowContext
    if AGENT_FRAMEWORK_AVAILABLE is False:
        raise ImportError(
            "agent-framework is required. Install with: pip install agent-framework --pre"
        )
    if AGENT_FRAMEWORK_AVAILABLE is True:
        return _executor, _WorkflowBuilder, _WorkflowContext
    try:
        from agent_framework import executor, WorkflowBuilder, WorkflowContext
        _executor = executor
        _WorkflowBuilder = WorkflowBuilder
        _WorkflowContext = WorkflowContext
        AGENT_FRAMEWORK_AVAILABLE = True
        logger.info("agent-framework imported successfully")
        return _executor, _WorkflowBuilder, _WorkflowContext
    except ImportError as e:
        AGENT_FRAMEWORK_AVAILABLE = False
        logger.error(
            f"agent-framework package is required but not available: {e}. "
            "Install with: pip install agent-framework --pre"
        )
        raise ImportError(
            "agent-framework is required. Install with: pip install agent-framework --pre"
        ) from e


class MAFOrchestrator:
    """Microsoft Agent Framework Orchestrator using agent-framework 100%."""
    
    def __init__(self, search_client: SearchClient, config: Optional[Dict[str, Any]] = None):
        """Initialize MAF Orchestrator with agent-framework."""
        try:
            executor, WorkflowBuilder, WorkflowContext = _import_agent_framework()
            if executor is None or WorkflowBuilder is None or WorkflowContext is None:
                raise ImportError(
                    "agent-framework imports failed. Install with: pip install agent-framework --pre"
                )
            self._executor = executor
            self._WorkflowBuilder = WorkflowBuilder
            self._WorkflowContext = WorkflowContext
        except ImportError:
            raise ImportError(
                "agent-framework is required. Install with: pip install agent-framework --pre"
            )
        
        self.config = config or {}
        self.search_client = search_client
        self._init_agents()
        self.workflow = self._create_workflow()
        self.max_retries = self._get_config_value("max_retries", 3)
        self.timeout_seconds = self._get_config_value("timeout_seconds", 300)
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0,
            "agent_performance": {}
        }
    
    def _init_agents(self) -> None:
        """Initialize all agents in the workflow."""
        agent_configs = self.config.get("agents", {})
        self.classification_agent = ClassificationAgent(config=agent_configs.get("classification", {}))
        self.user_profiling_agent = UserProfilingAgent(config=agent_configs.get("user_profiling", {}))
        self.legal_retrieval_agent = LegalRetrievalAgent(search_client=self.search_client, config=agent_configs.get("legal_retrieval", {}))
        self.legal_reasoning_agent = LegalReasoningAgent(config=agent_configs.get("legal_reasoning", {}))
        self.validation_agent = ValidationAgent(config=agent_configs.get("validation", {}))
        self.approval_agent = ApprovalAgent(config=agent_configs.get("approval", {}))
        self.agent_pipeline = [
            ("classification", self.classification_agent),
            ("user_profiling", self.user_profiling_agent),
            ("legal_retrieval", self.legal_retrieval_agent),
            ("legal_reasoning", self.legal_reasoning_agent),
            ("validation", self.validation_agent),
            ("approval", self.approval_agent)
        ]
    
    def _create_workflow(self):
        """Create agent-framework workflow with all agents."""
        builder = self._WorkflowBuilder()
        classification_executor = self._create_classification_executor()
        user_profiling_executor = self._create_user_profiling_executor()
        legal_retrieval_executor = self._create_legal_retrieval_executor()
        legal_reasoning_executor = self._create_legal_reasoning_executor()
        validation_executor = self._create_validation_executor()
        approval_executor = self._create_approval_executor()
        builder.set_start_executor(classification_executor)
        builder.add_edge(classification_executor, user_profiling_executor)
        builder.add_edge(user_profiling_executor, legal_retrieval_executor)
        builder.add_edge(legal_retrieval_executor, legal_reasoning_executor)
        builder.add_edge(legal_reasoning_executor, validation_executor)
        builder.add_edge(validation_executor, approval_executor)
        return builder.build()
    
    def _create_classification_executor(self):
        """Create classification agent executor."""
        WorkflowContext = self._WorkflowContext
        async def classification_executor_impl(state_json: str, ctx: WorkflowContext[str]) -> None:
            try:
                state_dict = json.loads(state_json)
                state = AgentState(**state_dict)
                agent_input = AgentInput(
                    state=state,
                    agent_config=self.config.get("agents", {}).get("classification", {})
                )
                agent_output = await self.classification_agent.execute(agent_input)
                state_dict = agent_output.updated_state.model_dump(mode='json')
                await ctx.send_message(json.dumps(state_dict))
            except Exception as e:
                logger.error(f"Classification agent failed: {e}")
                state_dict = json.loads(state_json)
                state_dict["metadata"]["error"] = str(e)
                await ctx.send_message(json.dumps(state_dict))
        return self._executor(classification_executor_impl)
    
    def _create_user_profiling_executor(self):
        """Create user profiling agent executor."""
        WorkflowContext = self._WorkflowContext
        async def user_profiling_executor_impl(state_json: str, ctx: WorkflowContext[str]) -> None:
            try:
                state_dict = json.loads(state_json)
                state = AgentState(**state_dict)
                agent_input = AgentInput(
                    state=state,
                    agent_config=self.config.get("agents", {}).get("user_profiling", {})
                )
                agent_output = await self.user_profiling_agent.execute(agent_input)
                state_dict = agent_output.updated_state.model_dump(mode='json')
                await ctx.send_message(json.dumps(state_dict))
            except Exception as e:
                logger.error(f"User profiling agent failed: {e}")
                state_dict = json.loads(state_json)
                state_dict["metadata"]["error"] = str(e)
                await ctx.send_message(json.dumps(state_dict))
        return self._executor(user_profiling_executor_impl)
    
    def _create_legal_retrieval_executor(self):
        """Create legal retrieval agent executor."""
        WorkflowContext = self._WorkflowContext
        async def legal_retrieval_executor_impl(state_json: str, ctx: WorkflowContext[str]) -> None:
            try:
                state_dict = json.loads(state_json)
                state = AgentState(**state_dict)
                agent_input = AgentInput(
                    state=state,
                    agent_config=self.config.get("agents", {}).get("legal_retrieval", {})
                )
                agent_output = await self.legal_retrieval_agent.execute(agent_input)
                state_dict = agent_output.updated_state.model_dump(mode='json')
                await ctx.send_message(json.dumps(state_dict))
            except Exception as e:
                logger.error(f"Legal retrieval agent failed: {e}")
                state_dict = json.loads(state_json)
                state_dict["metadata"]["error"] = str(e)
                await ctx.send_message(json.dumps(state_dict))
        return self._executor(legal_retrieval_executor_impl)
    
    def _create_legal_reasoning_executor(self):
        """Create legal reasoning agent executor."""
        WorkflowContext = self._WorkflowContext
        async def legal_reasoning_executor_impl(state_json: str, ctx: WorkflowContext[str]) -> None:
            try:
                state_dict = json.loads(state_json)
                state = AgentState(**state_dict)
                agent_input = AgentInput(
                    state=state,
                    agent_config=self.config.get("agents", {}).get("legal_reasoning", {})
                )
                agent_output = await self.legal_reasoning_agent.execute(agent_input)
                state_dict = agent_output.updated_state.model_dump(mode='json')
                await ctx.send_message(json.dumps(state_dict))
            except Exception as e:
                logger.error(f"Legal reasoning agent failed: {e}")
                state_dict = json.loads(state_json)
                state_dict["metadata"]["error"] = str(e)
                await ctx.send_message(json.dumps(state_dict))
        return self._executor(legal_reasoning_executor_impl)
    
    def _create_validation_executor(self):
        """Create validation agent executor."""
        WorkflowContext = self._WorkflowContext
        async def validation_executor_impl(state_json: str, ctx: WorkflowContext[str]) -> None:
            try:
                state_dict = json.loads(state_json)
                state = AgentState(**state_dict)
                agent_input = AgentInput(
                    state=state,
                    agent_config=self.config.get("agents", {}).get("validation", {})
                )
                agent_output = await self.validation_agent.execute(agent_input)
                state_dict = agent_output.updated_state.model_dump(mode='json')
                await ctx.send_message(json.dumps(state_dict))
            except Exception as e:
                logger.error(f"Validation agent failed: {e}")
                state_dict = json.loads(state_json)
                state_dict["metadata"]["error"] = str(e)
                await ctx.send_message(json.dumps(state_dict))
        return self._executor(validation_executor_impl)
    
    def _create_approval_executor(self):
        """Create approval agent executor."""
        WorkflowContext = self._WorkflowContext
        async def approval_executor_impl(state_json: str, ctx: WorkflowContext[str]) -> None:
            try:
                state_dict = json.loads(state_json)
                state = AgentState(**state_dict)
                agent_input = AgentInput(
                    state=state,
                    agent_config=self.config.get("agents", {}).get("approval", {})
                )
                agent_output = await self.approval_agent.execute(agent_input)
                state_dict = agent_output.updated_state.model_dump(mode='json')
                await ctx.send_message(json.dumps(state_dict))
            except Exception as e:
                logger.error(f"Approval agent failed: {e}")
                state_dict = json.loads(state_json)
                state_dict["metadata"]["error"] = str(e)
                await ctx.send_message(json.dumps(state_dict))
        return self._executor(approval_executor_impl)
    
    async def process_question(self, question: str, session_id: Optional[str] = None) -> FinalResponse:
        """Process a legal question through the MAF pipeline."""
        start_time = time.time()
        if not session_id:
            session_id = str(uuid.uuid4())
        state = AgentState(
            session_id=session_id,
            user_question=question,
            metadata={"orchestrator_start_time": datetime.now().isoformat()}
        )
        try:
            final_state = await self._execute_workflow(state)
            response = self._generate_final_response(final_state, start_time)
            self._update_execution_stats(True, time.time() - start_time)
            return response
        except Exception as e:
            error_response = self._generate_error_response(
                question=question,
                session_id=session_id,
                error=str(e),
                execution_time=time.time() - start_time
            )
            self._update_execution_stats(False, time.time() - start_time)
            return error_response
    
    async def _execute_workflow(self, state: AgentState) -> AgentState:
        """Execute the agent-framework workflow."""
        state_json = json.dumps(state.model_dump(mode='json'))
        try:
            result = await self.workflow.run(state_json)
            logger.info(f"Workflow result type: {type(result)}, is_list: {isinstance(result, list)}, length: {len(result) if isinstance(result, list) else 'N/A'}")
            final_data = None
            if isinstance(result, list):
                if len(result) == 0:
                    logger.warning("Workflow returned empty list, trying run_stream() as fallback")
                    async for event in self.workflow.run_stream(state_json):
                        if isinstance(event, str):
                            final_data = event
                        elif hasattr(event, 'data'):
                            final_data = event.data
                        elif hasattr(event, 'message'):
                            final_data = event.message
                    if final_data is None:
                        raise ValueError("Workflow returned empty list and run_stream() produced no output")
                else:
                    logger.info(f"Processing {len(result)} items from workflow")
                    if all(isinstance(item, str) for item in result if item is not None):
                        logger.info("Result is a list of JSON strings, using the last one")
                        for idx in range(len(result) - 1, -1, -1):
                            if result[idx] is not None and result[idx].strip():
                                final_data = result[idx].strip()
                                logger.info(f"Using JSON string from item {idx}, length: {len(final_data)}")
                                break
                    if final_data is None:
                        for idx in range(len(result) - 1, -1, -1):
                            item = result[idx]
                            if item is None:
                                continue
                            
                            logger.info(f"Checking item {idx}: type={type(item).__name__}, value preview: {str(item)[:200]}")
                            if isinstance(item, str) and item.strip():
                                final_data = item.strip()
                                logger.info(f"Found valid JSON string in item {idx}, length: {len(final_data)}")
                                break
                            elif isinstance(item, dict):
                                final_data = item
                                logger.info(f"Found dict in item {idx}")
                                break
                            elif hasattr(item, 'data') and item.data is not None:
                                final_data = item.data
                                logger.info(f"Found data in item {idx} via .data attribute")
                                break
                            elif hasattr(item, 'message') and item.message is not None:
                                final_data = item.message
                                logger.info(f"Found data in item {idx} via .message attribute")
                                break
                            elif hasattr(item, 'output') and item.output is not None:
                                final_data = item.output
                                logger.info(f"Found data in item {idx} via .output attribute")
                                break
                    if final_data is None and len(result) > 0:
                        last_item = result[-1]
                        logger.info(f"Trying detailed extraction from last item: type={type(last_item).__name__}")
                        if hasattr(last_item, 'data') and last_item.data is not None:
                            final_data = last_item.data
                            logger.info("Found data via .data attribute")
                        elif final_data is None and hasattr(last_item, 'message') and last_item.message is not None:
                            final_data = last_item.message
                            logger.info("Found data via .message attribute")
                        elif final_data is None and hasattr(last_item, 'output') and last_item.output is not None:
                            final_data = last_item.output
                            logger.info("Found data via .output attribute")
                        elif final_data is None and hasattr(last_item, '__dict__'):
                            event_dict = last_item.__dict__
                            logger.info(f"Event attributes: {list(event_dict.keys())}")
                            if 'data' in event_dict and event_dict['data'] is not None:
                                final_data = event_dict['data']
                                logger.info("Found data via __dict__['data']")
                        elif final_data is None:
                            event_str = str(last_item)
                            logger.warning(f"Could not extract data from event, trying string parsing. Event: {event_str[:300]}")
                            if 'data=' in event_str:
                                try:
                                    start_idx = event_str.find('data=') + 5
                                    while start_idx < len(event_str) and event_str[start_idx] in ' \n\t':
                                        start_idx += 1
                                    brace_count = 0
                                    end_idx = start_idx
                                    for i, char in enumerate(event_str[start_idx:], start_idx):
                                        if char == '{':
                                            brace_count += 1
                                        elif char == '}':
                                            brace_count -= 1
                                            if brace_count == 0:
                                                end_idx = i + 1
                                                break
                                    if end_idx > start_idx:
                                        final_data = event_str[start_idx:end_idx]
                                        logger.info(f"Extracted data from string representation: {final_data[:200]}...")
                                except Exception as e:
                                    logger.error(f"Error parsing data from string: {e}")
            elif hasattr(result, 'data'):
                final_data = result.data
                logger.info("Found data via result.data")
            elif hasattr(result, 'message'):
                final_data = result.message
                logger.info("Found data via result.message")
            elif hasattr(result, 'output'):
                final_data = result.output
                logger.info("Found data via result.output")
            elif isinstance(result, str) and result.strip():
                final_data = result.strip()
                logger.info("Result is a string")
            elif isinstance(result, dict):
                final_data = result
                logger.info("Result is a dict")
            
            if final_data is None:
                logger.error(f"Unable to extract data from workflow result. Result type: {type(result)}")
                logger.error(f"Result value (first 1000 chars): {str(result)[:1000]}")
                if isinstance(result, list):
                    logger.error(f"List length: {len(result)}")
                    for i, item in enumerate(result):
                        logger.error(f"  Item {i}: type={type(item)}, value={str(item)[:200]}")
                raise ValueError("No output data received from workflow")
            if isinstance(final_data, dict):
                state_dict = final_data
            elif isinstance(final_data, list):
                logger.info(f"final_data is a list with {len(final_data)} items, extracting last item")
                if len(final_data) == 0:
                    raise ValueError("final_data is an empty list")
                last_item = final_data[-1]
                logger.info(f"Last item type: {type(last_item)}, preview: {str(last_item)[:200]}")
                if isinstance(last_item, str):
                    try:
                        state_dict = json.loads(last_item)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse last item as JSON. Error: {e}")
                        logger.error(f"Last item (first 500 chars): {last_item[:500]}")
                        raise ValueError(f"Workflow output is not valid JSON: {str(e)}. Output preview: {last_item[:200]}")
                elif isinstance(last_item, dict):
                    state_dict = last_item
                else:
                    try:
                        state_dict = json.loads(str(last_item))
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.error(f"Unable to parse last item. Type: {type(last_item)}, Value: {str(last_item)[:500]}")
                        raise ValueError(f"Unable to parse workflow output: {type(last_item)}. Error: {str(e)}")
            elif isinstance(final_data, str):
                try:
                    state_dict = json.loads(final_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse workflow output as JSON. Error: {e}")
                    logger.error(f"Output (first 500 chars): {final_data[:500]}")
                    raise ValueError(f"Workflow output is not valid JSON: {str(e)}. Output preview: {final_data[:200]}")
            else:
                try:
                    state_dict = json.loads(str(final_data))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Unable to parse workflow output. Type: {type(final_data)}, Value: {str(final_data)[:500]}")
                    raise ValueError(f"Unable to parse workflow output: {type(final_data)}. Error: {str(e)}")
            return AgentState(**state_dict)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise ValueError(f"Failed to parse workflow output as JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Workflow execution error: {e}", exc_info=True)
            raise
    
    def _generate_final_response(self, state: AgentState, start_time: float) -> FinalResponse:
        """Generate final response from pipeline state."""
        execution_time_ms = int((time.time() - start_time) * 1000)
        if (state.classification_result and 
            state.classification_result.status == AgentStatus.CLARIFICATION_REQUIRED):
            return FinalResponse(
                answer=state.classification_result.question_to_user,
                sources=[],
                employee_type="Unknown",
                level=None,
                confidence=ConfidenceLevel.LOW,
                needs_human_review=False,
                session_id=state.session_id,
                processing_time_ms=execution_time_ms,
                agent_trace=state.metadata.get("agent_trace", [])
            )
        if (state.approval_result and 
            state.approval_result.requires_human_approval and 
            not state.approval_result.approved):
            
            return FinalResponse(
                answer="Your request has been submitted for human review. You will receive a response once it has been reviewed and approved.",
                sources=[],
                employee_type=state.user_profile.employee_type.value if state.user_profile else "Unknown",
                level=state.user_profile.level.value if state.user_profile and state.user_profile.level else None,
                confidence=ConfidenceLevel.MEDIUM,
                needs_human_review=True,
                session_id=state.session_id,
                processing_time_ms=execution_time_ms,
                agent_trace=state.metadata.get("agent_trace", [])
            )
        if state.reasoning_result and state.reasoning_result.answer:
            sources = []
            if state.reasoning_result.sources_used:
                sources = [
                    {
                        "document": source.get("document", ""),
                        "article": source.get("article", ""),
                        "content": source.get("content", "")[:200] + "..." if source.get("content") else ""
                    }
                    for source in state.reasoning_result.sources_used
                ]
            
            return FinalResponse(
                answer=state.reasoning_result.answer,
                sources=sources,
                employee_type=state.user_profile.employee_type.value if state.user_profile else "Unknown",
                level=state.user_profile.level.value if state.user_profile and state.user_profile.level else None,
                confidence=state.validation_result.confidence if state.validation_result else ConfidenceLevel.MEDIUM,
                needs_human_review=state.validation_result.needs_human_review if state.validation_result else False,
                session_id=state.session_id,
                processing_time_ms=execution_time_ms,
                agent_trace=state.metadata.get("agent_trace", [])
            )
        return FinalResponse(
            answer="I apologize, but I was unable to generate an answer to your question. Please try rephrasing your question or contact support for assistance.",
            sources=[],
            employee_type=state.user_profile.employee_type.value if state.user_profile else "Unknown",
            level=state.user_profile.level.value if state.user_profile and state.user_profile.level else None,
            confidence=ConfidenceLevel.LOW,
            needs_human_review=True,
            session_id=state.session_id,
            processing_time_ms=execution_time_ms,
            agent_trace=state.metadata.get("agent_trace", [])
        )
    
    def _generate_error_response(self, question: str, session_id: str, error: str, execution_time: float) -> FinalResponse:
        """Generate error response."""
        return FinalResponse(
            answer=f"I apologize, but an error occurred while processing your question. Please try again later. Error: {error}",
            sources=[],
            employee_type="Unknown",
            level=None,
            confidence=ConfidenceLevel.LOW,
            needs_human_review=True,
            session_id=session_id,
            processing_time_ms=int(execution_time * 1000),
            agent_trace=[{
                "agent_name": "orchestrator",
                "event": "error",
                "timestamp": datetime.now().isoformat(),
                "metadata": {"error": error}
            }]
        )
    
    def _update_execution_stats(self, success: bool, execution_time: float) -> None:
        """Update execution statistics."""
        self.execution_stats["total_executions"] += 1
        if success:
            self.execution_stats["successful_executions"] += 1
        else:
            self.execution_stats["failed_executions"] += 1
        total_time = (self.execution_stats["average_execution_time"] * 
                     (self.execution_stats["total_executions"] - 1) + 
                     execution_time)
        self.execution_stats["average_execution_time"] = total_time / self.execution_stats["total_executions"]
    
    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback."""
        return self.config.get(key, default)
    
    async def handle_clarification_response(self, session_id: str, clarification_response: str) -> FinalResponse:
        """Handle user response to clarification question."""
        enhanced_question = f"User clarification: {clarification_response}"
        return await self.process_question(enhanced_question, session_id)
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get orchestrator execution statistics."""
        return {
            "orchestrator_stats": self.execution_stats.copy(),
            "agent_stats": {
                name: agent.get_search_statistics() if hasattr(agent, 'get_search_statistics') else {}
                for name, agent in self.agent_pipeline
            },
            "configuration": {
                "max_retries": self.max_retries,
                "timeout_seconds": self.timeout_seconds,
                "total_agents": len(self.agent_pipeline)
            }
        }
    
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get pending approval requests."""
        return self.approval_agent.get_pending_requests()
    
    def approve_request(self, request_id: str, reviewer_id: str, reviewer_notes: Optional[str] = None) -> bool:
        """Approve a pending request."""
        return self.approval_agent.approve_request(request_id, reviewer_id, reviewer_notes)
    
    def reject_request(self, request_id: str, reviewer_id: str, rejection_reason: str) -> bool:
        """Reject a pending request."""
        return self.approval_agent.reject_request(request_id, reviewer_id, rejection_reason)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all agents."""
        health_status = {
            "orchestrator": "healthy",
            "agents": {},
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat()
        }
        
        for agent_name, agent in self.agent_pipeline:
            try:
                health_status["agents"][agent_name] = {
                    "status": "healthy",
                    "agent_type": type(agent).__name__,
                    "config_loaded": bool(agent.config)
                }
            except Exception as e:
                health_status["agents"][agent_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"
        
        return health_status