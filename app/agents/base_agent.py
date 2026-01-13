"""
Base agent class for Microsoft Agent Framework implementation.
"""

import time
import uuid
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

from .schemas import AgentInput, AgentOutput, AgentState, AgentStatus

# Configure logging
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Base class for all MAF agents.
    
    Provides common functionality for:
    - State management
    - Execution timing
    - Error handling
    - Logging and tracing
    """
    
    def __init__(self, agent_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base agent.
        
        Args:
            agent_name: Name of the agent
            config: Agent-specific configuration
        """
        self.agent_name = agent_name
        self.config = config or {}
        self.agent_id = str(uuid.uuid4())
        
    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """
        Execute the agent with timing and error handling.
        
        Args:
            input_data: Agent input with state and config
            
        Returns:
            AgentOutput with updated state and execution metadata
        """
        start_time = time.time()
        debug_mode = self._get_config_value("debug", False)
        
        try:
            # Debug logging
            if debug_mode:
                logger.debug(f"[{self.agent_name}] Starting execution")
                logger.debug(f"[{self.agent_name}] Input state: session_id={input_data.state.session_id}")
                logger.debug(f"[{self.agent_name}] Question: {input_data.state.user_question[:100]}...")
            
            # Update state with agent execution trace
            self._add_agent_trace(input_data.state, "started")
            
            # Execute agent-specific logic
            result = await self._execute_agent_logic(input_data)
            
            # Update execution time
            execution_time_ms = int((time.time() - start_time) * 1000)
            result.execution_time_ms = execution_time_ms
            
            # Add completion trace
            self._add_agent_trace(
                result.updated_state, 
                "completed", 
                {"execution_time_ms": execution_time_ms, "status": result.status.value}
            )
            
            # Debug logging
            if debug_mode:
                logger.debug(f"[{self.agent_name}] Completed in {execution_time_ms}ms")
                logger.debug(f"[{self.agent_name}] Status: {result.status.value}")
                if result.error_message:
                    logger.warning(f"[{self.agent_name}] Error: {result.error_message}")
            
            return result
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Error logging
            logger.error(f"[{self.agent_name}] Execution failed: {str(e)}", exc_info=True)
            
            # Add error trace
            self._add_agent_trace(
                input_data.state,
                "error",
                {"error": str(e), "execution_time_ms": execution_time_ms, "error_type": type(e).__name__}
            )
            
            return AgentOutput(
                status=AgentStatus.ERROR,
                updated_state=input_data.state,
                execution_time_ms=execution_time_ms,
                error_message=str(e)
            )
    
    @abstractmethod
    async def _execute_agent_logic(self, input_data: AgentInput) -> AgentOutput:
        """
        Agent-specific execution logic.
        
        Args:
            input_data: Agent input with state and config
            
        Returns:
            AgentOutput with updated state
        """
        pass
    
    def _add_agent_trace(
        self, 
        state: AgentState, 
        event: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add agent execution trace to state.
        
        Args:
            state: Current agent state
            event: Event type (started, completed, error)
            metadata: Additional metadata for the trace
        """
        trace_entry = {
            "agent_name": self.agent_name,
            "agent_id": self.agent_id,
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Initialize agent_trace if not exists
        if not hasattr(state, 'metadata') or 'agent_trace' not in state.metadata:
            if not hasattr(state, 'metadata'):
                state.metadata = {}
            state.metadata['agent_trace'] = []
        
        state.metadata['agent_trace'].append(trace_entry)
        state.updated_at = datetime.now()
    
    def _validate_state(self, state: AgentState) -> bool:
        """
        Validate agent state before processing.
        
        Args:
            state: Agent state to validate
            
        Returns:
            True if state is valid, False otherwise
        """
        if not state.session_id:
            return False
        if not state.user_question:
            return False
        return True
    
    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with fallback.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)