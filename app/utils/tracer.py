"""
Agent execution tracer for Agent Theater feature
Collects display-safe execution traces without performance impact
"""

import time
import uuid
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call within an agent execution"""
    tool_name: str
    duration_ms: int
    status: str  # 'success', 'error', 'timeout'
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentExecution:
    """Represents a single agent's execution"""
    agent_name: str
    display_name: str
    started_at: int  # timestamp in ms
    duration_ms: int = 0
    status: str = 'success'  # 'success', 'error', 'timeout'
    tool_calls: List[ToolCall] = field(default_factory=list)
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    confidence: Optional[float] = None
    sources_used: List[str] = field(default_factory=list)


@dataclass
class AgentTrace:
    """Complete trace of multi-agent execution"""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    total_duration_ms: int = 0
    agent_executions: List[AgentExecution] = field(default_factory=list)
    routing_decision: Optional[Dict[str, Any]] = None


class AgentTracer:
    """
    Lightweight tracer for collecting agent execution information
    Designed to have minimal performance impact on the system
    """
    
    def __init__(self):
        self.current_trace: Optional[AgentTrace] = None
        self.current_execution: Optional[AgentExecution] = None
        self._start_time: Optional[int] = None
    
    def start_trace(self) -> str:
        """Start a new trace session"""
        self.current_trace = AgentTrace()
        self._start_time = int(time.time() * 1000)
        logger.debug(f"Started trace: {self.current_trace.execution_id}")
        return self.current_trace.execution_id
    
    def set_routing_decision(self, primary_agent: str, confidence: float, fallback_used: Optional[str] = None):
        """Record the routing decision"""
        if not self.current_trace:
            return
            
        self.current_trace.routing_decision = {
            "primary_agent": primary_agent,
            "confidence": confidence,
            "fallback_used": fallback_used
        }
    
    @contextmanager
    def trace_agent(self, agent_name: str, display_name: str):
        """Context manager to trace an agent execution"""
        if not self.current_trace:
            yield None
            return
        
        # Map agent names to display names
        display_name_mapping = {
            'smart': 'Smart Chat',
            'offerpilot': 'OfferPilot', 
            'trustshield': 'TrustShield',
            'dispute': 'Dispute Copilot',
            'collections': 'Collections',
            'contracts': 'Contracts',
            'devcopilot': 'DevCopilot',
            'carecredit': 'WeCare',
            'narrator': 'Narrator',
            'imagegen': 'ImageGen',
            'rag': 'RAG System',
            'search': 'Web Search'
        }
        
        execution = AgentExecution(
            agent_name=agent_name,
            display_name=display_name_mapping.get(agent_name, display_name),
            started_at=int(time.time() * 1000)
        )
        
        self.current_execution = execution
        start_time = time.time()
        
        try:
            yield execution
            execution.status = 'success'
        except Exception as e:
            execution.status = 'error'
            logger.warning(f"Agent {agent_name} execution failed: {e}")
        finally:
            execution.duration_ms = int((time.time() - start_time) * 1000)
            self.current_trace.agent_executions.append(execution)
            self.current_execution = None
    
    @contextmanager
    def trace_tool(self, tool_name: str, input_data: Any = None):
        """Context manager to trace a tool call"""
        if not self.current_execution:
            yield None
            return
        
        tool_call = ToolCall(
            tool_name=tool_name,
            duration_ms=0,
            status='success'
        )
        
        # Create safe input summary (remove sensitive data)
        if input_data:
            tool_call.input_summary = self._create_safe_summary(input_data, max_length=100)
        
        start_time = time.time()
        
        try:
            result = yield tool_call
            tool_call.status = 'success'
            
            # Create safe output summary
            if result:
                tool_call.output_summary = self._create_safe_summary(result, max_length=100)
                
        except Exception as e:
            tool_call.status = 'error'
            logger.warning(f"Tool {tool_name} failed: {e}")
        finally:
            tool_call.duration_ms = int((time.time() - start_time) * 1000)
            self.current_execution.tool_calls.append(tool_call)
    
    def set_agent_confidence(self, confidence: float):
        """Set confidence score for current agent execution"""
        if self.current_execution:
            self.current_execution.confidence = confidence
    
    def add_sources(self, sources: List[str]):
        """Add sources used by current agent"""
        if self.current_execution:
            # Only keep filenames/short identifiers to avoid clutter
            safe_sources = []
            for source in sources:
                if isinstance(source, str):
                    # Extract just filename or short identifier
                    if '/' in source:
                        safe_sources.append(source.split('/')[-1])
                    else:
                        safe_sources.append(source[:50])  # Truncate long sources
                        
            self.current_execution.sources_used.extend(safe_sources)
    
    def set_agent_summaries(self, input_summary: str = None, output_summary: str = None):
        """Set input/output summaries for current agent"""
        if self.current_execution:
            if input_summary:
                self.current_execution.input_summary = self._create_safe_summary(input_summary, max_length=200)
            if output_summary:
                self.current_execution.output_summary = self._create_safe_summary(output_summary, max_length=200)
    
    def finish_trace(self) -> Optional[AgentTrace]:
        """Finish the current trace and return it"""
        if not self.current_trace or not self._start_time:
            return None
        
        self.current_trace.total_duration_ms = int(time.time() * 1000) - self._start_time
        
        # Log trace summary
        logger.info(f"Trace {self.current_trace.execution_id} completed: "
                   f"{len(self.current_trace.agent_executions)} agents, "
                   f"{self.current_trace.total_duration_ms}ms total")
        
        trace = self.current_trace
        self.current_trace = None
        self.current_execution = None
        self._start_time = None
        
        return trace
    
    def _create_safe_summary(self, data: Any, max_length: int = 100) -> str:
        """Create a safe, truncated summary of data for display"""
        if data is None:
            return ""
        
        # Convert to string
        if isinstance(data, dict):
            # For dicts, show key structure without values
            keys = list(data.keys())[:3]  # First 3 keys only
            summary = f"{{keys: {keys}"
            if len(data) > 3:
                summary += f", +{len(data)-3} more"
            summary += "}"
        elif isinstance(data, list):
            summary = f"[{len(data)} items]"
        elif isinstance(data, str):
            # Remove potential PII patterns and truncate
            summary = self._sanitize_string(data)
        else:
            summary = str(type(data).__name__)
        
        # Truncate to max length
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def _sanitize_string(self, text: str) -> str:
        """Remove potential PII from string summaries"""
        import re
        
        # Remove credit card patterns
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)
        # Remove SSN patterns  
        text = re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', '[SSN]', text)
        # Remove email patterns
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        # Remove phone patterns
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        return text.strip()


# Global tracer instance
tracer = AgentTracer()


def get_tracer() -> AgentTracer:
    """Get the global tracer instance"""
    return tracer