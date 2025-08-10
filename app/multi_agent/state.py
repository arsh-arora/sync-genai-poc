"""
LangGraph State Management for Multi-Agent Financial Services System
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from dataclasses import dataclass, field
from enum import Enum
import operator

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"        # Single agent can handle
    MODERATE = "moderate"    # 2-3 agents needed
    COMPLEX = "complex"      # Multiple agents + coordination needed


class ExecutionStrategy(Enum):
    """How agents should be executed"""
    SEQUENTIAL = "sequential"  # One after another
    PARALLEL = "parallel"     # Simultaneously  
    CONDITIONAL = "conditional"  # Based on results


@dataclass
class AgentTask:
    """Individual task for a specific agent"""
    agent_type: str
    query: str
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from an individual agent"""
    agent_type: str
    success: bool
    response: str
    confidence: float
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time: float = 0.0


class FinancialServicesState(TypedDict):
    """
    Comprehensive state for financial services multi-agent system
    Based on LangGraph best practices with financial domain specifics
    """
    # Core conversation state - using LangGraph's add_messages reducer
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Original user query and intent
    original_query: str
    user_intent: Optional[str]
    
    # Agent coordination
    current_agent: Optional[str] 
    supervisor_active: bool
    active_agents: List[str]
    completed_agents: List[str]
    
    # Task planning and execution
    task_complexity: Optional[TaskComplexity]
    execution_strategy: Optional[ExecutionStrategy] 
    agent_tasks: List[AgentTask]
    
    # Results aggregation
    agent_results: Dict[str, AgentResult]
    final_response: Optional[str]
    total_confidence: float
    
    # Financial services specific
    requires_pii_protection: bool
    compliance_check_needed: bool
    risk_level: str  # low, medium, high
    
    # Citation and source tracking
    all_sources: List[str]
    citation_map: Dict[str, List[str]]  # agent -> sources
    
    # User preferences and context
    allow_tavily: bool
    allow_llm_knowledge: bool 
    allow_web_search: bool
    user_context: Dict[str, Any]
    
    # Error handling and recovery
    errors: List[str]
    fallback_used: bool
    retry_count: int


class SupervisorState(TypedDict):
    """
    Simplified state for supervisor agent routing decisions
    Focuses on what supervisor needs to make routing decisions
    """
    messages: Annotated[List[BaseMessage], add_messages]
    original_query: str
    user_intent: Optional[str]
    task_complexity: Optional[TaskComplexity]
    next_agent: Optional[str]
    routing_reason: Optional[str]
    available_agents: List[str]
    user_preferences: Dict[str, bool]


class WorkerState(TypedDict):
    """
    State for individual worker agents
    Contains only what each agent needs to operate
    """
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    context: Dict[str, Any]
    user_preferences: Dict[str, bool]
    previous_results: Dict[str, Any]  # Results from other agents


# State reducer functions for complex state management
def reduce_agent_results(current: Dict[str, AgentResult], new: Dict[str, AgentResult]) -> Dict[str, AgentResult]:
    """Reducer for agent results - merge without overwriting"""
    merged = current.copy()
    merged.update(new)
    return merged


def reduce_sources(current: List[str], new: List[str]) -> List[str]:
    """Reducer for sources - combine and deduplicate"""
    combined = current + new
    return list(dict.fromkeys(combined))  # Preserves order while deduplicating


def reduce_errors(current: List[str], new: List[str]) -> List[str]:
    """Reducer for errors - append new errors"""
    return current + new


# Enhanced state with custom reducers
class EnhancedFinancialState(TypedDict):
    """Enhanced state with custom reducers for better state management"""
    # Core state
    messages: Annotated[List[BaseMessage], add_messages]
    original_query: str
    
    # Enhanced fields with custom reducers
    agent_results: Annotated[Dict[str, AgentResult], reduce_agent_results]
    all_sources: Annotated[List[str], reduce_sources] 
    errors: Annotated[List[str], reduce_errors]
    
    # Simple fields
    current_agent: Optional[str]
    final_response: Optional[str]
    total_confidence: float
    task_complexity: Optional[TaskComplexity]
    
    # User preferences
    allow_tavily: bool
    allow_llm_knowledge: bool
    allow_web_search: bool


# Factory functions for creating initial states
def create_initial_state(
    original_query: str,
    allow_tavily: bool = False,
    allow_llm_knowledge: bool = True, 
    allow_web_search: bool = False,
    user_context: Optional[Dict[str, Any]] = None
) -> FinancialServicesState:
    """Create initial state for multi-agent workflow"""
    return FinancialServicesState(
        messages=[],
        original_query=original_query,
        user_intent=None,
        current_agent=None,
        supervisor_active=True,
        active_agents=[],
        completed_agents=[],
        task_complexity=None,
        execution_strategy=None,
        agent_tasks=[],
        agent_results={},
        final_response=None,
        total_confidence=0.0,
        requires_pii_protection=False,
        compliance_check_needed=True,
        risk_level="low",
        all_sources=[],
        citation_map={},
        allow_tavily=allow_tavily,
        allow_llm_knowledge=allow_llm_knowledge,
        allow_web_search=allow_web_search,
        user_context=user_context or {},
        errors=[],
        fallback_used=False,
        retry_count=0
    )


def create_supervisor_state(base_state: FinancialServicesState) -> SupervisorState:
    """Extract supervisor-relevant state from full state"""
    return SupervisorState(
        messages=base_state["messages"],
        original_query=base_state["original_query"],
        user_intent=base_state["user_intent"],
        task_complexity=base_state["task_complexity"],
        next_agent=None,
        routing_reason=None,
        available_agents=[
            "trustshield", "offerpilot", "dispute", "collections", 
            "contracts", "devcopilot", "carecredit", "narrator", "imagegen"
        ],
        user_preferences={
            "allow_tavily": base_state["allow_tavily"],
            "allow_llm_knowledge": base_state["allow_llm_knowledge"],
            "allow_web_search": base_state["allow_web_search"]
        }
    )


def create_worker_state(
    base_state: FinancialServicesState, 
    agent_query: str,
    agent_context: Optional[Dict[str, Any]] = None
) -> WorkerState:
    """Create worker agent state from base state"""
    return WorkerState(
        messages=base_state["messages"], 
        query=agent_query,
        context=agent_context or {},
        user_preferences={
            "allow_tavily": base_state["allow_tavily"],
            "allow_llm_knowledge": base_state["allow_llm_knowledge"], 
            "allow_web_search": base_state["allow_web_search"]
        },
        previous_results={
            agent: result.response 
            for agent, result in base_state["agent_results"].items()
            if result.success
        }
    )