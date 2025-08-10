"""
Multi-Agent Orchestration System
Handles complex queries requiring multiple specialized agents
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from app.llm.gemini import chat
from app.router import route, AGENT_TYPES

logger = logging.getLogger(__name__)

class TaskType(Enum):
    ANALYSIS = "analysis"
    SEARCH = "search"
    GENERATION = "generation"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"

class ExecutionStrategy(Enum):
    PARALLEL = "parallel"      # All agents run simultaneously
    SEQUENTIAL = "sequential"  # Agents run in order
    CONDITIONAL = "conditional" # Next agent depends on previous results

@dataclass
class SubTask:
    """Individual task for a specific agent"""
    id: str
    agent_type: str
    task_type: TaskType
    query: str
    context: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on
    priority: int = 1  # Higher number = higher priority

@dataclass
class AgentResult:
    """Result from an individual agent"""
    task_id: str
    agent_type: str
    success: bool
    response: str
    confidence: float
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class OrchestrationResult:
    """Final orchestrated result"""
    success: bool
    synthesized_response: str
    agent_results: List[AgentResult]
    execution_trace: List[Dict[str, Any]]
    total_confidence: float
    sources: List[str] = field(default_factory=list)

class MultiAgentOrchestrator:
    """
    Orchestrates multiple agents to handle complex queries
    """
    
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.max_concurrent = 3  # Limit concurrent agent calls
        
    async def process_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> OrchestrationResult:
        """
        Main entry point for multi-agent processing
        """
        logger.info(f"Starting multi-agent orchestration for query: {query[:100]}...")
        
        try:
            # Step 1: Analyze if multi-agent approach is needed
            needs_multi_agent = await self._should_use_multi_agent(query)
            
            if not needs_multi_agent:
                logger.info("Query can be handled by single agent, routing normally")
                return await self._single_agent_fallback(query, context)
            
            # Step 2: Decompose query into sub-tasks
            sub_tasks = await self._decompose_query(query, context or {})
            
            # Step 3: Determine execution strategy
            strategy = self._determine_execution_strategy(sub_tasks)
            
            # Step 4: Execute agents based on strategy
            agent_results = await self._execute_agents(sub_tasks, strategy)
            
            # Step 5: Synthesize final response
            final_result = await self._synthesize_results(query, agent_results, sub_tasks)
            
            logger.info(f"Multi-agent orchestration completed successfully with {len(agent_results)} agents")
            return final_result
            
        except Exception as e:
            logger.error(f"Multi-agent orchestration failed: {e}")
            return OrchestrationResult(
                success=False,
                synthesized_response=f"I encountered an error processing your complex query: {str(e)}",
                agent_results=[],
                execution_trace=[{"error": str(e)}],
                total_confidence=0.0
            )
    
    async def _should_use_multi_agent(self, query: str) -> bool:
        """
        Determine if query requires multiple agents using LLM analysis
        """
        system_prompt = """You are a query analyzer for a multi-agent system. Determine if this query requires multiple specialized agents.

A query needs multiple agents if it:
- Has multiple distinct tasks (e.g., "analyze this contract AND check if merchant is trustworthy")
- Requires different types of expertise (e.g., technical + financial + legal)
- Has conditional logic (e.g., "if this is fraud, then dispute it")
- Needs validation from multiple perspectives

Single agent queries:
- Simple questions with one clear intent
- Questions that fit clearly into one domain

Respond with only JSON:
{"needs_multi_agent": true/false, "reasoning": "brief explanation", "estimated_agents": 2}"""

        try:
            response = chat([{"role": "user", "content": f"Analyze this query: {query}"}], system=system_prompt)
            result = json.loads(response.strip().replace('```json', '').replace('```', ''))
            
            return result.get("needs_multi_agent", False)
            
        except Exception as e:
            logger.error(f"Error in multi-agent analysis: {e}")
            return False  # Default to single agent on error
    
    async def _decompose_query(self, query: str, context: Dict[str, Any]) -> List[SubTask]:
        """
        Break down complex query into sub-tasks for different agents
        """
        system_prompt = f"""You are a query decomposition expert. Break down this complex query into specific sub-tasks for our specialized agents.

Available Agents:
{', '.join(AGENT_TYPES)}

Agent Capabilities:
- offer: Product search, deals, financing options
- trust: Fraud detection, scam analysis, PII protection
- dispute: Transaction disputes, chargebacks, billing issues
- collections: Payment plans, hardship assistance
- contracts: Contract analysis, terms review
- devcopilot: Code generation, API documentation
- carecredit: Medical/dental expense analysis
- narrator: Business analytics, portfolio insights
- imagegen: Image generation from text

Create specific, actionable sub-tasks. Each task should:
1. Be assignable to ONE agent
2. Have a clear, specific query
3. Include context/dependencies if needed

Respond with JSON array:
[
  {{
    "id": "task_1",
    "agent_type": "agent_name",
    "task_type": "analysis|search|generation|validation",
    "query": "specific query for this agent",
    "context": {{}},
    "dependencies": [],
    "priority": 1
  }}
]"""

        try:
            user_message = f"Decompose this query: {query}\nContext: {json.dumps(context)}"
            response = chat([{"role": "user", "content": user_message}], system=system_prompt)
            
            # Parse JSON response
            response_text = response.strip().replace('```json', '').replace('```', '').strip()
            tasks_data = json.loads(response_text)
            
            # Convert to SubTask objects
            sub_tasks = []
            for task_data in tasks_data:
                sub_task = SubTask(
                    id=task_data["id"],
                    agent_type=task_data["agent_type"],
                    task_type=TaskType(task_data["task_type"]),
                    query=task_data["query"],
                    context=task_data.get("context", {}),
                    dependencies=task_data.get("dependencies", []),
                    priority=task_data.get("priority", 1)
                )
                sub_tasks.append(sub_task)
            
            logger.info(f"Decomposed query into {len(sub_tasks)} sub-tasks")
            return sub_tasks
            
        except Exception as e:
            logger.error(f"Error in query decomposition: {e}")
            # Fallback: create single task
            return [SubTask(
                id="fallback_task",
                agent_type="trust",  # Default to trust agent
                task_type=TaskType.ANALYSIS,
                query=query,
                context=context
            )]
    
    def _determine_execution_strategy(self, sub_tasks: List[SubTask]) -> ExecutionStrategy:
        """
        Determine how to execute the sub-tasks (parallel, sequential, conditional)
        """
        # Check for dependencies
        has_dependencies = any(task.dependencies for task in sub_tasks)
        
        if has_dependencies:
            return ExecutionStrategy.SEQUENTIAL
        elif len(sub_tasks) <= self.max_concurrent:
            return ExecutionStrategy.PARALLEL
        else:
            return ExecutionStrategy.SEQUENTIAL
    
    async def _execute_agents(
        self, 
        sub_tasks: List[SubTask], 
        strategy: ExecutionStrategy
    ) -> List[AgentResult]:
        """
        Execute agents based on the determined strategy
        """
        results = []
        
        if strategy == ExecutionStrategy.PARALLEL:
            # Execute all tasks simultaneously
            tasks = [self._execute_single_agent(task, {}) for task in sub_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        elif strategy == ExecutionStrategy.SEQUENTIAL:
            # Execute tasks in order, building context
            context = {}
            for task in sorted(sub_tasks, key=lambda t: t.priority, reverse=True):
                result = await self._execute_single_agent(task, context)
                results.append(result)
                
                # Add result to context for next tasks
                if result.success:
                    context[f"{task.agent_type}_result"] = {
                        "response": result.response,
                        "confidence": result.confidence,
                        "sources": result.sources
                    }
        
        # Filter out exceptions and convert to AgentResult objects
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {sub_tasks[i].id} failed with exception: {result}")
                valid_results.append(AgentResult(
                    task_id=sub_tasks[i].id,
                    agent_type=sub_tasks[i].agent_type,
                    success=False,
                    response="",
                    confidence=0.0,
                    error=str(result)
                ))
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _execute_single_agent(self, task: SubTask, context: Dict[str, Any]) -> AgentResult:
        """
        Execute a single agent task
        """
        try:
            agent = self.agents.get(task.agent_type)
            if not agent:
                return AgentResult(
                    task_id=task.id,
                    agent_type=task.agent_type,
                    success=False,
                    response="",
                    confidence=0.0,
                    error=f"Agent {task.agent_type} not found"
                )
            
            # Prepare query with context
            enhanced_query = task.query
            if context:
                enhanced_query = f"{task.query}\n\nContext from previous agents: {json.dumps(context)}"
            
            # Execute agent based on type
            if task.agent_type == "trust":
                result = agent.scan(enhanced_query)
                return AgentResult(
                    task_id=task.id,
                    agent_type=task.agent_type,
                    success=True,
                    response=result.get("analysis", ""),
                    confidence=result.get("confidence", 0.5),
                    sources=result.get("citations", [])
                )
            
            elif task.agent_type == "offerpilot":
                result = agent.process_query(enhanced_query, None)
                return AgentResult(
                    task_id=task.id,
                    agent_type=task.agent_type,
                    success=True,
                    response=result.response,
                    confidence=result.confidence,
                    sources=result.sources
                )
            
            # Add more agent types as needed...
            else:
                # Generic agent execution
                result = agent.process_query(enhanced_query)
                return AgentResult(
                    task_id=task.id,
                    agent_type=task.agent_type,
                    success=True,
                    response=getattr(result, 'response', str(result)),
                    confidence=getattr(result, 'confidence', 0.5),
                    sources=getattr(result, 'sources', [])
                )
                
        except Exception as e:
            logger.error(f"Error executing agent {task.agent_type}: {e}")
            return AgentResult(
                task_id=task.id,
                agent_type=task.agent_type,
                success=False,
                response="",
                confidence=0.0,
                error=str(e)
            )
    
    async def _synthesize_results(
        self, 
        original_query: str,
        agent_results: List[AgentResult], 
        sub_tasks: List[SubTask]
    ) -> OrchestrationResult:
        """
        Synthesize multiple agent results into a coherent final response
        """
        successful_results = [r for r in agent_results if r.success]
        
        if not successful_results:
            return OrchestrationResult(
                success=False,
                synthesized_response="I was unable to process your request with any of the specialized agents.",
                agent_results=agent_results,
                execution_trace=[{"error": "No successful agent results"}],
                total_confidence=0.0
            )
        
        # Prepare synthesis prompt
        results_summary = []
        for result in successful_results:
            results_summary.append({
                "agent": result.agent_type,
                "response": result.response,
                "confidence": result.confidence,
                "sources": result.sources
            })
        
        system_prompt = """You are a response synthesizer for a multi-agent AI system. Your job is to create a coherent, comprehensive response from multiple specialized agent outputs.

Guidelines:
1. Synthesize information from all agents into one cohesive response
2. Maintain the expertise and insights from each agent
3. Resolve any conflicts between agent responses logically
4. Organize the response clearly with proper sections if needed
5. Preserve important details and citations
6. Make it feel like a single, intelligent response, not multiple separate answers

Format your response naturally, addressing the original user query completely."""

        try:
            synthesis_query = f"""Original user query: {original_query}

Agent Results:
{json.dumps(results_summary, indent=2)}

Please synthesize these results into a comprehensive response that fully addresses the user's original query."""

            synthesized_response = chat([{"role": "user", "content": synthesis_query}], system=system_prompt)
            
            # Combine all sources
            all_sources = []
            for result in successful_results:
                all_sources.extend(result.sources)
            
            # Calculate weighted average confidence
            total_confidence = sum(r.confidence for r in successful_results) / len(successful_results)
            
            return OrchestrationResult(
                success=True,
                synthesized_response=synthesized_response,
                agent_results=agent_results,
                execution_trace=[{
                    "agents_used": [r.agent_type for r in successful_results],
                    "synthesis_completed": True
                }],
                total_confidence=total_confidence,
                sources=list(set(all_sources))  # Remove duplicates
            )
            
        except Exception as e:
            logger.error(f"Error in response synthesis: {e}")
            # Fallback: concatenate responses
            fallback_response = f"Based on analysis from {len(successful_results)} specialized agents:\n\n"
            for result in successful_results:
                fallback_response += f"**{result.agent_type.title()} Agent:** {result.response}\n\n"
            
            return OrchestrationResult(
                success=True,
                synthesized_response=fallback_response,
                agent_results=agent_results,
                execution_trace=[{"synthesis_fallback": True}],
                total_confidence=sum(r.confidence for r in successful_results) / len(successful_results),
                sources=list(set(all_sources))
            )
    
    async def _single_agent_fallback(self, query: str, context: Optional[Dict[str, Any]]) -> OrchestrationResult:
        """
        Fallback to single agent routing when multi-agent isn't needed
        """
        from app.router import route
        
        route_result = route(query)
        agent_type = route_result["agent"]
        
        # Execute single agent
        task = SubTask(
            id="single_task",
            agent_type=agent_type,
            task_type=TaskType.ANALYSIS,
            query=query,
            context=context or {}
        )
        
        result = await self._execute_single_agent(task, {})
        
        return OrchestrationResult(
            success=result.success,
            synthesized_response=result.response,
            agent_results=[result],
            execution_trace=[{"single_agent": agent_type, "confidence": route_result["confidence"]}],
            total_confidence=result.confidence,
            sources=result.sources
        )