"""
LangGraph Supervisor Agent for Financial Services Multi-Agent System
Enhanced Smart Chat with multi-agent orchestration capabilities
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from langgraph.constants import Send
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.tool_node import tools_condition

from .state import (
    FinancialServicesState, SupervisorState, WorkerState,
    TaskComplexity, ExecutionStrategy, AgentTask, AgentResult,
    create_initial_state, create_supervisor_state, create_worker_state
)
from ..utils.tracer import get_tracer

logger = logging.getLogger(__name__)


class FinancialServicesSupervisor:
    """
    LangGraph-based supervisor for coordinating financial services agents
    Enhanced Smart Chat with full reasoning access and graceful degradation
    """
    
    def __init__(self, agents: Dict[str, Any], google_api_key: str):
        self.agents = agents
        # Available agents for routing (exclude trustshield as it's middleware only)
        self.available_agents = [name for name in agents.keys() if name != 'trustshield']
        
        # Initialize Gemini model for supervisor reasoning
        self.supervisor_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=google_api_key,
            temperature=0.1,  # Lower temperature for more consistent routing
            max_tokens=4096
        )
        
        # Initialize analysis model for query complexity
        self.analysis_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            api_key=google_api_key,
            temperature=0.0,  # Deterministic for analysis
            max_tokens=2048
        )
        
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph multi-agent workflow with hierarchical orchestration
        Following Tier-0 Master → Tier-1 Persona Supervisors → Tier-2 Agents pattern
        """
        # Create the graph with our financial services state
        graph = StateGraph(FinancialServicesState)
        
        # Tier-0: Master node - persona and platform detection
        graph.add_node("master", self._master_node)
        
        # Tier-1: Persona-specific supervisors
        graph.add_node("consumer_supervisor", self._consumer_supervisor_node)
        graph.add_node("partner_supervisor", self._partner_supervisor_node)
        
        # Tier-2: Analysis, planning, coordination, execution (reused from existing)
        graph.add_node("analyze_query", self._analyze_query_node)
        graph.add_node("plan_execution", self._plan_execution_node)
        graph.add_node("coordinator", self._coordinator_node)  # Routes to individual agents
        graph.add_node("synthesize_results", self._synthesize_results_node)
        graph.add_node("fallback_handler", self._fallback_node)
        
        # Add individual agent nodes (trustshield already excluded from available_agents)
        for agent_name in self.available_agents:
            graph.add_node(f"agent_{agent_name}", self._create_agent_node(agent_name))
        
        # Define the hierarchical workflow edges
        graph.add_edge(START, "master")
        
        # Master routes to persona-specific supervisors
        graph.add_conditional_edges(
            "master",
            self._master_routing,
            {
                "consumer": "consumer_supervisor",
                "partner": "partner_supervisor", 
                "fallback": "fallback_handler"
            }
        )
        
        # Persona supervisors route to analysis
        graph.add_edge("consumer_supervisor", "analyze_query")
        graph.add_edge("partner_supervisor", "analyze_query")
        
        # From analysis, decide single vs multi-agent
        graph.add_conditional_edges(
            "analyze_query",
            self._route_after_analysis,
            {
                "single_agent": "plan_execution",  # Still need planning for agent selection
                "multi_agent": "plan_execution",   # Plan multi-agent execution
                "fallback": "fallback_handler"     # Handle errors
            }
        )
        
        # From planning, route to coordinator 
        graph.add_conditional_edges(
            "plan_execution", 
            self._route_execution_strategy,
            {
                "sequential": "coordinator",  # Coordinator manages sequential execution
                "parallel": "coordinator",    # Coordinator handles parallel execution  
                "conditional": "coordinator", # Coordinator manages conditional logic
                "fallback": "fallback_handler"
            }
        )
        
        # Coordinator routes to specific agents
        graph.add_conditional_edges(
            "coordinator",
            self._coordinator_routing,
            {
                **{f"agent_{agent}": f"agent_{agent}" for agent in self.available_agents},
                "synthesize": "synthesize_results",
                "fallback": "fallback_handler"
            }
        )
        
        # All agent nodes return to coordinator for sequence management
        for agent_name in self.available_agents:
            graph.add_edge(f"agent_{agent_name}", "coordinator")
        
        # Synthesis completes or falls back
        graph.add_conditional_edges(
            "synthesize_results",
            self._check_synthesis_completion,
            {
                "complete": END,
                "fallback": "fallback_handler"
            }
        )
        
        # Fallback always tries to end gracefully
        graph.add_edge("fallback_handler", END)
        
        return graph.compile()
    
    async def process_query(
        self, 
        query: str,
        allow_tavily: bool = False,
        allow_llm_knowledge: bool = True,
        allow_web_search: bool = False,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for processing queries with multi-agent orchestration
        """
        logger.info(f"Processing query with enhanced Smart Chat: {query[:100]}...")
        
        # Start tracing for Agent Theater
        tracer = get_tracer()
        trace_id = tracer.start_trace()
        
        try:
            with tracer.trace_agent('smart', 'Smart Chat') as smart_execution:
                if smart_execution:
                    tracer.set_agent_summaries(
                        input_summary=query[:200],
                        output_summary="Processing multi-agent orchestration"
                    )
                
                # Create initial state
                initial_state = create_initial_state(
                    original_query=query,
                    allow_tavily=allow_tavily,
                    allow_llm_knowledge=allow_llm_knowledge,
                    allow_web_search=allow_web_search,
                    user_context=user_context
                )
                
                # Add user message to state
                initial_state["messages"] = [HumanMessage(content=query)]
                
                # Execute the graph workflow
                logger.info(f"🚀 Executing LangGraph with initial state: {list(initial_state.keys())}")
                try:
                    final_state = await self.graph.ainvoke(initial_state)
                    logger.info(f"✅ Graph execution completed. Final state keys: {list(final_state.keys())}")
                    logger.info(f"🔍 Graph final state final_response: {final_state.get('final_response')}")
                    
                    # Update smart agent execution with final response
                    if smart_execution:
                        response_preview = str(final_state.get('final_response', ''))[:200]
                        tracer.set_agent_summaries(output_summary=response_preview)
                except Exception as graph_error:
                    logger.error(f"Graph execution failed: {graph_error}", exc_info=True)
                    # Create a fallback final_state
                    final_state = {
                        "final_response": "I apologize, but I encountered an error while processing your request.",
                        "total_confidence": 0.0,
                        "all_sources": [],
                        "fallback_used": True,
                        "errors": [str(graph_error)]
                    }
            
            # Format response for API compatibility
            response = self._format_response(final_state)
            
            # Add trace to response
            trace = tracer.finish_trace()
            if trace:
                response['agent_trace'] = {
                    'execution_id': trace.execution_id,
                    'total_duration_ms': trace.total_duration_ms,
                    'agent_executions': [
                        {
                            'agent_name': execution.agent_name,
                            'display_name': execution.display_name,
                            'started_at': execution.started_at,
                            'duration_ms': execution.duration_ms,
                            'status': execution.status,
                            'tool_calls': [
                                {
                                    'tool_name': tc.tool_name,
                                    'duration_ms': tc.duration_ms,
                                    'status': tc.status,
                                    'input_summary': tc.input_summary,
                                    'output_summary': tc.output_summary,
                                    'metadata': tc.metadata
                                }
                                for tc in execution.tool_calls
                            ],
                            'input_summary': execution.input_summary,
                            'output_summary': execution.output_summary,
                            'confidence': execution.confidence,
                            'sources_used': execution.sources_used
                        }
                        for execution in trace.agent_executions
                    ],
                    'routing_decision': trace.routing_decision
                }
            
            return response
            
        except Exception as e:
            logger.error(f"Multi-agent processing failed: {e}", exc_info=True)
            
            # Ensure trace is finished even on error
            tracer.finish_trace()
            
            # Graceful degradation - fallback to single agent
            return await self._emergency_fallback(query, allow_tavily, allow_llm_knowledge, allow_web_search)
    
    async def _master_node(self, state: FinancialServicesState) -> Dict[str, Any]:
        """
        Tier-0 Master node: Detect user persona and platform from query context
        """
        query = state["original_query"]
        user_context = state.get("user_context", {})
        
        # Extract any existing user_type from context
        existing_user_type = user_context.get("user_type", "")
        
        persona_prompt = f"""You are a persona detection system for a financial services platform. Analyze the query and context to determine if this is from a Consumer or Partner.

Consumer indicators:
- Personal financial questions ("my account", "I was charged", "help me pay")
- Individual transactions and disputes
- Personal financing needs
- General customer service requests

Partner indicators:
- Business operations ("our campaign", "promo design", "widget integration")
- Merchant/provider language ("enrollment", "compliance", "co-branded")
- Developer/technical requests ("API", "webhook", "sandbox", "POS")
- Business analytics ("portfolio metrics", "campaign performance")
- Marketing/creative requests ("signage", "promotional copy")

Query: "{query}"
Existing context user_type: "{existing_user_type}"

Respond with JSON:
{{
    "persona": "consumer|partner",
    "confidence": 0.85,
    "reasoning": "brief explanation of decision",
    "detected_indicators": ["indicator1", "indicator2"]
}}"""

        try:
            messages = [
                SystemMessage(content=persona_prompt),
                HumanMessage(content=f"Analyze persona for: {query}")
            ]
            
            response = await self.analysis_model.ainvoke(messages)
            
            # Debug: Print raw response
            logger.info(f"Raw persona detection response: {repr(response.content)}")
            
            extracted_json = self._extract_json(response.content)
            logger.info(f"Extracted JSON: {repr(extracted_json)}")
            
            result = json.loads(extracted_json)
            
            detected_persona = result.get("persona", "consumer")
            confidence = result.get("confidence", 0.5)
            
            # Override with existing user_type if available and confident
            if existing_user_type in ["consumer", "partner"] and confidence < 0.8:
                detected_persona = existing_user_type
                logger.info(f"Using existing user_type: {existing_user_type}")
            
            logger.info(f"Master node detected persona: {detected_persona} (confidence: {confidence})")
            
            return {
                "detected_persona": detected_persona,
                "persona_confidence": confidence,
                "persona_reasoning": result.get("reasoning", ""),
                "user_context": {
                    **user_context,
                    "user_type": detected_persona
                }
            }
            
        except Exception as e:
            logger.error(f"Master persona detection failed: {e}")
            # Default to consumer
            return {
                "detected_persona": "consumer",
                "persona_confidence": 0.3,
                "fallback_used": True,
                "errors": [f"Persona detection failed: {str(e)}"]
            }
    
    def _master_routing(self, state: FinancialServicesState) -> str:
        """Route from master to appropriate persona supervisor"""
        if state.get("fallback_used"):
            return "fallback"
        
        persona = state.get("detected_persona", "consumer")
        logger.info(f"🔀 Master routing to {persona} supervisor")
        return persona
    
    async def _consumer_supervisor_node(self, state: FinancialServicesState) -> Dict[str, Any]:
        """
        Tier-1 Consumer supervisor: Filter to consumer-available agents
        """
        logger.info("Consumer supervisor: filtering agents for consumer persona")
        
        # Consumer-available agents (actual agents from initialization)
        consumer_agents = ['offerpilot', 'dispute', 'collections', 'contracts', 'carecredit', 'narrator']
        
        # Filter agent tasks to only include consumer agents
        original_tasks = state.get("agent_tasks", [])
        filtered_tasks = [task for task in original_tasks if task.agent_type in consumer_agents]
        
        return {
            "available_agents": consumer_agents,
            "agent_tasks": filtered_tasks,
            "persona_context": "consumer"
        }
    
    async def _partner_supervisor_node(self, state: FinancialServicesState) -> Dict[str, Any]:
        """
        Tier-1 Partner supervisor: Filter to partner-available agents  
        """
        logger.info("Partner supervisor: filtering agents for partner persona")
        
        # Partner-available agents (actual agents, no collections/dispute filing)  
        partner_agents = ['devcopilot', 'narrator', 'contracts', 'offerpilot', 'carecredit']
        
        # Filter agent tasks to only include partner agents
        original_tasks = state.get("agent_tasks", [])
        filtered_tasks = [task for task in original_tasks if task.agent_type in partner_agents]
        
        return {
            "available_agents": partner_agents,
            "agent_tasks": filtered_tasks,
            "persona_context": "partner"
        }

    async def _coordinator_node(self, state: FinancialServicesState) -> Dict[str, Any]:
        """
        Coordinator node: Routes to individual agents based on execution plan
        Replaces the removed supervisor node for agent orchestration
        """
        try:
            current_agent = state.get("current_agent")
            completed_agents = state.get("completed_agents", [])
            active_agents = state.get("active_agents", [])
            agent_results = state.get("agent_results", {})
            
            # Check if we're done with all agents
            if current_agent is None and active_agents:
                # Find the next agent to execute
                remaining_agents = [agent for agent in active_agents if agent not in completed_agents]
                if remaining_agents:
                    current_agent = remaining_agents[0]
                    logger.info(f"Coordinator setting current_agent to: {current_agent}")
                    return {"current_agent": current_agent}
            
            # If we've completed all agents, move to synthesis
            if len(completed_agents) >= len(active_agents) and active_agents:
                logger.info("All agents completed, coordinator moving to synthesis")
                return {"current_agent": None}  # Will route to synthesis
            
            # If we have a current agent, prepare for execution
            if current_agent and current_agent not in completed_agents:
                logger.info(f"Coordinator routing to agent: {current_agent}")
                return {"current_agent": current_agent}
            
            # Default case - move to synthesis
            return {"current_agent": None}
            
        except Exception as e:
            logger.error(f"Coordinator failed: {e}")
            return {
                "fallback_used": True,
                "errors": [f"Coordinator failed: {str(e)}"]
            }
    
    def _coordinator_routing(self, state: FinancialServicesState) -> str:
        """Route from coordinator to agents or synthesis"""
        if state.get("fallback_used"):
            return "fallback"
        
        current_agent = state.get("current_agent")
        
        if current_agent and current_agent in self.available_agents:
            logger.info(f"🔀 Coordinator routing to agent_{current_agent}")
            return f"agent_{current_agent}"
        else:
            logger.info("🔀 Coordinator routing to synthesize")
            return "synthesize"

    async def _analyze_query_node(self, state: FinancialServicesState) -> Dict[str, Any]:
        """
        Analyze query to determine if multi-agent approach is needed
        """
        query = state["original_query"]
        
        analysis_prompt = """You are a query analyzer for a financial services AI system. Determine if this query requires multiple specialized agents.

Available Specialized Agents:
- trustshield: Fraud detection, scam analysis, PII protection
- offerpilot: Product search, deals, financing options  
- dispute: Transaction disputes, chargebacks, billing issues
- collections: Payment plans, hardship assistance
- contracts: Contract analysis, legal terms review
- devcopilot: Code generation, API documentation, technical support
- carecredit: Medical/dental expense analysis, healthcare financing
- narrator: Business analytics, portfolio insights, spending analysis

Query needs MULTIPLE agents if:
- Has multiple distinct tasks (e.g., "check if this is fraud AND find financing options")
- Requires different expertise domains (e.g., legal + financial + technical)  
- Has conditional logic (e.g., "if this is legitimate, then help me dispute it")
- Needs validation from multiple perspectives
- Involves complex multi-step workflows

Query needs SINGLE agent if:
- Clear single intent that fits one domain
- Simple, straightforward question
- Can be fully answered by one specialist

Respond with JSON:
{
    "complexity": "simple|moderate|complex",
    "needs_multi_agent": true/false,
    "primary_agents": ["agent1", "agent2"],
    "execution_strategy": "sequential|parallel|conditional",
    "reasoning": "brief explanation",
    "confidence": 0.85
}"""

        try:
            messages = [
                SystemMessage(content=analysis_prompt),
                HumanMessage(content=f"Analyze this query: {query}")
            ]
            
            response = await self.analysis_model.ainvoke(messages)
            result = json.loads(self._extract_json(response.content))
            
            # Update state with analysis results
            strategy_mapping = {
                "sequential": ExecutionStrategy.SEQUENTIAL,
                "parallel": ExecutionStrategy.PARALLEL, 
                "conditional": ExecutionStrategy.CONDITIONAL,
                "direct": ExecutionStrategy.SEQUENTIAL  # Map invalid 'direct' to sequential
            }
            
            strategy_str = result.get("execution_strategy", "sequential")
            execution_strategy = strategy_mapping.get(strategy_str, ExecutionStrategy.SEQUENTIAL)
            
            # Normalize agent names to match our available agents
            agent_normalization = {
                "offeringpilot": "offerpilot",
                "offering": "offerpilot", 
                "trust": "trustshield",
                "care": "carecredit",
                "dev": "devcopilot",
                "image": "narrator"
            }
            
            normalized_agents = []
            for agent in result.get("primary_agents", []):
                normalized_agent = agent_normalization.get(agent, agent)
                if normalized_agent in ["trustshield", "offerpilot", "dispute", "collections", "contracts", "devcopilot", "carecredit", "narrator"]:
                    normalized_agents.append(normalized_agent)
            
            updates = {
                "task_complexity": TaskComplexity(result["complexity"]),
                "user_intent": result.get("reasoning", ""),
                "agent_tasks": [
                    AgentTask(agent_type=agent, query=query, priority=i+1)
                    for i, agent in enumerate(normalized_agents)
                ],
                "execution_strategy": execution_strategy
            }
            
            logger.info(f"Query analysis: {result['complexity']} complexity, multi-agent: {result['needs_multi_agent']}")
            
            return updates
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Default to single agent on analysis failure
            return {
                "task_complexity": TaskComplexity.SIMPLE,
                "fallback_used": True,
                "errors": [f"Analysis failed: {str(e)}"]
            }
    
    def _route_after_analysis(self, state: FinancialServicesState) -> str:
        """Route based on analysis results"""
        if state.get("fallback_used"):
            logger.info("🔀 Routing to fallback due to fallback_used flag")
            return "fallback"
        
        complexity = state.get("task_complexity")
        if complexity == TaskComplexity.SIMPLE:
            logger.info("🔀 Routing to single_agent (simple complexity)")
            return "single_agent"
        else:
            logger.info(f"🔀 Routing to multi_agent ({complexity} complexity)")
            return "multi_agent"
    
    async def _plan_execution_node(self, state: FinancialServicesState) -> Dict[str, Any]:
        """
        Plan execution strategy for multi-agent workflows with parallel support
        """
        try:
            agent_tasks = state.get("agent_tasks", [])
            strategy = state.get("execution_strategy", ExecutionStrategy.SEQUENTIAL)
            available_agents = state.get("available_agents", self.available_agents)
            
            # Filter tasks to only available agents for current persona
            filtered_tasks = [task for task in agent_tasks if task.agent_type in available_agents]
            
            logger.info(f"Planning execution: {strategy.value} strategy with {len(filtered_tasks)} agents")
            
            if not filtered_tasks:
                # No valid agents, fallback to simple routing
                from app.router import route
                route_result = route(state["original_query"])
                agent_name = route_result["agent"]
                
                # Check if routed agent is available for this persona
                if agent_name in available_agents:
                    filtered_tasks = [AgentTask(agent_type=agent_name, query=state["original_query"], priority=1)]
                else:
                    # Default to smart chat
                    filtered_tasks = [AgentTask(agent_type="smart", query=state["original_query"], priority=1)]
            
            active_agents = [task.agent_type for task in filtered_tasks]
            
            # For parallel execution, execute all agents simultaneously
            if strategy == ExecutionStrategy.PARALLEL and len(active_agents) > 1:
                logger.info(f"Executing {len(active_agents)} agents in parallel")
                # Execute all agents in parallel
                agent_results = {}
                import asyncio
                
                async def execute_single_agent(agent_name: str) -> tuple[str, AgentResult]:
                    result = await self._execute_agent(agent_name, state["original_query"], state)
                    return agent_name, result
                
                # Run all agents in parallel
                tasks = [execute_single_agent(agent) for agent in active_agents]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Parallel agent execution error: {result}")
                        continue
                    agent_name, agent_result = result
                    agent_results[agent_name] = agent_result
                
                return {
                    "active_agents": active_agents,
                    "completed_agents": active_agents,
                    "agent_results": agent_results,
                    "execution_strategy": strategy
                }
            else:
                # Sequential execution - set up for traditional flow
                return {
                    "active_agents": active_agents,
                    "current_agent": active_agents[0] if active_agents else None,
                    "execution_strategy": strategy
                }
            
        except Exception as e:
            logger.error(f"Execution planning failed: {e}")
            return {
                "fallback_used": True,
                "errors": [f"Planning failed: {str(e)}"]
            }
    
    def _route_execution_strategy(self, state: FinancialServicesState) -> str:
        """Route based on execution strategy"""
        if state.get("fallback_used"):
            return "fallback"
        
        strategy = state.get("execution_strategy", ExecutionStrategy.SEQUENTIAL)
        
        # All strategies are handled by supervisor for coordination
        # Return the strategy name that matches the conditional edges mapping
        if strategy == ExecutionStrategy.PARALLEL:
            return "parallel"
        elif strategy == ExecutionStrategy.CONDITIONAL:
            return "conditional"
        else:
            return "sequential"
    
    async def _supervisor_node(self, state: FinancialServicesState) -> Dict[str, Any]:
        """
        Central supervisor node - coordinates agent execution
        Implements full reasoning access with smart routing
        """
        try:
            current_agent = state.get("current_agent")
            completed_agents = state.get("completed_agents", [])
            active_agents = state.get("active_agents", [])
            agent_results = state.get("agent_results", {})
            
            # Check if we're done with all agents
            if current_agent is None and active_agents:
                # Find the next agent to execute
                remaining_agents = [agent for agent in active_agents if agent not in completed_agents]
                if remaining_agents:
                    current_agent = remaining_agents[0]
                    logger.info(f"Setting current_agent to: {current_agent}")
                    return {"current_agent": current_agent}
            
            # If we've completed all agents, move to synthesis
            if len(completed_agents) >= len(active_agents) and active_agents:
                logger.info("All agents completed, moving to synthesis")
                return {"current_agent": None}  # Will route to synthesis
            
            # Select next agent or continue current workflow
            if current_agent and current_agent not in completed_agents:
                # Normalize agent name first
                agent_normalization = {
                    "offeringpilot": "offerpilot",
                    "offering": "offerpilot", 
                    "trust": "trustshield",
                    "care": "carecredit",
                    "dev": "devcopilot",
                    "image": "narrator"
                }
                normalized_current_agent = agent_normalization.get(current_agent, current_agent)
                
                logger.info(f"Routing to agent: {normalized_current_agent} (from {current_agent})")
                
                # Prepare context with full reasoning from other agents
                previous_reasoning = {}
                for agent, result in agent_results.items():
                    if result.success:
                        previous_reasoning[agent] = {
                            "response": result.response,
                            "confidence": result.confidence,
                            "sources": result.sources,
                            "reasoning": result.metadata.get("reasoning", "")
                        }
                
                return {
                    "current_agent": normalized_current_agent,  # Use normalized name
                    "user_context": {
                        **state.get("user_context", {}),
                        "previous_agent_reasoning": previous_reasoning,
                        "current_step": f"{len(completed_agents) + 1} of {len(active_agents)}"
                    }
                }
            
            # Default case - move to synthesis or end
            return {"current_agent": None}
            
        except Exception as e:
            logger.error(f"Supervisor routing failed: {e}")
            return {
                "fallback_used": True,
                "errors": [f"Supervisor failed: {str(e)}"]
            }
    
    def _supervisor_routing(self, state: FinancialServicesState) -> str:
        """Determine where supervisor should route next"""
        if state.get("fallback_used"):
            logger.info("🔀 Supervisor routing to fallback")
            return "fallback"
        
        current_agent = state.get("current_agent")
        completed_agents = state.get("completed_agents", [])
        agent_results = state.get("agent_results", {})
        active_agents = state.get("active_agents", [])
        task_complexity = state.get("task_complexity")
        
        logger.info(f"🔍 Supervisor state: current_agent={current_agent}, completed={len(completed_agents)}, active={len(active_agents)}, complexity={task_complexity}")
        
        # For single agent tasks, route directly to the appropriate agent
        if task_complexity == TaskComplexity.SIMPLE and not current_agent and not completed_agents:
            # Need to determine which agent to use for single agent routing
            from app.router import route
            route_result = route(state["original_query"])
            agent_name = route_result["agent"]
            
            # Map legacy agent names to current names
            agent_name_mapping = {
                "offer": "offerpilot",
                "trust": "trustshield", 
                "dev": "devcopilot",
                "care": "carecredit",
                "image": "narrator",
                "dispute": "dispute",
                "collections": "collections",
                "contracts": "contracts",
                "devcopilot": "devcopilot",
                "carecredit": "carecredit",
                "narrator": "narrator",
            }
            
            mapped_agent = agent_name_mapping.get(agent_name, agent_name)
            logger.info(f"🔀 Single agent routing: {agent_name} -> {mapped_agent}")
            
            # Set current agent for routing
            return f"agent_{mapped_agent}"
            
        # For multi-agent tasks, execute the active agents
        if active_agents and not current_agent and not completed_agents:
            # Start with the first active agent
            first_agent = active_agents[0]
            logger.info(f"🔀 Multi-agent routing: starting with {first_agent}")
            return f"agent_{first_agent}"
        
        # Normalize current_agent name to match available agents
        if current_agent:
            agent_normalization = {
                "offeringpilot": "offerpilot",
                "offering": "offerpilot", 
                "trust": "trustshield",
                "care": "carecredit",
                "dev": "devcopilot",
                "image": "narrator"
            }
            normalized_current_agent = agent_normalization.get(current_agent, current_agent)
            
            if normalized_current_agent in self.available_agents:
                logger.info(f"🔀 Supervisor routing to agent_{normalized_current_agent} (normalized from {current_agent})")
                return f"agent_{normalized_current_agent}"
        elif agent_results and len(completed_agents) > 0:
            logger.info(f"🔀 Supervisor routing to synthesize (completed: {len(completed_agents)})")
            return "synthesize"
        else:
            logger.info("🔀 Supervisor routing to end")
            return "end"
    
    def _create_agent_node(self, agent_name: str):
        """Create a node function for a specific agent"""
        async def agent_node(state: FinancialServicesState) -> Dict[str, Any]:
            try:
                query = state["original_query"]
                user_context = state.get("user_context", {})
                
                # For ImageGen, always use original query to avoid redacted trigger words
                imagegen_query = query if agent_name == "imagegen" else query
                
                # Get full reasoning from previous agents
                previous_reasoning = user_context.get("previous_agent_reasoning", {})
                
                # Enhanced query with context if available
                enhanced_query = query
                if previous_reasoning:
                    context_summary = "\n".join([
                        f"{agent.title()}: {result['response'][:200]}..."
                        for agent, result in previous_reasoning.items()
                    ])
                    enhanced_query = f"{query}\n\nContext from other agents:\n{context_summary}"
                
                logger.info(f"Executing agent: {agent_name}")
                
                # Execute the actual agent
                result = await self._execute_agent(agent_name, enhanced_query, state)
                
                # Update state with results
                agent_results = state.get("agent_results", {}).copy()
                agent_results[agent_name] = result
                
                completed_agents = state.get("completed_agents", []).copy()
                if agent_name not in completed_agents:
                    completed_agents.append(agent_name)
                
                # Move to next agent in sequence
                active_agents = state.get("active_agents", [])
                try:
                    current_index = active_agents.index(agent_name)
                    next_agent = active_agents[current_index + 1] if current_index + 1 < len(active_agents) else None
                except (ValueError, IndexError):
                    next_agent = None
                
                # Extract citations from the agent result
                agent_citations = self._extract_citations_from_agent_result(result) if result.success else []
                
                return {
                    "agent_results": agent_results,
                    "completed_agents": completed_agents,
                    "current_agent": next_agent,
                    "all_sources": agent_citations
                }
                
            except Exception as e:
                logger.error(f"Agent {agent_name} execution failed: {e}")
                
                # Record error but continue workflow
                errors = state.get("errors", []).copy()
                errors.append(f"Agent {agent_name} failed: {str(e)}")
                
                return {
                    "errors": errors,
                    "agent_results": {
                        **state.get("agent_results", {}),
                        agent_name: AgentResult(
                            agent_type=agent_name,
                            success=False,
                            response="",
                            confidence=0.0,
                            error=str(e)
                        )
                    }
                }
        
        return agent_node
    
    async def _execute_agent(self, agent_name: str, query: str, state: FinancialServicesState) -> AgentResult:
        """Execute individual agent with error handling"""
        import time
        start_time = time.time()
        
        # Get tracer for this execution
        tracer = get_tracer()
        
        try:
            agent = self.agents.get(agent_name)
            if not agent:
                raise ValueError(f"Agent {agent_name} not found")
            
            # Trace this agent execution
            with tracer.trace_agent(agent_name, agent_name.title()) as agent_execution:
                if agent_execution:
                    tracer.set_agent_summaries(input_summary=query[:200])
                
                # Execute based on agent type with proper method calls
                if agent_name == "trustshield":
                    with tracer.trace_tool('fraud_scan', query):
                        result = agent.scan(query)
                    
                    if agent_execution:
                        tracer.set_agent_summaries(output_summary=result.get("analysis", "")[:200])
                        tracer.add_sources(result.get("citations", []))
                        tracer.set_agent_confidence(result.get("confidence", 0.5))
                    
                    return AgentResult(
                        agent_type=agent_name,
                        success=True,
                        response=result.get("analysis", "No analysis provided"),
                        confidence=result.get("confidence", 0.5),
                        sources=result.get("citations", []),
                        metadata={"threat_detected": result.get("threat_detected", False)},
                        processing_time=time.time() - start_time
                    )
            
                elif agent_name == "offerpilot":
                    with tracer.trace_tool('product_search', query):
                        result = agent.process_query(query, None)  # No budget specified
                    
                    # Use the response text from OfferPilotResponse
                    response_text = result.response
                    
                    # Get UI cards from metadata
                    ui_cards = result.metadata.get("ui_cards", [])
                    prequal = result.metadata.get("prequalification", {})
                    disclosures = result.metadata.get("disclosures", [])
                    
                    # Add disclosure information if available
                    if disclosures:
                        response_text += f"\n\n**Important Terms:**\n"
                        for disclosure in disclosures[:2]:  # Limit to first 2
                            response_text += f"• {disclosure}\n"
                    
                    # Extract citations from metadata
                    sources = []
                    if result.metadata and 'citations' in result.metadata:
                        for citation_dict in result.metadata['citations']:
                            sources.append(citation_dict)
                        logger.info(f"Extracted {len(sources)} citations from OfferPilot result")
                    
                    if agent_execution:
                        tracer.set_agent_summaries(output_summary=response_text[:200])
                        tracer.set_agent_confidence(0.8)
                    
                    return AgentResult(
                        agent_type=agent_name,
                        success=True,
                        response=response_text,
                        confidence=0.8,  # Default confidence for structured responses
                        sources=sources,
                        metadata={"offers_found": len(ui_cards), "prequalification": prequal},
                        processing_time=time.time() - start_time
                    )
            
            
                else:
                    # Generic agent execution for other agents with proper method mapping
                    result = None
                    response_text = ""
                    sources = []
                    confidence = 0.7
                    
                    # Try different method names based on agent
                    if agent_name == "narrator":
                        result = agent.process_question(query)
                        # Handle new NarratorResponse format with insights and UI cards
                        response_text = result.response
                        ui_cards = result.metadata.get("ui_cards", [])
                        sources = result.metadata.get("sources", [])
                        
                        # Add insights summary to response if available
                        if ui_cards:
                            insights_count = len(ui_cards)
                            kpis_analyzed = result.metadata.get("kpis_analyzed", 0)
                            response_text += f"\n\n💡 **Analysis Summary:** {insights_count} key insights from {kpis_analyzed} KPIs"
                        
                    elif agent_name == "dispute":
                        result = agent.process_dispute(query)
                        
                        # Handle new DisputeResponse format
                        response_text = result.response
                        ui_cards = result.metadata.get("ui_cards", [])
                        status = result.metadata.get("status", {})
                        handoffs = result.metadata.get("handoffs", [])
                        
                        # Add status information to response
                        if status:
                            response_text += f"\n\n**Status:** {status.get('stage', 'unknown').replace('_', ' ').title()}"
                            response_text += f" (Likelihood: {status.get('likelihood', 'unknown')})"
                            if not status.get('eligible', True):
                                response_text += f"\n⚠️ {status.get('eligibility_reason', 'Eligibility concerns')}"
                        
                        sources = []  # Enhanced dispute doesn't use traditional sources
                        confidence = 0.8
                            
                    elif agent_name == "collections":
                        if hasattr(agent, 'process_hardship'):
                            result = agent.process_hardship(query)
                        elif hasattr(agent, 'process_query'):
                            result = agent.process_query(query)
                        else:
                            result = {"response": "Collections agent processing not implemented", "confidence": 0.5}
                        
                        if isinstance(result, dict):
                            response_text = result.get('response', str(result))
                            confidence = result.get('confidence', 0.7)
                            sources = result.get('sources', [])
                        else:
                            response_text = str(result)
                            
                    elif agent_name == "contracts":
                        # Use enhanced contract analysis with systematic clause extraction
                        result = agent.analyze_contract(query)
                        
                        # Handle new ContractResponse format
                        response_text = result.response
                        ui_cards = result.metadata.get("ui_cards", [])
                        risk_flags = result.metadata.get("risk_flags", [])
                        handoffs = result.metadata.get("handoffs", [])
                        needs_legal_review = result.metadata.get("needs_legal_review", False)
                        
                        # Add risk flags to response if present
                        if risk_flags:
                            high_risks = [f for f in risk_flags if f.get("severity") == "high"]
                            if high_risks:
                                response_text += f"\n\n⚠️ **High Priority Risks:** {len(high_risks)} identified"
                            if needs_legal_review:
                                response_text += "\n📋 **Legal Review Required**"
                        
                        sources = []  # Enhanced contracts doesn't use traditional sources
                        confidence = 0.8
                            
                    elif agent_name == "devcopilot":
                        try:
                            result = agent.process_request(query)
                            response_text = result.response
                            confidence = 0.8
                            
                            # Extract sources from metadata
                            sources = result.metadata.get('sources', [])
                            
                        except Exception as e:
                            logger.error(f"DevCopilot error: {e}")
                            response_text = f"DevCopilot processing failed: {str(e)}"
                            confidence = 0.3
                            sources = []
                            
                    elif agent_name == "carecredit":
                        if hasattr(agent, 'process_care_query'):
                            result = agent.process_care_query(query)
                        elif hasattr(agent, 'process_query'):
                            result = agent.process_query(query)
                        else:
                            result = {"response": "CareCredit processing not implemented", "confidence": 0.5}
                            
                        if isinstance(result, dict):
                            response_text = result.get('response', str(result))
                            confidence = result.get('confidence', 0.7)
                            sources = result.get('sources', [])
                        else:
                            response_text = str(result)
                            
                    else:
                        # Fallback for unknown agents
                        if hasattr(agent, 'process_query'):
                            result = agent.process_query(query)
                        elif hasattr(agent, 'process'):
                            result = agent.process(query)
                        else:
                            result = {"response": f"Agent {agent_name} processing not implemented", "confidence": 0.3}
                        
                        # Handle different result types
                        if hasattr(result, 'response'):
                            response_text = result.response
                            confidence = getattr(result, 'confidence', 0.7)
                            sources = getattr(result, 'sources', [])
                        elif isinstance(result, dict):
                            response_text = result.get('response', str(result))
                            confidence = result.get('confidence', 0.7)
                            sources = result.get('sources', [])
                        else:
                            response_text = str(result)
                    
                    # Build metadata for agents with enhanced responses
                    metadata = {}
                    if agent_name == "narrator" and hasattr(result, 'metadata'):
                        metadata = result.metadata
                    elif agent_name in ["dispute", "collections", "carecredit"] and hasattr(result, 'metadata'):
                        metadata = result.metadata
                
                return AgentResult(
                    agent_type=agent_name,
                    success=True,
                    response=response_text,
                    confidence=confidence,
                    sources=sources,
                    metadata=metadata,
                    processing_time=time.time() - start_time
                )
                
        except Exception as e:
            logger.error(f"Agent {agent_name} execution error: {e}")
            return AgentResult(
                agent_type=agent_name,
                success=False,
                response="",
                confidence=0.0,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def _synthesize_results_node(self, state: FinancialServicesState) -> Dict[str, Any]:
        """
        Synthesize results from multiple agents into coherent response
        """
        try:
            agent_results = state.get("agent_results", {})
            successful_results = {k: v for k, v in agent_results.items() if v.success}
            
            if not successful_results:
                return {
                    "fallback_used": True,
                    "errors": ["No successful agent results to synthesize"]
                }
            
            original_query = state["original_query"]
            
            # Prepare synthesis prompt with full reasoning
            results_for_synthesis = []
            all_sources = []
            
            for agent_name, result in successful_results.items():
                results_for_synthesis.append({
                    "agent": agent_name,
                    "response": result.response,
                    "confidence": result.confidence,
                    "sources": result.sources,
                    "metadata": result.metadata,
                    "processing_time": result.processing_time
                })
                # Extract citations from agent results
                agent_sources = self._extract_citations_from_agent_result(result)
                all_sources.extend(agent_sources)
            
            synthesis_prompt = f"""You are a response synthesizer for a multi-agent financial services AI system. Create a comprehensive, coherent response from multiple specialized agent outputs.

Original User Query: {original_query}

Agent Results:
{json.dumps(results_for_synthesis, indent=2, default=str)}

IMPORTANT: Format your response using proper markdown:
- Use # for main titles
- Use ## for section headers
- Use ### for subsections  
- Use numbered lists (1. 2. 3.) for steps
- Use bullet points (-) for lists
- Use **bold** for emphasis and key terms
- Use `code` for technical terms
- Use proper paragraph spacing with empty lines
- Use > for important notes or quotes

Guidelines:
1. Synthesize information into one cohesive, natural response
2. Preserve expertise and insights from each agent  
3. Resolve conflicts between agents logically
4. Organize with clear sections using markdown headers
5. Maintain professional financial services tone
6. Include relevant details and maintain source attribution
7. Make it feel like a single intelligent response, not separate answers
8. Ensure proper spacing and formatting throughout

Format your response in clean, well-structured markdown that will render beautifully."""

            messages = [
                SystemMessage(content=synthesis_prompt),
                HumanMessage(content="Please synthesize these agent results into a comprehensive response.")
            ]
            
            response = await self.supervisor_model.ainvoke(messages)
            synthesized_response = response.content
            
            # Calculate weighted confidence
            total_confidence = sum(r.confidence for r in successful_results.values()) / len(successful_results)
            
            logger.info(f"Synthesized response from {len(successful_results)} agents")
            
            logger.info(f"✨ Synthesis successful. Response length: {len(synthesized_response)}")
            
            # Deduplicate sources by source field (since they're now dictionaries)
            seen_sources = set()
            unique_sources = []
            for source in all_sources:
                if isinstance(source, dict):
                    source_key = source.get('source', 'unknown')
                    if source_key not in seen_sources:
                        seen_sources.add(source_key)
                        unique_sources.append(source)
                elif isinstance(source, str):
                    if source not in seen_sources:
                        seen_sources.add(source)
                        unique_sources.append(source)
            
            return {
                "final_response": synthesized_response,
                "total_confidence": total_confidence,
                "all_sources": unique_sources
            }
            
        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            
            # Fallback: concatenate responses
            agent_results = state.get("agent_results", {})
            successful_results = {k: v for k, v in agent_results.items() if v.success}
            
            if successful_results:
                fallback_response = f"# Analysis Results\n\nBased on analysis from {len(successful_results)} specialized agents:\n\n"
                all_sources = []
                
                for agent_name, result in successful_results.items():
                    fallback_response += f"## {agent_name.title()} Analysis\n\n{result.response}\n\n"
                    all_sources.extend(result.sources)
                
                # Deduplicate fallback sources
                seen_sources = set()
                unique_sources = []
                for source in all_sources:
                    if isinstance(source, dict):
                        source_key = source.get('source', 'unknown')
                        if source_key not in seen_sources:
                            seen_sources.add(source_key)
                            unique_sources.append(source)
                    elif isinstance(source, str):
                        if source not in seen_sources:
                            seen_sources.add(source)
                            unique_sources.append(source)
                
                return {
                    "final_response": fallback_response,
                    "total_confidence": sum(r.confidence for r in successful_results.values()) / len(successful_results),
                    "all_sources": unique_sources,
                    "fallback_used": True
                }
            else:
                return {
                    "final_response": "I apologize, but I was unable to process your request successfully.",
                    "total_confidence": 0.0,
                    "fallback_used": True,
                    "errors": [f"Synthesis failed: {str(e)}"]
                }
    
    def _check_synthesis_completion(self, state: FinancialServicesState) -> str:
        """Check if synthesis is complete or needs fallback"""
        if state.get("final_response") and not state.get("fallback_used"):
            return "complete"
        else:
            return "fallback"
    
    async def _fallback_node(self, state: FinancialServicesState) -> Dict[str, Any]:
        """
        Graceful degradation - fallback to single agent routing
        """
        try:
            logger.info("Executing graceful fallback to single agent")
            
            original_query = state["original_query"]
            
            # Use simple routing for fallback
            from app.router import route
            route_result = route(original_query)
            agent_name = route_result["agent"]
            
            # Map legacy agent names to current names
            agent_name_mapping = {
                "offer": "offerpilot",
                "trust": "trustshield", 
                "dev": "devcopilot",
                "care": "carecredit",
                "image": "narrator",
                "dispute": "dispute",        # Already matches
                "collections": "collections", # Already matches
                "contracts": "contracts",    # Already matches
                "devcopilot": "devcopilot",  # Already matches
                "carecredit": "carecredit",  # Already matches
                "narrator": "narrator"       # Already matches
            }
            
            mapped_agent = agent_name_mapping.get(agent_name, agent_name)
            
            # Execute single agent
            result = await self._execute_agent(mapped_agent, original_query, state)
            
            if result.success:
                logger.info(f"🎯 Fallback agent {mapped_agent} succeeded. Response length: {len(result.response)}")
                return {
                    "final_response": result.response,
                    "total_confidence": result.confidence,
                    "all_sources": result.sources,
                    "fallback_used": True,
                    "agent_results": {mapped_agent: result}
                }
            else:
                # Ultimate fallback
                return {
                    "final_response": "I apologize, but I'm unable to process your request at this time. Please try again or contact support.",
                    "total_confidence": 0.1,
                    "fallback_used": True,
                    "errors": [f"Fallback agent {agent_name} also failed: {result.error}"]
                }
                
        except Exception as e:
            logger.error(f"Fallback handler failed: {e}")
            return {
                "final_response": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                "total_confidence": 0.0,
                "fallback_used": True,
                "errors": [f"Complete fallback failure: {str(e)}"]
            }
    
    def _format_response(self, final_state: FinancialServicesState) -> Dict[str, Any]:
        """Format final state into API-compatible response"""
        logger.info(f"🔍 Formatting response from final state keys: {list(final_state.keys())}")
        logger.info(f"🔍 final_response value: {final_state.get('final_response')}")
        logger.info(f"🔍 fallback_used: {final_state.get('fallback_used')}")
        
        agent_results = final_state.get("agent_results", {})
        successful_agents = [name for name, result in agent_results.items() if result.success]
        
        logger.info(f"🔍 Agent results count: {len(agent_results)}, successful: {len(successful_agents)}")
        
        # Determine primary agent for backwards compatibility
        primary_agent = "smart"  # Default to smart for multi-agent responses
        if len(successful_agents) == 1:
            primary_agent = successful_agents[0]
        
        final_response = final_state.get("final_response")
        if not final_response:
            # If no final_response, check if we have any agent results
            agent_results = final_state.get("agent_results", {})
            if agent_results:
                # Use the first successful agent result
                for agent_name, result in agent_results.items():
                    if result.success and result.response:
                        final_response = result.response
                        break
            
            if not final_response:
                final_response = "I apologize, but I was unable to generate a response. Please try again."
        
        # Collect UI cards and citations from all successful agent results
        all_ui_cards = []
        all_citations = []
        
        for agent_name, result in agent_results.items():
            if result.success and hasattr(result, 'metadata') and result.metadata:
                ui_cards = result.metadata.get('ui_cards', [])
                all_ui_cards.extend(ui_cards)
                
        # Extract citations from agent results (for both single and multi-agent cases)
        for agent_name, result in agent_results.items():
            if result.success:
                agent_citations = self._extract_citations_from_agent_result(result)
                all_citations.extend(agent_citations)
        
        # Use extracted citations if we don't have any from synthesis
        final_sources = final_state.get("all_sources", [])
        if not final_sources and all_citations:
            final_sources = all_citations
            logger.info(f"Using extracted citations from agent results: {len(final_sources)} citations")

        response_data = {
            "response": final_response,
            "agent": primary_agent,
            "confidence": final_state.get("total_confidence", 0.0),
            "sources": final_sources,
            "used_tavily": final_state.get("allow_tavily", False),
            "fallback_used": "yes" if final_state.get("fallback_used", False) else None,
            "document_assessment": {
                "multi_agent_used": len(successful_agents) > 1,
                "agents_involved": successful_agents,
                "task_complexity": final_state.get("task_complexity", "simple").value if final_state.get("task_complexity") else "simple",
                "execution_strategy": final_state.get("execution_strategy", "sequential").value if final_state.get("execution_strategy") else "sequential"
            },
            "metadata": {
                "agent_results": {
                    name: {
                        "success": result.success,
                        "confidence": result.confidence,
                        "processing_time": result.processing_time,
                        "error": result.error
                    }
                    for name, result in agent_results.items()
                },
                "ui_cards": all_ui_cards,
                "errors": final_state.get("errors", []),
                "retry_count": final_state.get("retry_count", 0)
            }
        }
        
        logger.info(f"🔍 Final response data: response='{response_data['response']}', agent={response_data['agent']}")
        logger.info(f"🔍 UI cards included: {len(all_ui_cards)} cards")
        if all_ui_cards:
            for i, card in enumerate(all_ui_cards):
                card_type = card.get('type', 'unknown')
                has_image = 'image' in card
                logger.info(f"🔍 UI Card {i+1}: type={card_type}, has_image={has_image}")
        
        return response_data
    
    def _extract_citations_from_agent_result(self, result) -> List[Dict[str, Any]]:
        """Extract and format citations from agent result"""
        citations = []
        
        try:
            # First, check if sources field contains citation dictionaries (main path for OfferPilot)
            if hasattr(result, 'sources') and result.sources:
                logger.info(f"Found {len(result.sources)} items in agent sources field")
                for source in result.sources:
                    if isinstance(source, dict):
                        # Already a properly formatted citation dictionary
                        citations.append({
                            'source': source.get('source', 'Unknown'),
                            'snippet': source.get('snippet', ''),
                            'rule_type': source.get('rule_type', 'Knowledge Base'),
                            'citation_title': source.get('citation_title', source.get('source', 'Unknown')),
                            'relevance_score': source.get('relevance_score', 0.8)
                        })
                    elif isinstance(source, str):
                        # Legacy string format
                        citations.append({
                            'source': source,
                            'snippet': f'Referenced from {source}',
                            'rule_type': 'Knowledge Base',
                            'citation_title': source,
                            'relevance_score': 0.7
                        })

            # Second, check if agent result has citations in metadata (alternative path)
            if not citations and hasattr(result, 'metadata') and result.metadata:
                agent_citations = result.metadata.get('citations', [])
                logger.info(f"Found {len(agent_citations)} citations in agent metadata")
                for citation in agent_citations:
                    if hasattr(citation, 'source') and hasattr(citation, 'snippet'):
                        # Citation object format
                        citations.append({
                            'source': citation.source,
                            'snippet': citation.snippet,
                            'rule_type': getattr(citation, 'rule_type', 'Knowledge Base'),
                            'citation_title': getattr(citation, 'citation_title', citation.source),
                            'relevance_score': getattr(citation, 'relevance_score', 0.8)
                        })
                    elif isinstance(citation, dict):
                        # Dictionary format
                        citations.append({
                            'source': citation.get('source', 'Unknown'),
                            'snippet': citation.get('snippet', ''),
                            'rule_type': citation.get('rule_type', 'Knowledge Base'),
                            'citation_title': citation.get('citation_title', citation.get('source', 'Unknown')),
                            'relevance_score': citation.get('relevance_score', 0.8)
                        })
            
            logger.info(f"Extracted {len(citations)} total citations from agent result")
            
        except Exception as e:
            logger.error(f"Error extracting citations from agent result: {e}")
        
        return citations
    
    async def _emergency_fallback(self, query: str, allow_tavily: bool, allow_llm_knowledge: bool, allow_web_search: bool) -> Dict[str, Any]:
        """Emergency fallback when entire system fails"""
        logger.warning("Executing emergency fallback - using basic routing")
        
        try:
            from app.router import route
            route_result = route(query)
            agent_name = route_result["agent"]
            
            # Map legacy agent names to current names
            agent_name_mapping = {
                "offer": "offerpilot",
                "trust": "trustshield", 
                "dev": "devcopilot",
                "care": "carecredit",
                "image": "narrator",
                "dispute": "dispute",        # Already matches
                "collections": "collections", # Already matches
                "contracts": "contracts",    # Already matches
                "devcopilot": "devcopilot",  # Already matches
                "carecredit": "carecredit",  # Already matches
                "narrator": "narrator"       # Already matches
            }
            
            mapped_agent = agent_name_mapping.get(agent_name, agent_name)
            
            # Create minimal state for emergency execution
            emergency_state = create_initial_state(
                original_query=query,
                allow_tavily=allow_tavily,
                allow_llm_knowledge=allow_llm_knowledge,
                allow_web_search=allow_web_search
            )
            
            result = await self._execute_agent(mapped_agent, query, emergency_state)
            
            return {
                "response": result.response if result.success else "I apologize, but I'm unable to process your request at this time.",
                "agent": mapped_agent,
                "confidence": result.confidence if result.success else 0.1,
                "sources": result.sources if result.success else [],
                "used_tavily": allow_tavily,
                "fallback_used": "emergency",
                "document_assessment": {
                    "emergency_fallback": True,
                    "error": result.error if not result.success else None
                }
            }
            
        except Exception as e:
            logger.error(f"Emergency fallback failed: {e}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                "agent": "system",
                "confidence": 0.0,
                "sources": [],
                "used_tavily": False,
                "fallback_used": "complete_failure",
                "document_assessment": {
                    "complete_system_failure": True,
                    "error": str(e)
                }
            }
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response that may contain markdown formatting"""
        import re
        
        if not text or not text.strip():
            return "{}"
        
        # Try to find JSON in code blocks (more flexible pattern)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
        if json_match:
            return json_match.group(1).strip()
        
        # Try to find JSON directly (non-greedy)
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()
        
        # Try to extract everything between first { and last }
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            return text[first_brace:last_brace+1].strip()
        
        # Fallback - return empty JSON
        logger.warning(f"Could not extract JSON from: {repr(text[:200])}")
        return '{"persona": "consumer", "confidence": 0.3, "reasoning": "Failed to parse response"}'