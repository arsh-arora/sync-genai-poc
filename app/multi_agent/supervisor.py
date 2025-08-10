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

logger = logging.getLogger(__name__)


class FinancialServicesSupervisor:
    """
    LangGraph-based supervisor for coordinating financial services agents
    Enhanced Smart Chat with full reasoning access and graceful degradation
    """
    
    def __init__(self, agents: Dict[str, Any], google_api_key: str):
        self.agents = agents
        self.available_agents = list(agents.keys())
        
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
        Build the LangGraph multi-agent workflow
        Following supervisor pattern with enhanced routing
        """
        # Create the graph with our financial services state
        graph = StateGraph(FinancialServicesState)
        
        # Add supervisor node - central coordinator
        graph.add_node("supervisor", self._supervisor_node)
        
        # Add analysis node - determine if multi-agent needed
        graph.add_node("analyze_query", self._analyze_query_node)
        
        # Add planning node - create execution plan
        graph.add_node("plan_execution", self._plan_execution_node)
        
        # Add synthesis node - combine agent results
        graph.add_node("synthesize_results", self._synthesize_results_node)
        
        # Add fallback node - graceful degradation
        graph.add_node("fallback_handler", self._fallback_node)
        
        # Add individual agent nodes
        for agent_name in self.available_agents:
            graph.add_node(f"agent_{agent_name}", self._create_agent_node(agent_name))
        
        # Define the workflow edges
        graph.add_edge(START, "analyze_query")
        
        # From analysis, decide single vs multi-agent
        graph.add_conditional_edges(
            "analyze_query",
            self._route_after_analysis,
            {
                "single_agent": "supervisor",  # Route directly to agent
                "multi_agent": "plan_execution",  # Plan multi-agent execution
                "fallback": "fallback_handler"  # Handle errors
            }
        )
        
        # From planning, execute agents based on strategy
        graph.add_conditional_edges(
            "plan_execution", 
            self._route_execution_strategy,
            {
                "sequential": "supervisor",  # Supervisor manages sequence
                "parallel": "supervisor",    # Supervisor handles parallel coordination
                "conditional": "supervisor", # Conditional logic via supervisor
                "fallback": "fallback_handler"
            }
        )
        
        # Supervisor routes to specific agents or synthesis
        graph.add_conditional_edges(
            "supervisor",
            self._supervisor_routing,
            {
                **{f"agent_{agent}": f"agent_{agent}" for agent in self.available_agents},
                "synthesize": "synthesize_results",
                "fallback": "fallback_handler",
                "end": END
            }
        )
        
        # All agent nodes return to supervisor for coordination
        for agent_name in self.available_agents:
            graph.add_edge(f"agent_{agent_name}", "supervisor")
        
        # Synthesis can end or continue to supervisor
        graph.add_conditional_edges(
            "synthesize_results",
            self._check_synthesis_completion,
            {
                "complete": END,
                "continue": "supervisor", 
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
        
        try:
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
            final_state = await self.graph.ainvoke(initial_state)
            logger.info(f"✅ Graph execution completed. Final state keys: {list(final_state.keys())}")
            logger.info(f"🔍 Graph final state final_response: {final_state.get('final_response')}")
            
            # Format response for API compatibility
            return self._format_response(final_state)
            
        except Exception as e:
            logger.error(f"Multi-agent processing failed: {e}", exc_info=True)
            
            # Graceful degradation - fallback to single agent
            return await self._emergency_fallback(query, allow_tavily, allow_llm_knowledge, allow_web_search)
    
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
- imagegen: Image generation from text descriptions

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
                "image": "imagegen"
            }
            
            normalized_agents = []
            for agent in result.get("primary_agents", []):
                normalized_agent = agent_normalization.get(agent, agent)
                if normalized_agent in ["trustshield", "offerpilot", "dispute", "collections", "contracts", "devcopilot", "carecredit", "narrator", "imagegen"]:
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
        Plan execution strategy for multi-agent workflows
        """
        try:
            agent_tasks = state.get("agent_tasks", [])
            strategy = state.get("execution_strategy", ExecutionStrategy.SEQUENTIAL)
            
            logger.info(f"Planning execution: {strategy.value} strategy with {len(agent_tasks)} agents")
            
            # Set active agents based on tasks
            active_agents = [task.agent_type for task in agent_tasks]
            
            return {
                "active_agents": active_agents,
                "current_agent": active_agents[0] if active_agents else None,
                "supervisor_active": True
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
                    "image": "imagegen"
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
                "image": "imagegen",
                "dispute": "dispute",
                "collections": "collections",
                "contracts": "contracts",
                "devcopilot": "devcopilot",
                "carecredit": "carecredit",
                "narrator": "narrator",
                "imagegen": "imagegen"
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
                "image": "imagegen"
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
                
                return {
                    "agent_results": agent_results,
                    "completed_agents": completed_agents,
                    "current_agent": next_agent,
                    "all_sources": result.sources if result.success else []
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
        
        try:
            agent = self.agents.get(agent_name)
            if not agent:
                raise ValueError(f"Agent {agent_name} not found")
            
            # Execute based on agent type with proper method calls
            if agent_name == "trustshield":
                result = agent.scan(query)
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
                result = agent.process_query(query, None)  # No budget specified
                
                # Convert OfferPilotResponse to readable text
                response_text = f"Found {len(result.items)} products matching your search:\n\n"
                
                for item in result.items:
                    response_text += f"**{item.title}** - ${item.price:,.2f} from {item.merchant}\n"
                    if item.offers:
                        response_text += f"Financing: {item.offers[0].months} months at {item.offers[0].apr}% APR\n"
                    response_text += "\n"
                
                if result.prequal.eligible:
                    response_text += f"✅ Pre-qualification: {result.prequal.reason}\n"
                else:
                    response_text += f"❌ Pre-qualification: {result.prequal.reason}\n"
                
                sources = [citation.source for citation in result.citations]
                
                return AgentResult(
                    agent_type=agent_name,
                    success=True,
                    response=response_text,
                    confidence=0.8,  # Default confidence for structured responses
                    sources=sources,
                    metadata={"offers_found": len(result.items)},
                    processing_time=time.time() - start_time
                )
            
            elif agent_name == "imagegen":
                # For imagegen, create a proper request object
                from app.agents.imagegen import ImageGenRequest
                request = ImageGenRequest(prompt=query, include_text=True, style_hints=[])
                result = agent.process_request(request)
                
                return AgentResult(
                    agent_type=agent_name,
                    success=True,
                    response=result.response,
                    confidence=result.confidence,
                    sources=result.sources,
                    metadata={
                        "image_generated": bool(result.image_data),
                        "image_format": result.image_format
                    },
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
                    # Convert NarratorResponse to readable text
                    response_text = f"Analysis Results:\n\n"
                    for finding in result.findings:
                        response_text += f"**{finding.title}**\n"
                        response_text += f"Evidence: {finding.evidence}\n\n"
                    
                    if result.actions:
                        response_text += "Recommended Actions:\n"
                        for action in result.actions:
                            response_text += f"• {action.hypothesis} (Owner: {action.owner})\n"
                    
                    sources = [citation.source for citation in result.citations]
                    
                elif agent_name == "dispute":
                    result = agent.process_dispute(query)
                    if isinstance(result, dict):
                        response_text = result.get('analysis', str(result))
                        sources = result.get('sources', [])
                        confidence = result.get('confidence', 0.7)
                    else:
                        response_text = str(result)
                        
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
                    if hasattr(agent, 'analyze_contract'):
                        result = agent.analyze_contract(query)
                    elif hasattr(agent, 'process_query'):
                        result = agent.process_query(query)
                    else:
                        result = {"response": "Contract analysis not implemented", "confidence": 0.5}
                        
                    if isinstance(result, dict):
                        response_text = result.get('response', str(result))
                        confidence = result.get('confidence', 0.7)
                        sources = result.get('sources', [])
                    else:
                        response_text = str(result)
                        
                elif agent_name == "devcopilot":
                    if hasattr(agent, 'process_technical_query'):
                        result = agent.process_technical_query(query)
                    elif hasattr(agent, 'process_query'):
                        result = agent.process_query(query)
                    else:
                        result = {"response": "DevCopilot processing not implemented", "confidence": 0.5}
                        
                    if isinstance(result, dict):
                        response_text = result.get('response', str(result))
                        confidence = result.get('confidence', 0.7)
                        sources = result.get('sources', [])
                    else:
                        response_text = str(result)
                        
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
                
                return AgentResult(
                    agent_type=agent_name,
                    success=True,
                    response=response_text,
                    confidence=confidence,
                    sources=sources,
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
                all_sources.extend(result.sources)
            
            synthesis_prompt = f"""You are a response synthesizer for a multi-agent financial services AI system. Create a comprehensive, coherent response from multiple specialized agent outputs.

Original User Query: {original_query}

Agent Results:
{json.dumps(results_for_synthesis, indent=2, default=str)}

Guidelines:
1. Synthesize information into one cohesive, natural response
2. Preserve expertise and insights from each agent  
3. Resolve conflicts between agents logically
4. Organize with clear sections if needed
5. Maintain professional financial services tone
6. Include relevant details and maintain source attribution
7. Make it feel like a single intelligent response, not separate answers

Format your response naturally, addressing the user's query completely."""

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
            
            return {
                "final_response": synthesized_response,
                "total_confidence": total_confidence,
                "all_sources": list(set(all_sources))  # Deduplicate sources
            }
            
        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            
            # Fallback: concatenate responses
            agent_results = state.get("agent_results", {})
            successful_results = {k: v for k, v in agent_results.items() if v.success}
            
            if successful_results:
                fallback_response = f"Based on analysis from {len(successful_results)} specialized agents:\n\n"
                all_sources = []
                
                for agent_name, result in successful_results.items():
                    fallback_response += f"**{agent_name.title()} Analysis:**\n{result.response}\n\n"
                    all_sources.extend(result.sources)
                
                return {
                    "final_response": fallback_response,
                    "total_confidence": sum(r.confidence for r in successful_results.values()) / len(successful_results),
                    "all_sources": list(set(all_sources)),
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
        """Check if synthesis is complete or needs more work"""
        if state.get("final_response") and not state.get("fallback_used"):
            return "complete"
        elif state.get("fallback_used") or state.get("errors"):
            return "fallback"
        else:
            return "continue"
    
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
                "image": "imagegen",
                "dispute": "dispute",        # Already matches
                "collections": "collections", # Already matches
                "contracts": "contracts",    # Already matches
                "devcopilot": "devcopilot",  # Already matches
                "carecredit": "carecredit",  # Already matches
                "narrator": "narrator",      # Already matches
                "imagegen": "imagegen"       # Already matches
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
        
        response_data = {
            "response": final_response,
            "agent": primary_agent,
            "confidence": final_state.get("total_confidence", 0.0),
            "sources": final_state.get("all_sources", []),
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
                "errors": final_state.get("errors", []),
                "retry_count": final_state.get("retry_count", 0)
            }
        }
        
        logger.info(f"🔍 Final response data: response='{response_data['response']}', agent={response_data['agent']}")
        return response_data
    
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
                "image": "imagegen",
                "dispute": "dispute",        # Already matches
                "collections": "collections", # Already matches
                "contracts": "contracts",    # Already matches
                "devcopilot": "devcopilot",  # Already matches
                "carecredit": "carecredit",  # Already matches
                "narrator": "narrator",      # Already matches
                "imagegen": "imagegen"       # Already matches
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
        
        # Try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Try to find JSON directly
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # Fallback
        return text