"""
Chat Agent for handling queries with RAG and optional Tavily fallback
"""

import os
import logging
from typing import Dict, Any, Optional

from tavily import TavilyClient

logger = logging.getLogger(__name__)

class ChatAgent:
    """
    Chat agent that processes queries using RAG pipeline with optional Tavily fallback
    """
    
    def __init__(self, rag_pipeline):
        """
        Initialize the chat agent.
        
        Args:
            rag_pipeline: The RAG pipeline instance
        """
        self.rag_pipeline = rag_pipeline
        self.tavily_client = None
        
        # Initialize Tavily client if enabled and API key is available
        if os.getenv("ALLOW_TAVILY", "false").lower() == "true":
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if tavily_api_key:
                try:
                    self.tavily_client = TavilyClient(api_key=tavily_api_key)
                    logger.info("Tavily client initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize Tavily client: {e}")
            else:
                logger.warning("ALLOW_TAVILY is true but TAVILY_API_KEY not found")
    
    async def process_query(
        self, 
        query: str, 
        use_tavily: bool = False,
        confidence_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Process a query using RAG pipeline and optionally Tavily fallback.
        
        Args:
            query: The user's query
            use_tavily: Whether to use Tavily fallback if RAG results are poor
            confidence_threshold: Minimum confidence score for RAG results
            
        Returns:
            Dictionary containing response, sources, and metadata
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # First, try RAG pipeline
            rag_result = await self.rag_pipeline.query(query)
            
            # Check if we should use Tavily fallback
            should_use_tavily = (
                use_tavily and 
                self.tavily_client and 
                self._should_fallback_to_tavily(rag_result, confidence_threshold)
            )
            
            if should_use_tavily:
                logger.info("RAG results below threshold, falling back to Tavily")
                tavily_result = await self._query_tavily(query)
                
                # Combine RAG and Tavily results
                return self._combine_results(rag_result, tavily_result)
            else:
                return {
                    **rag_result,
                    "used_tavily": False
                }
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"I apologize, but I encountered an error while processing your query: {str(e)}",
                "sources": [],
                "used_tavily": False,
                "error": str(e)
            }
    
    def _should_fallback_to_tavily(self, rag_result: Dict[str, Any], threshold: float) -> bool:
        """
        Determine if we should fallback to Tavily based on RAG result quality.
        
        Args:
            rag_result: Result from RAG pipeline
            threshold: Confidence threshold
            
        Returns:
            True if should fallback to Tavily
        """
        # Check if we have few or no retrieved documents
        if rag_result.get("retrieved_documents", 0) == 0:
            return True
        
        # Check if response indicates lack of information
        response = rag_result.get("response", "").lower()
        fallback_indicators = [
            "i don't have information",
            "not available in the documents",
            "cannot find",
            "no information",
            "not mentioned in the documents"
        ]
        
        return any(indicator in response for indicator in fallback_indicators)
    
    async def _query_tavily(self, query: str) -> Dict[str, Any]:
        """
        Query Tavily for additional information.
        
        Args:
            query: The search query
            
        Returns:
            Dictionary with Tavily search results
        """
        try:
            logger.info(f"Querying Tavily for: {query}")
            
            # Search with Tavily
            search_result = self.tavily_client.search(
                query=query,
                search_depth="basic",
                max_results=3
            )
            
            # Extract relevant information
            tavily_sources = []
            tavily_content = []
            
            for result in search_result.get("results", []):
                title = result.get("title", "")
                url = result.get("url", "")
                content = result.get("content", "")
                
                if content:
                    tavily_sources.append(f"{title} - {url}")
                    tavily_content.append(content)
            
            # Create a summary response
            if tavily_content:
                combined_content = "\n\n".join(tavily_content[:2])  # Use top 2 results
                tavily_response = (
                    f"Based on web search results:\n\n{combined_content}\n\n"
                    f"Please note: This information comes from web search and may need verification."
                )
            else:
                tavily_response = "No additional information found through web search."
            
            return {
                "response": tavily_response,
                "sources": tavily_sources,
                "search_results_count": len(search_result.get("results", []))
            }
            
        except Exception as e:
            logger.error(f"Error querying Tavily: {e}")
            return {
                "response": "Unable to retrieve additional information from web search.",
                "sources": [],
                "search_results_count": 0,
                "error": str(e)
            }
    
    def _combine_results(self, rag_result: Dict[str, Any], tavily_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine RAG and Tavily results into a single response.
        
        Args:
            rag_result: Result from RAG pipeline
            tavily_result: Result from Tavily search
            
        Returns:
            Combined result dictionary
        """
        # Combine responses
        rag_response = rag_result.get("response", "")
        tavily_response = tavily_result.get("response", "")
        
        if rag_response and tavily_response:
            combined_response = (
                f"From knowledge base:\n{rag_response}\n\n"
                f"Additional information from web search:\n{tavily_response}"
            )
        elif tavily_response:
            combined_response = tavily_response
        else:
            combined_response = rag_response or "No information available."
        
        # Combine sources
        rag_sources = rag_result.get("sources", [])
        tavily_sources = tavily_result.get("sources", [])
        
        combined_sources = []
        if rag_sources:
            combined_sources.extend([f"KB: {source}" for source in rag_sources])
        if tavily_sources:
            combined_sources.extend([f"Web: {source}" for source in tavily_sources])
        
        return {
            "response": combined_response,
            "sources": combined_sources,
            "used_tavily": True,
            "rag_documents": rag_result.get("retrieved_documents", 0),
            "tavily_results": tavily_result.get("search_results_count", 0),
            "retrieval_method": "hybrid_with_web_fallback"
        }
