"""
Tavily web search integration with document store indexing
"""

import os
import logging
from typing import List, Dict, Any, Optional
import time

from tavily import TavilyClient
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from app.rag.core import GeminiEmbedder, WebDisabled
from app.llm.gemini import chat

logger = logging.getLogger(__name__)

def web_search_into_docstore(
    docstore: InMemoryDocumentStore,
    embedder: GeminiEmbedder,
    query: str,
    max_results: int = 5
) -> List[Document]:
    """
    Perform web search with Tavily and add results to document store
    
    Args:
        docstore: Haystack document store to add results to
        embedder: Gemini embedder for generating embeddings
        query: Search query
        max_results: Maximum number of search results to process
        
    Returns:
        List of newly added Document objects
        
    Raises:
        WebDisabled: If ALLOW_TAVILY is not set to true
    """
    # Check if Tavily is enabled
    if os.getenv("ALLOW_TAVILY", "false").lower() != "true":
        raise WebDisabled("Web search is disabled. Set ALLOW_TAVILY=true to enable.")
    
    # Get Tavily API key
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required when ALLOW_TAVILY=true")
    
    try:
        # Initialize Tavily client
        client = TavilyClient(api_key=tavily_api_key)
        
        # Perform search
        logger.info(f"Performing web search for: {query}")
        search_result = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=True,
            include_raw_content=True
        )
        
        # Process search results
        documents = []
        results = search_result.get("results", [])
        
        if not results:
            logger.warning(f"No web search results found for query: {query}")
            return documents
        
        # Process each search result
        for i, result in enumerate(results):
            try:
                title = result.get("title", "")
                url = result.get("url", "")
                content = result.get("content", "")
                raw_content = result.get("raw_content", "")
                
                # Use raw content if available, otherwise use content
                full_content = raw_content if raw_content else content
                
                if not full_content:
                    logger.warning(f"No content found for URL: {url}")
                    continue
                
                # Summarize content using Gemini
                summary = _summarize_web_content(title, full_content, url, query)
                
                # Create document
                doc = Document(
                    content=summary,
                    meta={
                        "source": "web_search",
                        "source_url": url,
                        "title": title,
                        "type": "web_result",
                        "search_query": query,
                        "result_index": i,
                        "original_content_length": len(full_content),
                        "timestamp": time.time()
                    }
                )
                
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Error processing search result {i}: {e}")
                continue
        
        if not documents:
            logger.warning("No valid documents created from web search results")
            return documents
        
        # Generate embeddings for all documents
        logger.info(f"Generating embeddings for {len(documents)} web search results...")
        texts = [doc.content for doc in documents]
        embeddings = embedder.embed_texts(texts)
        
        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        
        # Write documents to store
        docstore.write_documents(documents)
        
        logger.info(f"Successfully added {len(documents)} web search results to document store")
        return documents
        
    except WebDisabled:
        raise
    except Exception as e:
        logger.error(f"Error during web search: {e}")
        return []

def _summarize_web_content(title: str, content: str, url: str, query: str) -> str:
    """
    Summarize web content using Gemini chat model
    
    Args:
        title: Page title
        content: Page content
        url: Page URL
        query: Original search query
        
    Returns:
        Summarized content
    """
    try:
        # Truncate content if too long (keep first 3000 characters)
        truncated_content = content[:3000] if len(content) > 3000 else content
        
        # Create summarization prompt
        system_prompt = (
            "You are a helpful assistant that summarizes web content. "
            "Create a concise but informative summary that captures the key information "
            "relevant to the user's search query. Focus on factual content and maintain accuracy."
        )
        
        user_message = f"""
Please summarize the following web content in relation to the search query: "{query}"

Title: {title}
URL: {url}
Content: {truncated_content}

Provide a clear, factual summary that highlights information most relevant to the search query.
Keep the summary concise but comprehensive (2-3 paragraphs maximum).
"""
        
        messages = [
            {"role": "user", "content": user_message}
        ]
        
        # Get summary from Gemini
        summary = chat(messages, system=system_prompt)
        
        # Add source attribution
        attributed_summary = f"{summary}\n\nSource: {title} ({url})"
        
        return attributed_summary
        
    except Exception as e:
        logger.error(f"Error summarizing content from {url}: {e}")
        # Return truncated original content as fallback
        fallback_content = content[:1000] if len(content) > 1000 else content
        return f"Content from {title} ({url}):\n\n{fallback_content}"

def search_and_get_results(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Perform web search and return formatted results without adding to docstore
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        List of formatted search results
        
    Raises:
        WebDisabled: If ALLOW_TAVILY is not set to true
    """
    # Check if Tavily is enabled
    if os.getenv("ALLOW_TAVILY", "false").lower() != "true":
        raise WebDisabled("Web search is disabled. Set ALLOW_TAVILY=true to enable.")
    
    # Get Tavily API key
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required when ALLOW_TAVILY=true")
    
    try:
        # Initialize Tavily client
        client = TavilyClient(api_key=tavily_api_key)
        
        # Perform search
        logger.info(f"Performing web search for: {query}")
        search_result = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=True
        )
        
        # Format results
        formatted_results = []
        results = search_result.get("results", [])
        
        for result in results:
            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0.0)
            })
        
        # Include answer if available
        answer = search_result.get("answer")
        if answer:
            formatted_results.insert(0, {
                "title": "Direct Answer",
                "url": "tavily://answer",
                "content": answer,
                "score": 1.0
            })
        
        logger.info(f"Retrieved {len(formatted_results)} web search results")
        return formatted_results
        
    except WebDisabled:
        raise
    except Exception as e:
        logger.error(f"Error during web search: {e}")
        return []
