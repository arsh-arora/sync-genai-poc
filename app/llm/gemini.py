"""
Gemini chat model integration for LLM functionality
"""

import os
import logging
from typing import List, Dict, Any, Optional

from google import genai

logger = logging.getLogger(__name__)

def chat(messages: List[Dict[str, str]], system: Optional[str] = None) -> str:
    """
    Generate chat response using Gemini chat model
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
                 Roles should be 'user' or 'assistant'
        system: Optional system prompt to prepend to the conversation
        
    Returns:
        Generated response string
        
    Raises:
        ValueError: If API key is not configured
        Exception: For API errors or other failures
    """
    # Get configuration
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    
    model_name = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    
    # Initialize the modern Gemini client
    client = genai.Client(api_key=api_key)
    
    try:
        # Build the prompt
        prompt_parts = []
        
        # Add system prompt if provided
        if system:
            prompt_parts.append(f"System: {system}\n")
        
        # Add conversation messages
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                # Handle unknown roles as user messages
                prompt_parts.append(f"User: {content}")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        # Combine into single prompt
        full_prompt = "\n\n".join(prompt_parts)
        
        logger.debug(f"Sending prompt to Gemini model {model_name}")
        
        # Generate response using modern API
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt
        )
        
        if not response.text:
            logger.warning("Empty response from Gemini API")
            return "I apologize, but I couldn't generate a response. Please try again."
        
        logger.debug(f"Received response from Gemini: {len(response.text)} characters")
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error generating chat response with Gemini: {e}")
        return f"I apologize, but I encountered an error while processing your request: {str(e)}"

def chat_with_context(
    query: str,
    context_documents: List[Dict[str, Any]],
    system_prompt: Optional[str] = None
) -> str:
    """
    Generate chat response with retrieved document context
    
    Args:
        query: User's query
        context_documents: List of retrieved documents with 'snippet' and 'source' keys
        system_prompt: Optional system prompt
        
    Returns:
        Generated response string
    """
    # Default system prompt for RAG
    default_system = (
        "You are a helpful assistant that answers questions based on provided documents. "
        "Use only the information from the documents to answer questions. "
        "If the information is not available in the documents, say so clearly. "
        "Provide specific references to the source documents when possible."
    )
    
    system = system_prompt or default_system
    
    # Build context from documents
    context_parts = []
    for i, doc in enumerate(context_documents, 1):
        snippet = doc.get("snippet", "")
        source = doc.get("source", "unknown")
        filename = doc.get("filename", "unknown")
        doc_type = doc.get("type", "unknown")
        
        context_parts.append(
            f"Document {i}:\n"
            f"Source: {filename} ({doc_type})\n"
            f"Content: {snippet}\n"
        )
    
    context_text = "\n".join(context_parts)
    
    # Create user message with context
    user_message = f"""
Based on the following documents, please answer this question: {query}

Documents:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the information in the documents above.
"""
    
    messages = [{"role": "user", "content": user_message}]
    
    return chat(messages, system=system)

def summarize_text(text: str, max_length: int = 500) -> str:
    """
    Summarize text using Gemini
    
    Args:
        text: Text to summarize
        max_length: Maximum length of summary in words
        
    Returns:
        Summarized text
    """
    system_prompt = (
        f"You are a helpful assistant that creates concise summaries. "
        f"Summarize the following text in no more than {max_length} words. "
        f"Focus on the key points and maintain accuracy."
    )
    
    messages = [{"role": "user", "content": f"Please summarize this text:\n\n{text}"}]
    
    return chat(messages, system=system_prompt)

def extract_key_points(text: str, num_points: int = 5) -> List[str]:
    """
    Extract key points from text using Gemini
    
    Args:
        text: Text to analyze
        num_points: Number of key points to extract
        
    Returns:
        List of key points as strings
    """
    system_prompt = (
        "You are a helpful assistant that extracts key points from text. "
        f"Extract the {num_points} most important points from the following text. "
        "Return each point as a separate line starting with a bullet point (•)."
    )
    
    messages = [{"role": "user", "content": f"Extract key points from this text:\n\n{text}"}]
    
    response = chat(messages, system=system_prompt)
    
    # Parse bullet points
    lines = response.split('\n')
    key_points = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('•') or line.startswith('-') or line.startswith('*'):
            # Remove bullet point and clean up
            point = line[1:].strip()
            if point:
                key_points.append(point)
    
    return key_points[:num_points]  # Ensure we don't exceed requested number

def classify_query(query: str, categories: List[str]) -> str:
    """
    Classify a query into one of the provided categories
    
    Args:
        query: Query to classify
        categories: List of possible categories
        
    Returns:
        Best matching category
    """
    categories_text = ", ".join(categories)
    
    system_prompt = (
        "You are a helpful assistant that classifies user queries. "
        f"Classify the following query into one of these categories: {categories_text}. "
        "Return only the category name, nothing else."
    )
    
    messages = [{"role": "user", "content": f"Classify this query: {query}"}]
    
    response = chat(messages, system=system_prompt).strip()
    
    # Find best match from categories
    response_lower = response.lower()
    for category in categories:
        if category.lower() in response_lower:
            return category
    
    # Return first category as fallback
    return categories[0] if categories else "general"
