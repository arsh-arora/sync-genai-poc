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

def chat_with_context_and_fallback(
    query: str,
    context_documents: List[Dict[str, Any]],
    allow_llm_knowledge: bool = True,
    allow_web_search: bool = False,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate chat response with fallback options when documents are insufficient
    
    Args:
        query: User's query
        context_documents: List of retrieved documents with 'snippet' and 'source' keys
        allow_llm_knowledge: Whether to fallback to LLM's own knowledge if documents insufficient
        allow_web_search: Whether to use web search if documents insufficient
        system_prompt: Optional system prompt
        
    Returns:
        Dictionary with response, fallback_used, and confidence
    """
    # Check document quality/relevance intelligently
    doc_quality = assess_document_quality(query, context_documents)
    
    fallback_used = None
    confidence = doc_quality["confidence"]
    
    if doc_quality["sufficient"]:
        # Use regular RAG response
        response = chat_with_context(query, context_documents, system_prompt)
    else:
        # Documents are insufficient - use fallback
        if allow_llm_knowledge:
            response = chat_with_llm_knowledge(query, context_documents, system_prompt)
            fallback_used = "llm_knowledge"
            confidence = max(0.6, confidence)  # Boost confidence for LLM knowledge fallback
        elif allow_web_search:
            try:
                # Import here to avoid circular dependency
                from app.tools.tavily_search import web_search_into_docstore
                
                # Perform web search (this should be done in the main endpoint, but for now we'll indicate it)
                response = chat_with_web_search_fallback(query, context_documents, system_prompt)
                fallback_used = "web_search"
                confidence = 0.7  # Higher confidence for web search results
            except ImportError:
                response = f"Based on the documents provided, there is insufficient information to answer '{query}'. Web search functionality is not available."
                fallback_used = "web_search_unavailable"
                confidence = 0.3
        else:
            response = f"Based on the documents provided, there is insufficient information to answer '{query}'. Consider enabling fallback options (LLM knowledge or web search) to access additional information sources."
            fallback_used = "none"
            confidence = 0.2
    
    return {
        "response": response,
        "fallback_used": fallback_used,
        "confidence": confidence,
        "document_assessment": doc_quality
    }

def assess_document_quality(query: str, context_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Intelligently assess whether retrieved documents are sufficient to answer the query
    
    Args:
        query: User's query
        context_documents: Retrieved documents
        
    Returns:
        Dictionary with assessment results
    """
    if not context_documents:
        return {
            "sufficient": False,
            "confidence": 0.0,
            "reason": "No documents found",
            "document_quality_score": 0.0,
            "semantic_relevance": 0.0,
            "coverage_score": 0.0
        }
    
    # Use LLM to assess document relevance and coverage
    assessment_prompt = f"""
Analyze whether these documents contain sufficient information to answer the user's question comprehensively.

Question: {query}

Documents:
{_format_docs_for_assessment(context_documents)}

Rate the documents on a scale of 0-10 for:
1. Semantic Relevance: How well do the documents relate to the question?
2. Information Coverage: How completely do they address what the user is asking?
3. Answer Sufficiency: Can the question be fully answered using only these documents?

Respond in this exact JSON format:
{{
    "semantic_relevance": <0-10>,
    "coverage_score": <0-10>, 
    "answer_sufficiency": <0-10>,
    "sufficient": <true/false>,
    "reasoning": "<brief explanation>"
}}
"""
    
    try:
        # Get LLM assessment
        messages = [{"role": "user", "content": assessment_prompt}]
        assessment_response = chat(messages, system="You are an expert at evaluating document relevance. Provide only valid JSON responses.")
        
        logger.debug(f"Raw Gemini assessment response: {assessment_response[:200]}...")
        
        # Parse JSON response with robust handling
        import json
        
        # Handle markdown-wrapped JSON responses
        response_text = assessment_response.strip()
        if response_text.startswith('```json'):
            # Extract JSON from markdown code block
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        elif response_text.startswith('```'):
            # Handle generic code blocks
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]).strip()
        
        # Check if response is empty or contains fallback message
        if not response_text or "I apologize, but I couldn't generate a response" in response_text:
            logger.error("Empty or fallback response from Gemini API for document assessment")
            raise ValueError("Empty or fallback response from Gemini API")
        
        logger.debug(f"Cleaned response text: {response_text[:200]}...")
        assessment = json.loads(response_text)
        
        # Validate assessment structure
        required_fields = ['semantic_relevance', 'coverage_score', 'answer_sufficiency', 'sufficient']
        missing_fields = [field for field in required_fields if field not in assessment]
        if missing_fields:
            logger.warning(f"Missing fields in assessment: {missing_fields}")
            # Set default values for missing fields
            for field in missing_fields:
                if field == 'sufficient':
                    assessment[field] = False
                else:
                    assessment[field] = 0
        
        # Calculate overall quality score
        quality_score = (
            assessment.get("semantic_relevance", 0) * 0.4 +
            assessment.get("coverage_score", 0) * 0.4 +
            assessment.get("answer_sufficiency", 0) * 0.2
        ) / 10.0
        
        # Determine sufficiency with intelligent thresholds
        sufficient = assessment.get("sufficient", False) and quality_score >= 0.6
        
        return {
            "sufficient": sufficient,
            "confidence": quality_score,
            "reason": assessment.get("reasoning", "Assessment completed"),
            "document_quality_score": quality_score,
            "semantic_relevance": assessment.get("semantic_relevance", 0) / 10.0,
            "coverage_score": assessment.get("coverage_score", 0) / 10.0,
            "answer_sufficiency": assessment.get("answer_sufficiency", 0) / 10.0
        }
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Gemini JSON response for assessment, falling back to heuristics. Response was: '{assessment_response[:100]}...' Error: {e}")
        
        # Fallback to enhanced heuristic assessment
    except Exception as e:
        logger.warning(f"Failed to get intelligent assessment, falling back to heuristics: {e}")
        
        # Fallback to enhanced heuristic assessment
        scores = [doc.get("score", 0.0) for doc in context_documents]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Enhanced heuristics
        keyword_matches = _count_keyword_matches(query, context_documents)
        length_score = min(1.0, sum(len(doc.get("snippet", "")) for doc in context_documents) / 1000)
        diversity_score = _calculate_document_diversity(context_documents)
        
        quality_score = (avg_score * 0.4) + (keyword_matches * 0.3) + (length_score * 0.2) + (diversity_score * 0.1)
        sufficient = quality_score >= 0.5 and len(context_documents) >= 1
        
        return {
            "sufficient": sufficient,
            "confidence": quality_score,
            "reason": f"Heuristic assessment - relevance: {avg_score:.2f}, keywords: {keyword_matches:.2f}, content: {length_score:.2f}",
            "document_quality_score": quality_score,
            "semantic_relevance": avg_score,
            "coverage_score": keyword_matches,
            "answer_sufficiency": length_score
        }

def chat_with_llm_knowledge(
    query: str,
    context_documents: List[Dict[str, Any]],
    system_prompt: Optional[str] = None
) -> str:
    """
    Generate chat response using LLM's own knowledge with document context as supporting info
    
    Args:
        query: User's query
        context_documents: Retrieved documents (used as supporting context)
        system_prompt: Optional system prompt
        
    Returns:
        Generated response string
    """
    # Enhanced system prompt that allows LLM knowledge
    enhanced_system = (
        "You are a helpful assistant with extensive knowledge. "
        "Answer the user's question using your knowledge and training. "
        "If relevant documents are provided, use them as supporting evidence, but you may "
        "also draw from your general knowledge to provide a comprehensive answer. "
        "Clearly distinguish between information from documents and your general knowledge. "
        "If the documents have limited relevance, focus more on your training knowledge."
    )
    
    system = system_prompt or enhanced_system
    
    # Build supporting context from documents if available
    if context_documents:
        context_parts = []
        for i, doc in enumerate(context_documents[:3], 1):  # Limit to top 3
            snippet = doc.get("snippet", "")[:400]  # Limit length
            filename = doc.get("filename", "unknown")
            score = doc.get("score", 0.0)
            
            context_parts.append(f"Supporting Document {i} ({filename}, relevance: {score:.2f}):\n{snippet}")
        
        context_text = "\n\n".join(context_parts)
        
        user_message = f"""
Question: {query}

Supporting documents (may have limited relevance):
{context_text}

Please answer the question comprehensively using your knowledge. You may reference the supporting documents if relevant, but feel free to provide additional context and information from your training to give a complete answer. If the documents don't fully address the question, supplement with your general knowledge.
"""
    else:
        user_message = f"Question: {query}\n\nPlease provide a comprehensive answer using your knowledge and training."
    
    messages = [{"role": "user", "content": user_message}]
    
    return chat(messages, system=system)

def _format_docs_for_assessment(context_documents: List[Dict[str, Any]]) -> str:
    """Format documents for LLM assessment"""
    formatted = []
    for i, doc in enumerate(context_documents[:5], 1):  # Limit to top 5 docs
        snippet = doc.get("snippet", "")[:300]  # Limit snippet length
        filename = doc.get("filename", "unknown")
        score = doc.get("score", 0.0)
        
        formatted.append(f"Doc {i} ({filename}, score: {score:.2f}):\n{snippet}")
    
    return "\n\n".join(formatted)

def _count_keyword_matches(query: str, context_documents: List[Dict[str, Any]]) -> float:
    """Count keyword matches between query and documents"""
    import re
    
    # Extract keywords from query (simple approach)
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    query_words = {w for w in query_words if len(w) > 2}  # Filter short words
    
    if len(query_words) == 0:
        return 0.0
    
    total_matches = 0
    for doc in context_documents:
        snippet = doc.get("snippet", "").lower()
        doc_words = set(re.findall(r'\b\w+\b', snippet))
        matches = len(query_words.intersection(doc_words))
        total_matches += matches
    
    # Normalize by query length and document count
    return min(1.0, total_matches / (len(query_words) * len(context_documents)))

def _calculate_document_diversity(context_documents: List[Dict[str, Any]]) -> float:
    """Calculate diversity score based on different document sources"""
    if len(context_documents) <= 1:
        return 0.5
    
    sources = set()
    for doc in context_documents:
        filename = doc.get("filename", "unknown")
        doc_type = doc.get("type", "unknown")
        sources.add(f"{filename}_{doc_type}")
    
    # More diverse sources = higher score
    diversity = len(sources) / len(context_documents)
    return min(1.0, diversity * 2)  # Scale up the impact

def chat_with_web_search_fallback(
    query: str,
    context_documents: List[Dict[str, Any]],
    system_prompt: Optional[str] = None
) -> str:
    """
    Generate chat response indicating web search would be performed
    (Actual web search should be handled in the main endpoint)
    
    Args:
        query: User's query
        context_documents: Retrieved documents (used as supporting context)
        system_prompt: Optional system prompt
        
    Returns:
        Generated response string
    """
    # Enhanced system prompt that acknowledges web search capability
    enhanced_system = (
        "You are a helpful assistant with access to both document knowledge and web search. "
        "The provided documents have limited relevance to the user's question, so you should "
        "indicate that web search would provide more current and comprehensive information. "
        "Acknowledge what limited information is available in the documents, then suggest "
        "that web search would be performed to get more complete answers."
    )
    
    system = system_prompt or enhanced_system
    
    # Build context information
    if context_documents:
        doc_info = f"Available documents ({len(context_documents)} found) have limited relevance to your question."
    else:
        doc_info = "No relevant documents were found in the knowledge base."
    
    user_message = f"""
Question: {query}

{doc_info}

Since the document knowledge is insufficient to fully answer your question, a web search would typically be performed here to find more current and comprehensive information. This would help provide you with up-to-date answers that go beyond the available document knowledge base.
    
Please provide a response acknowledging the limitation and indicating what a web search could help find.
"""
    
    messages = [{"role": "user", "content": user_message}]
    
    return chat(messages, system=system)
