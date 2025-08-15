"""
Message routing system using Gemini classifier
"""

import logging
from typing import Dict, List
from app.llm.gemini import chat

logger = logging.getLogger(__name__)

# Supported agent types
AGENT_TYPES = [
    "offer",
    "trust", 
    "dispute",
    "collections",
    "contracts",
    "devcopilot",
    "carecredit",
    "narrator",
]

def route(message: str) -> Dict[str, any]:
    """
    Route a message to the appropriate agent using Gemini classifier
    
    Args:
        message: User message to classify
        
    Returns:
        Dictionary with agent type and confidence score
    """
    try:
        # Create classification prompt
        system_prompt = """You are a message routing classifier for a financial services platform. 
Your job is to classify user messages into one of these agent categories based on intent and content.

Agent Categories:
- offer: Product offers, promotions, deals, discounts, marketing content
- trust: Security concerns, fraud detection, suspicious activity, safety issues
- dispute: Transaction disputes, chargebacks, billing issues, payment problems
- collections: Debt collection, overdue payments, payment reminders
- contracts: Contract terms, agreements, legal documents, merchant agreements
- devcopilot: Technical support, API questions, integration help, developer tools
- carecredit: Healthcare financing, medical payments, care credit specific queries
- narrator: General conversation, greetings, small talk, unclear intent

Respond with ONLY a JSON object in this exact format:
{"agent": "category_name", "confidence": 0.85}

The confidence should be a float between 0.0 and 1.0 representing how certain you are about the classification."""

        user_message = f"Classify this message: '{message}'"
        
        messages = [{"role": "user", "content": user_message}]
        
        # Get classification from Gemini
        response = chat(messages, system=system_prompt)
        
        # Parse JSON response
        import json
        try:
            # Handle markdown-wrapped JSON responses
            response_text = response.strip()
            if response_text.startswith('```json'):
                # Extract JSON from markdown code block
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                # Handle generic code blocks
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]).strip()
            
            result = json.loads(response_text)
            
            # Validate response format
            if not isinstance(result, dict) or "agent" not in result or "confidence" not in result:
                raise ValueError("Invalid response format")
            
            agent = result["agent"]
            confidence = float(result["confidence"])
            
            # Validate agent type
            if agent not in AGENT_TYPES:
                logger.warning(f"Unknown agent type '{agent}', defaulting to 'trust'")
                agent = "trust"
                confidence = 0.3
            
            # Validate confidence range
            confidence = max(0.0, min(1.0, confidence))
            
            # Apply default routing rule: if confidence < 0.5, default to "trust"
            if confidence < 0.5:
                logger.info(f"Low confidence ({confidence:.3f}) for agent '{agent}', defaulting to 'trust'")
                agent = "trust"
                confidence = 0.5
            
            logger.info(f"Routed message to '{agent}' with confidence {confidence:.3f}")
            
            return {
                "agent": agent,
                "confidence": confidence
            }
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse classification response: {e}")
            logger.debug(f"Raw response: {response}")
            
            # Fallback classification
            return {
                "agent": "trust",
                "confidence": 0.3
            }
            
    except Exception as e:
        logger.error(f"Error in message routing: {e}")
        
        # Default fallback
        return {
            "agent": "trust", 
            "confidence": 0.3
        }

def get_agent_description(agent_type: str) -> str:
    """
    Get description of what each agent handles
    
    Args:
        agent_type: The agent type
        
    Returns:
        Description string
    """
    descriptions = {
        "offer": "Handles product offers, promotions, deals, and marketing content",
        "trust": "Manages security concerns, fraud detection, and safety issues", 
        "dispute": "Processes transaction disputes, chargebacks, and billing issues",
        "collections": "Handles debt collection and overdue payment matters",
        "contracts": "Manages contract terms, agreements, and legal documents",
        "devcopilot": "Provides technical support and developer assistance",
        "carecredit": "Handles healthcare financing and medical payment queries",
        "narrator": "Manages general conversation and unclear intents"
    }
    
    return descriptions.get(agent_type, "Unknown agent type")

def route_with_fallback_analysis(message: str) -> Dict[str, any]:
    """
    Enhanced routing with fallback keyword analysis
    
    Args:
        message: User message to classify
        
    Returns:
        Dictionary with agent type, confidence, and reasoning
    """
    # First try Gemini classification
    primary_result = route(message)
    
    # If confidence is still low, try keyword-based fallback
    if primary_result["confidence"] < 0.6:
        keyword_result = _keyword_based_routing(message)
        
        # Use keyword result if it has higher confidence
        if keyword_result["confidence"] > primary_result["confidence"]:
            logger.info(f"Using keyword-based routing over Gemini classification")
            return {
                **keyword_result,
                "method": "keyword_fallback",
                "gemini_result": primary_result
            }
    
    return {
        **primary_result,
        "method": "gemini_classification"
    }

def _keyword_based_routing(message: str) -> Dict[str, any]:
    """
    Simple keyword-based routing as fallback
    
    Args:
        message: User message
        
    Returns:
        Classification result
    """
    message_lower = message.lower()
    
    # Define keyword patterns for each agent
    patterns = {
        "trust": [
            "fraud", "scam", "suspicious", "security", "hack", "phishing", 
            "stolen", "unauthorized", "breach", "compromise", "malware",
            "gift card", "wire transfer", "refund scam", "overpay"
        ],
        "dispute": [
            "dispute", "chargeback", "wrong charge", "billing error", 
            "refund", "cancel", "unauthorized charge", "double charge"
        ],
        "collections": [
            "overdue", "payment due", "debt", "collection", "past due",
            "late payment", "outstanding balance"
        ],
        "contracts": [
            "contract", "agreement", "terms", "legal", "merchant agreement",
            "terms of service", "privacy policy"
        ],
        "devcopilot": [
            "api", "integration", "technical", "developer", "code", "sdk",
            "documentation", "endpoint", "webhook"
        ],
        "carecredit": [
            "care credit", "medical", "healthcare", "dental", "veterinary",
            "health financing"
        ],
        "offer": [
            "offer", "promotion", "deal", "discount", "sale", "special",
            "limited time", "bonus"
        ]
    }
    
    # Score each agent based on keyword matches
    scores = {}
    for agent, keywords in patterns.items():
        score = sum(1 for keyword in keywords if keyword in message_lower)
        if score > 0:
            scores[agent] = score / len(keywords)  # Normalize by number of keywords
    
    if scores:
        # Get the agent with highest score
        best_agent = max(scores, key=scores.get)
        confidence = min(0.8, scores[best_agent] * 2)  # Cap at 0.8 for keyword matching
        
        return {
            "agent": best_agent,
            "confidence": confidence
        }
    
    # No keywords matched
    return {
        "agent": "narrator",
        "confidence": 0.4
    }
