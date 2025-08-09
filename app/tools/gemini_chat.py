"""
Gemini Chat Generator for Haystack 2.x
"""

import os
import logging
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole

logger = logging.getLogger(__name__)

@component
class GeminiChatGenerator:
    """
    A component for generating chat responses using Google's Gemini API.
    
    This component integrates with Haystack 2.x pipelines and provides
    chat completion functionality using Gemini models.
    """
    
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize the Gemini Chat Generator.
        
        Args:
            model: The Gemini model to use
            api_key: Google API key (if not provided, uses GOOGLE_API_KEY env var)
            generation_config: Generation configuration parameters
            safety_settings: Safety settings for content filtering
        """
        self.model_name = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key is required")
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        logger.info(f"Initialized Gemini Chat Generator with model: {self.model_name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dictionary."""
        return default_to_dict(
            self,
            model=self.model_name,
            api_key=self.api_key,
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeminiChatGenerator":
        """Deserialize component from dictionary."""
        return default_from_dict(cls, data)
    
    def _convert_messages_to_gemini_format(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        """Convert Haystack ChatMessage format to Gemini format."""
        gemini_messages = []
        
        for message in messages:
            # Map Haystack roles to Gemini roles
            if message.role == ChatRole.SYSTEM:
                # Gemini doesn't have a system role, so we'll prepend system messages to user messages
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": f"System: {message.content}"}]
                })
            elif message.role == ChatRole.USER:
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": message.content}]
                })
            elif message.role == ChatRole.ASSISTANT:
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": message.content}]
                })
        
        return gemini_messages
    
    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """
        Generate a chat response using Gemini.
        
        Args:
            messages: List of ChatMessage objects representing the conversation
            
        Returns:
            Dictionary with 'replies' key containing generated ChatMessage responses
        """
        try:
            logger.info(f"Generating response with {len(messages)} input messages")
            
            # Convert messages to Gemini format
            gemini_messages = self._convert_messages_to_gemini_format(messages)
            
            # For Gemini, we need to handle the conversation differently
            # We'll combine all messages into a single prompt for simplicity
            combined_prompt = ""
            for msg in messages:
                if msg.role == ChatRole.SYSTEM:
                    combined_prompt += f"System: {msg.content}\n\n"
                elif msg.role == ChatRole.USER:
                    combined_prompt += f"User: {msg.content}\n\n"
                elif msg.role == ChatRole.ASSISTANT:
                    combined_prompt += f"Assistant: {msg.content}\n\n"
            
            # Generate response
            response = self.model.generate_content(combined_prompt)
            
            if not response.text:
                logger.warning("Empty response from Gemini API")
                return {"replies": []}
            
            # Create ChatMessage response
            reply = ChatMessage.from_assistant(response.text)
            
            logger.info("Successfully generated response")
            return {"replies": [reply]}
            
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            # Return empty response on error
            return {"replies": []}
