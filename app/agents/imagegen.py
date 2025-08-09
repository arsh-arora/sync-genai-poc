"""
ImageGen - AI Image Generation Agent
Creates images using Gemini's native image generation capabilities
"""

import logging
import base64
from io import BytesIO
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from PIL import Image

from google import genai
from google.genai import types
from pydantic import BaseModel

logger = logging.getLogger(__name__)

@dataclass
class GeneratedImage:
    """Generated image with metadata"""
    image_data: bytes
    format: str
    description: str
    prompt: str
    base64_data: str

class ImageGenRequest(BaseModel):
    prompt: str
    include_text: bool = True
    style_hints: Optional[List[str]] = None

class ImageGenResponse(BaseModel):
    success: bool
    prompt: str
    generated_text: Optional[str] = None
    image_base64: Optional[str] = None
    image_format: str = "PNG"
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ImageGenAgent:
    """
    AI Image Generation Agent using Gemini's native capabilities
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize ImageGen agent"""
        self.client = None
        self.model_name = "gemini-2.0-flash-preview-image-generation"
        
        try:
            import os
            api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key is required for image generation")
            
            self.client = genai.Client(api_key=api_key)
            logger.info("ImageGen agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ImageGen agent: {e}")
            raise
    
    def generate_image(self, prompt: str, include_text: bool = True, 
                      style_hints: Optional[List[str]] = None) -> ImageGenResponse:
        """
        Generate an image based on the text prompt
        
        Args:
            prompt: Text description of the image to generate
            include_text: Whether to include descriptive text with the image
            style_hints: Optional style hints to improve generation
            
        Returns:
            ImageGenResponse with generated image and metadata
        """
        try:
            logger.info(f"Generating image for prompt: {prompt[:100]}...")
            
            # Enhance prompt with style hints if provided
            enhanced_prompt = self._enhance_prompt(prompt, style_hints)
            
            # Configure generation parameters
            config = types.GenerateContentConfig(
                response_modalities=['IMAGE'] + (['TEXT'] if include_text else [])
            )
            
            # Generate content using Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=enhanced_prompt,
                config=config
            )
            
            # Process the response
            generated_text = None
            generated_image = None
            
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                
                for part in candidate.content.parts:
                    if part.text is not None and include_text:
                        generated_text = part.text
                    elif part.inline_data is not None:
                        # Convert image data to base64
                        image_data = part.inline_data.data
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
                        
                        # Create PIL image for validation
                        image = Image.open(BytesIO(image_data))
                        
                        generated_image = GeneratedImage(
                            image_data=image_data,
                            format="PNG",
                            description=generated_text or prompt,
                            prompt=enhanced_prompt,
                            base64_data=image_base64
                        )
                        
                        logger.info(f"Successfully generated {image.size[0]}x{image.size[1]} image")
            
            if generated_image:
                return ImageGenResponse(
                    success=True,
                    prompt=enhanced_prompt,
                    generated_text=generated_text,
                    image_base64=generated_image.base64_data,
                    image_format=generated_image.format,
                    metadata={
                        "model": self.model_name,
                        "original_prompt": prompt,
                        "enhanced_prompt": enhanced_prompt,
                        "include_text": include_text,
                        "style_hints": style_hints
                    }
                )
            else:
                return ImageGenResponse(
                    success=False,
                    prompt=enhanced_prompt,
                    error_message="No image was generated in the response"
                )
                
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return ImageGenResponse(
                success=False,
                prompt=prompt,
                error_message=str(e)
            )
    
    def _enhance_prompt(self, prompt: str, style_hints: Optional[List[str]] = None) -> str:
        """
        Enhance the prompt with style hints and quality improvements
        
        Args:
            prompt: Original prompt
            style_hints: Optional style hints
            
        Returns:
            Enhanced prompt string
        """
        enhanced = prompt
        
        # Add style hints if provided
        if style_hints:
            style_text = ", ".join(style_hints)
            enhanced = f"{enhanced}, {style_text}"
        
        # Add default quality enhancers if not already present
        quality_terms = ["high quality", "detailed", "professional", "4k", "8k", "hd"]
        if not any(term in enhanced.lower() for term in quality_terms):
            enhanced = f"{enhanced}, high quality, detailed"
        
        # Add artistic style guidance if none specified
        style_terms = ["photorealistic", "digital art", "painting", "3d render", "concept art", "illustration"]
        if not any(term in enhanced.lower() for term in style_terms):
            enhanced = f"{enhanced}, digital art"
        
        return enhanced
    
    def process_request(self, request: ImageGenRequest) -> ImageGenResponse:
        """
        Process an image generation request
        
        Args:
            request: ImageGenRequest object
            
        Returns:
            ImageGenResponse with results
        """
        return self.generate_image(
            prompt=request.prompt,
            include_text=request.include_text,
            style_hints=request.style_hints
        )

# Utility functions for image handling
def save_generated_image(image_response: ImageGenResponse, 
                        filename: Optional[str] = None) -> Optional[str]:
    """
    Save a generated image to disk
    
    Args:
        image_response: ImageGenResponse with image data
        filename: Optional filename, will auto-generate if not provided
        
    Returns:
        Saved filename or None if failed
    """
    try:
        if not image_response.success or not image_response.image_base64:
            return None
        
        # Generate filename if not provided
        if not filename:
            import time
            timestamp = int(time.time())
            filename = f"generated_image_{timestamp}.{image_response.image_format.lower()}"
        
        # Decode and save image
        image_data = base64.b64decode(image_response.image_base64)
        image = Image.open(BytesIO(image_data))
        image.save(filename)
        
        logger.info(f"Saved generated image to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return None

def get_image_data_uri(image_response: ImageGenResponse) -> Optional[str]:
    """
    Get data URI for displaying image in HTML
    
    Args:
        image_response: ImageGenResponse with image data
        
    Returns:
        Data URI string or None if failed
    """
    try:
        if not image_response.success or not image_response.image_base64:
            return None
        
        format_lower = image_response.image_format.lower()
        return f"data:image/{format_lower};base64,{image_response.image_base64}"
        
    except Exception as e:
        logger.error(f"Error creating data URI: {e}")
        return None

# Test cases for ImageGen agent
def test_imagegen():
    """Test ImageGen agent with various prompts"""
    print("🎨 Testing ImageGen Agent")
    print("=" * 40)
    
    try:
        agent = ImageGenAgent()
        
        test_cases = [
            {
                "name": "Simple Object",
                "prompt": "A red apple on a wooden table",
                "expected_success": True
            },
            {
                "name": "Complex Scene", 
                "prompt": "A futuristic city with flying cars and neon lights at night",
                "style_hints": ["cyberpunk", "neon", "cinematic"],
                "expected_success": True
            },
            {
                "name": "Fantasy Character",
                "prompt": "A wizard casting a spell in an enchanted forest",
                "style_hints": ["fantasy art", "magical", "mystical"],
                "expected_success": True
            }
        ]
        
        passed = 0
        total = len(test_cases)
        
        for i, case in enumerate(test_cases, 1):
            print(f"{i}. {case['name']}")
            print(f"   Prompt: '{case['prompt']}'")
            
            try:
                result = agent.generate_image(
                    prompt=case["prompt"],
                    style_hints=case.get("style_hints")
                )
                
                success = result.success == case["expected_success"]
                
                if success and result.success:
                    print(f"   ✅ PASS - Image generated successfully")
                    if result.generated_text:
                        print(f"   📝 Description: {result.generated_text[:100]}...")
                    print(f"   🖼️ Format: {result.image_format}")
                    passed += 1
                elif success:
                    print(f"   ✅ PASS - Expected behavior matched")
                    passed += 1
                else:
                    print(f"   ❌ FAIL - {result.error_message}")
                    
            except Exception as e:
                print(f"   ❌ FAIL - Exception: {str(e)}")
            
            print()
        
        print(f"📊 ImageGen Results: {passed}/{total} tests passed")
        return passed == total
        
    except Exception as e:
        print(f"❌ Failed to initialize ImageGen agent: {e}")
        return False

if __name__ == "__main__":
    test_imagegen()