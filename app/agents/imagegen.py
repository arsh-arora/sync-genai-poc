"""
ImageGen - B2B Marketing Studio with Real Image Generation
Template-driven creative generation with co-branding, dynamic disclosures, compliance redlining, and actual image generation
"""

import logging
import base64
import re
import os
from io import BytesIO
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from pydantic import BaseModel, Field
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

from app.llm.gemini import chat

logger = logging.getLogger(__name__)

# Pydantic models for B2B marketing studio
class BrandSettings(BaseModel):
    name: str
    hex_color: str = "#2E86DE"
    placement: str = "top_left"

class CreativeCopy(BaseModel):
    headline: str
    body: Optional[str] = None
    cta: str
    legal: str

class RedlineChange(BaseModel):
    original: str
    replacement: str
    reason: str
    severity: str

class ImageGenResponse(BaseModel):
    response: str  # Brief creative summary + legal footer statement
    metadata: Dict[str, Any]  # ui_cards, disclosures, handoffs, image data

class ImageGenAgent:
    """
    B2B Marketing Studio with Real Image Generation
    Template selection, co-branding, dynamic disclosures, compliance redlining, and Gemini native image generation
    """
    
    def __init__(self, docstore=None, embedder=None, retriever=None, rules_loader=None):
        """Initialize ImageGen with marketing rules and Gemini image generation"""
        self.docstore = docstore
        self.embedder = embedder
        self.retriever = retriever
        self.rules_loader = rules_loader
        
        # Initialize Gemini image generation client
        self.client = None
        self.model_name = "gemini-2.0-flash-preview-image-generation"
        
        try:
            # Load environment variables
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                os.environ['GOOGLE_API_KEY'] = api_key  # Ensure env var is set
                self.client = genai.Client()
                logger.info("ImageGen Gemini client initialized for gemini-2.0-flash-preview-image-generation")
            else:
                logger.warning("GOOGLE_API_KEY not found - image generation will be disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
        
        # Load marketing rules
        if rules_loader:
            self.imagegen_rules = rules_loader.get_rules('imagegen') or {}
            logger.info("ImageGenAgent loaded rules from centralized rules loader")
        else:
            self.imagegen_rules = self._load_fallback_rules()
        
        # Extract rule components
        self.templates = self.imagegen_rules.get("templates", {})
        self.co_brand_settings = self.imagegen_rules.get("co_brand_settings", {})
        self.dynamic_disclosures = self.imagegen_rules.get("dynamic_disclosures", {})
        self.banned_phrases = self.imagegen_rules.get("banned_phrases", {})
        self.legal_footers = self.imagegen_rules.get("legal_footers", {})
    
    def _load_fallback_rules(self) -> Dict[str, Any]:
        """Fallback rules if centralized loader not available"""
        return {
            "templates": {
                "a4_instore": {
                    "format": "A4", "dimensions": "8.5x11", "orientation": "portrait",
                    "layout": ["headline", "subhead", "price_callout", "cta", "legal_footer"],
                    "brand_placement": "top_left"
                },
                "social_square": {
                    "format": "square", "dimensions": "1080x1080", "orientation": "square", 
                    "layout": ["headline", "caption_with_legal"], "brand_placement": "bottom_right"
                }
            },
            "co_brand_settings": {
                "default_hex": "#2E86DE", "placeholder_style": "initials_in_circle"
            },
            "dynamic_disclosures": {
                "equal_payment": {"triggers": ["equal payment"], "disclosure_id": "equal_payment_generic"}
            },
            "banned_phrases": {
                "guaranteed_approval": {"phrases": ["guaranteed approval"], "replacement": "subject to credit approval", "severity": "high"}
            },
            "legal_footers": {
                "equal_payment_generic": "Equal monthly payments required. Subject to credit approval."
            }
        }
    
    def generate_creative(
        self,
        message: str,
        template: str = "a4_instore",
        brand: Optional[BrandSettings] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ImageGenResponse:
        """
        Main creative generation pipeline with compliance, co-branding, and actual image generation
        
        Args:
            message: Marketing message content
            template: Template selection (a4_instore, social_square, etc.)
            brand: Brand settings for co-branding
            context: Additional context for creative generation
            
        Returns:
            ImageGenResponse with creative summary, metadata, and actual generated image
        """
        try:
            logger.info(f"Generating creative with template: {template}")
            
            # Step 1: Validate template selection
            template_config = self._validate_template(template)
            
            # Step 2: Redline pass - check for banned phrases using TrustShield-style logic
            redlined_message, redline_changes = self._redline_pass(message)
            
            # Step 3: Dynamic disclosure detection
            required_disclosures = self._detect_disclosures(redlined_message)
            
            # Step 4: Generate creative copy
            creative_copy = self._generate_creative_copy(redlined_message, template_config)
            
            # Step 5: Apply co-branding
            brand_config = self._apply_co_branding(brand, template_config)
            
            # Step 6: Generate actual image using Gemini
            image_data = self._generate_image(creative_copy, template_config, brand_config, redlined_message)
            
            # Step 7: Create preview reference and UI cards with image
            preview_ref, ui_cards = self._create_ui_cards(template, creative_copy, brand_config, image_data)
            
            # Step 8: Detect handoffs
            handoffs = self._detect_handoffs(message, redline_changes)
            
            # Step 9: Generate response summary
            response_text = self._generate_response_summary(creative_copy, redline_changes, required_disclosures, image_data)
            
            return ImageGenResponse(
                response=response_text,
                metadata={
                    "ui_cards": ui_cards,
                    "disclosures": required_disclosures,
                    "handoffs": handoffs,
                    "template": template,
                    "brand_config": brand_config,
                    "redline_changes": [change.model_dump() for change in redline_changes],
                    "preview_ref": preview_ref,
                    "image_generated": image_data is not None,
                    "image_base64": image_data.get("base64") if image_data else None,
                    "image_format": image_data.get("format") if image_data else None
                }
            )
            
        except Exception as e:
            logger.error(f"Creative generation error: {e}")
            return self._error_response(f"Creative generation failed: {str(e)}")
    
    def _generate_image(
        self, 
        creative_copy: CreativeCopy, 
        template_config: Dict[str, Any], 
        brand_config: Dict[str, Any],
        original_message: str
    ) -> Optional[Dict[str, Any]]:
        """Generate actual image using Gemini native image generation"""
        
        if not self.client:
            logger.warning("Gemini client not available - skipping image generation")
            return None
        
        try:
            # Build comprehensive prompt for image generation
            prompt = self._build_image_prompt(creative_copy, template_config, brand_config, original_message)
            logger.info(f"Generating image with prompt: {prompt[:150]}...")
            
            # Generate image using Gemini 2.0 Flash Preview Image Generation (matching working test)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            # Process the response - extract image from response parts (matching working test structure)
            if response.candidates and len(response.candidates) > 0:
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        # Get image data directly from inline_data
                        image_bytes = part.inline_data.data
                        
                        # Convert to PIL Image to get dimensions and validate format
                        try:
                            pil_image = Image.open(BytesIO(image_bytes))
                            width, height = pil_image.size
                            
                            # Convert to base64 string for UI
                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                            
                            logger.info(f"Successfully generated {width}x{height} image")
                            
                            return {
                                "base64": image_base64,
                                "format": "PNG",
                                "size": f"{width}x{height}",
                                "prompt": prompt
                            }
                        except Exception as img_error:
                            logger.error(f"Error processing generated image: {img_error}")
                            return None
            
            logger.warning("No image generated in Gemini response")
            return None
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None
    
    def _build_image_prompt(
        self, 
        creative_copy: CreativeCopy, 
        template_config: Dict[str, Any], 
        brand_config: Dict[str, Any],
        original_message: str
    ) -> str:
        """Build comprehensive prompt for image generation"""
        
        # Template specifications
        template_format = template_config.get("format", "A4")
        dimensions = template_config.get("dimensions", "8.5x11")
        orientation = template_config.get("orientation", "portrait")
        layout = template_config.get("layout", [])
        
        # Brand specifications
        brand_name = brand_config.get("brand_name", "Partner Brand")
        brand_hex = brand_config.get("brand_hex", "#2E86DE")
        brand_placement = brand_config.get("brand_placement", "top_left")
        brand_initials = brand_config.get("initials", "PB")
        
        # Build prompt parts
        prompt_parts = []
        
        # Base description
        prompt_parts.append(f"Create a professional {template_format} marketing {template_format.lower()} in {orientation} orientation")
        
        # Layout structure
        if "headline" in layout:
            prompt_parts.append(f"with prominent headline text '{creative_copy.headline}'")
        
        if creative_copy.body and ("body" in layout or "subhead" in layout):
            prompt_parts.append(f"body text '{creative_copy.body[:50]}...'")
        
        if "cta" in layout:
            prompt_parts.append(f"call-to-action button '{creative_copy.cta}'")
        
        # Brand elements
        prompt_parts.append(f"include {brand_name} branding in {brand_placement} position")
        prompt_parts.append(f"use brand color {brand_hex} as accent color")
        prompt_parts.append(f"show brand initials '{brand_initials}' in a circle logo placeholder")
        
        # Legal footer
        if "legal_footer" in layout:
            prompt_parts.append(f"small legal text at bottom: '{creative_copy.legal[:30]}...'")
        
        # Context from original message
        if "furniture" in original_message.lower():
            prompt_parts.append("background should suggest furniture/home decor theme")
        elif "dental" in original_message.lower() or "medical" in original_message.lower():
            prompt_parts.append("background should suggest healthcare/medical theme") 
        
        # Style specifications
        prompt_parts.extend([
            "professional financial services design",
            "clean modern layout with good typography",
            "high contrast for readability",
            "marketing poster style",
            "high quality, professional appearance"
        ])
        
        return ", ".join(prompt_parts)
    
    def _validate_template(self, template: str) -> Dict[str, Any]:
        """Validate and return template configuration"""
        if template not in self.templates:
            available = list(self.templates.keys())
            raise ValueError(f"Template '{template}' not found. Available: {available}")
        
        return self.templates[template]
    
    def _redline_pass(self, message: str) -> Tuple[str, List[RedlineChange]]:
        """TrustShield-style redline pass to flag/replace banned phrases"""
        redlined_message = message
        redline_changes = []
        
        try:
            for violation_type, config in self.banned_phrases.items():
                phrases = config.get("phrases", [])
                replacement = config.get("replacement", "[REDLINED]")
                severity = config.get("severity", "medium")
                
                for phrase in phrases:
                    # Case-insensitive search and replace
                    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                    matches = pattern.findall(redlined_message)
                    
                    if matches:
                        for match in matches:
                            redline_changes.append(RedlineChange(
                                original=match,
                                replacement=replacement,
                                reason=f"Banned phrase: {violation_type}",
                                severity=severity
                            ))
                        
                        redlined_message = pattern.sub(replacement, redlined_message)
            
            if redline_changes:
                logger.info(f"Applied {len(redline_changes)} redline changes")
            
            return redlined_message, redline_changes
            
        except Exception as e:
            logger.error(f"Redline pass error: {e}")
            return message, []
    
    def _detect_disclosures(self, message: str) -> List[str]:
        """Detect required disclosures based on message content"""
        required_disclosures = []
        message_lower = message.lower()
        
        try:
            for disclosure_type, config in self.dynamic_disclosures.items():
                triggers = config.get("triggers", [])
                disclosure_id = config.get("disclosure_id", "")
                
                # Check if any trigger phrases are present
                for trigger in triggers:
                    if trigger.lower() in message_lower:
                        if disclosure_id not in required_disclosures:
                            required_disclosures.append(disclosure_id)
                        break
            
            logger.info(f"Detected {len(required_disclosures)} required disclosures")
            return required_disclosures
            
        except Exception as e:
            logger.error(f"Disclosure detection error: {e}")
            return []
    
    def _generate_creative_copy(self, message: str, template_config: Dict[str, Any]) -> CreativeCopy:
        """Generate creative copy based on message and template layout"""
        try:
            layout = template_config.get("layout", [])
            
            # Use LLM to generate structured creative copy
            system_prompt = f"""You are a B2B marketing copywriter. Generate creative copy based on the layout requirements.

Layout elements needed: {', '.join(layout)}
Template format: {template_config.get('format', 'Unknown')}

Generate concise, professional copy with:
- Compelling headline (max 8 words)
- Clear body text if needed (max 25 words)
- Strong call-to-action (max 5 words)
- Appropriate legal language

Return as JSON with keys: headline, body, cta, legal"""
            
            user_message = f"Create marketing copy for: {message}"
            messages = [{"role": "user", "content": user_message}]
            
            response = chat(messages, system=system_prompt)
            
            # Try to parse LLM response as JSON
            try:
                import json
                copy_data = json.loads(response.strip())
                return CreativeCopy(
                    headline=copy_data.get("headline", "Your Financial Solution"),
                    body=copy_data.get("body"),
                    cta=copy_data.get("cta", "Learn More"),
                    legal=copy_data.get("legal", "See terms and conditions.")
                )
            except json.JSONDecodeError:
                # Fallback to basic parsing
                return self._fallback_copy_generation(message)
                
        except Exception as e:
            logger.error(f"Creative copy generation error: {e}")
            return self._fallback_copy_generation(message)
    
    def _fallback_copy_generation(self, message: str) -> CreativeCopy:
        """Fallback creative copy generation"""
        return CreativeCopy(
            headline="Special Financing Available",
            body="Flexible payment options for your purchase",
            cta="Apply Today",
            legal="Subject to credit approval. See terms."
        )
    
    def _apply_co_branding(self, brand: Optional[BrandSettings], template_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply co-branding with brand name and hex colors"""
        default_hex = self.co_brand_settings.get("default_hex", "#2E86DE")
        placeholder_style = self.co_brand_settings.get("placeholder_style", "initials_in_circle")
        
        if brand:
            # Generate initials from brand name
            initials = ''.join([word[0].upper() for word in brand.name.split()[:2]])
            
            return {
                "brand_name": brand.name,
                "brand_hex": brand.hex_color,
                "brand_placement": brand.placement,
                "logo_placeholder": f"{placeholder_style}:{initials}",
                "initials": initials
            }
        else:
            return {
                "brand_name": "Partner Brand",
                "brand_hex": default_hex,
                "brand_placement": template_config.get("brand_placement", "top_left"),
                "logo_placeholder": f"{placeholder_style}:PB",
                "initials": "PB"
            }
    
    def _create_ui_cards(
        self, 
        template: str, 
        creative_copy: CreativeCopy, 
        brand_config: Dict[str, Any],
        image_data: Optional[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Create preview reference and UI cards for creative output with actual image"""
        preview_ref = f"creative_preview_{template}_{hash(creative_copy.headline) % 10000}"
        
        ui_card = {
            "type": "creative",
            "template": template,
            "preview_ref": preview_ref,
            "copy": {
                "headline": creative_copy.headline,
                "body": creative_copy.body,
                "cta": creative_copy.cta,
                "legal": creative_copy.legal
            },
            "brand": brand_config,
            "dimensions": self.templates[template].get("dimensions", "Unknown")
        }
        
        # Add actual image data if generated
        if image_data:
            ui_card["image"] = {
                "base64": image_data["base64"],
                "format": image_data["format"],
                "size": image_data["size"],
                "data_uri": f"data:image/{image_data['format'].lower()};base64,{image_data['base64']}"
            }
        
        ui_cards = [ui_card]
        
        return preview_ref, ui_cards
    
    def _detect_handoffs(self, original_message: str, redline_changes: List[RedlineChange]) -> List[str]:
        """Detect handoffs to other agents"""
        handoffs = []
        message_lower = original_message.lower()
        
        # Handoff to narrator for campaign measurement
        if any(word in message_lower for word in ['campaign', 'measure', 'analytics', 'performance', 'roi']):
            handoffs.append("narrator")
        
        # Handoff back to imagegen for variants
        if any(word in message_lower for word in ['variant', 'alternative', 'different', 'another']):
            handoffs.append("imagegen")
        
        # If high severity redlines, might need manual review
        high_severity_redlines = [change for change in redline_changes if change.severity == "high"]
        if high_severity_redlines:
            handoffs.append("manual_review")
        
        return handoffs
    
    def _generate_response_summary(
        self, 
        creative_copy: CreativeCopy, 
        redline_changes: List[RedlineChange],
        required_disclosures: List[str],
        image_data: Optional[Dict[str, Any]]
    ) -> str:
        """Generate brief creative summary with legal footer and image status"""
        
        summary_parts = []
        
        # Creative summary
        summary_parts.append(f"**Creative Generated:** {creative_copy.headline}")
        
        if creative_copy.body:
            summary_parts.append(f"Body copy focuses on {creative_copy.body[:50]}...")
        
        summary_parts.append(f"Call-to-action: '{creative_copy.cta}'")
        
        # Image generation status
        if image_data:
            summary_parts.append(f"🖼️ **Image generated:** {image_data['size']} {image_data['format']}")
        else:
            summary_parts.append("⚠️ **Image generation unavailable** (API key required)")
        
        # Compliance notes
        if redline_changes:
            high_severity = len([c for c in redline_changes if c.severity == "high"])
            if high_severity > 0:
                summary_parts.append(f"⚠️ **{high_severity} high-priority redlines applied** for compliance")
            else:
                summary_parts.append(f"📝 **{len(redline_changes)} redlines applied** for compliance")
        
        if required_disclosures:
            summary_parts.append(f"📋 **{len(required_disclosures)} disclosures** automatically added")
        
        # Legal footer from first disclosure or default
        legal_footer = ""
        if required_disclosures:
            first_disclosure = required_disclosures[0]
            legal_footer = self.legal_footers.get(first_disclosure, "")
        
        if not legal_footer:
            legal_footer = "Subject to credit approval. See terms and conditions."
        
        response_text = "\n".join(summary_parts)
        response_text += f"\n\n**Legal Footer:** {legal_footer}"
        
        return response_text
    
    def _error_response(self, message: str) -> ImageGenResponse:
        """Create error response"""
        return ImageGenResponse(
            response=f"**Creative Generation Error:** {message}",
            metadata={
                "ui_cards": [],
                "disclosures": [],
                "handoffs": [],
                "error": message,
                "image_generated": False
            }
        )

# Test cases for ImageGen B2B marketing studio with real image generation
def test_imagegen():
    """Test ImageGen with B2B marketing studio and image generation"""
    print("🧪 Testing ImageGen B2B Marketing Studio with Real Image Generation")
    print("=" * 70)
    
    imagegen = ImageGenAgent()
    
    test_cases = [
        {
            "name": "Equal payment promotional creative with image",
            "message": "Get equal monthly payments on your furniture purchase with no money down!",
            "template": "a4_instore",
            "brand": BrandSettings(name="Urban Living Co", hex_color="#FF6B35"),
            "expected_disclosures": ["equal_payment_generic"],
            "expected_redlines": 0,
            "expect_image": True
        },
        {
            "name": "Banned phrase redlining with image",
            "message": "Guaranteed approval for everyone! No credit check required!",
            "template": "social_square", 
            "brand": BrandSettings(name="FastCash", hex_color="#E74C3C"),
            "expected_disclosures": [],
            "expected_redlines": 2,
            "expect_image": True
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        try:
            print(f"{i}. {case['name']}")
            
            result = imagegen.generate_creative(
                message=case["message"],
                template=case["template"],
                brand=case["brand"]
            )
            
            # Validate response structure
            valid_structure = (
                hasattr(result, 'response') and
                hasattr(result, 'metadata') and
                'ui_cards' in result.metadata and
                'disclosures' in result.metadata
            )
            
            # Check disclosures
            disclosures = result.metadata.get('disclosures', [])
            disclosures_ok = len(disclosures) >= len(case['expected_disclosures'])
            
            # Check redlines
            redline_changes = result.metadata.get('redline_changes', [])
            redlines_ok = len(redline_changes) == case['expected_redlines']
            
            # Check UI cards
            ui_cards = result.metadata.get('ui_cards', [])
            ui_cards_ok = len(ui_cards) > 0 and ui_cards[0].get('type') == 'creative'
            
            # Check image generation
            image_generated = result.metadata.get('image_generated', False)
            image_ok = image_generated or not case['expect_image']  # OK if image generated OR not expected
            
            success = valid_structure and disclosures_ok and redlines_ok and ui_cards_ok and image_ok
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Template: {case['template']}")
            print(f"   Brand: {case['brand'].name} ({case['brand'].hex_color})")
            print(f"   Disclosures: {len(disclosures)} (expected >= {len(case['expected_disclosures'])})")
            print(f"   Redlines: {len(redline_changes)} (expected {case['expected_redlines']})")
            print(f"   UI Cards: {len(ui_cards)}")
            print(f"   Image Generated: {image_generated}")
            print(f"   Status: {status}")
            
            if success:
                passed += 1
            else:
                print(f"   Failure reasons:")
                if not valid_structure:
                    print(f"     - Invalid response structure")
                if not disclosures_ok:
                    print(f"     - Disclosure count mismatch")
                if not redlines_ok:
                    print(f"     - Redline count mismatch")
                if not ui_cards_ok:
                    print(f"     - UI cards issue")
                if not image_ok:
                    print(f"     - Image generation issue")
            
            print()
            
        except Exception as e:
            print(f"   ❌ FAIL - Exception: {str(e)}")
            print()
    
    print(f"📊 Results: {passed}/{total} tests passed")
    return passed == total

if __name__ == "__main__":
    test_imagegen()