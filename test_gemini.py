#!/usr/bin/env python3
"""
Simple test script to check Gemini API connectivity
"""

import os
from dotenv import load_dotenv
from app.llm.gemini import chat

def test_gemini_api():
    """Test basic Gemini API functionality"""
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment")
        return False
        
    print(f"🔑 Using API key: {api_key[:10]}...")
    
    try:
        # Test basic chat functionality
        print("🧪 Testing basic chat...")
        messages = [{"role": "user", "content": "Say 'Hello, API test successful!'"}]
        response = chat(messages)
        
        print(f"✅ Basic chat response: {response}")
        
        # Test JSON response parsing
        print("\n🧪 Testing JSON response...")
        json_messages = [{"role": "user", "content": "Return this as JSON: {'test': 'success', 'status': 'working'}"}]
        json_response = chat(json_messages, system="You must return valid JSON only, no other text.")
        
        print(f"✅ JSON response: {json_response}")
        
        # Test persona detection format
        print("\n🧪 Testing persona detection format...")
        persona_messages = [{"role": "user", "content": "I need help with my credit card payment plan"}]
        persona_system = """You are a persona detection system. Respond with ONLY a JSON object:
{"persona": "consumer", "confidence": 0.85, "reasoning": "mentions personal credit card"}"""
        
        persona_response = chat(persona_messages, system=persona_system)
        print(f"✅ Persona detection response: {persona_response}")
        
        # Try to parse as JSON
        import json
        import re
        
        def extract_json(text: str) -> str:
            """Extract JSON from LLM response that may contain markdown formatting"""
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
            return '{"persona": "consumer", "confidence": 0.3, "reasoning": "Failed to parse response"}'
        
        try:
            extracted = extract_json(persona_response)
            print(f"🔧 Extracted JSON: {extracted}")
            
            parsed = json.loads(extracted)
            print(f"✅ JSON parsing successful: {parsed}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"Raw response: {repr(persona_response)}")
            
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Gemini API connectivity...\n")
    success = test_gemini_api()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")