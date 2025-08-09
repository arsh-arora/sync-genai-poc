#!/usr/bin/env python3
"""
Debug script specifically for TrustShield Gemini integration
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add app to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gemini_chat_function():
    """Test the exact chat function used by TrustShield"""
    print("🔍 Testing app.llm.gemini.chat function")
    print("=" * 45)
    
    try:
        from app.llm.gemini import chat
        
        # Exact same call as TrustShield
        system_prompt = """You are a fraud detection specialist. Analyze the following text and classify it into one of these categories:

Categories:
- refund_scam: Attempts to trick users into paying for fake refunds
- account_takeover: Attempts to gain unauthorized access to accounts  
- pii_risk: Requests for sensitive personal information
- safe: Normal, legitimate communication

Respond with ONLY a JSON object:
{"category": "category_name", "confidence": 0.85, "reasoning": "brief explanation"}

Focus on detecting sophisticated social engineering attempts."""

        user_message = "Analyze this text: 'Hi'"
        messages = [{"role": "user", "content": user_message}]
        
        print("📤 Sending request to Gemini...")
        print(f"   System prompt length: {len(system_prompt)}")
        print(f"   User message: '{user_message}'")
        
        response = chat(messages, system=system_prompt)
        
        print(f"📥 Raw response from chat function:")
        print(f"   Type: {type(response)}")
        print(f"   Length: {len(response) if response else 0}")
        print(f"   Content: '{response}'")
        
        if response:
            # Try parsing as JSON like TrustShield does
            try:
                result = json.loads(response.strip())
                print(f"✅ JSON parsing successful: {result}")
                return True
            except json.JSONDecodeError as je:
                print(f"❌ JSON parsing failed: {je}")
                print(f"   Raw response to parse: '{response.strip()}'")
                return False
        else:
            print("❌ Empty response from chat function")
            return False
            
    except Exception as e:
        print(f"❌ Error testing chat function: {e}")
        return False

def test_direct_gemini_client():
    """Test direct Gemini client for comparison"""
    print("\n🔍 Testing Direct Gemini Client")
    print("=" * 35)
    
    try:
        from google import genai
        
        api_key = os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)
        
        prompt = """You are a fraud detection specialist. Respond with ONLY a JSON object:
{"category": "safe", "confidence": 0.95, "reasoning": "normal greeting"}

Analyze this text: 'Hi'"""
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        print(f"📥 Direct client response:")
        print(f"   Type: {type(response)}")
        print(f"   Has text: {hasattr(response, 'text')}")
        if hasattr(response, 'text'):
            print(f"   Text length: {len(response.text) if response.text else 0}")
            print(f"   Content: '{response.text}'")
        
        return response and hasattr(response, 'text') and response.text
        
    except Exception as e:
        print(f"❌ Error with direct client: {e}")
        return False

def test_simple_chat():
    """Test with very simple input"""
    print("\n🔍 Testing Simple Chat")
    print("=" * 25)
    
    try:
        from app.llm.gemini import chat
        
        messages = [{"role": "user", "content": "Say hello"}]
        response = chat(messages, system="Reply with just 'Hello'")
        
        print(f"📥 Simple response: '{response}'")
        return bool(response and len(response) > 0)
        
    except Exception as e:
        print(f"❌ Simple chat error: {e}")
        return False

def test_with_debug_logging():
    """Test with debug logging enabled"""
    print("\n🔍 Testing with Debug Logging")
    print("=" * 32)
    
    import logging
    
    # Enable debug logging
    logging.basicConfig(level=logging.DEBUG)
    gemini_logger = logging.getLogger('app.llm.gemini')
    gemini_logger.setLevel(logging.DEBUG)
    
    try:
        from app.llm.gemini import chat
        
        messages = [{"role": "user", "content": "Test"}]
        response = chat(messages, system="Respond with: OK")
        
        print(f"📥 Debug response: '{response}'")
        return bool(response)
        
    except Exception as e:
        print(f"❌ Debug test error: {e}")
        return False

def main():
    """Run all TrustShield debug tests"""
    print("🛡️ TrustShield Gemini Debug Suite")
    print("=" * 35)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment")
        return
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    tests = [
        ("Simple Chat Test", test_simple_chat),
        ("Direct Gemini Client", test_direct_gemini_client),
        ("TrustShield Chat Function", test_gemini_chat_function),
        ("Debug Logging Test", test_with_debug_logging),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 {test_name}")
        print('='*50)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Debug Summary")
    print('='*50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if not results.get("Simple Chat Test"):
        print("   - Basic chat function is broken - check app.llm.gemini")
    
    if results.get("Direct Gemini Client") and not results.get("TrustShield Chat Function"):
        print("   - Direct API works but chat wrapper fails - fix chat function")
    
    if not any(results.values()):
        print("   - All tests failed - likely API quota/auth issue")
    
    print(f"\n🎯 Next Steps:")
    print("   1. Fix the chat function in app/llm/gemini.py")
    print("   2. Add better error handling to TrustShield")
    print("   3. Consider fallback responses when Gemini fails")

if __name__ == "__main__":
    main()