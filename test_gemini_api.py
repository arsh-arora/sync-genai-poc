#!/usr/bin/env python3
"""
Test script for Google Gemini API to verify models and functionality
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

def test_gemini_models():
    """Test different Gemini models to see which ones work"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment")
        return False
    
    # Initialize modern Gemini client
    client = genai.Client(api_key=api_key)
    
    # Models to test
    models_to_test = [
        "gemini-2.5-flash",
        "gemini-2.5-flash", 
        "gemini-1.5-flash", 
        "gemini-1.5-pro",
        "gemini-2.0-flash-exp",
    ]
    
    working_models = []
    
    print("🧪 Testing Gemini Chat Models")
    print("=" * 50)
    
    for model_name in models_to_test:
        print(f"\n🔍 Testing: {model_name}")
        try:
            # Simple test message using modern API
            response = client.models.generate_content(
                model=model_name,
                contents="Say 'Hello World' in JSON format"
            )
            
            if response and response.text:
                print(f"   ✅ SUCCESS: {model_name}")
                print(f"   📝 Response: {response.text[:100]}...")
                working_models.append(model_name)
            else:
                print(f"   ❌ FAILED: Empty response from {model_name}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {model_name} - {str(e)}")
            # Check if it's a quota error
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"   ⏰ Rate limit hit - waiting 10 seconds...")
                time.sleep(10)
        
        # Small delay between tests
        time.sleep(2)
    
    print(f"\n📊 Results: {len(working_models)} working models found")
    for model in working_models:
        print(f"   ✅ {model}")
    
    return working_models

def test_gemini_embeddings():
    """Test Gemini embedding models"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment")
        return False
    
    client = genai.Client(api_key=api_key)
    
    embedding_models = [
        "text-embedding-004",
        "embedding-001"
    ]
    
    working_embedding_models = []
    
    print("\n🧪 Testing Gemini Embedding Models")
    print("=" * 50)
    
    for model_name in embedding_models:
        print(f"\n🔍 Testing: {model_name}")
        try:
            # Test embedding using modern API
            result = client.models.embed_content(
                model=model_name,
                content="This is a test sentence for embedding"
            )
            
            if result and hasattr(result, 'embedding'):
                embedding_dim = len(result.embedding.values)
                print(f"   ✅ SUCCESS: {model_name} (dim: {embedding_dim})")
                working_embedding_models.append(model_name)
            else:
                print(f"   ❌ FAILED: No embedding returned from {model_name}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {model_name} - {str(e)}")
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"   ⏰ Rate limit hit - waiting 10 seconds...")
                time.sleep(10)
        
        time.sleep(2)
    
    print(f"\n📊 Embedding Results: {len(working_embedding_models)} working models found")
    for model in working_embedding_models:
        print(f"   ✅ {model}")
    
    return working_embedding_models

def test_api_quotas():
    """Test current API quota status"""
    print("\n🔍 Testing API Quota Status")
    print("=" * 30)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    
    try:
        # Try a minimal request
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hi"
        )
        
        if response.text:
            print("   ✅ API is responding normally")
            print(f"   📝 Response: {response.text}")
            return True
        else:
            print("   ⚠️ API responded but with empty content")
            return False
            
    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ API Error: {error_msg}")
        
        if "429" in error_msg:
            print("   📊 Rate limit information:")
            if "quota_metric" in error_msg:
                print("   - Hit multiple quota limits")
                print("   - Suggestions:")
                print("     1. Wait 1-24 hours for quota reset")
                print("     2. Upgrade to paid tier")
                print("     3. Use more efficient models (flash vs pro)")
        
        return False

def test_json_parsing():
    """Test JSON response parsing"""
    print("\n🔍 Testing JSON Response Parsing")
    print("=" * 35)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    
    try:
        # Request specifically formatted JSON
        prompt = '''Return exactly this JSON and nothing else:
{"status": "working", "message": "test successful"}'''
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        if response.text:
            print(f"   📝 Raw response: '{response.text}'")
            
            # Try to parse as JSON
            try:
                parsed = json.loads(response.text.strip())
                print(f"   ✅ JSON parsing successful: {parsed}")
                return True
            except json.JSONDecodeError as je:
                print(f"   ❌ JSON parsing failed: {je}")
                print(f"   🔍 Response length: {len(response.text)}")
                print(f"   🔍 First 100 chars: '{response.text[:100]}'")
                return False
        else:
            print("   ❌ Empty response - this is the root cause!")
            return False
            
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
        return False

def generate_recommendations():
    """Generate recommendations based on test results"""
    print("\n💡 Recommendations")
    print("=" * 20)
    
    print("Based on the test results:")
    print("1. If all tests fail with 429 errors:")
    print("   - Wait 24 hours for quota reset")
    print("   - Consider upgrading to paid tier")
    print("")
    print("2. If some models work:")
    print("   - Update your app to use working models")
    print("   - Prefer 'flash' models for better quotas")
    print("")
    print("3. If JSON parsing fails:")
    print("   - Add response validation in your code")
    print("   - Handle empty responses gracefully")
    print("")
    print("4. Quick fixes for your app:")
    print("   - Add retry logic with exponential backoff")
    print("   - Switch to gemini-2.5-flash for better limits")
    print("   - Add fallback responses when API fails")

def main():
    """Run all tests"""
    print("🚀 Gemini API Test Suite")
    print("=" * 25)
    
    # Test quota status first
    quota_ok = test_api_quotas()
    
    if quota_ok:
        # Test chat models
        working_chat_models = test_gemini_models()
        
        # Test embeddings  
        working_embedding_models = test_gemini_embeddings()
        
        # Test JSON parsing
        json_ok = test_json_parsing()
        
        # Generate config recommendations
        if working_chat_models:
            print(f"\n🔧 Recommended chat model: {working_chat_models[0]}")
        if working_embedding_models:
            print(f"🔧 Recommended embedding model: {working_embedding_models[0]}")
    
    # Always show recommendations
    generate_recommendations()
    
    print(f"\n✅ Test complete! Check results above.")

if __name__ == "__main__":
    main()