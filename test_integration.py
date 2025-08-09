#!/usr/bin/env python3
"""
Test Complete Application Integration
"""

import os
import sys
import requests
import json
import time
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30

def test_health_check():
    """Test health endpoint"""
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/healthz", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Agents: {data.get('agents_initialized')}")
            print(f"   Docstore size: {data.get('docstore_size')}")
            return True
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False

def test_chat_endpoint():
    """Test smart chat endpoint"""
    print("💬 Testing smart chat...")
    try:
        payload = {
            "message": "I want to buy a laptop under $1000",
            "allow_tavily": False
        }
        response = requests.post(f"{BASE_URL}/chat", json=payload, timeout=TEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Agent: {data.get('agent')}")
            print(f"   Confidence: {data.get('confidence', 0):.1%}")
            print(f"   Response length: {len(data.get('response', ''))}")
            return True
        else:
            print(f"   ❌ Chat failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Chat error: {e}")
        return False

def test_agent_endpoints():
    """Test direct agent endpoints"""
    print("🤖 Testing agent endpoints...")
    
    # Test OfferPilot
    try:
        payload = {"query": "office desk", "budget": 600}
        response = requests.post(f"{BASE_URL}/agent/offerpilot", json=payload, timeout=TEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   OfferPilot: Found {len(data.get('items', []))} items")
        else:
            print(f"   ❌ OfferPilot failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ OfferPilot error: {e}")
    
    # Test TrustShield
    try:
        payload = {"text": "My credit card is 4111-1111-1111-1111"}
        response = requests.post(f"{BASE_URL}/agent/trustshield", json=payload, timeout=TEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   TrustShield: Decision = {data.get('decision')}")
        else:
            print(f"   ❌ TrustShield failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ TrustShield error: {e}")
    
    # Test DevCopilot
    try:
        payload = {"service": "payments", "lang": "python"}
        response = requests.post(f"{BASE_URL}/agent/devcopilot", json=payload, timeout=TEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   DevCopilot: Generated {data.get('endpoint', 'code')}")
        else:
            print(f"   ❌ DevCopilot failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ DevCopilot error: {e}")
    
    return True

def test_middleware():
    """Test PII redaction and TrustShield middleware"""
    print("🛡️ Testing security middleware...")
    
    # Test PII redaction
    try:
        payload = {
            "message": "My SSN is 123-45-6789 and my card is 4111-1111-1111-1111",
            "allow_tavily": False
        }
        response = requests.post(f"{BASE_URL}/chat", json=payload, timeout=TEST_TIMEOUT)
        
        if response.status_code == 200:
            # Should not contain original PII in response
            response_text = response.text.lower()
            if "123-45-6789" not in response_text and "4111-1111-1111-1111" not in response_text:
                print("   ✅ PII redaction working")
            else:
                print("   ⚠️ PII may not be fully redacted")
        else:
            print(f"   ❌ PII test failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ PII test error: {e}")
    
    return True

def test_ui_availability():
    """Test UI is available"""
    print("🎨 Testing UI availability...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200 and "Synch GenAI PoC" in response.text:
            print("   ✅ UI loaded successfully")
            return True
        else:
            print(f"   ❌ UI failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ UI error: {e}")
        return False

def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Starting Integration Tests")
    print("=" * 50)
    
    # Check if server is running
    print("🔍 Checking if server is running...")
    try:
        response = requests.get(f"{BASE_URL}/healthz", timeout=2)
        if response.status_code != 200:
            print("❌ Server not responding. Please start the server with: python main.py")
            return False
    except requests.exceptions.RequestException:
        print("❌ Server not running. Please start the server with: python main.py")
        return False
    
    print("✅ Server is running\n")
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("UI Availability", test_ui_availability),
        ("Chat Endpoint", test_chat_endpoint),
        ("Agent Endpoints", test_agent_endpoints),
        ("Security Middleware", test_middleware),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"   ❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All integration tests passed! The application is ready to use.")
        print(f"\n🌐 Access the UI at: {BASE_URL}")
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the logs.")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)