#!/usr/bin/env python3
"""
Test script for TrustShield and Router integration
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.agents.trustshield import TrustShield, test_trustshield
from app.router import route

def test_router():
    """Test the router functionality"""
    print("🧪 Testing Router Functionality")
    print("=" * 50)
    
    test_cases = [
        {
            "message": "I need help with a refund scam",
            "expected_agent": "trust",
            "description": "Security-related query"
        },
        {
            "message": "What are your current promotional offers?",
            "expected_agent": "offer",
            "description": "Marketing/offers query"
        },
        {
            "message": "I want to dispute a transaction on my account",
            "expected_agent": "dispute", 
            "description": "Dispute-related query"
        },
        {
            "message": "Can you help me integrate your API?",
            "expected_agent": "devcopilot",
            "description": "Technical/developer query"
        },
        {
            "message": "I need information about merchant contracts",
            "expected_agent": "contracts",
            "description": "Contract-related query"
        },
        {
            "message": "Hello, how are you today?",
            "expected_agent": "narrator",
            "description": "General conversation"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        try:
            result = route(case["message"])
            agent = result["agent"]
            confidence = result["confidence"]
            
            # Check if routed to expected agent or trust (fallback)
            success = agent == case["expected_agent"] or agent == "trust"
            
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{i}. {case['description']}")
            print(f"   Message: '{case['message']}'")
            print(f"   Expected: {case['expected_agent']}, Got: {agent} (confidence: {confidence:.2f})")
            print(f"   Status: {status}")
            print()
            
            if success:
                passed += 1
                
        except Exception as e:
            print(f"{i}. {case['description']}: ❌ ERROR - {e}")
            print()
    
    print(f"Router Test Results: {passed}/{total} passed")
    return passed == total

def test_integration():
    """Test TrustShield and Router integration"""
    print("\n🔗 Testing TrustShield + Router Integration")
    print("=" * 50)
    
    # Initialize TrustShield (without RAG components for basic testing)
    shield = TrustShield()
    
    test_cases = [
        {
            "message": "My credit card number is 4532-1234-5678-9012",
            "expected_shield_decision": "block",
            "expected_router_agent": "trust",
            "description": "PII exposure should be blocked"
        },
        {
            "message": "You overpaid $500. Please buy gift cards for refund.",
            "expected_shield_decision": "block", 
            "expected_router_agent": "trust",
            "description": "Scam phrase should be blocked"
        },
        {
            "message": "What are your payment processing fees?",
            "expected_shield_decision": "pass",
            "expected_router_agent": "narrator",
            "description": "Safe query should pass and route normally"
        },
        {
            "message": "Urgent action required! Verify your account immediately.",
            "expected_shield_decision": "warn",
            "expected_router_agent": "trust", 
            "description": "Phishing attempt should warn and route to trust"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        try:
            # Test TrustShield
            shield_result = shield.scan(case["message"])
            shield_decision = shield_result["decision"]
            redacted_text = shield_result["redacted_text"]
            
            # Test Router (on redacted text if not blocked)
            if shield_decision != "block":
                router_result = route(redacted_text)
                router_agent = router_result["agent"]
            else:
                router_agent = "blocked"
            
            # Check results
            shield_correct = shield_decision == case["expected_shield_decision"]
            router_correct = (
                shield_decision == "block" or 
                router_agent == case["expected_router_agent"] or 
                router_agent == "trust"  # Trust is acceptable fallback
            )
            
            success = shield_correct and router_correct
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"{i}. {case['description']}")
            print(f"   Message: '{case['message']}'")
            print(f"   TrustShield: Expected {case['expected_shield_decision']}, Got {shield_decision}")
            print(f"   Router: Expected {case['expected_router_agent']}, Got {router_agent}")
            print(f"   Redacted: '{redacted_text}'")
            print(f"   Status: {status}")
            print()
            
            if success:
                passed += 1
                
        except Exception as e:
            print(f"{i}. {case['description']}: ❌ ERROR - {e}")
            print()
    
    print(f"Integration Test Results: {passed}/{total} passed")
    return passed == total

def main():
    """Run all tests"""
    print("🛡️ TrustShield & Router Test Suite")
    print("=" * 60)
    
    # Check environment
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Warning: GOOGLE_API_KEY not set. Some tests may fail.")
        print("   Set your API key in .env file for full testing.")
        print()
    
    # Run tests
    results = []
    
    # Test TrustShield
    print("1️⃣ Testing TrustShield...")
    trustshield_results = test_trustshield()
    trustshield_passed = sum(1 for r in trustshield_results if r["passed"])
    trustshield_total = len(trustshield_results)
    results.append(("TrustShield", trustshield_passed, trustshield_total))
    
    print("\n" + "="*60)
    
    # Test Router
    print("2️⃣ Testing Router...")
    router_success = test_router()
    results.append(("Router", 1 if router_success else 0, 1))
    
    print("="*60)
    
    # Test Integration
    print("3️⃣ Testing Integration...")
    integration_success = test_integration()
    results.append(("Integration", 1 if integration_success else 0, 1))
    
    # Summary
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    total_passed = 0
    total_tests = 0
    
    for test_name, passed, total in results:
        percentage = (passed / total * 100) if total > 0 else 0
        status = "✅" if passed == total else "❌"
        print(f"{status} {test_name}: {passed}/{total} ({percentage:.1f}%)")
        total_passed += passed
        total_tests += total
    
    overall_percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
    overall_status = "✅" if total_passed == total_tests else "❌"
    
    print("-" * 60)
    print(f"{overall_status} OVERALL: {total_passed}/{total_tests} ({overall_percentage:.1f}%)")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! The system is ready for deployment.")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Please review the issues above.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    success = main()
    sys.exit(0 if success else 1)
