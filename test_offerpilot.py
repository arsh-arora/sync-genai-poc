#!/usr/bin/env python3
"""
Test script for OfferPilot agent
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.agents.offerpilot import OfferPilot, test_offerpilot

def test_offerpilot_tools():
    """Test individual OfferPilot tools"""
    print("🔧 Testing OfferPilot Tools")
    print("=" * 50)
    
    pilot = OfferPilot()
    
    # Test marketplace search
    print("1. Testing marketplace search...")
    search_results = pilot.marketplace_search("laptop", max_results=5)
    print(f"   Found {len(search_results)} products for 'laptop'")
    if search_results:
        print(f"   Top result: {search_results[0]['title']} - ${search_results[0]['price']}")
    
    # Test offers lookup
    print("\n2. Testing offers lookup...")
    offers = pilot.offers_lookup("Apple Store", 1199.99)
    print(f"   Found {len(offers)} offers for Apple Store purchase of $1199.99")
    if offers:
        print(f"   Best offer: {offers[0]['name']} - {offers[0]['apr']}% APR")
    
    # Test payment simulation
    print("\n3. Testing payment simulation...")
    payment = pilot.payments_simulate(1199.99, 18, 0.0)
    print(f"   $1199.99 over 18 months at 0% APR:")
    print(f"   Monthly: ${payment['monthly']}")
    print(f"   Total: ${payment['total_cost']}")
    
    # Test credit screening
    print("\n4. Testing credit screening...")
    from app.agents.offerpilot import UserStub
    user = UserStub(credit_score=750, income=80000)
    prequal = pilot.credit_quickscreen(user)
    print(f"   Credit score 750, Income $80k:")
    print(f"   Eligible: {prequal['eligible']}")
    print(f"   Reason: {prequal['reason']}")
    
    print("\n✅ Tool tests completed")

def test_offerpilot_integration():
    """Test OfferPilot integration scenarios"""
    print("\n🔗 Testing OfferPilot Integration")
    print("=" * 50)
    
    pilot = OfferPilot()
    
    test_scenarios = [
        {
            "name": "Budget-conscious furniture shopping",
            "query": "desk for home office",
            "budget": 400.0,
            "description": "Should find affordable desk options with financing"
        },
        {
            "name": "Premium electronics purchase",
            "query": "MacBook laptop",
            "budget": None,
            "description": "Should find Apple products with promotional financing"
        },
        {
            "name": "Healthcare financing",
            "query": "dental crown treatment",
            "budget": None,
            "description": "Should find dental services with CareCredit options"
        },
        {
            "name": "Veterinary care",
            "query": "pet surgery",
            "budget": 3000.0,
            "description": "Should find vet services with CareCredit financing"
        }
    ]
    
    passed = 0
    total = len(test_scenarios)
    
    for i, scenario in enumerate(test_scenarios, 1):
        try:
            print(f"{i}. {scenario['name']}")
            print(f"   {scenario['description']}")
            
            result = pilot.process_query(scenario["query"], scenario["budget"])
            
            # Validate response structure
            items_count = len(result.items)
            has_offers = any(len(item.offers) > 0 for item in result.items)
            prequal_valid = isinstance(result.prequal.eligible, bool)
            
            # Check for appropriate financing types
            financing_types = set()
            for item in result.items:
                for offer in item.offers:
                    if "CARECREDIT" in offer.id:
                        financing_types.add("CareCredit")
                    elif "SYNC" in offer.id:
                        financing_types.add("Synchrony")
            
            success = items_count > 0 and prequal_valid
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Items: {items_count}, Financing: {has_offers}")
            print(f"   Pre-qualified: {result.prequal.eligible}")
            print(f"   Financing types: {', '.join(financing_types) if financing_types else 'None'}")
            print(f"   Citations: {len(result.citations)}")
            print(f"   Status: {status}")
            print()
            
            if success:
                passed += 1
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            print()
    
    print(f"Integration Test Results: {passed}/{total} passed")
    return passed == total

def test_offerpilot_edge_cases():
    """Test OfferPilot edge cases and error handling"""
    print("\n⚠️  Testing OfferPilot Edge Cases")
    print("=" * 50)
    
    pilot = OfferPilot()
    
    edge_cases = [
        {
            "name": "Empty query",
            "query": "",
            "budget": None,
            "should_handle": True
        },
        {
            "name": "No matching products",
            "query": "quantum computer",
            "budget": None,
            "should_handle": True
        },
        {
            "name": "Very low budget",
            "query": "laptop",
            "budget": 50.0,
            "should_handle": True
        },
        {
            "name": "Very high budget",
            "query": "desk",
            "budget": 10000.0,
            "should_handle": True
        }
    ]
    
    passed = 0
    total = len(edge_cases)
    
    for i, case in enumerate(edge_cases, 1):
        try:
            print(f"{i}. {case['name']}")
            
            result = pilot.process_query(case["query"], case["budget"])
            
            # Should always return valid response structure
            valid_structure = (
                hasattr(result, 'items') and
                hasattr(result, 'prequal') and
                hasattr(result, 'citations') and
                isinstance(result.items, list) and
                isinstance(result.prequal.eligible, bool)
            )
            
            success = valid_structure
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Query: '{case['query']}', Budget: {case['budget']}")
            print(f"   Valid structure: {valid_structure}")
            print(f"   Items returned: {len(result.items)}")
            print(f"   Status: {status}")
            print()
            
            if success:
                passed += 1
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            print()
    
    print(f"Edge Case Test Results: {passed}/{total} passed")
    return passed == total

def main():
    """Run all OfferPilot tests"""
    print("🛒 OfferPilot Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test individual tools
    test_offerpilot_tools()
    
    print("\n" + "="*60)
    
    # Test golden paths
    print("🌟 Testing Golden Paths...")
    golden_success = test_offerpilot()
    results.append(("Golden Paths", 1 if golden_success else 0, 1))
    
    print("="*60)
    
    # Test integration scenarios
    integration_success = test_offerpilot_integration()
    results.append(("Integration", 1 if integration_success else 0, 1))
    
    print("="*60)
    
    # Test edge cases
    edge_success = test_offerpilot_edge_cases()
    results.append(("Edge Cases", 1 if edge_success else 0, 1))
    
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
        print("\n🎉 All OfferPilot tests passed! Ready for marketplace discovery.")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Please review the issues above.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    success = main()
    sys.exit(0 if success else 1)
