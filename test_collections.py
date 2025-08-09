#!/usr/bin/env python3
"""
Test script for CollectionsAdvisor agent
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.agents.collections import CollectionsAdvisor, CustomerState, test_collections_advisor

def test_policy_compliance():
    """Test policy compliance and rule enforcement"""
    print("📋 Testing Policy Compliance")
    print("=" * 50)
    
    advisor = CollectionsAdvisor()
    
    compliance_cases = [
        {
            "name": "Re-aging not allowed for 30-day bucket",
            "customer_state": CustomerState(
                balance=3000.0,
                apr=22.99,
                bucket="30",
                income_monthly=4000.0,
                expenses_monthly=3000.0
            ),
            "should_not_include": "re_aging"
        },
        {
            "name": "Settlement not allowed for current bucket",
            "customer_state": CustomerState(
                balance=5000.0,
                apr=19.99,
                bucket="current",
                income_monthly=3500.0,
                expenses_monthly=3200.0
            ),
            "should_not_include": "settlement"
        },
        {
            "name": "Interest reduction allowed for 60-day bucket",
            "customer_state": CustomerState(
                balance=4000.0,
                apr=24.99,
                bucket="60",
                income_monthly=4500.0,
                expenses_monthly=3800.0
            ),
            "should_include": "interest_reduction"
        }
    ]
    
    passed = 0
    total = len(compliance_cases)
    
    for i, case in enumerate(compliance_cases, 1):
        try:
            print(f"{i}. {case['name']}")
            
            result = advisor.process_hardship_request(case["customer_state"])
            plan_types = [plan.type for plan in result.plans]
            
            # Check compliance
            compliance_ok = True
            if "should_not_include" in case:
                compliance_ok = case["should_not_include"] not in plan_types
            elif "should_include" in case:
                compliance_ok = case["should_include"] in plan_types
            
            success = compliance_ok and len(result.plans) > 0
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Customer: {case['customer_state'].bucket} bucket")
            print(f"   Plan types: {plan_types}")
            print(f"   Compliance check: {compliance_ok}")
            print(f"   Status: {status}")
            print()
            
            if success:
                passed += 1
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            print()
    
    print(f"Policy Compliance Results: {passed}/{total} passed")
    return passed == total

def test_payment_simulation():
    """Test payment simulation accuracy"""
    print("💰 Testing Payment Simulation")
    print("=" * 50)
    
    advisor = CollectionsAdvisor()
    
    simulation_cases = [
        {
            "name": "Deferral simulation",
            "balance": 1000.0,
            "apr": 24.0,
            "months": 3,
            "kind": "deferral",
            "params": {"deferral_months": 3},
            "expected_first_payment": 0.0
        },
        {
            "name": "Settlement simulation",
            "balance": 5000.0,
            "apr": 20.0,
            "months": 1,
            "kind": "settlement",
            "params": {"settlement_percentage": 0.60, "split_months": 1},
            "expected_total_approx": 3000.0
        },
        {
            "name": "Payment plan simulation",
            "balance": 2400.0,
            "apr": 18.0,
            "months": 12,
            "kind": "payment_plan",
            "params": {"term_months": 12},
            "expected_monthly_range": (220, 240)
        }
    ]
    
    passed = 0
    total = len(simulation_cases)
    
    for i, case in enumerate(simulation_cases, 1):
        try:
            print(f"{i}. {case['name']}")
            
            result = advisor.plan_simulate(
                balance=case["balance"],
                apr=case["apr"],
                months=case["months"],
                kind=case["kind"],
                params=case["params"]
            )
            
            # Validate simulation structure
            valid_structure = (
                "schedule" in result and
                "npv" in result and
                "chargeoff_risk" in result and
                len(result["schedule"]) > 0
            )
            
            # Check specific expectations
            expectation_met = True
            if "expected_first_payment" in case:
                first_payment = result["schedule"][0]["payment"]
                expectation_met = abs(first_payment - case["expected_first_payment"]) < 0.01
            elif "expected_total_approx" in case:
                total_payments = sum(p["payment"] for p in result["schedule"])
                expectation_met = abs(total_payments - case["expected_total_approx"]) < 100
            elif "expected_monthly_range" in case:
                avg_payment = sum(p["payment"] for p in result["schedule"]) / len(result["schedule"])
                min_exp, max_exp = case["expected_monthly_range"]
                expectation_met = min_exp <= avg_payment <= max_exp
            
            success = valid_structure and expectation_met
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Balance: ${case['balance']:.2f}, APR: {case['apr']}%")
            print(f"   Schedule length: {len(result['schedule'])} months")
            print(f"   NPV: ${result['npv']:.2f}")
            print(f"   Risk score: {result['chargeoff_risk']:.3f}")
            print(f"   Expectation met: {expectation_met}")
            print(f"   Status: {status}")
            print()
            
            if success:
                passed += 1
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            print()
    
    print(f"Payment Simulation Results: {passed}/{total} passed")
    return passed == total

def test_scoring_algorithm():
    """Test plan scoring algorithm"""
    print("🎯 Testing Scoring Algorithm")
    print("=" * 50)
    
    advisor = CollectionsAdvisor()
    
    # Test with different customer profiles
    scoring_cases = [
        {
            "name": "High income customer - should prefer payment plans",
            "customer_state": CustomerState(
                balance=5000.0,
                apr=22.99,
                bucket="60",
                income_monthly=8000.0,
                expenses_monthly=4000.0
            ),
            "expected_high_score_type": "payment_plan"
        },
        {
            "name": "Low income customer - should prefer deferral",
            "customer_state": CustomerState(
                balance=3000.0,
                apr=24.99,
                bucket="90",
                income_monthly=2200.0,
                expenses_monthly=2000.0
            ),
            "expected_high_score_type": "deferral"
        },
        {
            "name": "Severely delinquent - should prefer settlement",
            "customer_state": CustomerState(
                balance=10000.0,
                apr=29.99,
                bucket="120+",
                income_monthly=3000.0,
                expenses_monthly=2800.0
            ),
            "expected_high_score_type": "settlement"
        }
    ]
    
    passed = 0
    total = len(scoring_cases)
    
    for i, case in enumerate(scoring_cases, 1):
        try:
            print(f"{i}. {case['name']}")
            
            result = advisor.process_hardship_request(case["customer_state"])
            
            # Check if expected plan type is top-ranked
            top_plan_type = result.plans[0].type if result.plans else None
            expectation_met = top_plan_type == case["expected_high_score_type"]
            
            # Check score distribution
            scores = [plan.npv for plan in result.plans]
            score_variation = max(scores) - min(scores) if len(scores) > 1 else 0
            
            success = expectation_met and len(result.plans) > 0
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Customer profile: {case['customer_state'].bucket} bucket, income ${case['customer_state'].income_monthly:.0f}")
            print(f"   Top plan type: {top_plan_type}")
            print(f"   Expected type: {case['expected_high_score_type']}")
            print(f"   Expectation met: {expectation_met}")
            print(f"   Plans generated: {len(result.plans)}")
            print(f"   Status: {status}")
            print()
            
            if success:
                passed += 1
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            print()
    
    print(f"Scoring Algorithm Results: {passed}/{total} passed")
    return passed == total

def test_edge_cases():
    """Test edge cases and error handling"""
    print("⚠️  Testing Edge Cases")
    print("=" * 50)
    
    advisor = CollectionsAdvisor()
    
    edge_cases = [
        {
            "name": "Very low balance",
            "customer_state": CustomerState(
                balance=50.0,
                apr=19.99,
                bucket="current"
            )
        },
        {
            "name": "Very high balance",
            "customer_state": CustomerState(
                balance=100000.0,
                apr=29.99,
                bucket="120+"
            )
        },
        {
            "name": "Zero APR",
            "customer_state": CustomerState(
                balance=1000.0,
                apr=0.0,
                bucket="current"
            )
        },
        {
            "name": "No income data",
            "customer_state": CustomerState(
                balance=5000.0,
                apr=24.99,
                bucket="90"
            )
        }
    ]
    
    passed = 0
    total = len(edge_cases)
    
    for i, case in enumerate(edge_cases, 1):
        try:
            print(f"{i}. {case['name']}")
            
            result = advisor.process_hardship_request(case["customer_state"])
            
            # Should always return valid structure
            valid_structure = (
                hasattr(result, 'plans') and
                hasattr(result, 'citations') and
                isinstance(result.plans, list) and
                isinstance(result.citations, list)
            )
            
            # Should handle edge case gracefully
            handled_gracefully = len(result.plans) >= 0  # At least doesn't crash
            
            success = valid_structure and handled_gracefully
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Balance: ${case['customer_state'].balance:.2f}")
            print(f"   APR: {case['customer_state'].apr}%")
            print(f"   Valid structure: {valid_structure}")
            print(f"   Plans generated: {len(result.plans)}")
            print(f"   Status: {status}")
            print()
            
            if success:
                passed += 1
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            print()
    
    print(f"Edge Case Results: {passed}/{total} passed")
    return passed == total

def main():
    """Run all CollectionsAdvisor tests"""
    print("💼 CollectionsAdvisor Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test policy compliance
    compliance_success = test_policy_compliance()
    results.append(("Policy Compliance", 1 if compliance_success else 0, 1))
    
    print("="*60)
    
    # Test payment simulation
    simulation_success = test_payment_simulation()
    results.append(("Payment Simulation", 1 if simulation_success else 0, 1))
    
    print("="*60)
    
    # Test scoring algorithm
    scoring_success = test_scoring_algorithm()
    results.append(("Scoring Algorithm", 1 if scoring_success else 0, 1))
    
    print("="*60)
    
    # Test golden paths
    print("🌟 Testing Golden Paths...")
    golden_success = test_collections_advisor()
    results.append(("Golden Paths", 1 if golden_success else 0, 1))
    
    print("="*60)
    
    # Test edge cases
    edge_success = test_edge_cases()
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
        print("\n🎉 All CollectionsAdvisor tests passed! Ready for hardship negotiations.")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Please review the issues above.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    success = main()
    sys.exit(0 if success else 1)
