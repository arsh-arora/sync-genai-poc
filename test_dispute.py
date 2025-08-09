#!/usr/bin/env python3
"""
Test script for DisputeCopilot agent
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.agents.dispute import DisputeCopilot, test_dispute_copilot

def test_receipt_extraction():
    """Test receipt extraction functionality"""
    print("📄 Testing Receipt Extraction")
    print("=" * 50)
    
    copilot = DisputeCopilot()
    
    test_receipts = [
        {
            "name": "Best Buy Receipt",
            "text": """BEST BUY
Store #1234
123 Main St, Anytown USA

Date: 12/15/2024
Time: 14:30

Laptop Computer         $899.99
Extended Warranty       $199.99
Tax                     $65.99
                       --------
TOTAL                 $1,165.97

Payment: Credit Card
Card: ****1234
Auth: 123456""",
            "expected_merchant": "Best Buy",
            "expected_amount": 1165.97
        },
        {
            "name": "Amazon Receipt",
            "text": """Amazon.com
Order #123-4567890-1234567
Order Date: December 10, 2024

Items:
- Wireless Headphones    $79.99
- Phone Case            $24.99
- Shipping              $0.00

Subtotal: $104.98
Tax: $8.40
Total: $113.38

Payment Method: Credit Card ending in 5678""",
            "expected_merchant": "Amazon",
            "expected_amount": 113.38
        },
        {
            "name": "Restaurant Receipt",
            "text": """Mario's Italian Restaurant
456 Oak Avenue
City, State 12345

Table: 12
Server: Sarah
Date: 01/08/2025

2x Pasta Primavera      $32.00
1x Caesar Salad         $12.00
2x Soft Drinks          $6.00

Subtotal               $50.00
Tax                    $4.00
Tip                    $10.00
TOTAL                  $64.00

Thank you for dining with us!""",
            "expected_merchant": "Mario's Italian Restaurant",
            "expected_amount": 64.00
        }
    ]
    
    passed = 0
    total = len(test_receipts)
    
    for i, receipt in enumerate(test_receipts, 1):
        try:
            print(f"{i}. {receipt['name']}")
            
            result = copilot.receipt_extract(receipt["text"])
            
            # Check extraction results
            merchant_ok = result.get("merchant") is not None
            amount_ok = result.get("amount") is not None
            items_ok = isinstance(result.get("items", []), list)
            
            success = merchant_ok and amount_ok and items_ok
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Merchant: {result.get('merchant', 'None')}")
            print(f"   Amount: ${result.get('amount', 0):.2f}")
            print(f"   Date: {result.get('date', 'None')}")
            print(f"   Items: {len(result.get('items', []))} found")
            print(f"   Status: {status}")
            print()
            
            if success:
                passed += 1
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            print()
    
    print(f"Receipt Extraction Results: {passed}/{total} passed")
    return passed == total

def test_dispute_triage():
    """Test dispute triage classification"""
    print("🎯 Testing Dispute Triage")
    print("=" * 50)
    
    copilot = DisputeCopilot()
    
    triage_cases = [
        {
            "narrative": "I was charged twice for the same item at Target",
            "expected": "billing_error",
            "description": "Duplicate charge scenario"
        },
        {
            "narrative": "Someone used my card at a gas station in another state while I was at home",
            "expected": "fraud",
            "description": "Clear fraud case"
        },
        {
            "narrative": "I ordered a phone online but it never arrived, even though I was charged",
            "expected": "goods_not_received",
            "description": "Non-delivery case"
        },
        {
            "narrative": "The restaurant charged me $150 but the bill was only $50",
            "expected": "billing_error",
            "description": "Wrong amount charged"
        },
        {
            "narrative": "I see charges on my card that I never made and don't recognize",
            "expected": "fraud",
            "description": "Unrecognized charges"
        },
        {
            "narrative": "I paid for a service but the company never provided it",
            "expected": "goods_not_received",
            "description": "Service not provided"
        }
    ]
    
    passed = 0
    total = len(triage_cases)
    
    for i, case in enumerate(triage_cases, 1):
        try:
            print(f"{i}. {case['description']}")
            
            result = copilot._triage_dispute(case["narrative"])
            
            success = result == case["expected"]
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Narrative: {case['narrative'][:50]}...")
            print(f"   Expected: {case['expected']}")
            print(f"   Got: {result}")
            print(f"   Status: {status}")
            print()
            
            if success:
                passed += 1
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            print()
    
    print(f"Dispute Triage Results: {passed}/{total} passed")
    return passed == total

def test_dispute_integration():
    """Test full dispute processing integration"""
    print("🔗 Testing Dispute Integration")
    print("=" * 50)
    
    copilot = DisputeCopilot()
    
    integration_cases = [
        {
            "name": "Complete Billing Error Case",
            "narrative": "I was charged $299.99 twice for the same laptop purchase at Best Buy on December 15th",
            "merchant": "Best Buy",
            "amount": 299.99,
            "uploaded_text": """BEST BUY
Date: 12/15/2024
Laptop Computer - $299.99
TOTAL: $299.99""",
            "expected_triage": "billing_error"
        },
        {
            "name": "Goods Not Received with Receipt",
            "narrative": "I ordered headphones from Amazon but they never arrived",
            "merchant": "Amazon",
            "amount": 79.99,
            "uploaded_text": """Amazon.com
Order Date: 12/01/2024
Wireless Headphones - $79.99
Expected Delivery: 12/05/2024
Total: $79.99""",
            "expected_triage": "goods_not_received"
        },
        {
            "name": "Fraud Case",
            "narrative": "There's a $500 charge at a store I've never been to",
            "merchant": None,
            "amount": 500.00,
            "uploaded_text": None,
            "expected_triage": "fraud"
        }
    ]
    
    passed = 0
    total = len(integration_cases)
    
    for i, case in enumerate(integration_cases, 1):
        try:
            print(f"{i}. {case['name']}")
            
            result = copilot.process_dispute(
                narrative=case["narrative"],
                merchant=case["merchant"],
                amount=case["amount"],
                uploaded_text=case["uploaded_text"]
            )
            
            # Validate complete response
            triage_ok = result.triage == case["expected_triage"]
            resolution_ok = len(result.merchant_resolution.message) > 50
            packet_ok = len(result.packet.letter) > 100
            fields_ok = result.packet.fields.amount > 0
            
            success = triage_ok and resolution_ok and packet_ok and fields_ok
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Triage: {result.triage} (expected: {case['expected_triage']})")
            print(f"   Resolution: {len(result.merchant_resolution.message)} chars")
            print(f"   Letter: {len(result.packet.letter)} chars")
            print(f"   Checklist: {len(result.merchant_resolution.checklist)} items")
            print(f"   Attachments: {len(result.packet.attachments)} required")
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

def test_dispute_edge_cases():
    """Test edge cases and error handling"""
    print("⚠️  Testing Dispute Edge Cases")
    print("=" * 50)
    
    copilot = DisputeCopilot()
    
    edge_cases = [
        {
            "name": "Empty narrative",
            "narrative": "",
            "merchant": None,
            "amount": None,
            "uploaded_text": None
        },
        {
            "name": "Very long narrative",
            "narrative": "This is a very long dispute narrative that goes on and on about various issues with a transaction that occurred at a merchant location where I purchased multiple items and had various problems with the service and the products and the billing and the delivery and many other issues that need to be resolved through the dispute process.",
            "merchant": "Test Merchant",
            "amount": 100.0,
            "uploaded_text": None
        },
        {
            "name": "Invalid receipt text",
            "narrative": "I have a dispute",
            "merchant": None,
            "amount": None,
            "uploaded_text": "This is not a valid receipt format at all just random text"
        },
        {
            "name": "Missing all optional fields",
            "narrative": "I need help with a dispute",
            "merchant": None,
            "amount": None,
            "uploaded_text": None
        }
    ]
    
    passed = 0
    total = len(edge_cases)
    
    for i, case in enumerate(edge_cases, 1):
        try:
            print(f"{i}. {case['name']}")
            
            result = copilot.process_dispute(
                narrative=case["narrative"],
                merchant=case["merchant"],
                amount=case["amount"],
                uploaded_text=case["uploaded_text"]
            )
            
            # Should always return valid structure
            valid_structure = (
                hasattr(result, 'triage') and
                hasattr(result, 'merchant_resolution') and
                hasattr(result, 'packet') and
                hasattr(result, 'citations') and
                isinstance(result.triage, str) and
                len(result.merchant_resolution.message) > 0 and
                len(result.packet.letter) > 0
            )
            
            success = valid_structure
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Valid structure: {valid_structure}")
            print(f"   Triage: {result.triage}")
            print(f"   Has resolution: {len(result.merchant_resolution.message) > 0}")
            print(f"   Has letter: {len(result.packet.letter) > 0}")
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
    """Run all DisputeCopilot tests"""
    print("⚖️  DisputeCopilot Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test receipt extraction
    receipt_success = test_receipt_extraction()
    results.append(("Receipt Extraction", 1 if receipt_success else 0, 1))
    
    print("="*60)
    
    # Test dispute triage
    triage_success = test_dispute_triage()
    results.append(("Dispute Triage", 1 if triage_success else 0, 1))
    
    print("="*60)
    
    # Test golden paths
    print("🌟 Testing Golden Paths...")
    golden_success = test_dispute_copilot()
    results.append(("Golden Paths", 1 if golden_success else 0, 1))
    
    print("="*60)
    
    # Test integration
    integration_success = test_dispute_integration()
    results.append(("Integration", 1 if integration_success else 0, 1))
    
    print("="*60)
    
    # Test edge cases
    edge_success = test_dispute_edge_cases()
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
        print("\n🎉 All DisputeCopilot tests passed! Ready for dispute processing.")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Please review the issues above.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    success = main()
    sys.exit(0 if success else 1)
