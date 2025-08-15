"""
Dispute & Benefits Copilot - Reg-E-like Packet Processing
Turns messy customer narratives + receipts into compliant dispute packets with category detection,
eligibility windows, evidence classification, and structured packet assembly
"""

import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.llm.gemini import chat
from app.rag.core import retrieve
from app.tools.tavily_search import web_search_into_docstore

logger = logging.getLogger(__name__)

# Enhanced Pydantic models for Reg-E-like processing
class EvidenceItem(BaseModel):
    type: str  # merchant, date, amount, order_id, chat, etc.
    value: str
    confidence: float
    source: str  # narrative, uploaded_text, extracted

class TimelineEvent(BaseModel):
    date: str
    event: str
    source: str

class DisputeStatus(BaseModel):
    stage: str  # intake, compiled, ready_to_submit, submitted, under_review
    likelihood: str  # low, medium, high
    next_milestone: str
    eligible: bool
    eligibility_reason: str

class PacketPreview(BaseModel):
    category: str
    merchant: str
    amount: float
    timeline: List[TimelineEvent]
    evidence: List[EvidenceItem]
    next_update_date: str

class DisputeResponse(BaseModel):
    response: str  # Short human summary + "ready to submit" note
    metadata: Dict[str, Any]  # Contains status, ui_cards, handoffs

class DisputeCopilot:
    """
    Dispute & Benefits Copilot for Reg-E-like dispute packet processing with category detection,
    eligibility windows, evidence classification, and structured packet assembly
    """
    
    def __init__(self, docstore=None, embedder=None, retriever=None, rules_loader=None):
        """Initialize DisputeCopilot with RAG components and rules-based processing"""
        self.docstore = docstore
        self.embedder = embedder
        self.retriever = retriever
        self.rules_loader = rules_loader
        
        # Load dispute rules from centralized loader or use defaults
        if rules_loader:
            self.dispute_rules = rules_loader.get_rules('dispute') or {}
            logger.info("DisputeCopilot loaded rules from centralized rules loader")
        else:
            self.dispute_rules = self._load_fallback_rules()
        
        # Extract rule components
        self.categories = self.dispute_rules.get("categories", {})
        self.demo_clocks = self.dispute_rules.get("demo_clocks", {"posting_window_days": 60, "purchase_window_days": 90})
        self.status_pipeline = self.dispute_rules.get("status_pipeline", ["intake", "compiled", "ready_to_submit", "submitted", "under_review"])
        self.likelihood_buckets = self.dispute_rules.get("likelihood_buckets", ["low", "medium", "high"])
        
    def _load_fallback_rules(self) -> Dict[str, Any]:
        """Fallback rules if centralized loader not available"""
        return {
            "categories": {
                "duplicate_charge": {"required_evidence": ["statement_screenshot", "merchant_name", "date", "amount"]},
                "item_not_received": {"required_evidence": ["order_confirmation", "expected_delivery", "merchant_contact_attempt"]},
                "wrong_amount": {"required_evidence": ["receipt", "statement_screenshot", "correct_amount"]},
                "canceled_but_charged": {"required_evidence": ["cancellation_confirmation", "statement_screenshot"]}
            },
            "demo_clocks": {"posting_window_days": 60, "purchase_window_days": 90},
            "status_pipeline": ["intake", "compiled", "ready_to_submit", "submitted", "under_review"],
            "likelihood_buckets": ["low", "medium", "high"]
        }
    
    def process_dispute(self, narrative: str, merchant: Optional[str] = None, 
                       amount: Optional[float] = None, uploaded_text: Optional[str] = None) -> DisputeResponse:
        """
        Main processing pipeline for dispute cases
        
        Args:
            narrative: Customer's description of the dispute
            merchant: Merchant name (optional)
            amount: Disputed amount (optional)
            uploaded_text: Receipt or document text (optional)
            
        Returns:
            DisputeResponse with triage, resolution, packet, and citations
        """
        try:
            logger.info(f"Processing dispute: {narrative[:100]}...")
            
            # Step 1: Triage with Gemini
            triage_result = self._triage_dispute(narrative)
            logger.info(f"Dispute triaged as: {triage_result}")
            
            # Step 2: Extract receipt information if uploaded
            receipt_data = {}
            if uploaded_text:
                receipt_data = self.receipt_extract(uploaded_text)
                logger.info(f"Extracted receipt data: {receipt_data}")
            
            # Step 3: Merge user inputs with extracted data
            merged_data = self._merge_dispute_data(
                narrative, merchant, amount, receipt_data
            )
            
            # Step 4: Retrieve merchant policy citations
            citations = self._get_policy_citations(merged_data.get("merchant"), triage_result)
            
            # Step 5: Generate merchant resolution (unless fraud)
            merchant_resolution = None
            if triage_result != "fraud":
                merchant_resolution = self._generate_merchant_resolution(
                    triage_result, merged_data, citations
                )
            else:
                merchant_resolution = MerchantResolution(
                    message="For fraudulent transactions, contact your card issuer immediately. Do not attempt merchant resolution.",
                    checklist=["Report fraud immediately", "Request account freeze", "File police report if amount > $500"]
                )
            
            # Step 6: Compose formal dispute packet
            dispute_packet = self._compose_dispute_packet(
                triage_result, merged_data, citations
            )
            
            return DisputeResponse(
                triage=triage_result,
                merchant_resolution=merchant_resolution,
                packet=dispute_packet,
                citations=citations
            )
            
        except Exception as e:
            logger.error(f"Error processing dispute: {e}")
            # Return minimal response on error
            return DisputeResponse(
                triage="billing_error",
                merchant_resolution=MerchantResolution(
                    message="Please contact customer service for assistance with your dispute.",
                    checklist=["Gather transaction documentation", "Contact customer service"]
                ),
                packet=DisputePacket(
                    summary="System error occurred during dispute processing",
                    fields=DisputeFields(
                        merchant=merchant or "Unknown",
                        date=datetime.now().strftime("%Y-%m-%d"),
                        amount=amount or 0.0,
                        reason="System error"
                    ),
                    letter="Please contact customer service for assistance.",
                    attachments=[]
                ),
                citations=[]
            )
    
    def receipt_extract(self, text: str) -> Dict[str, Any]:
        """
        Extract receipt information using Gemini + regex fallback
        
        Args:
            text: Receipt text to extract from
            
        Returns:
            Dictionary with merchant, date, amount, and items
        """
        try:
            # Use Gemini for structured extraction
            system_prompt = """You are a receipt parser. Extract structured information from receipt text.

Return ONLY a JSON object with these fields:
{
  "merchant": "merchant name or null",
  "date": "YYYY-MM-DD format or null", 
  "amount": "total amount as float or null",
  "items": ["list of purchased items or empty array"]
}

Be precise and only extract information that is clearly present."""

            user_message = f"Extract information from this receipt:\n\n{text}"
            messages = [{"role": "user", "content": user_message}]
            
            response = chat(messages, system=system_prompt)
            
            # Parse Gemini response
            try:
                extracted = json.loads(response.strip())
                logger.info("Successfully extracted receipt data with Gemini")
                return extracted
            except json.JSONDecodeError:
                logger.warning("Gemini response not valid JSON, falling back to regex")
                return self._regex_receipt_fallback(text)
                
        except Exception as e:
            logger.error(f"Error in receipt extraction: {e}")
            return self._regex_receipt_fallback(text)
    
    def _regex_receipt_fallback(self, text: str) -> Dict[str, Any]:
        """Regex-based fallback for receipt extraction"""
        result = {
            "merchant": None,
            "date": None,
            "amount": None,
            "items": []
        }
        
        # Extract merchant (usually first line or after common patterns)
        merchant_patterns = [
            r'^([A-Z][A-Za-z\s&]+)(?:\n|\r)',  # First line capitalized
            r'(?:MERCHANT|STORE|RETAILER):\s*([^\n\r]+)',
            r'^([A-Z\s]+(?:INC|LLC|CORP|CO))',  # Corporate suffixes
        ]
        
        for pattern in merchant_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                result["merchant"] = match.group(1).strip()
                break
        
        # Extract date
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # MM/DD/YYYY or MM-DD-YYYY
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',    # YYYY/MM/DD or YYYY-MM-DD
            r'((?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{1,2},?\s+\d{4})'  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    # Try to parse and standardize date
                    date_str = match.group(1)
                    # Simple conversion - in production would use dateutil
                    if '/' in date_str or '-' in date_str:
                        result["date"] = date_str
                    break
                except:
                    continue
        
        # Extract total amount
        amount_patterns = [
            r'(?:TOTAL|AMOUNT|BALANCE):\s*\$?(\d+\.?\d*)',
            r'TOTAL\s+\$?(\d+\.?\d*)',
            r'\$(\d+\.\d{2})\s*(?:TOTAL|$)',
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    result["amount"] = float(match.group(1))
                    break
                except ValueError:
                    continue
        
        # Extract items (simple line-based extraction)
        lines = text.split('\n')
        items = []
        for line in lines:
            line = line.strip()
            # Look for lines that might be items (have price patterns)
            if re.search(r'\$\d+\.?\d*', line) and len(line) > 5:
                # Clean up the line
                item = re.sub(r'\s*\$\d+\.?\d*.*$', '', line).strip()
                if item and len(item) > 2:
                    items.append(item)
        
        result["items"] = items[:10]  # Limit to 10 items
        
        return result
    
    def _triage_dispute(self, narrative: str) -> str:
        """
        Triage dispute using Gemini classification
        
        Args:
            narrative: Customer dispute narrative
            
        Returns:
            Dispute type: fraud, goods_not_received, or billing_error
        """
        try:
            system_prompt = """You are a dispute triage specialist. Classify customer disputes into one of these categories:

Categories:
- fraud: Unauthorized transactions, stolen card, identity theft
- goods_not_received: Items/services paid for but not delivered or provided
- billing_error: Duplicate charges, wrong amounts, mathematical errors

Respond with ONLY a JSON object:
{"category": "category_name", "confidence": 0.85, "reasoning": "brief explanation"}

Focus on the primary issue described by the customer."""

            user_message = f"Classify this dispute: '{narrative}'"
            messages = [{"role": "user", "content": user_message}]
            
            response = chat(messages, system=system_prompt)
            
            result = json.loads(response.strip())
            category = result.get("category", "billing_error")
            
            # Validate category
            valid_categories = ["fraud", "goods_not_received", "billing_error"]
            if category not in valid_categories:
                logger.warning(f"Invalid category '{category}', defaulting to billing_error")
                category = "billing_error"
            
            return category
            
        except Exception as e:
            logger.error(f"Error in dispute triage: {e}")
            return "billing_error"  # Safe default
    
    def _merge_dispute_data(self, narrative: str, merchant: Optional[str], 
                           amount: Optional[float], receipt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user inputs with extracted receipt data"""
        merged = {
            "narrative": narrative,
            "merchant": merchant or receipt_data.get("merchant"),
            "amount": amount or receipt_data.get("amount"),
            "date": receipt_data.get("date"),
            "items": receipt_data.get("items", [])
        }
        
        # Clean up merchant name
        if merged["merchant"]:
            merged["merchant"] = merged["merchant"].strip().title()
        
        return merged
    
    def _get_policy_citations(self, merchant: Optional[str], dispute_type: str) -> List[Citation]:
        """
        Retrieve merchant policy citations from knowledge base
        
        Args:
            merchant: Merchant name
            dispute_type: Type of dispute
            
        Returns:
            List of policy citations
        """
        citations = []
        
        if not all([self.retriever, self.embedder]):
            logger.warning("RAG components not available for policy citations")
            return citations
        
        try:
            # Create query for dispute policies
            if merchant:
                query = f"{merchant} dispute policy {dispute_type}"
            else:
                query = f"card dispute documentation {dispute_type} required items"
            
            # Retrieve relevant policy documents
            results = retrieve(self.retriever, self.embedder, query, k=3)
            
            for result in results:
                citations.append(Citation(
                    source=result.get("filename", "Dispute Policy"),
                    snippet=result.get("snippet", "")[:300] + "..."
                ))
            
            # If no local citations found, search web for card dispute documentation
            if not citations:
                logger.info("No local dispute policies found, searching web...")
                try:
                    web_docs = web_search_into_docstore(
                        self.docstore,
                        self.embedder,
                        "card dispute documentation duplicate charge required items",
                        max_results=2
                    )
                    
                    # Re-retrieve after adding web content
                    if web_docs:
                        results = retrieve(self.retriever, self.embedder, 
                                         f"dispute {dispute_type} documentation", k=2)
                        
                        for result in results:
                            citations.append(Citation(
                                source=result.get("filename", "Web Search"),
                                snippet=result.get("snippet", "")[:300] + "..."
                            ))
                            
                except Exception as e:
                    logger.warning(f"Web search for dispute policies failed: {e}")
            
        except Exception as e:
            logger.error(f"Error retrieving policy citations: {e}")
        
        return citations
    
    def _generate_merchant_resolution(self, dispute_type: str, data: Dict[str, Any], 
                                    citations: List[Citation]) -> MerchantResolution:
        """Generate merchant resolution draft"""
        merchant = data.get("merchant", "the merchant")
        amount = data.get("amount", 0.0)
        
        if dispute_type == "billing_error":
            message = f"""Dear {merchant} Customer Service,

I am writing to request resolution of a billing error on my account. I have identified a discrepancy in my recent transaction that requires your immediate attention.

Transaction Details:
- Date: {data.get('date', 'Not specified')}
- Amount: ${amount:.2f}
- Issue: {data.get('narrative', 'Billing discrepancy')}

I have reviewed my records and believe this charge is incorrect. Please investigate this matter and provide a correction or refund as appropriate.

I would appreciate your prompt response within 5-7 business days. Please confirm the resolution in writing.

Thank you for your attention to this matter."""

            checklist = [
                "Contact merchant customer service within 2-3 business days",
                "Provide transaction details and explanation",
                "Request written confirmation of resolution",
                "Keep records of all communication",
                "Allow 5-7 business days for merchant response",
                "Escalate to supervisor if initial contact unsuccessful"
            ]
            
        elif dispute_type == "goods_not_received":
            message = f"""Dear {merchant} Customer Service,

I am writing regarding an order that I paid for but have not received. I need your assistance in resolving this delivery issue.

Order Details:
- Date of Purchase: {data.get('date', 'Not specified')}
- Amount Paid: ${amount:.2f}
- Items: {', '.join(data.get('items', ['Not specified']))}

Despite payment being processed, I have not received the goods/services. Please investigate the status of my order and provide either:
1. Immediate shipment with tracking information, or
2. Full refund of the purchase amount

I request resolution within 10-15 business days as per standard delivery expectations.

Please confirm your resolution plan in writing."""

            checklist = [
                "Contact merchant within 30 days of expected delivery",
                "Provide order confirmation and payment proof",
                "Request tracking information if applicable",
                "Ask for delivery timeline or refund",
                "Document all communication attempts",
                "Allow 10-15 business days for resolution",
                "Request written confirmation of solution"
            ]
        
        else:  # Default case
            message = f"""Dear {merchant} Customer Service,

I need assistance resolving an issue with a recent transaction on my account.

Transaction Details:
- Date: {data.get('date', 'Not specified')}
- Amount: ${amount:.2f}
- Issue: {data.get('narrative', 'Transaction dispute')}

Please review this transaction and provide appropriate resolution.

Thank you for your prompt attention to this matter."""

            checklist = [
                "Contact merchant customer service",
                "Provide transaction documentation",
                "Explain the issue clearly",
                "Request written response",
                "Allow reasonable time for resolution"
            ]
        
        return MerchantResolution(message=message, checklist=checklist)
    
    def _compose_dispute_packet(self, dispute_type: str, data: Dict[str, Any], 
                               citations: List[Citation]) -> DisputePacket:
        """Compose formal dispute packet"""
        
        # Create summary
        merchant = data.get("merchant", "Unknown Merchant")
        amount = data.get("amount", 0.0)
        date = data.get("date", "Unknown Date")
        
        summary = f"Dispute for {dispute_type.replace('_', ' ')} involving {merchant} for ${amount:.2f} on {date}"
        
        # Create fields
        fields = DisputeFields(
            merchant=merchant,
            date=date,
            amount=amount,
            reason=data.get("narrative", "No reason provided")
        )
        
        # Generate formal dispute letter
        letter = self._generate_formal_letter(dispute_type, data, citations)
        
        # Determine required attachments
        attachments = self._get_required_attachments(dispute_type)
        
        return DisputePacket(
            summary=summary,
            fields=fields,
            letter=letter,
            attachments=attachments
        )
    
    def _generate_formal_letter(self, dispute_type: str, data: Dict[str, Any], 
                               citations: List[Citation]) -> str:
        """Generate formal dispute letter"""
        
        merchant = data.get("merchant", "Unknown Merchant")
        amount = data.get("amount", 0.0)
        date = data.get("date", "Unknown Date")
        narrative = data.get("narrative", "")
        
        current_date = datetime.now().strftime("%B %d, %Y")
        
        if dispute_type == "fraud":
            letter = f"""Date: {current_date}

To: Card Issuer Dispute Department

RE: Fraudulent Transaction Dispute

Dear Dispute Resolution Team,

I am writing to formally dispute fraudulent charges on my account. I did not authorize these transactions and believe my account has been compromised.

DISPUTED TRANSACTION DETAILS:
- Merchant: {merchant}
- Transaction Date: {date}
- Amount: ${amount:.2f}
- Description: {narrative}

FRAUD DECLARATION:
I hereby declare under penalty of perjury that:
1. I did not authorize this transaction
2. I did not participate in this transaction
3. I did not give permission for anyone else to use my account
4. I have not received any goods or services from this transaction

IMMEDIATE ACTIONS TAKEN:
- Reported fraud immediately upon discovery
- Reviewed all recent account activity
- Secured account and changed passwords

I request immediate provisional credit and permanent removal of this fraudulent charge. Please investigate this matter urgently and provide written confirmation of resolution.

Sincerely,
[Cardholder Name]
Account Number: [Account Number]"""

        elif dispute_type == "goods_not_received":
            letter = f"""Date: {current_date}

To: Card Issuer Dispute Department

RE: Goods Not Received Dispute

Dear Dispute Resolution Team,

I am formally disputing a charge for goods/services that were paid for but never received.

TRANSACTION DETAILS:
- Merchant: {merchant}
- Transaction Date: {date}
- Amount: ${amount:.2f}
- Items/Services: {', '.join(data.get('items', ['Not specified']))}

ISSUE DESCRIPTION:
{narrative}

MERCHANT CONTACT ATTEMPTS:
I have attempted to resolve this matter directly with the merchant through:
- Initial contact on [Date]
- Follow-up contact on [Date]
- No satisfactory resolution provided

SUPPORTING EVIDENCE:
- Original purchase receipt/confirmation
- Communication records with merchant
- Proof of non-delivery

I request provisional credit while this matter is investigated. The merchant has failed to deliver the goods/services as promised, and I should not be held liable for this charge.

Please investigate and provide permanent credit for this transaction.

Sincerely,
[Cardholder Name]
Account Number: [Account Number]"""

        else:  # billing_error
            letter = f"""Date: {current_date}

To: Card Issuer Dispute Department

RE: Billing Error Dispute

Dear Dispute Resolution Team,

I am writing to dispute a billing error on my account that requires correction.

TRANSACTION DETAILS:
- Merchant: {merchant}
- Transaction Date: {date}
- Disputed Amount: ${amount:.2f}
- Error Description: {narrative}

BILLING ERROR DETAILS:
This charge appears to be incorrect due to:
- Duplicate processing of the same transaction
- Incorrect amount charged
- Mathematical or computational error
- Charge for goods/services not received

MERCHANT RESOLUTION ATTEMPT:
I contacted the merchant on [Date] to resolve this billing error. [Outcome of merchant contact]

REQUESTED RESOLUTION:
Please investigate this billing error and provide appropriate correction to my account. I have supporting documentation available upon request.

I request provisional credit during the investigation period as provided under the Fair Credit Billing Act.

Sincerely,
[Cardholder Name]
Account Number: [Account Number]"""

        return letter
    
    def _get_required_attachments(self, dispute_type: str) -> List[str]:
        """Get list of required attachments for dispute type"""
        
        common_attachments = [
            "Copy of credit card statement showing the disputed charge",
            "Account holder identification",
            "Completed dispute form"
        ]
        
        if dispute_type == "fraud":
            return common_attachments + [
                "Affidavit of unauthorized use",
                "Police report (if filed)",
                "List of all unauthorized transactions",
                "Documentation of account security measures taken"
            ]
        
        elif dispute_type == "goods_not_received":
            return common_attachments + [
                "Original purchase receipt or confirmation",
                "Shipping/tracking information (if applicable)",
                "Communication records with merchant",
                "Proof of expected delivery date",
                "Evidence of non-delivery"
            ]
        
        else:  # billing_error
            return common_attachments + [
                "Original receipt showing correct amount",
                "Documentation of the billing error",
                "Calculation showing the discrepancy",
                "Correspondence with merchant (if any)"
            ]

# Test cases for dispute scenarios
def test_dispute_copilot():
    """Test DisputeCopilot with common dispute scenarios"""
    print("🧪 Testing DisputeCopilot")
    print("=" * 50)
    
    copilot = DisputeCopilot()
    
    test_cases = [
        {
            "name": "Duplicate Charge",
            "narrative": "I was charged twice for the same purchase at Best Buy. The amount of $299.99 appears twice on my statement for the same day.",
            "merchant": "Best Buy",
            "amount": 299.99,
            "uploaded_text": None,
            "expected_triage": "billing_error"
        },
        {
            "name": "Goods Not Received",
            "narrative": "I ordered a laptop from Dell on December 1st but never received it. They charged my card $899 but the item was never delivered.",
            "merchant": "Dell",
            "amount": 899.00,
            "uploaded_text": "DELL TECHNOLOGIES\nOrder Date: 12/01/2024\nLaptop Computer - $899.00\nTotal: $899.00\nExpected Delivery: 12/15/2024",
            "expected_triage": "goods_not_received"
        },
        {
            "name": "Fraudulent Transaction",
            "narrative": "I see a charge for $500 at a store I've never been to. This is definitely fraud as I was out of town that day.",
            "merchant": None,
            "amount": 500.00,
            "uploaded_text": None,
            "expected_triage": "fraud"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        try:
            print(f"{i}. {case['name']}")
            
            result = copilot.process_dispute(
                narrative=case["narrative"],
                merchant=case["merchant"],
                amount=case["amount"],
                uploaded_text=case["uploaded_text"]
            )
            
            # Validate response structure
            valid_structure = (
                hasattr(result, 'triage') and
                hasattr(result, 'merchant_resolution') and
                hasattr(result, 'packet') and
                hasattr(result, 'citations')
            )
            
            # Check triage accuracy
            triage_correct = result.triage == case["expected_triage"]
            
            # Check packet completeness
            packet_complete = (
                result.packet.fields.merchant and
                result.packet.fields.amount > 0 and
                len(result.packet.letter) > 100
            )
            
            success = valid_structure and triage_correct and packet_complete
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Narrative: {case['narrative'][:60]}...")
            print(f"   Triage: Expected {case['expected_triage']}, Got {result.triage}")
            print(f"   Merchant Resolution: {len(result.merchant_resolution.message)} chars")
            print(f"   Dispute Letter: {len(result.packet.letter)} chars")
            print(f"   Citations: {len(result.citations)}")
            print(f"   Status: {status}")
            print()
            
            if success:
                passed += 1
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            print()
    
    print(f"DisputeCopilot Test Results: {passed}/{total} passed")
    return passed == total

if __name__ == "__main__":
    # Run tests
    success = test_dispute_copilot()
    print(f"\n{'🎉 All tests passed!' if success else '⚠️ Some tests failed.'}")
