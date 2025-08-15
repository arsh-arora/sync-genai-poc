"""
Dispute & Benefits Copilot - Reg-E-like Packet Processing
Enhanced version with category detection, eligibility windows, evidence classification, and structured packet assembly
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
    Enhanced Dispute & Benefits Copilot for Reg-E-like dispute packet processing
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
        Enhanced Reg-E-like dispute processing pipeline with category detection, eligibility windows,
        evidence classification, and structured packet assembly
        
        Args:
            narrative: Customer's description of the dispute
            merchant: Merchant name (optional)
            amount: Disputed amount (optional)
            uploaded_text: Receipt or document text (optional)
            
        Returns:
            DisputeResponse with structured metadata for UI cards, status, and handoffs
        """
        try:
            logger.info(f"Processing Reg-E dispute: {narrative[:100]}...")
            
            # Step 1: Category Detection against rules/dispute.yml
            category = self._detect_category(narrative)
            logger.info(f"Dispute categorized as: {category}")
            
            # Step 2: Evidence Classification by regex/heuristic
            evidence = self._classify_evidence(narrative, uploaded_text, merchant, amount)
            logger.info(f"Classified {len(evidence)} evidence items")
            
            # Step 3: Eligibility Windows Check (demo clocks)
            eligibility = self._check_eligibility(evidence)
            logger.info(f"Eligibility check: {eligibility.eligible} - {eligibility.eligibility_reason}")
            
            # Step 4: Timeline Construction
            timeline = self._construct_timeline(evidence)
            
            # Step 5: Status Assessment
            status = self._assess_status(category, evidence, eligibility)
            
            # Step 6: Packet Assembly
            packet_preview = self._assemble_packet(category, evidence, timeline, merchant or "Unknown", amount or 0.0)
            
            # Step 7: Check for handoffs
            handoffs = self._detect_handoffs(narrative, category)
            
            # Step 8: Generate human summary
            response_summary = self._generate_summary(category, status, packet_preview)
            
            return DisputeResponse(
                response=response_summary,
                metadata={
                    "status": status.dict(),
                    "ui_cards": [packet_preview.dict()],
                    "handoffs": handoffs,
                    "category": category,
                    "evidence_count": len(evidence),
                    "timeline_events": len(timeline)
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing Reg-E dispute: {e}")
            # Return fallback response
            return DisputeResponse(
                response="Unable to process dispute. Please contact customer service for assistance.",
                metadata={
                    "status": {"stage": "intake", "likelihood": "low", "next_milestone": "manual_review", "eligible": False, "eligibility_reason": "Processing error"},
                    "ui_cards": [],
                    "handoffs": [],
                    "error": str(e)
                }
            )
    
    def _detect_category(self, narrative: str) -> str:
        """Detect dispute category against rules/dispute.yml categories"""
        narrative_lower = narrative.lower()
        
        # Category detection patterns
        category_patterns = {
            "duplicate_charge": ["charged twice", "double charge", "duplicate", "same transaction", "multiple charges"],
            "item_not_received": ["not received", "never arrived", "didn't get", "not delivered", "missing order", "not shipped"],
            "wrong_amount": ["wrong amount", "charged more", "overcharged", "incorrect price", "different amount"],
            "canceled_but_charged": ["canceled", "cancelled", "refund", "returned", "charged after cancel"]
        }
        
        for category, patterns in category_patterns.items():
            if any(pattern in narrative_lower for pattern in patterns):
                return category
        
        return "duplicate_charge"  # Default category
    
    def _classify_evidence(self, narrative: str, uploaded_text: Optional[str], 
                          merchant: Optional[str], amount: Optional[float]) -> List[EvidenceItem]:
        """Classify evidence by regex/heuristic (merchant, dates, amounts, order id, chats)"""
        evidence = []
        
        # Extract from narrative
        if narrative:
            # Merchant names
            merchant_matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', narrative)
            for match in merchant_matches[:3]:  # Limit to first 3
                if len(match) > 2:
                    evidence.append(EvidenceItem(
                        type="merchant",
                        value=match,
                        confidence=0.7,
                        source="narrative"
                    ))
            
            # Dates
            date_patterns = [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
            ]
            for pattern in date_patterns:
                dates = re.findall(pattern, narrative, re.IGNORECASE)
                for date in dates:
                    evidence.append(EvidenceItem(
                        type="date",
                        value=date,
                        confidence=0.8,
                        source="narrative"
                    ))
            
            # Amounts
            amount_pattern = r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
            amounts = re.findall(amount_pattern, narrative)
            for amt in amounts:
                evidence.append(EvidenceItem(
                    type="amount",
                    value=f"${amt}",
                    confidence=0.9,
                    source="narrative"
                ))
            
            # Order IDs
            order_patterns = [
                r'\b(?:order|confirmation|ref|reference)[\s#:]*([A-Z0-9]{6,20})\b',
                r'\b[A-Z0-9]{8,16}\b'
            ]
            for pattern in order_patterns:
                orders = re.findall(pattern, narrative, re.IGNORECASE)
                for order in orders:
                    evidence.append(EvidenceItem(
                        type="order_id",
                        value=order,
                        confidence=0.6,
                        source="narrative"
                    ))
        
        # Add provided merchant and amount
        if merchant:
            evidence.append(EvidenceItem(
                type="merchant",
                value=merchant,
                confidence=1.0,
                source="provided"
            ))
        
        if amount:
            evidence.append(EvidenceItem(
                type="amount",
                value=f"${amount:.2f}",
                confidence=1.0,
                source="provided"
            ))
        
        # Extract from uploaded text if available
        if uploaded_text:
            # Simple receipt parsing
            receipt_amounts = re.findall(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', uploaded_text)
            for amt in receipt_amounts[:2]:  # Limit to first 2
                evidence.append(EvidenceItem(
                    type="amount",
                    value=f"${amt}",
                    confidence=0.9,
                    source="uploaded_text"
                ))
        
        return evidence
    
    def _check_eligibility(self, evidence: List[EvidenceItem]) -> DisputeStatus:
        """Check eligibility windows (60d from posting or 90d from purchase) - demo clocks"""
        today = datetime.now()
        posting_window = timedelta(days=self.demo_clocks.get("posting_window_days", 60))
        purchase_window = timedelta(days=self.demo_clocks.get("purchase_window_days", 90))
        
        # Extract dates from evidence
        dates = [item for item in evidence if item.type == "date"]
        
        if not dates:
            # No dates found - assume within window for demo
            return DisputeStatus(
                stage="intake",
                likelihood="medium",
                next_milestone="evidence_review",
                eligible=True,
                eligibility_reason="No specific dates provided; assuming within demo window (illustrative only)"
            )
        
        # Check against demo clocks
        eligible = True
        reason = f"Within demo eligibility windows: {self.demo_clocks['posting_window_days']}d from posting or {self.demo_clocks['purchase_window_days']}d from purchase (illustrative only)"
        
        # Simple date parsing for demo purposes
        for date_item in dates:
            try:
                # Try to parse the date
                date_str = date_item.value
                if '/' in date_str:
                    date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                elif '-' in date_str:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                else:
                    continue
                
                # Check if outside windows
                days_ago = (today - date_obj).days
                if days_ago > self.demo_clocks.get("purchase_window_days", 90):
                    eligible = False
                    reason = f"Date {date_str} is {days_ago} days ago, outside demo window (illustrative only)"
                    break
                    
            except ValueError:
                continue
        
        return DisputeStatus(
            stage="intake",
            likelihood="medium",
            next_milestone="eligibility_confirmed",
            eligible=eligible,
            eligibility_reason=reason
        )
    
    def _construct_timeline(self, evidence: List[EvidenceItem]) -> List[TimelineEvent]:
        """Construct timeline from evidence"""
        timeline = []
        
        # Add dates from evidence
        for item in evidence:
            if item.type == "date":
                timeline.append(TimelineEvent(
                    date=item.value,
                    event=f"Transaction date (from {item.source})",
                    source=item.source
                ))
        
        # Add current date as intake
        timeline.append(TimelineEvent(
            date=datetime.now().strftime("%Y-%m-%d"),
            event="Dispute intake initiated",
            source="system"
        ))
        
        return timeline
    
    def _assess_status(self, category: str, evidence: List[EvidenceItem], eligibility: DisputeStatus) -> DisputeStatus:
        """Assess dispute status based on category, evidence, and eligibility"""
        
        # Get required evidence for category
        required = self.categories.get(category, {}).get("required_evidence", [])
        evidence_types = [item.type for item in evidence]
        
        # Calculate likelihood based on evidence completeness
        evidence_score = 0
        for req in required:
            if any(req.replace("_", "") in etype or etype in req for etype in evidence_types):
                evidence_score += 1
        
        completion_ratio = evidence_score / len(required) if required else 0.8
        
        if completion_ratio >= 0.8:
            likelihood = "high"
            stage = "ready_to_submit"
            next_milestone = "submit_packet"
        elif completion_ratio >= 0.5:
            likelihood = "medium"
            stage = "compiled"
            next_milestone = "gather_remaining_evidence"
        else:
            likelihood = "low"
            stage = "intake"
            next_milestone = "evidence_collection"
        
        return DisputeStatus(
            stage=stage,
            likelihood=likelihood,
            next_milestone=next_milestone,
            eligible=eligibility.eligible,
            eligibility_reason=eligibility.eligibility_reason
        )
    
    def _assemble_packet(self, category: str, evidence: List[EvidenceItem], 
                        timeline: List[TimelineEvent], merchant: str, amount: float) -> PacketPreview:
        """Assemble packet with timeline, evidence bullets, and next update date"""
        
        # Next update date (today + 10 days)
        next_update = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
        
        return PacketPreview(
            category=category,
            merchant=merchant,
            amount=amount,
            timeline=timeline,
            evidence=evidence,
            next_update_date=next_update
        )
    
    def _detect_handoffs(self, narrative: str, category: str) -> List[str]:
        """Detect if handoffs to contracts are needed for policy clause clarity"""
        handoffs = []
        
        # Check for contract/policy related terms
        contract_terms = ["terms", "policy", "contract", "agreement", "clause", "warranty", "guarantee"]
        narrative_lower = narrative.lower()
        
        if any(term in narrative_lower for term in contract_terms):
            handoffs.append("contracts")
        
        return handoffs
    
    def _generate_summary(self, category: str, status: DisputeStatus, packet: PacketPreview) -> str:
        """Generate short human summary + ready to submit note"""
        
        category_names = {
            "duplicate_charge": "Duplicate Charge",
            "item_not_received": "Item Not Received", 
            "wrong_amount": "Wrong Amount",
            "canceled_but_charged": "Canceled But Charged"
        }
        
        category_display = category_names.get(category, category.replace("_", " ").title())
        
        summary = f"**{category_display} Dispute** for ${packet.amount:.2f} at {packet.merchant}. "
        summary += f"Evidence: {len(packet.evidence)} items collected. "
        summary += f"Timeline: {len(packet.timeline)} events documented. "
        
        if status.eligible:
            if status.stage == "ready_to_submit":
                summary += "✅ **Ready to submit** - all required evidence collected."
            else:
                summary += f"Status: {status.stage.replace('_', ' ').title()} ({status.likelihood} likelihood)."
        else:
            summary += f"⚠️ Eligibility issue: {status.eligibility_reason}"
        
        summary += f" Next update by {packet.next_update_date}."
        
        return summary

# Backwards compatibility - create alias for existing import
DisputeCopilot_Enhanced = DisputeCopilot