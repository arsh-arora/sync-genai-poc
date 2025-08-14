"""
TrustShield - Real-time Scam & PII Defense System
Comprehensive fraud detection, PII protection, and safety guidance
"""

import re
import logging
import hashlib
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from app.llm.gemini import chat
from app.rag.core import retrieve

logger = logging.getLogger(__name__)

@dataclass
class ThreatEvidence:
    """Evidence of a potential threat"""
    type: str
    evidence: str
    confidence: float
    severity: str  # low, medium, high, critical

@dataclass
class SafetyGuidance:
    """Safety guidance for users"""
    label: str
    url: str

@dataclass
class Citation:
    """Citation from knowledge base"""
    source: str
    snippet: str

class TrustShield:
    """
    Advanced threat detection and PII protection system
    """
    
    def __init__(self, docstore=None, embedder=None, retriever=None):
        """Initialize TrustShield with optional RAG components for safety guidance"""
        self.docstore = docstore
        self.embedder = embedder
        self.retriever = retriever
        
        # Initialize Presidio engines
        try:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            logger.info("Presidio engines initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Presidio engines: {e}")
            self.analyzer = None
            self.anonymizer = None
        
        # Scam detection patterns
        self.scam_patterns = self._initialize_scam_patterns()
        
        # High-entropy patterns for gift cards, etc.
        self.entropy_patterns = self._initialize_entropy_patterns()
    
    def scan(self, text: str, user_type: str = "consumer") -> Dict[str, Any]:
        """
        Comprehensive scan of text for threats, PII, and scams
        
        Args:
            text: Text to analyze
            user_type: consumer or partner (for logging only)
            
        Returns:
            Dictionary with decision, reasons, redacted text, next steps, and citations
        """
        try:
            logger.info(f"TrustShield scanning text for {user_type}: {text[:100]}...")
            
            # Initialize result structure
            result = {
                "decision": "pass",
                "reasons": [],
                "redacted_text": text,
                "next_step": None,
                "citations": []
            }
            
            # 1. PII Detection using Presidio
            pii_threats = self._detect_pii(text)
            result["reasons"].extend(pii_threats)
            
            # 2. Heuristic-based detection (regex patterns)
            heuristic_threats = self._detect_heuristic_threats(text)
            result["reasons"].extend(heuristic_threats)
            
            # 3. High-entropy detection (gift cards, etc.)
            entropy_threats = self._detect_high_entropy_patterns(text)
            result["reasons"].extend(entropy_threats)
            
            # 4. Scam phrase detection
            phrase_threats = self._detect_scam_phrases(text)
            result["reasons"].extend(phrase_threats)
            
            # 5. Gemini-based intent classification
            intent_threats = self._classify_intent_with_gemini(text)
            result["reasons"].extend(intent_threats)
            
            # 6. Determine overall decision based on threat levels
            decision, next_step = self._make_decision(result["reasons"])
            result["decision"] = decision
            result["next_step"] = next_step
            
            # 7. Get safety guidance citations if high risk
            if decision in ["block", "warn"]:
                result["citations"] = self._get_safety_citations(result["reasons"])
            
            # 8. Redact text if needed
            if decision != "block":
                result["redacted_text"] = self._redact_sensitive_content(text, result["reasons"])
            else:
                result["redacted_text"] = "[BLOCKED - High risk content detected]"
            
            logger.info(f"TrustShield decision: {decision} with {len(result['reasons'])} threats detected")
            return result
            
        except Exception as e:
            logger.error(f"Error in TrustShield scan: {e}")
            return {
                "decision": "warn",
                "reasons": [ThreatEvidence("system_error", str(e), 0.8, "medium")],
                "redacted_text": text,
                "next_step": {"label": "Contact Support", "url": "/support"},
                "citations": []
            }
    
    def _detect_pii(self, text: str) -> List[ThreatEvidence]:
        """Detect PII using Presidio analyzer"""
        threats = []
        
        if not self.analyzer:
            return threats
        
        try:
            # Analyze text for PII
            results = self.analyzer.analyze(
                text=text,
                language='en',
                entities=None  # Detect all supported entities
            )
            
            for result in results:
                # Map Presidio confidence to our threat levels
                if result.score >= 0.8:
                    severity = "high"
                elif result.score >= 0.6:
                    severity = "medium"
                else:
                    severity = "low"
                
                evidence_text = text[result.start:result.end]
                
                threats.append(ThreatEvidence(
                    type="pii_detected",
                    evidence=f"{result.entity_type}: {evidence_text}",
                    confidence=result.score,
                    severity=severity
                ))
            
        except Exception as e:
            logger.error(f"Error in PII detection: {e}")
        
        return threats
    
    def _detect_heuristic_threats(self, text: str) -> List[ThreatEvidence]:
        """Detect threats using regex patterns"""
        threats = []
        
        # Credit card patterns (Luhn algorithm validation)
        cc_matches = re.finditer(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', text)
        for match in cc_matches:
            cc_number = re.sub(r'[-\s]', '', match.group())
            if self._validate_luhn(cc_number):
                threats.append(ThreatEvidence(
                    type="credit_card",
                    evidence=f"Credit card number detected: {match.group()}",
                    confidence=0.9,
                    severity="critical"
                ))
        
        # SSN patterns
        ssn_matches = re.finditer(r'\b\d{3}-?\d{2}-?\d{4}\b', text)
        for match in ssn_matches:
            threats.append(ThreatEvidence(
                type="ssn",
                evidence=f"SSN detected: {match.group()}",
                confidence=0.85,
                severity="critical"
            ))
        
        # CVV patterns (3-4 digits, context-aware)
        cvv_matches = re.finditer(r'\b(?:cvv|cvc|security code|card code)[\s:]*(\d{3,4})\b', text, re.IGNORECASE)
        for match in cvv_matches:
            threats.append(ThreatEvidence(
                type="cvv",
                evidence=f"CVV code detected: {match.group()}",
                confidence=0.9,
                severity="critical"
            ))
        
        return threats
    
    def _detect_high_entropy_patterns(self, text: str) -> List[ThreatEvidence]:
        """Detect high-entropy strings that might be gift card codes"""
        threats = []
        
        # Look for potential gift card codes (high entropy alphanumeric strings)
        potential_codes = re.findall(r'\b[A-Z0-9]{10,20}\b', text)
        
        for code in potential_codes:
            entropy = self._calculate_entropy(code)
            
            # High entropy suggests random generation (like gift card codes)
            if entropy > 3.5 and len(code) >= 12:
                # Check if it's in a suspicious context
                context_words = ["gift card", "card code", "redeem", "activation", "voucher"]
                context_found = any(word in text.lower() for word in context_words)
                
                confidence = 0.7 if context_found else 0.5
                severity = "high" if context_found else "medium"
                
                threats.append(ThreatEvidence(
                    type="gift_card_code",
                    evidence=f"Potential gift card code: {code}",
                    confidence=confidence,
                    severity=severity
                ))
        
        return threats
    
    def _detect_scam_phrases(self, text: str) -> List[ThreatEvidence]:
        """Detect common scam phrases"""
        threats = []
        text_lower = text.lower()
        
        # High-risk scam phrases
        high_risk_phrases = [
            "overpay and send gift cards",
            "refund via gift card",
            "wire money to",
            "send gift cards for refund",
            "pay with gift cards",
            "buy gift cards and send codes",
            "verification fee required",
            "advance fee required",
            "send money via western union",
            "send bitcoin for verification"
        ]
        
        # Medium-risk phrases
        medium_risk_phrases = [
            "urgent action required",
            "account will be closed",
            "verify your account immediately",
            "click here to verify",
            "suspended account",
            "unusual activity detected",
            "confirm your identity",
            "update payment information"
        ]
        
        # Check high-risk phrases
        for phrase in high_risk_phrases:
            if phrase in text_lower:
                threats.append(ThreatEvidence(
                    type="scam_phrase",
                    evidence=f"High-risk scam phrase detected: '{phrase}'",
                    confidence=0.9,
                    severity="critical"
                ))
        
        # Check medium-risk phrases
        for phrase in medium_risk_phrases:
            if phrase in text_lower:
                threats.append(ThreatEvidence(
                    type="phishing_phrase",
                    evidence=f"Phishing phrase detected: '{phrase}'",
                    confidence=0.7,
                    severity="medium"
                ))
        
        return threats
    
    def _classify_intent_with_gemini(self, text: str) -> List[ThreatEvidence]:
        """Use Gemini to classify intent for sophisticated threats"""
        threats = []
        
        try:
            system_prompt = """You are a fraud detection specialist. Analyze the following text and classify it into one of these categories:

Categories:
- refund_scam: Attempts to trick users into paying for fake refunds
- account_takeover: Attempts to gain unauthorized access to accounts  
- pii_risk: Requests for sensitive personal information
- safe: Normal, legitimate communication

Respond with ONLY a JSON object:
{"category": "category_name", "confidence": 0.85, "reasoning": "brief explanation"}

Focus on detecting sophisticated social engineering attempts."""

            user_message = f"Analyze this text: '{text}'"
            messages = [{"role": "user", "content": user_message}]
            
            response = chat(messages, system=system_prompt)
            
            import json
            # Handle markdown-wrapped JSON responses
            response_text = response.strip()
            if response_text.startswith('```json'):
                # Extract JSON from markdown code block
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                # Handle generic code blocks
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]).strip()
            
            result = json.loads(response_text)
            
            category = result.get("category", "safe")
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "")
            
            # Map categories to threat levels
            if category == "refund_scam":
                threats.append(ThreatEvidence(
                    type="refund_scam",
                    evidence=f"Refund scam detected: {reasoning}",
                    confidence=confidence,
                    severity="critical"
                ))
            elif category == "account_takeover":
                threats.append(ThreatEvidence(
                    type="account_takeover",
                    evidence=f"Account takeover attempt: {reasoning}",
                    confidence=confidence,
                    severity="critical"
                ))
            elif category == "pii_risk":
                threats.append(ThreatEvidence(
                    type="pii_request",
                    evidence=f"Suspicious PII request: {reasoning}",
                    confidence=confidence,
                    severity="high"
                ))
            
        except Exception as e:
            logger.error(f"Error in Gemini intent classification: {e}")
        
        return threats
    
    def _make_decision(self, threats: List[ThreatEvidence]) -> Tuple[str, Optional[Dict[str, str]]]:
        """
        Make overall decision based on detected threats using weighted risk scoring
        """
        if not threats:
            return "pass", None
        
        # Calculate weighted risk score
        risk_score = self._calculate_risk_score(threats)
        
        # Count threats by severity for additional context
        critical_count = sum(1 for t in threats if t.severity == "critical")
        high_count = sum(1 for t in threats if t.severity == "high")
        medium_count = sum(1 for t in threats if t.severity == "medium")
        low_count = sum(1 for t in threats if t.severity == "low")
        
        # Get highest severity threat types for context-specific guidance
        threat_types = [t.type for t in threats]
        highest_severity_threats = [t for t in threats if t.severity == "critical"]
        if not highest_severity_threats:
            highest_severity_threats = [t for t in threats if t.severity == "high"]
        
        logger.info(f"Risk assessment: score={risk_score:.2f}, critical={critical_count}, high={high_count}, medium={medium_count}, low={low_count}")
        
        # Enhanced decision logic with risk score thresholds
        if risk_score >= 8.0 or critical_count > 0:
            # Immediate block for critical threats or very high risk
            return "block", self._get_next_step_for_threats(highest_severity_threats, "critical")
            
        elif risk_score >= 6.0 or (high_count >= 2) or (high_count >= 1 and medium_count >= 3):
            # Block for high cumulative risk
            return "block", self._get_next_step_for_threats(highest_severity_threats, "high")
            
        elif risk_score >= 4.0 or high_count >= 1 or medium_count >= 2:
            # Warning for moderate risk
            return "warn", self._get_next_step_for_threats(highest_severity_threats, "medium")
            
        elif risk_score >= 2.0 or medium_count >= 1 or low_count >= 3:
            # Pass with monitoring for low risk
            return "pass", {
                "label": "Low Risk Detected - Monitor Activity",
                "url": "/security/monitor"
            }
        else:
            # Clean pass
            return "pass", None
    
    def _calculate_risk_score(self, threats: List[ThreatEvidence]) -> float:
        """
        Calculate weighted risk score based on threat severity, confidence, and type
        """
        if not threats:
            return 0.0
        
        # Severity weights
        severity_weights = {
            "critical": 10.0,
            "high": 6.0,
            "medium": 3.0,
            "low": 1.0
        }
        
        # Threat type multipliers (some threats are inherently more dangerous)
        type_multipliers = {
            "credit_card": 1.5,
            "ssn": 1.5,
            "cvv": 1.4,
            "scam_phrase": 1.3,
            "refund_scam": 1.4,
            "account_takeover": 1.3,
            "pii_request": 1.2,
            "gift_card_code": 1.1,
            "phishing_phrase": 1.0,
            "pii_detected": 1.0,
            "system_error": 0.8
        }
        
        total_score = 0.0
        threat_count = len(threats)
        
        for threat in threats:
            # Base score from severity and confidence
            base_score = severity_weights.get(threat.severity, 1.0) * threat.confidence
            
            # Apply threat type multiplier
            type_multiplier = type_multipliers.get(threat.type, 1.0)
            threat_score = base_score * type_multiplier
            
            total_score += threat_score
        
        # Apply diminishing returns for multiple threats of same type
        unique_types = len(set(t.type for t in threats))
        diversity_factor = min(1.0, unique_types / threat_count + 0.3)
        
        # Apply escalation factor for multiple threats
        if threat_count > 1:
            escalation_factor = 1.0 + (threat_count - 1) * 0.2
            total_score *= escalation_factor
        
        # Apply diversity factor
        final_score = total_score * diversity_factor
        
        logger.debug(f"Risk calculation: base={total_score:.2f}, diversity={diversity_factor:.2f}, final={final_score:.2f}")
        
        return final_score
    
    def _get_next_step_for_threats(self, threats: List[ThreatEvidence], severity_level: str) -> Dict[str, str]:
        """
        Get context-specific next step guidance based on threat types
        """
        if not threats:
            return {"label": "Contact Support", "url": "/support"}
        
        # Categorize threats
        threat_types = set(t.type for t in threats)
        
        # PII exposure threats
        pii_threats = {"credit_card", "ssn", "cvv", "pii_detected", "pii_request"}
        if threat_types.intersection(pii_threats):
            if severity_level == "critical":
                return {
                    "label": "CRITICAL: PII Exposure Detected - Immediate Action Required",
                    "url": "/security/pii-breach"
                }
            else:
                return {
                    "label": "PII Risk Detected - Review and Secure Information",
                    "url": "/security/pii-guidance"
                }
        
        # Scam/fraud threats
        scam_threats = {"scam_phrase", "refund_scam", "account_takeover", "phishing_phrase"}
        if threat_types.intersection(scam_threats):
            if severity_level == "critical":
                return {
                    "label": "FRAUD ALERT: Scam Detected - Do Not Proceed",
                    "url": "/security/fraud-alert"
                }
            else:
                return {
                    "label": "Potential Scam Detected - Verify Before Proceeding",
                    "url": "/security/scam-guidance"
                }
        
        # Gift card/payment threats
        payment_threats = {"gift_card_code"}
        if threat_types.intersection(payment_threats):
            return {
                "label": "Suspicious Payment Method - Verify Legitimacy",
                "url": "/security/payment-verification"
            }
        
        # Default based on severity
        if severity_level == "critical":
            return {
                "label": "Critical Security Threat - Contact Support Immediately",
                "url": "/security/critical-support"
            }
        elif severity_level == "high":
            return {
                "label": "High Risk Activity - Security Review Required",
                "url": "/security/high-risk-review"
            }
        else:
            return {
                "label": "Security Warning - Proceed with Caution",
                "url": "/security/general-guidance"
            }
    
    def _get_safety_citations(self, threats: List[ThreatEvidence]) -> List[Dict[str, str]]:
        """Get safety guidance citations from knowledge base"""
        citations = []
        
        if not all([self.retriever, self.embedder]):
            return citations
        
        try:
            # Create query based on threat types
            threat_types = [t.type for t in threats]
            query = f"safety guidance for {', '.join(set(threat_types))} threats"
            
            # Retrieve relevant safety documents
            from app.rag.core import retrieve
            results = retrieve(self.retriever, self.embedder, query, k=3)
            
            for result in results:
                citations.append({
                    "source": result.get("filename", "Safety Guidelines"),
                    "snippet": result.get("snippet", "")[:200] + "..."
                })
                
        except Exception as e:
            logger.error(f"Error retrieving safety citations: {e}")
        
        return citations
    
    def _redact_sensitive_content(self, text: str, threats: List[ThreatEvidence]) -> str:
        """Redact sensitive content from text"""
        if not self.anonymizer:
            return text
        
        try:
            # Use Presidio to redact PII
            analyzer_results = self.analyzer.analyze(text=text, language='en')
            
            # Configure anonymization
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators={
                    "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[CREDIT_CARD]"}),
                    "SSN": OperatorConfig("replace", {"new_value": "[SSN]"}),
                    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
                    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
                    "PERSON": OperatorConfig("replace", {"new_value": "[NAME]"}),
                    "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})
                }
            )
            
            redacted_text = anonymized_result.text
            
            # Additional redaction for gift card codes and other patterns
            redacted_text = re.sub(r'\b[A-Z0-9]{10,20}\b', '[GIFT_CARD_CODE]', redacted_text)
            
            return redacted_text
            
        except Exception as e:
            logger.error(f"Error in text redaction: {e}")
            return text
    
    def _validate_luhn(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm"""
        try:
            digits = [int(d) for d in card_number if d.isdigit()]
            if len(digits) < 13 or len(digits) > 19:
                return False
            
            checksum = 0
            for i, digit in enumerate(reversed(digits)):
                if i % 2 == 1:
                    digit *= 2
                    if digit > 9:
                        digit -= 9
                checksum += digit
            
            return checksum % 10 == 0
        except:
            return False
    
    def _calculate_entropy(self, string: str) -> float:
        """Calculate Shannon entropy of a string"""
        if not string:
            return 0
        
        # Count character frequencies
        char_counts = {}
        for char in string:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0
        length = len(string)
        for count in char_counts.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _initialize_scam_patterns(self) -> Dict[str, List[str]]:
        """Initialize scam detection patterns"""
        return {
            "refund_scams": [
                r"overpay.*gift card",
                r"refund.*gift card",
                r"send.*gift card.*refund",
                r"buy.*gift card.*verification"
            ],
            "wire_fraud": [
                r"wire.*money.*to",
                r"western union.*transfer",
                r"send.*money.*urgent"
            ],
            "crypto_scams": [
                r"send.*bitcoin.*verification",
                r"crypto.*wallet.*verification",
                r"bitcoin.*address.*refund"
            ]
        }
    
    def _initialize_entropy_patterns(self) -> Dict[str, float]:
        """Initialize entropy thresholds for different pattern types"""
        return {
            "gift_card_codes": 3.5,
            "crypto_addresses": 4.0,
            "random_strings": 3.0
        }

# Unit tests for common scam scenarios
def test_trustshield():
    """Deterministic unit tests for TrustShield"""
    shield = TrustShield()
    
    test_cases = [
        {
            "name": "Refund Scam",
            "text": "You overpaid $500. Please buy gift cards and send the codes for your refund.",
            "expected_decision": "block",
            "expected_threats": ["scam_phrase", "refund_scam"]
        },
        {
            "name": "Credit Card Exposure",
            "text": "My card number is 4532-1234-5678-9012 and CVV is 123",
            "expected_decision": "block", 
            "expected_threats": ["credit_card", "cvv"]
        },
        {
            "name": "SSN Exposure",
            "text": "My social security number is 123-45-6789",
            "expected_decision": "block",
            "expected_threats": ["ssn"]
        },
        {
            "name": "Phishing Attempt",
            "text": "Urgent action required! Your account will be closed. Click here to verify immediately.",
            "expected_decision": "warn",
            "expected_threats": ["phishing_phrase"]
        },
        {
            "name": "Safe Message",
            "text": "Hello, I'd like to know about your payment processing fees.",
            "expected_decision": "pass",
            "expected_threats": []
        }
    ]
    
    results = []
    for case in test_cases:
        result = shield.scan(case["text"])
        
        # Check decision
        decision_correct = result["decision"] == case["expected_decision"]
        
        # Check threat types
        detected_types = [threat.type for threat in result["reasons"]]
        threats_correct = all(expected in detected_types for expected in case["expected_threats"])
        
        test_result = {
            "name": case["name"],
            "passed": decision_correct and threats_correct,
            "expected_decision": case["expected_decision"],
            "actual_decision": result["decision"],
            "expected_threats": case["expected_threats"],
            "detected_threats": detected_types
        }
        
        results.append(test_result)
        
        print(f"Test '{case['name']}': {'PASS' if test_result['passed'] else 'FAIL'}")
        if not test_result['passed']:
            print(f"  Expected: {case['expected_decision']}, Got: {result['decision']}")
            print(f"  Expected threats: {case['expected_threats']}")
            print(f"  Detected threats: {detected_types}")
    
    return results

if __name__ == "__main__":
    # Run tests
    test_results = test_trustshield()
    passed = sum(1 for r in test_results if r["passed"])
    total = len(test_results)
    print(f"\nTest Results: {passed}/{total} passed")
