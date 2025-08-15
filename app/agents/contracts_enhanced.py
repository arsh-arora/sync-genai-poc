"""
Contract Intelligence - Hybrid Rules + LLM Approach
First applies rules-based clause extraction from contracts_lexicon.yml,
then falls back to LLM + Tavily web search for insufficient results
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple

from pydantic import BaseModel

from app.llm.gemini import chat
from app.rag.core import retrieve
from app.tools.tavily_search import web_search_into_docstore

logger = logging.getLogger(__name__)

# Enhanced Pydantic models for systematic clause extraction
class ClauseChip(BaseModel):
    name: str  # apr, promotional_terms, late_fees, etc.
    snippet: str  # captured sentence/paragraph
    start: int  # character offset start
    end: int  # character offset end
    flags: List[str] = []  # risk flags

class RiskFlag(BaseModel):
    type: str  # missing_clause, conflicting_numbers, promo_ambiguity
    description: str
    severity: str  # low, medium, high
    clause_type: Optional[str] = None

class ContractResponse(BaseModel):
    response: str  # Plain English answer with 1-2 clause cites
    metadata: Dict[str, Any]  # Contains ui_cards, handoffs, risk_flags

class ContractIntelligence:
    """
    Hybrid Contract Intelligence: Rules-first, then LLM+Tavily fallback
    """
    
    def __init__(self, docstore=None, embedder=None, retriever=None, rules_loader=None):
        """Initialize ContractIntelligence with RAG components and rules-based processing"""
        self.docstore = docstore
        self.embedder = embedder
        self.retriever = retriever
        self.rules_loader = rules_loader
        
        # Load contract rules from centralized loader or use defaults
        if rules_loader:
            self.contract_rules = rules_loader.get_rules('contracts_lexicon') or {}
            logger.info("ContractIntelligence loaded rules from centralized rules loader")
        else:
            self.contract_rules = self._load_fallback_rules()
        
        # Extract rule components
        self.clauses = self.contract_rules.get("clauses", {})
        self.risk_flags = self.contract_rules.get("risk_flags", {})
        self.ask_legal_triggers = self.contract_rules.get("ask_legal_triggers", [])
        
    def _load_fallback_rules(self) -> Dict[str, Any]:
        """Fallback rules if centralized loader not available"""
        return {
            "clauses": {
                "apr": {"keywords": ["APR", "annual percentage", "interest rate", "finance charge"]},
                "promotional_terms": {"keywords": ["equal payment", "deferred interest", "paid in full", "promo period", "no interest"]},
                "late_fees": {"keywords": ["late fee", "returned payment", "penalty", "grace period"]},
                "dispute_resolution": {"keywords": ["arbitration", "binding", "small claims", "waiver", "class action"]},
                "data_sharing": {"keywords": ["share", "third party", "marketing partners", "opt out", "sell"]}
            },
            "risk_flags": {
                "missing_clause": "Key clause not found",
                "conflicting_numbers": "Conflicting numerical terms detected", 
                "promo_ambiguity": "Promotional wording lacks 'paid in full' condition"
            },
            "ask_legal_triggers": ["conflicting_numbers", "promo_ambiguity", "missing_clause"]
        }
    
    def analyze_contract(self, contract_text: str, question: Optional[str] = None) -> ContractResponse:
        """
        Hybrid analysis: Rules-first, then LLM+Tavily fallback
        
        Args:
            contract_text: The contract document text
            question: Specific question about the contract (optional)
            
        Returns:
            ContractResponse with plain English answer, clause chips, and handoffs
        """
        try:
            logger.info(f"Starting hybrid contract analysis: {len(contract_text)} chars")
            
            # PHASE 1: Rules-based extraction
            clause_chips = self._extract_clauses(contract_text)
            risk_flags = self._analyze_risks(clause_chips, contract_text)
            
            logger.info(f"Rules phase: {len(clause_chips)} clauses, {len(risk_flags)} risks")
            
            # PHASE 2: Check if rules results are sufficient
            is_sufficient = self._assess_sufficiency(clause_chips, question)
            
            # PHASE 3: LLM+Tavily fallback if insufficient
            llm_response = ""
            citations = []
            
            if not is_sufficient:
                logger.info("Rules insufficient, using LLM+Tavily fallback")
                llm_response, additional_chips, citations = self._llm_fallback(
                    contract_text, question, clause_chips
                )
                clause_chips.extend(additional_chips)
            
            # PHASE 4: Generate response
            if question:
                response_text = self._answer_question(question, clause_chips, llm_response)
            else:
                response_text = self._generate_summary(clause_chips, risk_flags, llm_response)
            
            # PHASE 5: Detect handoffs
            handoffs = self._detect_handoffs(question, risk_flags, clause_chips)
            needs_legal_review = self._check_legal_triggers(risk_flags)
            
            return ContractResponse(
                response=response_text,
                metadata={
                    "ui_cards": [chip.model_dump() for chip in clause_chips],
                    "handoffs": handoffs,
                    "risk_flags": [flag.model_dump() for flag in risk_flags],
                    "needs_legal_review": needs_legal_review,
                    "method_used": "hybrid" if not is_sufficient else "rules_only",
                    "citations": [{"source": c.get("source", ""), "snippet": c.get("snippet", "")} for c in citations]
                }
            )
            
        except Exception as e:
            logger.error(f"Contract analysis error: {e}")
            return ContractResponse(
                response="Contract analysis failed. Please try again or contact support.",
                metadata={"error": str(e), "ui_cards": [], "handoffs": [], "risk_flags": []}
            )
    
    def _extract_clauses(self, contract_text: str) -> List[ClauseChip]:
        """Extract clauses using keywords from contracts_lexicon.yml with precise locations"""
        clause_chips = []
        
        for clause_name, clause_config in self.clauses.items():
            keywords = clause_config.get("keywords", [])
            
            # Find keyword matches in text
            for keyword in keywords:
                # Case-insensitive search for keyword
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                
                for match in pattern.finditer(contract_text):
                    # Capture surrounding context (sentence or paragraph)
                    snippet, start, end = self._capture_context(contract_text, match.start(), match.end())
                    
                    # Check for duplicates (same clause type in similar location)
                    if not self._is_duplicate_clause(clause_chips, clause_name, start, end):
                        clause_chip = ClauseChip(
                            name=clause_name,
                            snippet=snippet,
                            start=start,
                            end=end,
                            flags=[]
                        )
                        clause_chips.append(clause_chip)
        
        return clause_chips
    
    def _capture_context(self, text: str, keyword_start: int, keyword_end: int) -> Tuple[str, int, int]:
        """Capture nearest sentence/paragraph around keyword with char offsets"""
        
        # Find sentence boundaries around the keyword
        sentence_start = keyword_start
        sentence_end = keyword_end
        
        # Look backward for sentence start (period, exclamation, or start of text)
        for i in range(keyword_start - 1, max(0, keyword_start - 500), -1):
            if text[i] in '.!?\n' and i > 0:
                sentence_start = i + 1
                break
            elif i == 0:
                sentence_start = 0
                break
        
        # Look forward for sentence end (period, exclamation, or end of text)
        for i in range(keyword_end, min(len(text), keyword_end + 500)):
            if text[i] in '.!?\n':
                sentence_end = i + 1
                break
            elif i == len(text) - 1:
                sentence_end = len(text)
                break
        
        # If sentence is too short, expand to paragraph
        snippet = text[sentence_start:sentence_end].strip()
        if len(snippet) < 50:
            # Expand to paragraph (double newline or larger context)
            para_start = max(0, keyword_start - 200)
            para_end = min(len(text), keyword_end + 200)
            
            # Look for natural paragraph breaks
            for i in range(keyword_start - 1, para_start, -1):
                if text[i:i+2] == '\n\n':
                    para_start = i + 2
                    break
            
            for i in range(keyword_end, para_end):
                if text[i:i+2] == '\n\n':
                    para_end = i
                    break
            
            snippet = text[para_start:para_end].strip()
            sentence_start = para_start
            sentence_end = para_end
        
        return snippet, sentence_start, sentence_end
    
    def _is_duplicate_clause(self, existing_chips: List[ClauseChip], clause_name: str, 
                           start: int, end: int) -> bool:
        """Check if clause already captured in similar location"""
        for chip in existing_chips:
            if (chip.name == clause_name and 
                abs(chip.start - start) < 100):  # Within 100 chars
                return True
        return False
    
    def _analyze_risks(self, clause_chips: List[ClauseChip], contract_text: str) -> List[RiskFlag]:
        """Analyze risks: missing_clause, conflicting_numbers, promo_ambiguity"""
        risk_flags = []
        
        # Check for missing clauses
        expected_clauses = set(self.clauses.keys())
        found_clauses = set(chip.name for chip in clause_chips)
        missing_clauses = expected_clauses - found_clauses
        
        for missing in missing_clauses:
            risk_flags.append(RiskFlag(
                type="missing_clause",
                description=f"Missing {missing.replace('_', ' ')} clause",
                severity="medium",
                clause_type=missing
            ))
        
        # Check for conflicting numbers
        numbers_by_clause = {}
        for chip in clause_chips:
            numbers = re.findall(r'(\d+(?:\.\d+)?%?)', chip.snippet)
            if numbers:
                if chip.name not in numbers_by_clause:
                    numbers_by_clause[chip.name] = []
                numbers_by_clause[chip.name].extend(numbers)
        
        for clause_name, numbers in numbers_by_clause.items():
            unique_numbers = set(numbers)
            if len(unique_numbers) > 1 and clause_name in ['apr', 'late_fees']:
                risk_flags.append(RiskFlag(
                    type="conflicting_numbers",
                    description=f"Multiple different {clause_name} values found: {', '.join(unique_numbers)}",
                    severity="high",
                    clause_type=clause_name
                ))
        
        # Check for promotional ambiguity
        promo_chips = [chip for chip in clause_chips if chip.name == "promotional_terms"]
        for chip in promo_chips:
            snippet_lower = chip.snippet.lower()
            if ("deferred interest" in snippet_lower or "no interest" in snippet_lower):
                if "paid in full" not in snippet_lower:
                    risk_flags.append(RiskFlag(
                        type="promo_ambiguity",
                        description="Promotional terms lack 'paid in full' condition clarity",
                        severity="medium",
                        clause_type="promotional_terms"
                    ))
                    # Add flag to the clause chip
                    chip.flags.append("promo_ambiguity")
        
        return risk_flags
    
    def _check_legal_triggers(self, risk_flags: List[RiskFlag]) -> bool:
        """Check if any risk flags trigger legal review"""
        risk_types = [flag.type for flag in risk_flags]
        return any(trigger in risk_types for trigger in self.ask_legal_triggers)
    
    def _assess_sufficiency(self, clause_chips: List[ClauseChip], question: Optional[str]) -> bool:
        """Determine if rules-based results are sufficient"""
        # If specific question asked, check if we have relevant clauses
        if question:
            question_lower = question.lower()
            relevant_found = False
            
            # Check for question-relevant clauses
            if any(word in question_lower for word in ["apr", "interest", "rate"]):
                relevant_found = any(chip.name == "apr" for chip in clause_chips)
            elif any(word in question_lower for word in ["late", "fee", "penalty"]):
                relevant_found = any(chip.name == "late_fees" for chip in clause_chips)
            elif any(word in question_lower for word in ["promo", "promotional", "deferred"]):
                relevant_found = any(chip.name == "promotional_terms" for chip in clause_chips)
            elif any(word in question_lower for word in ["dispute", "arbitration"]):
                relevant_found = any(chip.name == "dispute_resolution" for chip in clause_chips)
            elif any(word in question_lower for word in ["data", "sharing", "privacy"]):
                relevant_found = any(chip.name == "data_sharing" for chip in clause_chips)
            else:
                # Generic question - sufficient if we have any clauses
                relevant_found = len(clause_chips) >= 2
            
            return relevant_found
        
        # For general analysis, sufficient if we found most expected clauses
        expected_clauses = len(self.clauses)
        found_clauses = len(set(chip.name for chip in clause_chips))
        
        return found_clauses >= (expected_clauses * 0.6)  # 60% threshold
    
    def _llm_fallback(self, contract_text: str, question: Optional[str], 
                     existing_chips: List[ClauseChip]) -> Tuple[str, List[ClauseChip], List[Dict]]:
        """Phase 2: LLM+Tavily fallback for insufficient results"""
        
        # Build context from existing rules results
        context = f"Rules-based analysis found {len(existing_chips)} clauses: "
        context += ", ".join(set(chip.name for chip in existing_chips))
        
        # LLM analysis with structured prompt
        if question:
            system_prompt = f"""You are a contract analysis expert. The initial rules-based analysis was insufficient.

CONTEXT: {context}

Answer the specific question about this contract. Focus on finding information not captured by the rules.
Provide specific quotes from the contract with location context.
If you cannot find the answer in the contract, say so clearly."""

            user_message = f"Question: {question}\n\nContract text:\n{contract_text[:6000]}"
        else:
            system_prompt = f"""You are a contract analysis expert. The initial rules-based analysis was insufficient.

CONTEXT: {context}

Analyze this contract for key terms not captured by basic keyword matching.
Look for: complex fee structures, SLA terms, termination conditions, liability clauses, warranties.
Provide specific quotes and explain their significance."""

            user_message = f"Analyze this contract:\n\n{contract_text[:6000]}"
        
        try:
            messages = [{"role": "user", "content": user_message}]
            llm_response = chat(messages, system=system_prompt)
            logger.info("LLM analysis completed")
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            llm_response = "LLM analysis unavailable"
        
        # Try to get additional context from RAG + Tavily
        citations = []
        additional_chips = []
        
        if self.retriever and self.embedder:
            try:
                # RAG search for contract terms
                query = question if question else "contract terms analysis"
                rag_results = retrieve(self.retriever, self.embedder, query, k=2)
                
                for result in rag_results:
                    citations.append({
                        "source": result.get("filename", "Knowledge Base"),
                        "snippet": result.get("snippet", "")[:200]
                    })
                
                # Tavily web search fallback
                if len(citations) < 2 and self.docstore:
                    try:
                        web_query = f"contract {question}" if question else "contract analysis terms"
                        web_docs = web_search_into_docstore(
                            self.docstore, self.embedder, web_query, max_results=2
                        )
                        
                        if web_docs:
                            web_results = retrieve(self.retriever, self.embedder, query, k=2)
                            for result in web_results:
                                citations.append({
                                    "source": result.get("filename", "Web Search"),
                                    "snippet": result.get("snippet", "")[:200]
                                })
                        
                    except Exception as e:
                        logger.warning(f"Tavily search failed: {e}")
                
            except Exception as e:
                logger.error(f"RAG search failed: {e}")
        
        return llm_response, additional_chips, citations

    def _answer_question(self, question: str, clause_chips: List[ClauseChip], 
                        llm_response: str) -> str:
        """Answer specific question with 1-2 clause citations"""
        question_lower = question.lower()
        
        # Find most relevant clauses based on question keywords
        relevant_chips = []
        relevance_scores = {}
        
        for chip in clause_chips:
            score = 0
            chip_text = chip.snippet.lower()
            
            # Score based on keyword overlap
            question_words = set(question_lower.split())
            chip_words = set(chip_text.split())
            common_words = question_words & chip_words
            score += len(common_words) * 2
            
            # Score based on clause type relevance
            if chip.name in question_lower:
                score += 5
            
            # Specific question patterns
            if "apr" in question_lower or "interest" in question_lower:
                if chip.name == "apr":
                    score += 10
            if "late" in question_lower or "fee" in question_lower:
                if chip.name == "late_fees":
                    score += 10
            if "promo" in question_lower or "promotional" in question_lower:
                if chip.name == "promotional_terms":
                    score += 10
            
            if score > 0:
                relevance_scores[chip] = score
        
        # Sort by relevance and take top 2
        sorted_chips = sorted(relevance_scores.keys(), 
                            key=lambda x: relevance_scores[x], reverse=True)
        relevant_chips = sorted_chips[:2]
        
        if not relevant_chips and not llm_response:
            return f"No specific information found for: '{question}'. The contract may not contain relevant terms."
        
        # Build response
        response_parts = []
        
        if relevant_chips:
            response_parts.append("**Contract Analysis:**\n")
            for i, chip in enumerate(relevant_chips, 1):
                key_info = self._extract_key_info(chip)
                response_parts.append(f"{i}. **{chip.name.replace('_', ' ').title()}:** {key_info}")
                response_parts.append(f"   *\"...{chip.snippet[:150]}...\"*\n")
        
        if llm_response and llm_response != "LLM analysis unavailable":
            response_parts.append("**Additional Analysis:**")
            response_parts.append(llm_response[:500] + ("..." if len(llm_response) > 500 else ""))
        
        return "\n".join(response_parts)
    
    def _extract_key_info(self, chip: ClauseChip) -> str:
        """Extract key information from clause snippet"""
        snippet = chip.snippet
        
        if chip.name == "apr":
            # Extract APR percentage
            numbers = re.findall(r'(\d+(?:\.\d+)?%)', snippet)
            if numbers:
                return f"Interest rate of {numbers[0]} identified"
            else:
                return "Interest rate terms present but specific rate not clearly stated"
        
        elif chip.name == "late_fees":
            # Extract fee amounts
            fees = re.findall(r'\$(\d+(?:\.\d+)?)', snippet)
            if fees:
                return f"Late fee of ${fees[0]} specified"
            else:
                return "Late fee terms present"
        
        elif chip.name == "promotional_terms":
            # Check for specific promotional periods
            months = re.findall(r'(\d+)\s*months?', snippet, re.IGNORECASE)
            if months:
                return f"Promotional period of {months[0]} months identified"
            else:
                return "Promotional financing terms available"
        
        elif chip.name == "dispute_resolution":
            if "arbitration" in snippet.lower():
                return "Arbitration required for dispute resolution"
            else:
                return "Dispute resolution process specified"
        
        elif chip.name == "data_sharing":
            if "opt out" in snippet.lower():
                return "Data sharing with opt-out option available"
            else:
                return "Data sharing practices disclosed"
        
        return f"{chip.name.replace('_', ' ').title()} terms identified"
    
    def _generate_summary(self, clause_chips: List[ClauseChip], risk_flags: List[RiskFlag], 
                         llm_response: str) -> str:
        """Generate contract analysis summary"""
        clause_types = set(chip.name for chip in clause_chips)
        
        summary_parts = []
        summary_parts.append(f"**Contract Analysis:** {len(clause_chips)} clauses identified across {len(clause_types)} categories.")
        
        if clause_types:
            categories = [name.replace('_', ' ').title() for name in sorted(clause_types)]
            summary_parts.append(f"**Categories:** {', '.join(categories)}")
        
        if risk_flags:
            high_risks = [f for f in risk_flags if f.severity == "high"]
            if high_risks:
                summary_parts.append(f"⚠️ **{len(high_risks)} High-Risk Items** require attention")
            else:
                summary_parts.append(f"📋 **{len(risk_flags)} Risk Items** identified")
        
        if llm_response and llm_response != "LLM analysis unavailable":
            summary_parts.append("**Enhanced Analysis:** Additional LLM insights included")
        
        return "\n".join(summary_parts)
    
    def _detect_handoffs(self, question: Optional[str], risk_flags: List[RiskFlag], 
                        clause_chips: List[ClauseChip]) -> List[str]:
        """Detect handoffs: back to dispute (remedy path) or imagegen/offer for compliant copy"""
        handoffs = []
        
        # Check question content for handoff triggers
        if question:
            question_lower = question.lower()
            
            # Dispute handoff triggers
            dispute_terms = ["dispute", "remedy", "resolution", "complaint", "chargeback", "refund"]
            if any(term in question_lower for term in dispute_terms):
                handoffs.append("dispute")
            
            # Offer/marketing handoff triggers
            offer_terms = ["copy", "marketing", "compliant", "disclosure", "advertisement"]
            if any(term in question_lower for term in offer_terms):
                handoffs.append("offerpilot")
            
            # Image generation for compliant copy
            image_terms = ["visual", "image", "graphic", "banner", "design"]
            if any(term in question_lower for term in image_terms):
                handoffs.append("imagegen")
        
        # Risk-based handoffs
        high_risk_flags = [f for f in risk_flags if f.severity == "high"]
        if high_risk_flags:
            handoffs.append("dispute")  # High risk may lead to disputes
        
        # Promotional ambiguity may need offer clarification
        promo_flags = [f for f in risk_flags if f.type == "promo_ambiguity"]
        if promo_flags:
            handoffs.append("offerpilot")
        
        return list(set(handoffs))  # Remove duplicates

# Backwards compatibility - create alias for existing import
ContractIntelligence_Enhanced = ContractIntelligence