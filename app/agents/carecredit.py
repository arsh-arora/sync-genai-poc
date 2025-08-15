"""
CareCredit Treatment Translator
Converts medical estimates into plain-language options with provider availability and financing plans
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.llm.gemini import chat
from app.agents.offerpilot import OfferPilot
from app.tools.tavily_search import web_search_into_docstore

logger = logging.getLogger(__name__)

# Pydantic models for structured data
class InsuranceInfo(BaseModel):
    deductible_left: float
    coinsurance: float

class LineItem(BaseModel):
    name: str
    unit_cost: float
    qty: int = 1
    subtotal: float
    procedure_code: Optional[str] = None

class Provider(BaseModel):
    name: str
    address: str
    phone: str
    next_appt_days: int

class FinancingOption(BaseModel):
    offer_id: str
    months: int
    apr: float
    monthly: float
    total_cost: float

class OOPPResult(BaseModel):
    estimated_total: float
    assumptions: Dict[str, Any]

class Citation(BaseModel):
    source: str
    snippet: str

class CreditResponse(BaseModel):
    response: str  # counts + 12-mo explanation + next steps
    metadata: Dict[str, Any]  # ui_cards and disclosures

@dataclass
class ParsedProcedure:
    """Represents a parsed medical/dental procedure"""
    procedure_code: Optional[str]
    name: str
    unit_cost: float
    qty: int = 1

class CareCredit:
    """
    CareCredit Treatment Translator for medical/dental estimates
    """
    
    def __init__(self, docstore=None, embedder=None, retriever=None, rules_loader=None):
        """Initialize CareCredit agent with required components"""
        self.docstore = docstore
        self.embedder = embedder
        self.retriever = retriever
        self.rules_loader = rules_loader
        
        # Initialize OfferPilot for financing offers
        self.offer_pilot = OfferPilot(docstore, embedder, retriever)
        
        # Load provider data and rules
        self._load_provider_data()
        self._load_carecredit_rules()
    
    def _load_provider_data(self):
        """Load healthcare provider directory"""
        try:
            providers_path = Path("synchrony-demo-rules-repo/fixtures/providers.json")
            with open(providers_path, 'r') as f:
                data = json.load(f)
                self.providers = data["providers"]
                self.specialties = data.get("specialties", {})
                self.procedure_mappings = data.get("procedure_mappings", {})
            logger.info(f"Loaded {len(self.providers)} healthcare providers")
        except Exception as e:
            logger.error(f"Failed to load provider data: {e}")
            self.providers = []
            self.specialties = {}
            self.procedure_mappings = {}
    
    def _load_carecredit_rules(self):
        """Load CareCredit rules from YAML"""
        try:
            if self.rules_loader:
                self.carecredit_rules = self.rules_loader.get_rules('carecredit') or {}
                logger.info("Loaded CareCredit rules from rules_loader")
            else:
                self.carecredit_rules = {}
                logger.warning("No rules_loader provided - using defaults")
        except Exception as e:
            logger.error(f"Failed to load CareCredit rules: {e}")
            self.carecredit_rules = {}
    
    def process_estimate(
        self,
        estimate_text: str,
        location: Optional[str] = None,
        insurance: Optional[Dict[str, float]] = None
    ) -> CreditResponse:
        """
        Main processing pipeline: treatment → providers → OOPP → financing
        
        Args:
            estimate_text: Medical/dental estimate text
            location: Patient location for provider search
            insurance: Insurance info with deductible_left and coinsurance
            
        Returns:
            CreditResponse with counts + 12-mo explanation + next steps + metadata
        """
        try:
            logger.info("Processing CareCredit treatment estimate")
            
            # Step 0: Validate that this is actually a medical/dental estimate
            if not self._is_medical_estimate(estimate_text):
                logger.info("Input is not a medical estimate - treating as general CareCredit query")
                return self._handle_general_query(estimate_text)
            
            # Step 1: Parse estimate into line items (deterministic parser)
            parsed_items = self.estimate_parse_table(estimate_text)
            logger.info(f"Parsed {len(parsed_items)} line items from estimate")
            
            # Step 2: Calculate total cost
            total_cost = sum(item.unit_cost * item.qty for item in parsed_items)
            
            # Step 3: Identify specialty from procedures
            specialty = self._identify_specialty(parsed_items, estimate_text)
            logger.info(f"Identified specialty: {specialty}")
            
            # Step 4: Provider shortlist (max 3) filtered by specialty/city
            providers = self.providers_search({"specialty": specialty, "location": location})
            
            # Step 5: OOPP estimate using rules/carecredit.yml defaults
            oopp_result = self.oopp_simulate(total_cost, None, insurance)
            
            # Step 6: Financing pairing - attach EP/DI options
            financing_offers = self._get_carecredit_offers(total_cost)
            
            # Step 7: Generate response with counts + 12-mo explanation + next steps
            response_text = self._generate_response_text(parsed_items, providers, financing_offers, oopp_result, total_cost)
            
            # Step 8: Build UI cards for metadata
            ui_cards = self._build_ui_cards(parsed_items, providers, financing_offers, total_cost)
            
            # Step 9: Always append carecredit_generic disclosure
            disclosures = self._get_disclosures(financing_offers)
            
            return CreditResponse(
                response=response_text,
                metadata={
                    "ui_cards": ui_cards,
                    "disclosures": disclosures,
                    "total_procedures": len(parsed_items),
                    "total_cost": total_cost,
                    "providers_found": len(providers),
                    "financing_options": len(financing_offers),
                    "oopp_estimate": oopp_result.estimated_total,
                    "specialty": specialty
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing CareCredit estimate: {e}")
            return CreditResponse(
                response=f"Error processing estimate: {str(e)}",
                metadata={
                    "ui_cards": [],
                    "disclosures": ["carecredit_generic"],
                    "error": str(e)
                }
            )
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for supervisor integration
        
        Args:
            query: User query about healthcare financing
            
        Returns:
            Dict with response, metadata, sources for supervisor
        """
        try:
            # Extract location if mentioned
            location = self._extract_location(query)
            
            # Extract insurance info if mentioned
            insurance = self._extract_insurance(query)
            
            # Process the estimate
            result = self.process_estimate(query, location, insurance)
            
            # Convert to supervisor-compatible format
            return {
                "response": result.response,
                "confidence": 0.8,
                "sources": [],
                "metadata": result.metadata
            }
            
        except Exception as e:
            logger.error(f"CareCredit process_query error: {e}")
            return {
                "response": f"Error processing CareCredit query: {str(e)}",
                "confidence": 0.2,
                "sources": [],
                "metadata": {"error": str(e)}
            }
    
    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location from query text"""
        import re
        
        # Look for city patterns
        city_patterns = [
            r"in ([A-Z][a-z]+ ?[A-Z]?[a-z]*)",  # "in New York", "in Chicago"
            r"I'm in ([A-Z][a-z]+ ?[A-Z]?[a-z]*)",  # "I'm in New York"
            r"([A-Z][a-z]+) area",  # "Chicago area"
        ]
        
        for pattern in city_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).title()
        
        return None
    
    def _extract_insurance(self, query: str) -> Optional[Dict[str, float]]:
        """Extract insurance information from query text"""
        import re
        
        insurance = {}
        
        # Look for deductible
        deductible_match = re.search(r'\$?(\d+) deductible', query, re.IGNORECASE)
        if deductible_match:
            insurance["deductible_left"] = float(deductible_match.group(1))
        
        # Look for coinsurance
        coinsurance_match = re.search(r'(\d+)% coinsurance', query, re.IGNORECASE)
        if coinsurance_match:
            insurance["coinsurance"] = float(coinsurance_match.group(1)) / 100
        
        # Look for "no insurance"
        if re.search(r'no insurance|without insurance', query, re.IGNORECASE):
            return None
        
        return insurance if insurance else None
    
    def estimate_parse_table(self, text: str) -> List[ParsedProcedure]:
        """
        Parse medical estimate table using Gemini with regex fallback
        
        Args:
            text: Estimate text to parse
            
        Returns:
            List of parsed procedures
        """
        try:
            # First, try Gemini-powered parsing with table hints
            system_prompt = """You are a medical billing expert. Parse the following medical/dental estimate into a structured list.

Extract each procedure/service with:
- Procedure code (if present, like D0120, 99213, etc.)
- Procedure name/description  
- Unit cost (dollar amount)
- Quantity (default to 1 if not specified)

Return as JSON array with objects containing: procedure_code, name, unit_cost, qty

Focus on actual billable procedures, ignore totals and administrative text."""

            user_message = f"Medical/dental estimate to parse:\n\n{text}"
            messages = [{"role": "user", "content": user_message}]
            
            response = chat(messages, system=system_prompt)
            
            # Try to parse Gemini response as JSON
            try:
                # Debug: Print the raw response
                logger.info(f"Raw Gemini response: {repr(response[:200])}")
                
                # Clean the response - handle markdown code blocks
                response_clean = self._extract_json(response)
                
                logger.info(f"Cleaned response: {repr(response_clean[:200])}")
                
                parsed_json = json.loads(response_clean)
                procedures = []
                
                for item in parsed_json:
                    if isinstance(item, dict) and "name" in item and "unit_cost" in item:
                        procedures.append(ParsedProcedure(
                            procedure_code=item.get("procedure_code"),
                            name=item["name"],
                            unit_cost=float(item["unit_cost"]),
                            qty=int(item.get("qty", 1))
                        ))
                
                if procedures:
                    logger.info("Successfully parsed estimate using Gemini")
                    return procedures
                    
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to parse Gemini JSON response: {e}")
            
            # Fallback to regex parsing
            logger.info("Falling back to regex parsing")
            return self._regex_parse_estimate(text)
            
        except Exception as e:
            logger.error(f"Error parsing estimate: {e}")
            return self._regex_parse_estimate(text)
    
    def _regex_parse_estimate(self, text: str) -> List[ParsedProcedure]:
        """
        Fallback regex-based estimate parsing
        
        Args:
            text: Estimate text
            
        Returns:
            List of parsed procedures
        """
        procedures = []
        
        # Common patterns for medical estimates
        patterns = [
            # Pattern 1: Code | Description | $Amount
            r'([A-Z]?\d{4,5})\s*[|\-\s]+([^|\$]+?)\s*[|\-\s]*\$?\s*(\d+(?:\.\d{2})?)',
            # Pattern 2: Description $Amount
            r'([A-Za-z][^$\n]{10,50}?)\s+\$(\d+(?:\.\d{2})?)',
            # Pattern 3: Description ... $Amount
            r'([A-Za-z][^$\n]{5,40}?)\.{2,}\$?(\d+(?:\.\d{2})?)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                try:
                    if len(match) == 3:  # Code, description, amount
                        code, name, amount = match
                        procedures.append(ParsedProcedure(
                            procedure_code=code.strip(),
                            name=name.strip(),
                            unit_cost=float(amount),
                            qty=1
                        ))
                    elif len(match) == 2:  # Description, amount
                        name, amount = match
                        procedures.append(ParsedProcedure(
                            procedure_code=None,
                            name=name.strip(),
                            unit_cost=float(amount),
                            qty=1
                        ))
                except (ValueError, IndexError):
                    continue
        
        # Remove duplicates and clean up
        seen_names = set()
        clean_procedures = []
        for proc in procedures:
            if proc.name.lower() not in seen_names and proc.unit_cost > 0:
                seen_names.add(proc.name.lower())
                clean_procedures.append(proc)
        
        # If still no results, create generic procedure from any dollar amounts
        if not clean_procedures:
            dollar_matches = re.findall(r'\$(\d+(?:\.\d{2})?)', text)
            for amount in dollar_matches[:3]:  # Take first 3 amounts
                try:
                    cost = float(amount)
                    if cost > 10:  # Reasonable minimum
                        clean_procedures.append(ParsedProcedure(
                            procedure_code=None,
                            name=f"Medical procedure (${cost})",
                            unit_cost=cost,
                            qty=1
                        ))
                except ValueError:
                    continue
        
        return clean_procedures
    
    def providers_search(self, criteria: Dict[str, str]) -> List[Provider]:
        """
        Search providers by specialty and city (max 3 results)
        
        Args:
            criteria: Dict with specialty and location/city
            
        Returns:
            List of matching providers (max 3)
        """
        specialty = criteria.get("specialty", "").lower()
        city = criteria.get("location", "").lower()
        
        matching_providers = []
        max_results = self.carecredit_rules.get("provider_search", {}).get("max_results", 3)
        
        for provider in self.providers:
            # Check specialty match
            if specialty and provider.get("specialty", "").lower() != specialty:
                continue
            
            # Check city match (if specified) 
            if city and city not in provider.get("city", "").lower():
                continue
            
            # Only include enrolled/CareCredit accepting providers
            if provider.get("accepts_carecredit", False) or provider.get("enrolled", False):
                matching_providers.append(Provider(
                    name=provider["name"],
                    address=provider.get("address", f"{provider.get('city', 'Unknown')}"),
                    phone=provider.get("phone", "Contact provider"),
                    next_appt_days=provider.get("next_appt_days", 7)
                ))
        
        # Sort by appointment availability (sooner appointments first)
        matching_providers.sort(key=lambda p: p.next_appt_days)
        
        return matching_providers[:max_results]  # Return max 3
    
    def _get_carecredit_offers(self, total_cost: float) -> List[FinancingOption]:
        """
        Get CareCredit financing offers with EP/DI options
        
        Args:
            total_cost: Total treatment cost
            
        Returns:
            List of financing options (EP + DI)
        """
        financing_options = []
        
        try:
            # Get financing rules from carecredit.yml
            financing_rules = self.carecredit_rules.get("financing_options", {})
            ep_rules = financing_rules.get("equal_payment", {})
            di_rules = financing_rules.get("deferred_interest", {})
            
            # Generate Equal Payment (EP) options
            ep_terms = ep_rules.get("terms", [12, 24, 36, 48, 60])
            ep_min = ep_rules.get("min_amount", 200.0)
            ep_max = ep_rules.get("max_amount", 25000.0)
            ep_promo_apr = ep_rules.get("apr_range", {}).get("promotional", 0.0)
            
            if ep_min <= total_cost <= ep_max:
                for months in ep_terms:
                    if months <= 24:  # Promotional 0% APR for shorter terms
                        apr = ep_promo_apr
                        offer_id = f"EP_{months}_0"
                    else:  # Standard APR for longer terms  
                        apr = 14.90  # Use standard rate
                        offer_id = f"EP_{months}_1490"
                    
                    # Calculate payment
                    if apr == 0:
                        monthly = total_cost / months
                        total_with_interest = total_cost
                    else:
                        monthly_rate = apr / 100 / 12
                        monthly = (total_cost * monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
                        total_with_interest = monthly * months
                    
                    financing_options.append(FinancingOption(
                        offer_id=offer_id,
                        months=months,
                        apr=apr,
                        monthly=round(monthly, 2),
                        total_cost=round(total_with_interest, 2)
                    ))
            
            # Generate Deferred Interest (DI) options  
            di_terms = di_rules.get("terms", [6, 12, 18, 24])
            di_min = di_rules.get("min_amount", 200.0)
            di_max = di_rules.get("max_amount", 25000.0)
            di_standard_apr = di_rules.get("standard_apr", 26.99)
            
            if di_min <= total_cost <= di_max:
                for months in di_terms:
                    offer_id = f"DI_{months}_0"
                    
                    # DI: 0% during promo period, standard APR if not paid off
                    financing_options.append(FinancingOption(
                        offer_id=offer_id,
                        months=months,
                        apr=0.0,  # Promotional rate during term
                        monthly=round(total_cost / months, 2),  # Minimum payment
                        total_cost=total_cost  # If paid during promo period
                    ))
            
            # Sort by APR, then by months (0% APR first, shorter terms first)
            financing_options.sort(key=lambda x: (x.apr > 0, x.apr, x.months))
            
        except Exception as e:
            logger.error(f"Error getting CareCredit offers: {e}")
            # Fallback default offers
            financing_options = [
                FinancingOption(offer_id="EP_12_0", months=12, apr=0.0, monthly=round(total_cost/12, 2), total_cost=total_cost),
                FinancingOption(offer_id="DI_12_0", months=12, apr=0.0, monthly=round(total_cost/12, 2), total_cost=total_cost)
            ]
        
        return financing_options
    
    def oopp_simulate(
        self, 
        total: float, 
        promo: Optional[FinancingOption], 
        insurance: Optional[Dict[str, float]]
    ) -> OOPPResult:
        """
        Simulate out-of-pocket costs using rules/carecredit.yml defaults
        
        Args:
            total: Total procedure cost
            promo: Selected financing option
            insurance: Insurance details
            
        Returns:
            OOPP simulation result
        """
        assumptions = {}
        
        try:
            # Start with total cost
            estimated_total = total
            assumptions["original_total"] = total
            
            # Get defaults from rules
            oopp_defaults = self.carecredit_rules.get("oopp_defaults", {})
            deductible_defaults = oopp_defaults.get("deductible", {})
            coinsurance_defaults = oopp_defaults.get("coinsurance", {})
            
            # Apply insurance if provided, otherwise use defaults
            if insurance:
                deductible_left = insurance.get("deductible_left", deductible_defaults.get("individual", 1500.0))
                coinsurance = insurance.get("coinsurance", coinsurance_defaults.get("in_network", 0.2))
            else:
                # Use rules defaults
                deductible_left = deductible_defaults.get("individual", 1500.0)
                coinsurance = coinsurance_defaults.get("in_network", 0.2)
                assumptions["insurance_status"] = f"Using default deductible ${deductible_left} and {coinsurance*100}% coinsurance"
            
            # Apply deductible
            after_deductible = max(0, total - deductible_left)
            deductible_used = min(total, deductible_left)
            
            # Apply coinsurance to remaining amount
            insurance_pays = after_deductible * (1 - coinsurance)
            patient_coinsurance = after_deductible * coinsurance
            
            estimated_total = deductible_used + patient_coinsurance
            
            assumptions.update({
                "deductible_applied": deductible_used,
                "coinsurance_rate": coinsurance,
                "insurance_payment": insurance_pays,
                "patient_portion": estimated_total,
                "caveat": self.carecredit_rules.get("oop_estimator", {}).get("caveat", "Illustrative only")
            })
            
            # Factor in financing if selected
            if promo:
                assumptions.update({
                    "financing_option": f"{promo.months} months at {promo.apr}% APR",
                    "monthly_payment": promo.monthly,
                    "total_with_interest": promo.total_cost
                })
                
                # For 0% APR, total cost doesn't change
                if promo.apr == 0:
                    assumptions["financing_benefit"] = "0% APR - no interest charges"
                else:
                    interest_cost = promo.total_cost - estimated_total
                    assumptions["interest_cost"] = interest_cost
            
        except Exception as e:
            logger.error(f"Error simulating OOPP: {e}")
            assumptions["error"] = str(e)
        
        return OOPPResult(
            estimated_total=round(estimated_total, 2),
            assumptions=assumptions
        )
    
    def terms_retrieve(self, query: str) -> List[Citation]:
        """
        Retrieve CareCredit terms with Tavily fallback
        
        Args:
            query: Terms query
            
        Returns:
            List of citations
        """
        citations = []
        
        try:
            # First try local knowledge base
            if self.retriever and self.embedder:
                from app.rag.core import retrieve
                results = retrieve(self.retriever, self.embedder, query, k=2)
                
                for result in results:
                    citations.append(Citation(
                        source=result.get("filename", "Terms Document"),
                        snippet=result.get("snippet", "")[:200] + "..."
                    ))
            
            # If no local results, search web with Tavily
            if not citations and self.docstore and self.embedder:
                logger.info("No local terms found, searching web for CareCredit terms")
                
                try:
                    web_docs = web_search_into_docstore(
                        self.docstore,
                        self.embedder,
                        "CareCredit promotional financing terms conditions healthcare",
                        max_results=2
                    )
                    
                    if web_docs:
                        # Re-retrieve after adding web content
                        from app.rag.core import retrieve
                        results = retrieve(self.retriever, self.embedder, query, k=2)
                        
                        for result in results:
                            citations.append(Citation(
                                source=result.get("filename", "Web Search - CareCredit"),
                                snippet=result.get("snippet", "")[:200] + "..."
                            ))
                
                except Exception as e:
                    logger.warning(f"Web search for CareCredit terms failed: {e}")
            
            # Add default citation if nothing found
            if not citations:
                citations.append(Citation(
                    source="CareCredit Terms",
                    snippet="Subject to credit approval. Minimum monthly payments required. See carecredit.com for full terms and conditions."
                ))
            
        except Exception as e:
            logger.error(f"Error retrieving terms: {e}")
        
        return citations
    
    def _identify_specialty(self, procedures: List[ParsedProcedure], text: str) -> str:
        """
        Identify medical specialty from procedures and text
        
        Args:
            procedures: Parsed procedures
            text: Original estimate text
            
        Returns:
            Medical specialty
        """
        # Check procedure codes first
        for proc in procedures:
            if proc.procedure_code:
                if proc.procedure_code in self.procedure_mappings:
                    return self.procedure_mappings[proc.procedure_code]["specialty"]
        
        # Check keywords in text and procedure names
        text_lower = text.lower()
        all_text = text_lower + " " + " ".join(proc.name.lower() for proc in procedures if proc.name)
        
        specialty_scores = {}
        for specialty, keywords in self.specialties.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            if score > 0:
                specialty_scores[specialty] = score
        
        if specialty_scores:
            return max(specialty_scores.items(), key=lambda x: x[1])[0]
        
        # Default to dental (most common CareCredit usage)
        return "dental"
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response that may contain markdown formatting"""
        import re
        
        # Try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Try to find JSON directly
        json_match = re.search(r'(\{.*?\}|\[.*?\])', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Fallback
        return text.strip()
    
    def _is_medical_estimate(self, text: str) -> bool:
        """
        Check if the input text is a medical/dental estimate with procedures and costs
        
        Args:
            text: Input text to validate
            
        Returns:
            True if this appears to be a medical estimate, False otherwise
        """
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Indicators that this is NOT a medical estimate (general queries)
        query_indicators = [
            "find me", "show me", "i need", "looking for", "search for",
            "what are", "how much", "can you", "help me find",
            "equipment", "financing options", "carecredit options"
        ]
        
        if any(indicator in text_lower for indicator in query_indicators):
            return False
        
        # Indicators that this IS a medical estimate
        estimate_indicators = [
            "$", "cost", "total", "procedure", "treatment", "exam", "cleaning",
            "filling", "crown", "implant", "extraction", "surgery", "consultation",
            "x-ray", "root canal", "orthodontic", "periodontal"
        ]
        
        # Medical billing patterns
        billing_patterns = [
            r'\$[\d,]+\.?\d*',  # Dollar amounts
            r'procedure.*\$',   # Procedure with cost
            r'treatment.*\$',   # Treatment with cost  
            r'\d+\.\d+',       # Decimal numbers (costs)
        ]
        
        # Check for cost indicators
        has_cost_indicators = any(indicator in text_lower for indicator in estimate_indicators)
        
        # Check for billing patterns
        import re
        has_billing_patterns = any(re.search(pattern, text_lower) for pattern in billing_patterns)
        
        # Must have either cost indicators or billing patterns
        return has_cost_indicators or has_billing_patterns
    
    def _handle_general_query(self, query: str) -> CreditResponse:
        """
        Handle general CareCredit queries (not medical estimates)
        
        Args:
            query: General query about CareCredit options
            
        Returns:
            CreditResponse with general information
        """
        # Extract any budget information from the query
        import re
        budget_match = re.search(r'under \$?([0-9,]+)', query.lower())
        budget = None
        if budget_match:
            budget = float(budget_match.group(1).replace(',', ''))
        
        # Generate a helpful response about CareCredit in general
        response_text = f"""I can help you understand CareCredit financing options for healthcare expenses.
        
**CareCredit Overview:**
• Special healthcare credit card for medical, dental, veterinary, and vision expenses
• Promotional financing options including 6, 12, 18, and 24-month plans
• No interest if paid in full within promotional period
• Accepted at over 260,000 healthcare providers nationwide

**Next Steps:**
1. **Get a specific treatment estimate** from your healthcare provider
2. **Upload the estimate** here for detailed financing analysis
3. **Apply for CareCredit** at carecredit.com if you haven't already
4. **Use the CareCredit mobile app** to manage your account

For detailed financing calculations, please provide a specific treatment estimate with procedures and costs."""

        if budget:
            response_text += f"\n\n**Budget Note:** For expenses around ${budget:,.0f}, CareCredit offers several promotional periods that can help manage costs."

        return CreditResponse(
            response=response_text,
            metadata={
                "ui_cards": [],
                "disclosures": ["carecredit_generic"],
                "query_type": "general_inquiry",
                "budget_mentioned": budget
            }
        )
    
    def _generate_response_text(
        self,
        procedures: List[ParsedProcedure],
        providers: List[Provider],
        financing: List[FinancingOption],
        oopp: OOPPResult,
        total_cost: float
    ) -> str:
        """
        Generate response: counts + 12-mo explanation + next steps
        
        Args:
            procedures: Parsed procedures
            providers: Available providers
            financing: Financing options
            oopp: Out-of-pocket projection
            total_cost: Total treatment cost
            
        Returns:
            Response with counts, 12-mo explanation, and next steps
        """
        response_parts = []
        
        # Counts summary
        response_parts.append(
            f"**Treatment Summary:** {len(procedures)} procedure{'s' if len(procedures) != 1 else ''} "
            f"totaling ${total_cost:,.2f}"
        )
        
        # OOPP estimate
        response_parts.append(
            f"**Out-of-Pocket:** ${oopp.estimated_total:,.2f} "
            f"({oopp.assumptions.get('caveat', 'estimate only')})"
        )
        
        # 12-month financing explanation
        if financing:
            best_12mo = next((f for f in financing if f.months == 12), financing[0])
            if best_12mo.apr == 0:
                response_parts.append(
                    f"**12-Month Financing:** ${best_12mo.monthly:.2f}/month with 0% APR promotional financing"
                )
            else:
                response_parts.append(
                    f"**12-Month Financing:** ${best_12mo.monthly:.2f}/month at {best_12mo.apr}% APR"
                )
        
        # Provider availability
        if providers:
            next_appt = min(p.next_appt_days for p in providers)
            response_parts.append(
                f"**Providers:** {len(providers)} CareCredit providers available, "
                f"next appointment in {next_appt} days"
            )
        
        # Next steps
        next_steps = []
        if providers:
            next_steps.append("1. Contact a provider to schedule consultation")
        next_steps.append("2. Apply for CareCredit at carecredit.com")
        next_steps.append("3. Bring your CareCredit card to your appointment")
        
        if next_steps:
            response_parts.append(f"**Next Steps:** {' | '.join(next_steps)}")
        
        return "\n".join(response_parts)
    
    def _build_ui_cards(
        self,
        procedures: List[ParsedProcedure],
        providers: List[Provider],
        financing: List[FinancingOption],
        total_cost: float
    ) -> List[Dict[str, Any]]:
        """
        Build UI cards: estimate, providers, financing
        
        Returns:
            List of UI cards for metadata
        """
        ui_cards = []
        
        # Estimate card
        estimate_items = [
            {
                "name": proc.name,
                "code": proc.procedure_code or "",
                "unit_cost": proc.unit_cost,
                "qty": proc.qty,
                "subtotal": proc.unit_cost * proc.qty
            }
            for proc in procedures
        ]
        
        ui_cards.append({
            "type": "estimate",
            "items": estimate_items,
            "total": total_cost
        })
        
        # Providers card
        provider_items = [
            {
                "name": prov.name,
                "address": prov.address,
                "phone": prov.phone,
                "next_appt_days": prov.next_appt_days
            }
            for prov in providers
        ]
        
        ui_cards.append({
            "type": "providers",
            "items": provider_items
        })
        
        # Financing card
        financing_items = [
            {
                "offer_id": fin.offer_id,
                "months": fin.months,
                "apr": fin.apr,
                "monthly_payment": fin.monthly,
                "total_cost": fin.total_cost,
                "offer_type": "EP" if fin.offer_id.startswith("EP_") else "DI"
            }
            for fin in financing
        ]
        
        ui_cards.append({
            "type": "financing",
            "options": financing_items
        })
        
        return ui_cards
    
    def _get_disclosures(self, financing: List[FinancingOption]) -> List[str]:
        """
        Get disclosure list - always append carecredit_generic
        
        Args:
            financing: Financing options
            
        Returns:
            List of disclosure IDs
        """
        disclosures = ["carecredit_generic"]  # Always include
        
        # Add equal_payment_generic if EP options present
        if any(f.offer_id.startswith("EP_") for f in financing):
            disclosures.append("equal_payment_generic")
        
        # Add deferred_interest_generic if DI options present  
        if any(f.offer_id.startswith("DI_") for f in financing):
            disclosures.append("deferred_interest_generic")
        
        return disclosures

# Test cases for CareCredit scenarios  
def test_carecredit():
    """Test CareCredit with golden-path scenarios"""
    print("🧪 Testing CareCredit Treatment Translator")
    print("=" * 50)
    
    carecredit = CareCredit()
    
    test_cases = [
        {
            "name": "Dental cleaning estimate",
            "estimate_text": """
            Dental Estimate - City Dental Care
            
            D0120 | Periodic oral evaluation | $85.00
            D1110 | Prophylaxis - adult cleaning | $120.00
            D0274 | Bitewing X-rays (4 films) | $65.00
            
            Total: $270.00
            """,
            "location": "New York, NY",
            "insurance": {"deductible_left": 150.0, "coinsurance": 0.2},
            "expected_specialty": "dental",
            "expected_items": 3
        },
        {
            "name": "Dermatology procedure",
            "estimate_text": """
            Dermatology Treatment Estimate
            
            Mole removal procedure ........... $450.00
            Pathology examination ............ $125.00
            Follow-up visit .................. $85.00
            
            Estimated Total: $660.00
            """,
            "location": "Chicago, IL", 
            "insurance": None,
            "expected_specialty": "dermatology",
            "expected_items": 3
        },
        {
            "name": "Veterinary emergency",
            "estimate_text": """
            Pet Emergency Treatment
            
            Emergency exam: $150
            X-rays (2 views): $180
            Pain medication: $45
            
            Total due: $375
            """,
            "location": "Phoenix, AZ",
            "insurance": None,
            "expected_specialty": "veterinary", 
            "expected_items": 3
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        try:
            print(f"{i}. {case['name']}")
            
            result = carecredit.process_estimate(
                estimate_text=case["estimate_text"],
                location=case["location"],
                insurance=case["insurance"]
            )
            
            # Validate response structure
            valid_structure = (
                isinstance(result, CreditResponse) and
                isinstance(result.line_items, list) and
                isinstance(result.providers, list) and
                isinstance(result.financing, list) and
                isinstance(result.oopp, OOPPResult) and
                isinstance(result.citations, list)
            )
            
            # Check line items parsing
            items_ok = len(result.line_items) >= case["expected_items"]
            
            # Check providers found
            providers_ok = len(result.providers) > 0
            
            # Check financing options
            financing_ok = len(result.financing) > 0
            
            # Check explanation generated
            explanation_ok = len(result.explanation) > 50
            
            # Check OOPP calculation
            oopp_ok = result.oopp.estimated_total > 0
            
            success = (valid_structure and items_ok and providers_ok and 
                      financing_ok and explanation_ok and oopp_ok)
            
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Location: {case['location']}")
            print(f"   Line items parsed: {len(result.line_items)}")
            print(f"   Providers found: {len(result.providers)}")
            print(f"   Financing options: {len(result.financing)}")
            print(f"   Out-of-pocket estimate: ${result.oopp.estimated_total:.2f}")
            print(f"   Status: {status}")
            
            if success:
                passed += 1
            else:
                print(f"   Failure reasons:")
                if not valid_structure:
                    print(f"     - Invalid response structure")
                if not items_ok:
                    print(f"     - Insufficient line items parsed")
                if not providers_ok:
                    print(f"     - No providers found")
                if not financing_ok:
                    print(f"     - No financing options")
                if not explanation_ok:
                    print(f"     - Poor explanation generated")
                if not oopp_ok:
                    print(f"     - OOPP calculation failed")
            
            print()
            
        except Exception as e:
            print(f"   ❌ FAIL - Exception: {str(e)}")
            print()
    
    print(f"📊 CareCredit Test Results: {passed}/{total} tests passed")
    return passed == total

if __name__ == "__main__":
    test_carecredit()