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
    explanation: str
    line_items: List[LineItem]
    providers: List[Provider]
    financing: List[FinancingOption]
    oopp: OOPPResult
    citations: List[Citation]

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
    
    def __init__(self, docstore=None, embedder=None, retriever=None):
        """Initialize CareCredit agent with required components"""
        self.docstore = docstore
        self.embedder = embedder
        self.retriever = retriever
        
        # Initialize OfferPilot for financing offers
        self.offer_pilot = OfferPilot(docstore, embedder, retriever)
        
        # Load provider data
        self._load_provider_data()
    
    def _load_provider_data(self):
        """Load healthcare provider directory"""
        try:
            providers_path = Path("app/data/providers.json")
            with open(providers_path, 'r') as f:
                data = json.load(f)
                self.providers = data["providers"]
                self.specialties = data["specialties"]
                self.procedure_mappings = data["procedure_mappings"]
            logger.info(f"Loaded {len(self.providers)} healthcare providers")
        except Exception as e:
            logger.error(f"Failed to load provider data: {e}")
            self.providers = []
            self.specialties = {}
            self.procedure_mappings = {}
    
    def process_estimate(
        self,
        estimate_text: str,
        location: Optional[str] = None,
        insurance: Optional[Dict[str, float]] = None
    ) -> CreditResponse:
        """
        Main processing pipeline for treatment estimates
        
        Args:
            estimate_text: Medical/dental estimate text
            location: Patient location for provider search
            insurance: Insurance info with deductible_left and coinsurance
            
        Returns:
            CreditResponse with explanation, providers, financing options
        """
        try:
            logger.info("Processing CareCredit treatment estimate")
            
            # Step 1: Parse estimate into line items
            parsed_items = self.estimate_parse_table(estimate_text)
            logger.info(f"Parsed {len(parsed_items)} line items from estimate")
            
            # Step 2: Calculate total cost
            total_cost = sum(item.unit_cost * item.qty for item in parsed_items)
            
            # Step 3: Identify specialty from procedures
            specialty = self._identify_specialty(parsed_items, estimate_text)
            logger.info(f"Identified specialty: {specialty}")
            
            # Step 4: Search for providers
            providers = self.providers_search({"specialty": specialty, "location": location})
            
            # Step 5: Get CareCredit financing offers
            financing_offers = self._get_carecredit_offers(total_cost)
            
            # Step 6: Calculate out-of-pocket costs
            oopp_result = self.oopp_simulate(total_cost, financing_offers[0] if financing_offers else None, insurance)
            
            # Step 7: Get terms and citations
            citations = self.terms_retrieve("CareCredit promotional financing")
            
            # Step 8: Generate plain-language explanation
            explanation = self._generate_explanation(parsed_items, providers, financing_offers, oopp_result)
            
            # Convert parsed items to LineItem models
            line_items = [
                LineItem(
                    name=item.name,
                    unit_cost=item.unit_cost,
                    qty=item.qty,
                    subtotal=item.unit_cost * item.qty,
                    procedure_code=item.procedure_code
                )
                for item in parsed_items
            ]
            
            return CreditResponse(
                explanation=explanation,
                line_items=line_items,
                providers=providers,
                financing=financing_offers,
                oopp=oopp_result,
                citations=citations
            )
            
        except Exception as e:
            logger.error(f"Error processing CareCredit estimate: {e}")
            return CreditResponse(
                explanation=f"Error processing estimate: {str(e)}",
                line_items=[],
                providers=[],
                financing=[],
                oopp=OOPPResult(estimated_total=0.0, assumptions={}),
                citations=[]
            )
    
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
                parsed_json = json.loads(response.strip())
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
        Search providers by specialty and location
        
        Args:
            criteria: Dict with specialty and location
            
        Returns:
            List of matching providers
        """
        specialty = criteria.get("specialty", "").lower()
        location = criteria.get("location", "").lower()
        
        matching_providers = []
        
        for provider in self.providers:
            # Check specialty match
            if specialty and provider["specialty"] != specialty:
                continue
            
            # Check location match (if specified)
            if location and location not in provider["location"].lower():
                continue
            
            # Only include CareCredit accepting providers
            if provider.get("accepts_carecredit", False):
                matching_providers.append(Provider(
                    name=provider["name"],
                    address=provider["address"],
                    phone=provider["phone"],
                    next_appt_days=provider["next_appt_days"]
                ))
        
        # Sort by appointment availability
        matching_providers.sort(key=lambda p: p.next_appt_days)
        
        return matching_providers[:5]  # Return top 5
    
    def _get_carecredit_offers(self, total_cost: float) -> List[FinancingOption]:
        """
        Get CareCredit financing offers for healthcare merchants
        
        Args:
            total_cost: Total treatment cost
            
        Returns:
            List of financing options
        """
        financing_options = []
        
        try:
            # Look for healthcare-specific offers using OfferPilot
            healthcare_merchants = ["CareCredit", "Healthcare Financing", "Medical Credit"]
            
            for merchant in healthcare_merchants:
                offers = self.offer_pilot.offers_lookup(merchant, total_cost)
                
                for offer in offers:
                    # Calculate payment details
                    payment_sim = self.offer_pilot.payments_simulate(
                        total_cost, offer["months"], offer["apr"]
                    )
                    
                    financing_options.append(FinancingOption(
                        offer_id=offer["id"],
                        months=offer["months"],
                        apr=offer["apr"],
                        monthly=payment_sim["monthly"],
                        total_cost=payment_sim["total_cost"]
                    ))
            
            # Add default CareCredit options if none found
            if not financing_options:
                default_offers = [
                    {"id": "CARECREDIT_12_0", "months": 12, "apr": 0.0},
                    {"id": "CARECREDIT_24_0", "months": 24, "apr": 0.0},
                    {"id": "CARECREDIT_60_1499", "months": 60, "apr": 14.99},
                ]
                
                for offer in default_offers:
                    # Only show 0% APR for qualifying amounts
                    if offer["apr"] == 0 and total_cost < 200:
                        continue
                        
                    payment_sim = self.offer_pilot.payments_simulate(
                        total_cost, offer["months"], offer["apr"]
                    )
                    
                    financing_options.append(FinancingOption(
                        offer_id=offer["id"],
                        months=offer["months"],
                        apr=offer["apr"],
                        monthly=payment_sim["monthly"],
                        total_cost=payment_sim["total_cost"]
                    ))
            
            # Sort by APR (0% first, then ascending)
            financing_options.sort(key=lambda x: (x.apr > 0, x.apr))
            
        except Exception as e:
            logger.error(f"Error getting CareCredit offers: {e}")
        
        return financing_options
    
    def oopp_simulate(
        self, 
        total: float, 
        promo: Optional[FinancingOption], 
        insurance: Optional[Dict[str, float]]
    ) -> OOPPResult:
        """
        Simulate out-of-pocket costs with insurance and financing
        
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
            
            # Apply insurance if provided
            if insurance:
                deductible_left = insurance.get("deductible_left", 0)
                coinsurance = insurance.get("coinsurance", 0.2)  # Default 20%
                
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
                    "patient_portion": estimated_total
                })
            else:
                assumptions["insurance_status"] = "No insurance information provided"
            
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
        all_text = text_lower + " " + " ".join(proc.name.lower() for proc in procedures)
        
        specialty_scores = {}
        for specialty, keywords in self.specialties.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            if score > 0:
                specialty_scores[specialty] = score
        
        if specialty_scores:
            return max(specialty_scores.items(), key=lambda x: x[1])[0]
        
        # Default to dental (most common CareCredit usage)
        return "dental"
    
    def _generate_explanation(
        self,
        procedures: List[ParsedProcedure],
        providers: List[Provider],
        financing: List[FinancingOption],
        oopp: OOPPResult
    ) -> str:
        """
        Generate plain-language explanation of treatment options
        
        Args:
            procedures: Parsed procedures
            providers: Available providers
            financing: Financing options
            oopp: Out-of-pocket projection
            
        Returns:
            Plain-language explanation
        """
        explanation_parts = []
        
        # Treatment summary
        total_procedures = len(procedures)
        total_cost = sum(proc.unit_cost * proc.qty for proc in procedures)
        
        explanation_parts.append(
            f"Your treatment estimate includes {total_procedures} procedure{'s' if total_procedures != 1 else ''} "
            f"with a total cost of ${total_cost:,.2f}."
        )
        
        # Insurance impact
        if "insurance_payment" in oopp.assumptions:
            insurance_saves = oopp.assumptions["insurance_payment"]
            explanation_parts.append(
                f"With your insurance, you'll be responsible for approximately ${oopp.estimated_total:,.2f} "
                f"out-of-pocket (insurance covers ${insurance_saves:,.2f})."
            )
        else:
            explanation_parts.append(
                f"Without insurance information, your estimated out-of-pocket cost is ${oopp.estimated_total:,.2f}."
            )
        
        # Financing options
        if financing:
            best_offer = financing[0]  # First is best (sorted by APR)
            if best_offer.apr == 0:
                explanation_parts.append(
                    f"Great news! You may qualify for 0% APR financing with CareCredit, "
                    f"allowing you to pay just ${best_offer.monthly:.2f} per month for {best_offer.months} months."
                )
            else:
                explanation_parts.append(
                    f"CareCredit financing is available starting at {best_offer.apr}% APR, "
                    f"with payments as low as ${best_offer.monthly:.2f} per month."
                )
        
        # Provider availability
        if providers:
            next_available = min(p.next_appt_days for p in providers)
            explanation_parts.append(
                f"We found {len(providers)} CareCredit-accepting providers in your area, "
                f"with appointments available as soon as {next_available} days."
            )
        
        return " ".join(explanation_parts)

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