"""
OfferPilot - Marketplace Discovery → Financing → Apply
Conversational shopping with grounded products, promo financing, and credit pre-qualification
"""

import json
import logging
import math
import hashlib
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.rag.core import retrieve
from app.tools.tavily_search import web_search_into_docstore

logger = logging.getLogger(__name__)

# Enhanced Pydantic models for rules-aware financing
class PromoOffer(BaseModel):
    type: str  # "equal_payment" or "deferred_interest"
    months: int
    est_monthly: Optional[float] = None
    disclosure_key: str
    min_purchase: int = 0

class ProductCard(BaseModel):
    title: str
    price: int  # in cents
    partner: str
    promos: List[PromoOffer]
    warnings: List[str] = []

class PrequalResult(BaseModel):
    status: str  # "eligible", "uncertain", "ineligible"
    explanation: str

class OfferPilotResponse(BaseModel):
    response: str  # 3-5 lines summary
    metadata: Dict[str, Any]  # Contains ui_cards, disclosures, handoffs

class Citation(BaseModel):
    source: str
    snippet: str

@dataclass
class UserStub:
    """Mock user data for credit screening"""
    credit_score: int = 720
    income: float = 75000.0
    debt_to_income: float = 0.25
    employment_status: str = "employed"
    years_at_job: int = 3

class OfferPilot:
    """
    Marketplace discovery agent with financing options and credit pre-qualification
    """
    
    def __init__(self, docstore=None, embedder=None, retriever=None, rules_loader=None):
        """Initialize OfferPilot with RAG components and rules-based financing"""
        self.docstore = docstore
        self.embedder = embedder
        self.retriever = retriever
        self.rules_loader = rules_loader
        
        # Load rules from centralized loader or fallback to local loading
        if rules_loader:
            self.promotions_rules = rules_loader.get_rules('promotions') or {}
            self.disclosures_rules = rules_loader.get_rules('disclosures') or {}
            self.prequalification_rules = rules_loader.get_rules('prequalification') or {}
            logger.info("OfferPilot loaded rules from centralized rules loader")
        else:
            self._load_rules()
        
        # Load data files
        self._load_marketplace_data()
        self._load_financing_data()
    
    def _load_rules(self):
        """Load rules from synchrony-demo-rules-repo"""
        try:
            rules_base = Path("synchrony-demo-rules-repo/rules")
            
            # Load promotions rules
            with open(rules_base / "promotions.yml", 'r') as f:
                self.promotions_rules = yaml.safe_load(f)
            
            # Load disclosures
            with open(rules_base / "disclosures.yml", 'r') as f:
                self.disclosures_rules = yaml.safe_load(f)
            
            # Load prequalification rules
            with open(rules_base / "prequalification.yml", 'r') as f:
                self.prequalification_rules = yaml.safe_load(f)
                
            logger.info("Loaded rules from synchrony-demo-rules-repo")
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            # Fallback to empty rules
            self.promotions_rules = {"partners": [], "generic_defaults": {"equal_payment_months": [6, 12, 18], "deferred_interest_months": [6, 12]}}
            self.disclosures_rules = {"disclosures": {}}
            self.prequalification_rules = {"demo_scoring": {"buckets": []}}
    
    def _load_marketplace_data(self):
        """Load marketplace catalog from rules repo"""
        try:
            # First try centralized rules loader
            if self.rules_loader:
                products_data = self.rules_loader.get_fixture('products')
                merchants_data = self.rules_loader.get_fixture('merchants')
                
                if products_data and merchants_data:
                    self.marketplace_products = products_data
                    self.merchants_data = {m["partner_id"]: m for m in merchants_data}
                    logger.info(f"Loaded {len(self.marketplace_products)} products from centralized rules loader")
                    return
            
            # Fallback to direct file access
            products_path = Path("synchrony-demo-rules-repo/fixtures/products.json")
            merchants_path = Path("synchrony-demo-rules-repo/fixtures/merchants.json")
            
            if products_path.exists():
                with open(products_path, 'r') as f:
                    self.marketplace_products = json.load(f)
                with open(merchants_path, 'r') as f:
                    self.merchants_data = {m["partner_id"]: m for m in json.load(f)}
                logger.info(f"Loaded {len(self.marketplace_products)} products from rules repo")
            else:
                # Fallback to original catalog
                catalog_path = Path("app/data/marketplace_catalog.json")
                with open(catalog_path, 'r') as f:
                    data = json.load(f)
                    self.marketplace_products = data["products"]
                    self.merchants_data = {}
                logger.info(f"Loaded {len(self.marketplace_products)} products from fallback catalog")
                
        except Exception as e:
            logger.error(f"Failed to load marketplace data: {e}")
            self.marketplace_products = []
            self.merchants_data = {}
    
    def _load_financing_data(self):
        """Load financing offers from JSON file"""
        try:
            offers_path = Path("app/data/financing_offers.json")
            with open(offers_path, 'r') as f:
                data = json.load(f)
                self.financing_offers = {offer["id"]: offer for offer in data["offers"]}
                self.merchant_offers = data["merchant_offers"]
                self.offers_metadata = data["metadata"]
            logger.info(f"Loaded {len(self.financing_offers)} financing offers")
        except Exception as e:
            logger.error(f"Failed to load financing offers: {e}")
            self.financing_offers = {}
            self.merchant_offers = {}
            self.offers_metadata = {}
    
    def process_query(self, query: str, budget: Optional[float] = None, user_type: str = "consumer", user_id: str = "demo_user") -> OfferPilotResponse:
        """
        Enhanced rules-aware processing pipeline for OfferPilot
        
        Args:
            query: User search query
            budget: Optional budget constraint  
            user_type: consumer or partner
            user_id: User identifier for deterministic prequalification
            
        Returns:
            OfferPilotResponse with structured UI cards, disclosures, and handoffs
        """
        try:
            logger.info(f"Processing rules-aware OfferPilot query: {query}, user_type: {user_type}")
            
            # Step 1: Search and filter products
            matching_products = self._search_products(query, budget)
            if not matching_products:
                return self._empty_response("No products found matching your query.")
            
            # Step 2: Generate product cards with financing options
            ui_cards = []
            all_disclosures = set()
            
            for product in matching_products[:3]:  # Top 3 products
                card, disclosures = self._create_product_card(product)
                ui_cards.append(card)
                all_disclosures.update(disclosures)
            
            # Step 3: Deterministic prequalification
            if ui_cards:
                avg_price = sum(card.price for card in ui_cards) / len(ui_cards)
                prequal = self._deterministic_prequalification(user_id, avg_price)
            else:
                prequal = PrequalResult(status="ineligible", explanation="No products available")
            
            # Step 4: Generate response summary
            response_text = self._generate_response_summary(ui_cards, prequal)
            
            # Step 5: Check for handoffs
            handoffs = self._detect_handoffs(query)
            
            return OfferPilotResponse(
                response=response_text,
                metadata={
                    "ui_cards": [card.dict() for card in ui_cards],
                    "disclosures": list(all_disclosures),
                    "handoffs": handoffs,
                    "prequalification": prequal.dict()
                }
            )
            
        except Exception as e:
            logger.error(f"Error in OfferPilot processing: {e}")
            return self._empty_response("Unable to process your request. Please try again.")
    
    def marketplace_search(self, query: str, max_results: int = 8) -> List[Dict[str, Any]]:
        """
        Search marketplace catalog for products matching query
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of matching products
        """
        query_lower = query.lower()
        results = []
        
        for product in self.marketplace_products:
            score = 0
            
            # Title matching
            if any(word in product["title"].lower() for word in query_lower.split()):
                score += 3
            
            # Category matching
            if query_lower in product["category"].lower():
                score += 2
            
            # Features matching (if features field exists)
            if "features" in product and product["features"]:
                for feature in product["features"]:
                    if any(word in feature.lower() for word in query_lower.split()):
                        score += 1
            
            # Description matching (if description field exists)
            if "description" in product and product["description"]:
                if any(word in product["description"].lower() for word in query_lower.split()):
                    score += 1
            
            if score > 0:
                product_copy = product.copy()
                product_copy["relevance_score"] = score
                results.append(product_copy)
        
        # Sort by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return results[:max_results]
    
    def offers_lookup(self, merchant: str, amount: float) -> List[Dict[str, Any]]:
        """
        Look up financing offers available for merchant and amount
        
        Args:
            merchant: Merchant name
            amount: Purchase amount
            
        Returns:
            List of applicable financing offers
        """
        applicable_offers = []
        
        # Get offers for this merchant
        merchant_offer_ids = self.merchant_offers.get(merchant, [])
        
        for offer_id in merchant_offer_ids:
            offer = self.financing_offers.get(offer_id)
            if not offer:
                continue
            
            # Check if amount qualifies for this offer
            if (amount >= offer["min_purchase"] and 
                amount <= offer["max_purchase"]):
                applicable_offers.append(offer)
        
        # Sort by APR (0% first, then ascending)
        applicable_offers.sort(key=lambda x: (x["apr"] > 0, x["apr"]))
        
        return applicable_offers
    
    def payments_simulate(self, amount: float, months: int, apr: float) -> Dict[str, Any]:
        """
        Simulate payment calculations for financing
        
        Args:
            amount: Principal amount
            months: Number of months
            apr: Annual percentage rate
            
        Returns:
            Payment simulation results
        """
        if apr == 0:
            # 0% APR - simple equal payments
            monthly = round(amount / months, 2)
            total_cost = amount
        else:
            # Standard loan calculation
            monthly_rate = apr / 100 / 12
            if monthly_rate == 0:
                monthly = round(amount / months, 2)
            else:
                monthly = round(
                    amount * (monthly_rate * (1 + monthly_rate) ** months) / 
                    ((1 + monthly_rate) ** months - 1), 2
                )
            total_cost = round(monthly * months, 2)
        
        payoff_date = datetime.now() + timedelta(days=months * 30)
        
        return {
            "monthly": monthly,
            "total_cost": total_cost,
            "payoff_date": payoff_date.strftime("%Y-%m-%d")
        }
    
    def credit_quickscreen(self, user_stub: UserStub) -> Dict[str, Any]:
        """
        Mock credit pre-qualification screening (deterministic)
        
        Args:
            user_stub: User information for screening
            
        Returns:
            Pre-qualification result
        """
        # Deterministic scoring based on user attributes
        score = 0
        reasons = []
        
        # Credit score evaluation
        if user_stub.credit_score >= 750:
            score += 40
        elif user_stub.credit_score >= 700:
            score += 30
        elif user_stub.credit_score >= 650:
            score += 20
        else:
            reasons.append("Credit score below preferred range")
        
        # Income evaluation
        if user_stub.income >= 60000:
            score += 25
        elif user_stub.income >= 40000:
            score += 15
        else:
            reasons.append("Income below minimum threshold")
        
        # Debt-to-income ratio
        if user_stub.debt_to_income <= 0.3:
            score += 20
        elif user_stub.debt_to_income <= 0.4:
            score += 10
        else:
            reasons.append("High debt-to-income ratio")
        
        # Employment stability
        if user_stub.employment_status == "employed" and user_stub.years_at_job >= 2:
            score += 15
        elif user_stub.employment_status == "employed":
            score += 10
        else:
            reasons.append("Employment history concerns")
        
        # Determine eligibility
        eligible = score >= 70
        
        if eligible:
            reason = "Pre-qualified based on credit profile"
        else:
            reason = "; ".join(reasons) if reasons else "Does not meet minimum requirements"
        
        return {
            "eligible": eligible,
            "reason": reason,
            "score": score
        }
    
    def _rank_items(self, items: List[ProductCard], query: str, budget: Optional[float]) -> List[ProductCard]:
        """
        Rank items by relevance × affordability × promo fit
        
        Args:
            items: List of product items with offers
            query: Original search query
            budget: Budget constraint
            
        Returns:
            Ranked list of items
        """
        scored_items = []
        
        for item in items:
            # Relevance score (based on title/query match)
            relevance = sum(1 for word in query.lower().split() 
                          if word in item.title.lower()) / len(query.split())
            
            # Affordability score (lower price = higher score)
            if budget:
                budget_cents = budget * 100
                affordability = max(0, (budget_cents - item.price) / budget_cents)
            else:
                # Use relative affordability within result set
                max_price = max(i.price for i in items)
                affordability = 1 - (item.price / max_price) if max_price > 0 else 0
            
            # Promo fit score (Equal Payment and DI both score well)
            promo_score = 0
            if item.promos:
                # Score based on number of promo options available
                promo_score = min(1.0, len(item.promos) * 0.3)
                # Bonus for having equal payment options (easier to understand)
                if any(promo.type == "equal_payment" for promo in item.promos):
                    promo_score += 0.2
            
            # Combined score
            combined_score = (relevance * 0.4 + affordability * 0.3 + promo_score * 0.3)
            
            scored_items.append((item, combined_score))
        
        # Sort by combined score
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        return [item for item, score in scored_items]
    
    def _get_promotional_citations(self, items: List[ProductCard]) -> List[Citation]:
        """
        Get promotional terms citations from knowledge base
        
        Args:
            items: List of product items
            
        Returns:
            List of citations
        """
        citations = []
        
        if not all([self.retriever, self.embedder]):
            logger.warning("RAG components not available for citations")
            return citations
        
        try:
            # Create query for promotional terms
            offer_types = set()
            for item in items:
                for promo in item.promos:
                    if promo.type == "deferred_interest":
                        offer_types.add("0% APR deferred interest financing")
                    elif promo.type == "equal_payment":
                        offer_types.add("equal payment financing")
            
            if offer_types:
                query = f"promotional terms {' '.join(offer_types)}"
                
                # Retrieve relevant terms documents
                results = retrieve(self.retriever, self.embedder, query, k=3)
                
                for result in results:
                    citations.append(Citation(
                        source=result.get("filename", "Promotional Terms"),
                        snippet=result.get("snippet", "")[:200] + "..."
                    ))
            
            # If no local terms found, search web for Synchrony terms
            if not citations:
                logger.info("No local promotional terms found, searching web...")
                try:
                    web_docs = web_search_into_docstore(
                        self.docstore, 
                        self.embedder, 
                        "0% APR equal monthly payments Synchrony terms",
                        max_results=2
                    )
                    
                    # Re-retrieve after adding web content
                    if web_docs:
                        results = retrieve(self.retriever, self.embedder, 
                                         "promotional financing terms", k=2)
                        
                        for result in results:
                            citations.append(Citation(
                                source=result.get("filename", "Web Search"),
                                snippet=result.get("snippet", "")[:200] + "..."
                            ))
                            
                except Exception as e:
                    logger.warning(f"Web search for terms failed: {e}")
            
        except Exception as e:
            logger.error(f"Error retrieving promotional citations: {e}")
        
        return citations
    
    # New rules-aware helper methods
    def _search_products(self, query: str, budget: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search products using existing marketplace_search but with budget filter"""
        results = self.marketplace_search(query, max_results=5)
        
        if budget:
            # Convert budget from dollars to cents for comparison
            budget_cents = int(budget * 100)
            results = [p for p in results if p.get("price", 0) <= budget_cents]
        
        return results
    
    def _create_product_card(self, product: Dict[str, Any]) -> tuple[ProductCard, List[str]]:
        """Create a ProductCard with financing options based on rules"""
        partner_id = product.get("partner_id", "unknown")
        category = product.get("category", "")
        price = product.get("price", 0)
        
        # Get applicable promotions for this partner/category
        promos = []
        disclosures_used = []
        warnings = []
        
        # Check partner-specific promotions
        partner_promos = self._get_partner_promotions(partner_id, category, price)
        
        if partner_promos:
            for promo_config in partner_promos:
                promo = PromoOffer(
                    type=promo_config["type"],
                    months=promo_config["months"],
                    disclosure_key=promo_config["disclosure_key"],
                    min_purchase=promo_config["min_purchase"]
                )
                
                # Calculate estimated monthly payment
                if promo_config["type"] == "equal_payment":
                    promo.est_monthly = price / promo_config["months"]
                else:  # deferred_interest
                    promo.est_monthly = 0  # No payments during promo period
                    warnings.append("Deferred Interest: Interest accrues if not paid in full during promotional period")
                
                promos.append(promo)
                disclosures_used.append(promo_config["disclosure_key"])
        else:
            # Use generic defaults
            defaults = self.promotions_rules["generic_defaults"]
            
            # Add Equal Payment options
            for months in defaults["equal_payment_months"]:
                if price >= 10000:  # Minimum $100 threshold
                    promos.append(PromoOffer(
                        type="equal_payment",
                        months=months,
                        est_monthly=price / months,
                        disclosure_key="equal_payment_generic"
                    ))
                    disclosures_used.append("equal_payment_generic")
            
            # Add Deferred Interest options  
            for months in defaults["deferred_interest_months"]:
                if price >= 10000:
                    promos.append(PromoOffer(
                        type="deferred_interest", 
                        months=months,
                        est_monthly=0,
                        disclosure_key="deferred_interest_generic"
                    ))
                    disclosures_used.append("deferred_interest_generic")
                    warnings.append("DI accrues if not paid in full within promotional period")
        
        # Get merchant name
        merchant_name = self.merchants_data.get(partner_id, {}).get("name", partner_id.title())
        
        card = ProductCard(
            title=product.get("title", ""),
            price=price,
            partner=merchant_name,
            promos=promos,
            warnings=warnings
        )
        
        return card, disclosures_used
    
    def _get_partner_promotions(self, partner_id: str, category: str, price: int) -> List[Dict[str, Any]]:
        """Get partner-specific promotions based on rules"""
        partner_config = None
        for partner in self.promotions_rules["partners"]:
            if partner["partner_id"] == partner_id:
                partner_config = partner
                break
        
        if not partner_config:
            return []
        
        # Check category eligibility
        if category not in partner_config.get("categories_allowed", []):
            return []
        
        # Filter promotions by minimum purchase
        applicable_promos = []
        for promo in partner_config.get("promos", []):
            if price >= promo.get("min_purchase", 0):
                applicable_promos.append(promo)
        
        return applicable_promos
    
    def _deterministic_prequalification(self, user_id: str, price: float) -> PrequalResult:
        """Deterministic prequalification based on stable hash"""
        # Create stable hash from user_id and price
        hash_input = f"{user_id}:{int(price)}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        
        # Use hash to determine bucket
        buckets = self.prequalification_rules["demo_scoring"]["buckets"]
        
        for bucket in buckets:
            if price <= bucket["price_upper"]:
                # Use hash to add some variation within bucket
                if bucket["name"] == "eligible":
                    # 80% eligible, 20% uncertain within eligible range
                    status = "eligible" if hash_value % 10 < 8 else "uncertain"
                elif bucket["name"] == "uncertain":
                    # 60% uncertain, 40% eligible within uncertain range  
                    status = "uncertain" if hash_value % 10 < 6 else "eligible"
                else:  # ineligible
                    status = "ineligible"
                    
                return PrequalResult(
                    status=status,
                    explanation=bucket["explanation"]
                )
        
        # Fallback
        return PrequalResult(status="ineligible", explanation="Above demo threshold")
    
    def _generate_response_summary(self, ui_cards: List[ProductCard], prequal: PrequalResult) -> str:
        """Generate 3-5 line response summary"""
        if not ui_cards:
            return "No products found matching your criteria."
        
        best_card = ui_cards[0]
        best_promo = best_card.promos[0] if best_card.promos else None
        
        summary = f"Found {len(ui_cards)} great options! "
        summary += f"Top pick: {best_card.title} for ${best_card.price/100:.2f} from {best_card.partner}. "
        
        if best_promo:
            if best_promo.type == "equal_payment":
                summary += f"Available: {best_promo.months}-month equal payments of ${best_promo.est_monthly/100:.2f}/month. "
            else:
                summary += f"Available: {best_promo.months}-month deferred interest (0% if paid in full). "
        
        if prequal.status == "eligible":
            summary += "✅ You're prequalified!"
        elif prequal.status == "uncertain": 
            summary += "Additional info may be needed for approval."
        else:
            summary += "Higher amounts may require additional review."
            
        return summary
    
    def _detect_handoffs(self, query: str) -> List[str]:
        """Detect if query should be handed off to other agents"""
        handoffs = []
        
        query_lower = query.lower()
        
        # Check for dispute-related terms
        dispute_terms = ["double charged", "charged twice", "dispute", "chargeback", "refund", "unauthorized"]
        if any(term in query_lower for term in dispute_terms):
            handoffs.append("dispute")
        
        # Check for collections-related terms  
        collections_terms = ["can't pay", "hardship", "payment plan", "financial difficulty"]
        if any(term in query_lower for term in collections_terms):
            handoffs.append("collections")
        
        return handoffs
    
    def _empty_response(self, message: str) -> OfferPilotResponse:
        """Generate empty response with error message"""
        return OfferPilotResponse(
            response=message,
            metadata={
                "ui_cards": [],
                "disclosures": [],
                "handoffs": [],
                "prequalification": {"status": "ineligible", "explanation": "No evaluation performed"}
            }
        )

# Golden-path tests
def test_offerpilot():
    """Test OfferPilot with golden-path scenarios"""
    print("🧪 Testing OfferPilot Golden Paths")
    print("=" * 50)
    
    # Initialize OfferPilot (without RAG for basic testing)
    pilot = OfferPilot()
    
    test_cases = [
        {
            "name": "Desk under $600",
            "query": "office desk",
            "budget": 600.0,
            "expected_items": 2,  # Should find both desk options
            "expected_financing": True
        },
        {
            "name": "Laptop around $1000",
            "query": "laptop computer",
            "budget": 1200.0,
            "expected_items": 3,  # Should find laptop options
            "expected_financing": True
        },
        {
            "name": "Healthcare with CareCredit",
            "query": "dental treatment",
            "budget": None,
            "expected_items": 2,  # Should find dental services
            "expected_carecredit": True
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        try:
            result = pilot.process_query(case["query"], case["budget"])
            
            # Validate response structure
            assert isinstance(result, OfferPilotResponse)
            assert isinstance(result.items, list)
            assert isinstance(result.prequal, PrequalResult)
            assert isinstance(result.citations, list)
            
            # Check item count
            items_found = len(result.items)
            items_ok = items_found > 0
            
            # Check financing offers
            has_financing = any(len(item.offers) > 0 for item in result.items)
            
            # Check for CareCredit if healthcare query
            has_carecredit = False
            if case.get("expected_carecredit"):
                has_carecredit = any(
                    any("CARECREDIT" in offer.id for offer in item.offers)
                    for item in result.items
                )
            
            # Check pre-qualification
            prequal_ok = isinstance(result.prequal.eligible, bool)
            
            # Overall success
            success = (items_ok and 
                      (has_financing if case["expected_financing"] else True) and
                      (has_carecredit if case.get("expected_carecredit") else True) and
                      prequal_ok)
            
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"{i}. {case['name']}")
            print(f"   Query: '{case['query']}', Budget: {case['budget']}")
            print(f"   Items found: {items_found}")
            print(f"   Has financing: {has_financing}")
            print(f"   Pre-qualified: {result.prequal.eligible}")
            if case.get("expected_carecredit"):
                print(f"   Has CareCredit: {has_carecredit}")
            print(f"   Status: {status}")
            print()
            
            if success:
                passed += 1
                
        except Exception as e:
            print(f"{i}. {case['name']}: ❌ ERROR - {e}")
            print()
    
    print(f"OfferPilot Test Results: {passed}/{total} passed")
    return passed == total

if __name__ == "__main__":
    # Run tests
    success = test_offerpilot()
    print(f"\n{'🎉 All tests passed!' if success else '⚠️ Some tests failed.'}")
