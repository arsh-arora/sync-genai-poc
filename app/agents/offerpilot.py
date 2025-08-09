"""
OfferPilot - Marketplace Discovery → Financing → Apply
Conversational shopping with grounded products, promo financing, and credit pre-qualification
"""

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.rag.core import retrieve
from app.tools.tavily_search import web_search_into_docstore

logger = logging.getLogger(__name__)

# Pydantic models for strict output schema enforcement
class FinancingOffer(BaseModel):
    id: str
    months: int
    apr: float
    monthly: float
    total_cost: float
    disclaimer: str

class ProductItem(BaseModel):
    sku: str
    title: str
    price: float
    merchant: str
    offers: List[FinancingOffer]

class PrequalResult(BaseModel):
    eligible: bool
    reason: str

class Citation(BaseModel):
    source: str
    snippet: str

class OfferPilotResponse(BaseModel):
    items: List[ProductItem]
    prequal: PrequalResult
    citations: List[Citation]

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
    
    def __init__(self, docstore=None, embedder=None, retriever=None):
        """Initialize OfferPilot with RAG components for terms retrieval"""
        self.docstore = docstore
        self.embedder = embedder
        self.retriever = retriever
        
        # Load data files
        self._load_marketplace_data()
        self._load_financing_data()
    
    def _load_marketplace_data(self):
        """Load marketplace catalog from JSON file"""
        try:
            catalog_path = Path("app/data/marketplace_catalog.json")
            with open(catalog_path, 'r') as f:
                data = json.load(f)
                self.marketplace_products = data["products"]
                self.marketplace_metadata = data["metadata"]
            logger.info(f"Loaded {len(self.marketplace_products)} products from marketplace catalog")
        except Exception as e:
            logger.error(f"Failed to load marketplace catalog: {e}")
            self.marketplace_products = []
            self.marketplace_metadata = {}
    
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
    
    def process_query(self, query: str, budget: Optional[float] = None) -> OfferPilotResponse:
        """
        Main processing pipeline for OfferPilot
        
        Args:
            query: User search query
            budget: Optional budget constraint
            
        Returns:
            OfferPilotResponse with products, financing, and pre-qualification
        """
        try:
            logger.info(f"Processing OfferPilot query: {query}, budget: {budget}")
            
            # Step 1: Search marketplace
            search_results = self.marketplace_search(query, max_results=8)
            logger.info(f"Found {len(search_results)} products from marketplace search")
            
            # Step 2: Filter by budget if provided
            if budget:
                search_results = [p for p in search_results if p["price"] <= budget]
                logger.info(f"Filtered to {len(search_results)} products within budget ${budget}")
            
            # Step 3: Get financing offers for each product
            enriched_items = []
            for product in search_results[:5]:  # Top 5 products
                offers = self.offers_lookup(product["merchant"], product["price"])
                
                # Step 4: Simulate payments for each offer
                financing_offers = []
                for offer in offers:
                    payment_sim = self.payments_simulate(
                        product["price"], 
                        offer["months"], 
                        offer["apr"]
                    )
                    
                    financing_offers.append(FinancingOffer(
                        id=offer["id"],
                        months=offer["months"],
                        apr=offer["apr"],
                        monthly=payment_sim["monthly"],
                        total_cost=payment_sim["total_cost"],
                        disclaimer=offer["disclaimer"]
                    ))
                
                enriched_items.append(ProductItem(
                    sku=product["sku"],
                    title=product["title"],
                    price=product["price"],
                    merchant=product["merchant"],
                    offers=financing_offers
                ))
            
            # Step 5: Rank by relevance × affordability × promo fit
            ranked_items = self._rank_items(enriched_items, query, budget)
            
            # Step 6: Get promotional terms citations
            citations = self._get_promotional_citations(ranked_items)
            
            # Step 7: Credit pre-qualification
            user_stub = UserStub()  # Mock user data
            prequal = self.credit_quickscreen(user_stub)
            
            return OfferPilotResponse(
                items=ranked_items,
                prequal=PrequalResult(
                    eligible=prequal["eligible"],
                    reason=prequal["reason"]
                ),
                citations=citations
            )
            
        except Exception as e:
            logger.error(f"Error in OfferPilot processing: {e}")
            return OfferPilotResponse(
                items=[],
                prequal=PrequalResult(eligible=False, reason="System error occurred"),
                citations=[]
            )
    
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
            
            # Features matching
            for feature in product["features"]:
                if any(word in feature.lower() for word in query_lower.split()):
                    score += 1
            
            # Description matching
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
    
    def _rank_items(self, items: List[ProductItem], query: str, budget: Optional[float]) -> List[ProductItem]:
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
                affordability = max(0, (budget - item.price) / budget)
            else:
                # Use relative affordability within result set
                max_price = max(i.price for i in items)
                affordability = 1 - (item.price / max_price)
            
            # Promo fit score (0% APR offers get higher score)
            promo_score = 0
            if item.offers:
                best_offer = min(item.offers, key=lambda x: x.apr)
                if best_offer.apr == 0:
                    promo_score = 1.0
                else:
                    promo_score = max(0, (25 - best_offer.apr) / 25)  # Scale from 25% APR
            
            # Combined score
            combined_score = (relevance * 0.4 + affordability * 0.3 + promo_score * 0.3)
            
            scored_items.append((item, combined_score))
        
        # Sort by combined score
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        return [item for item, score in scored_items]
    
    def _get_promotional_citations(self, items: List[ProductItem]) -> List[Citation]:
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
                for offer in item.offers:
                    if offer.apr == 0:
                        offer_types.add("0% APR promotional financing")
                    else:
                        offer_types.add("standard APR financing")
            
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
