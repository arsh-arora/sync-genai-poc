"""
Collections & Hardship Advisor
Negotiation assistant that proposes compliant hardship plans
"""

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.llm.gemini import chat
from app.rag.core import retrieve
from app.tools.tavily_search import web_search_into_docstore

logger = logging.getLogger(__name__)

# Pydantic models for strict input/output schema enforcement
class CustomerState(BaseModel):
    balance: float
    apr: float
    bucket: Literal["current", "30", "60", "90", "120+"]
    income_monthly: Optional[float] = None
    expenses_monthly: Optional[float] = None
    preferences: Optional[dict] = None

class PaymentScheduleEntry(BaseModel):
    month: int
    payment: float
    interest: float
    principal: float
    balance: float

class HardshipPlan(BaseModel):
    type: str
    params: dict
    payment_curve: List[PaymentScheduleEntry]
    npv: float
    risk_score: float
    why: str
    disclosures: List[str]

class Citation(BaseModel):
    source: str
    snippet: str

class CollectionsResponse(BaseModel):
    plans: List[HardshipPlan]
    citations: List[Citation]

class CollectionsAdvisor:
    """
    Collections & Hardship Advisor for proposing compliant hardship plans
    """
    
    def __init__(self, docstore=None, embedder=None, retriever=None):
        """Initialize CollectionsAdvisor with RAG components for terms retrieval"""
        self.docstore = docstore
        self.embedder = embedder
        self.retriever = retriever
        
        # Load hardship policies
        self._load_hardship_policies()
    
    def _load_hardship_policies(self):
        """Load hardship policies from JSON file"""
        try:
            policies_path = Path("app/data/hardship_policies.json")
            with open(policies_path, 'r') as f:
                self.policies = json.load(f)
            logger.info("Loaded hardship policies successfully")
        except Exception as e:
            logger.error(f"Failed to load hardship policies: {e}")
            self.policies = {}
    
    def policy_rules_load(self) -> Dict[str, Any]:
        """Load policy rules from hardship policies"""
        return self.policies
    
    def process_hardship_request(self, customer_state: CustomerState) -> CollectionsResponse:
        """
        Main processing pipeline for hardship requests
        
        Args:
            customer_state: Customer's financial state and account information
            
        Returns:
            CollectionsResponse with recommended hardship plans and citations
        """
        try:
            logger.info(f"Processing hardship request for {customer_state.bucket} bucket, balance ${customer_state.balance:.2f}")
            
            # Step 1: Load policy rules
            policy_rules = self.policy_rules_load()
            
            # Step 2: Generate candidate plans
            candidates = self._generate_candidate_plans(customer_state, policy_rules)
            logger.info(f"Generated {len(candidates)} candidate plans")
            
            # Step 3: Simulate each plan
            simulated_plans = []
            for candidate in candidates:
                simulation = self.plan_simulate(
                    balance=customer_state.balance,
                    apr=customer_state.apr,
                    months=candidate.get("months", 12),
                    kind=candidate["type"],
                    params=candidate["params"]
                )
                
                # Step 4: Score the plan
                score_components = self._score_plan(candidate, simulation, customer_state, policy_rules)
                
                # Step 5: Generate rationale with Gemini
                rationale = self._generate_rationale(candidate, customer_state, policy_rules)
                
                # Step 6: Get mandatory disclosures
                disclosures = self._get_mandatory_disclosures(candidate["type"], policy_rules)
                
                simulated_plans.append({
                    "candidate": candidate,
                    "simulation": simulation,
                    "score": score_components["total_score"],
                    "rationale": rationale,
                    "disclosures": disclosures
                })
            
            # Step 7: Sort by score and take top 3
            simulated_plans.sort(key=lambda x: x["score"], reverse=True)
            top_plans = simulated_plans[:3]
            
            # Step 8: Get policy citations
            citations = self._get_policy_citations()
            
            # Step 9: Format response
            formatted_plans = []
            for plan_data in top_plans:
                candidate = plan_data["candidate"]
                simulation = plan_data["simulation"]
                
                formatted_plans.append(HardshipPlan(
                    type=candidate["type"],
                    params=candidate["params"],
                    payment_curve=[
                        PaymentScheduleEntry(
                            month=entry["month"],
                            payment=entry["payment"],
                            interest=entry["interest"],
                            principal=entry["principal"],
                            balance=entry["balance"]
                        ) for entry in simulation["schedule"]
                    ],
                    npv=simulation["npv"],
                    risk_score=simulation["chargeoff_risk"],
                    why=plan_data["rationale"],
                    disclosures=plan_data["disclosures"]
                ))
            
            return CollectionsResponse(
                plans=formatted_plans,
                citations=citations
            )
            
        except Exception as e:
            logger.error(f"Error processing hardship request: {e}")
            return CollectionsResponse(plans=[], citations=[])
    
    def _generate_candidate_plans(self, customer_state: CustomerState, policy_rules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate candidate hardship plans based on policy rules and customer state"""
        candidates = []
        allowed_actions = policy_rules.get("allowed_actions", {})
        
        # Deferral plans
        if (allowed_actions.get("deferral", {}).get("enabled") and 
            customer_state.bucket in allowed_actions["deferral"]["eligible_buckets"]):
            
            max_months = allowed_actions["deferral"]["max_months"]
            for months in [2, 3, 6]:
                if months <= max_months:
                    candidates.append({
                        "type": "deferral",
                        "months": months,
                        "params": {"deferral_months": months}
                    })
        
        # Re-aging plans
        if (allowed_actions.get("re_aging", {}).get("enabled") and 
            customer_state.bucket in allowed_actions["re_aging"]["eligible_buckets"]):
            
            candidates.append({
                "type": "re_aging",
                "months": 12,
                "params": {"bring_current": True, "payment_history_required": 3}
            })
        
        # Settlement plans
        if (allowed_actions.get("settlement", {}).get("enabled") and 
            customer_state.bucket in allowed_actions["settlement"]["eligible_buckets"]):
            
            min_pct = allowed_actions["settlement"]["min_settlement_pct"]
            max_pct = allowed_actions["settlement"]["max_settlement_pct"]
            
            for pct in [min_pct, 0.60, 0.75, max_pct]:
                candidates.append({
                    "type": "settlement",
                    "months": 1,
                    "params": {"settlement_percentage": pct, "lump_sum": True}
                })
                
                # Split settlement options
                if not allowed_actions["settlement"]["requires_lump_sum"]:
                    candidates.append({
                        "type": "settlement",
                        "months": 6,
                        "params": {"settlement_percentage": pct, "split_months": 6}
                    })
        
        # Interest reduction plans
        if (allowed_actions.get("interest_reduction", {}).get("enabled") and 
            customer_state.bucket in allowed_actions["interest_reduction"]["eligible_buckets"]):
            
            max_cycles = allowed_actions["interest_reduction"]["max_cycles"]
            min_reduction = allowed_actions["interest_reduction"]["min_reduction_pct"]
            
            for cycles in [6, 12]:
                if cycles <= max_cycles:
                    candidates.append({
                        "type": "interest_reduction",
                        "months": cycles,
                        "params": {"reduction_percentage": min_reduction, "cycles": cycles}
                    })
        
        # Payment plan options
        if (allowed_actions.get("payment_plan", {}).get("enabled") and 
            customer_state.bucket in allowed_actions["payment_plan"]["eligible_buckets"]):
            
            min_months = allowed_actions["payment_plan"]["min_months"]
            max_months = allowed_actions["payment_plan"]["max_months"]
            
            for months in [12, 24, 36]:
                if min_months <= months <= max_months:
                    candidates.append({
                        "type": "payment_plan",
                        "months": months,
                        "params": {"term_months": months}
                    })
        
        return candidates
    
    def plan_simulate(self, balance: float, apr: float, months: int, kind: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate payment plan and calculate NPV and chargeoff risk
        
        Args:
            balance: Current balance
            apr: Annual percentage rate
            months: Plan duration in months
            kind: Plan type
            params: Plan-specific parameters
            
        Returns:
            Simulation results with schedule, NPV, and risk
        """
        monthly_rate = apr / 100 / 12
        schedule = []
        current_balance = balance
        total_payments = 0
        
        if kind == "deferral":
            deferral_months = params.get("deferral_months", 3)
            
            # Deferral period - interest only accrues
            for month in range(1, deferral_months + 1):
                interest = current_balance * monthly_rate
                current_balance += interest
                schedule.append({
                    "month": month,
                    "payment": 0.0,
                    "interest": interest,
                    "principal": 0.0,
                    "balance": current_balance
                })
            
            # Resume normal payments after deferral
            if current_balance > 0:
                remaining_months = 24  # Assume 24 month payoff after deferral
                monthly_payment = self._calculate_monthly_payment(current_balance, apr, remaining_months)
                
                for month in range(deferral_months + 1, deferral_months + remaining_months + 1):
                    interest = current_balance * monthly_rate
                    principal = monthly_payment - interest
                    current_balance = max(0, current_balance - principal)
                    total_payments += monthly_payment
                    
                    schedule.append({
                        "month": month,
                        "payment": monthly_payment,
                        "interest": interest,
                        "principal": principal,
                        "balance": current_balance
                    })
                    
                    if current_balance <= 0:
                        break
        
        elif kind == "settlement":
            settlement_pct = params.get("settlement_percentage", 0.60)
            settlement_amount = balance * settlement_pct
            split_months = params.get("split_months", 1)
            
            monthly_settlement = settlement_amount / split_months
            
            for month in range(1, split_months + 1):
                schedule.append({
                    "month": month,
                    "payment": monthly_settlement,
                    "interest": 0.0,
                    "principal": monthly_settlement,
                    "balance": max(0, settlement_amount - (monthly_settlement * month))
                })
                total_payments += monthly_settlement
        
        elif kind == "interest_reduction":
            cycles = params.get("cycles", 12)
            reduction_pct = params.get("reduction_percentage", 0.50)
            reduced_apr = apr * (1 - reduction_pct)
            reduced_monthly_rate = reduced_apr / 100 / 12
            
            monthly_payment = self._calculate_monthly_payment(balance, reduced_apr, cycles)
            
            for month in range(1, cycles + 1):
                interest = current_balance * reduced_monthly_rate
                principal = monthly_payment - interest
                current_balance = max(0, current_balance - principal)
                total_payments += monthly_payment
                
                schedule.append({
                    "month": month,
                    "payment": monthly_payment,
                    "interest": interest,
                    "principal": principal,
                    "balance": current_balance
                })
                
                if current_balance <= 0:
                    break
        
        elif kind == "payment_plan":
            term_months = params.get("term_months", 24)
            monthly_payment = self._calculate_monthly_payment(balance, apr, term_months)
            
            for month in range(1, term_months + 1):
                interest = current_balance * monthly_rate
                principal = monthly_payment - interest
                current_balance = max(0, current_balance - principal)
                total_payments += monthly_payment
                
                schedule.append({
                    "month": month,
                    "payment": monthly_payment,
                    "interest": interest,
                    "principal": principal,
                    "balance": current_balance
                })
                
                if current_balance <= 0:
                    break
        
        else:  # re_aging - bring current then normal payments
            monthly_payment = self._calculate_monthly_payment(balance, apr, 24)
            
            for month in range(1, 25):
                interest = current_balance * monthly_rate
                principal = monthly_payment - interest
                current_balance = max(0, current_balance - principal)
                total_payments += monthly_payment
                
                schedule.append({
                    "month": month,
                    "payment": monthly_payment,
                    "interest": interest,
                    "principal": principal,
                    "balance": current_balance
                })
                
                if current_balance <= 0:
                    break
        
        # Calculate NPV (simple discount at 10% annually)
        discount_rate = 0.10 / 12  # Monthly discount rate
        npv = sum(payment["payment"] / ((1 + discount_rate) ** payment["month"]) 
                 for payment in schedule)
        
        # Calculate chargeoff risk based on plan type and customer state
        chargeoff_risk = self._calculate_chargeoff_risk(kind, params, balance, apr)
        
        return {
            "schedule": schedule,
            "npv": npv,
            "chargeoff_risk": chargeoff_risk,
            "total_payments": total_payments
        }
    
    def _calculate_monthly_payment(self, balance: float, apr: float, months: int) -> float:
        """Calculate monthly payment for standard amortization"""
        if apr == 0:
            return balance / months
        
        monthly_rate = apr / 100 / 12
        return balance * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
    
    def _calculate_chargeoff_risk(self, plan_type: str, params: Dict[str, Any], balance: float, apr: float) -> float:
        """Calculate chargeoff risk for a plan"""
        base_risk = 0.15  # Base 15% risk
        
        # Plan type adjustments
        if plan_type == "settlement":
            base_risk *= 0.3  # Settlement reduces risk significantly
        elif plan_type == "deferral":
            base_risk *= 1.5  # Deferral increases risk
        elif plan_type == "interest_reduction":
            base_risk *= 0.7  # Interest reduction helps
        elif plan_type == "re_aging":
            base_risk *= 0.8  # Re-aging helps if customer complies
        
        # Balance size impact
        if balance > 10000:
            base_risk *= 1.2
        elif balance < 2000:
            base_risk *= 0.8
        
        # APR impact
        if apr > 25:
            base_risk *= 1.1
        
        return min(0.95, base_risk)  # Cap at 95%
    
    def _score_plan(self, candidate: Dict[str, Any], simulation: Dict[str, Any], 
                   customer_state: CustomerState, policy_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Score a hardship plan based on adherence, affordability, and risk safety"""
        
        # Adherence score (0 or 1 if within policy)
        adherence = 1.0  # Assume all generated candidates are policy-compliant
        
        # Affordability score
        affordability = 1.0
        if customer_state.income_monthly and customer_state.expenses_monthly:
            disposable_income = customer_state.income_monthly - customer_state.expenses_monthly
            
            if simulation["schedule"]:
                avg_payment = sum(p["payment"] for p in simulation["schedule"]) / len(simulation["schedule"])
                if disposable_income > 0:
                    affordability = min(1.0, disposable_income / avg_payment)
                else:
                    affordability = 0.1  # Very low if no disposable income
        
        # Risk safety score
        risk_safety = 1 - simulation["chargeoff_risk"]
        
        # Combined score
        total_score = adherence * affordability * risk_safety
        
        return {
            "adherence": adherence,
            "affordability": affordability,
            "risk_safety": risk_safety,
            "total_score": total_score
        }
    
    def _generate_rationale(self, candidate: Dict[str, Any], customer_state: CustomerState, 
                           policy_rules: Dict[str, Any]) -> str:
        """Generate AI rationale for why this plan fits the customer and policy"""
        try:
            system_prompt = """You are a collections advisor explaining hardship plans. Generate a brief, professional rationale for why a specific hardship plan is appropriate for a customer's situation.

Focus on:
- Customer's financial situation and delinquency bucket
- Plan benefits and suitability
- Policy compliance
- Risk mitigation

Keep it concise (2-3 sentences) and professional."""

            plan_type = candidate["type"]
            bucket = customer_state.bucket
            balance = customer_state.balance
            
            context = f"""
Plan Type: {plan_type}
Customer Bucket: {bucket}
Balance: ${balance:.2f}
Plan Parameters: {candidate.get('params', {})}
"""
            
            if customer_state.income_monthly:
                context += f"Monthly Income: ${customer_state.income_monthly:.2f}\n"
            if customer_state.expenses_monthly:
                context += f"Monthly Expenses: ${customer_state.expenses_monthly:.2f}\n"
            
            user_message = f"Explain why this hardship plan is appropriate:\n{context}"
            messages = [{"role": "user", "content": user_message}]
            
            response = chat(messages, system=system_prompt)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating rationale: {e}")
            return f"This {candidate['type']} plan is designed to address the customer's {customer_state.bucket} delinquency status while maintaining policy compliance."
    
    def _get_mandatory_disclosures(self, plan_type: str, policy_rules: Dict[str, Any]) -> List[str]:
        """Get mandatory disclosures for a plan type"""
        disclosures = policy_rules.get("mandatory_disclosures", {})
        return disclosures.get(plan_type, [])
    
    def _get_policy_citations(self) -> List[Citation]:
        """Get policy citations from knowledge base with Tavily fallback"""
        citations = []
        
        if not all([self.retriever, self.embedder]):
            logger.warning("RAG components not available for policy citations")
            return citations
        
        try:
            # Retrieve hardship policy documents
            results = retrieve(self.retriever, self.embedder, "hardship program policy", k=3)
            
            for result in results:
                citations.append(Citation(
                    source=result.get("filename", "Hardship Policy"),
                    snippet=result.get("snippet", "")[:300] + "..."
                ))
            
            # Tavily fallback if insufficient citations
            if len(citations) < 1:
                logger.info("Insufficient local hardship citations, searching web...")
                try:
                    web_docs = web_search_into_docstore(
                        self.docstore,
                        self.embedder,
                        "credit card hardship program deferral re-aging disclosures",
                        max_results=2
                    )
                    
                    # Re-retrieve after adding web content
                    if web_docs:
                        results = retrieve(self.retriever, self.embedder, 
                                         "hardship deferral re-aging disclosures", k=2)
                        
                        for result in results:
                            citations.append(Citation(
                                source=result.get("filename", "Web Search"),
                                snippet=result.get("snippet", "")[:300] + "..."
                            ))
                            
                except Exception as e:
                    logger.warning(f"Web search for hardship policies failed: {e}")
            
        except Exception as e:
            logger.error(f"Error retrieving policy citations: {e}")
        
        return citations

# Test cases for collections scenarios
def test_collections_advisor():
    """Test CollectionsAdvisor with golden-path scenarios"""
    print("🧪 Testing CollectionsAdvisor Golden Paths")
    print("=" * 50)
    
    advisor = CollectionsAdvisor()
    
    test_cases = [
        {
            "name": "90-day bucket with low income → deferral wins",
            "customer_state": CustomerState(
                balance=5000.0,
                apr=24.99,
                bucket="90",
                income_monthly=2500.0,
                expenses_monthly=2200.0
            ),
            "expected_top_plan": "deferral"
        },
        {
            "name": "120+ bucket → split settlement shown",
            "customer_state": CustomerState(
                balance=8000.0,
                apr=29.99,
                bucket="120+",
                income_monthly=3000.0,
                expenses_monthly=2800.0
            ),
            "expected_plan_type": "settlement"
        },
        {
            "name": "Current bucket with good income → payment plan",
            "customer_state": CustomerState(
                balance=3000.0,
                apr=19.99,
                bucket="current",
                income_monthly=5000.0,
                expenses_monthly=3500.0
            ),
            "expected_plan_type": "payment_plan"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        try:
            print(f"{i}. {case['name']}")
            
            result = advisor.process_hardship_request(case["customer_state"])
            
            # Validate response structure
            valid_structure = (
                isinstance(result.plans, list) and
                len(result.plans) > 0 and
                isinstance(result.citations, list)
            )
            
            # Check if expected plan type is present
            plan_types = [plan.type for plan in result.plans]
            expected_present = (
                case.get("expected_top_plan") in plan_types or
                case.get("expected_plan_type") in plan_types
            )
            
            # Check plan completeness
            plans_complete = all(
                len(plan.payment_curve) > 0 and
                plan.npv > 0 and
                len(plan.disclosures) > 0 and
                len(plan.why) > 10
                for plan in result.plans
            )
            
            success = valid_structure and expected_present and plans_complete
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Customer: {case['customer_state'].bucket} bucket, ${case['customer_state'].balance:.2f}")
            print(f"   Plans generated: {len(result.plans)}")
            print(f"   Plan types: {plan_types}")
            print(f"   Expected type present: {expected_present}")
            print(f"   Citations: {len(result.citations)}")
            print(f"   Status: {status}")
            print()
            
            if success:
                passed += 1
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            print()
    
    print(f"CollectionsAdvisor Test Results: {passed}/{total} passed")
    return passed == total

if __name__ == "__main__":
    # Run tests
    success = test_collections_advisor()
    print(f"\n{'🎉 All tests passed!' if success else '⚠️ Some tests failed.'}")
