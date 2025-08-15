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

class PaymentPlan(BaseModel):
    plan_id: str
    name: str
    description: str
    monthly_payment: float
    duration_months: int
    total_paid: float
    payment_schedule: List[PaymentScheduleEntry]
    rationale: str
    score: float
    affordability_score: float
    bank_npv_score: float
    cure_probability_score: float

class CollectionsResponse(BaseModel):
    response: str  # top 2-3 options + why
    metadata: Dict[str, Any]  # ui_cards, disclosures, handoffs

class CollectionsAdvisor:
    """
    Collections & Hardship Advisor for proposing compliant hardship plans
    """
    
    def __init__(self, docstore=None, embedder=None, retriever=None, rules_loader=None):
        """Initialize CollectionsAdvisor with RAG components and rules loader"""
        self.docstore = docstore
        self.embedder = embedder
        self.retriever = retriever
        self.rules_loader = rules_loader
        
        # Load collections rules and policies
        self._load_collections_rules()
        self._load_hardship_policies()
    
    def _load_collections_rules(self):
        """Load collections rules from YAML"""
        try:
            if self.rules_loader:
                self.collections_rules = self.rules_loader.get_rules('collections') or {}
                logger.info("Loaded collections rules from rules_loader")
            else:
                self.collections_rules = {}
                logger.warning("No rules_loader provided - using defaults")
        except Exception as e:
            logger.error(f"Failed to load collections rules: {e}")
            self.collections_rules = {}
    
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
        Main processing pipeline: humane, policy-aware plans
        
        Args:
            customer_state: Customer's financial state and account information
            
        Returns:
            CollectionsResponse with top 2-3 options + rationale, UI cards, disclosures
        """
        try:
            logger.info(f"Processing hardship request for {customer_state.bucket} bucket, balance ${customer_state.balance:.2f}")
            
            # Step 1: Get plan menu from rules/collections.yml
            available_plans = self._get_plan_menu(customer_state)
            
            # Step 2: Simulate payment schedules with demo APR
            simulated_plans = self._simulate_payment_plans(available_plans, customer_state)
            
            # Step 3: Rank plans with weights (affordability, bank NPV, cure probability)
            ranked_plans = self._rank_payment_plans(simulated_plans, customer_state)
            
            # Step 4: Select top 2-3 options
            top_plans = ranked_plans[:3]
            
            # Step 5: Generate response with rationale
            response_text = self._generate_response_text(top_plans, customer_state)
            
            # Step 6: Build UI cards for metadata
            ui_cards = self._build_plan_ui_cards(top_plans)
            
            # Step 7: Include mandatory disclosures and handoffs
            disclosures = ["collections_generic"]
            handoffs = ["contracts"]  # for terms clarifications
            
            return CollectionsResponse(
                response=response_text,
                metadata={
                    "ui_cards": ui_cards,
                    "disclosures": disclosures,
                    "handoffs": handoffs,
                    "total_plans_evaluated": len(simulated_plans),
                    "customer_bucket": customer_state.bucket,
                    "balance": customer_state.balance
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing hardship request: {e}")
            return CollectionsResponse(
                response=f"Error processing hardship request: {str(e)}",
                metadata={
                    "ui_cards": [],
                    "disclosures": ["collections_generic"],
                    "handoffs": [],
                    "error": str(e)
                }
            )
    
    def _get_plan_menu(self, customer_state: CustomerState) -> List[Dict[str, Any]]:
        """
        Get available plans from rules/collections.yml filtered by eligibility
        
        Args:
            customer_state: Customer financial information
            
        Returns:
            List of eligible payment plans
        """
        available_plans = []
        rules_plans = self.collections_rules.get("plans", {})
        
        for plan_id, plan_config in rules_plans.items():
            # Check eligibility
            eligibility = plan_config.get("eligibility", {})
            min_balance = eligibility.get("min_balance", 0)
            max_balance = eligibility.get("max_balance", float('inf'))
            
            if min_balance <= customer_state.balance <= max_balance:
                available_plans.append({
                    "plan_id": plan_id,
                    "config": plan_config
                })
                
        logger.info(f"Found {len(available_plans)} eligible plans for balance ${customer_state.balance}")
        return available_plans
    
    def _simulate_payment_plans(self, available_plans: List[Dict[str, Any]], customer_state: CustomerState) -> List[PaymentPlan]:
        """
        Simulate payment schedules with demo APR and compute totals
        
        Args:
            available_plans: List of eligible plans
            customer_state: Customer financial information
            
        Returns:
            List of simulated payment plans
        """
        simulated_plans = []
        sim_params = self.collections_rules.get("simulation_params", {})
        demo_apr = sim_params.get("demo_apr", 0.2399)  # 23.99%
        
        for plan_data in available_plans:
            plan_id = plan_data["plan_id"]
            config = plan_data["config"]
            
            # Calculate payment schedule based on plan type
            payment_schedule = self._calculate_payment_schedule(
                balance=customer_state.balance,
                apr=demo_apr,
                plan_config=config
            )
            
            # Calculate totals
            total_paid = sum(entry.payment for entry in payment_schedule)
            avg_monthly = total_paid / len(payment_schedule) if payment_schedule else 0
            
            plan = PaymentPlan(
                plan_id=plan_id,
                name=config.get("name", plan_id.replace("_", " ").title()),
                description=config.get("description", ""),
                monthly_payment=avg_monthly,
                duration_months=config.get("duration_months", 12),
                total_paid=total_paid,
                payment_schedule=payment_schedule,
                rationale="",  # Will be filled in ranking
                score=0.0,     # Will be calculated in ranking
                affordability_score=0.0,
                bank_npv_score=0.0,
                cure_probability_score=0.0
            )
            
            simulated_plans.append(plan)
        
        logger.info(f"Simulated {len(simulated_plans)} payment plans")
        return simulated_plans
    
    def _calculate_payment_schedule(self, balance: float, apr: float, plan_config: Dict[str, Any]) -> List[PaymentScheduleEntry]:
        """
        Calculate payment schedule based on plan type
        
        Args:
            balance: Outstanding balance
            apr: Annual percentage rate
            plan_config: Plan configuration from rules
            
        Returns:
            List of payment schedule entries
        """
        payment_schedule = []
        monthly_rate = apr / 12
        duration_months = plan_config.get("duration_months", 12)
        payment_type = plan_config.get("payment_type", "fixed")
        min_payment_percent = plan_config.get("min_payment_percent", 0.025)
        
        current_balance = balance
        
        for month in range(1, duration_months + 1):
            if payment_type == "interest_only":
                # Interest-only payments
                interest = current_balance * monthly_rate
                principal = 0
                payment = interest
                
            elif payment_type == "deferred":
                # Deferred payments - only interest accrues
                interest = current_balance * monthly_rate
                principal = 0
                payment = 0
                current_balance += interest  # Interest compounds
                
            elif payment_type == "reduced":
                # Reduced payments (minimum percent of balance)
                payment = max(current_balance * min_payment_percent, 25.0)  # Min $25
                interest = current_balance * monthly_rate
                principal = max(0, payment - interest)
                
            elif payment_type == "fixed":
                # Fixed payment plan to pay off balance
                if month == 1:
                    # Calculate fixed payment using amortization formula
                    if monthly_rate > 0:
                        fixed_payment = (balance * monthly_rate * (1 + monthly_rate) ** duration_months) / ((1 + monthly_rate) ** duration_months - 1)
                    else:
                        fixed_payment = balance / duration_months
                
                payment = fixed_payment
                interest = current_balance * monthly_rate
                principal = payment - interest
            
            else:
                # Default to minimum payment
                payment = current_balance * min_payment_percent
                interest = current_balance * monthly_rate
                principal = max(0, payment - interest)
            
            # Update balance
            current_balance = max(0, current_balance - principal)
            
            payment_schedule.append(PaymentScheduleEntry(
                month=month,
                payment=round(payment, 2),
                interest=round(interest, 2),
                principal=round(principal, 2),
                balance=round(current_balance, 2)
            ))
            
            # Break if balance is paid off
            if current_balance <= 0:
                break
        
        return payment_schedule
    
    def _rank_payment_plans(self, plans: List[PaymentPlan], customer_state: CustomerState) -> List[PaymentPlan]:
        """
        Rank plans using weighted scoring: affordability, bank NPV, cure probability
        
        Args:
            plans: List of simulated payment plans
            customer_state: Customer financial information
            
        Returns:
            List of ranked payment plans (highest score first)
        """
        weights = self.collections_rules.get("scoring", {}).get("weights", {})
        affordability_weight = weights.get("affordability", 0.4)
        bank_npv_weight = weights.get("bank_npv", 0.3)
        cure_probability_weight = weights.get("cure_probability", 0.3)
        
        for plan in plans:
            # Calculate affordability score (lower monthly payment = higher score)
            max_monthly = max(p.monthly_payment for p in plans) if plans else plan.monthly_payment
            plan.affordability_score = 1.0 - (plan.monthly_payment / max_monthly) if max_monthly > 0 else 1.0
            
            # Calculate bank NPV score (higher total paid = higher score for bank)
            max_total = max(p.total_paid for p in plans) if plans else plan.total_paid
            plan.bank_npv_score = plan.total_paid / max_total if max_total > 0 else 1.0
            
            # Calculate cure probability (shorter term + reasonable payment = higher score)
            duration_score = 1.0 - (plan.duration_months / 24.0)  # Prefer shorter terms
            payment_ratio = plan.monthly_payment / customer_state.balance if customer_state.balance > 0 else 0
            payment_reasonableness = 1.0 - min(payment_ratio, 0.1) / 0.1  # Reasonable if <10% of balance
            plan.cure_probability_score = (duration_score + payment_reasonableness) / 2
            
            # Calculate weighted total score
            plan.score = (
                plan.affordability_score * affordability_weight +
                plan.bank_npv_score * bank_npv_weight +
                plan.cure_probability_score * cure_probability_weight
            )
            
            # Generate rationale
            plan.rationale = self._generate_plan_rationale(plan, customer_state)
        
        # Sort by score (highest first)
        ranked_plans = sorted(plans, key=lambda p: p.score, reverse=True)
        logger.info(f"Ranked {len(ranked_plans)} plans, top score: {ranked_plans[0].score:.2f}")
        
        return ranked_plans
    
    def _generate_plan_rationale(self, plan: PaymentPlan, customer_state: CustomerState) -> str:
        """Generate rationale for why this plan is recommended"""
        rationale_parts = []
        
        if plan.affordability_score > 0.7:
            rationale_parts.append("Affordable monthly payments")
        elif plan.affordability_score > 0.4:
            rationale_parts.append("Moderate monthly payments")
        else:
            rationale_parts.append("Higher payments but faster resolution")
        
        if plan.duration_months <= 6:
            rationale_parts.append("short-term relief")
        elif plan.duration_months <= 12:
            rationale_parts.append("balanced timeline")
        else:
            rationale_parts.append("extended payment period")
        
        if customer_state.bucket in ["90", "120+"]:
            rationale_parts.append("suitable for delinquent accounts")
        
        return " • ".join(rationale_parts).capitalize()
    
    def _generate_response_text(self, top_plans: List[PaymentPlan], customer_state: CustomerState) -> str:
        """
        Generate response text with top 2-3 options + why
        
        Args:
            top_plans: Top ranked payment plans
            customer_state: Customer financial information
            
        Returns:
            Response text explaining the recommended options
        """
        response_parts = []
        
        # Header
        response_parts.append(f"**Payment Plan Options for ${customer_state.balance:,.2f} Balance**")
        response_parts.append(f"Account Status: {customer_state.bucket} days")
        response_parts.append("")
        
        # Top options
        for i, plan in enumerate(top_plans, 1):
            response_parts.append(f"**Option {i}: {plan.name}**")
            response_parts.append(f"• Monthly Payment: ${plan.monthly_payment:.2f}")
            response_parts.append(f"• Duration: {plan.duration_months} months")
            response_parts.append(f"• Total Paid: ${plan.total_paid:,.2f}")
            response_parts.append(f"• Why: {plan.rationale}")
            response_parts.append("")
        
        # Next steps
        response_parts.append("**Next Steps:**")
        response_parts.append("1. Review plan details with our specialist")
        response_parts.append("2. Provide income verification if required")
        response_parts.append("3. Set up automatic payments")
        response_parts.append("4. Contact us for terms clarification if needed")
        
        return "\n".join(response_parts)
    
    def _build_plan_ui_cards(self, plans: List[PaymentPlan]) -> List[Dict[str, Any]]:
        """
        Build UI cards for payment plans
        
        Args:
            plans: List of payment plans
            
        Returns:
            List of UI cards for metadata
        """
        ui_cards = []
        
        for plan in plans:
            # Build payment schedule summary (first 3 months + last month)
            schedule_summary = []
            for entry in plan.payment_schedule[:3]:
                schedule_summary.append({
                    "month": entry.month,
                    "payment": entry.payment,
                    "balance": entry.balance
                })
            
            if len(plan.payment_schedule) > 3:
                last_entry = plan.payment_schedule[-1]
                schedule_summary.append({
                    "month": last_entry.month,
                    "payment": last_entry.payment,
                    "balance": last_entry.balance
                })
            
            ui_cards.append({
                "type": "plan",
                "name": plan.name,
                "description": plan.description,
                "monthly_payment": plan.monthly_payment,
                "duration_months": plan.duration_months,
                "total_paid": plan.total_paid,
                "schedule": schedule_summary,
                "rationale": plan.rationale,
                "score": round(plan.score, 2),
                "affordability": round(plan.affordability_score, 2),
                "bank_npv": round(plan.bank_npv_score, 2),
                "cure_probability": round(plan.cure_probability_score, 2)
            })
        
        return ui_cards
    
    # Compatibility method for supervisor integration
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process collections query for supervisor integration
        
        Args:
            query: User query about collections/hardship
            
        Returns:
            Dict with response, metadata, confidence, sources
        """
        try:
            # Extract balance and bucket from query (simplified)
            import re
            
            balance_match = re.search(r'\$?([\d,]+(?:\.\d{2})?)', query)
            balance = float(balance_match.group(1).replace(',', '')) if balance_match else 1000.0
            
            # Determine bucket based on query keywords
            if any(word in query.lower() for word in ['overdue', 'late', 'behind']):
                bucket = "90"
            elif any(word in query.lower() for word in ['current', 'up to date']):
                bucket = "current"
            else:
                bucket = "60"  # Default
            
            customer_state = CustomerState(
                balance=balance,
                apr=0.2399,  # Default APR
                bucket=bucket
            )
            
            result = self.process_hardship_request(customer_state)
            
            return {
                "response": result.response,
                "metadata": result.metadata,
                "confidence": 0.8,
                "sources": []
            }
            
        except Exception as e:
            logger.error(f"Collections process_query error: {e}")
            return {
                "response": f"Error processing collections query: {str(e)}",
                "confidence": 0.2,
                "sources": [],
                "metadata": {"error": str(e)}
            }
