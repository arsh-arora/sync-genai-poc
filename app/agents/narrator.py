"""
Portfolio Intel Narrator - Insights that Drive Action
Generates actionable insights from KPI analysis with impact ranking and action suggestions
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.llm.gemini import chat

logger = logging.getLogger(__name__)

# Pydantic models for structured responses
class Insight(BaseModel):
    metric: str
    value: float
    delta: float
    delta_percent: float
    confidence: float
    impact_score: float
    total_score: float
    action: str
    target_agent: str
    rationale: str

class NarratorResponse(BaseModel):
    response: str  # 2-4 bullets with numbers + 1-2 "try next"
    metadata: Dict[str, Any]  # ui_cards with insights

class PortfolioIntelNarrator:
    """
    Portfolio Intel Narrator - Insights that Drive Action
    Generates actionable insights from KPI analysis with impact ranking
    """
    
    def __init__(self, docstore=None, embedder=None, retriever=None, rules_loader=None):
        """Initialize Narrator with rules and mock data"""
        self.docstore = docstore
        self.embedder = embedder  
        self.retriever = retriever
        self.rules_loader = rules_loader
        
        # Load narrator rules and mock data
        self._load_narrator_rules()
        self._load_portfolio_metrics()
    
    def _load_narrator_rules(self):
        """Load narrator rules from YAML"""
        try:
            if self.rules_loader:
                self.narrator_rules = self.rules_loader.get_rules('narrator') or {}
                logger.info("Loaded narrator rules from rules_loader")
            else:
                self.narrator_rules = {}
                logger.warning("No rules_loader provided - using defaults")
        except Exception as e:
            logger.error(f"Failed to load narrator rules: {e}")
            self.narrator_rules = {}
    
    def _load_portfolio_metrics(self):
        """Load mock portfolio metrics from JSON"""
        try:
            metrics_path = Path("synchrony-demo-rules-repo/fixtures/narrator/mock_portfolio_metrics.json")
            with open(metrics_path, 'r') as f:
                self.portfolio_data = json.load(f)
            logger.info("Loaded portfolio metrics data")
        except Exception as e:
            logger.error(f"Failed to load portfolio metrics: {e}")
            self.portfolio_data = {"segments": []}
    
    def generate_insights(self, query: str = "") -> NarratorResponse:
        """
        Main entry point: generate insights that drive action
        
        Args:
            query: Optional query to focus insights (unused for now - always analyzes all KPIs)
            
        Returns:
            NarratorResponse with insights and action suggestions
        """
        try:
            logger.info("Generating portfolio insights with KPI analysis")
            
            # Step 1: Calculate KPIs using formulas from rules
            kpi_results = self._calculate_kpis()
            
            # Step 2: Generate insights with impact and confidence scoring
            insights = self._generate_insights(kpi_results)
            
            # Step 3: Rank insights by weighted score (impact 0.6, confidence 0.4) 
            ranked_insights = self._rank_insights(insights)
            
            # Step 4: Select top insights and add action suggestions
            actionable_insights = self._add_action_suggestions(ranked_insights[:4])
            
            # Step 5: Generate response text (2-4 bullets + try next)
            response_text = self._generate_response_text(actionable_insights)
            
            # Step 6: Build UI cards for metadata
            ui_cards = self._build_insight_ui_cards(actionable_insights)
            
            return NarratorResponse(
                response=response_text,
                metadata={
                    "ui_cards": ui_cards,
                    "kpis_analyzed": len(kpi_results),
                    "insights_generated": len(insights),
                    "top_insights": len(actionable_insights),
                    "data_date": self.portfolio_data.get("date", "unknown")
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return NarratorResponse(
                response=f"Error generating portfolio insights: {str(e)}",
                metadata={
                    "ui_cards": [],
                    "error": str(e)
                }
            )
    
    def _calculate_kpis(self) -> List[Dict[str, Any]]:
        """
        Calculate KPIs using formulas from rules/narrator.yml
        
        Returns:
            List of KPI calculations with current/previous values
        """
        kpi_results = []
        kpis = self.narrator_rules.get("kpis", {})
        
        for segment in self.portfolio_data.get("segments", []):
            platform = segment["platform"]
            
            for kpi_name, kpi_config in kpis.items():
                try:
                    # Get current and previous values
                    current_value = segment.get(kpi_name)
                    previous_value = segment.get("previous_values", {}).get(kpi_name)
                    
                    if current_value is not None and previous_value is not None:
                        # Calculate delta
                        delta = current_value - previous_value
                        delta_percent = (delta / previous_value) * 100 if previous_value != 0 else 0
                        
                        # Calculate confidence based on data completeness
                        confidence = self._calculate_confidence(segment, kpi_name, kpi_config)
                        
                        kpi_results.append({
                            "platform": platform,
                            "metric": kpi_name,
                            "current_value": current_value,
                            "previous_value": previous_value,
                            "delta": delta,
                            "delta_percent": delta_percent,
                            "confidence": confidence,
                            "formula": kpi_config.get("formula", ""),
                            "caveats": kpi_config.get("caveats", ""),
                            "thresholds": kpi_config.get("thresholds", {})
                        })
                        
                except Exception as e:
                    logger.warning(f"Error calculating KPI {kpi_name} for {platform}: {e}")
                    continue
        
        logger.info(f"Calculated {len(kpi_results)} KPI results")
        return kpi_results
    
    def _calculate_confidence(self, segment: Dict[str, Any], kpi_name: str, kpi_config: Dict[str, Any]) -> float:
        """Calculate confidence score based on data completeness and reliability"""
        confidence_factors = []
        
        # Data completeness
        required_fields = ["total_apps", "total_txn", "total_balance", "total_accounts"]
        completeness = sum(1 for field in required_fields if segment.get(field, 0) > 0) / len(required_fields)
        confidence_factors.append(completeness)
        
        # Sample size (higher sample = higher confidence)
        total_apps = segment.get("total_apps", 0)
        sample_score = min(total_apps / 10000, 1.0) if total_apps > 0 else 0.5
        confidence_factors.append(sample_score)
        
        # Threshold alignment (values in expected ranges)
        thresholds = kpi_config.get("thresholds", {})
        current_value = segment.get(kpi_name, 0)
        if thresholds:
            low = thresholds.get("low", 0)
            high = thresholds.get("high", 1)
            if low <= current_value <= high:
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.7)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _generate_insights(self, kpi_results: List[Dict[str, Any]]) -> List[Insight]:
        """
        Generate insights from KPI calculations
        
        Args:
            kpi_results: List of KPI calculation results
            
        Returns:
            List of insights with impact scores
        """
        insights = []
        
        for kpi_data in kpi_results:
            try:
                # Calculate impact score based on delta magnitude and business importance
                impact_score = self._calculate_impact_score(kpi_data)
                
                # Only create insights for significant changes
                if abs(kpi_data["delta_percent"]) >= 5.0 or impact_score >= 0.6:
                    insight = Insight(
                        metric=f"{kpi_data['platform']}_{kpi_data['metric']}",
                        value=kpi_data["current_value"],
                        delta=kpi_data["delta"],
                        delta_percent=kpi_data["delta_percent"],
                        confidence=kpi_data["confidence"],
                        impact_score=impact_score,
                        total_score=0.0,  # Will be calculated in ranking
                        action="",        # Will be added later
                        target_agent="",  # Will be added later  
                        rationale=f"{kpi_data['metric']} changed {kpi_data['delta_percent']:.1f}% for {kpi_data['platform']}"
                    )
                    insights.append(insight)
                    
            except Exception as e:
                logger.warning(f"Error generating insight for {kpi_data.get('metric', 'unknown')}: {e}")
                continue
        
        logger.info(f"Generated {len(insights)} insights from {len(kpi_results)} KPIs")
        return insights
    
    def _calculate_impact_score(self, kpi_data: Dict[str, Any]) -> float:
        """Calculate business impact score for a KPI change"""
        metric = kpi_data["metric"]
        delta_percent = abs(kpi_data["delta_percent"])
        platform = kpi_data["platform"]
        
        # Base impact from change magnitude
        magnitude_score = min(delta_percent / 20.0, 1.0)  # Cap at 20% change
        
        # Business importance weights
        importance_weights = {
            "approval_rate": 0.9,    # High impact - affects acquisition
            "charge_off_rate": 0.95, # Very high impact - affects losses
            "promo_uptake": 0.7,     # Medium-high impact - affects revenue  
            "revolve_rate": 0.8,     # High impact - affects profitability
            "funnel_conversion": 0.6, # Medium impact - affects efficiency
            "acquisition_cost": 0.7, # Medium-high impact - affects unit economics
            "portfolio_yield": 0.85  # High impact - affects revenue
        }
        
        business_weight = importance_weights.get(metric, 0.5)
        
        # Platform importance (higher volume = higher impact)
        platform_weights = {"digital": 1.0, "home_auto": 0.8, "carecredit": 0.7}
        platform_weight = platform_weights.get(platform, 0.5)
        
        # Direction matters (negative changes in good metrics = higher impact)
        direction_multiplier = 1.0
        if metric in ["approval_rate", "promo_uptake", "revolve_rate", "funnel_conversion", "portfolio_yield"]:
            # Good metrics - declining is worse
            if kpi_data["delta"] < 0:
                direction_multiplier = 1.2
        elif metric in ["charge_off_rate", "acquisition_cost"]:
            # Bad metrics - increasing is worse  
            if kpi_data["delta"] > 0:
                direction_multiplier = 1.2
        
        impact_score = magnitude_score * business_weight * platform_weight * direction_multiplier
        return min(impact_score, 1.0)
    
    def _rank_insights(self, insights: List[Insight]) -> List[Insight]:
        """
        Rank insights by weighted score: impact (0.6) + confidence (0.4)
        
        Args:
            insights: List of insights to rank
            
        Returns:
            List of insights sorted by total score (highest first)
        """
        weights = self.narrator_rules.get("insight_ranking", {}).get("weights", {})
        impact_weight = weights.get("impact", 0.6)
        confidence_weight = weights.get("confidence", 0.4)
        
        for insight in insights:
            insight.total_score = (insight.impact_score * impact_weight + 
                                 insight.confidence * confidence_weight)
        
        ranked_insights = sorted(insights, key=lambda x: x.total_score, reverse=True)
        logger.info(f"Ranked {len(ranked_insights)} insights, top score: {ranked_insights[0].total_score:.2f}")
        
        return ranked_insights
    
    def _add_action_suggestions(self, top_insights: List[Insight]) -> List[Insight]:
        """
        Add action suggestions linking to imagegen or offer agents
        
        Args:
            top_insights: Top ranked insights
            
        Returns:
            Insights with action suggestions added
        """
        action_rules = self.narrator_rules.get("action_suggestions", {})
        
        for insight in top_insights:
            # Extract base metric name (remove platform prefix)
            base_metric = insight.metric.split("_", 1)[-1]
            current_value = insight.value
            
            # Determine action based on metric performance
            action_key = None
            if base_metric == "approval_rate" and current_value < 0.5:
                action_key = "low_approval_rate"
            elif base_metric == "promo_uptake" and current_value < 0.3:
                action_key = "low_promo_uptake"
            elif base_metric == "charge_off_rate" and current_value > 0.02:
                action_key = "high_charge_off"
            elif base_metric == "funnel_conversion" and current_value < 0.3:
                action_key = "low_funnel_conversion"
            
            if action_key and action_key in action_rules:
                rule = action_rules[action_key]
                insight.action = rule["action"]
                insight.target_agent = rule["target_agent"]
                insight.rationale += f" → {rule['rationale']}"
            else:
                # Default actions based on trend
                if insight.delta < 0:
                    insight.action = "launch_asset"
                    insight.target_agent = "imagegen"
                    insight.rationale += " → Create promotional materials to reverse trend"
                else:
                    insight.action = "monitor"
                    insight.target_agent = "narrator"
                    insight.rationale += " → Continue monitoring performance"
        
        return top_insights
    
    def _generate_response_text(self, insights: List[Insight]) -> str:
        """
        Generate response text: 2-4 bullets with numbers + 1-2 "try next"
        
        Args:
            insights: Top actionable insights
            
        Returns:
            Response text with key findings and next actions
        """
        response_parts = []
        
        # Header
        response_parts.append("**📊 Portfolio Insights & Action Items**")
        response_parts.append("")
        
        # Key insights (2-4 bullets with numbers)
        insights_to_show = insights[:4]
        for i, insight in enumerate(insights_to_show, 1):
            platform = insight.metric.split("_")[0]
            metric_name = insight.metric.split("_", 1)[1].replace("_", " ").title()
            
            # Format the insight with numbers
            direction = "↑" if insight.delta > 0 else "↓"
            response_parts.append(
                f"**{i}. {platform.title()} {metric_name}: {insight.value:.1%} {direction}**"
            )
            response_parts.append(
                f"   Changed {insight.delta_percent:+.1f}% • Impact: {insight.impact_score:.2f} • Confidence: {insight.confidence:.2f}"
            )
            response_parts.append("")
        
        # Try next actions (1-2 suggestions)
        response_parts.append("**🎯 Try Next:**")
        
        action_suggestions = []
        for insight in insights_to_show[:2]:  # Top 2 actions
            if insight.action == "launch_asset":
                action_suggestions.append(f"Launch {insight.target_agent} asset for {insight.metric.split('_')[0]} performance")
            elif insight.action == "promo_tuning":
                action_suggestions.append(f"Optimize {insight.target_agent} promotional terms")
            elif insight.action == "portfolio_review":
                action_suggestions.append(f"Review {insight.target_agent} risk policies")
        
        # Deduplicate and add
        unique_actions = list(dict.fromkeys(action_suggestions))
        for i, action in enumerate(unique_actions[:2], 1):
            response_parts.append(f"{i}. {action}")
        
        return "\n".join(response_parts)
    
    def _build_insight_ui_cards(self, insights: List[Insight]) -> List[Dict[str, Any]]:
        """
        Build UI cards for insights
        
        Args:
            insights: List of insights
            
        Returns:
            List of UI cards for metadata
        """
        ui_cards = []
        
        for insight in insights:
            platform = insight.metric.split("_")[0]
            metric_name = insight.metric.split("_", 1)[1]
            
            ui_cards.append({
                "type": "insight",
                "platform": platform,
                "metric": metric_name,
                "value": round(insight.value, 4),
                "delta": round(insight.delta, 4),
                "delta_percent": round(insight.delta_percent, 1),
                "confidence": round(insight.confidence, 2),
                "impact_score": round(insight.impact_score, 2),
                "total_score": round(insight.total_score, 2),
                "action": insight.action,
                "target_agent": insight.target_agent,
                "rationale": insight.rationale
            })
        
        return ui_cards
    
    # Compatibility method for supervisor integration
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process narrator query for supervisor integration
        
        Args:
            query: User query about portfolio insights
            
        Returns:
            Dict with response, metadata, confidence, sources
        """
        try:
            result = self.generate_insights(query)
            
            return {
                "response": result.response,
                "metadata": result.metadata,
                "confidence": 0.8,
                "sources": []
            }
            
        except Exception as e:
            logger.error(f"Narrator process_query error: {e}")
            return {
                "response": f"Error generating portfolio insights: {str(e)}",
                "confidence": 0.2,
                "sources": [],
                "metadata": {"error": str(e)}
            }
