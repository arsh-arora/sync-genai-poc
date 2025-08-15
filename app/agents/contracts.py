"""
Merchant Onboarding & Contract Intelligence
Auto-ingest merchant agreements, extract key terms, generate operational checklists
"""

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.llm.gemini import chat
from app.rag.core import retrieve
from app.tools.tavily_search import web_search_into_docstore

logger = logging.getLogger(__name__)

# Pydantic models for strict input/output schema enforcement
class ContractSection(BaseModel):
    heading: str
    text: str
    page_span: str

class ChecklistItem(BaseModel):
    task: str
    owner: str
    due_in_days: int

class RiskFlag(BaseModel):
    severity: str  # low|med|high
    note: str

class Delta(BaseModel):
    field: str
    from_value: Any = None
    to_value: Any = None

class Citation(BaseModel):
    source: str
    snippet: str

class ContractResponse(BaseModel):
    summary: str
    extractions: Dict[str, Any]
    checklist: List[ChecklistItem]
    risk_flags: List[RiskFlag]
    deltas: List[Delta]
    citations: List[Citation]

class ContractIntelligence:
    """
    Merchant Onboarding & Contract Intelligence for automated contract analysis
    """
    
    def __init__(self, docstore=None, embedder=None, retriever=None, rules_loader=None):
        """Initialize ContractIntelligence with RAG components for terms retrieval"""
        self.docstore = docstore
        self.embedder = embedder
        self.retriever = retriever
        
        # Load contract ontology
        self._load_contract_ontology()
    
    def _load_contract_ontology(self):
        """Load contract ontology from knowledge base"""
        try:
            ontology_path = Path("app/kb/contract_ontology.md")
            if ontology_path.exists():
                self.ontology_doc = ontology_path.read_text()
                logger.info("Loaded contract ontology successfully")
            else:
                self.ontology_doc = ""
                logger.warning("Contract ontology not found")
        except Exception as e:
            logger.error(f"Failed to load contract ontology: {e}")
            self.ontology_doc = ""
    
    def process_contract(self, file_path: str, prev_version_path: Optional[str] = None) -> ContractResponse:
        """
        Main processing pipeline for contract analysis
        
        Args:
            file_path: Path to the contract file to analyze
            prev_version_path: Optional path to previous version for delta analysis
            
        Returns:
            ContractResponse with extractions, checklist, risk flags, and deltas
        """
        try:
            logger.info(f"Processing contract: {file_path}")
            
            # Step 1: Parse contract into sections
            sections = self.contract_parse(file_path)
            logger.info(f"Parsed {len(sections)} contract sections")
            
            # Step 2: Map sections to ontology using Gemini
            extractions = self.ontology_map(sections, self.ontology_doc)
            logger.info("Completed ontology mapping")
            
            # Step 3: Delta detection if previous version provided
            deltas = []
            if prev_version_path:
                prev_sections = self.contract_parse(prev_version_path)
                prev_extractions = self.ontology_map(prev_sections, self.ontology_doc)
                deltas = self.delta_detect(extractions, prev_extractions)
                logger.info(f"Detected {len(deltas)} deltas from previous version")
            
            # Step 4: Build operational checklist
            checklist = self._build_checklist(extractions)
            logger.info(f"Generated {len(checklist)} checklist items")
            
            # Step 5: Identify risk flags
            risk_flags = self._identify_risk_flags(extractions)
            logger.info(f"Identified {len(risk_flags)} risk flags")
            
            # Step 6: Get policy citations
            citations = self._get_contract_citations(extractions)
            
            # Step 7: Generate summary
            summary = self._generate_summary(extractions, risk_flags, deltas)
            
            return ContractResponse(
                summary=summary,
                extractions=extractions,
                checklist=checklist,
                risk_flags=risk_flags,
                deltas=deltas,
                citations=citations
            )
            
        except Exception as e:
            logger.error(f"Error processing contract: {e}")
            return ContractResponse(
                summary=f"Error processing contract: {str(e)}",
                extractions={},
                checklist=[],
                risk_flags=[],
                deltas=[],
                citations=[]
            )
    
    def contract_parse(self, path: str) -> List[ContractSection]:
        """
        Parse contract file into structured sections
        
        Args:
            path: Path to contract file (supports .md, .txt, and basic .pdf)
            
        Returns:
            List of ContractSection objects with heading, text, and page_span
        """
        try:
            file_path = Path(path)
            
            if not file_path.exists():
                logger.error(f"Contract file not found: {path}")
                return []
            
            # Read file content based on extension
            if file_path.suffix.lower() == '.pdf':
                # For PDF files, assume pre-extracted markdown version exists
                md_path = file_path.with_suffix('.md')
                if md_path.exists():
                    content = md_path.read_text(encoding='utf-8')
                    logger.info(f"Using pre-extracted markdown: {md_path}")
                else:
                    # Basic PDF text extraction (would use pdftotext in production)
                    logger.warning(f"PDF extraction not implemented, using placeholder for: {path}")
                    content = f"# Contract Document\n\nPDF content from {path} would be extracted here."
            else:
                content = file_path.read_text(encoding='utf-8')
            
            # Parse sections based on markdown headers and common contract patterns
            sections = []
            current_section = {"heading": "Preamble", "text": "", "page_span": "1"}
            
            lines = content.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Detect section headers
                if self._is_section_header(line):
                    # Save previous section if it has content
                    if current_section["text"].strip():
                        sections.append(ContractSection(
                            heading=current_section["heading"],
                            text=current_section["text"].strip(),
                            page_span=current_section["page_span"]
                        ))
                    
                    # Start new section
                    current_section = {
                        "heading": self._clean_header(line),
                        "text": "",
                        "page_span": str((i // 50) + 1)  # Rough page estimation
                    }
                else:
                    # Add line to current section
                    if line:  # Skip empty lines
                        current_section["text"] += line + "\n"
            
            # Add final section
            if current_section["text"].strip():
                sections.append(ContractSection(
                    heading=current_section["heading"],
                    text=current_section["text"].strip(),
                    page_span=current_section["page_span"]
                ))
            
            logger.info(f"Parsed {len(sections)} sections from {path}")
            return sections
            
        except Exception as e:
            logger.error(f"Error parsing contract {path}: {e}")
            return []
    
    def _is_section_header(self, line: str) -> bool:
        """Detect if a line is a section header"""
        line = line.strip()
        
        # Markdown headers
        if line.startswith('#'):
            return True
        
        # Numbered sections
        if re.match(r'^\d+\.?\s+[A-Z]', line):
            return True
        
        # Article/Section patterns
        if re.match(r'^(ARTICLE|SECTION|SCHEDULE|APPENDIX|EXHIBIT)\s+[IVX\d]', line, re.IGNORECASE):
            return True
        
        # All caps headers (common in legal documents)
        if len(line) > 3 and line.isupper() and not re.search(r'[.!?]$', line):
            return True
        
        return False
    
    def _clean_header(self, line: str) -> str:
        """Clean and normalize section headers"""
        line = line.strip()
        
        # Remove markdown markers
        line = re.sub(r'^#+\s*', '', line)
        
        # Remove numbering
        line = re.sub(r'^\d+\.?\s*', '', line)
        
        # Remove article/section prefixes
        line = re.sub(r'^(ARTICLE|SECTION|SCHEDULE|APPENDIX|EXHIBIT)\s+[IVX\d]+\.?\s*', '', line, flags=re.IGNORECASE)
        
        return line.strip()
    
    def ontology_map(self, sections: List[ContractSection], ontology_doc: str) -> Dict[str, Any]:
        """
        Map contract sections to ontology using Gemini structured extraction
        
        Args:
            sections: Parsed contract sections
            ontology_doc: Contract ontology documentation
            
        Returns:
            Dictionary with extracted terms mapped to ontology categories
        """
        try:
            # Combine all section text for analysis
            full_text = "\n\n".join([f"## {section.heading}\n{section.text}" for section in sections])
            
            system_prompt = f"""You are a contract analysis expert. Extract key terms from the contract text according to the provided ontology.

ONTOLOGY REFERENCE:
{ontology_doc[:3000]}...

Extract information for these categories:
1. fees (setup_fees, monthly_fees, transaction_fees, volume_discounts, penalty_fees, termination_fees, payment_terms, fee_escalation)
2. sla (uptime_guarantee, response_times, resolution_times, processing_times, reporting_frequency, maintenance_windows, escalation_procedures, performance_penalties)
3. brand_usage (logo_usage_rights, trademark_usage, co_branding_requirements, marketing_approval, brand_guidelines, exclusivity_rights, attribution_requirements, usage_restrictions)
4. data_sharing (data_types_shared, data_retention_period, data_security_requirements, third_party_sharing, customer_consent_requirements, data_deletion_rights, compliance_standards, breach_notification)
5. security (security_certifications, encryption_requirements, access_controls, vulnerability_management, incident_response, audit_requirements, employee_screening, physical_security)
6. termination (termination_notice_period, termination_for_cause, termination_without_cause, data_return_obligations, transition_assistance, post_termination_restrictions, survival_clauses, termination_fees)
7. penalties (late_payment_penalties, performance_penalties, compliance_violations, data_breach_penalties, liquidated_damages, penalty_caps, cure_periods, escalating_penalties)
8. audit_rights (audit_frequency, audit_scope, audit_notice_period, audit_costs, audit_access_rights, third_party_audits, audit_remediation, audit_reporting)
9. marketing_obligations (marketing_spend_commitments, promotional_requirements, event_participation, content_creation, lead_generation, co_marketing_activities, marketing_performance_metrics, marketing_approval_process)

Return ONLY a JSON object with the extracted information. Use null for fields not found in the contract."""

            user_message = f"Extract key terms from this contract:\n\n{full_text[:8000]}"  # Limit for token constraints
            messages = [{"role": "user", "content": user_message}]
            
            response = chat(messages, system=system_prompt)
            
            # Parse Gemini response
            try:
                extractions = json.loads(response.strip())
                logger.info("Successfully extracted contract terms with Gemini")
                return extractions
            except json.JSONDecodeError:
                logger.warning("Gemini response not valid JSON, using fallback extraction")
                return self._fallback_extraction(sections)
                
        except Exception as e:
            logger.error(f"Error in ontology mapping: {e}")
            return self._fallback_extraction(sections)
    
    def _fallback_extraction(self, sections: List[ContractSection]) -> Dict[str, Any]:
        """Fallback extraction using regex patterns"""
        extractions = {
            "fees": {},
            "sla": {},
            "brand_usage": {},
            "data_sharing": {},
            "security": {},
            "termination": {},
            "penalties": {},
            "audit_rights": {},
            "marketing_obligations": {}
        }
        
        full_text = " ".join([section.text for section in sections])
        
        # Basic fee extraction
        fee_patterns = [
            (r'setup fee[:\s]+\$?([\d,]+)', 'setup_fees'),
            (r'monthly fee[:\s]+\$?([\d,]+)', 'monthly_fees'),
            (r'transaction fee[:\s]+(\d+\.?\d*%|\$[\d.]+)', 'transaction_fees'),
            (r'termination fee[:\s]+\$?([\d,]+)', 'termination_fees'),
            (r'net (\d+)', 'payment_terms')
        ]
        
        for pattern, field in fee_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                extractions["fees"][field] = match.group(1)
        
        # Basic SLA extraction
        sla_patterns = [
            (r'(\d+\.?\d*)%\s+uptime', 'uptime_guarantee'),
            (r'response.*?(\d+)\s+(hours?|days?)', 'response_times'),
            (r'resolution.*?(\d+)\s+(hours?|days?)', 'resolution_times')
        ]
        
        for pattern, field in sla_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                extractions["sla"][field] = f"{match.group(1)} {match.group(2) if len(match.groups()) > 1 else ''}"
        
        # Basic termination extraction
        termination_patterns = [
            (r'(\d+)\s+days?\s+notice', 'termination_notice_period'),
            (r'immediate termination', 'termination_for_cause')
        ]
        
        for pattern, field in termination_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                if field == 'termination_for_cause':
                    extractions["termination"][field] = True
                else:
                    extractions["termination"][field] = match.group(1)
        
        return extractions
    
    def delta_detect(self, curr_extractions: Dict[str, Any], prev_extractions: Dict[str, Any]) -> List[Delta]:
        """
        Detect changes between current and previous contract extractions
        
        Args:
            curr_extractions: Current contract extractions
            prev_extractions: Previous contract extractions
            
        Returns:
            List of Delta objects representing changes
        """
        deltas = []
        
        try:
            # Compare each category
            for category in curr_extractions:
                if category not in prev_extractions:
                    # New category added
                    deltas.append(Delta(
                        field=category,
                        from_value=None,
                        to_value=curr_extractions[category]
                    ))
                    continue
                
                curr_cat = curr_extractions[category]
                prev_cat = prev_extractions[category]
                
                if isinstance(curr_cat, dict) and isinstance(prev_cat, dict):
                    # Compare fields within category
                    for field in curr_cat:
                        curr_val = curr_cat[field]
                        prev_val = prev_cat.get(field)
                        
                        if curr_val != prev_val:
                            deltas.append(Delta(
                                field=f"{category}.{field}",
                                from_value=prev_val,
                                to_value=curr_val
                            ))
                    
                    # Check for removed fields
                    for field in prev_cat:
                        if field not in curr_cat:
                            deltas.append(Delta(
                                field=f"{category}.{field}",
                                from_value=prev_cat[field],
                                to_value=None
                            ))
                
                elif curr_cat != prev_cat:
                    # Direct value comparison
                    deltas.append(Delta(
                        field=category,
                        from_value=prev_cat,
                        to_value=curr_cat
                    ))
            
            # Check for removed categories
            for category in prev_extractions:
                if category not in curr_extractions:
                    deltas.append(Delta(
                        field=category,
                        from_value=prev_extractions[category],
                        to_value=None
                    ))
            
            logger.info(f"Detected {len(deltas)} deltas between contract versions")
            return deltas
            
        except Exception as e:
            logger.error(f"Error in delta detection: {e}")
            return []
    
    def _build_checklist(self, extractions: Dict[str, Any]) -> List[ChecklistItem]:
        """Build operational checklist based on contract extractions"""
        checklist = []
        
        try:
            # Data sharing obligations
            if extractions.get("data_sharing", {}).get("data_types_shared"):
                checklist.append(ChecklistItem(
                    task="Implement data-share audit procedures",
                    owner="Risk",
                    due_in_days=30
                ))
                
                checklist.append(ChecklistItem(
                    task="Set up data retention and deletion processes",
                    owner="Operations",
                    due_in_days=45
                ))
            
            # Security requirements
            security = extractions.get("security", {})
            if security.get("security_certifications"):
                checklist.append(ChecklistItem(
                    task="Obtain required security certifications",
                    owner="Risk",
                    due_in_days=90
                ))
            
            if security.get("audit_requirements"):
                checklist.append(ChecklistItem(
                    task="Schedule security audit",
                    owner="Risk",
                    due_in_days=60
                ))
            
            # SLA monitoring
            sla = extractions.get("sla", {})
            if sla.get("uptime_guarantee") or sla.get("response_times"):
                checklist.append(ChecklistItem(
                    task="Set up SLA monitoring and alerting",
                    owner="Operations",
                    due_in_days=14
                ))
            
            # Brand usage compliance
            brand = extractions.get("brand_usage", {})
            if brand.get("marketing_approval"):
                checklist.append(ChecklistItem(
                    task="Establish marketing approval workflow",
                    owner="Marketing",
                    due_in_days=21
                ))
            
            if brand.get("brand_guidelines"):
                checklist.append(ChecklistItem(
                    task="Review and implement brand guidelines",
                    owner="Marketing",
                    due_in_days=14
                ))
            
            # Fee structure implementation
            fees = extractions.get("fees", {})
            if fees.get("setup_fees") or fees.get("monthly_fees"):
                checklist.append(ChecklistItem(
                    task="Configure billing system for fee structure",
                    owner="Finance",
                    due_in_days=30
                ))
            
            # Marketing obligations
            marketing = extractions.get("marketing_obligations", {})
            if marketing.get("marketing_spend_commitments"):
                checklist.append(ChecklistItem(
                    task="Plan marketing budget allocation",
                    owner="Marketing",
                    due_in_days=30
                ))
            
            if marketing.get("event_participation"):
                checklist.append(ChecklistItem(
                    task="Schedule required event participation",
                    owner="Marketing",
                    due_in_days=60
                ))
            
            # Termination procedures
            termination = extractions.get("termination", {})
            if termination.get("data_return_obligations"):
                checklist.append(ChecklistItem(
                    task="Document data return procedures",
                    owner="Legal",
                    due_in_days=45
                ))
            
            # Audit rights preparation
            audit = extractions.get("audit_rights", {})
            if audit.get("audit_frequency"):
                checklist.append(ChecklistItem(
                    task="Prepare audit documentation and procedures",
                    owner="Risk",
                    due_in_days=30
                ))
            
            logger.info(f"Generated {len(checklist)} checklist items")
            return checklist
            
        except Exception as e:
            logger.error(f"Error building checklist: {e}")
            return []
    
    def _identify_risk_flags(self, extractions: Dict[str, Any]) -> List[RiskFlag]:
        """Identify risk flags based on contract terms"""
        risk_flags = []
        
        try:
            # High risk: Uncapped penalties
            penalties = extractions.get("penalties", {})
            if penalties and not penalties.get("penalty_caps"):
                if any(penalties.get(key) for key in ["late_payment_penalties", "performance_penalties", "data_breach_penalties"]):
                    risk_flags.append(RiskFlag(
                        severity="high",
                        note="Uncapped penalties detected - unlimited liability exposure"
                    ))
            
            # High risk: Very high SLA requirements
            sla = extractions.get("sla", {})
            uptime = sla.get("uptime_guarantee", "")
            if isinstance(uptime, str) and "99.9" in uptime:
                risk_flags.append(RiskFlag(
                    severity="high",
                    note="Very high uptime SLA (>99.9%) - difficult to achieve"
                ))
            
            # Medium risk: Short termination notice
            termination = extractions.get("termination", {})
            notice_period = termination.get("termination_notice_period")
            if notice_period:
                try:
                    days = int(re.search(r'\d+', str(notice_period)).group())
                    if days < 30:
                        risk_flags.append(RiskFlag(
                            severity="high",
                            note=f"Short termination notice period ({days} days)"
                        ))
                    elif days < 90:
                        risk_flags.append(RiskFlag(
                            severity="med",
                            note=f"Moderate termination notice period ({days} days)"
                        ))
                except:
                    pass
            
            # Medium risk: Significant fees
            fees = extractions.get("fees", {})
            setup_fee = fees.get("setup_fees", "")
            if isinstance(setup_fee, str):
                try:
                    amount = int(re.sub(r'[^\d]', '', setup_fee))
                    if amount > 10000:
                        risk_flags.append(RiskFlag(
                            severity="med",
                            note=f"High setup fee (${amount:,})"
                        ))
                except:
                    pass
            
            # Medium risk: Broad audit rights
            audit = extractions.get("audit_rights", {})
            if audit.get("audit_frequency") and "annual" not in str(audit.get("audit_frequency", "")).lower():
                risk_flags.append(RiskFlag(
                    severity="med",
                    note="Frequent audit rights - operational burden"
                ))
            
            # High risk: Exclusive rights
            brand = extractions.get("brand_usage", {})
            if brand.get("exclusivity_rights"):
                risk_flags.append(RiskFlag(
                    severity="high",
                    note="Exclusive rights granted - limits future partnerships"
                ))
            
            # Medium risk: Complex data sharing
            data = extractions.get("data_sharing", {})
            if data.get("third_party_sharing") and data.get("compliance_standards"):
                risk_flags.append(RiskFlag(
                    severity="med",
                    note="Complex data sharing with compliance requirements"
                ))
            
            # Low risk: Standard terms
            if not risk_flags:
                risk_flags.append(RiskFlag(
                    severity="low",
                    note="Standard contract terms with typical risk profile"
                ))
            
            logger.info(f"Identified {len(risk_flags)} risk flags")
            return risk_flags
            
        except Exception as e:
            logger.error(f"Error identifying risk flags: {e}")
            return [RiskFlag(severity="med", note="Error in risk assessment")]
    
    def _get_contract_citations(self, extractions: Dict[str, Any]) -> List[Citation]:
        """Get contract policy citations from knowledge base with Tavily fallback"""
        citations = []
        
        if not all([self.retriever, self.embedder]):
            logger.warning("RAG components not available for contract citations")
            return citations
        
        try:
            # Create queries for key contract terms
            queries = []
            
            # Add queries based on extracted terms
            if extractions.get("audit_rights"):
                queries.append("audit rights contract terms")
            if extractions.get("sla"):
                queries.append("SLA days service level agreement")
            if extractions.get("data_sharing"):
                queries.append("data sharing reconciliation")
            
            # Default query if no specific terms
            if not queries:
                queries = ["contract terms merchant agreement"]
            
            # Retrieve relevant documents for each query
            for query in queries[:3]:  # Limit to 3 queries
                results = retrieve(self.retriever, self.embedder, query, k=2)
                
                for result in results:
                    citations.append(Citation(
                        source=result.get("filename", "Contract Policy"),
                        snippet=result.get("snippet", "")[:300] + "..."
                    ))
            
            # Tavily fallback if insufficient citations
            if len(citations) < 1:
                logger.info("Insufficient local contract citations, searching web...")
                try:
                    web_docs = web_search_into_docstore(
                        self.docstore,
                        self.embedder,
                        "contract audit rights SLA days reconciliation terms",
                        max_results=2
                    )
                    
                    # Re-retrieve after adding web content
                    if web_docs:
                        results = retrieve(self.retriever, self.embedder, 
                                         "contract terms audit SLA", k=2)
                        
                        for result in results:
                            citations.append(Citation(
                                source=result.get("filename", "Web Search"),
                                snippet=result.get("snippet", "")[:300] + "..."
                            ))
                            
                except Exception as e:
                    logger.warning(f"Web search for contract terms failed: {e}")
            
        except Exception as e:
            logger.error(f"Error retrieving contract citations: {e}")
        
        return citations
    
    def _generate_summary(self, extractions: Dict[str, Any], risk_flags: List[RiskFlag], 
                         deltas: List[Delta]) -> str:
        """Generate executive summary of contract analysis"""
        try:
            # Count non-empty categories
            populated_categories = sum(1 for cat in extractions.values() if cat)
            
            # Risk summary
            high_risks = len([r for r in risk_flags if r.severity == "high"])
            med_risks = len([r for r in risk_flags if r.severity == "med"])
            
            # Key terms summary
            key_terms = []
            
            fees = extractions.get("fees", {})
            if fees.get("setup_fees"):
                key_terms.append(f"Setup fee: {fees['setup_fees']}")
            if fees.get("monthly_fees"):
                key_terms.append(f"Monthly fee: {fees['monthly_fees']}")
            
            sla = extractions.get("sla", {})
            if sla.get("uptime_guarantee"):
                key_terms.append(f"Uptime SLA: {sla['uptime_guarantee']}")
            
            termination = extractions.get("termination", {})
            if termination.get("termination_notice_period"):
                key_terms.append(f"Termination notice: {termination['termination_notice_period']}")
            
            # Build summary
            summary_parts = [
                f"Contract analysis completed with {populated_categories} categories extracted."
            ]
            
            if key_terms:
                summary_parts.append(f"Key terms: {', '.join(key_terms[:3])}.")
            
            if high_risks > 0:
                summary_parts.append(f"⚠️ {high_risks} high-risk items require immediate attention.")
            elif med_risks > 0:
                summary_parts.append(f"📋 {med_risks} medium-risk items identified for review.")
            else:
                summary_parts.append("✅ Standard risk profile with typical contract terms.")
            
            if deltas:
                summary_parts.append(f"🔄 {len(deltas)} changes detected from previous version.")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Contract analysis completed with basic extraction."

# Test cases for contract scenarios
def test_contract_intelligence():
    """Test ContractIntelligence with golden-path scenarios"""
    print("🧪 Testing ContractIntelligence Golden Paths")
    print("=" * 50)
    
    intelligence = ContractIntelligence()
    
    test_cases = [
        {
            "name": "MD contract with audit rights → checklist includes audit procedures",
            "file_path": "app/contracts/merchant_agreement.md",
            "expected_checklist_task": "audit",
            "expected_risk_flags": True
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        try:
            print(f"{i}. {case['name']}")
            
            result = intelligence.process_contract(case["file_path"])
            
            # Validate response structure
            valid_structure = (
                hasattr(result, 'summary') and
                hasattr(result, 'extractions') and
                hasattr(result, 'checklist') and
                hasattr(result, 'risk_flags') and
                hasattr(result, 'deltas') and
                hasattr(result, 'citations')
            )
            
            # Check for expected checklist task
            checklist_ok = any(
                case["expected_checklist_task"].lower() in item.task.lower()
                for item in result.checklist
            )
            
            # Check for risk flags
            risk_flags_ok = len(result.risk_flags) > 0 if case["expected_risk_flags"] else True
            
            success = valid_structure and checklist_ok and risk_flags_ok
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   File: {case['file_path']}")
            print(f"   Extractions: {len(result.extractions)} categories")
            print(f"   Checklist: {len(result.checklist)} items")
            print(f"   Risk flags: {len(result.risk_flags)}")
            print(f"   Status: {status}")
            
            if success:
                passed += 1
            else:
                if not valid_structure:
                    print(f"   ❌ Invalid response structure")
                if not checklist_ok:
                    print(f"   ❌ Expected checklist task '{case['expected_checklist_task']}' not found")
                if not risk_flags_ok:
                    print(f"   ❌ Expected risk flags not found")
            
            print()
            
        except Exception as e:
            print(f"   ❌ FAIL - Exception: {e}")
            print()
    
    # Test delta detection with v1→v2 penalty clause addition
    print("Testing Delta Detection: v1→v2 penalty clause addition")
    print("-" * 50)
    
    try:
        # Create mock extractions for v1 and v2
        v1_extractions = {
            "fees": {"setup_fees": "$5000", "monthly_fees": "$1000"},
            "sla": {"uptime_guarantee": "99.5%"},
            "penalties": {}
        }
        
        v2_extractions = {
            "fees": {"setup_fees": "$5000", "monthly_fees": "$1000"},
            "sla": {"uptime_guarantee": "99.5%"},
            "penalties": {"late_payment_penalties": "5% per month", "penalty_caps": None}
        }
        
        deltas = intelligence.delta_detect(v2_extractions, v1_extractions)
        
        # Check if penalty clause delta is detected
        penalty_delta_found = any(
            "penalties" in delta.field and delta.to_value
            for delta in deltas
        )
        
        if penalty_delta_found:
            print("✅ PASS - Penalty clause delta detected")
            print(f"   Deltas found: {len(deltas)}")
            for delta in deltas:
                print(f"   - {delta.field}: {delta.from_value} → {delta.to_value}")
        else:
            print("❌ FAIL - Penalty clause delta not detected")
    
    except Exception as e:
        print(f"❌ FAIL - Delta detection test failed: {e}")
    
    print()
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    return passed == total


def main():
    """Main function for testing and demonstration"""
    print("🔍 Contract Intelligence Agent")
    print("=" * 50)
    print("Analyzing merchant agreements and extracting key terms...")
    print()
    
    # Run tests
    success = test_contract_intelligence()
    
    if success:
        print("🎉 All tests passed! Contract Intelligence is working correctly.")
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
    
    # Example usage
    print("\n💡 Example Usage:")
    print("-" * 20)
    print("from app.agents.contracts import ContractIntelligence")
    print("intelligence = ContractIntelligence(docstore, embedder, retriever)")
    print("result = intelligence.process_contract('path/to/contract.md')")
    print("print(result.summary)")
    print("for item in result.checklist:")
    print("    print(f'- {item.task} ({item.owner}, {item.due_in_days} days)')")


if __name__ == "__main__":
    main()
