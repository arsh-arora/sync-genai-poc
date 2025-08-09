# Contract Ontology - Key Terms Extraction Guide

## Overview

This document defines the standard ontology for extracting key terms from merchant agreements, partnership contracts, and vendor agreements. It serves as a reference for automated contract analysis and obligation mapping.

## Core Extraction Categories

### 1. Fees Structure

**Definition**: All monetary obligations, charges, and payment terms

**Key Fields to Extract:**
- `setup_fees`: One-time implementation or onboarding costs
- `monthly_fees`: Recurring monthly charges
- `transaction_fees`: Per-transaction charges (fixed or percentage)
- `volume_discounts`: Tiered pricing based on transaction volume
- `penalty_fees`: Charges for non-compliance or violations
- `termination_fees`: Costs associated with contract termination
- `payment_terms`: When payments are due (net 30, etc.)
- `fee_escalation`: Annual increases or adjustment mechanisms

**Example Patterns:**
- "Setup fee of $5,000 due within 30 days"
- "Monthly service fee: $2,500"
- "Transaction fee: 2.5% + $0.30 per transaction"
- "Volume discount: >$1M monthly = 2.0% rate"

### 2. Service Level Agreements (SLAs)

**Definition**: Performance commitments and operational requirements

**Key Fields to Extract:**
- `uptime_guarantee`: System availability commitments (99.9%)
- `response_times`: Support response requirements
- `resolution_times`: Issue resolution commitments
- `processing_times`: Transaction or request processing speeds
- `reporting_frequency`: How often reports are provided
- `maintenance_windows`: Scheduled downtime allowances
- `escalation_procedures`: How issues are escalated
- `performance_penalties`: Consequences for SLA breaches

**Example Patterns:**
- "99.5% uptime guarantee"
- "Support response within 4 business hours"
- "Critical issues resolved within 24 hours"
- "Transaction processing within 2 seconds"

### 3. Brand Usage Rights

**Definition**: Intellectual property usage, marketing rights, and brand guidelines

**Key Fields to Extract:**
- `logo_usage_rights`: Permission to use company logos
- `trademark_usage`: Rights to use trademarks and service marks
- `co_branding_requirements`: Joint branding obligations
- `marketing_approval`: Requirement for marketing material approval
- `brand_guidelines`: Standards for brand representation
- `exclusivity_rights`: Exclusive usage or territory rights
- `attribution_requirements`: How to credit the brand
- `usage_restrictions`: Limitations on brand usage

**Example Patterns:**
- "Merchant may use Company logo in approved marketing materials"
- "All marketing materials require prior written approval"
- "Exclusive rights within the healthcare vertical"
- "Logo usage must comply with brand guidelines v2.1"

### 4. Data Sharing & Privacy

**Definition**: Data handling, sharing, and privacy obligations

**Key Fields to Extract:**
- `data_types_shared`: Categories of data being exchanged
- `data_retention_period`: How long data is kept
- `data_security_requirements`: Security standards and controls
- `third_party_sharing`: Rights to share data with third parties
- `customer_consent_requirements`: Consent obligations
- `data_deletion_rights`: Right to delete or return data
- `compliance_standards`: GDPR, CCPA, PCI-DSS requirements
- `breach_notification`: Data breach notification procedures

**Example Patterns:**
- "Customer transaction data retained for 7 years"
- "PCI-DSS Level 1 compliance required"
- "Data may be shared with approved payment processors"
- "Breach notification within 72 hours"

### 5. Security Requirements

**Definition**: Cybersecurity, physical security, and operational security obligations

**Key Fields to Extract:**
- `security_certifications`: Required certifications (SOC 2, ISO 27001)
- `encryption_requirements`: Data encryption standards
- `access_controls`: User access and authentication requirements
- `vulnerability_management`: Security testing and patching
- `incident_response`: Security incident procedures
- `audit_requirements`: Security audit obligations
- `employee_screening`: Background check requirements
- `physical_security`: Facility security requirements

**Example Patterns:**
- "SOC 2 Type II certification required annually"
- "AES-256 encryption for data at rest"
- "Multi-factor authentication for all admin access"
- "Quarterly vulnerability assessments required"

### 6. Termination Clauses

**Definition**: Contract termination conditions, procedures, and consequences

**Key Fields to Extract:**
- `termination_notice_period`: Required notice before termination
- `termination_for_cause`: Conditions allowing immediate termination
- `termination_without_cause`: General termination rights
- `data_return_obligations`: Requirements to return or delete data
- `transition_assistance`: Support during transition period
- `post_termination_restrictions`: Ongoing obligations after termination
- `survival_clauses`: Terms that survive contract termination
- `termination_fees`: Costs associated with early termination

**Example Patterns:**
- "90 days written notice required for termination"
- "Immediate termination for material breach"
- "All customer data returned within 30 days"
- "Early termination fee: 6 months of monthly fees"

### 7. Penalties & Compliance

**Definition**: Consequences for non-compliance and violation remedies

**Key Fields to Extract:**
- `late_payment_penalties`: Charges for overdue payments
- `performance_penalties`: Consequences for SLA breaches
- `compliance_violations`: Penalties for regulatory non-compliance
- `data_breach_penalties`: Consequences for security incidents
- `liquidated_damages`: Pre-agreed damage amounts
- `penalty_caps`: Maximum penalty amounts
- `cure_periods`: Time allowed to remedy violations
- `escalating_penalties`: Increasing penalties for repeated violations

**Example Patterns:**
- "Late payment penalty: 1.5% per month"
- "SLA breach penalty: $1,000 per hour of downtime"
- "Data breach penalty: up to $100,000"
- "30-day cure period for material breaches"

### 8. Audit Rights

**Definition**: Rights to inspect, audit, and verify compliance

**Key Fields to Extract:**
- `audit_frequency`: How often audits may be conducted
- `audit_scope`: What can be audited
- `audit_notice_period`: Advance notice required for audits
- `audit_costs`: Who pays for audit expenses
- `audit_access_rights`: What records and systems can be accessed
- `third_party_audits`: Rights to use external auditors
- `audit_remediation`: Requirements to address audit findings
- `audit_reporting`: Audit report sharing requirements

**Example Patterns:**
- "Annual compliance audits permitted with 30 days notice"
- "Audit costs borne by requesting party unless violations found"
- "Access to all relevant records and systems"
- "Third-party auditors subject to confidentiality agreements"

### 9. Marketing Obligations

**Definition**: Marketing, promotion, and business development commitments

**Key Fields to Extract:**
- `marketing_spend_commitments`: Required marketing investments
- `promotional_requirements`: Mandatory promotional activities
- `event_participation`: Trade show or conference obligations
- `content_creation`: Requirements to create marketing content
- `lead_generation`: Lead generation and sharing obligations
- `co_marketing_activities`: Joint marketing initiatives
- `marketing_performance_metrics`: Success measurement criteria
- `marketing_approval_process`: Review and approval procedures

**Example Patterns:**
- "Minimum $50,000 annual marketing spend required"
- "Participation in 2 major industry conferences annually"
- "Monthly case study or success story required"
- "Joint webinar quarterly"

## Extraction Guidelines

### Document Structure Recognition

**Section Identification Patterns:**
- Headers: "FEES", "SERVICE LEVELS", "TERMINATION", "DATA PRIVACY"
- Numbered sections: "3. Payment Terms", "7. Intellectual Property"
- Article references: "Article IV - Confidentiality"
- Appendix references: "Schedule A - Fee Structure"

### Risk Flag Triggers

**High Risk Indicators:**
- Uncapped penalties or unlimited liability
- SLA requirements >99.9% uptime
- Termination notice <30 days
- Exclusive rights or exclusivity clauses
- Personal guarantees or joint liability

**Medium Risk Indicators:**
- SLA requirements 95-99% uptime
- Termination notice 30-90 days
- Significant setup or termination fees
- Complex fee structures with multiple tiers
- Broad audit rights or frequent audits

**Low Risk Indicators:**
- Standard industry terms
- Reasonable notice periods (90+ days)
- Capped penalties and liability
- Clear termination procedures
- Standard data protection requirements

### Owner Assignment Heuristics

**Legal Team:**
- Contract amendments and modifications
- Intellectual property and trademark issues
- Liability and indemnification matters
- Regulatory compliance requirements

**Risk Team:**
- Data privacy and security requirements
- Audit rights and compliance monitoring
- Risk assessment and mitigation
- Insurance and liability coverage

**Finance Team:**
- Fee structures and payment terms
- Revenue recognition implications
- Cost analysis and budgeting
- Financial reporting requirements

**Operations Team:**
- SLA monitoring and performance
- System integration and technical requirements
- Day-to-day contract administration
- Vendor relationship management

**Marketing Team:**
- Brand usage and co-marketing
- Marketing obligations and commitments
- Promotional requirements
- Content creation and approval

## Common Contract Patterns

### Payment Terms Variations
- "Net 30" = Payment due 30 days after invoice
- "2/10 Net 30" = 2% discount if paid within 10 days, otherwise net 30
- "Monthly in advance" = Payment due at start of service period
- "Quarterly in arrears" = Payment due at end of quarter

### SLA Measurement Methods
- "Calendar time" = 24/7 including weekends and holidays
- "Business hours" = Typically 8 AM - 6 PM, Monday-Friday
- "Scheduled maintenance excluded" = Planned downtime doesn't count
- "Force majeure excluded" = Natural disasters and external factors excluded

### Termination Notice Calculations
- "Business days" = Excludes weekends and holidays
- "Calendar days" = Includes all days
- "Written notice" = Formal documentation required
- "Email acceptable" = Electronic delivery permitted

This ontology serves as the foundation for automated contract analysis, ensuring consistent extraction of key terms and proper risk assessment across all merchant agreements and partnership contracts.
