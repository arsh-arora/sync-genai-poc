"""
Developer & Partner API Copilot with Partner Enablement
Mini doc index, stepwise checklists, and test flows for real partner enablement
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal

from pydantic import BaseModel, Field

from app.llm.gemini import chat

logger = logging.getLogger(__name__)

# Pydantic models for partner enablement
class ChecklistStep(BaseModel):
    step: str
    description: str
    code_sample: Optional[str] = None
    test_payload: Optional[Dict[str, Any]] = None

class TestFlow(BaseModel):
    name: str
    description: str
    payload: Dict[str, Any]
    expected_response: Dict[str, Any]

class DocSnippet(BaseModel):
    source: str  # "doc#heading" format
    content: str

class DevCopilotResponse(BaseModel):
    response: str  # 3-5 step guide + what to try now
    metadata: Dict[str, Any]  # ui_cards with checklist/samples, sources

class DevCopilot:
    """
    Developer & Partner API Copilot with Partner Enablement
    Mini doc index, stepwise checklists, and test flows for real partner enablement
    """
    
    def __init__(self, docstore=None, embedder=None, retriever=None, rules_loader=None):
        """Initialize DevCopilot with RAG components and dev docs index"""
        self.docstore = docstore
        self.embedder = embedder
        self.retriever = retriever
        self.rules_loader = rules_loader
        
        # Load dev docs index from fixtures
        self.dev_docs_path = Path("synchrony-demo-rules-repo/fixtures/dev_docs")
        self.doc_index = self._load_doc_index()
        
        # Common integration patterns
        self.common_patterns = {
            "embed prequal": self._get_prequal_checklist,
            "receive dispute webhooks": self._get_webhook_checklist,
            "sandbox run": self._get_sandbox_checklist
        }
    
    def process_request(
        self, 
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> DevCopilotResponse:
        """
        Main processing pipeline for partner enablement with stepwise guides
        
        Args:
            query: Partner's question or integration ask
            context: Optional context (partner_id, environment, etc.)
            
        Returns:
            DevCopilotResponse with 3-5 step guide and what to try now
        """
        try:
            logger.info(f"Processing DevCopilot request: {query}")
            
            # Step 1: Detect common integration patterns
            pattern = self._detect_pattern(query)
            
            if pattern:
                # Step 2: Generate stepwise checklist
                checklist = self.common_patterns[pattern]()
                
                # Step 3: Get relevant doc snippets
                doc_snippets = self._search_docs(query)
                
                # Step 4: Generate test flows
                test_flows = self._generate_test_flows(pattern)
                
                # Step 5: Build response
                response = self._build_guide_response(pattern, checklist, doc_snippets, test_flows)
                
                return response
            else:
                # General query - search docs and provide guidance
                return self._handle_general_query(query, context)
            
        except Exception as e:
            logger.error(f"Error processing DevCopilot request: {e}")
            return self._error_response(f"Error: {str(e)}")
    
    def _load_doc_index(self) -> Dict[str, Dict[str, str]]:
        """Load mini doc index from fixtures/dev_docs/*.md with anchored snippets"""
        doc_index = {}
        
        try:
            if not self.dev_docs_path.exists():
                logger.warning(f"Dev docs path not found: {self.dev_docs_path}")
                return doc_index
            
            for doc_file in self.dev_docs_path.glob("*.md"):
                doc_name = doc_file.stem
                content = doc_file.read_text(encoding='utf-8')
                
                # Parse headings for anchored snippets
                sections = {}
                current_heading = "main"
                current_content = []
                
                for line in content.split('\n'):
                    if line.startswith('#'):
                        # Save previous section
                        if current_content:
                            sections[current_heading] = '\n'.join(current_content).strip()
                        
                        # Start new section
                        current_heading = line.strip('#').strip().lower().replace(' ', '_')
                        current_content = []
                    else:
                        current_content.append(line)
                
                # Save final section
                if current_content:
                    sections[current_heading] = '\n'.join(current_content).strip()
                
                doc_index[doc_name] = sections
                logger.info(f"Loaded {len(sections)} sections from {doc_name}.md")
            
            return doc_index
            
        except Exception as e:
            logger.error(f"Error loading doc index: {e}")
            return {}
    
    def _detect_pattern(self, query: str) -> Optional[str]:
        """Detect common integration patterns from query"""
        query_lower = query.lower()
        
        # Pattern matching with keywords
        if any(word in query_lower for word in ['embed', 'widget', 'prequal', 'qualification']):
            return "embed prequal"
        elif any(word in query_lower for word in ['webhook', 'dispute', 'status', 'callback']):
            return "receive dispute webhooks"
        elif any(word in query_lower for word in ['sandbox', 'test', 'run', 'try']):
            return "sandbox run"
        
        return None
    
    def _get_prequal_checklist(self) -> List[ChecklistStep]:
        """Generate prequalification widget embed checklist"""
        return [
            ChecklistStep(
                step="1. Get Sandbox Keys",
                description="Request sandbox API keys from partner portal",
                code_sample="""
# Set your sandbox credentials
API_KEY = "sk_sandbox_..."
PARTNER_ID = "partner_123"
ENVIRONMENT = "sandbox"
                """.strip()
            ),
            ChecklistStep(
                step="2. Add Widget Script",
                description="Include prequalification widget script in your page",
                code_sample="""
<script src="https://widgets.sandbox.syncpay.com/prequal.js"></script>
<div id="prequal-widget"></div>
                """.strip(),
                test_payload={
                    "amount": 2500.00,
                    "partner_id": "partner_123",
                    "customer_context": {"zip": "10001"}
                }
            ),
            ChecklistStep(
                step="3. Initialize Widget",
                description="Initialize with amount and partner context",
                code_sample="""
SyncPay.Prequal.init({
    containerId: 'prequal-widget',
    amount: 2500.00,
    partnerId: 'partner_123',
    environment: 'sandbox',
    onStart: (data) => console.log('Started:', data),
    onComplete: (result) => handlePrequalResult(result)
});
                """.strip()
            ),
            ChecklistStep(
                step="4. Handle Callbacks",
                description="Store prequalification outcomes and display offers",
                code_sample="""
function handlePrequalResult(result) {
    if (result.approved) {
        // Display financing options
        showFinancingOffers(result.offers);
    } else {
        // Handle decline gracefully
        showAlternativeOptions();
    }
}
                """.strip(),
                test_payload={
                    "prequal_id": "pq_123",
                    "approved": True,
                    "offers": [{"term_months": 12, "apr": 0.0}]
                }
            )
        ]
    
    def _get_webhook_checklist(self) -> List[ChecklistStep]:
        """Generate dispute webhook handling checklist"""
        return [
            ChecklistStep(
                step="1. Set Webhook Endpoint",
                description="Configure your endpoint to receive dispute status updates",
                code_sample="""
# Example Flask endpoint
@app.route('/webhooks/dispute-status', methods=['POST'])
def handle_dispute_webhook():
    payload = request.get_json()
    
    # Verify webhook signature
    if not verify_webhook_signature(request.headers, payload):
        return 'Unauthorized', 401
    
    process_dispute_update(payload)
    return 'OK', 200
                """.strip()
            ),
            ChecklistStep(
                step="2. Handle Status Changes",
                description="Process dispute status transitions: intake→compiled→ready_to_submit→submitted→under_review",
                code_sample="""
def process_dispute_update(payload):
    dispute_id = payload['dispute_id']
    status = payload['status']
    
    # Update dispute record
    dispute = Dispute.get(dispute_id)
    dispute.status = status
    dispute.save()
    
    # Trigger appropriate actions
    if status == 'ready_to_submit':
        notify_customer(dispute, 'ready_for_submission')
    elif status == 'under_review':
        notify_customer(dispute, 'being_reviewed')
                """.strip(),
                test_payload={
                    "dispute_id": "disp_123",
                    "status": "ready_to_submit",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "merchant": "Urban Living Co",
                    "amount": 350.00
                }
            ),
            ChecklistStep(
                step="3. Implement Retry Logic",
                description="Handle failed webhooks with exponential backoff",
                code_sample="""
# Webhook retry configuration
WEBHOOK_RETRY_ATTEMPTS = 3
WEBHOOK_RETRY_DELAY = [1, 5, 15]  # seconds

def send_webhook_with_retry(url, payload):
    for attempt in range(WEBHOOK_RETRY_ATTEMPTS):
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        
        time.sleep(WEBHOOK_RETRY_DELAY[attempt])
    
    # Log failed webhook for manual processing
    log_failed_webhook(url, payload)
    return False
                """.strip()
            )
        ]
    
    def _get_sandbox_checklist(self) -> List[ChecklistStep]:
        """Generate sandbox testing checklist"""
        return [
            ChecklistStep(
                step="1. Switch to Sandbox Environment",
                description="Configure your app to use sandbox URLs and keys",
                code_sample="""
# Sandbox configuration
BASE_URL = "https://api.sandbox.syncpay.com"
API_KEY = "sk_sandbox_test_..."
WIDGET_URL = "https://widgets.sandbox.syncpay.com"

# Enable debug mode
DEBUG_MODE = True
LOG_LEVEL = "DEBUG"
                """.strip()
            ),
            ChecklistStep(
                step="2. Use Test Credit Profiles",
                description="Test with deterministic credit profiles for predictable results",
                test_payload={
                    "test_profiles": [
                        {"ssn_last4": "1111", "result": "approved", "max_amount": 5000},
                        {"ssn_last4": "2222", "result": "declined", "reason": "insufficient_credit"},
                        {"ssn_last4": "3333", "result": "manual_review", "timeline": "1-2 business days"}
                    ]
                }
            ),
            ChecklistStep(
                step="3. Test Key Flows",
                description="Run through critical integration paths",
                code_sample="""
# Test prequalification flow
test_prequal = {
    "amount": 1500.00,
    "customer": {"ssn_last4": "1111"},  # Always approves
    "partner_id": "partner_sandbox"
}

# Test dispute creation
test_dispute = {
    "merchant": "Test Merchant",
    "amount": 299.99,
    "reason": "duplicate_charge",
    "evidence": ["receipt.pdf"]
}

# Run automated tests
run_integration_tests()
                """.strip()
            ),
            ChecklistStep(
                step="4. Validate Webhook Delivery",
                description="Ensure webhooks are received correctly in sandbox",
                code_sample="""
# Test webhook endpoint using ngrok or similar
# 1. Start your local server
# 2. Expose with ngrok: ngrok http 3000
# 3. Configure webhook URL in sandbox dashboard
# 4. Trigger test events

# Webhook test payload
webhook_test = {
    "dispute_id": "disp_sandbox_123",
    "status": "compiled",
    "test_mode": True
}
                """.strip(),
                test_payload={
                    "webhook_url": "https://your-ngrok-url.ngrok.io/webhooks/test",
                    "test_events": ["dispute.status_changed", "prequal.completed"]
                }
            )
        ]
    
    def _search_docs(self, query: str) -> List[DocSnippet]:
        """Search doc index for relevant snippets"""
        query_lower = query.lower()
        relevant_snippets = []
        
        try:
            # Search through doc index
            for doc_name, sections in self.doc_index.items():
                for section_name, content in sections.items():
                    # Check if query terms appear in content
                    content_lower = content.lower()
                    
                    # Simple relevance scoring
                    score = 0
                    query_words = query_lower.split()
                    
                    for word in query_words:
                        if len(word) > 2:  # Skip short words
                            if word in content_lower:
                                score += content_lower.count(word)
                    
                    if score > 0:
                        relevant_snippets.append(DocSnippet(
                            source=f"{doc_name}.md#{section_name}",
                            content=content[:300] + "..." if len(content) > 300 else content
                        ))
            
            # Sort by relevance and return top 3
            return relevant_snippets[:3]
            
        except Exception as e:
            logger.error(f"Error searching docs: {e}")
            return []
    
    def _generate_test_flows(self, pattern: str) -> List[TestFlow]:
        """Generate deterministic test flows for pattern"""
        if pattern == "embed prequal":
            return [
                TestFlow(
                    name="Approved Prequalification",
                    description="Test with auto-approval profile",
                    payload={
                        "amount": 2500.00,
                        "customer": {"ssn_last4": "1111", "zip": "10001"},
                        "partner_id": "partner_sandbox"
                    },
                    expected_response={
                        "approved": True,
                        "prequal_id": "pq_approved_123",
                        "offers": [
                            {"term_months": 12, "apr": 0.0, "monthly_payment": 208.33},
                            {"term_months": 24, "apr": 9.99, "monthly_payment": 115.38}
                        ]
                    }
                ),
                TestFlow(
                    name="Declined Prequalification",
                    description="Test decline handling",
                    payload={
                        "amount": 2500.00,
                        "customer": {"ssn_last4": "2222", "zip": "10001"},
                        "partner_id": "partner_sandbox"
                    },
                    expected_response={
                        "approved": False,
                        "prequal_id": "pq_declined_123",
                        "reason": "insufficient_credit_history"
                    }
                )
            ]
        
        elif pattern == "receive dispute webhooks":
            return [
                TestFlow(
                    name="Dispute Status Update",
                    description="Test webhook delivery for status change",
                    payload={
                        "dispute_id": "disp_test_123",
                        "status": "ready_to_submit",
                        "previous_status": "compiled",
                        "timestamp": "2024-01-01T12:00:00Z",
                        "test_mode": True
                    },
                    expected_response={
                        "received": True,
                        "processed": True,
                        "next_action": "notify_customer"
                    }
                )
            ]
        
        elif pattern == "sandbox run":
            return [
                TestFlow(
                    name="Full Integration Test",
                    description="End-to-end sandbox test",
                    payload={
                        "test_scenario": "happy_path",
                        "prequal_amount": 1500.00,
                        "customer_profile": "approved",
                        "dispute_test": True
                    },
                    expected_response={
                        "prequal_passed": True,
                        "webhook_delivered": True,
                        "integration_score": "100%"
                    }
                )
            ]
        
        return []
    
    def _build_guide_response(
        self, 
        pattern: str, 
        checklist: List[ChecklistStep], 
        doc_snippets: List[DocSnippet],
        test_flows: List[TestFlow]
    ) -> DevCopilotResponse:
        """Build step-by-step guide response"""
        
        # Generate response text (3-5 step guide)
        guide_steps = []
        for i, step in enumerate(checklist, 1):
            guide_steps.append(f"{i}. {step.step}: {step.description}")
        
        response_text = f"""**{pattern.title()} Integration Guide:**

{chr(10).join(guide_steps)}

**What to try now:**
- Use the code samples in the checklist below
- Test with the provided payloads in sandbox mode
- Check the documentation snippets for additional details
- Validate webhook delivery if applicable"""
        
        # Build metadata for UI cards
        ui_cards = []
        
        # Add checklist cards
        for step in checklist:
            card = {
                "type": "checklist",
                "title": step.step,
                "content": step.description
            }
            
            if step.code_sample:
                card["code_sample"] = step.code_sample
            
            if step.test_payload:
                card["test_payload"] = step.test_payload
            
            ui_cards.append(card)
        
        # Add test flow cards
        for flow in test_flows:
            ui_cards.append({
                "type": "test_flow",
                "title": flow.name,
                "description": flow.description,
                "payload": flow.payload,
                "expected_response": flow.expected_response
            })
        
        # Build sources from doc snippets
        sources = [snippet.source for snippet in doc_snippets]
        
        return DevCopilotResponse(
            response=response_text,
            metadata={
                "ui_cards": ui_cards,
                "sources": sources,
                "pattern": pattern,
                "test_flows": [flow.model_dump() for flow in test_flows],
                "doc_snippets": [snippet.model_dump() for snippet in doc_snippets]
            }
        )
    
    def _handle_general_query(self, query: str, context: Optional[Dict[str, Any]]) -> DevCopilotResponse:
        """Handle general queries not matching specific patterns"""
        
        # Search docs for relevant content
        doc_snippets = self._search_docs(query)
        
        # Use LLM to generate response if needed
        try:
            system_prompt = """You are a developer support specialist. Help partners integrate with our APIs.
            
            Provide a concise 3-5 step guide based on the documentation snippets.
            Focus on actionable steps and what to try next."""
            
            context_text = ""
            if doc_snippets:
                context_text = "\n\nRelevant documentation:\n"
                for snippet in doc_snippets:
                    context_text += f"- {snippet.source}: {snippet.content}\n"
            
            user_message = f"Partner question: {query}{context_text}"
            messages = [{"role": "user", "content": user_message}]
            
            llm_response = chat(messages, system=system_prompt)
            
            # Build basic response
            return DevCopilotResponse(
                response=llm_response,
                metadata={
                    "ui_cards": [
                        {
                            "type": "general_help",
                            "title": "General Guidance",
                            "content": llm_response
                        }
                    ],
                    "sources": [snippet.source for snippet in doc_snippets],
                    "doc_snippets": [snippet.model_dump() for snippet in doc_snippets]
                }
            )
            
        except Exception as e:
            logger.error(f"Error with LLM response: {e}")
            
            # Fallback to doc-only response
            if doc_snippets:
                response_text = "Based on the documentation:\n\n"
                for snippet in doc_snippets:
                    response_text += f"**{snippet.source}:**\n{snippet.content}\n\n"
            else:
                response_text = "I don't have specific documentation for that question. Please check the partner portal or contact support."
            
            return DevCopilotResponse(
                response=response_text,
                metadata={
                    "ui_cards": [],
                    "sources": [snippet.source for snippet in doc_snippets],
                    "doc_snippets": [snippet.model_dump() for snippet in doc_snippets]
                }
            )
    
    def _error_response(self, message: str) -> DevCopilotResponse:
        """Create error response"""
        return DevCopilotResponse(
            response=f"**Error:** {message}\n\nPlease try rephrasing your question or contact support.",
            metadata={
                "ui_cards": [
                    {
                        "type": "error",
                        "title": "Error",
                        "content": message
                    }
                ],
                "sources": [],
                "error": message
            }
        )

# Test cases for DevCopilot partner enablement scenarios
def test_devcopilot():
    """Test DevCopilot with partner enablement scenarios"""
    print("🧪 Testing DevCopilot Partner Enablement")
    print("=" * 50)
    
    copilot = DevCopilot()
    
    test_cases = [
        {
            "name": "Embed prequalification widget",
            "query": "How do I embed the prequalification widget on my checkout page?",
            "expected_pattern": "embed prequal",
            "expected_checklist_steps": 4
        },
        {
            "name": "Receive dispute webhooks",
            "query": "I need to handle dispute status webhooks in my app",
            "expected_pattern": "receive dispute webhooks",
            "expected_checklist_steps": 3
        },
        {
            "name": "Sandbox testing",
            "query": "How can I test my integration in the sandbox?",
            "expected_pattern": "sandbox run",
            "expected_checklist_steps": 4
        },
        {
            "name": "General query",
            "query": "What's the API rate limit?",
            "expected_pattern": None,
            "expected_checklist_steps": 0
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        try:
            print(f"{i}. {case['name']}")
            
            result = copilot.process_request(case["query"])
            
            # Validate response structure
            valid_structure = (
                hasattr(result, 'response') and
                hasattr(result, 'metadata') and
                'ui_cards' in result.metadata and
                'sources' in result.metadata
            )
            
            # Check pattern detection
            detected_pattern = result.metadata.get('pattern')
            pattern_ok = detected_pattern == case['expected_pattern']
            
            # Check checklist steps
            ui_cards = result.metadata.get('ui_cards', [])
            checklist_cards = [card for card in ui_cards if card.get('type') == 'checklist']
            checklist_ok = len(checklist_cards) == case['expected_checklist_steps']
            
            # Check response content
            response_ok = len(result.response) > 50 and 'step' in result.response.lower()
            
            success = valid_structure and pattern_ok and checklist_ok and response_ok
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Query: {case['query']}")
            print(f"   Pattern: {detected_pattern}")
            print(f"   Checklist Steps: {len(checklist_cards)}")
            print(f"   UI Cards: {len(ui_cards)}")
            print(f"   Sources: {len(result.metadata.get('sources', []))}")
            print(f"   Status: {status}")
            
            if success:
                passed += 1
            else:
                print(f"   Failure reasons:")
                if not valid_structure:
                    print(f"     - Invalid response structure")
                if not pattern_ok:
                    print(f"     - Pattern mismatch: expected {case['expected_pattern']}, got {detected_pattern}")
                if not checklist_ok:
                    print(f"     - Checklist steps: expected {case['expected_checklist_steps']}, got {len(checklist_cards)}")
                if not response_ok:
                    print(f"     - Response content issue")
            
            print()
            
        except Exception as e:
            print(f"   ❌ FAIL - Exception: {str(e)}")
            print()
    
    print(f"📊 Results: {passed}/{total} tests passed")
    return passed == total

if __name__ == "__main__":
    test_devcopilot()