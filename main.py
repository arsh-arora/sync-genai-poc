"""
Synch GenAI PoC - Main FastAPI Application
Complete app wiring with middleware, agent endpoints, and UI
"""

import os
import re
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json

from app.rag.core import init_docstore, GeminiEmbedder, index_markdown, build_retriever, retrieve
from app.llm.gemini import chat_with_context
from app.tools.tavily_search import web_search_into_docstore, WebDisabled
from app.router import route
from app.agents.trustshield import TrustShield
from app.agents.offerpilot import OfferPilot
from app.agents.dispute import DisputeCopilot
from app.agents.collections import CollectionsAdvisor, CustomerState
from app.agents.devcopilot import DevCopilot
from app.agents.carecredit import CareCredit
from app.agents.narrator import PortfolioIntelNarrator
from app.agents.imagegen import ImageGenAgent
from app.services.pdf_processor import LandingAIPDFProcessor, ProcessedPDF

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Synch GenAI PoC",
    description="Complete GenAI-powered financial services platform",
    version="2.0.0"
)

# Global components
docstore = None
embedder = None
retriever = None
agents = {}  # Agent registry
pdf_processor = None
uploaded_pdfs = {}  # Store processed PDFs

# PII Redactor Middleware
class PIIRedactorMiddleware(BaseHTTPMiddleware):
    """Redacts PAN, SSN, CVV from request bodies"""
    
    def __init__(self, app):
        super().__init__(app)
        # PII patterns
        self.patterns = {
            'PAN': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            'SSN': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'CVV': re.compile(r'\b\d{3,4}\b(?=.*(?:cvv|cvc|security|code))', re.IGNORECASE),
        }
    
    async def dispatch(self, request: Request, call_next):
        # Only process POST requests with JSON bodies
        if request.method == "POST" and request.headers.get("content-type", "").startswith("application/json"):
            body = await request.body()
            if body:
                try:
                    body_str = body.decode('utf-8')
                    original_str = body_str
                    
                    # Redact PII patterns
                    for pii_type, pattern in self.patterns.items():
                        body_str = pattern.sub(f'[REDACTED_{pii_type}]', body_str)
                    
                    # Log if redaction occurred
                    if body_str != original_str:
                        logger.warning(f"PII redaction performed on request to {request.url.path}")
                    
                    # Create new request with redacted body
                    from fastapi.requests import Request as FastAPIRequest
                    async def receive():
                        return {"type": "http.request", "body": body_str.encode('utf-8')}
                    
                    # Modify the request
                    request._body = body_str.encode('utf-8')
                    
                except Exception as e:
                    logger.error(f"Error in PII redaction: {e}")
        
        response = await call_next(request)
        return response

# TrustShield Blocking Middleware
class TrustShieldMiddleware(BaseHTTPMiddleware):
    """TrustShield security scanning and blocking"""
    
    def __init__(self, app):
        super().__init__(app)
        self.protected_endpoints = ['/chat', '/agent/', '/api/']
    
    async def dispatch(self, request: Request, call_next):
        # Check if endpoint needs protection
        needs_protection = any(request.url.path.startswith(endpoint) 
                             for endpoint in self.protected_endpoints)
        
        if needs_protection and request.method == "POST":
            # Get request body for TrustShield scan
            body = await request.body()
            if body and agents.get('trustshield'):
                try:
                    body_data = json.loads(body.decode('utf-8'))
                    
                    # Extract text to scan (look for common field names)
                    text_to_scan = ""
                    for field in ['message', 'query', 'narrative', 'question', 'estimate_text']:
                        if field in body_data:
                            text_to_scan = body_data[field]
                            break
                    
                    if text_to_scan:
                        # Run TrustShield scan
                        shield_result = agents['trustshield'].scan(text_to_scan)
                        
                        if shield_result["decision"] == "block":
                            logger.warning(f"TrustShield BLOCKED request to {request.url.path}")
                            return JSONResponse(
                                status_code=403,
                                content={
                                    "error": "Request blocked by security policy",
                                    "reason": shield_result.get('next_step', {}).get('label', 'Security violation detected'),
                                    "blocked": True
                                }
                            )
                        
                        # Replace original text with redacted version
                        for field in ['message', 'query', 'narrative', 'question', 'estimate_text']:
                            if field in body_data:
                                body_data[field] = shield_result["redacted_text"]
                        
                        # Update request body
                        new_body = json.dumps(body_data).encode('utf-8')
                        request._body = new_body
                    
                except Exception as e:
                    logger.error(f"Error in TrustShield middleware: {e}")
        
        response = await call_next(request)
        return response

# Add middleware (order matters - PII redaction first, then TrustShield)
app.add_middleware(TrustShieldMiddleware)
app.add_middleware(PIIRedactorMiddleware)

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global docstore, embedder, retriever, agents, pdf_processor
    
    try:
        logger.info("Initializing document store...")
        docstore = init_docstore()
        
        logger.info("Initializing Gemini embedder...")
        embedder = GeminiEmbedder()
        
        logger.info("Indexing markdown documents...")
        doc_count = index_markdown(docstore, embedder)
        logger.info(f"Indexed {doc_count} document chunks")
        
        logger.info("Building retriever...")
        retriever = build_retriever(docstore)
        
        logger.info("Initializing PDF processor...")
        pdf_processor = LandingAIPDFProcessor()
        
        logger.info("Initializing agents...")
        agents = {
            'trustshield': TrustShield(docstore=docstore, embedder=embedder, retriever=retriever),
            'offerpilot': OfferPilot(docstore=docstore, embedder=embedder, retriever=retriever),
            'dispute': DisputeCopilot(docstore=docstore, embedder=embedder, retriever=retriever),
            'collections': CollectionsAdvisor(docstore=docstore, embedder=embedder, retriever=retriever),
            'devcopilot': DevCopilot(),
            'carecredit': CareCredit(docstore=docstore, embedder=embedder, retriever=retriever),
            'narrator': PortfolioIntelNarrator(docstore=docstore, embedder=embedder, retriever=retriever),
            'imagegen': ImageGenAgent(),
        }
        
        logger.info(f"Application startup complete - {len(agents)} agents initialized")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    allow_tavily: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    agent: str
    confidence: float
    sources: list[str] = []
    used_tavily: bool = False
    image_data: Optional[str] = None
    image_format: Optional[str] = None

class OfferRequest(BaseModel):
    query: str
    budget: Optional[float] = None

class DisputeRequest(BaseModel):
    narrative: str
    merchant: Optional[str] = None
    amount: Optional[float] = None
    uploaded_text: Optional[str] = None

class CollectionsRequest(BaseModel):
    balance: float
    apr: float
    bucket: str
    income_monthly: Optional[float] = None
    expenses_monthly: Optional[float] = None
    preferences: Optional[dict] = None

class DevCopilotRequest(BaseModel):
    service: str
    endpoint: Optional[str] = None
    lang: str = "python"
    sample: Optional[dict] = None

class CareCreditRequest(BaseModel):
    estimate_text: str
    location: Optional[str] = None
    insurance: Optional[dict] = None

class NarratorRequest(BaseModel):
    question: str

class ImageGenRequest(BaseModel):
    prompt: str
    include_text: Optional[bool] = True
    style_hints: Optional[list[str]] = None

# UI Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the modern React UI"""
    try:
        ui_path = Path("app/ui/modern.html")
        if ui_path.exists():
            return HTMLResponse(content=ui_path.read_text(encoding='utf-8'), status_code=200)
        else:
            return HTMLResponse(content="<h1>Modern UI not found</h1><p>Please check app/ui/modern.html</p>", status_code=404)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading UI</h1><p>{str(e)}</p>", status_code=500)

@app.get("/classic", response_class=HTMLResponse)
async def classic_ui():
    """Serve the classic UI as backup"""
    return HTMLResponse(content=UI_HTML, status_code=200)

# Smart Chat Endpoint
@app.post("/chat", response_model=ChatResponse)
async def smart_chat(request: ChatRequest):
    """Smart chat with router dispatch to appropriate agent"""
    if not all([docstore, embedder, retriever]) or not agents:
        raise HTTPException(status_code=500, detail="System components not initialized")
    
    try:
        logger.info(f"Processing chat message: {request.message}")
        
        # Route the query to appropriate agent
        routing_result = route(request.message)
        routed_agent = routing_result["agent"]
        confidence = routing_result["confidence"]
        
        logger.info(f"Query routed to '{routed_agent}' agent with confidence {confidence:.3f}")
        
        # Initialize Tavily usage flag
        used_tavily = False
        
        # Dispatch to specific agent based on routing
        if routed_agent == "offerpilot" and "offerpilot" in agents:
            agent_result = agents["offerpilot"].process_query(request.message)
            response_text = f"Found {len(agent_result.items)} products. Pre-qualification: {'Eligible' if agent_result.prequal.eligible else 'Not eligible'}"
            sources = [f"Products: {len(agent_result.items)}", f"Pre-qual: {agent_result.prequal.reason}"]
            
        elif routed_agent == "dispute" and "dispute" in agents:
            agent_result = agents["dispute"].process_dispute(request.message)
            response_text = f"Dispute triage: {agent_result.triage}. Merchant resolution available."
            sources = [f"Triage: {agent_result.triage}", f"Resolution steps: {len(agent_result.merchant_resolution.checklist)}"]
            
        elif routed_agent == "collections" and "collections" in agents:
            # Need customer state for collections - create from message parsing
            customer_state = CustomerState(balance=1000.0, apr=24.99, bucket="30-60")
            agent_result = agents["collections"].process_hardship_request(customer_state)
            response_text = f"Generated {len(agent_result.plans)} hardship plans"
            sources = [f"Plans available: {len(agent_result.plans)}"]
            
        elif routed_agent == "imagegen" and "imagegen" in agents:
            from app.agents.imagegen import ImageGenRequest as AgentImageGenRequest
            image_request = AgentImageGenRequest(prompt=request.message)
            agent_result = agents["imagegen"].process_request(image_request)
            
            if agent_result.success:
                response_text = agent_result.generated_text or f"Generated image: {request.message}"
                sources = [f"Image generated successfully", f"Format: {agent_result.image_format}"]
                
                # Add image data to response (we'll handle this in UI)
                return ChatResponse(
                    response=response_text,
                    agent=routed_agent,
                    confidence=confidence,
                    sources=sources,
                    used_tavily=used_tavily,
                    image_data=agent_result.image_base64,  # Add this field
                    image_format=agent_result.image_format
                )
            else:
                response_text = f"Failed to generate image: {agent_result.error_message}"
                sources = ["Image generation failed"]
            
        else:
            # Fallback to RAG-based response
            retrieved_docs = retrieve(retriever, embedder, request.message, k=5)
            
            # Check for Tavily usage - trigger if enabled and low quality/relevance results
            should_use_web = (
                request.allow_tavily and 
                os.getenv("ALLOW_TAVILY", "false").lower() == "true" and
                (len(retrieved_docs) < 2 or 
                 (len(retrieved_docs) > 0 and all(doc.get('score', 0) < 0.7 for doc in retrieved_docs)))
            )
            
            if should_use_web:
                try:
                    web_docs = web_search_into_docstore(docstore, embedder, request.message, max_results=3)
                    used_tavily = len(web_docs) > 0
                    logger.info(f"Used Tavily web search: {used_tavily}")
                    
                    # Re-retrieve documents now that we have web content in the docstore
                    if used_tavily:
                        logger.info("Re-retrieving documents including web search results...")
                        retrieved_docs = retrieve(retriever, embedder, request.message, k=5)
                        logger.info(f"Re-retrieved {len(retrieved_docs)} documents with web content")
                        
                except Exception as e:
                    logger.warning(f"Web search failed: {e}")
                    used_tavily = False
            
            if retrieved_docs:
                response_text = chat_with_context(request.message, retrieved_docs)
                sources = [f"{doc['filename']} - Score: {doc['score']:.3f}" for doc in retrieved_docs[:3]]
            else:
                response_text = "I couldn't find relevant information to answer your question."
                sources = []
        
        # Add routing info to sources
        sources.insert(0, f"🎯 Routed to: {routed_agent} ({confidence:.1%} confidence)")
        
        return ChatResponse(
            response=response_text,
            agent=routed_agent,
            confidence=confidence,
            sources=sources,
            used_tavily=used_tavily
        )
        
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Agent-specific endpoints for tabbed UI testing
@app.post("/agent/offerpilot")
async def agent_offerpilot(request: OfferRequest):
    """OfferPilot direct endpoint"""
    if "offerpilot" not in agents:
        raise HTTPException(status_code=500, detail="OfferPilot not initialized")
    
    try:
        result = agents["offerpilot"].process_query(request.query, request.budget)
        return result.dict()
    except Exception as e:
        logger.error(f"OfferPilot error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/dispute")
async def agent_dispute(request: DisputeRequest):
    """Dispute Copilot direct endpoint"""
    if "dispute" not in agents:
        raise HTTPException(status_code=500, detail="DisputeCopilot not initialized")
    
    try:
        result = agents["dispute"].process_dispute(
            narrative=request.narrative,
            merchant=request.merchant,
            amount=request.amount,
            uploaded_text=request.uploaded_text
        )
        return result.dict()
    except Exception as e:
        logger.error(f"DisputeCopilot error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/collections")
async def agent_collections(request: CollectionsRequest):
    """Collections Advisor direct endpoint"""
    if "collections" not in agents:
        raise HTTPException(status_code=500, detail="CollectionsAdvisor not initialized")
    
    try:
        customer_state = CustomerState(
            balance=request.balance,
            apr=request.apr,
            bucket=request.bucket,
            income_monthly=request.income_monthly,
            expenses_monthly=request.expenses_monthly,
            preferences=request.preferences
        )
        result = agents["collections"].process_hardship_request(customer_state)
        return result.dict()
    except Exception as e:
        logger.error(f"CollectionsAdvisor error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/devcopilot")
async def agent_devcopilot(request: DevCopilotRequest):
    """DevCopilot direct endpoint"""
    if "devcopilot" not in agents:
        raise HTTPException(status_code=500, detail="DevCopilot not initialized")
    
    try:
        result = agents["devcopilot"].generate_code_guide(
            service=request.service,
            endpoint=request.endpoint,
            lang=request.lang,
            sample=request.sample
        )
        return result.dict()
    except Exception as e:
        logger.error(f"DevCopilot error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/carecredit")
async def agent_carecredit(request: CareCreditRequest):
    """CareCredit direct endpoint"""
    if "carecredit" not in agents:
        raise HTTPException(status_code=500, detail="CareCredit not initialized")
    
    try:
        result = agents["carecredit"].process_estimate(
            estimate_text=request.estimate_text,
            location=request.location,
            insurance=request.insurance
        )
        return result.dict()
    except Exception as e:
        logger.error(f"CareCredit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/narrator")
async def agent_narrator(request: NarratorRequest):
    """Portfolio Intel Narrator direct endpoint"""
    if "narrator" not in agents:
        raise HTTPException(status_code=500, detail="PortfolioIntelNarrator not initialized")
    
    try:
        result = agents["narrator"].process_question(request.question)
        return result.dict()
    except Exception as e:
        logger.error(f"PortfolioIntelNarrator error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/imagegen")
async def agent_imagegen(request: ImageGenRequest):
    """ImageGen direct endpoint"""
    if "imagegen" not in agents:
        raise HTTPException(status_code=500, detail="ImageGen not initialized")
    
    try:
        result = agents["imagegen"].process_request(request)
        return result.dict()
    except Exception as e:
        logger.error(f"ImageGen error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/trustshield")
async def agent_trustshield(request: dict):
    """TrustShield direct endpoint"""
    if "trustshield" not in agents:
        raise HTTPException(status_code=500, detail="TrustShield not initialized")
    
    try:
        text = request.get("text", "")
        result = agents["trustshield"].scan(text)
        return result
    except Exception as e:
        logger.error(f"TrustShield error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF document"""
    if not pdf_processor:
        raise HTTPException(status_code=500, detail="PDF processor not initialized")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Limit file size (10MB)
    max_size = 10 * 1024 * 1024
    
    try:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            
            if len(content) > max_size:
                raise HTTPException(status_code=400, detail="File too large (max 10MB)")
            
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Process PDF
        logger.info(f"Processing uploaded PDF: {file.filename}")
        processed_pdf = pdf_processor.process_pdf(tmp_path, chunk_strategy="semantic")
        
        # Store processed PDF
        pdf_id = f"pdf_{int(time.time())}_{len(uploaded_pdfs)}"
        uploaded_pdfs[pdf_id] = processed_pdf
        
        # Add chunks to document store for RAG
        documents = []
        for chunk in processed_pdf.chunks:
            doc = chunk.to_document(file.filename)
            doc.meta["pdf_id"] = pdf_id
            documents.append(doc)
        
        # Generate embeddings and add to docstore
        if documents:
            texts = [doc.content for doc in documents]
            embeddings = embedder.embed_texts(texts)
            
            for doc, embedding in zip(documents, embeddings):
                doc.embedding = embedding
            
            docstore.write_documents(documents)
            logger.info(f"Added {len(documents)} PDF chunks to docstore")
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return {
            "success": True,
            "pdf_id": pdf_id,
            "filename": file.filename,
            "total_pages": processed_pdf.total_pages,
            "chunks_extracted": len(processed_pdf.chunks),
            "processing_time": processed_pdf.processing_time,
            "file_size": processed_pdf.file_size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF upload: {e}")
        # Clean up temp file if it exists
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@app.get("/pdf/{pdf_id}/info")
async def get_pdf_info(pdf_id: str):
    """Get detailed information about a processed PDF including chunks"""
    if pdf_id not in uploaded_pdfs:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    pdf = uploaded_pdfs[pdf_id]
    return {
        "pdf_id": pdf_id,
        "filename": pdf.filename,
        "total_pages": pdf.total_pages,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                "page_number": chunk.page_number,
                "bbox": chunk.bbox.to_dict(),
                "confidence": chunk.confidence
            }
            for chunk in pdf.chunks
        ],
        "processing_time": pdf.processing_time
    }

@app.get("/pdf/{pdf_id}")
async def get_pdf_summary(pdf_id: str):
    """Get summary information about a processed PDF"""
    if pdf_id not in uploaded_pdfs:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    pdf = uploaded_pdfs[pdf_id]
    return {
        "pdf_id": pdf_id,
        "filename": pdf.filename,
        "total_pages": pdf.total_pages,
        "chunks": len(pdf.chunks),
        "processing_time": pdf.processing_time
    }

@app.get("/pdf/{pdf_id}/page/{page_num}")
async def get_pdf_page(pdf_id: str, page_num: int):
    """Get a specific page image with optional bounding box overlays"""
    if pdf_id not in uploaded_pdfs:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    pdf = uploaded_pdfs[pdf_id]
    
    if page_num >= len(pdf.page_images) or page_num < 0:
        raise HTTPException(status_code=404, detail="Page not found")
    
    # Get chunks for this page
    page_chunks = [chunk for chunk in pdf.chunks if chunk.page_number == page_num]
    
    return {
        "pdf_id": pdf_id,
        "page_number": page_num,
        "image_base64": pdf.page_images[page_num],
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "bbox": chunk.bbox.to_dict(),
                "confidence": chunk.confidence
            }
            for chunk in page_chunks
        ]
    }

# Health endpoint
@app.get("/healthz")
async def health_check():
    """Enhanced health check with docstore size and env flags"""
    try:
        docstore_size = docstore.count_documents() if docstore else 0
        
        env_flags = {
            "ALLOW_TAVILY": os.getenv("ALLOW_TAVILY", "false").lower() == "true",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
            "DEBUG": os.getenv("DEBUG", "false").lower() == "true",
        }
        
        agent_status = {name: agent is not None for name, agent in agents.items()}
        
        return {
            "status": "healthy" if docstore and len(agents) > 0 else "degraded",
            "docstore_size": docstore_size,
            "agents_initialized": len(agents),
            "agent_status": agent_status,
            "env_flags": env_flags,
            "components": {
                "docstore": docstore is not None,
                "embedder": embedder is not None, 
                "retriever": retriever is not None,
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Minimal SPA UI
UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Synch GenAI PoC</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #f8fafc; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px 0; margin-bottom: 30px; border-radius: 12px; }
        .header h1 { text-align: center; font-size: 2.5rem; font-weight: 700; margin-bottom: 10px; }
        .header p { text-align: center; font-size: 1.1rem; opacity: 0.9; }
        
        .tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 30px; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
        .tab { padding: 12px 20px; background: #f1f5f9; border: none; border-radius: 8px; cursor: pointer; font-weight: 500; transition: all 0.2s; color: #64748b; }
        .tab:hover { background: #e2e8f0; }
        .tab.active { background: #3b82f6; color: white; }
        
        .tab-content { display: none; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
        .tab-content.active { display: block; }
        
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; font-weight: 600; margin-bottom: 8px; color: #374151; }
        .form-group input, .form-group textarea, .form-group select { width: 100%; padding: 12px 16px; border: 2px solid #e5e7eb; border-radius: 8px; font-size: 14px; transition: border-color 0.2s; }
        .form-group input:focus, .form-group textarea:focus, .form-group select:focus { outline: none; border-color: #3b82f6; }
        .form-group textarea { min-height: 100px; resize: vertical; }
        
        .btn { background: #3b82f6; color: white; border: none; padding: 14px 28px; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 14px; transition: background 0.2s; }
        .btn:hover { background: #2563eb; }
        .btn:disabled { background: #9ca3af; cursor: not-allowed; }
        
        .result { margin-top: 30px; padding: 20px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #3b82f6; }
        .result-json { background: #1f2937; color: #f9fafb; padding: 20px; border-radius: 8px; font-family: 'SF Mono', Monaco, monospace; font-size: 13px; overflow-x: auto; white-space: pre; }
        
        .loading { display: none; text-align: center; padding: 20px; color: #6b7280; }
        .loading.show { display: block; }
        
        .error { background: #fef2f2; border: 1px solid #fecaca; color: #dc2626; padding: 16px; border-radius: 8px; margin-top: 20px; }
        
        .example { background: #f0f9ff; border: 1px solid #bae6fd; padding: 16px; border-radius: 8px; margin-bottom: 20px; font-size: 14px; }
        .example-title { font-weight: 600; color: #0369a1; margin-bottom: 8px; }
        
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; border: 1px solid #e5e7eb; }
        .card h3 { margin-bottom: 12px; color: #374151; }
        .card-content { color: #6b7280; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Synch GenAI PoC</h1>
            <p>Complete AI-powered financial services platform with 8 specialized agents</p>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('chat')">💬 Smart Chat</button>
            <button class="tab" onclick="showTab('offerpilot')">🛍️ OfferPilot</button>
            <button class="tab" onclick="showTab('dispute')">⚖️ Dispute Copilot</button>
            <button class="tab" onclick="showTab('collections')">💳 Collections</button>
            <button class="tab" onclick="showTab('devcopilot')">👨‍💻 DevCopilot</button>
            <button class="tab" onclick="showTab('carecredit')">🏥 CareCredit</button>
            <button class="tab" onclick="showTab('narrator')">📊 Portfolio Intel</button>
            <button class="tab" onclick="showTab('trustshield')">🛡️ TrustShield</button>
        </div>
        
        <!-- Smart Chat Tab -->
        <div id="chat" class="tab-content active">
            <div class="example">
                <div class="example-title">💡 Smart Chat Example:</div>
                "I want to buy a laptop under $1000 with 0% financing"
            </div>
            <div class="form-group">
                <label>Message:</label>
                <textarea id="chat-message" placeholder="Ask anything - the router will send you to the right agent..."></textarea>
            </div>
            <div class="form-group">
                <label><input type="checkbox" id="chat-tavily"> Allow web search (Tavily)</label>
            </div>
            <button class="btn" onclick="runChat()">💬 Send Message</button>
            <div class="loading" id="chat-loading">Processing your message...</div>
            <div id="chat-result"></div>
        </div>
        
        <!-- OfferPilot Tab -->
        <div id="offerpilot" class="tab-content">
            <div class="example">
                <div class="example-title">🛍️ OfferPilot Example:</div>
                Query: "office desk" | Budget: 600
            </div>
            <div class="form-group">
                <label>Search Query:</label>
                <input type="text" id="offer-query" placeholder="office desk" value="office desk">
            </div>
            <div class="form-group">
                <label>Budget (optional):</label>
                <input type="number" id="offer-budget" placeholder="600" value="600">
            </div>
            <button class="btn" onclick="runOfferPilot()">🛍️ Search Products</button>
            <div class="loading" id="offerpilot-loading">Searching products...</div>
            <div id="offerpilot-result"></div>
        </div>
        
        <!-- Dispute Copilot Tab -->
        <div id="dispute" class="tab-content">
            <div class="example">
                <div class="example-title">⚖️ Dispute Example:</div>
                "I was charged twice for the same Amazon purchase on my Synchrony card"
            </div>
            <div class="form-group">
                <label>Dispute Narrative:</label>
                <textarea id="dispute-narrative" placeholder="Describe your dispute..." value="I was charged twice for the same Amazon purchase on my Synchrony card"></textarea>
            </div>
            <div class="form-group">
                <label>Merchant (optional):</label>
                <input type="text" id="dispute-merchant" placeholder="Amazon" value="Amazon">
            </div>
            <div class="form-group">
                <label>Amount (optional):</label>
                <input type="number" id="dispute-amount" placeholder="150.00" value="150.00">
            </div>
            <button class="btn" onclick="runDispute()">⚖️ Process Dispute</button>
            <div class="loading" id="dispute-loading">Processing dispute...</div>
            <div id="dispute-result"></div>
        </div>
        
        <!-- Collections Tab -->
        <div id="collections" class="tab-content">
            <div class="example">
                <div class="example-title">💳 Collections Example:</div>
                Balance: 5000 | APR: 24.99% | Bucket: "90-120"
            </div>
            <div class="form-group">
                <label>Balance:</label>
                <input type="number" id="collections-balance" placeholder="5000" value="5000">
            </div>
            <div class="form-group">
                <label>APR (%):</label>
                <input type="number" id="collections-apr" placeholder="24.99" value="24.99" step="0.01">
            </div>
            <div class="form-group">
                <label>Delinquency Bucket:</label>
                <select id="collections-bucket">
                    <option value="30-60">30-60 days</option>
                    <option value="60-90">60-90 days</option>
                    <option value="90-120" selected>90-120 days</option>
                    <option value="120+">120+ days</option>
                </select>
            </div>
            <div class="form-group">
                <label>Monthly Income (optional):</label>
                <input type="number" id="collections-income" placeholder="4000">
            </div>
            <button class="btn" onclick="runCollections()">💳 Generate Hardship Plans</button>
            <div class="loading" id="collections-loading">Generating plans...</div>
            <div id="collections-result"></div>
        </div>
        
        <!-- DevCopilot Tab -->
        <div id="devcopilot" class="tab-content">
            <div class="example">
                <div class="example-title">👨‍💻 DevCopilot Example:</div>
                Service: "payments" | Language: "python"
            </div>
            <div class="form-group">
                <label>Service:</label>
                <input type="text" id="dev-service" placeholder="payments" value="payments">
            </div>
            <div class="form-group">
                <label>Endpoint (optional):</label>
                <input type="text" id="dev-endpoint" placeholder="/payments">
            </div>
            <div class="form-group">
                <label>Language:</label>
                <select id="dev-language">
                    <option value="python" selected>Python</option>
                    <option value="javascript">JavaScript</option>
                    <option value="java">Java</option>
                    <option value="curl">cURL</option>
                </select>
            </div>
            <button class="btn" onclick="runDevCopilot()">👨‍💻 Generate Code</button>
            <div class="loading" id="devcopilot-loading">Generating code...</div>
            <div id="devcopilot-result"></div>
        </div>
        
        <!-- CareCredit Tab -->
        <div id="carecredit" class="tab-content">
            <div class="example">
                <div class="example-title">🏥 CareCredit Example:</div>
                Medical estimate with procedure codes and costs
            </div>
            <div class="form-group">
                <label>Medical Estimate:</label>
                <textarea id="care-estimate" placeholder="Paste your medical estimate..." value="Dental Estimate - City Dental Care

D0120 | Periodic oral evaluation | $85.00
D1110 | Prophylaxis - adult cleaning | $120.00
D0274 | Bitewing X-rays (4 films) | $65.00

Total: $270.00"></textarea>
            </div>
            <div class="form-group">
                <label>Location (optional):</label>
                <input type="text" id="care-location" placeholder="New York, NY" value="New York, NY">
            </div>
            <button class="btn" onclick="runCareCredit()">🏥 Analyze Treatment</button>
            <div class="loading" id="carecredit-loading">Analyzing treatment...</div>
            <div id="carecredit-result"></div>
        </div>
        
        <!-- Portfolio Intel Narrator Tab -->
        <div id="narrator" class="tab-content">
            <div class="example">
                <div class="example-title">📊 Portfolio Intel Example:</div>
                "Why did spend drop after 2025-07-31?"
            </div>
            <div class="form-group">
                <label>Business Question:</label>
                <textarea id="narrator-question" placeholder="Ask about portfolio metrics..." value="Why did spend drop after 2025-07-31?"></textarea>
            </div>
            <button class="btn" onclick="runNarrator()">📊 Analyze Metrics</button>
            <div class="loading" id="narrator-loading">Analyzing portfolio data...</div>
            <div id="narrator-result"></div>
        </div>
        
        <!-- TrustShield Tab -->
        <div id="trustshield" class="tab-content">
            <div class="example">
                <div class="example-title">🛡️ TrustShield Example:</div>
                "My SSN is 123-45-6789 and my credit card is 4111-1111-1111-1111"
            </div>
            <div class="form-group">
                <label>Text to Scan:</label>
                <textarea id="trust-text" placeholder="Enter text to scan for security issues..." value="My SSN is 123-45-6789 and my credit card is 4111-1111-1111-1111"></textarea>
            </div>
            <button class="btn" onclick="runTrustShield()">🛡️ Security Scan</button>
            <div class="loading" id="trustshield-loading">Scanning for security issues...</div>
            <div id="trustshield-result"></div>
        </div>
    </div>

    <script>
        function showTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }
        
        function showLoading(agentName, show = true) {
            const loading = document.getElementById(`${agentName}-loading`);
            if (loading) {
                loading.classList.toggle('show', show);
            }
        }
        
        function showResult(agentName, result) {
            const resultDiv = document.getElementById(`${agentName}-result`);
            if (!resultDiv) return;
            
            // Create human-readable and JSON views
            const humanReadable = createHumanReadableView(result);
            const jsonView = JSON.stringify(result, null, 2);
            
            resultDiv.innerHTML = `
                <div class="result">
                    <h3>📋 Summary</h3>
                    ${humanReadable}
                    <h3 style="margin-top: 25px;">📄 Raw JSON Response</h3>
                    <div class="result-json">${jsonView}</div>
                </div>
            `;
        }
        
        function createHumanReadableView(result) {
            // Create cards/tiles based on result type
            if (result.items && Array.isArray(result.items)) {
                // OfferPilot results
                return `
                    <div class="grid">
                        ${result.items.map(item => `
                            <div class="card">
                                <h3>${item.title}</h3>
                                <div class="card-content">
                                    <p><strong>Price:</strong> $${item.price}</p>
                                    <p><strong>Merchant:</strong> ${item.merchant}</p>
                                    <p><strong>Offers:</strong> ${item.offers.length} financing options</p>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                `;
            } else if (result.findings && Array.isArray(result.findings)) {
                // Narrator results
                return `
                    <div class="grid">
                        ${result.findings.map(finding => `
                            <div class="card">
                                <h3>${finding.title}</h3>
                                <div class="card-content">
                                    <pre>${JSON.stringify(finding.evidence, null, 2)}</pre>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                `;
            } else if (result.line_items && Array.isArray(result.line_items)) {
                // CareCredit results
                return `
                    <div class="card">
                        <h3>💰 Treatment Costs</h3>
                        <div class="card-content">
                            ${result.line_items.map(item => `
                                <p><strong>${item.name}:</strong> $${item.subtotal} (${item.qty}x $${item.unit_cost})</p>
                            `).join('')}
                            <hr>
                            <p><strong>Total:</strong> $${result.oopp.estimated_total}</p>
                        </div>
                    </div>
                    <div class="card">
                        <h3>🏥 Providers Found</h3>
                        <div class="card-content">
                            ${result.providers.map(provider => `
                                <p><strong>${provider.name}</strong><br>
                                Next appointment: ${provider.next_appt_days} days</p>
                            `).join('')}
                        </div>
                    </div>
                `;
            } else if (result.snippet && result.code) {
                // DevCopilot results
                return `
                    <div class="card">
                        <h3>💻 Generated Code</h3>
                        <div class="result-json">${result.snippet.code}</div>
                    </div>
                `;
            } else if (result.response) {
                // Chat results
                return `
                    <div class="card">
                        <h3>💬 Response</h3>
                        <div class="card-content">
                            <p>${result.response}</p>
                            <p><strong>Agent:</strong> ${result.agent} (${(result.confidence * 100).toFixed(1)}% confidence)</p>
                        </div>
                    </div>
                `;
            }
            
            return '<p>Response received successfully.</p>';
        }
        
        function showError(agentName, error) {
            const resultDiv = document.getElementById(`${agentName}-result`);
            if (!resultDiv) return;
            
            resultDiv.innerHTML = `<div class="error">❌ Error: ${error}</div>`;
        }
        
        async function apiCall(endpoint, data) {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return response.json();
        }
        
        async function runChat() {
            const message = document.getElementById('chat-message').value;
            const allowTavily = document.getElementById('chat-tavily').checked;
            
            if (!message.trim()) return;
            
            showLoading('chat', true);
            try {
                const result = await apiCall('/chat', { message, allow_tavily: allowTavily });
                showResult('chat', result);
            } catch (error) {
                showError('chat', error.message);
            } finally {
                showLoading('chat', false);
            }
        }
        
        async function runOfferPilot() {
            const query = document.getElementById('offer-query').value;
            const budget = parseFloat(document.getElementById('offer-budget').value) || null;
            
            if (!query.trim()) return;
            
            showLoading('offerpilot', true);
            try {
                const result = await apiCall('/agent/offerpilot', { query, budget });
                showResult('offerpilot', result);
            } catch (error) {
                showError('offerpilot', error.message);
            } finally {
                showLoading('offerpilot', false);
            }
        }
        
        async function runDispute() {
            const narrative = document.getElementById('dispute-narrative').value;
            const merchant = document.getElementById('dispute-merchant').value || null;
            const amount = parseFloat(document.getElementById('dispute-amount').value) || null;
            
            if (!narrative.trim()) return;
            
            showLoading('dispute', true);
            try {
                const result = await apiCall('/agent/dispute', { narrative, merchant, amount });
                showResult('dispute', result);
            } catch (error) {
                showError('dispute', error.message);
            } finally {
                showLoading('dispute', false);
            }
        }
        
        async function runCollections() {
            const balance = parseFloat(document.getElementById('collections-balance').value);
            const apr = parseFloat(document.getElementById('collections-apr').value);
            const bucket = document.getElementById('collections-bucket').value;
            const income = parseFloat(document.getElementById('collections-income').value) || null;
            
            if (!balance || !apr) return;
            
            showLoading('collections', true);
            try {
                const result = await apiCall('/agent/collections', { balance, apr, bucket, income_monthly: income });
                showResult('collections', result);
            } catch (error) {
                showError('collections', error.message);
            } finally {
                showLoading('collections', false);
            }
        }
        
        async function runDevCopilot() {
            const service = document.getElementById('dev-service').value;
            const endpoint = document.getElementById('dev-endpoint').value || null;
            const lang = document.getElementById('dev-language').value;
            
            if (!service.trim()) return;
            
            showLoading('devcopilot', true);
            try {
                const result = await apiCall('/agent/devcopilot', { service, endpoint, lang });
                showResult('devcopilot', result);
            } catch (error) {
                showError('devcopilot', error.message);
            } finally {
                showLoading('devcopilot', false);
            }
        }
        
        async function runCareCredit() {
            const estimate_text = document.getElementById('care-estimate').value;
            const location = document.getElementById('care-location').value || null;
            
            if (!estimate_text.trim()) return;
            
            showLoading('carecredit', true);
            try {
                const result = await apiCall('/agent/carecredit', { estimate_text, location });
                showResult('carecredit', result);
            } catch (error) {
                showError('carecredit', error.message);
            } finally {
                showLoading('carecredit', false);
            }
        }
        
        async function runNarrator() {
            const question = document.getElementById('narrator-question').value;
            
            if (!question.trim()) return;
            
            showLoading('narrator', true);
            try {
                const result = await apiCall('/agent/narrator', { question });
                showResult('narrator', result);
            } catch (error) {
                showError('narrator', error.message);
            } finally {
                showLoading('narrator', false);
            }
        }
        
        async function runTrustShield() {
            const text = document.getElementById('trust-text').value;
            
            if (!text.trim()) return;
            
            showLoading('trustshield', true);
            try {
                const result = await apiCall('/agent/trustshield', { text });
                showResult('trustshield', result);
            } catch (error) {
                showError('trustshield', error.message);
            } finally {
                showLoading('trustshield', false);
            }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )