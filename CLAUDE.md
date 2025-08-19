# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Frontend Development
- `npm install` - Install Node.js dependencies
- `npm run dev` - Start Vite development server on port 3000
- `npm run build` - Build frontend for production (TypeScript compilation + Vite build)
- `npm run preview` - Preview production build

### Backend Development  
- `pip install -r requirements.txt` - Install Python dependencies
- `python main.py` - Start FastAPI backend server on port 8000

### Testing and Quality
- No lint/test commands configured - check with user before running any quality checks
- TypeScript compilation is included in `npm run build`

### Full Development Environment
```bash
# Terminal 1 - Backend
python main.py

# Terminal 2 - Frontend  
npm run dev
```

## Architecture Overview

This is a financial services AI platform with 8 specialized agents built on FastAPI (backend) and React/TypeScript (frontend). The system uses a defensive security approach with built-in PII protection and fraud detection.

### Key Components

**Multi-Agent System**: 8 specialized AI agents handle different financial domains:
- Smart Chat Agent (intelligent routing hub)
- OfferPilot (product search & financing)  
- TrustShield (fraud detection & PII protection)
- Dispute Copilot (credit card disputes)
- Collections Agent (payment assistance)
- DevCopilot (technical support & API docs)
- CareCredit (medical expense analysis)
- Narrator (portfolio analytics)
- ImageGen (AI image generation)

**Intelligent Routing**: Uses Gemini LLM for intent classification with keyword fallback. Routes queries to the most appropriate specialist agent based on confidence scores.

**RAG System**: ChromaDB vector store with Gemini embeddings for document retrieval. Supports markdown documents and PDF processing with bounding box extraction via LandingAI.

**Fallback Mechanisms**: Multi-tiered fallbacks when documents are insufficient:
1. Document assessment via LLM
2. LLM knowledge fallback  
3. Web search (Tavily integration)

**Security Features**: Microsoft Presidio for PII detection/redaction, defensive design principles, fraud pattern recognition.

### File Structure
- `main.py` - FastAPI application entry point with all agent endpoints
- `app/agents/` - Individual agent implementations (8 specialized agents)
  - `chat_agent.py` - Smart Chat routing agent
  - `offerpilot.py` - Product search and financing
  - `trustshield.py` - Fraud detection and PII protection  
  - `dispute_reg_e.py` - Dispute assistance
  - `collections.py` - Payment plans and hardship support
  - `devcopilot.py` - Technical support and API docs
  - `carecredit.py` - Healthcare financing specialist
  - `narrator.py` - Portfolio analytics
  - `imagegen.py` - AI image generation
- `app/rag/` - RAG system (embeddings, retrieval, document store)
  - `core.py` - ChromaDB integration and retrieval logic
  - `pipeline.py` - Document processing pipeline
- `app/llm/gemini.py` - Gemini LLM integration with fallback logic
- `app/router.py` - Intelligent query routing system (confidence-based)
- `app/services/pdf_processor.py` - LandingAI PDF processing pipeline
- `app/config/rules_loader.py` - YAML rules configuration loader
- `src/` - React/TypeScript frontend components
- `synchrony-demo-rules-repo/` - Business rules and knowledge base files

### Frontend Components
- `ChatPane.tsx` - Main chat interface with PDF drag-and-drop
- `LeftRail.tsx` - Agent selector sidebar
- `RightInspector.tsx` - Citations, tool traces, PDF viewer
- `src/config/agents.ts` - Agent configuration and examples

### Environment Variables Required
```bash
GOOGLE_API_KEY=your_gemini_api_key
VISION_AGENT_API_KEY=your_landing_ai_key  
TAVILY_API_KEY=your_tavily_key  # Optional
ALLOW_TAVILY=true  # Optional
```

## Key Development Patterns

**Agent Structure**: Each agent follows a consistent pattern with RAG integration, confidence scoring, and citation tracking.

**Intelligent Routing**: Uses Gemini LLM for intent classification with keyword fallback. Confidence threshold of 0.6 determines routing method.

**RAG Integration**: Documents are indexed as markdown chunks with semantic search. PDF processing extracts text with bounding boxes for interactive viewing.

**Fallback System**: Multi-tiered approach - RAG → LLM Knowledge → Web Search → Legacy Tavily when documents are insufficient.

**Error Handling**: Comprehensive fallback systems ensure graceful degradation when primary knowledge sources fail.

**Security First**: All agents designed for defensive security use only. PII is automatically detected and redacted using Microsoft Presidio.

**TypeScript Frontend**: Strict TypeScript configuration with React, Tailwind CSS, and proper markdown rendering with syntax highlighting.

**Rules-Based Configuration**: YAML files in `synchrony-demo-rules-repo/rules/` define agent behaviors and business logic.

## API Endpoints

### Core Chat
- `POST /chat` - Main chat interface with fallback options
- `POST /agent/{agent_name}` - Direct agent invocation

### PDF Management
- `POST /upload/pdf` - PDF upload and processing  
- `GET /pdf/{pdf_id}/page/{page_num}` - Get PDF page with annotations
- `GET /pdf/{pdf_id}/info` - Get PDF metadata and chunks

### Health Check
- `GET /healthz` - Application health status

The system uses CORS middleware for cross-origin requests and serves static files from the built frontend.

## Important Implementation Notes

**Agent Routing**: The system first attempts Gemini-based classification. If confidence < 0.6, falls back to keyword matching. Default route is TrustShield for security.

**PDF Processing**: Uses LandingAI for text extraction with bounding box coordinates. Processed chunks are automatically indexed into the RAG system.

**PII Protection**: Microsoft Presidio automatically detects and redacts sensitive information (SSN, phone numbers, addresses, etc.) in real-time.

**Document Assessment**: Before falling back to LLM knowledge or web search, the system uses AI to assess whether retrieved documents contain sufficient information to answer the query.

**Business Rules**: The `synchrony-demo-rules-repo/` contains fixtures, golden tests, and YAML configuration files that define agent behaviors and example interactions.