# Synch GenAI PoC - Project Roadmap

## High-Level Goals

### Core Infrastructure
- [x] Set up project structure with required directories
- [x] Create FastAPI application with basic endpoints
- [x] Integrate Haystack 2.x with in-memory document store
- [x] Set up Gemini API integration for chat and embeddings
- [x] Implement hybrid RAG pipeline for knowledge base queries
- [x] Create simple UI for local demo

### Knowledge Base Management
- [x] Set up markdown policies/terms in /kb directory
- [x] Create JSON/CSV data stubs for catalog, offers, providers, metrics, policies
- [x] Add sample data files for demonstration
- [x] Set up merchant contracts handling

### AI Integration
- [x] Implement Gemini chat model integration
- [x] Set up hybrid retrieval (BM25 + semantic embeddings)
- [x] Configure Tavily search as fallback (optional)
- [x] Create agent system for handling different query types

### Deployment & Configuration
- [x] Create comprehensive .env.example with all required variables
- [x] Set up requirements.txt with pinned dependencies
- [x] Create run_local.sh script for easy startup
- [x] Write comprehensive README with setup instructions

## Completion Criteria
- FastAPI server runs on port 8000
- UI is accessible and functional
- RAG system can query knowledge base effectively
- Gemini integration works for both chat and embeddings
- All environment variables are properly documented
- Project can be run locally with minimal setup

## Completed Tasks
- [x] Created project directory structure
- [x] Set up cline_docs folder with essential documentation
