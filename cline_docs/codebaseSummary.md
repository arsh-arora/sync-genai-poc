# Codebase Summary - Synch GenAI PoC

## Project Structure
```
synch-genai-poc/
├── cline_docs/          # Project documentation
├── app/                 # Main application code
│   ├── agents/          # AI agents for different query types
│   ├── tools/           # Utility tools and helpers
│   ├── rag/             # RAG pipeline implementation
│   ├── kb/              # Knowledge base (markdown policies, extracted PDFs)
│   ├── data/            # JSON/CSV stubs for catalog, offers, providers
│   ├── openapi/         # Dummy partner API specifications
│   ├── contracts/       # Merchant contracts (md or pdf+md extract)
│   └── ui/              # Simple SPA for local demo
├── .env.example         # Environment variable template
├── requirements.txt     # Python dependencies
├── run_local.sh         # Local development startup script
└── README.md            # Setup and usage instructions
```

## Key Components and Their Interactions

### Core Application (Planned)
- **FastAPI Server**: Main web server handling API requests and serving UI
- **RAG Pipeline**: Haystack-based retrieval system for knowledge base queries
- **Gemini Integration**: Chat and embedding services via Google AI SDK
- **Agent System**: Different agents for handling various query types

### Data Flow (Planned)
1. User queries come through UI or API endpoints
2. Agents determine query type and route to appropriate handler
3. RAG system searches knowledge base using Gemini embeddings
4. If no relevant results, optionally fall back to Tavily search
5. Gemini chat model generates response based on retrieved context
6. Response returned to user through UI

### External Dependencies
- **Google AI API**: For Gemini chat and embedding models
- **Tavily API**: Optional web search fallback
- **Haystack 2.x**: Document indexing and retrieval
- **FastAPI ecosystem**: Web framework and ASGI server

## Recent Significant Changes
- Initial project structure created
- Documentation framework established
- Technology stack defined and documented

## User Feedback Integration and Its Impact on Development
- Project structure follows user specifications exactly
- No databases as per user requirements
- Focus on minimal viable PoC implementation
- Local development optimized setup as requested

## Current Status
- Project structure: ✅ Complete
- Documentation: ✅ Complete
- Dependencies: ✅ Complete
- Core implementation: ✅ Complete
- Hybrid RAG pipeline: ✅ Complete
- Gemini integration: ✅ Complete
- UI: ✅ Complete
- Sample data: ✅ Complete
- Ready for testing: ✅ Complete
