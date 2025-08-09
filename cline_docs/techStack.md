# Tech Stack - Synch GenAI PoC

## Core Framework
- **FastAPI**: Modern, fast web framework for building APIs with Python
- **Uvicorn**: ASGI server for running FastAPI applications
- **Pydantic**: Data validation and settings management using Python type annotations

## AI/ML Components
- **Haystack 2.x**: Modern NLP framework for building search systems and RAG pipelines
  - Using in-memory document store for PoC simplicity
  - Provides retrieval and indexing capabilities
- **Google Generative AI SDK**: Official Python SDK for Gemini API
  - Chat model: gemini-2.5-flash
  - Embedding model: gemini-embedding-001
- **Tavily Python SDK**: Web search API for fallback when local KB doesn't have answers
  - Optional component, controlled by ALLOW_TAVILY environment variable

## Data Processing
- **NumPy**: Numerical computing for vector operations
- **Pandas**: Data manipulation for handling CSV/JSON data stubs
- **Jinja2**: Template engine for UI rendering

## Configuration & Environment
- **Python-dotenv**: Environment variable management from .env files

## Architecture Decisions
- **In-memory storage**: Using Haystack's in-memory document store for simplicity in PoC
- **Single-file UI**: Simple HTML/JavaScript SPA served directly by FastAPI
- **Environment-based configuration**: All API keys and settings via environment variables
- **Modular structure**: Separate directories for agents, tools, RAG, and knowledge base
- **Local development focus**: Optimized for easy local setup and testing

## Development Tools
- **run_local.sh**: Shell script for easy local development server startup
- **requirements.txt**: Pinned dependencies for reproducible builds
