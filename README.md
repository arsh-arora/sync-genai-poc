# Synch GenAI PoC

A complete GenAI-powered financial services platform with 8 specialized AI agents, intelligent document processing, and enhanced markdown rendering.

## Quick Start

### Install Dependencies
```bash
# Install Node.js dependencies
npm install

# Install Python dependencies (if not already done)
pip install -r requirements.txt
```

### Development
```bash
# Start backend server (Terminal 1)
python main.py

# Start frontend development server (Terminal 2)
npm run dev
```

### Production
```bash
# Build frontend
npm run build

# Start production server
python main.py
```

## Features

### 🧠 Intelligent Fallback System
- **LLM Knowledge Fallback**: Uses the LLM's training knowledge when documents are insufficient
- **Web Search Integration**: Tavily search for current information
- **Smart Document Assessment**: AI-powered evaluation of document relevance and coverage

### 🎨 Enhanced UI
- **React + TypeScript**: Modern, type-safe frontend with proper markdown rendering
- **Syntax Highlighting**: Beautiful code blocks with language detection
- **Drag & Drop PDF**: Upload documents directly in chat
- **Fallback Indicators**: Visual feedback for knowledge sources used

### 🤖 8 Specialized AI Agents
- **Smart Chat**: AI router with intelligent fallbacks
- **OfferPilot**: Product search with financing
- **Dispute Copilot**: Credit card dispute assistance
- **Collections**: Hardship and payment plans
- **DevCopilot**: Code generation and API docs
- **CareCredit**: Medical expense analysis
- **Narrator**: Portfolio analytics
- **ImageGen**: AI image generation

## Environment Variables

Create `.env` file with:
```bash
GOOGLE_API_KEY=your_gemini_api_key
VISION_AGENT_API_KEY=your_landing_ai_key
TAVILY_API_KEY=your_tavily_key  # Optional
ALLOW_TAVILY=true  # Optional
```

## Architecture

- **Backend**: FastAPI with intelligent RAG and fallback systems
- **Frontend**: React + TypeScript with proper markdown rendering
- **AI**: Google Gemini with document assessment and knowledge fallbacks
- **PDF Processing**: Landing AI with bounding box extraction
- **Security**: PII redaction and TrustShield content filtering# synchrony-final
