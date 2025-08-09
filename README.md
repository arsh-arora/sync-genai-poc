# Synch GenAI PoC

A proof-of-concept application demonstrating intelligent knowledge base querying using hybrid retrieval (BM25 + semantic search) with Google Gemini AI and optional Tavily web search fallback.

## Features

- **Hybrid Retrieval**: Combines keyword-based (BM25) and semantic (embedding-based) search for optimal results
- **Gemini Integration**: Uses Google's Gemini 1.5 Pro for chat and gemini-embedding-001 for embeddings
- **Web Search Fallback**: Optional Tavily integration for queries not answered by the knowledge base
- **Modern UI**: Clean, responsive web interface for easy interaction
- **FastAPI Backend**: High-performance API with automatic documentation
- **In-Memory Storage**: Simple setup with no external database dependencies

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web UI        │    │   FastAPI        │    │   RAG Pipeline  │
│   (HTML/JS)     │◄──►│   Server         │◄──►│   (Haystack)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                │                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Chat Agent     │    │ Hybrid Retrieval│
                       │   (Tavily)       │    │ BM25 + Semantic │
                       └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ Knowledge Base  │
                                               │ (Policies, etc.)│
                                               └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Google AI API key (for Gemini)
- Tavily API key (optional, for web search)

### Installation & Setup

1. **Clone and navigate to the project:**
   ```bash
   cd synch-genai-poc
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the application:**
   ```bash
   ./run_local.sh
   ```

The script will automatically:
- Create a virtual environment
- Install dependencies
- Start the FastAPI server on port 8000

### Access the Application

- **Web UI**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google AI API key for Gemini | `AIza...` |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_CHAT_MODEL` | Gemini chat model | `gemini-2.5-flash` |
| `GEMINI_EMBED_MODEL` | Gemini embedding model | `gemini-embedding-001` |
| `TAVILY_API_KEY` | Tavily API key for web search | - |
| `ALLOW_TAVILY` | Enable Tavily fallback | `false` |
| `DEBUG` | Enable debug mode | `true` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Knowledge Base Structure

The application loads documents from these directories:

```
app/
├── kb/              # Markdown policies and terms
├── contracts/       # Merchant contracts (markdown)
├── data/           # JSON/CSV data files
└── openapi/        # API specifications (for reference)
```

### Adding Your Own Data

1. **Policies & Terms**: Add `.md` files to `app/kb/`
2. **Contracts**: Add `.md` files to `app/contracts/`
3. **Data Files**: Add `.json` files to `app/data/`

The system will automatically index new files on startup.

## API Endpoints

### POST /api/query

Query the knowledge base with optional web search fallback.

**Request:**
```json
{
  "query": "What are the data privacy policies?",
  "use_tavily": false
}
```

**Response:**
```json
{
  "response": "Based on our privacy policy...",
  "sources": ["privacy_policy.md (policy)"],
  "used_tavily": false,
  "retrieved_documents": 3,
  "retrieval_method": "hybrid"
}
```

### GET /api/health

Check system health and initialization status.

**Response:**
```json
{
  "status": "healthy",
  "rag_initialized": true,
  "chat_agent_initialized": true
}
```

## Technology Stack

### Core Framework
- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### AI/ML Components
- **Haystack 2.x**: RAG pipeline framework
- **Google Generative AI**: Gemini chat and embeddings
- **Sentence Transformers**: Local embeddings for hybrid retrieval
- **Tavily**: Web search API (optional)

### Data Processing
- **NumPy**: Numerical operations
- **Pandas**: Data manipulation
- **Jinja2**: Template rendering

## Development

### Project Structure

```
synch-genai-poc/
├── app/
│   ├── agents/          # Chat agents
│   ├── tools/           # Gemini integrations
│   ├── rag/            # RAG pipeline
│   ├── kb/             # Knowledge base files
│   ├── data/           # Data files
│   ├── contracts/      # Contract documents
│   └── ui/             # Web interface
├── cline_docs/         # Project documentation
├── main.py             # FastAPI application
├── requirements.txt    # Python dependencies
├── .env.example       # Environment template
├── run_local.sh       # Startup script
└── README.md          # This file
```

### Key Components

1. **RAGPipeline** (`app/rag/pipeline.py`): Hybrid retrieval using BM25 + semantic search
2. **GeminiChatGenerator** (`app/tools/gemini_chat.py`): Gemini API integration
3. **ChatAgent** (`app/agents/chat_agent.py`): Query processing with Tavily fallback
4. **Web UI** (`app/ui/index.html`): Interactive chat interface

### Running in Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GOOGLE_API_KEY="your_key_here"

# Run with auto-reload
python main.py
```

## Customization

### Adding New Retrieval Methods

Extend the `RAGPipeline` class to add new retrieval strategies:

```python
# In app/rag/pipeline.py
def _build_custom_retrieval_pipeline(self):
    # Add your custom retrieval logic
    pass
```

### Custom Chat Models

Create new chat generators by extending the base pattern:

```python
# In app/tools/
@component
class CustomChatGenerator:
    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        # Your implementation
        pass
```

### UI Customization

Modify `app/ui/index.html` to customize the interface:
- Update styles in the `<style>` section
- Modify JavaScript for new functionality
- Add new UI components as needed

## Troubleshooting

### Common Issues

1. **"Google API key is required"**
   - Ensure `GOOGLE_API_KEY` is set in your `.env` file
   - Verify the API key is valid and has Gemini API access

2. **"Failed to initialize RAG pipeline"**
   - Check if all dependencies are installed: `pip install -r requirements.txt`
   - Verify Python version is 3.8 or higher

3. **"No documents found in knowledge base"**
   - Add `.md` files to `app/kb/` or `app/contracts/`
   - The system will use sample documents if none are found

4. **Slow startup**
   - First run downloads sentence transformer models (~100MB)
   - Subsequent runs will be faster

### Performance Optimization

- Use GPU acceleration for embeddings (modify device settings in pipeline)
- Implement document caching for large knowledge bases
- Consider external vector databases for production use

## Security Considerations

- Store API keys securely (never commit `.env` files)
- Implement rate limiting for production deployments
- Validate and sanitize user inputs
- Use HTTPS in production environments

## License

This is a proof-of-concept application. Please review and comply with the terms of service for:
- Google AI/Gemini API
- Tavily API
- All open-source dependencies

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review API documentation at `/docs` when running
3. Check logs for detailed error messages

---

**Note**: This is a proof-of-concept application designed for demonstration and development purposes. For production use, consider additional security, scalability, and monitoring requirements.
# sync-genai-poc
