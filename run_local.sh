#!/bin/bash

# Synch GenAI PoC - Local Development Server
# This script starts the FastAPI server on port 8000

echo "🚀 Starting Synch GenAI PoC..."
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if [ ! -f "venv/pyvenv.cfg" ] || [ ! -d "venv/lib/python*/site-packages/fastapi" ]; then
    echo "📦 Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Please copy .env.example to .env and configure your API keys:"
    echo "   cp .env.example .env"
    echo "   # Then edit .env with your API keys"
    echo ""
    echo "Required environment variables:"
    echo "  - GOOGLE_API_KEY: Your Google AI API key"
    echo "  - TAVILY_API_KEY: Your Tavily API key (optional)"
    echo ""
    read -p "Do you want to continue without .env file? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Exiting. Please configure .env file first."
        exit 1
    fi
fi

# Set default environment variables if not set
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-8000}
export DEBUG=${DEBUG:-"true"}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}

echo ""
echo "🌟 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Debug: $DEBUG"
echo "   Log Level: $LOG_LEVEL"
echo ""

# Start the server
echo "🎯 Starting FastAPI server..."
echo "   Access the application at: http://localhost:$PORT"
echo "   API documentation at: http://localhost:$PORT/docs"
echo "   Health check at: http://localhost:$PORT/api/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================"

# Run the application
python main.py
