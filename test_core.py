#!/usr/bin/env python3
"""
Simple test script for the core RAG functionality
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.rag.core import init_docstore, GeminiEmbedder, index_markdown, build_retriever, retrieve
from app.llm.gemini import chat_with_context

def test_core_functionality():
    """Test the core RAG functionality"""
    print("🧪 Testing Core RAG Functionality")
    print("=" * 50)
    
    # Check environment variables
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not set. Please set it in your .env file.")
        return False
    
    try:
        # Initialize components
        print("1. Initializing document store...")
        docstore = init_docstore()
        print("✅ Document store initialized")
        
        print("2. Initializing Gemini embedder...")
        embedder = GeminiEmbedder()
        print("✅ Gemini embedder initialized")
        
        print("3. Indexing markdown documents...")
        doc_count = index_markdown(docstore, embedder)
        print(f"✅ Indexed {doc_count} document chunks")
        
        if doc_count == 0:
            print("⚠️  No documents found. Make sure you have .md files in app/kb/ or app/contracts/")
            return False
        
        print("4. Building retriever...")
        retriever = build_retriever(docstore)
        print("✅ Retriever built")
        
        # Test retrieval
        print("5. Testing retrieval...")
        test_query = "What are the data privacy policies?"
        retrieved_docs = retrieve(retriever, embedder, test_query, k=3)
        print(f"✅ Retrieved {len(retrieved_docs)} documents for test query")
        
        if retrieved_docs:
            print("\nTop retrieved document:")
            top_doc = retrieved_docs[0]
            print(f"  Source: {top_doc['filename']}")
            print(f"  Type: {top_doc['type']}")
            print(f"  Score: {top_doc['score']:.3f}")
            print(f"  Snippet: {top_doc['snippet'][:200]}...")
        
        # Test chat generation
        print("\n6. Testing chat generation...")
        if retrieved_docs:
            response = chat_with_context(test_query, retrieved_docs[:2])
            print("✅ Generated chat response")
            print(f"\nResponse preview: {response[:300]}...")
        
        print("\n🎉 All tests passed! Core functionality is working.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    success = test_core_functionality()
    sys.exit(0 if success else 1)
