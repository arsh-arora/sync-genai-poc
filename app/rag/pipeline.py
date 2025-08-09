"""
Hybrid RAG Pipeline Implementation using Haystack 2.x and Gemini API
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

from app.tools.gemini_chat import GeminiChatGenerator

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Hybrid RAG Pipeline using Haystack 2.x with BM25 + Embedding retrieval and Gemini chat"""
    
    def __init__(self):
        self.document_store = None
        self.indexing_pipeline = None
        self.retrieval_pipeline = None
        self.rag_pipeline = None
        self.chat_generator = None
        
        # Initialize Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        
    async def initialize(self):
        """Initialize the hybrid RAG pipeline components"""
        try:
            logger.info("Initializing document store...")
            self.document_store = InMemoryDocumentStore()
            
            logger.info("Building indexing pipeline...")
            self._build_indexing_pipeline()
            
            logger.info("Loading knowledge base documents...")
            await self._load_knowledge_base()
            
            logger.info("Building hybrid retrieval pipeline...")
            self._build_retrieval_pipeline()
            
            logger.info("Building RAG pipeline...")
            self._build_rag_pipeline()
            
            logger.info("Hybrid RAG pipeline initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    def _build_indexing_pipeline(self):
        """Build the document indexing pipeline"""
        # Document splitter for chunking
        document_splitter = DocumentSplitter(
            split_by="word", 
            split_length=512, 
            split_overlap=32
        )
        
        # Document embedder using sentence transformers
        document_embedder = SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"  # Lightweight model for PoC
        )
        
        # Document writer
        document_writer = DocumentWriter(self.document_store)
        
        # Build indexing pipeline
        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component("document_splitter", document_splitter)
        self.indexing_pipeline.add_component("document_embedder", document_embedder)
        self.indexing_pipeline.add_component("document_writer", document_writer)
        
        # Connect components
        self.indexing_pipeline.connect("document_splitter", "document_embedder")
        self.indexing_pipeline.connect("document_embedder", "document_writer")
    
    def _build_retrieval_pipeline(self):
        """Build the hybrid retrieval pipeline"""
        # Text embedder for query embedding
        text_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"  # Same model as document embedder
        )
        
        # Retrievers
        embedding_retriever = InMemoryEmbeddingRetriever(self.document_store)
        bm25_retriever = InMemoryBM25Retriever(self.document_store)
        
        # Document joiner to combine results
        document_joiner = DocumentJoiner()
        
        # Ranker to score and rank combined results
        ranker = TransformersSimilarityRanker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2"  # Lightweight cross-encoder
        )
        
        # Build retrieval pipeline
        self.retrieval_pipeline = Pipeline()
        self.retrieval_pipeline.add_component("text_embedder", text_embedder)
        self.retrieval_pipeline.add_component("embedding_retriever", embedding_retriever)
        self.retrieval_pipeline.add_component("bm25_retriever", bm25_retriever)
        self.retrieval_pipeline.add_component("document_joiner", document_joiner)
        self.retrieval_pipeline.add_component("ranker", ranker)
        
        # Connect components
        self.retrieval_pipeline.connect("text_embedder", "embedding_retriever")
        self.retrieval_pipeline.connect("bm25_retriever", "document_joiner")
        self.retrieval_pipeline.connect("embedding_retriever", "document_joiner")
        self.retrieval_pipeline.connect("document_joiner", "ranker")
    
    def _build_rag_pipeline(self):
        """Build the complete RAG pipeline with chat generation"""
        # Initialize Gemini chat generator
        self.chat_generator = GeminiChatGenerator(
            model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
        )
        
        # Define the prompt template
        prompt_template = [
            ChatMessage.from_system(
                "You are a helpful assistant that answers questions based on the provided documents. "
                "Use only the information from the documents to answer questions. "
                "If the information is not available in the documents, say so clearly. "
                "Provide specific references to the source documents when possible."
            ),
            ChatMessage.from_user(
                "Given these documents, answer the question.\n\n"
                "Documents:\n{% for doc in documents %}{{ doc.content }}\n"
                "Source: {{ doc.meta.get('filename', 'Unknown') }} ({{ doc.meta.get('type', 'unknown') }})\n"
                "---\n{% endfor %}\n"
                "Question: {{question}}\n\n"
                "Answer:"
            )
        ]
        
        # Create prompt builder
        prompt_builder = ChatPromptBuilder(
            template=prompt_template,
            required_variables={"question", "documents"}
        )
        
        # Build RAG pipeline
        self.rag_pipeline = Pipeline()
        
        # Add retrieval components
        self.rag_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        ))
        self.rag_pipeline.add_component("embedding_retriever", InMemoryEmbeddingRetriever(self.document_store))
        self.rag_pipeline.add_component("bm25_retriever", InMemoryBM25Retriever(self.document_store))
        self.rag_pipeline.add_component("document_joiner", DocumentJoiner())
        self.rag_pipeline.add_component("ranker", TransformersSimilarityRanker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        ))
        
        # Add generation components
        self.rag_pipeline.add_component("prompt_builder", prompt_builder)
        self.rag_pipeline.add_component("llm", self.chat_generator)
        
        # Connect retrieval components
        self.rag_pipeline.connect("text_embedder", "embedding_retriever")
        self.rag_pipeline.connect("bm25_retriever", "document_joiner")
        self.rag_pipeline.connect("embedding_retriever", "document_joiner")
        self.rag_pipeline.connect("document_joiner", "ranker")
        
        # Connect to generation
        self.rag_pipeline.connect("ranker", "prompt_builder.documents")
        self.rag_pipeline.connect("prompt_builder", "llm.messages")
    
    async def _load_knowledge_base(self):
        """Load documents from the knowledge base directories"""
        documents = []
        
        # Load from kb directory (markdown policies/terms)
        kb_path = Path("app/kb")
        if kb_path.exists():
            for file_path in kb_path.glob("*.md"):
                content = file_path.read_text(encoding="utf-8")
                doc = Document(
                    content=content,
                    meta={
                        "source": str(file_path),
                        "type": "policy",
                        "filename": file_path.name
                    }
                )
                documents.append(doc)
        
        # Load from contracts directory
        contracts_path = Path("app/contracts")
        if contracts_path.exists():
            for file_path in contracts_path.glob("*.md"):
                content = file_path.read_text(encoding="utf-8")
                doc = Document(
                    content=content,
                    meta={
                        "source": str(file_path),
                        "type": "contract",
                        "filename": file_path.name
                    }
                )
                documents.append(doc)
        
        # Load from data directory (JSON/CSV stubs)
        data_path = Path("app/data")
        if data_path.exists():
            for file_path in data_path.glob("*.json"):
                content = file_path.read_text(encoding="utf-8")
                doc = Document(
                    content=f"Data file: {file_path.name}\n\n{content}",
                    meta={
                        "source": str(file_path),
                        "type": "data",
                        "filename": file_path.name
                    }
                )
                documents.append(doc)
        
        if not documents:
            # Add some sample documents for testing
            logger.warning("No documents found in knowledge base, adding sample documents")
            documents = [
                Document(
                    content="This is a sample policy document about data privacy and user rights. "
                    "Users have the right to access, modify, and delete their personal data. "
                    "We collect minimal data necessary for service provision.",
                    meta={"source": "sample", "type": "policy", "filename": "sample_policy.md"}
                ),
                Document(
                    content="This is a sample contract document outlining merchant terms and conditions. "
                    "Merchants must comply with payment processing standards and maintain accurate records. "
                    "Commission rates vary based on transaction volume.",
                    meta={"source": "sample", "type": "contract", "filename": "sample_contract.md"}
                ),
                Document(
                    content="Sample data: {'providers': ['Provider A', 'Provider B'], 'offers': ['Offer 1', 'Offer 2'], "
                    "'metrics': {'total_transactions': 1000, 'success_rate': 0.95}}",
                    meta={"source": "sample", "type": "data", "filename": "sample_data.json"}
                )
            ]
        
        logger.info(f"Indexing {len(documents)} documents...")
        
        # Run indexing pipeline
        self.indexing_pipeline.run({"document_splitter": {"documents": documents}})
        
        logger.info(f"Successfully indexed {self.document_store.count_documents()} document chunks")
    
    async def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the hybrid RAG pipeline"""
        if not self.rag_pipeline:
            raise RuntimeError("RAG pipeline not initialized")
        
        try:
            logger.info(f"Processing query: {question}")
            
            # Run the RAG pipeline
            results = self.rag_pipeline.run({
                "text_embedder": {"text": question},
                "bm25_retriever": {"query": question, "top_k": top_k},
                "embedding_retriever": {"top_k": top_k},
                "ranker": {"query": question, "top_k": top_k},
                "prompt_builder": {"question": question}
            })
            
            # Extract response and sources
            response = results["llm"]["replies"][0].content if results["llm"]["replies"] else "No response generated"
            
            # Get source information from ranked documents
            ranked_docs = results.get("ranker", {}).get("documents", [])
            sources = [
                f"{doc.meta.get('filename', 'Unknown')} ({doc.meta.get('type', 'unknown')}) - Score: {doc.score:.3f}"
                for doc in ranked_docs[:3]  # Top 3 sources
            ]
            
            return {
                "response": response,
                "sources": sources,
                "retrieved_documents": len(ranked_docs),
                "retrieval_method": "hybrid"
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get the number of documents in the document store"""
        if not self.document_store:
            return 0
        return self.document_store.count_documents()
