"""
Core RAG functionality with Gemini embeddings and Haystack integration
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from google import genai
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

logger = logging.getLogger(__name__)

class WebDisabled(Exception):
    """Raised when web search is disabled but requested"""
    pass

def init_docstore() -> InMemoryDocumentStore:
    """Initialize Haystack InMemoryDocumentStore"""
    return InMemoryDocumentStore()

class GeminiEmbedder:
    """Gemini embeddings API wrapper with batching and retry logic"""
    
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key is required")
        
        self.client = genai.Client(api_key=self.api_key)
        logger.info(f"Initialized GeminiEmbedder with model: {self.model}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 10, max_retries: int = 3) -> List[List[float]]:
        """
        Embed a list of texts using Gemini embeddings API
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of embedding vectors, same length as input texts
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._embed_batch_with_retry(batch, max_retries)
            all_embeddings.extend(batch_embeddings)
        
        logger.info(f"Generated embeddings for {len(texts)} texts")
        return all_embeddings
    
    def _embed_batch_with_retry(self, batch: List[str], max_retries: int) -> List[List[float]]:
        """Embed a batch of texts with retry logic"""
        for attempt in range(max_retries + 1):
            try:
                # Use modern Gemini embeddings API
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=batch
                )
                
                # Extract embeddings from response
                if hasattr(result, 'embeddings'):
                    embeddings = []
                    for embedding in result.embeddings:
                        if hasattr(embedding, 'values'):
                            embeddings.append(embedding.values)
                        else:
                            embeddings.append(embedding)
                    return embeddings
                else:
                    # Fallback - return zero vectors
                    return [[0.0] * 768 for _ in batch]
                
            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Embedding attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate embeddings after {max_retries + 1} attempts: {e}")
                    # Return zero vectors as fallback
                    return [[0.0] * 768 for _ in batch]  # Default dimension
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query text"""
        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=query
            )
            
            if hasattr(result, 'embeddings') and result.embeddings:
                # Get first embedding for single query
                embedding = result.embeddings[0]
                if hasattr(embedding, 'values'):
                    return embedding.values
                else:
                    return embedding
            else:
                # Fallback
                return [0.0] * 768
                
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return [0.0] * 768  # Default dimension fallback

def _count_tokens(text: str) -> int:
    """Rough token count estimation (1 token ≈ 4 characters)"""
    return len(text) // 4

def _split_text_into_chunks(text: str, max_tokens: int = 1500, overlap_tokens: int = 150) -> List[Dict[str, Any]]:
    """
    Split text into chunks with token limits and overlap
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks
        
    Returns:
        List of chunk dictionaries with text and line span info
    """
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_tokens = 0
    start_line = 0
    
    for i, line in enumerate(lines):
        line_tokens = _count_tokens(line)
        
        # If adding this line would exceed max tokens, finalize current chunk
        if current_tokens + line_tokens > max_tokens and current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start_line': start_line,
                'end_line': i - 1,
                'token_count': current_tokens
            })
            
            # Start new chunk with overlap
            overlap_lines = []
            overlap_tokens = 0
            
            # Add lines from the end of current chunk for overlap
            for j in range(len(current_chunk) - 1, -1, -1):
                line_overlap_tokens = _count_tokens(current_chunk[j])
                if overlap_tokens + line_overlap_tokens <= overlap_tokens:
                    overlap_lines.insert(0, current_chunk[j])
                    overlap_tokens += line_overlap_tokens
                else:
                    break
            
            current_chunk = overlap_lines + [line]
            current_tokens = overlap_tokens + line_tokens
            start_line = max(0, i - len(overlap_lines))
        else:
            current_chunk.append(line)
            current_tokens += line_tokens
            if not current_chunk or len(current_chunk) == 1:
                start_line = i
    
    # Add final chunk if it exists
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'start_line': start_line,
            'end_line': len(lines) - 1,
            'token_count': current_tokens
        })
    
    return chunks

def index_markdown(docstore: InMemoryDocumentStore, embedder: GeminiEmbedder, base_dir: str = "app") -> int:
    """
    Index markdown files from /kb and /contracts directories
    
    Args:
        docstore: Haystack document store
        embedder: Gemini embedder instance
        base_dir: Base directory path
        
    Returns:
        Number of documents indexed
    """
    base_path = Path(base_dir)
    kb_path = base_path / "kb"
    contracts_path = base_path / "contracts"
    
    documents = []
    
    # Process both directories
    for dir_path in [kb_path, contracts_path]:
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            continue
            
        for md_file in dir_path.glob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                file_type = "policy" if dir_path.name == "kb" else "contract"
                
                # Split into chunks
                chunks = _split_text_into_chunks(content)
                
                for i, chunk in enumerate(chunks):
                    doc_id = f"{md_file.stem}_chunk_{i}"
                    
                    doc = Document(
                        content=chunk['text'],
                        meta={
                            "source": str(md_file),
                            "filename": md_file.name,
                            "type": file_type,
                            "chunk_id": i,
                            "start_line": chunk['start_line'],
                            "end_line": chunk['end_line'],
                            "token_count": chunk['token_count']
                        }
                    )
                    documents.append(doc)
                    
            except Exception as e:
                logger.error(f"Error processing {md_file}: {e}")
    
    if not documents:
        logger.warning("No documents found to index")
        return 0
    
    # Generate embeddings for all documents
    logger.info(f"Generating embeddings for {len(documents)} document chunks...")
    texts = [doc.content for doc in documents]
    embeddings = embedder.embed_texts(texts)
    
    # Add embeddings to documents
    for doc, embedding in zip(documents, embeddings):
        doc.embedding = embedding
    
    # Write to document store
    docstore.write_documents(documents)
    
    logger.info(f"Successfully indexed {len(documents)} document chunks")
    return len(documents)

def build_retriever(docstore: InMemoryDocumentStore) -> InMemoryEmbeddingRetriever:
    """Build Haystack InMemoryEmbeddingRetriever"""
    return InMemoryEmbeddingRetriever(document_store=docstore)

def retrieve(retriever: InMemoryEmbeddingRetriever, embedder: GeminiEmbedder, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents for a query with enhanced citation information
    
    Args:
        retriever: Haystack embedding retriever
        embedder: Gemini embedder for query embedding
        query: Search query
        k: Number of results to return
        
    Returns:
        List of dictionaries with source, snippet, score, and citation details
    """
    try:
        # Embed the query
        query_embedding = embedder.embed_query(query)
        
        # Retrieve documents
        result = retriever.run(query_embedding=query_embedding, top_k=k)
        documents = result.get("documents", [])
        
        # Format results with enhanced citation information
        formatted_results = []
        for doc in documents:
            # Get full content - don't truncate for citations
            content = doc.content if doc.content else ""
            
            # Create enhanced citation with rule/passage details
            citation_result = {
                "source": doc.meta.get("source", "unknown"),
                "snippet": content,  # Full content, not truncated
                "score": getattr(doc, 'score', 0.0),
                "filename": doc.meta.get("filename", "unknown"),
                "type": doc.meta.get("type", "unknown"),
                "chunk_id": doc.meta.get("chunk_id", 0),
                "line_span": f"{doc.meta.get('start_line', 0)}-{doc.meta.get('end_line', 0)}",
                # Enhanced citation fields
                "citation_title": _extract_citation_title(content, doc.meta.get("filename", "unknown")),
                "rule_type": _determine_rule_type(doc.meta.get("filename", ""), doc.meta.get("type", "")),
                "content_preview": _create_content_preview(content),
                "relevance_score": float(getattr(doc, 'score', 0.0))
            }
            formatted_results.append(citation_result)
        
        logger.info(f"Retrieved {len(formatted_results)} documents for query: {query}")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return []

def _extract_citation_title(content: str, filename: str) -> str:
    """Extract a meaningful title from content or filename"""
    if not content:
        return filename.replace('.md', '').replace('_', ' ').title()
    
    # Look for markdown headers
    lines = content.split('\n')
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if line.startswith('# '):
            return line[2:].strip()
        elif line.startswith('## '):
            return line[3:].strip()
    
    # Fallback to filename
    return filename.replace('.md', '').replace('_', ' ').title()

def _determine_rule_type(filename: str, doc_type: str) -> str:
    """Determine the type of rule/policy based on filename and type"""
    filename_lower = filename.lower()
    
    if 'promotion' in filename_lower or 'offer' in filename_lower:
        return "Promotional Terms"
    elif 'privacy' in filename_lower:
        return "Privacy Policy"
    elif 'security' in filename_lower:
        return "Security Guidelines"
    elif 'dispute' in filename_lower:
        return "Dispute Policy"
    elif 'contract' in filename_lower:
        return "Contract Terms"
    elif 'trustshield' in filename_lower:
        return "Security Rules"
    elif 'collection' in filename_lower:
        return "Collections Policy"
    elif doc_type == "policy":
        return "Policy Document"
    elif doc_type == "contract":
        return "Contract Document"
    elif doc_type == "data":
        return "Data Reference"
    else:
        return "Knowledge Base"

def _create_content_preview(content: str, max_length: int = 150) -> str:
    """Create a preview of the content for display"""
    if not content:
        return ""
    
    # Clean up content
    clean_content = content.replace('\n', ' ').replace('\t', ' ')
    # Remove extra whitespace
    clean_content = ' '.join(clean_content.split())
    
    if len(clean_content) <= max_length:
        return clean_content
    
    # Truncate at word boundary
    truncated = clean_content[:max_length]
    last_space = truncated.rfind(' ')
    if last_space > 0:
        truncated = truncated[:last_space]
    
    return truncated + "..."
