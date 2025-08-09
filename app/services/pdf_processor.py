"""
PDF Processing Service with Landing AI Document Extraction
Processes uploaded PDFs and extracts structured content with bounding boxes
"""

import os
import logging
import base64
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import time

import requests
from PIL import Image
import fitz  # PyMuPDF
from haystack import Document

try:
    from agentic_doc.parse import parse
    AGENTIC_DOC_AVAILABLE = True
except ImportError:
    AGENTIC_DOC_AVAILABLE = False
    logging.warning("agentic_doc not available - falling back to REST API")

logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    """Bounding box coordinates"""
    x: float
    y: float
    width: float
    height: float
    page: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "page": self.page
        }

@dataclass
class PDFChunk:
    """Extracted PDF chunk with location and content"""
    text: str
    bbox: BoundingBox
    page_number: int
    chunk_id: str
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    def to_document(self, pdf_name: str) -> Document:
        """Convert to Haystack document"""
        return Document(
            content=self.text,
            meta={
                "source": pdf_name,
                "type": "pdf",
                "chunk_id": self.chunk_id,
                "page_number": self.page_number,
                "bbox": self.bbox.to_dict(),
                "confidence": self.confidence,
                "metadata": self.metadata or {}
            }
        )

@dataclass
class ProcessedPDF:
    """Complete processed PDF with chunks and page images"""
    filename: str
    chunks: List[PDFChunk]
    page_images: List[str]  # Base64 encoded page images
    total_pages: int
    file_size: int
    processing_time: float
    
class LandingAIPDFProcessor:
    """
    PDF processor using Landing AI's Document Extraction API
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("VISION_AGENT_API_KEY")
        if not self.api_key:
            logger.warning("Vision Agent API key not found - using fallback processing")
        
        self.base_url = "https://predict.app.landing.ai/inference/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
    
    def process_pdf(self, pdf_path: str, chunk_strategy: str = "semantic") -> ProcessedPDF:
        """
        Process PDF using Landing AI or fallback method
        
        Args:
            pdf_path: Path to PDF file
            chunk_strategy: Strategy for chunking ("semantic", "page", "paragraph")
            
        Returns:
            ProcessedPDF with extracted chunks and metadata
        """
        start_time = time.time()
        pdf_name = Path(pdf_path).name
        
        try:
            # Try Landing AI SDK first if available and API key is set
            if AGENTIC_DOC_AVAILABLE and self.api_key:
                logger.info(f"Processing {pdf_name} with Landing AI SDK...")
                result = self._process_with_agentic_doc(pdf_path, chunk_strategy)
            elif self.api_key:
                logger.info(f"Processing {pdf_name} with Landing AI REST API...")
                result = self._process_with_landing_ai(pdf_path, chunk_strategy)
            else:
                logger.info(f"Processing {pdf_name} with fallback method...")
                result = self._process_with_fallback(pdf_path, chunk_strategy)
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            logger.info(f"Successfully processed {pdf_name}: {len(result.chunks)} chunks, "
                       f"{result.total_pages} pages in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_name}: {e}")
            # Fallback to basic processing
            return self._process_with_fallback(pdf_path, chunk_strategy)
    
    def _process_with_agentic_doc(self, pdf_path: str, chunk_strategy: str) -> ProcessedPDF:
        """Process PDF using Landing AI agentic_doc SDK"""
        
        # Set API key as environment variable for the SDK
        if self.api_key:
            os.environ["VISION_AGENT_API_KEY"] = self.api_key
        
        # First, extract page images for visualization
        page_images = self._extract_page_images(pdf_path)
        
        try:
            # Parse PDF using agentic_doc
            logger.info(f"Calling agentic_doc.parse on {pdf_path}")
            result = parse(pdf_path)
            
            if not result:
                raise Exception("No results from agentic_doc.parse")
            
            # Get the first document result
            doc_result = result[0]
            
            # Extract chunks from the result
            chunks = []
            if hasattr(doc_result, 'chunks') and doc_result.chunks:
                # Open PDF once for all coordinate conversions
                pdf_doc = fitz.open(pdf_path)
                scale_factor = 2  # We render images at 2x scale
                
                try:
                    for i, chunk_data in enumerate(doc_result.chunks):
                        # Get text from chunk
                        text = chunk_data.text if hasattr(chunk_data, 'text') else str(chunk_data)
                        
                        # Extract grounding information for bounding boxes
                        if hasattr(chunk_data, 'grounding') and chunk_data.grounding:
                            ground = chunk_data.grounding[0]  # Take first grounding
                            page_num = ground.page
                            box = ground.box
                            
                            # Get page dimensions for coordinate conversion
                            if page_num < pdf_doc.page_count:
                                page = pdf_doc[page_num]
                                page_rect = page.rect
                                
                                # Convert normalized coordinates to pixel coordinates
                                # Landing AI format: l, t, r, b (normalized 0-1)
                                l = box.l
                                t = box.t
                                r = box.r
                                b = box.b
                                
                                # Convert to absolute coordinates with scaling
                                x = l * page_rect.width * scale_factor
                                y = t * page_rect.height * scale_factor
                                width = (r - l) * page_rect.width * scale_factor
                                height = (b - t) * page_rect.height * scale_factor
                                
                                bbox = BoundingBox(
                                    x=x,
                                    y=y,
                                    width=width,
                                    height=height,
                                    page=page_num
                                )
                            else:
                                # Invalid page number - use first page
                                page = pdf_doc[0]
                                page_rect = page.rect
                                bbox = BoundingBox(
                                    x=0, y=0,
                                    width=page_rect.width * scale_factor,
                                    height=page_rect.height * scale_factor,
                                    page=0
                                )
                                page_num = 0
                        else:
                            # No grounding info - use whole first page
                            page = pdf_doc[0]
                            page_rect = page.rect
                            bbox = BoundingBox(
                                x=0, y=0,
                                width=page_rect.width * scale_factor,
                                height=page_rect.height * scale_factor,
                                page=0
                            )
                            page_num = 0
                        
                        # Create chunk with proper confidence handling
                        confidence = 1.0
                        if hasattr(chunk_data, 'confidence') and chunk_data.confidence is not None:
                            confidence = float(chunk_data.confidence)
                        
                        chunk = PDFChunk(
                            text=text,
                            bbox=bbox,
                            page_number=page_num,
                            chunk_id=f"chunk_{i}",
                            confidence=confidence,
                            metadata={}
                        )
                        chunks.append(chunk)
                
                finally:
                    pdf_doc.close()
            
            # Get total pages from the PDF
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            doc.close()
            
            return ProcessedPDF(
                filename=Path(pdf_path).name,
                chunks=chunks,
                page_images=page_images,
                total_pages=total_pages,
                file_size=os.path.getsize(pdf_path),
                processing_time=0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"agentic_doc processing failed: {e}")
            raise

    def _process_with_landing_ai(self, pdf_path: str, chunk_strategy: str) -> ProcessedPDF:
        """Process PDF using Landing AI API"""
        
        # First, extract page images for visualization
        page_images = self._extract_page_images(pdf_path)
        
        # Encode PDF for API
        with open(pdf_path, 'rb') as f:
            pdf_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Landing AI Document Extraction API call
        payload = {
            "file_content": pdf_base64,
            "file_type": "pdf",
            "extraction_config": {
                "chunk_strategy": chunk_strategy,
                "include_bboxes": True,
                "include_confidence": True,
                "page_images": True
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/documents/extract",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_landing_ai_response(data, page_images, pdf_path)
            else:
                logger.error(f"Landing AI API error: {response.status_code} - {response.text}")
                raise Exception(f"API returned {response.status_code}")
                
        except Exception as e:
            logger.error(f"Landing AI processing failed: {e}")
            raise
    
    def _process_with_fallback(self, pdf_path: str, chunk_strategy: str) -> ProcessedPDF:
        """Fallback processing using PyMuPDF"""
        
        doc = fitz.open(pdf_path)
        chunks = []
        page_images = []
        
        try:
            # Extract page images
            for page_num in range(doc.page_count):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better quality
                img_data = pix.tobytes("png")
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                page_images.append(img_base64)
            
            # Extract text chunks based on strategy
            if chunk_strategy == "page":
                chunks = self._chunk_by_page(doc)
            elif chunk_strategy == "paragraph":
                chunks = self._chunk_by_paragraph(doc)
            else:  # semantic (default)
                chunks = self._chunk_semantically(doc)
            
            return ProcessedPDF(
                filename=Path(pdf_path).name,
                chunks=chunks,
                page_images=page_images,
                total_pages=doc.page_count,
                file_size=os.path.getsize(pdf_path),
                processing_time=0  # Will be set by caller
            )
            
        finally:
            doc.close()
    
    def _extract_page_images(self, pdf_path: str) -> List[str]:
        """Extract page images as base64 strings"""
        doc = fitz.open(pdf_path)
        images = []
        
        try:
            for page_num in range(doc.page_count):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                images.append(img_base64)
        finally:
            doc.close()
            
        return images
    
    def _parse_landing_ai_response(self, data: Dict[str, Any], 
                                  page_images: List[str], pdf_path: str) -> ProcessedPDF:
        """Parse Landing AI API response into ProcessedPDF"""
        
        chunks = []
        extracted_chunks = data.get("chunks", [])
        
        # Get page dimensions for coordinate conversion
        doc = fitz.open(pdf_path)
        
        try:
            for i, chunk_data in enumerate(extracted_chunks):
                # Landing AI uses normalized coordinates in 'grounding' field
                grounding = chunk_data.get("grounding", [])
                if not grounding:
                    continue
                
                ground = grounding[0]  # Take first grounding
                page_num = ground.get("page", 0)
                box = ground.get("box", {})
                
                # Get page dimensions (PDF uses 2x scale for images)
                page = doc[page_num]
                page_rect = page.rect
                scale_factor = 2  # We render images at 2x scale
                
                # Convert normalized coordinates to pixel coordinates
                # Landing AI format: l, t, r, b (normalized 0-1)
                l = box.get("l", 0)  # Left
                t = box.get("t", 0)  # Top
                r = box.get("r", 0)  # Right  
                b = box.get("b", 0)  # Bottom
                
                # Convert to absolute coordinates with scaling
                x = l * page_rect.width * scale_factor
                y = t * page_rect.height * scale_factor
                width = (r - l) * page_rect.width * scale_factor
                height = (b - t) * page_rect.height * scale_factor
                
                bbox = BoundingBox(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    page=page_num
                )
                
                chunk = PDFChunk(
                    text=chunk_data.get("text", ""),
                    bbox=bbox,
                    page_number=page_num,
                    chunk_id=f"chunk_{i}",
                    confidence=chunk_data.get("confidence", 1.0),
                    metadata=chunk_data.get("metadata", {})
                )
                chunks.append(chunk)
        
        finally:
            doc.close()
        
        return ProcessedPDF(
            filename=Path(pdf_path).name,
            chunks=chunks,
            page_images=page_images,
            total_pages=data.get("total_pages", len(page_images)),
            file_size=os.path.getsize(pdf_path),
            processing_time=0
        )
    
    def _chunk_by_page(self, doc: fitz.Document) -> List[PDFChunk]:
        """Chunk PDF by page"""
        chunks = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                # Get page dimensions for bounding box
                rect = page.rect
                bbox = BoundingBox(
                    x=0, y=0,
                    width=rect.width,
                    height=rect.height,
                    page=page_num
                )
                
                chunk = PDFChunk(
                    text=text.strip(),
                    bbox=bbox,
                    page_number=page_num,
                    chunk_id=f"page_{page_num}",
                    confidence=1.0
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_paragraph(self, doc: fitz.Document) -> List[PDFChunk]:
        """Chunk PDF by paragraphs using text blocks"""
        chunks = []
        chunk_id = 0
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:  # Text block
                    text_parts = []
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_parts.append(span["text"])
                    
                    text = "".join(text_parts).strip()
                    if text and len(text) > 20:  # Minimum length filter
                        
                        # Create bounding box from block (apply 2x scale to match image rendering)
                        scale_factor = 2
                        bbox = BoundingBox(
                            x=block["bbox"][0] * scale_factor,
                            y=block["bbox"][1] * scale_factor,
                            width=(block["bbox"][2] - block["bbox"][0]) * scale_factor,
                            height=(block["bbox"][3] - block["bbox"][1]) * scale_factor,
                            page=page_num
                        )
                        
                        chunk = PDFChunk(
                            text=text,
                            bbox=bbox,
                            page_number=page_num,
                            chunk_id=f"para_{chunk_id}",
                            confidence=0.9
                        )
                        chunks.append(chunk)
                        chunk_id += 1
        
        return chunks
    
    def _chunk_semantically(self, doc: fitz.Document) -> List[PDFChunk]:
        """Semantic chunking - group related paragraphs"""
        # For now, use paragraph chunking with larger minimum sizes
        paragraphs = self._chunk_by_paragraph(doc)
        
        # Merge small adjacent paragraphs
        merged_chunks = []
        current_chunk = None
        
        for para in paragraphs:
            if current_chunk is None:
                current_chunk = para
            elif (len(current_chunk.text) < 300 and 
                  para.page_number == current_chunk.page_number):
                # Merge with previous chunk
                current_chunk.text += "\n\n" + para.text
                # Expand bounding box
                current_chunk.bbox.width = max(
                    current_chunk.bbox.x + current_chunk.bbox.width,
                    para.bbox.x + para.bbox.width
                ) - current_chunk.bbox.x
                current_chunk.bbox.height = max(
                    current_chunk.bbox.y + current_chunk.bbox.height,
                    para.bbox.y + para.bbox.height
                ) - current_chunk.bbox.y
            else:
                merged_chunks.append(current_chunk)
                current_chunk = para
        
        if current_chunk:
            merged_chunks.append(current_chunk)
        
        return merged_chunks

def save_processed_pdf(processed_pdf: ProcessedPDF, 
                      output_dir: str = "processed_pdfs") -> str:
    """Save processed PDF data to disk for caching"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from original
    base_name = Path(processed_pdf.filename).stem
    output_file = Path(output_dir) / f"{base_name}_processed.json"
    
    # Convert to serializable format
    data = {
        "filename": processed_pdf.filename,
        "chunks": [
            {
                "text": chunk.text,
                "bbox": chunk.bbox.to_dict(),
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "confidence": chunk.confidence,
                "metadata": chunk.metadata
            }
            for chunk in processed_pdf.chunks
        ],
        "page_images": processed_pdf.page_images,
        "total_pages": processed_pdf.total_pages,
        "file_size": processed_pdf.file_size,
        "processing_time": processed_pdf.processing_time
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved processed PDF to {output_file}")
    return str(output_file)

# Test function
def test_pdf_processor():
    """Test PDF processing with sample file"""
    print("🔍 Testing PDF Processor")
    print("=" * 40)
    
    processor = LandingAIPDFProcessor()
    
    # Create a simple test PDF if none exists
    test_file = "test_document.pdf"
    
    try:
        result = processor.process_pdf(test_file, chunk_strategy="semantic")
        
        print(f"✅ Processed: {result.filename}")
        print(f"   Pages: {result.total_pages}")
        print(f"   Chunks: {len(result.chunks)}")
        print(f"   Processing time: {result.processing_time:.2f}s")
        
        for i, chunk in enumerate(result.chunks[:3]):
            print(f"   Chunk {i}: {chunk.text[:50]}...")
            print(f"     Page: {chunk.page_number}, BBox: {chunk.bbox.to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_pdf_processor()