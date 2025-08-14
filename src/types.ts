export interface Agent {
  name: string;
  icon: string;
  color: string;
  example: string;
  tooltip: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  agent: string;
  confidence?: number;
  sources?: string[];
  used_tavily?: boolean;
  fallback_used?: string | null;
  document_assessment?: DocumentAssessment | null;
  image_data?: string | null;
  image_format?: string | null;
  timestamp: Date;
  pdfData?: PdfData;
}

export interface DocumentAssessment {
  sufficient: boolean;
  confidence: number;
  reason: string;
  document_quality_score: number;
  semantic_relevance: number;
  coverage_score: number;
  answer_sufficiency: number;
}

export interface PdfData {
  filename: string;
  total_pages: number;
  chunks_extracted: number;
  processing_time: number;
  file_size: number;
}

export interface ChatRequest {
  message: string;
  allow_tavily?: boolean;
  allow_llm_knowledge?: boolean;
  allow_web_search?: boolean;
  user_type?: UserType;
}

export interface ChatResponse {
  response: string;
  agent: string;
  confidence: number;
  sources: string[];
  used_tavily: boolean;
  fallback_used?: string | null;
  document_assessment?: DocumentAssessment | null;
  image_data?: string | null;
  image_format?: string | null;
}

export interface UploadedPdf {
  pdf_id: string;
  filename: string;
  total_pages: number;
  chunks: number;
  processing_time: number;
}

export type UserType = 'consumer' | 'partner';