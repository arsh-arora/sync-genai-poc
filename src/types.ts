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
  agent_trace?: AgentTrace;
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
  agent_trace?: AgentTrace;
}

export interface UploadedPdf {
  pdf_id: string;
  filename: string;
  total_pages: number;
  chunks: number;
  processing_time: number;
}

export type UserType = 'consumer' | 'partner';

export interface PersonaDetectionResponse {
  persona: UserType;
  confidence: number;
  reasoning: string;
  available_agents: string[];
  is_confident: boolean;
  error?: string;
}

export interface ChatHistory {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
  persona?: UserType;
}

// Agent Theater Types
export interface ToolCall {
  tool_name: string;
  duration_ms: number;
  status: 'success' | 'error' | 'timeout';
  input_summary?: string;
  output_summary?: string;
  metadata?: Record<string, any>;
}

export interface AgentExecution {
  agent_name: string;
  display_name: string;
  started_at: number;
  duration_ms: number;
  status: 'success' | 'error' | 'timeout';
  tool_calls: ToolCall[];
  input_summary?: string;
  output_summary?: string;
  confidence?: number;
  sources_used?: string[];
}

export interface AgentTrace {
  execution_id: string;
  total_duration_ms: number;
  agent_executions: AgentExecution[];
  routing_decision?: {
    primary_agent: string;
    confidence: number;
    fallback_used?: string;
  };
}