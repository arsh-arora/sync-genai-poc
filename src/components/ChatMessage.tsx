import React from 'react';
import { Message } from '../types';
import { AGENTS } from '../config/agents';
import MarkdownRenderer from './MarkdownRenderer';
import AgentTheater from './AgentTheater';

interface ChatMessageProps {
  message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  if (message.role === 'user') {
    return (
      <div className="chat-message animate-fade-in">
        <div className="flex items-start space-x-3 justify-end">
          <div className="max-w-3xl">
            <div className="bg-teal-600 text-white px-4 py-2 rounded-2xl rounded-tr-sm">
              <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
            </div>
            {message.pdfData && (
              <div className="mt-2 p-3 bg-red-50 border border-red-200 rounded-lg max-w-sm">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-red-100 rounded flex items-center justify-center flex-shrink-0">
                    <i className="fas fa-file-pdf text-red-600"></i>
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <i className="fas fa-file-pdf text-red-600"></i>
                      <span className="font-medium text-sm text-slate-800">
                        {message.pdfData.filename}
                      </span>
                    </div>
                    <div className="text-xs text-slate-600 space-y-1">
                      <div>📄 {message.pdfData.total_pages} pages</div>
                      <div>🧩 {message.pdfData.chunks_extracted} chunks extracted</div>
                      <div>⚡ Processed in {message.pdfData.processing_time?.toFixed(2)}s</div>
                    </div>
                    <div className="text-xs text-slate-500 mt-2">
                      Ready for questions about this document
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
          <div className="w-8 h-8 rounded-full bg-teal-600 flex items-center justify-center text-white flex-shrink-0">
            <i className="fas fa-user text-sm"></i>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-message animate-fade-in">
      <div className="flex items-start space-x-4">
        {/* Agent Avatar with Color Ring */}
        <div className="relative flex-shrink-0">
          <div className={`w-10 h-10 rounded-full ${AGENTS[message.agent]?.color?.replace('text-', 'bg-') || 'bg-slate-500'} bg-opacity-10 flex items-center justify-center ring-2 ring-white shadow-sm`}>
            <i className={`fas ${AGENTS[message.agent]?.icon || 'fa-robot'} ${AGENTS[message.agent]?.color || 'text-slate-600'} text-lg`}></i>
          </div>
          {message.confidence !== undefined && (
            <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-white rounded-full flex items-center justify-center shadow-sm border border-slate-200">
              <span className="text-xs font-medium text-slate-600">
                {(message.confidence * 100).toFixed(0)}%
              </span>
            </div>
          )}
        </div>

        <div className="flex-1 max-w-5xl">
          {/* Enhanced Message Header */}
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden hover:shadow-md transition-shadow duration-200">
            <div className={`px-4 py-3 ${AGENTS[message.agent]?.color?.replace('text-', 'bg-') || 'bg-slate-500'} bg-opacity-5 border-b border-slate-100`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <span className="font-semibold text-slate-800">
                    {AGENTS[message.agent]?.name || 'AI Assistant'}
                  </span>
                  <span className="text-xs text-slate-500">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                {message.fallback_used && (
                  <div className="flex items-center space-x-2">
                    <i className="fas fa-route text-amber-600"></i>
                    <span className="text-xs text-amber-700 font-medium">
                      Fallback: {message.fallback_used.replace('_', ' ')}
                    </span>
                  </div>
                )}
              </div>
              
              {/* Agent Theater - Show execution trace in header */}
              {message.agent_trace && (
                <div className="mt-2">
                  <AgentTheater 
                    trace={message.agent_trace} 
                    isVisible={true} 
                    compact={true}
                    onExpandToPanel={() => {
                      console.log('Switch to Agent Theater panel');
                    }}
                  />
                </div>
              )}
            </div>

            {/* Enhanced Message Content */}
            <div className="px-6 py-5">
              <div className="prose prose-slate prose-lg max-w-none">
                <MarkdownRenderer content={message.content} />
              </div>
            </div>

            {/* Image Display */}
            {message.image_data && (
              <div className="px-4 pb-4">
                <div className="border border-slate-200 rounded-lg overflow-hidden bg-slate-50">
                  <div className="px-3 py-2 border-b border-slate-200 bg-white">
                    <span className="text-sm text-slate-600 flex items-center font-medium">
                      <i className="fas fa-image mr-2 text-purple-600"></i>
                      Generated Image ({message.image_format})
                    </span>
                  </div>
                  <div className="p-4">
                    <img
                      src={`data:image/${message.image_format};base64,${message.image_data}`}
                      alt="Generated content"
                      className="max-w-full h-auto rounded shadow-sm"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Footer with Sources and Metadata */}
            <div className="px-4 py-3 bg-slate-50 border-t border-slate-100">
              <div className="flex items-center justify-between">
                {/* Sources */}
                {message.sources && message.sources.length > 0 ? (
                  <div className="flex flex-wrap gap-1">
                    <span className="text-xs text-slate-600 mr-2">Sources:</span>
                    {message.sources.slice(0, 3).map((source, index) => (
                      <button
                        key={index}
                        className="text-xs bg-white hover:bg-slate-100 text-slate-600 px-2 py-1 rounded-md border border-slate-200 transition-colors"
                        title={source}
                      >
                        <i className="fas fa-quote-left mr-1"></i>
                        {index + 1}
                      </button>
                    ))}
                    {message.sources.length > 3 && (
                      <span className="text-xs text-slate-500">
                        +{message.sources.length - 3} more
                      </span>
                    )}
                  </div>
                ) : (
                  <div className="text-xs text-slate-500">
                    <i className="fas fa-brain mr-1"></i>
                    AI Knowledge
                  </div>
                )}
                
                {/* Document Assessment */}
                {message.document_assessment && (
                  <div className="text-xs text-slate-500">
                    Quality: {(message.document_assessment.document_quality_score * 100).toFixed(0)}%
                  </div>
                )}
                
                {/* Used Tavily indicator */}
                {message.used_tavily && (
                  <div className="flex items-center space-x-1 text-xs text-blue-600">
                    <i className="fas fa-search"></i>
                    <span>Web Search</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;