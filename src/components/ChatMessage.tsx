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
      <div className="flex items-start space-x-3">
        <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center">
          <i className={`fas ${AGENTS[message.agent]?.icon || 'fa-robot'} text-slate-600`}></i>
        </div>
        <div className="flex-1 max-w-4xl">
          {/* Message Header */}
          <div className="flex items-center space-x-2 mb-2">
            <span className="font-medium text-slate-700">
              {AGENTS[message.agent]?.name || 'AI Assistant'}
            </span>
            {message.confidence !== undefined && (
              <span className="px-2 py-0.5 text-xs bg-slate-100 text-slate-600 rounded-full">
                {(message.confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>

          {/* Agent Theater - Show execution trace */}
          {message.agent_trace && (
            <AgentTheater 
              trace={message.agent_trace} 
              isVisible={true} 
              compact={true}
              onExpandToPanel={() => {
                // This would need to be passed down from App component to work
                console.log('Switch to Agent Theater panel');
              }}
            />
          )}

          {/* Message Content */}
          <MarkdownRenderer content={message.content} />

          {/* Image Display */}
          {message.image_data && (
            <div className="mt-4 max-w-lg">
              <div className="border border-slate-200 rounded-lg overflow-hidden">
                <div className="bg-slate-50 px-3 py-2 border-b border-slate-200">
                  <span className="text-sm text-slate-600 flex items-center">
                    <i className="fas fa-image mr-2"></i>
                    Generated Image ({message.image_format})
                  </span>
                </div>
                <div className="p-4">
                  <img
                    src={`data:image/${message.image_format};base64,${message.image_data}`}
                    alt="Generated content"
                    className="max-w-full h-auto rounded"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Sources */}
          {message.sources && message.sources.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1">
              {message.sources.slice(0, 5).map((source, index) => (
                <button
                  key={index}
                  className="tool-chip text-xs bg-slate-100 hover:bg-slate-200 text-slate-600 px-2 py-1 rounded-md"
                  title={source}
                >
                  <i className="fas fa-quote-left mr-1"></i>
                  ({index + 1})
                </button>
              ))}
            </div>
          )}
          
          {/* Fallback Information */}
          {message.fallback_used && (
            <div className="mt-2 px-3 py-2 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-center space-x-2 text-sm">
                <i className="fas fa-route text-blue-600"></i>
                <span className="font-medium text-blue-800">
                  Fallback Used: {message.fallback_used.replace('_', ' ')}
                </span>
              </div>
              {message.document_assessment && (
                <div className="mt-1 text-xs text-blue-700">
                  <div>Quality Score: {(message.document_assessment.document_quality_score * 100).toFixed(0)}%</div>
                  <div>Assessment: {message.document_assessment.reason}</div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;