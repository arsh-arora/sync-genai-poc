import React, { useState, useRef } from 'react';
import { Message } from '../types';
import { AGENTS } from '../config/agents';
import ChatMessage from './ChatMessage';

interface ChatPaneProps {
  messages: Message[];
  isLoading: boolean;
  inputText: string;
  setInputText: (text: string) => void;
  selectedAgent: string;
  onSendMessage: () => void;
  messagesEndRef: React.RefObject<HTMLDivElement>;
  uploadPdf: (file: File) => void;
}

const ChatPane: React.FC<ChatPaneProps> = ({
  messages,
  isLoading,
  inputText,
  setInputText,
  selectedAgent,
  onSendMessage,
  messagesEndRef,
  uploadPdf
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      if (e.shiftKey || e.ctrlKey) {
        // Allow new line on Shift+Enter or Ctrl+Enter
        return;
      } else {
        // Send message on plain Enter
        e.preventDefault();
        if (inputText.trim()) {
          onSendMessage();
        }
      }
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    const files = Array.from(e.dataTransfer.files);
    console.log('Dropped files:', files.map(f => ({ name: f.name, type: f.type, size: f.size })));
    
    const pdfFiles = files.filter(file => 
      file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
    );

    if (pdfFiles.length === 0) {
      alert('Please drop PDF files only.');
      return;
    }

    // Upload each PDF file
    for (const file of pdfFiles) {
      try {
        console.log(`Uploading PDF: ${file.name}`);
        await uploadPdf(file);
        console.log(`Successfully uploaded: ${file.name}`);
      } catch (error) {
        console.error(`Error uploading PDF ${file.name}:`, error);
      }
    }
  };

  const handleFileUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      uploadPdf(file);
    }
  };

  return (
    <div 
      className="flex-1 flex flex-col bg-white relative"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag overlay */}
      {isDragOver && (
        <div className="absolute inset-0 bg-teal-50 border-2 border-dashed border-teal-300 flex items-center justify-center z-10">
          <div className="text-center">
            <i className="fas fa-file-pdf text-4xl text-teal-600 mb-4"></i>
            <p className="text-teal-800 font-medium">Drop PDF files here</p>
          </div>
        </div>
      )}

      {/* Chat Header */}
      <div className="bg-white border-b border-slate-200 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <i className={`fas ${AGENTS[selectedAgent]?.icon || 'fa-robot'} ${AGENTS[selectedAgent]?.color || 'text-slate-600'}`}></i>
            <span className="font-medium text-slate-800">{AGENTS[selectedAgent]?.name || 'AI Assistant'}</span>
          </div>
          <span className="px-2 py-1 text-xs bg-slate-100 text-slate-600 rounded-md">
            {messages.length} messages
          </span>
        </div>
        <div className="flex items-center space-x-2 text-sm text-slate-500">
          <i className="fas fa-circle text-green-500"></i>
          <span>Online</span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 ? (
          <div className="text-center py-12">
            <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <i className={`fas ${AGENTS[selectedAgent]?.icon || 'fa-robot'} ${AGENTS[selectedAgent]?.color || 'text-slate-400'} text-2xl`}></i>
            </div>
            <h3 className="text-lg font-medium text-slate-800 mb-2">
              Chat with {AGENTS[selectedAgent]?.name || 'AI Assistant'}
            </h3>
            <p className="text-slate-600 max-w-md mx-auto mb-4">
              {AGENTS[selectedAgent]?.tooltip || 'Start a conversation by typing your message below.'}
            </p>
            <div className="text-sm text-slate-500">
              <p>Try: "{AGENTS[selectedAgent]?.example || 'Hello, how can you help me?'}"</p>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))
        )}
        
        {isLoading && (
          <div className="flex items-center space-x-3 animate-fade-in">
            <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center">
              <i className={`fas ${AGENTS[selectedAgent]?.icon || 'fa-robot'} text-slate-600`}></i>
            </div>
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
              <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="bg-white border-t border-slate-200 p-4">
        <div className="flex items-end space-x-3">
          <button
            onClick={handleFileUpload}
            className="p-2 text-slate-600 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition-colors"
            title="Upload PDF"
          >
            <i className="fas fa-plus text-lg"></i>
          </button>
          
          <div className="flex-1">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={`Ask to ${AGENTS[selectedAgent]?.name || 'AI Assistant'}...`}
              className="w-full px-4 py-3 border border-slate-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent"
              rows={1}
              disabled={isLoading}
            />
          </div>
          
          <button
            onClick={onSendMessage}
            disabled={!inputText.trim() || isLoading}
            className="px-4 py-3 bg-teal-600 text-white rounded-lg hover:bg-teal-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <i className="fas fa-paper-plane"></i>
          </button>
        </div>
        <div className="mt-2 flex items-center justify-between text-xs text-slate-500">
          <span>Use Tavily • Shift+Enter for new line</span>
          <span>0 chars • Shift+Enter to send</span>
        </div>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
};

export default ChatPane;