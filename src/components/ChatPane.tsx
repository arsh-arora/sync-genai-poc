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
  currentChatTitle: string;
}

const ChatPane: React.FC<ChatPaneProps> = ({
  messages,
  isLoading,
  inputText,
  setInputText,
  selectedAgent,
  onSendMessage,
  messagesEndRef,
  uploadPdf,
  currentChatTitle
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
      <div className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-3">
            <i className="fas fa-comments text-blue-600 text-xl"></i>
            <span className="font-semibold text-lg text-slate-800">{currentChatTitle}</span>
          </div>
          <span className="px-3 py-1 text-sm bg-slate-100 text-slate-600 rounded-full font-medium">
            {messages.length} messages
          </span>
        </div>
        <div className="flex items-center space-x-2 text-sm text-slate-500">
          <i className="fas fa-circle text-green-500"></i>
          <span className="font-medium">AI Assistant</span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto bg-gradient-to-b from-slate-50 to-white">
        <div className="p-6 space-y-6">
        {messages.length === 0 ? (
          <div className="text-center py-16">
            <div className="w-20 h-20 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <i className="fas fa-comments text-slate-400 text-3xl"></i>
            </div>
            <h3 className="text-xl font-semibold text-slate-800 mb-3">
              Start a New Conversation
            </h3>
            <p className="text-base text-slate-600 max-w-lg mx-auto mb-6 leading-relaxed">
              Ask me anything about financial services, healthcare financing, or business partnerships. I'll route your question to the right specialist.
            </p>
            <div className="bg-slate-50 rounded-xl p-4 max-w-md mx-auto">
              <p className="text-sm text-slate-500 mb-2 font-medium">Try asking:</p>
              <p className="text-sm text-slate-700 italic">"I need help financing a dental procedure for $3,500"</p>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))
        )}
        
        {isLoading && (
          <div className="flex items-start space-x-4 animate-fade-in">
            <div className="w-10 h-10 rounded-full bg-slate-100 flex items-center justify-center animate-pulse">
              <i className={`fas ${AGENTS[selectedAgent]?.icon || 'fa-robot'} ${AGENTS[selectedAgent]?.color || 'text-slate-600'}`}></i>
            </div>
            <div className="flex-1">
              <div className="bg-white border border-slate-200 rounded-lg shadow-sm overflow-hidden">
                <div className="px-4 py-3 bg-slate-50 border-b border-slate-100">
                  <span className="font-semibold text-slate-700">
                    {AGENTS[selectedAgent]?.name || 'AI Assistant'}
                  </span>
                </div>
                <div className="px-4 py-4">
                  <div className="flex items-center space-x-3">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                    </div>
                    <span className="text-sm text-slate-500">
                      Analyzing your request...
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="bg-white border-t border-slate-200 p-6">
        <div className="flex items-end space-x-4">
          <button
            onClick={handleFileUpload}
            className="p-3 text-slate-600 hover:text-slate-800 hover:bg-slate-100 rounded-xl transition-colors"
            title="Upload PDF"
          >
            <i className="fas fa-plus text-xl"></i>
          </button>
          
          <div className="flex-1">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={`Ask ${AGENTS[selectedAgent]?.name || 'AI Assistant'}...`}
              className="w-full px-5 py-4 text-base border border-slate-300 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent placeholder-slate-400"
              rows={1}
              disabled={isLoading}
            />
          </div>
          
          <button
            onClick={onSendMessage}
            disabled={!inputText.trim() || isLoading}
            className="px-5 py-4 bg-teal-600 text-white rounded-xl hover:bg-teal-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <i className="fas fa-paper-plane text-base"></i>
          </button>
        </div>
        <div className="mt-3 flex items-center justify-between text-sm text-slate-500">
          <span className="font-medium">Press Enter to send • Shift+Enter for new line</span>
          <span className="font-medium">{inputText.length} characters</span>
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