import React, { useState, useEffect, useRef } from 'react';
import { Message, ChatRequest, ChatResponse, UploadedPdf, UserType } from './types';
import { AGENTS } from './config/agents';
import Header from './components/Header';
import LeftRail from './components/LeftRail';
import ChatPane from './components/ChatPane';
import RightInspector from './components/RightInspector';
import LandingPage from './components/LandingPage';

function App() {
  const [userType, setUserType] = useState<UserType | null>(() => {
    const stored = localStorage.getItem('userType');
    return stored as UserType | null;
  });
  const [selectedAgent, setSelectedAgent] = useState('smart');
  const [allowTavily, setAllowTavily] = useState(false);
  const [allowLlmKnowledge, setAllowLlmKnowledge] = useState(true);
  const [allowWebSearch, setAllowWebSearch] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [rightPanel, setRightPanel] = useState('citations');
  const [citations, setCitations] = useState<string[]>([]);
  const [toolTrace, setToolTrace] = useState<any[]>([]);
  const [uploadedPdfs, setUploadedPdfs] = useState<UploadedPdf[]>([]);
  const [selectedPdfChunk, setSelectedPdfChunk] = useState(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputText,
      agent: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const endpoint = selectedAgent === 'smart' ? '/chat' : `/agent/${selectedAgent}`;
      const payload: ChatRequest | any = selectedAgent === 'smart' 
        ? { 
            message: inputText, 
            allow_tavily: allowTavily,
            allow_llm_knowledge: allowLlmKnowledge,
            allow_web_search: allowWebSearch,
            user_type: userType
          }
        : selectedAgent === 'imagegen'
        ? { prompt: inputText, include_text: true, style_hints: [], user_type: userType }
        : { query: inputText, user_type: userType };

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result: ChatResponse = await response.json();
      
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: result.response,
        agent: result.agent,
        confidence: result.confidence,
        sources: result.sources,
        used_tavily: result.used_tavily,
        fallback_used: result.fallback_used,
        document_assessment: result.document_assessment,
        image_data: result.image_data,
        image_format: result.image_format,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiMessage]);
      setCitations(result.sources || []);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`,
        agent: 'system',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const uploadPdf = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/upload/pdf', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Upload failed with status ${response.status}:`, errorText);
        throw new Error(`Upload failed: ${response.status} - ${errorText}`);
      }

      const result = await response.json();
      
      // Add PDF upload message
      const pdfMessage: Message = {
        id: Date.now().toString(),
        role: 'user',
        content: `Uploaded PDF: ${file.name}`,
        agent: 'user',
        timestamp: new Date(),
        pdfData: {
          filename: result.filename,
          total_pages: result.total_pages,
          chunks_extracted: result.chunks_extracted,
          processing_time: result.processing_time,
          file_size: result.file_size
        }
      };

      setMessages(prev => [...prev, pdfMessage]);
      setUploadedPdfs(prev => [...prev, {
        pdf_id: result.pdf_id,
        filename: result.filename,
        total_pages: result.total_pages,
        chunks: result.chunks_extracted,
        processing_time: result.processing_time
      }]);
    } catch (error) {
      console.error('PDF upload error:', error);
    }
  };

  const handleSelectAgent = (agent: string) => {
    setSelectedAgent(agent);
  };

  const handleUseExample = (agent: string) => {
    setSelectedAgent(agent);
    setInputText(AGENTS[agent]?.example || '');
  };

  const clearChat = () => {
    setMessages([]);
    setCitations([]);
    setToolTrace([]);
  };

  const exportJSON = () => {
    const lastMessage = messages[messages.length - 1];
    if (lastMessage && lastMessage.role === 'assistant') {
      const blob = new Blob([JSON.stringify(lastMessage, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `genai-response-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  const handleUserTypeSelect = (selectedUserType: UserType) => {
    setUserType(selectedUserType);
    localStorage.setItem('userType', selectedUserType);
  };

  const handleResetPersona = () => {
    setUserType(null);
    localStorage.removeItem('userType');
    setMessages([]);
    setCitations([]);
    setToolTrace([]);
  };

  // Show landing page if user type not selected
  if (!userType) {
    return <LandingPage onUserTypeSelect={handleUserTypeSelect} />;
  }

  return (
    <div className="h-screen flex flex-col bg-slate-50">
      {/* Header */}
      <Header 
        allowTavily={allowTavily}
        setAllowTavily={setAllowTavily}
        allowLlmKnowledge={allowLlmKnowledge}
        setAllowLlmKnowledge={setAllowLlmKnowledge}
        allowWebSearch={allowWebSearch}
        setAllowWebSearch={setAllowWebSearch}
        onClear={clearChat}
        onExport={exportJSON}
        userType={userType}
        onResetPersona={handleResetPersona}
      />

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Rail */}
        <LeftRail 
          selectedAgent={selectedAgent}
          onSelectAgent={handleSelectAgent}
          onUseExample={handleUseExample}
          userType={userType}
        />

        {/* Chat Pane */}
        <ChatPane 
          messages={messages}
          isLoading={isLoading}
          inputText={inputText}
          setInputText={setInputText}
          selectedAgent={selectedAgent}
          onSendMessage={sendMessage}
          messagesEndRef={messagesEndRef}
          uploadPdf={uploadPdf}
        />

        {/* Right Inspector */}
        <RightInspector 
          activePanel={rightPanel}
          setActivePanel={setRightPanel}
          citations={citations}
          toolTrace={toolTrace}
          uploadedPdfs={uploadedPdfs}
          selectedPdfChunk={selectedPdfChunk}
          setSelectedPdfChunk={setSelectedPdfChunk}
        />
      </div>
    </div>
  );
}

export default App;