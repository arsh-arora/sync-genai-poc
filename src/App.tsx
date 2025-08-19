import React, { useState, useEffect, useRef } from 'react';
import { Message, ChatRequest, ChatResponse, UploadedPdf, UserType, PersonaDetectionResponse, ChatHistory } from './types';
import { AGENTS } from './config/agents';
import Header from './components/Header';
import LeftRail from './components/LeftRail';
import ChatPane from './components/ChatPane';
import RightInspector from './components/RightInspector';

function App() {
  // Chat History Management
  const [chatHistories, setChatHistories] = useState<ChatHistory[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [rightPanelCollapsed, setRightPanelCollapsed] = useState(false);
  
  // Current Chat State
  const [userType, setUserType] = useState<UserType>('consumer');
  const [selectedAgent, setSelectedAgent] = useState('smart');
  const [availableAgents, setAvailableAgents] = useState<string[]>([]);
  const [personaDetected, setPersonaDetected] = useState(false);
  const [personaConfidence, setPersonaConfidence] = useState(0);
  const [allowTavily, setAllowTavily] = useState(false);
  const [allowLlmKnowledge, setAllowLlmKnowledge] = useState(true);
  const [allowWebSearch, setAllowWebSearch] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [rightPanel, setRightPanel] = useState('citations');
  const [citations, setCitations] = useState<(string | import('./types').Citation)[]>([]);
  const [toolTrace, setToolTrace] = useState<any[]>([]);
  const [agentTrace, setAgentTrace] = useState<any>(null);
  const [uploadedPdfs, setUploadedPdfs] = useState<UploadedPdf[]>([]);
  const [selectedPdfChunk, setSelectedPdfChunk] = useState(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Get current chat
  const currentChat = currentChatId ? chatHistories.find(chat => chat.id === currentChatId) : null;

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load chat history from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('chatHistories');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        const histories = parsed.map((h: any) => ({
          ...h,
          createdAt: new Date(h.createdAt),
          updatedAt: new Date(h.updatedAt),
          messages: h.messages.map((m: any) => ({
            ...m,
            timestamp: new Date(m.timestamp)
          }))
        }));
        setChatHistories(histories);
      } catch (e) {
        console.error('Failed to load chat histories:', e);
      }
    }
  }, []);

  // Save chat histories to localStorage whenever they change
  useEffect(() => {
    if (chatHistories.length > 0) {
      localStorage.setItem('chatHistories', JSON.stringify(chatHistories));
    }
  }, [chatHistories]);

  // Update current chat messages when messages change
  useEffect(() => {
    if (currentChatId && messages.length > 0) {
      setChatHistories(prev => prev.map(chat => 
        chat.id === currentChatId 
          ? { ...chat, messages: [...messages], updatedAt: new Date() }
          : chat
      ));
    }
  }, [messages, currentChatId]);

  // Update chat history persona when userType changes
  useEffect(() => {
    if (currentChatId && personaDetected) {
      setChatHistories(prev => prev.map(chat => 
        chat.id === currentChatId 
          ? { ...chat, persona: userType, updatedAt: new Date() }
          : chat
      ));
    }
  }, [userType, currentChatId, personaDetected]);

  // Chat Management Functions
  const createNewChat = () => {
    const newChatId = Date.now().toString();
    const newChat: ChatHistory = {
      id: newChatId,
      title: 'New Chat',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
      persona: userType
    };
    
    setChatHistories(prev => [newChat, ...prev]);
    setCurrentChatId(newChatId);
    
    // Reset current chat state
    setMessages([]);
    setCitations([]);
    setToolTrace([]);
    setAgentTrace(null);
    setPersonaDetected(false);
    setPersonaConfidence(0);
    setAvailableAgents([]);
  };

  const selectChat = (chatId: string) => {
    const chat = chatHistories.find(c => c.id === chatId);
    if (chat) {
      setCurrentChatId(chatId);
      setMessages(chat.messages);
      setUserType(chat.persona || 'consumer');
      
      // Reset other state
      setCitations([]);
      setToolTrace([]);
      setAgentTrace(null);
      
      // Extract citations and traces from last message if available
      const lastMessage = chat.messages[chat.messages.length - 1];
      if (lastMessage?.role === 'assistant') {
        setCitations(lastMessage.sources || []);
        setAgentTrace(lastMessage.agent_trace || null);
      }
    }
  };

  const deleteChat = (chatId: string) => {
    setChatHistories(prev => prev.filter(chat => chat.id !== chatId));
    if (currentChatId === chatId) {
      setCurrentChatId(null);
      setMessages([]);
      setCitations([]);
      setToolTrace([]);
      setAgentTrace(null);
    }
  };

  const updateChatTitle = (chatId: string, title: string) => {
    setChatHistories(prev => prev.map(chat => 
      chat.id === chatId 
        ? { ...chat, title, updatedAt: new Date() }
        : chat
    ));
  };

  const generateChatTitle = async (message: string, chatId: string) => {
    try {
      const response = await fetch('/generate-chat-title', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });

      if (response.ok) {
        const result = await response.json();
        updateChatTitle(chatId, result.title);
      } else {
        // Fallback to truncated message
        const fallbackTitle = message.length > 50 ? message.substring(0, 47) + '...' : message;
        updateChatTitle(chatId, fallbackTitle);
      }
    } catch (error) {
      console.error('Failed to generate chat title:', error);
      // Fallback to truncated message
      const fallbackTitle = message.length > 50 ? message.substring(0, 47) + '...' : message;
      updateChatTitle(chatId, fallbackTitle);
    }
  };

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    // Create new chat if none exists
    if (!currentChatId) {
      createNewChat();
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputText,
      agent: 'user',
      timestamp: new Date()
    };

    // Auto-generate chat title from first message using LLM
    if (messages.length === 0 && currentChatId) {
      generateChatTitle(inputText, currentChatId);
    }

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      // Detect persona if not already detected
      if (!personaDetected && messages.length === 0) {
        try {
          const personaResponse = await fetch('/detect-persona-and-agents', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              message: inputText,
              allow_tavily: allowTavily,
              allow_llm_knowledge: allowLlmKnowledge,
              allow_web_search: allowWebSearch,
              user_type: userType
            })
          });

          if (personaResponse.ok) {
            const personaData: PersonaDetectionResponse = await personaResponse.json();
            setUserType(personaData.persona);
            setAvailableAgents(personaData.available_agents);
            setPersonaDetected(true);
            setPersonaConfidence(personaData.confidence);
          }
        } catch (personaError) {
          console.warn('Persona detection failed:', personaError);
          // Continue with default settings
        }
      }
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
        agent_trace: result.agent_trace,  // 🎭 Add the missing agent_trace!
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiMessage]);
      setCitations(result.sources || []);
      setAgentTrace(result.agent_trace || null);
      
      // Auto-switch to Tool Trace if agent trace exists
      if (result.agent_trace && result.agent_trace.agent_executions?.length > 1) {
        setRightPanel('tools');
      }
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
    createNewChat();
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

  const handleResetPersona = () => {
    setUserType('consumer');
    setAvailableAgents([]);
    setPersonaDetected(false);
    setPersonaConfidence(0);
    setMessages([]);
    setCitations([]);
    setToolTrace([]);
    setAgentTrace(null);
  };

  // Auto-detect persona from context instead of manual selection
  // No landing page needed - context-based routing handles persona detection

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
        onResetPersona={handleResetPersona}
      />

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Rail */}
        <LeftRail 
          chatHistories={chatHistories}
          currentChatId={currentChatId}
          onSelectChat={selectChat}
          onCreateNewChat={createNewChat}
          onDeleteChat={deleteChat}
          onUpdateChatTitle={updateChatTitle}
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
          currentChatTitle={currentChat?.title || 'New Chat'}
        />

        {/* Right Inspector */}
        <RightInspector 
          activePanel={rightPanel}
          setActivePanel={setRightPanel}
          citations={citations}
          toolTrace={toolTrace}
          agentTrace={agentTrace}
          uploadedPdfs={uploadedPdfs}
          selectedPdfChunk={selectedPdfChunk}
          setSelectedPdfChunk={setSelectedPdfChunk}
          isCollapsed={rightPanelCollapsed}
          onToggleCollapse={() => setRightPanelCollapsed(!rightPanelCollapsed)}
        />
      </div>
    </div>
  );
}

export default App;