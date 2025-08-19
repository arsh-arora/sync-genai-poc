import React, { useState } from 'react';
import { ChatHistory, UserType } from '../types';

interface LeftRailProps {
  chatHistories: ChatHistory[];
  currentChatId: string | null;
  onSelectChat: (chatId: string) => void;
  onCreateNewChat: () => void;
  onDeleteChat: (chatId: string) => void;
  onUpdateChatTitle: (chatId: string, title: string) => void;
  userType: UserType;
}

interface ChatHistoryItemProps {
  chat: ChatHistory;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onUpdateTitle: (title: string) => void;
}

const ChatHistoryItem: React.FC<ChatHistoryItemProps> = ({ 
  chat, 
  isSelected, 
  onSelect, 
  onDelete, 
  onUpdateTitle 
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editTitle, setEditTitle] = useState(chat.title);

  const handleSaveTitle = () => {
    if (editTitle.trim() && editTitle !== chat.title) {
      onUpdateTitle(editTitle.trim());
    }
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSaveTitle();
    } else if (e.key === 'Escape') {
      setEditTitle(chat.title);
      setIsEditing(false);
    }
  };

  return (
    <div className="border-b border-slate-100 last:border-0">
      <div
        className={`group p-3 cursor-pointer transition-all hover:bg-slate-50 ${
          isSelected 
            ? 'bg-blue-50 border-r-2 border-blue-500' 
            : 'hover:bg-slate-50'
        }`}
        onClick={onSelect}
      >
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            {isEditing ? (
              <input
                type="text"
                value={editTitle}
                onChange={(e) => setEditTitle(e.target.value)}
                onBlur={handleSaveTitle}
                onKeyDown={handleKeyDown}
                className="w-full text-sm font-medium bg-transparent border-none outline-none focus:ring-1 focus:ring-blue-500 rounded px-1"
                autoFocus
                onClick={(e) => e.stopPropagation()}
              />
            ) : (
              <h3 className={`font-medium ${isSelected ? 'text-blue-700' : 'text-slate-700'} text-sm truncate`}>
                {chat.title}
              </h3>
            )}
            <div className="flex items-center justify-between mt-1">
              <p className="text-xs text-slate-500">
                {chat.messages.length} messages
              </p>
              <p className="text-xs text-slate-400">
                {new Date(chat.updatedAt).toLocaleDateString()}
              </p>
            </div>
            {chat.persona && (
              <span className={`inline-block mt-1 px-2 py-0.5 text-xs rounded-full ${
                chat.persona === 'consumer' 
                  ? 'bg-green-100 text-green-700' 
                  : 'bg-purple-100 text-purple-700'
              }`}>
                {chat.persona}
              </span>
            )}
          </div>
          <div className="opacity-0 group-hover:opacity-100 transition-opacity flex items-center space-x-1 ml-2">
            <button
              onClick={(e) => { 
                e.stopPropagation(); 
                setIsEditing(true);
              }}
              className="p-1 hover:bg-slate-200 rounded"
              title="Edit title"
            >
              <i className="fas fa-edit text-xs text-slate-400"></i>
            </button>
            <button
              onClick={(e) => { 
                e.stopPropagation(); 
                if (confirm('Delete this chat?')) onDelete();
              }}
              className="p-1 hover:bg-red-100 rounded"
              title="Delete chat"
            >
              <i className="fas fa-trash text-xs text-red-400"></i>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

const LeftRail: React.FC<LeftRailProps> = ({ 
  chatHistories,
  currentChatId,
  onSelectChat,
  onCreateNewChat,
  onDeleteChat,
  onUpdateChatTitle,
  userType
}) => {
  
  return (
    <div className="w-72 bg-white border-r border-slate-200 flex flex-col">
      <div className="p-4 border-b border-slate-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-slate-800">Chat History</h2>
          <button
            onClick={onCreateNewChat}
            className="p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
            title="New Chat"
          >
            <i className="fas fa-plus text-sm"></i>
          </button>
        </div>
        <div className="text-xs text-slate-400 bg-slate-50 px-2 py-1 rounded">
          {userType} persona
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto">
        {chatHistories.length === 0 ? (
          <div className="p-6 text-center">
            <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <i className="fas fa-comments text-slate-400 text-xl"></i>
            </div>
            <h3 className="font-medium text-slate-700 mb-2">No chats yet</h3>
            <p className="text-sm text-slate-500 mb-4">Start a new conversation to see your chat history</p>
            <button
              onClick={onCreateNewChat}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors"
            >
              Start New Chat
            </button>
          </div>
        ) : (
          chatHistories.map((chat) => (
            <ChatHistoryItem
              key={chat.id}
              chat={chat}
              isSelected={chat.id === currentChatId}
              onSelect={() => onSelectChat(chat.id)}
              onDelete={() => onDeleteChat(chat.id)}
              onUpdateTitle={(title) => onUpdateChatTitle(chat.id, title)}
            />
          ))
        )}
      </div>
      
      <div className="p-3 border-t border-slate-100 bg-slate-50">
        <p className="text-xs text-slate-500 text-center">
          {chatHistories.length} chat{chatHistories.length !== 1 ? 's' : ''} saved
        </p>
      </div>
    </div>
  );
};

export default LeftRail;