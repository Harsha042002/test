import { useEffect, useState } from 'react';
import { Plus, X, MessageSquare, User } from 'lucide-react';
import { Chat } from '../types';
import { Logo } from './Logo';
import { UserChats } from './UserChats';
import { useAuth } from '../hooks/useAuth';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  chats: Chat[];
  onChatSelect: (chatId: string) => void;
  selectedChatId: string | null;
  onNewChat: () => void;
  onLoadConversation: (conversationId: string) => void;
}

export function Sidebar({ 
  isOpen, 
  onClose, 
  chats, 
  onChatSelect, 
  selectedChatId, 
  onNewChat,
  onLoadConversation
}: SidebarProps) {
  const isAuthenticated = useAuth();
  const [activeTab, setActiveTab] = useState<'session' | 'account'>('session');

  // Handle escape key to close sidebar
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  // Prevent body scroll when sidebar is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  // Handle user conversation selection
  const handleUserChatSelect = (conversationId: string) => {
    onLoadConversation(conversationId);
    onClose();
  };

  return (
    <aside
      className={`fixed top-0 left-0 h-full w-64 bg-[var(--color-sidebar-bg)] border-r border-gray-200 dark:border-dark-border ${
        isOpen ? 'translate-x-0 shadow-xl opacity-100' : '-translate-x-full opacity-0'
      } z-50 transition-transform duration-300`}
    >
      {/* Sidebar Header */}
      <div
        className="fixed top-0 left-0 w-64 h-16 z-60 bg-[var(--color-sidebar-bg)] dark:bg-[#000000] flex items-center justify-between px-4 border-b border-gray-200 dark:border-dark-border"
      >
        <button
          onClick={onClose}
          className="p-2 hover:bg-gray-100 dark:hover:bg-dark-hover rounded-lg transition-colors duration-200"
          aria-label="Close Sidebar"
        >
          <X size={24} className="text-gray-700 dark:text-gray-300" />
        </button>
        <div>
          <Logo className="h-6 sm:h-8 w-auto" />
        </div>
      </div>

      {/* New Chat Option */}
      <div className="mt-16 p-4">
        <button
          onClick={() => {
            onNewChat();
            onClose();
          }}
          className="w-full flex items-center justify-center gap-2 p-2 bg-blue-500 text-white rounded-lg focus:outline-none hover:bg-blue-600 transition-colors"
        >
          <Plus size={18} />
          New Chat
        </button>
      </div>

      {/* Tabs - Only show if authenticated */}
      {isAuthenticated && (
        <div className="flex border-b border-gray-200 dark:border-gray-800">
          <button
            onClick={() => setActiveTab('session')}
            className={`flex-1 py-2 px-4 text-sm font-medium text-center transition-colors ${
              activeTab === 'session'
                ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center justify-center gap-1">
              <MessageSquare size={16} />
              <span>Session</span>
            </div>
          </button>
          <button
            onClick={() => setActiveTab('account')}
            className={`flex-1 py-2 px-4 text-sm font-medium text-center transition-colors ${
              activeTab === 'account'
                ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center justify-center gap-1">
              <User size={16} />
              <span>My Chats</span>
            </div>
          </button>
        </div>
      )}

      {/* Content based on active tab */}
      <div className="h-[calc(100vh-16rem)] overflow-y-auto hide-scrollbar pb-4">
        {activeTab === 'session' ? (
          /* Session-based Chat List */
          <div>
            {chats.map((chat) => (
              <button
                key={chat.id}
                onClick={() => {
                  onChatSelect(chat.id);
                  onClose();
                }}
                className={`w-full p-4 text-left hover:bg-gray-100 dark:hover:bg-dark-hover transition-colors duration-200 ${
                  selectedChatId === chat.id ? 'bg-gray-100 dark:bg-dark-surface' : ''
                }`}
                aria-selected={selectedChatId === chat.id}
              >
                <h3 className="font-medium truncate text-gray-900 dark:text-gray-100">{chat.title}</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 truncate">
                  {chat.messages[chat.messages.length - 1]?.content}
                </p>
              </button>
            ))}
          </div>
        ) : (
          /* User Account-based Chat List */
          <UserChats onChatSelect={handleUserChatSelect} selectedChatId={selectedChatId} />
        )}
      </div>
    </aside>
  );
}
