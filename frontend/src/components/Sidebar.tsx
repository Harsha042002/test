import { useEffect } from 'react';
import { Plus, X } from 'lucide-react';
import { Chat } from '../types';
import { Logo } from './Logo';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  chats: Chat[];
  onChatSelect: (chatId: string) => void;
  selectedChatId: string | null;
  onNewChat: () => void;
}

export function Sidebar({ isOpen, onClose, chats, onChatSelect, selectedChatId, onNewChat }: SidebarProps) {
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
      <div className="mt-16 p-4"> {/* Added `mt-16` to account for the fixed header */}
        <button
          onClick={() => {
            onNewChat();
            onClose();
          }}
          className="w-full flex items-center gap-2 p-2 bg-blue-500 text-white rounded-lg focus:outline-none"
        >
          <Plus size={18} />
          New Chat
        </button>
      </div>

      {/* Chat List */}
      <div className="h-[calc(100vh-16rem)] overflow-y-auto hide-scrollbar pb-4"> {/* Removed `mt-16` */}
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
    </aside>
  );
}
