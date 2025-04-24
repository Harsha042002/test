import { useEffect, useState } from 'react';
import { Plus, X } from 'lucide-react';
import { Chat } from '../types';
import { Logo } from './Logo';
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
  onNewChat,
  onLoadConversation
}: SidebarProps) {
  const isAuthenticated = useAuth();
  const [userChats, setUserChats] = useState<{ session_id: string; preview: string }[]>([]);

  // Fetch previous chats from backend when authenticated
  useEffect(() => {
    const fetchChats = async () => {
      if (!isAuthenticated) {
        setUserChats([]);
        return;
      }
      const userStr = localStorage.getItem('user');
      let user: { id?: string } = {};
      try {
        user = userStr && userStr !== "undefined" ? JSON.parse(userStr) : {};
      } catch {
        user = {};
      }
      if (!user.id) return;
      try {
        const res = await fetch(`http://localhost:8000/conversations?user_id=${user.id}`);
        if (res.ok) {
          const data = await res.json();
          setUserChats(data.conversations || []);
        }
      } catch (err) {
        setUserChats([]);
      }
    };
    fetchChats();
  }, [isAuthenticated]);

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

      {/* Previous conversations */}
      {isAuthenticated && (
        <div className="h-[calc(100vh-8rem)] overflow-y-auto hide-scrollbar pb-4">
          <div>
            {userChats.length === 0 ? (
              <div className="text-center text-gray-400 mt-8">No previous chats found.</div>
            ) : (
              userChats.map((chat) => (
                <button
                  key={chat.session_id}
                  onClick={() => {
                    onLoadConversation(chat.session_id);
                    onClose();
                  }}
                  className="w-full p-4 text-left hover:bg-gray-100 dark:hover:bg-dark-hover transition-colors duration-200"
                >
                  <h3 className="font-medium truncate text-gray-900 dark:text-gray-100">
                    {chat.preview ? chat.preview.substring(0, 30) : "Previous Chat"}
                  </h3>
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </aside>
  );
}
