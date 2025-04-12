// UserChats.tsx
import { useState, useEffect } from 'react';
import { useTheme } from '../context/ThemeContext';
import { useAuth } from '../hooks/useAuth';
import { MessageSquare, RefreshCw, Trash2 } from 'lucide-react';

interface UserChat {
    conversation_id: string;
    timestamp: string;
    user_query: string;
    message_count: number;
}

interface UserChatsProps {
    onChatSelect: (chatId: string) => void;
    selectedChatId: string | null;
}

export function UserChats({ onChatSelect, selectedChatId }: UserChatsProps) {
    const { theme } = useTheme();
    const isAuthenticated = useAuth();
    const [userChats, setUserChats] = useState<UserChat[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchUserChats = async () => {
        if (!isAuthenticated) {
            setUserChats([]);
            return;
        }

        try {
            setIsLoading(true);
            setError(null);
            
            const response = await fetch('/user/conversations', {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to fetch chats: ${response.status}`);
            }

            const data = await response.json();
            setUserChats(data);
        } catch (err) {
            setError('Failed to load your conversations');
            console.error('Error fetching user chats:', err);
        } finally {
            setIsLoading(false);
        }
    };

    const deleteUserChats = async () => {
        if (!window.confirm('Are you sure you want to delete all your chat history?')) {
            return;
        }

        try {
            setIsLoading(true);
            const response = await fetch('/user/conversations', {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to delete chats: ${response.status}`);
            }

            setUserChats([]);
        } catch (err) {
            setError('Failed to delete conversations');
            console.error('Error deleting user chats:', err);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchUserChats();
    }, [isAuthenticated]);

    if (!isAuthenticated) {
        return (
            <div className="p-4 text-center text-gray-500 dark:text-gray-400">
                <p>Log in to view your chat history</p>
            </div>
        );
    }

    return (
        <div className="mt-2">
            <div className="flex justify-between items-center px-4 py-2">
                <h3 className={`font-medium ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                    Your Conversations
                </h3>
                <div className="flex gap-2">
                    <button 
                        onClick={fetchUserChats} 
                        className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800"
                        title="Refresh conversations"
                    >
                        <RefreshCw size={16} className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} />
                    </button>
                    {userChats.length > 0 && (
                        <button 
                            onClick={deleteUserChats} 
                            className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800"
                            title="Delete all conversations"
                        >
                            <Trash2 size={16} className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} />
                        </button>
                    )}
                </div>
            </div>

            {isLoading ? (
                <div className="flex justify-center py-4">
                    <div className="w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full animate-spin"></div>
                </div>
            ) : error ? (
                <div className="p-4 text-center text-red-500">
                    <p>{error}</p>
                    <button 
                        onClick={fetchUserChats}
                        className="mt-2 text-sm underline"
                    >
                        Try again
                    </button>
                </div>
            ) : userChats.length === 0 ? (
                <div className="p-4 text-center text-gray-500 dark:text-gray-400">
                    <p>No conversations found</p>
                </div>
            ) : (
                <div className="space-y-1">
                    {userChats.map(chat => (
                        <button
                            key={chat.conversation_id}
                            onClick={() => onChatSelect(chat.conversation_id)}
                            className={`w-full px-4 py-3 text-left transition-colors hover:bg-gray-100 dark:hover:bg-gray-800 flex items-start gap-3 ${
                                selectedChatId === chat.conversation_id 
                                    ? 'bg-gray-100 dark:bg-gray-800 border-l-2 border-blue-500 dark:border-blue-400' 
                                    : ''
                            }`}
                        >
                            <MessageSquare 
                                size={18} 
                                className={theme === 'dark' ? 'text-gray-400 mt-1' : 'text-gray-600 mt-1'} 
                            />
                            <div className="overflow-hidden flex-1">
                                <p className={`truncate font-medium ${theme === 'dark' ? 'text-gray-200' : 'text-gray-800'}`}>
                                    {chat.user_query || 'New conversation'}
                                </p>
                                <div className="flex justify-between items-center mt-1">
                                    <span className="text-xs text-gray-500">
                                        {new Date(chat.timestamp).toLocaleDateString()} · {chat.message_count} messages
                                    </span>
                                </div>
                            </div>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}