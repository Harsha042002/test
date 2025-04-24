import { useState, useCallback, useEffect, useRef } from 'react';
import { useAuth } from './hooks/useAuth';
import { Moon, Sun } from 'lucide-react';
import { ThemeProvider, useTheme } from './context/ThemeContext';
import { Sidebar } from './components/Sidebar';
import { ChatMessage } from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import { Chat, Message} from './types'; // Import all needed types
import { Logo } from './components/Logo';
import LoginModal from './components/LoginModal';
import { LoginModalProvider, useLoginModal } from './context/loginModalContext';
import { ToastContainer, toast } from 'react-toastify';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import 'react-toastify/dist/ReactToastify.css';
import { Toaster } from 'react-hot-toast';
import { authService } from './services/api'; // Import authService for logout functionality
import ErrorBoundary from './components/ErrorBoundary';
import { conversationService } from './services/conversationService'; // Import conversationService




const mockChats: Chat[] = [
    {
        id: '1',
        title: 'Getting Started',
        lastUpdated: new Date(),
        messages: [],
    },
];

function Layout() {
    const { theme, toggleTheme } = useTheme();
    const { onOpen } = useLoginModal();
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [selectedChatId, setSelectedChatId] = useState<string>('1');
    const [chats, setChats] = useState<Chat[]>(mockChats);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const isAuthenticated = useAuth();
    const [showLogout, setShowLogout] = useState(false);
    // const [isChatStarted, setIsChatStarted] = useState(false); // Track if chat has started

    // Add booking state and handler
    const [isBookingLoading, setIsBookingLoading] = useState(false);

    const handleBookingProcess = useCallback(async (callback: () => Promise<any>) => {
        setIsBookingLoading(true);
        try {
            await callback();
            toast.success('Booking successful!');
        } catch (error: any) {
            console.error('Booking error:', error);
            toast.error(error.message || 'An error occurred during booking');
        } finally {
            setIsBookingLoading(false);
        }
    }, []);

    // Keep existing handlers
    const toggleSidebar = useCallback(() => {
        setIsSidebarOpen(prev => !prev);
    }, []);


    const handleSendMessage = useCallback(async (content: string) => {
        const userStr = localStorage.getItem('user');
let user: { id?: string; name?: string; mobile?: string } = {};
try {
          user = userStr && userStr !== "undefined" ? JSON.parse(userStr) : {};
} catch {
                        user = {};
}
console.log('DEBUG user object:', user); // <--- ADD THIS LINE
if (!user.id || !user.mobile) {
    toast.error("Please login to chat.");
    return;
}

        const newMessageId = Date.now().toString();

        // Add the user's message to the chat immediately
        const userMessage: Message = {
            id: newMessageId,
            content,
            role: 'user',
            timestamp: new Date(),
        };

        setChats((prevChats) =>
            prevChats.map((chat) =>
                chat.id === selectedChatId
                    ? { ...chat, messages: [...chat.messages, userMessage], lastUpdated: new Date() }
                    : chat
            )
        );

        // Add a temporary loading message for the AI response
        const loadingMessageId = `${newMessageId}-loading`;
        setChats((prevChats) =>
            prevChats.map((chat) =>
                chat.id === selectedChatId
                    ? {
                        ...chat,
                        messages: [
                            ...chat.messages,
                            {
                                id: loadingMessageId,
                                content: '',
                                role: 'assistant',
                                timestamp: new Date(),
                                isLoading: true,
                            },
                        ],
                        lastUpdated: new Date(),
                    }
                    : chat
            )
        );

        try {
            // Prepare request body
const session_id = localStorage.getItem('sessionId') || undefined;
            const body = {
                query: content,
                id: Number(user.id),
                name: user.name,
                mobile: user.mobile, // always use mobile
                ...(session_id && { session_id }),
            };

// Use fetchWithRefresh for token refresh logic
            const response = await authService.fetchWithRefresh('http://localhost:8000/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                            });

            if (!response.ok || !response.body) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

// Update sessionId from response header if present
            const newSessionId = response.headers.get('x-session-id');
            if (newSessionId) localStorage.setItem('sessionId', newSessionId);

            // Stream the plain text answer
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let assistantMessage = '';
            
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
assistantMessage += decoder.decode(value, { stream: true });

                                // Update the loading message with the streamed content
                                setChats((prevChats) =>
                                    prevChats.map((chat) =>
                                        chat.id === selectedChatId
                                            ? {
                                                ...chat,
                                                messages: chat.messages.map((message) =>
                                                    message.id === loadingMessageId
                                                        ? {
                                                            ...message,
                                                            content: assistantMessage,
                                                            isLoading: false,
                                                        }
                                                        : message
                                                ),
                                                lastUpdated: new Date(),
                                            }
                                            : chat
                                    )
                                );
}

// Fetch & render history by user_id and session_id (only once)
if (user.id && newSessionId) {
    const histRes = await authService.fetchWithRefresh(
        `http://localhost:8000/history?user_id=${user.id}&session_id=${newSessionId}`
    );
    if (histRes && histRes.ok) {
const js = await histRes.json();
                setChats((prevChats) =>
            prevChats.map((chat) =>
                chat.id === selectedChatId
                    ? {
                        ...chat,
                        messages: js.history.map((msg: any, idx: number) => ({
                            id: `${selectedChatId}-${idx}`,
                            role: msg.role,
                            content: msg.content,
                            timestamp: new Date(),
                        })),
                        lastUpdated: new Date(),
                    }
                    : chat
            )
        );
                    }
            }
        } catch (error) {
            console.error('Error sending query:', error);
            setChats((prevChats) =>
                prevChats.map((chat) =>
                    chat.id === selectedChatId
                        ? {
                            ...chat,
                            messages: chat.messages.map((message) =>
                                message.id === loadingMessageId
                                    ? {
                                        ...message,
                                        content: 'Sorry, something went wrong. Please try again.',
                                        isLoading: false,
                                    }
                                    : message
                            ),
                            lastUpdated: new Date(),
                        }
                        : chat
                )
            );
        }
    }, [selectedChatId]);

    const handleNewChat = useCallback(() => {
        const currentChat = chats.find(chat => chat.id === selectedChatId);

        if (currentChat && currentChat.messages.length <= 1) {
            toast.error('Please start a conversation before creating a new chat');
            return;
        }

        const newChat: Chat = {
            id: Date.now().toString(),
            title: 'New Chat',
            messages: [],
            lastUpdated: new Date(),
        };
        setChats(prevChats => [newChat, ...prevChats]);
        setSelectedChatId(newChat.id);
    }, [selectedChatId, chats]);

    const loadConversation = useCallback(async (conversationId: string) => {
        try {
            const conversation = await conversationService.getConversation(conversationId);

            const formattedMessages: Message[] = conversation.messages.map((msg: any, index) => {
                // Create the base message
                const formattedMessage: Message = {
                    id: `${conversationId}-${index}`,
                    role: msg.role as 'user' | 'assistant',
                    content: msg.content,
                    timestamp: new Date(),
                };

                // Only add rawData if it exists in the original message
                if (msg.rawData) {
                    formattedMessage.rawData = msg.rawData;
                }

                // Only add busRoutes if it exists in the original message
                if (msg.busRoutes) {
                    formattedMessage.busRoutes = msg.busRoutes;
                }

                return formattedMessage;
            });

            const loadedChat: Chat = {
                id: conversationId,
                title: formattedMessages[0]?.content.substring(0, 30) || 'Loaded Conversation',
                messages: formattedMessages,
                lastUpdated: new Date(conversation.timestamp)
            };

            setChats(prevChats => {
                const exists = prevChats.some(c => c.id === conversationId);
                if (exists) {
                    return prevChats.map(c => c.id === conversationId ? loadedChat : c);
                } else {
                    return [loadedChat, ...prevChats];
                }
            });

            setSelectedChatId(conversationId);
            toast.success('Conversation loaded successfully');
        } catch (error) {
            console.error('Error loading conversation:', error);
            toast.error('Failed to load conversation');
        }
    }, []);

    // Add new useEffect for booking handler
    useEffect(() => {
        const bookingHandler = () => {
            if (isAuthenticated && !isBookingLoading) {
                console.log('Booking handler is set up');
                // Example usage:
                // handleBookingProcess(async () => {
                //     await new Promise(resolve => setTimeout(resolve, 1000));
                //     return { success: true };
                // });
            }
        };

        bookingHandler();
    }, [isAuthenticated, isBookingLoading, handleBookingProcess]);

    // Keep existing scroll effect
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [selectedChatId, chats]);

    const selectedChat = chats.find(chat => chat.id === selectedChatId);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [selectedChat?.messages]);
    


    return (
        <div className="h-[100dvh] flex flex-col bg-[var(--color-app-bg)] text-[var(--color-text)] overflow-hidden">
            <Toaster position="top-center" />
            <header
                className="fixed top-0 left-0 w-full h-16 flex items-center justify-between px-2 sm:px-5 bg-[var(--color-header-bg)] whitespace-nowrap z-50"
            >
                <div className="flex items-center gap-1 flex-shrink-0">
                    <button
                        onClick={toggleSidebar}
                        className="p-2 hover:bg-gray-100 dark:hover:bg-dark-hover rounded-lg transition-colors duration-200"
                        aria-label={isSidebarOpen ? "Close sidebar" : "Open sidebar"}
                    >
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 24 24"
                            className="w-5 sm:w-6 h-5 sm:h-6"
                        >
                            <path
                                d="M18 4H6C4.89543 4 4 4.89543 4 6V18C4 19.1046 4.89543 20 6 20H18C19.1046 20 20 19.1046 20 18V6C20 4.89543 19.1046 4 18 4Z"
                                fill="none"
                                strokeWidth="1.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                className="stroke-gray-700 dark:stroke-gray-300"
                            />
                            <path
                                d="M9 4V20"
                                strokeWidth="1.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                className="stroke-gray-700 dark:stroke-gray-300"
                            />
                            <circle
                                cx="6.5"
                                cy="8"
                                r="1"
                                className="fill-gray-700 dark:fill-gray-300"
                            />
                            <circle
                                cx="6.5"
                                cy="12"
                                r="1"
                                className="fill-gray-700 dark:fill-gray-300"
                            />
                        </svg>
                    </button>

                    {/* Adjusted logo size */}
                    <Logo className="h-6 sm:h-8 w-auto" />
                </div>

                {/* Hide "Ṧ.AI" on mobile */}
                <div className="hidden sm:block absolute left-1/2 transform -translate-x-1/2 text-center font-semibold text-base sm:text-lg">
                    <span className="text-[#1765f3] dark:text-[#fbe822]">Ṧ</span>.AI
                </div>

                <div className="ml-auto flex items-center gap-1 flex-shrink-0">
                    {isAuthenticated ? (
                        <div className="relative">
                            <div
                                className="w-6 sm:w-8 h-6 sm:h-8 cursor-pointer"
                                onClick={() => setShowLogout((prev) => !prev)}
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
                                    <circle
                                        cx="256"
                                        cy="256"
                                        r="256"
                                        fill={theme === "dark" ? "#FBE822" : "#1765F3"}
                                    />
                                    <circle
                                        cx="256"
                                        cy="192"
                                        r="80"
                                        fill={theme === "dark" ? "#1765F3" : "#FBE822"}
                                    />
                                    <path
                                        d="M256 288 C 160 288, 80 352, 80 432 L 432 432 C 432 352, 352 288, 256 288 Z"
                                        fill={theme === "dark" ? "#1765F3" : "#FBE822"}
                                    />
                                </svg>
                            </div>
                            {showLogout && (
                                <button
                                    onClick={async () => {
await authService.logout();
                                        window.dispatchEvent(new Event("storage"));
                                        toast.success("Logged out successfully!");
                                        setShowLogout(false);
                                    }}
                                    className={`absolute top-10 left-1/2 transform -translate-x-1/2 px-3 py-1 text-xs sm:text-sm rounded-lg font-medium transition-all duration-200 ${theme === "dark"
                                            ? "bg-[#FBE822] text-[#1765F3] hover:bg-[#fcef4d]"
                                            : "bg-[#1765F3] text-[#FBE822] hover:bg-[#1e7af3]"
                                        }`}
                                >
                                    Logout
                                </button>
                            )}
                        </div>
                    ) : (
                        <button
                            onClick={onOpen}
                            className={`px-3 py-1 text-xs sm:text-sm rounded-lg font-medium transition-all duration-200 ${theme === "dark"
                                    ? "bg-[#FBE822] text-[#1765F3] hover:bg-[#fcef4d]"
                                    : "bg-[#1765F3] text-[#FBE822] hover:bg-[#1e7af3]"
                                }`}
                        >
                            User Login
                        </button>
                    )}
                    <button
                        onClick={toggleTheme}
                        className="p-2 hover:bg-gray-100 dark:hover:bg-dark-hover rounded-lg transition-colors duration-200"
                        aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
                    >
                        {theme === "light" ? (
                            <Moon size={20} className="text-gray-700" />
                        ) : (
                            <Sun size={20} className="text-gray-300" />
                        )}
                    </button>
                </div>
            </header>

            {/* Main Content */}
            <div className="flex-1 flex flex-col" style={{ marginTop: '9vh' }}>
                <div className="flex-1 flex overflow-hidden">
                    <Sidebar
                        isOpen={isSidebarOpen}
                        onClose={() => setIsSidebarOpen(false)}
                        chats={chats}
                        selectedChatId={selectedChatId}
                        onChatSelect={setSelectedChatId}
                        onNewChat={handleNewChat}
                        onLoadConversation={loadConversation} // Pass the loadConversation function
                    />
                    <ErrorBoundary>
                        <div className="flex-1 flex flex-col items-center justify-center" >
                            {(chats[0].messages.length === 0) ? (
                                <div className="flex flex-col items-center justify-center w-[90%] gap-4 translate-y-[-30%]">
                                    <Logo className="h-16 w-auto" /> {/* Freshbus logo */}
                                    <div className="flex items-center justify-center gap-1 font-semibold text-base sm:text-lg">
                                        <span className="text-[#1765f3] dark:text-[#fbe822]">Ṧ</span>.AI
                                        <span className="text-gray-700 dark:text-gray-300">- Your assistant for Freshbus bookings</span>
                                    </div>
                                    <div className="w-[100%] mx-auto max-w-md">
                                        <ChatInput onSend={handleSendMessage} />
                                    </div>
                                </div>
                            ) : (
                                <div className="flex flex-col h-full w-full">
                                    {/* Scrollable Chat Messages */}
                                    <div className="flex-1  overflow-y-auto hide-scrollbar" style={{ maxHeight: 'calc(100vh - 9vh - 6rem)'}}>
                                        <div className="max-w-4xl sm:max-w-5xl lg:max-w-6xl mx-auto px-2 sm:px-4 lg:px-6">
                                            <div className="py-1.5 space-y-1">
                                                {selectedChat?.messages.map((message) => (
                                                    <ChatMessage key={message.id} message={message} />
                                                ))}
                                                <div ref={messagesEndRef} />
                                                </div>
                                            </div>
                                        </div>

                                        {/* Fixed Chat Input */}
                                        <div
                                            className="bg-[var(--color-app-bg)] fixed bottom-3 left-0 w-full flex items-center justify-center"
                                        >
                                            <div className="w-full max-w-4xl sm:max-w-5xl lg:max-w-6xl mx-auto px-2 sm:px-4 lg:px-6">
                                                <ChatInput onSend={handleSendMessage} />
                                            </div>
                                        </div>
                                    </div>
                            )}
                        </div>
                    </ErrorBoundary>
                </div>
            </div>
        </div>
    );
}

function App() {
    return (
        <BrowserRouter>
            <ThemeProvider>
                <LoginModalProvider>
                    <Routes>
                        <Route path="/" element={<Layout />} />
                        <Route path="/dashboard" element={<Layout />} />
                    </Routes>
                    <LoginModal />
                    <ToastContainer
                        position="top-right"
                        autoClose={3000}
                        hideProgressBar={false}
                        newestOnTop
                        closeOnClick
                        rtl={false}
                        pauseOnFocusLoss
                        draggable
                        pauseOnHover
                    />
                </LoginModalProvider>
            </ThemeProvider>
        </BrowserRouter>
    );
}

export default App;