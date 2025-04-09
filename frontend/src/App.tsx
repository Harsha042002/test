import { useState, useCallback, useEffect, useRef } from 'react';
import { useAuth } from './hooks/useAuth';
import { Moon, Sun } from 'lucide-react';
import { ThemeProvider, useTheme } from './context/ThemeContext';
import { Sidebar } from './components/Sidebar';
import { ChatMessage } from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import { Chat, Message, BusRoute } from './types';
import { Logo } from './components/Logo';
import LoginModal from './components/LoginModal';
import { LoginModalProvider, useLoginModal } from './context/loginModalContext';
import { ToastContainer, toast } from 'react-toastify';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import 'react-toastify/dist/ReactToastify.css';
import { Toaster } from 'react-hot-toast';
import { authService } from './services/api'; // Import authService for logout functionality

const mockChats: Chat[] = [
    {
        id: '1',
        title: 'Getting Started',
        lastUpdated: new Date(),
        messages: [
            {
                id: '1',
                role: 'assistant',
                content: 'Hello! How can I help you today?',
                timestamp: new Date(),
            },
        ],
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
    const [showLogout, setShowLogout] = useState(false); // State to toggle logout button
    console.log('Authentication state:', isAuthenticated); // Debugging log

    const toggleSidebar = useCallback(() => {
        setIsSidebarOpen(prev => !prev);
    }, []);

    const handleSendMessage = useCallback((content: string) => {
        const newMessage: Message = {
            id: Date.now().toString(),
            content,
            role: 'user',
            timestamp: new Date(),
        };

        setChats(prevChats =>
            prevChats.map(chat =>
                chat.id === selectedChatId
                    ? { ...chat, messages: [...chat.messages, newMessage], lastUpdated: new Date() }
                    : chat
            )
        );

        if (
            (content.toLowerCase().includes('hyderabad') || content.toLowerCase().includes('hyd')) &&
            (content.toLowerCase().includes('vijayawada') || content.toLowerCase().includes('vij'))
        ) {
            setTimeout(() => {
                const sampleBusRoutes: BusRoute[] = [
                    {
                        id: 'bus-1',
                        from: 'Hyderabad',
                        to: 'Vijayawada',
                        rating: 4.5,
                        duration: '8 hrs',
                        startTime: '06:00',
                        endTime: '14:00',
                        boardingPoints: ['Begumpet', 'Madhapur'],
                        droppingPoints: ['Vijayawada Bus Stand', 'Guntur'],
                        seats: [
                            { id: 'R1', type: 'Regular', price: 500, available: true, label: 'R1' },
                            { id: 'R2', type: 'Regular', price: 500, available: true, label: 'R2' },
                            { id: 'B1', type: 'Budget-Friendly', price: 480, available: true, label: 'B1' },
                            { id: 'B2', type: 'Budget-Friendly', price: 480, available: true, label: 'B2' },
                            { id: 'P1', type: 'Premium', price: 800, available: true, label: 'P1' },
                            { id: 'P2', type: 'Premium', price: 800, available: true, label: 'P2' },
                        ],
                    },
                    {
                        id: 'bus-2',
                        from: 'Hyderabad',
                        to: 'Vijayawada',
                        rating: 4.2,
                        duration: '7.5 hrs',
                        startTime: '07:00',
                        endTime: '14:30',
                        boardingPoints: ['Secunderabad', 'Begumpet'],
                        droppingPoints: ['Vijayawada Bus Stand', 'Guntur'],
                        seats: [
                            { id: 'R1', type: 'Regular', price: 480, available: true, label: 'R1' },
                            { id: 'R2', type: 'Regular', price: 480, available: true, label: 'R2' },
                            { id: 'B1', type: 'Budget-Friendly', price: 450, available: true, label: 'B1' },
                            { id: 'B2', type: 'Budget-Friendly', price: 450, available: true, label: 'B2' },
                            { id: 'P1', type: 'Premium', price: 850, available: true, label: 'P1' },
                            { id: 'P2', type: 'Premium', price: 850, available: true, label: 'P2' },
                        ],
                    },
                    {
                        id: 'bus-3',
                        from: 'Hyderabad',
                        to: 'Vijayawada',
                        rating: 4.7,
                        duration: '8.5 hrs',
                        startTime: '05:30',
                        endTime: '14:00',
                        boardingPoints: ['Abids', 'Koti'],
                        droppingPoints: ['Vijayawada Bus Stand', 'Moghal'],
                        seats: [
                            { id: 'R1', type: 'Regular', price: 510, available: true, label: 'R1' },
                            { id: 'R2', type: 'Regular', price: 510, available: true, label: 'R2' },
                            { id: 'B1', type: 'Budget-Friendly', price: 470, available: true, label: 'B1' },
                            { id: 'B2', type: 'Budget-Friendly', price: 470, available: true, label: 'B2' },
                            { id: 'P1', type: 'Premium', price: 900, available: true, label: 'P1' },
                            { id: 'P2', type: 'Premium', price: 900, available: true, label: 'P2' },
                        ],
                    },
                ];

                const busMessage: Message = {
                    id: Date.now().toString() + '-bus',
                    content: '',
                    role: 'assistant',
                    timestamp: new Date(),
                    busRoutes: sampleBusRoutes,
                };

                setChats(prevChats =>
                    prevChats.map(chat =>
                        chat.id === selectedChatId
                            ? { ...chat, messages: [...chat.messages, busMessage], lastUpdated: new Date() }
                            : chat
                    )
                );
            }, 1000);
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
            messages: [
                {
                    id: Date.now().toString() + '-assistant',
                    role: 'assistant',
                    content: 'Hello! How can I help you today?',
                    timestamp: new Date(),
                },
            ],
            lastUpdated: new Date(),
        };
        setChats(prevChats => [newChat, ...prevChats]);
        setSelectedChatId(newChat.id);
    }, [selectedChatId, chats]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [selectedChatId, chats]);

    const selectedChat = chats.find(chat => chat.id === selectedChatId);
    const chatAreaClass = isSidebarOpen
      ? "flex-1 transition-all duration-300 flex flex-col md:w-[calc(100%-16rem)]" // Squeeze on larger screens
      : "flex-1 transition-all duration-300 flex flex-col w-full"; // Full width on smaller screens

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
            onClick={() => {
              authService.clearAuth();
              window.dispatchEvent(new Event("storage"));
              toast.success("Logged out successfully!");
              setShowLogout(false);
            }}
            className={`absolute top-12 right-0 px-3 py-1 text-xs sm:text-sm rounded-lg font-medium transition-all duration-200 ${
              theme === "dark"
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
        className={`px-3 py-1 text-xs sm:text-sm rounded-lg font-medium transition-all duration-200 ${
          theme === "dark"
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
      />
      <div className={`${chatAreaClass} flex flex-col h-full`}>
        <div className="flex-1 overflow-hidden">
          <div className="h-full overflow-y-auto hide-scrollbar"> {/* Ensure scrollability */}
            <div className="max-w-3xl mx-auto px-2">
              <div className="py-1.5 space-y-1">
                {selectedChat?.messages.map((message) => (
                  <ChatMessage key={message.id} message={message} />
                ))}
                <div ref={messagesEndRef} />
              </div>
            </div>
          </div>
        </div>
        <div className="flex-shrink-0 bg-[var(--color-app-bg)]">
          <div className="max-w-3xl mx-auto px-2 py-1.5">
            <ChatInput onSend={handleSendMessage} />
          </div>
        </div>
      </div>
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
                        <Route path="/dashboard" element={< Layout />} />
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
