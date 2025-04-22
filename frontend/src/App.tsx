import { useState, useCallback, useEffect, useRef } from 'react';
import { useAuth } from './hooks/useAuth';
import { Moon, Sun } from 'lucide-react';
import { ThemeProvider, useTheme } from './context/ThemeContext';
import { Sidebar } from './components/Sidebar';
import { ChatMessage } from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import { Chat, Message, BusRoute, Seat } from './types'; // Import all needed types
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

// Define a type for the trip data
interface TripRecommendation {
    window?: {
        seatNumber: string;
        price: string;
        seat_id: string | number;
    };
    aisle?: {
        seatNumber: string;
        price: string;
        seat_id: string | number;
    };
}

interface Trip {
    tripId: string;
    from: string;
    to: string;
    rating: string;
    duration: string;
    departureTime: string;
    arrivalTime: string;
    boardingPoint: string;
    droppingPoint: string;
    all_boarding_points?: any[]; // Add this to support boarding points dropdown
    all_dropping_points?: any[]; // Add this to support dropping points dropdown
    recommendations: {
        reasonable: TripRecommendation;
        premium: TripRecommendation;
        budget_friendly: TripRecommendation;
        [key: string]: TripRecommendation; // Allow any other categories
    };
}

interface JsonData {
    trips?: Trip[];
    [key: string]: any; // Allow any other properties
}

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
        // if (!isChatStarted) {
        //     setIsChatStarted(true); // Mark chat as started
        // }

        const newMessageId = Date.now().toString();

        // Add the user's message to the chat
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
                                content: '', // Empty content initially
                                role: 'assistant',
                                timestamp: new Date(),
                                isLoading: true, // Mark this message as loading
                            },
                        ],
                        lastUpdated: new Date(),
                    }
                    : chat
            )
        );

        try {
            const response = await fetch('http://localhost:8000/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream',
                },
                body: JSON.stringify({
                    query: content,
                    session_id: localStorage.getItem('sessionId') || null,
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const reader = response.body?.getReader();
            const decoder = new TextDecoder();

            if (!reader) {
                throw new Error('No response body reader available');
            }

            let assistantMessage = '';
            let jsonData: JsonData | null = null;

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.error) {
                                throw new Error(data.error);
                            }

                            if (data.json_data) {
                                jsonData = data.json_data;
                                console.log("Received JSON data for bus routes:", jsonData);

                                // Add default boarding/dropping points if not provided
                                if (jsonData && jsonData.trips && Array.isArray(jsonData.trips)) {
                                    jsonData.trips = jsonData.trips.map(trip => {
                                        // Create default boarding points array if not provided
                                        if (!trip.all_boarding_points) {
                                            trip.all_boarding_points = [{
                                                id: `${trip.tripId}-boarding-default`,
                                                name: trip.boardingPoint,
                                                landmark: 'Main Stop',
                                                time: trip.departureTime
                                            }];
                                        }

                                        // Create default dropping points array if not provided
                                        if (!trip.all_dropping_points) {
                                            trip.all_dropping_points = [{
                                                id: `${trip.tripId}-dropping-default`,
                                                name: trip.droppingPoint,
                                                landmark: 'Main Stop',
                                                time: trip.arrivalTime
                                            }];
                                        }

                                        return trip;
                                    });
                                }
                            }

                            if (data.text) {
                                assistantMessage += data.text;

                                setChats((prevChats) =>
                                    prevChats.map((chat) =>
                                        chat.id === selectedChatId
                                            ? {
                                                ...chat,
                                                messages: chat.messages.map((message) =>
                                                    message.id === loadingMessageId
                                                        ? {
                                                            ...message,
                                                            content: assistantMessage, // Update with backend response
                                                            isLoading: false, // Remove loading state
                                                            rawData: jsonData || message.rawData, // Add raw data if available
                                                        }
                                                        : message
                                                ),
                                                lastUpdated: new Date(),
                                            }
                                            : chat
                                    )
                                );

                                // Process JSON data into bus routes if available
                                if (jsonData && jsonData.trips) {
                                    const busRoutes: BusRoute[] = [];

                                    console.log("Processing trips for bus routes:", jsonData.trips.length);

                                    for (const trip of jsonData.trips) {
                                        if (trip.tripId && trip.from && trip.to && trip.recommendations) {
                                            console.log(`Processing trip ${trip.tripId}`);

                                            const busRoute: BusRoute = {
                                                id: trip.tripId,
                                                from: trip.from,
                                                to: trip.to,
                                                rating: parseFloat(trip.rating) || 4.5,
                                                duration: trip.duration || "Unknown",
                                                startTime: trip.departureTime || "Unknown",
                                                endTime: trip.arrivalTime || "Unknown",
                                                boardingPoints: [trip.boardingPoint || "Default Boarding"],
                                                droppingPoints: [trip.droppingPoint || "Default Dropping"],
                                                seats: [],
                                            };

                                            const seats: Seat[] = [];
                                            for (const category in trip.recommendations) {
                                                const catData = trip.recommendations[category];

                                                if (catData.window && catData.window.seatNumber) {
                                                    seats.push({
                                                        id: catData.window.seat_id || `${trip.tripId}-${catData.window.seatNumber}-w`,
                                                        x: 0,
                                                        y: parseInt(catData.window.seatNumber) || 0,
                                                        seatName: catData.window.seatNumber,
                                                        price: parseInt(catData.window.price) || 0,
                                                        totalFare: parseInt(catData.window.price) || 0,
                                                        isOccupied: false,
                                                        availabilityStatus: 'A',
                                                        isReservedForFemales: false,
                                                        isReservedForMales: false,
                                                        fare: { "Base Fare": parseInt(catData.window.price) || 0, GST: 0, Discount: 0 },
                                                        label: `Window ${catData.window.seatNumber}`,
                                                        available: true,
                                                        hasStaticFare: true,
                                                        isDummy: false,
                                                        type: category === 'premium' ? 'Premium' :
                                                            category === 'budget_friendly' ? 'Budget-Friendly' : 'Regular',
                                                    });
                                                }

                                                if (catData.aisle && catData.aisle.seatNumber) {
                                                    seats.push({
                                                        id: catData.aisle.seat_id || `${trip.tripId}-${catData.aisle.seatNumber}-a`,
                                                        x: 1,
                                                        y: parseInt(catData.aisle.seatNumber) || 0,
                                                        seatName: catData.aisle.seatNumber,
                                                        price: parseInt(catData.aisle.price) || 0,
                                                        totalFare: parseInt(catData.aisle.price) || 0,
                                                        isOccupied: false,
                                                        availabilityStatus: 'A',
                                                        isReservedForFemales: false,
                                                        isReservedForMales: false,
                                                        fare: { "Base Fare": parseInt(catData.aisle.price) || 0, GST: 0, Discount: 0 },
                                                        label: `Aisle ${catData.aisle.seatNumber}`,
                                                        available: true,
                                                        hasStaticFare: true,
                                                        isDummy: false,
                                                        type: category === 'premium' ? 'Premium' :
                                                            category === 'budget_friendly' ? 'Budget-Friendly' : 'Regular',
                                                    });
                                                }
                                            }

                                            console.log(`Created ${seats.length} seats for trip ${trip.tripId}`);

                                            busRoute.seats = seats;

                                            if (seats.length > 0) {
                                                busRoutes.push(busRoute);
                                                console.log(`Added bus route ${trip.tripId} with ${seats.length} seats`);
                                            } else {
                                                console.warn(`Bus route ${trip.tripId} has no valid seats, not displaying card`);
                                            }
                                        } else {
                                            console.warn("Trip missing critical data:", trip);
                                        }
                                    }

                                    if (busRoutes.length > 0) {
                                        setChats((prevChats) =>
                                            prevChats.map((chat) =>
                                                chat.id === selectedChatId
                                                    ? {
                                                        ...chat,
                                                        messages: chat.messages.map((message) =>
                                                            message.id === loadingMessageId
                                                                ? {
                                                                    ...message,
                                                                    busRoutes, // Add bus routes to the message
                                                                }
                                                                : message
                                                        ),
                                                        lastUpdated: new Date(),
                                                    }
                                                    : chat
                                            )
                                        );
                                        console.log(`Added ${busRoutes.length} bus route cards to message:`, busRoutes);
                                    } else {
                                        console.warn("No valid bus routes found in response");
                                    }
                                }
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Error sending query:', error);

            // Update the loading message with an error message
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
                                        isLoading: false, // Remove loading state
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
                                    onClick={() => {
                                        authService.clearAuth();
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