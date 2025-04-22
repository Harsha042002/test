import ReactMarkdown from 'react-markdown';
import { Message } from '../types';
import { BusCard } from './Buscard';
import { toast } from 'react-toastify';
import { useEffect, useState } from 'react';
import { useTheme } from '../context/ThemeContext';
import { useBusRoutes } from '../hooks/useBusRoutes';



interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const extractedBusRoutes = useBusRoutes(message);
  
  // Add user profile state
  const [userProfile, setUserProfile] = useState({
    mobile: '',
    email: '',
    name: '',
    gender: '',
  });
  
  // Use state to track which bus routes to display
  const [displayRoutes, setDisplayRoutes] = useState(message.busRoutes || []);
  const { theme } = useTheme();
  const isLoading = message.isLoading || false;

  // Add useEffect for loading user profile
  useEffect(() => {
    // Get user info from localStorage if available
    const userString = localStorage.getItem('user');
    if (userString) {
      try {
        const userData = JSON.parse(userString);
        setUserProfile({
          mobile: userData.mobile || '',
          email: userData.email || '',
          name: userData.name || '',
          gender: userData.gender || 'Male'
        });
      } catch (error) {
        console.error('Error parsing user data:', error);
      }
    }
  }, []);

  const handleSeatSelect = (seatId: number | string, boardingPoint: any, droppingPoint: any) => {
    console.log("Selected seat ID:", seatId);
    console.log("Selected boarding point:", boardingPoint);
    console.log("Selected dropping point:", droppingPoint);
    
    // Here you would trigger the booking process with the selected points
    toast.success(`Booking seat ${seatId} from ${boardingPoint.name} to ${droppingPoint.name}`);
  };

  // Extract JSON data from the content
  const getJsonData = () => {
    try {
      if (message.content && message.content.includes('```json')) {
        const jsonStr = message.content.split('```json')[1].split('```')[0];
        return JSON.parse(jsonStr.trim());
      }
    } catch (e) {
      console.error('Error parsing JSON from content:', e);
    }
    return null;
  };

  // Get boarding/dropping points from various sources
  const getBoardingPoints = (routeId: string) => {
    // Try to get from rawData first
    if (message.rawData && message.rawData.trips) {
      const trip = message.rawData.trips.find((t: any) => t.tripId === routeId);
      if (trip) {
        if (trip.all_boarding_points && Array.isArray(trip.all_boarding_points)) {
          return trip.all_boarding_points;
        }
        
        if (trip.boardingPoint) {
          return [{
            id: `${routeId}-boarding`,
            name: trip.boardingPoint,
            landmark: 'Main Stop'
          }];
        }
      }
    }
    
    // Try to extract from JSON in content
    const jsonData = getJsonData();
    if (jsonData && jsonData.trips) {
      const trip = jsonData.trips.find((t: any) => t.tripId === routeId);
      if (trip) {
        if (trip.all_boarding_points && Array.isArray(trip.all_boarding_points)) {
          return trip.all_boarding_points;
        }
        
        if (trip.boardingPoint) {
          return [{
            id: `${routeId}-boarding`,
            name: trip.boardingPoint,
            landmark: 'Main Stop'
          }];
        }
      }
    }
    
    // Default fallback
    return [];
  };

  const getDroppingPoints = (routeId: string) => {
    // Try to get from rawData first
    if (message.rawData && message.rawData.trips) {
      const trip = message.rawData.trips.find((t: any) => t.tripId === routeId);
      if (trip) {
        if (trip.all_dropping_points && Array.isArray(trip.all_dropping_points)) {
          return trip.all_dropping_points;
        }
        
        if (trip.droppingPoint) {
          return [{
            id: `${routeId}-dropping`,
            name: trip.droppingPoint,
            landmark: 'Main Stop'
          }];
        }
      }
    }
    
    // Try to extract from JSON in content
    const jsonData = getJsonData();
    if (jsonData && jsonData.trips) {
      const trip = jsonData.trips.find((t: any) => t.tripId === routeId);
      if (trip) {
        if (trip.all_dropping_points && Array.isArray(trip.all_dropping_points)) {
          return trip.all_dropping_points;
        }
        
        if (trip.droppingPoint) {
          return [{
            id: `${routeId}-dropping`,
            name: trip.droppingPoint,
            landmark: 'Main Stop'
          }];
        }
      }
    }
    
    // Default fallback
    return [];
  };

  // Enhanced debugging logs
  useEffect(() => {
    console.log("Message object:", message);
    console.log("Has bus routes:", !!message.busRoutes);
    console.log("Has JSON content:", message.content?.includes('```json'));
    console.log("Extracted routes:", extractedBusRoutes);
    
    // Check for JSON in content
    if (message.content && message.content.includes('```json')) {
      try {
        const jsonStr = message.content.split('```json')[1].split('```')[0];
        const jsonData = JSON.parse(jsonStr.trim());
        console.log("Parsed JSON data:", jsonData);
        if (jsonData && jsonData.trips) {
          console.log("JSON has trips:", jsonData.trips.length);
        }
      } catch (e) {
        console.error("Error parsing JSON from content:", e);
      }
    }
    
    // Log routes details
    if (message.busRoutes && message.busRoutes.length > 0) {
      console.log(`Message has ${message.busRoutes.length} bus routes`);
    }
    
    if (extractedBusRoutes && extractedBusRoutes.length > 0) {
      console.log(`Extracted ${extractedBusRoutes.length} bus routes`);
    }
  }, [message, extractedBusRoutes]);

  // Update displayRoutes when extractedBusRoutes changes
  useEffect(() => {
    if (message.busRoutes && message.busRoutes.length > 0) {
      setDisplayRoutes(message.busRoutes);
    } else if (extractedBusRoutes && extractedBusRoutes.length > 0) {
      setDisplayRoutes(extractedBusRoutes);
    }
  }, [message.busRoutes, extractedBusRoutes]);

  // Determine if we should show bus routes
  const hasBusRoutes = !!(message.busRoutes && message.busRoutes.length > 0);
  const hasExtractedRoutes = !!(extractedBusRoutes && extractedBusRoutes.length > 0);
  const showBusRoutes = hasBusRoutes || hasExtractedRoutes;

  // Debug: log the final decision
  useEffect(() => {
    console.log("Will show bus routes:", showBusRoutes);
    console.log("Routes to show:", displayRoutes.length);
  }, [showBusRoutes, displayRoutes]);

  return (
    <div className={`p-1 ${isUser ? 'text-right' : 'text-left'}`}> 
      <div className={`flex gap-2 py-2 ${isUser ? 'justify-end' : 'justify-start'}`}>
        {!isUser && (
          <div className="flex-shrink-0">
            <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center ">
              <img
                src="/src/assets/aiimg.png"
                alt="AI Icon"
                className="w-5 h-5 object-contain"
                style={{ transform: 'scale(1.5)' }}
              />
            </div>
          </div>
        )}
        <div className={`flex-1 space-y-2 ${isUser ? 'text-right' : 'text-left'}`}>
          <div className="flex items-center gap-2 justify-between">
          <span className="font-medium text-gray-900 dark:text-gray-100">
             {!isUser && (<><span className="text-[#1765f3] dark:text-[#fbe822]">Ṧ</span>.AI</>)}
          </span>
          </div>

          {/* Show loading animation or message content */}
          {!isUser && isLoading ? (
            <div className="flex justify-start items-center">
              <div className="w-2 h-2 bg-blue-500 dark:bg-yellow-500 rounded-full animate-dots"></div>
              <div className="w-2 h-2 bg-blue-500 dark:bg-yellow-500 rounded-full animate-dots"></div>
              <div className="w-2 h-2 bg-blue-500 dark:bg-yellow-500 rounded-full animate-dots"></div>
            </div>
          ) : showBusRoutes ? (  
            <div className="w-full">
              {/* First show the text response */}
              {message.content && (
                <div className="prose dark:prose-invert max-w-none mb-4">
                  <ReactMarkdown>
                    {message.content.includes('```json') 
                      ? message.content.split('```json')[0] 
                      : message.content}
                  </ReactMarkdown>
                </div>
              )}
              
              {/* Then display the bus cards */}
              <div className="grid grid-cols-3 gap-2">
                {displayRoutes.map(route => (
                  <BusCard
                    key={route.id}
                    {...route}
                    onSeatSelect={handleSeatSelect}
                    allBoardingPoints={getBoardingPoints(route.id)}
                    allDroppingPoints={getDroppingPoints(route.id)}
                    userProfile={userProfile}
                  />
                ))}
              </div>
            </div>
          ) : (
            // Regular message without bus routes
            <div className="prose dark:prose-invert max-w-none">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>
        {isUser && (
          <div className="flex-shrink-0">
            <div className="w-8 h-8 rounded-full bg-gray-300 dark:bg-gray-600 flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" className="w-7 h-7">
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
          </div>
        )}
      </div>
    </div>
  );
}