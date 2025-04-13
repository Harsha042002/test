import { format } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import { Message } from '../types';
import { BusCard } from './Buscard';
import { toast } from 'react-toastify';
import { useEffect, useState } from 'react';
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
    gender: 'Male'
  });
  
  // Use state to track which bus routes to display
  const [displayRoutes, setDisplayRoutes] = useState(message.busRoutes || []);

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
    <div className="p-2"> 
      <div className={`flex gap-4 py-4 ${isUser ? 'justify-end' : 'justify-start'}`}>
        {!isUser && (
          <div className="flex-shrink-0">
            <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center overflow-hidden">
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
              {isUser 
                ? 'You'
                : (<><span className="text-[#1765f3] dark:text-[#fbe822]">Ṧ</span>.AI</>)}
            </span>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {format(message.timestamp, 'h:mm a')}
            </span>
          </div>
          
          {/* Check if message has bus routes - either from props or extracted */}
          {showBusRoutes ? (
            <div className="w-full">
              {/* First show the text response - FIXED: filter out JSON part */}
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
              <div className="grid grid-cols-1 gap-6">
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
              <span className="text-sm font-medium text-gray-700 dark:text-gray-200">You</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}