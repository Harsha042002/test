import { format } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import { Message } from '../types';
import { BusCard } from './Buscard';
import { toast } from 'react-toastify';

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';
  
  const handleSeatSelect = (seatId: number | string, boardingPoint: any, droppingPoint: any) => {
    console.log("Selected seat ID:", seatId);
    console.log("Selected boarding point:", boardingPoint);
    console.log("Selected dropping point:", droppingPoint);
    
    // Here you would trigger the booking process with the selected points
    toast.success(`Booking seat ${seatId} from ${boardingPoint.name} to ${droppingPoint.name}`);
    
    // This could open a modal, navigate to a booking page, or similar
    // Example: openBookingModal(seatId, boardingPoint, droppingPoint);
  };

  // Get boarding/dropping points from the raw data if available
  const getBoardingPoints = (routeId: string) => {
    if (!message.rawData || !message.rawData.trips) return [];
    
    const trip = message.rawData.trips.find((t: any) => t.tripId === routeId);
    if (!trip) return [];
    
    // If all_boarding_points exists, use that
    if (trip.all_boarding_points && Array.isArray(trip.all_boarding_points)) {
      return trip.all_boarding_points;
    }
    
    // Otherwise create a simple point from the default boarding point
    if (trip.boardingPoint) {
      return [{
        id: `${routeId}-boarding`,
        name: trip.boardingPoint,
        landmark: 'Main Stop'
      }];
    }
    
    return [];
  };

  const getDroppingPoints = (routeId: string) => {
    if (!message.rawData || !message.rawData.trips) return [];
    
    const trip = message.rawData.trips.find((t: any) => t.tripId === routeId);
    if (!trip) return [];
    
    // If all_dropping_points exists, use that
    if (trip.all_dropping_points && Array.isArray(trip.all_dropping_points)) {
      return trip.all_dropping_points;
    }
    
    // Otherwise create a simple point from the default dropping point
    if (trip.droppingPoint) {
      return [{
        id: `${routeId}-dropping`,
        name: trip.droppingPoint,
        landmark: 'Main Stop'
      }];
    }
    
    return [];
  };

  // For debugging
  const hasRawData = !!message.rawData;
  const hasTrips = hasRawData && Array.isArray(message.rawData?.trips);
  
  if (message.busRoutes && message.busRoutes.length > 0) {
    console.log("Message has bus routes:", message.busRoutes.length);
    console.log("Message has raw data:", hasRawData);
    console.log("Message has trips array:", hasTrips);
    
    if (hasTrips) {
      const firstRoute = message.busRoutes[0];
      const boardingPoints = getBoardingPoints(firstRoute.id);
      const droppingPoints = getDroppingPoints(firstRoute.id);
      
      console.log("First route boarding points:", boardingPoints);
      console.log("First route dropping points:", droppingPoints);
    }
  }

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
          
          {/* Check if message has bus routes */}
          {message.busRoutes && message.busRoutes.length > 0 ? (
            <div className="w-full">
              {/* First show the text response */}
              {message.content && (
                <div className="prose dark:prose-invert max-w-none mb-4">
                  <ReactMarkdown>{message.content.split('```json')[0]}</ReactMarkdown>
                </div>
              )}
              
              {/* Then display the bus cards */}
              <div className="grid grid-cols-1 gap-6">
                {message.busRoutes.map(route => (
                  <BusCard
                    key={route.id}
                    {...route}
                    onSeatSelect={handleSeatSelect}
                    // Pass the full boarding and dropping points lists
                    allBoardingPoints={getBoardingPoints(route.id)}
                    allDroppingPoints={getDroppingPoints(route.id)}
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