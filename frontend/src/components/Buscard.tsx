import { useState, useEffect } from 'react';
import { Bus, Check, ChevronDown, Loader2, Clock, User, Phone, Mail, Star } from 'lucide-react';
import { toast } from 'react-toastify';
import { BusRoute, Seat } from '../types';

interface LocationPoint {
  id: string | number;
  name: string;
  landmark?: string;
  time?: string;
  latitude?: number;
  longitude?: number;
  address?: string;
}

interface UserProfile {
  mobile: string;
  email: string;
  name: string;
  gender: string;
}

interface BusCardProps extends BusRoute {
  onSeatSelect: (seatId: number | string, boardingPoint: any, droppingPoint: any) => void;
  allBoardingPoints?: LocationPoint[];
  allDroppingPoints?: LocationPoint[];
  userProfile: UserProfile;
}

export function BusCard({ 
  id, 
  from, 
  to, 
  rating, 
  duration, 
  startTime, 
  endTime, 
  boardingPoints, 
  droppingPoints, 
  seats,
  allBoardingPoints = [],
  allDroppingPoints = [],
  userProfile
}: BusCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [selectedSeat, setSelectedSeat] = useState<Seat | null>(null);
  const [isBoardingOpen, setIsBoardingOpen] = useState(false);
  const [isDroppingOpen, setIsDroppingOpen] = useState(false);
  const [selectedBoardingPoint, setSelectedBoardingPoint] = useState<LocationPoint | null>(null);
  const [selectedDroppingPoint, setSelectedDroppingPoint] = useState<LocationPoint | null>(null);
  const [confirmStep, setConfirmStep] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [bookingError, setBookingError] = useState<string | null>(null);
  const [paymentUrl, setPaymentUrl] = useState<string | null>(null);

  // Get formatted boarding and dropping points for display
  useEffect(() => {
    // Initialize with default boarding point if available
    if (allBoardingPoints && allBoardingPoints.length > 0 && !selectedBoardingPoint) {
      setSelectedBoardingPoint(allBoardingPoints[0]);
    }
    
    // Initialize with default dropping point if available
    if (allDroppingPoints && allDroppingPoints.length > 0 && !selectedDroppingPoint) {
      setSelectedDroppingPoint(allDroppingPoints[0]);
    }
  }, [allBoardingPoints, allDroppingPoints]);

  // Group seats by type
  const seatsByType = seats.reduce<Record<string, Seat[]>>((acc, seat) => {
    const type = seat.type || 'Regular';
    if (!acc[type]) {
      acc[type] = [];
    }
    acc[type].push(seat);
    return acc;
  }, {});

  // Calculate total fare including GST and discounts
  const calculateTotalFare = (seat: Seat): number => {
    // If the seat has a fare breakdown, calculate total from components
    if (seat.fare) {
      const baseFare = parseFloat((seat.fare["Base Fare"] || 0).toFixed(2));
      const gst = parseFloat((seat.fare["GST"] || 0).toFixed(2));
      const discount = parseFloat((seat.fare["Discount"] || 0).toFixed(2));
      
      // Calculate total with proper decimal precision
      const total = parseFloat((baseFare + gst + discount).toFixed(2));
      return total;
    }
    
    // If totalFare is available, use that
    if (seat.totalFare !== undefined) {
      return parseFloat(seat.totalFare.toFixed(2));
    }
    
    // Fallback to price if available
    if (seat.price !== undefined) {
      return parseFloat(seat.price.toFixed(2));
    }
    
    return 0;
  };

  const handleSeatClick = (seat: Seat) => {
    setSelectedSeat(seat);
    setExpanded(true);
  };

  const handleProceedClick = () => {
    if (!selectedSeat) {
      toast.error('Please select a seat first');
      return;
    }
    
    if (!selectedBoardingPoint) {
      toast.error('Please select a boarding point');
      return;
    }
    
    if (!selectedDroppingPoint) {
      toast.error('Please select a dropping point');
      return;
    }
    
    setConfirmStep(true);
  };

  const handleBookTicket = async () => {
    if (!selectedSeat || !selectedBoardingPoint || !selectedDroppingPoint) {
      setBookingError("Please select seat, boarding point, and dropping point");
      return;
    }
    
    setIsProcessing(true);
    setBookingError(null);
    
    try {
      // Calculate total fare with detailed logging
      const totalFare = calculateTotalFare(selectedSeat);
      
      // Log fare calculation details
      console.log(`Calculating fare for seat ${selectedSeat.id}:`);
      console.log(`- Total fare from seat object: ${selectedSeat.totalFare}`);
      console.log(`- Base fare: ${selectedSeat.fare?.["Base Fare"] || 0}`);
      console.log(`- GST: ${selectedSeat.fare?.["GST"] || 0}`);
      console.log(`- Discount: ${selectedSeat.fare?.["Discount"] || 0}`);
      console.log(`- Calculated total: ${totalFare}`);

      // Create tomorrow's date for the boarding/dropping times
      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 1);
      tomorrow.setUTCHours(0, 0, 0, 0);

      // Add boarding time (23:20:00)
      const boardingTime = new Date(tomorrow);
      boardingTime.setUTCHours(17, 50, 0, 0); // 23:20 IST = 17:50 UTC

      // Add dropping time (07:30:00 next day)
      const droppingTime = new Date(tomorrow);
      droppingTime.setDate(droppingTime.getDate() + 1);
      droppingTime.setUTCHours(2, 0, 0, 0); // 07:30 IST = 02:00 UTC

      const bookingData = {
        mobile: userProfile.mobile || "9154227800",
        email: userProfile.email || "syed@b.com",
        seat_map: [
          {
            passenger_age: 25,
            seat_id: selectedSeat.id,
            passenger_name: userProfile.name || "John",
            gender: userProfile.gender || "Male"
          }
        ],
        trip_id: parseInt(id.toString()),
        boarding_point_id: selectedBoardingPoint.id,
        dropping_point_id: selectedDroppingPoint.id,
        boarding_point_time: boardingTime.toISOString(),
        dropping_point_time: droppingTime.toISOString(),
        total_collect_amount: totalFare,
        main_category: 1,
        freshcardId: 1,
        freshcard: false,
        return_url: window.location.origin + "/booking-confirmation"
      };

      console.log("Sending booking request:", bookingData);
      console.log("Request URL:", "http://localhost:8000/tickets/block");

      // Make API call to block ticket
      const response = await fetch('http://localhost:8000/tickets/block', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
        },
        body: JSON.stringify(bookingData)
      });
      
      console.log("Response status:", response.status);
      
      // Important: Check if the response is empty before trying to parse it
      const responseText = await response.text();
      console.log("Raw response:", responseText);
      
      if (!responseText) {
        throw new Error('Server returned an empty response');
      }
      
      // Try to parse the response text as JSON
      let data;
      try {
        data = JSON.parse(responseText);
      } catch (error) {
        console.error('Failed to parse response as JSON:', responseText);
        throw new Error('Invalid response format from server');
      }
      
      if (!response.ok) {
        throw new Error(data?.message || `Server error: ${response.status}`);
      }
      
      console.log("Booking response:", data);
      
      // Check if we have payment details with a web URL
      if (data.payment_links && data.payment_links.web) {
        // Open payment page in iframe or new window
        setPaymentUrl(data.payment_links.web);
        toast.success("Ticket blocked successfully! Please complete payment.");
      } 
      // Check for payment_details.payment_url (from your original code)
      else if (data.payment_details && data.payment_details.payment_url) {
        setPaymentUrl(data.payment_details.payment_url);
        toast.success("Ticket blocked successfully! Please complete payment.");
      } 
      // For other types of successful responses
      else if (data.status === "NEW" || data.status === "SUCCESS") {
        toast.success("Ticket blocked successfully!");
        // Check if there's an SDK payload with a client payload
        if (data.sdk_payload && data.sdk_payload.payload) {
          toast.info("Please complete payment to confirm your booking.");
          // You could also use the SDK payload here if needed
        }
        setConfirmStep(false);
      } else {
        // No payment URL but request was successful
        toast.success("Ticket blocked successfully!");
        toast.info("Please check your email for payment instructions.");
        setConfirmStep(false);
      }
    } catch (error: any) {
      console.error("Error blocking ticket:", error);
      setBookingError(error.message || "Failed to block ticket");
      toast.error(error.message || "Failed to block ticket");
    } finally {
      setIsProcessing(false);
    }
  };

  const LocationDropdown = ({ 
    label, 
    points, 
    selectedPoint, 
    setSelectedPoint, 
    isOpen, 
    setIsOpen,
    defaultPoint
  }: { 
    label: string; 
    points: LocationPoint[]; 
    selectedPoint: LocationPoint | null;
    setSelectedPoint: (point: LocationPoint) => void;
    isOpen: boolean;
    setIsOpen: (isOpen: boolean) => void;
    defaultPoint?: string;
  }) => {
    // Use provided points or create default points from strings
    const displayPoints = points.length > 0 ? points : 
      (defaultPoint ? [{ id: `${id}-${defaultPoint.replace(/\s+/g, '-').toLowerCase()}`, name: defaultPoint, landmark: 'Main Stop' }] : []);
      
    return (
      <div className="relative">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 leading-tight">{label}</h4>
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="w-full mt-1 px-3 py-2 bg-gray-50 dark:bg-gray-700 flex justify-between items-center rounded-lg text-left border border-gray-200 dark:border-gray-600 transition-colors hover:bg-gray-100 dark:hover:bg-gray-600"
        >
          <div>
            {/* FIX HERE - Access the name property instead of rendering the object */}
            <p className="text-base text-gray-900 dark:text-gray-100 leading-tight">
              {selectedPoint ? selectedPoint.name : 'Select Location'}
            </p>
            {selectedPoint && selectedPoint.landmark && (
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                {selectedPoint.landmark}
              </p>
            )}
          </div>
          <ChevronDown className={`h-4 w-4 text-gray-500 transform transition-transform ${isOpen ? 'rotate-180' : ''}`} />
        </button>
        
        {isOpen && displayPoints.length > 0 && (
          <div className="absolute z-10 mt-1 w-full rounded-md bg-white dark:bg-gray-800 shadow-lg max-h-60 overflow-auto border border-gray-200 dark:border-gray-600">
            {displayPoints.map((point) => (
              <button
                key={point.id}
                type="button"
                className="w-full px-3 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-700 border-b border-gray-100 dark:border-gray-700 last:border-0"
                onClick={() => {
                  setSelectedPoint(point);
                  setIsOpen(false);
                }}
              >
                {/* Each point is rendered correctly here with its properties */}
                <p className="text-sm font-medium text-gray-900 dark:text-gray-100">{point.name}</p>
                {point.landmark && (
                  <p className="text-xs text-gray-500 dark:text-gray-400">{point.landmark}</p>
                )}
                {point.time && (
                  <p className="text-xs text-blue-500 dark:text-blue-400">{point.time}</p>
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className={`bg-white dark:bg-gray-800 shadow-md rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 transition-all duration-200 ${expanded ? 'pb-4' : ''}`}>
      {/* Bus header */}
      <div className="px-4 py-4 flex justify-between items-center">
        <div className="flex items-center">
          <div className="bg-yellow-100 dark:bg-yellow-900 p-2 rounded-full mr-3">
            <Bus className="h-5 w-5 text-yellow-500 dark:text-yellow-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{from} to {to}</h3>
            <div className="flex items-center text-sm text-gray-500 dark:text-gray-400 mt-1">
              <div className="flex items-center">
                <Star className="h-3 w-3 text-yellow-500 mr-1" />
                <span>{rating.toFixed(1)}</span>
              </div>
              <span className="mx-2">•</span>
              <span>{duration}</span>
            </div>
          </div>
        </div>
        <button 
          onClick={() => setExpanded(!expanded)}
          className="bg-yellow-100 dark:bg-gray-700 text-yellow-600 dark:text-yellow-400 p-2 rounded-lg text-sm font-medium flex items-center space-x-1"
        >
          <span>{expanded ? 'Hide' : 'Select'}</span>
        </button>
      </div>
      
      {/* Bus details */}
      <div className="px-4 py-2 bg-gray-50 dark:bg-gray-750 flex justify-between text-sm">
        <div className="flex items-center">
          <Clock className="h-4 w-4 text-gray-500 dark:text-gray-400 mr-1" />
          <span className="text-gray-900 dark:text-gray-100 font-medium">{startTime}</span>
        </div>
        <div className="text-gray-500 dark:text-gray-400">
          {duration}
        </div>
        <div className="flex items-center">
          <Clock className="h-4 w-4 text-gray-500 dark:text-gray-400 mr-1" />
          <span className="text-gray-900 dark:text-gray-100 font-medium">{endTime}</span>
        </div>
      </div>
      
      {/* Seat selection area (shown when expanded) */}
      {expanded && (
        <div className="px-4 py-4 space-y-4">
          {confirmStep ? (
            /* Ticket confirmation form */
            <div className="space-y-4 bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Confirm Your Booking</h3>
              
              {/* Display selected seat and boarding/dropping points */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Selected Seat:</span>
                  <span className="font-medium">{selectedSeat?.seatName} (₹{selectedSeat?.price})</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Boarding Point:</span>
                  <span className="font-medium">{selectedBoardingPoint?.name}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Dropping Point:</span>
                  <span className="font-medium">{selectedDroppingPoint?.name}</span>
                </div>
              </div>
              
              {/* Passenger info fields */}
              <div className="space-y-2">
                <h4 className="text-md font-medium text-gray-900 dark:text-white">Passenger Information</h4>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Name</label>
                  <div className="flex items-center px-3 py-2 bg-gray-50 dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600">
                    <User className="h-4 w-4 text-gray-500 dark:text-gray-400 mr-2" />
                    <span>{userProfile.name || "Not provided"}</span>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Phone</label>
                  <div className="flex items-center px-3 py-2 bg-gray-50 dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600">
                    <Phone className="h-4 w-4 text-gray-500 dark:text-gray-400 mr-2" />
                    <span>{userProfile.mobile || "Not provided"}</span>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Email</label>
                  <div className="flex items-center px-3 py-2 bg-gray-50 dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600">
                    <Mail className="h-4 w-4 text-gray-500 dark:text-gray-400 mr-2" />
                    <span>{userProfile.email || "Not provided"}</span>
                  </div>
                </div>
              </div>
              
              {/* Total fare calculation */}
              <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                <div className="flex justify-between font-medium">
                  <span>Total Fare:</span>
                  <span>₹{selectedSeat ? calculateTotalFare(selectedSeat) : '0'}</span>
                </div>
              </div>
              
              {/* Error message */}
              {bookingError && (
                <div className="text-red-500 text-sm bg-red-50 dark:bg-red-900/20 p-2 rounded">
                  {bookingError}
                </div>
              )}
              
              {/* Action buttons */}
              <div className="flex space-x-2 pt-2">
                <button
                  onClick={() => setConfirmStep(false)}
                  className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg flex-1 transition-colors hover:bg-gray-200 dark:hover:bg-gray-600"
                >
                  Back
                </button>
                <button
                  onClick={handleBookTicket}
                  disabled={isProcessing}
                  className="px-4 py-2 bg-yellow-500 text-white rounded-lg flex-1 transition-colors hover:bg-yellow-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="animate-spin h-4 w-4 mr-2" />
                      Processing...
                    </>
                  ) : (
                    'Confirm & Pay'
                  )}
                </button>
              </div>
            </div>
          ) : (
            /* Seat selection view */
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Seat selection - left column */}
                <div>
                  <h3 className="text-md font-medium text-gray-900 dark:text-white mb-3">Select Seat</h3>
                  
                  {/* Display seats by type */}
                  <div className="space-y-4">
                    {Object.keys(seatsByType).map(type => (
                      <div key={type} className="space-y-2">
                        <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400">{type} Seats</h4>
                        <div className="grid grid-cols-2 gap-2">
                          {seatsByType[type].map(seat => (
                            <button
                              key={seat.id}
                              onClick={() => handleSeatClick(seat)}
                              className={`p-2 rounded-lg border text-left ${
                                selectedSeat?.id === seat.id 
                                  ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20' 
                                  : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-750'
                              }`}
                            >
                              <div className="flex justify-between items-center">
                                <span className="font-medium text-gray-900 dark:text-white">
                                  {seat.label || `Seat ${seat.seatName}`}
                                </span>
                                {selectedSeat?.id === seat.id && (
                                  <Check className="h-4 w-4 text-yellow-500" />
                                )}
                              </div>
                              <span className="text-sm text-gray-500 dark:text-gray-400">₹{seat.price}</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                
                {/* Boarding & dropping selection - right column */}
                <div className="space-y-4">
                  {/* Boarding points selector */}
                  <LocationDropdown
                    label="Boarding Point"
                    points={allBoardingPoints || []}
                    selectedPoint={selectedBoardingPoint}
                    setSelectedPoint={setSelectedBoardingPoint}
                    isOpen={isBoardingOpen}
                    setIsOpen={setIsBoardingOpen}
                    defaultPoint={typeof boardingPoints[0] === 'string' ? boardingPoints[0] : undefined}
                  />
                  
                  {/* Dropping points selector */}
                  <LocationDropdown
                    label="Dropping Point"
                    points={allDroppingPoints || []}
                    selectedPoint={selectedDroppingPoint}
                    setSelectedPoint={setSelectedDroppingPoint}
                    isOpen={isDroppingOpen}
                    setIsOpen={setIsDroppingOpen}
                    defaultPoint={typeof droppingPoints[0] === 'string' ? droppingPoints[0] : undefined}
                  />
                  
                  {/* Proceed button */}
                  <div className="pt-4">
                    <button
                      onClick={handleProceedClick}
                      disabled={!selectedSeat || !selectedBoardingPoint || !selectedDroppingPoint}
                      className="w-full px-4 py-2 bg-yellow-500 text-white rounded-lg transition-colors hover:bg-yellow-600 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Proceed
                    </button>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}
      
      {/* Payment iframe modal */}
      {paymentUrl && (
        <PaymentIframe 
          paymentUrl={paymentUrl} 
          onClose={() => {
            setPaymentUrl(null);
            // Optionally refresh or reset the form after closing
            setConfirmStep(false); 
          }} 
        />
      )}
    </div>
  );
}

// Payment iframe component
const PaymentIframe = ({ paymentUrl, onClose }: { paymentUrl: string; onClose: () => void }) => {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Set a timeout to hide the loading indicator
    const timer = setTimeout(() => {
      setLoading(false);
    }, 3000);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75">
      <div className="bg-white dark:bg-gray-800 p-4 rounded-lg flex flex-col w-full max-w-4xl h-[90vh] relative">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">Complete Payment</h2>
          <button
            onClick={onClose}
            className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
          >
            <svg 
              className="w-6 h-6 text-gray-500" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white dark:bg-gray-800 z-10">
            <div className="loader animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
            <p className="ml-2 text-gray-600 dark:text-gray-300">Loading payment page...</p>
          </div>
        )}
        
        <iframe 
          src={paymentUrl}
          className="flex-1 w-full border-0 rounded"
          title="Payment Page"
          sandbox="allow-forms allow-scripts allow-same-origin allow-top-navigation allow-popups"
        />
        
        <div className="mt-4 text-sm text-gray-500 dark:text-gray-400">
          This payment is securely processed by our payment partner. Do not close this window until payment is complete.
        </div>
      </div>
    </div>
  );
};