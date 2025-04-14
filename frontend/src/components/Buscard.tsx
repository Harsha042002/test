import { useState, useEffect } from 'react';
import { Bus, Check, ChevronDown, Loader2, Clock, User, Phone, Mail, Star } from 'lucide-react';
import { toast } from 'react-toastify';
import { BusRoute, Seat, LocationPoint, FareDetails } from '../types';

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
    console.log(`Calculating fare for seat ${seat.id || seat.seat_id}:`);
    
    // First try to find fare details from various possible places in the data structure
    let fareDetails: FareDetails | null = null;
    
    if (seat.fare) {
      fareDetails = seat.fare;
    } 
    else if (seat.window && seat.window.fare) {
      fareDetails = seat.window.fare;
    } 
    else if (seat.aisle && seat.aisle.fare) {
      fareDetails = seat.aisle.fare;
    }
    
    // If we found fare details, use them to calculate total
    if (fareDetails) {
      const baseFare = fareDetails["Base Fare"] || 0;
      const gst = fareDetails["GST"] || 0;
      const discount = fareDetails["Discount"] || 0;
      
      console.log(`- Base fare: ${baseFare}`);
      console.log(`- GST: ${gst}`);
      console.log(`- Discount: ${discount}`);
      
      const total = baseFare + gst + discount;
      console.log(`- Sum of components: ${total}`);
      return total;
    }
    
    // Fallbacks if no fare details found
    if (seat.totalFare !== undefined) {
      console.log(`- Using provided totalFare as fallback: ${seat.totalFare}`);
      return seat.totalFare;
    }
    
    if (seat.price !== undefined) {
      const price = typeof seat.price === 'string' ? parseFloat(seat.price) : seat.price;
      console.log(`- Using price as fallback: ${price}`);
      return price;
    }
    
    console.log(`- No fare information available, defaulting to 0`);
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
      // Get auth token from localStorage
      const authToken = localStorage.getItem('auth_token');
      if (!authToken) {
        throw new Error('Authentication required');
      }
  
      // Extract exact fare components from the seat data
      // Reference the original fare data structure
      let fareDetails: FareDetails = { "Base Fare": 0, "GST": 0, "Discount": 0 };
      
      // First priority: Get fare from the direct fare property on the seat
      if (selectedSeat.fare) {
        fareDetails = selectedSeat.fare;
      } 
      // Second priority: Try to get from window/aisle properties
      else if (selectedSeat.window && selectedSeat.window.fare) {
        fareDetails = selectedSeat.window.fare;
      } 
      else if (selectedSeat.aisle && selectedSeat.aisle.fare) {
        fareDetails = selectedSeat.aisle.fare;
      }
      // Otherwise, construct a basic fare object from price
      else if (selectedSeat.price) {
        const price = typeof selectedSeat.price === 'string' ? parseFloat(selectedSeat.price) : selectedSeat.price;
        fareDetails = {
          "Base Fare": price,
          "GST": 0,
          "Discount": 0
        };
      }
      
      // Calculate the exact fare
      const baseFare = fareDetails["Base Fare"] || 0;
      const gst = fareDetails["GST"] || 0;
      const discount = fareDetails["Discount"] || 0;
      const exactFare = baseFare + gst + discount;
      
      // Log the exact fare details for debugging
      console.log("Using exact fare components from API:");
      console.log(`- Base fare: ${baseFare}`);
      console.log(`- GST: ${gst}`);
      console.log(`- Discount: ${discount}`);
      console.log(`- Total: ${exactFare}`);
  
      // Create tomorrow's date for the boarding/dropping times
      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 1);
      
      // Set boarding time (using the startTime prop)
      const boardingTime = new Date(tomorrow);
      const [boardingHours, boardingMinutes] = startTime.split(':').map(part => parseInt(part));
      boardingTime.setHours(boardingHours, boardingMinutes, 0, 0);
      
      // Set dropping time (using the endTime prop)
      const droppingTime = new Date(tomorrow);
      const [droppingHours, droppingMinutes] = endTime.split(':').map(part => parseInt(part));
      droppingTime.setHours(droppingHours, droppingMinutes, 0, 0);
      
      // If dropping time is earlier than boarding time, add 1 day to dropping time
      if (droppingTime < boardingTime) {
        droppingTime.setDate(droppingTime.getDate() + 1);
      }
  
      // Create the booking data - use the exact fare breakdown from the API
      const bookingData = {
        mobile: userProfile.mobile || "9154227800",
        email: userProfile.email || "syed@b.com",
        seat_map: [{
          passenger_age: 25,
          seat_id: selectedSeat.id || selectedSeat.seat_id,
          passenger_name: userProfile.name || "John",
          gender: userProfile.gender || "Male"
        }],
        trip_id: parseInt(id.toString()),
        boarding_point_id: selectedBoardingPoint.id,
        dropping_point_id: selectedDroppingPoint.id,
        boarding_point_time: boardingTime.toISOString(),
        dropping_point_time: droppingTime.toISOString(),
        // Use the exact fare sum as the total amount
        total_collect_amount: exactFare,
        // Include the fare components EXACTLY as they appear in the API response
        fare: fareDetails,
        main_category: 1,
        freshcardId: 1,
        freshcard: false,
        return_url: window.location.origin + "/booking-confirmation"
      };
  
      // Log the complete booking data for debugging
      console.log("Sending booking data:", JSON.stringify(bookingData, null, 2));
  
      // Make the API call with auth token
      const response = await fetch('http://localhost:8000/tickets/block', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify(bookingData)
      });
  
      console.log("Response status:", response.status);
  
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || `Server error: ${response.status}`);
      }
  
      // Process successful response
      const data = await response.json();
      console.log("Full booking response:", data);

      // Get the payment URL from the response
      let paymentUrl: string | undefined;

      if (data.payment_details?.payment_links?.web) {
        paymentUrl = data.payment_details.payment_links.web;
      } else if (data.payment_links?.web) {
        paymentUrl = data.payment_links.web;
      } else if (data.id) {
        paymentUrl = `https://sandbox.assets.juspay.in/payment-page/order/${data.id}`;
      }

      // Redirect to payment URL if found
      if (paymentUrl) {
        toast.success("Redirecting to payment page...");
        window.location.href = paymentUrl; // Redirect to payment URL
      } else {
        console.error("Payment URL not found in response:", data);
        throw new Error('Payment URL not found in response');
      }
  
    } catch (error: any) {
      console.error("Error blocking ticket:", error);
      if (error.message === 'Authentication required') {
        toast.error("Please login to book tickets");
      } else {
        setBookingError(error.message || "Failed to block ticket");
        toast.error(error.message || "Failed to block ticket");
      }
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
                  <span className="font-medium">{selectedSeat?.seatName} (₹{selectedSeat ? calculateTotalFare(selectedSeat) : '0'})</span>
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
              
              {/* Total fare calculation with detailed breakdown */}
              <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                {selectedSeat && (
                  <>
                    {/* Display fare breakdown if available */}
                    {(selectedSeat.fare || 
                     (selectedSeat.window && selectedSeat.window.fare) || 
                     (selectedSeat.aisle && selectedSeat.aisle.fare)) && (
                      <>
                        <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400">
                          <span>Base Fare:</span>
                          <span>₹{
                            selectedSeat.fare ? selectedSeat.fare["Base Fare"] :
                            selectedSeat.window && selectedSeat.window.fare ? selectedSeat.window.fare["Base Fare"] :
                            selectedSeat.aisle && selectedSeat.aisle.fare ? selectedSeat.aisle.fare["Base Fare"] : 0
                          }</span>
                        </div>
                        <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400">
                          <span>GST:</span>
                          <span>₹{
                            selectedSeat.fare ? selectedSeat.fare["GST"] :
                            selectedSeat.window && selectedSeat.window.fare ? selectedSeat.window.fare["GST"] :
                            selectedSeat.aisle && selectedSeat.aisle.fare ? selectedSeat.aisle.fare["GST"] : 0
                          }</span>
                        </div>
                        <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400">
                          <span>Discount:</span>
                          <span>₹{
                            selectedSeat.fare ? selectedSeat.fare["Discount"] :
                            selectedSeat.window && selectedSeat.window.fare ? selectedSeat.window.fare["Discount"] :
                            selectedSeat.aisle && selectedSeat.aisle.fare ? selectedSeat.aisle.fare["Discount"] : 0
                          }</span>
                        </div>
                      </>
                    )}
                  </>
                )}
                <div className="flex justify-between font-medium mt-2">
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
                              key={seat.id || seat.seat_id}
                              onClick={() => handleSeatClick(seat)}
                              className={`p-2 rounded-lg border text-left ${
                                selectedSeat?.id === seat.id || selectedSeat?.seat_id === seat.seat_id
                                  ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20' 
                                  : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-750'
                              }`}
                            >
                              <div className="flex justify-between items-center">
                                <span className="font-medium text-gray-900 dark:text-white">
                                  {seat.label || `Seat ${seat.seatName || seat.seatNumber}`}
                                </span>
                                {(selectedSeat?.id === seat.id || selectedSeat?.seat_id === seat.seat_id) && (
                                  <Check className="h-4 w-4 text-yellow-500" />
                                )}
                              </div>
                              <span className="text-sm text-gray-500 dark:text-gray-400">₹{calculateTotalFare(seat)}</span>
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
    </div>
  );
}