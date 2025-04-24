import { useState, useEffect } from 'react';
import { ChevronDown, Loader2, Star } from 'lucide-react';
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
  const [selectedSeat, setSelectedSeat] = useState<Seat | null>(null);
  const [isBoardingOpen, setIsBoardingOpen] = useState(false);
  const [isDroppingOpen, setIsDroppingOpen] = useState(false);
  const [selectedBoardingPoint, setSelectedBoardingPoint] = useState<LocationPoint | null>(null);
  const [selectedDroppingPoint, setSelectedDroppingPoint] = useState<LocationPoint | null>(null);
  const [confirmStep, setConfirmStep] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [bookingError, setBookingError] = useState<string | null>(null);
const [isCardBodyVisible, setIsCardBodyVisible] = useState(false); // State to toggle card body visibility

  const toggleCardBody = () => {
    setIsCardBodyVisible(!isCardBodyVisible);
  };

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
    setConfirmStep(true);
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
      const authToken = localStorage.getItem('access_token');
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
            <p className="text-sm text-gray-900 dark:text-gray-100 leading-tight">
              {selectedPoint ? selectedPoint.name : 'Select Location'}
            </p>
            {selectedPoint && selectedPoint.landmark && (
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
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
                  <p className="text-sm text-gray-500 dark:text-gray-400">{point.landmark}</p>
                )}
                {point.time && (
                  <p className="text-sm text-blue-500 dark:text-blue-400">{point.time}</p>
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="bg-white dark:bg-gray-800 shadow-md rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 transition-all duration-200">
     {/* Bus header */}
<div className="w-full px-2 py-2 flex justify-between items-center bg-[#1765F3]">
  <div>
    <h3 className="text-lg sm:text-xl md:text-2xl lg:text-lg xl:text-lg font-semibold text-white">
      {from} to {to}
    </h3>
    <div className="flex justify-between items-center mt-1">
      {/* Star and Rating */}
      <div className="flex items-center gap-1">
        <Star className="h-3 w-3 sm:h-4 sm:w-4 md:h-5 md:w-5 text-[#FBE822] fill-current mr-1" />
        <span className="text-sm sm:text-base md:text-lg lg:text-lg xl:text-lg text-white">
          {rating.toFixed(1)}
        </span>
      </div>
      {/* Duration */}
      <div>
        <span className="text-sm sm:text-base md:text-lg lg:text-lg xl:text-lg text-white">
          {duration}
        </span>
      </div>
    </div>
  </div>
{/* Select/Hide Button */}
        <button
          onClick={toggleCardBody}
          className="px-3 py-1 bg-white text-[#1765F3] font-medium text-sm rounded-lg shadow-md hover:bg-gray-100 transition"
        >
          {isCardBodyVisible ? 'Hide' : 'Select'}
        </button>
</div>
      
{/* Conditional rendering for card body */}
      {isCardBodyVisible && (
        <div>
            {/* Conditional rendering for selection or confirmation step */}
                {confirmStep ? (
            /* Confirmation Step */
            <div className="px-4 py-4 space-y-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                   Confirm Your Booking
              </h3>
                            <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                  Selected Seat:
                  </span>
                  <span className="font-medium text-sm">
                   {selectedSeat?.seatName} (₹{selectedSeat ? calculateTotalFare(selectedSeat) : '0'})
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                  Boarding Point:
                  </span>
                  <span className="font-medium text-sm">
                   {selectedBoardingPoint?.name}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                  Dropping Point:
                  </span>
                  <span className="font-medium text-sm">
                    {selectedDroppingPoint?.name}
                  </span>
                </div>
              </div>
                            
              {/* Total fare calculation with detailed breakdown */}
              <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                {selectedSeat && (
                  <>
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
                <div className="flex justify-between font-medium mt-2 text-sm">
                  <span>Total Fare:</span>
                  <span>₹{selectedSeat ? calculateTotalFare(selectedSeat) : '0'}</span>
                </div>
              </div>
                            
              {/* Action buttons */}
              <div className="flex space-x-2 pt-2">
                <button
                  onClick={() => setConfirmStep(false)} // Go back to selection step
                  className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg flex-1 transition-colors hover:bg-gray-200 dark:hover:bg-gray-600 text-sm"
                >
                  Back
                </button>
                <button
                  onClick={handleBookTicket}
                  disabled={isProcessing}
                  className="px-2 py-1 bg-yellow-500 text-white rounded-lg flex-1 transition-colors hover:bg-yellow-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center text-sm"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="animate-spin h-3 w-3 mr-1" />
                      Processing...
                    </>
                  ) : (
                    'Confirm & Pay'
                  )}
                </button>
              </div>
            </div>
          ) : (
            /* Selection Step */
            <div className="px-4 py-4 space-y-4">
              {/* Pickup and dropping dropdowns */}
          <div className="space-y-4">
            <LocationDropdown
              label="Boarding Point"
              points={allBoardingPoints || []}
              selectedPoint={selectedBoardingPoint}
              setSelectedPoint={setSelectedBoardingPoint}
              isOpen={isBoardingOpen}
              setIsOpen={setIsBoardingOpen}
              defaultPoint={typeof boardingPoints[0] === 'string' ? boardingPoints[0] : undefined}
            />
            <LocationDropdown
              label="Dropping Point"
              points={allDroppingPoints || []}
              selectedPoint={selectedDroppingPoint}
              setSelectedPoint={setSelectedDroppingPoint}
              isOpen={isDroppingOpen}
              setIsOpen={setIsDroppingOpen}
              defaultPoint={typeof droppingPoints[0] === 'string' ? droppingPoints[0] : undefined}
            />
          </div>

          {/* Seat selection area */}
          <div>
            <h3 className="text-md font-medium text-gray-900 dark:text-white mb-3">
              Select Seat
            </h3>
            <div className="space-y-4">
              {Object.keys(seatsByType).map((type) => (
                <div key={type} className="space-y-2">
                  <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400">
                    {type} Seats
                  </h4>
                  <div className="grid grid-cols-2 gap-2">
                    {seatsByType[type].map((seat) => (
                      <button
                        key={seat.id || seat.seat_id}
                        onClick={() => handleSeatClick(seat)} // Directly proceed to the next step
                        className={`p-1 rounded-lg border text-left ${
                          selectedSeat?.id === seat.id || selectedSeat?.seat_id === seat.seat_id
                            ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
                            : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-750'
                        }`}
                      >
                        <div className="flex justify-between items-center">
                          <span className="font-medium text-sm text-gray-900 dark:text-white">
                            {seat.label || `Seat ${seat.seatName || seat.seatNumber}`}
                          </span>
                        </div>
                        <span className="text-sm text-gray-500 dark:text-gray-400">
                          ₹{calculateTotalFare(seat)}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
</div>
          )}
                                </div>
                    )}
    </div>
  );
}