// src/components/Buscard.tsx
import React, { useState, useEffect } from 'react';
import { BusRoute, Seat } from '../types';
import { toast } from 'react-toastify';
import { ChevronDown } from 'lucide-react';

// Add interfaces for boarding/dropping points from the data
interface LocationPoint {
  id: number | string;
  name: string;
  landmark?: string;
  time?: string;
  address?: string;
}

interface BusCardProps extends BusRoute {
  onSeatSelect: (seatId: number | string, boardingPoint: LocationPoint, droppingPoint: LocationPoint) => void;
  allBoardingPoints?: LocationPoint[];
  allDroppingPoints?: LocationPoint[];
}

export const BusCard: React.FC<BusCardProps> = ({
  id,
  rating,
  duration,
  startTime,
  endTime,
  boardingPoints,
  droppingPoints,
  seats,
  onSeatSelect,
  from,
  to,
  allBoardingPoints = [],
  allDroppingPoints = []
}) => {
  const [selectedSeat, setSelectedSeat] = useState<Seat | null>(null);
  const [confirmStep, setConfirmStep] = useState(false);
  const [bookingError, setBookingError] = useState<string | null>(null);
  const [boardingPointDropdownOpen, setBoardingPointDropdownOpen] = useState(false);
  const [droppingPointDropdownOpen, setDroppingPointDropdownOpen] = useState(false);
  const [selectedBoardingPoint, setSelectedBoardingPoint] = useState<LocationPoint | null>(null);
  const [selectedDroppingPoint, setSelectedDroppingPoint] = useState<LocationPoint | null>(null);
  
  // Add this useEffect for debug logging right after the state declarations
  useEffect(() => {
    // Log debugging info when the component mounts
    console.log(`BusCard mounted for route ${id} (${from} to ${to})`);
    console.log(`This card has ${seats?.length || 0} seats`);
    
    // Ensure we have valid boarding points
    if (!allBoardingPoints || allBoardingPoints.length === 0) {
      console.warn(`No boarding points provided for bus route ${id}`);
    }
    
    // Ensure we have valid dropping points
    if (!allDroppingPoints || allDroppingPoints.length === 0) {
      console.warn(`No dropping points provided for bus route ${id}`);
    }
    
    // Make sure we have seats
    if (!seats || seats.length === 0) {
      console.warn(`No seats provided for bus route ${id}`);
    }
  }, [id, from, to, seats, allBoardingPoints, allDroppingPoints]);

  // Create default boarding/dropping points if none provided
  useEffect(() => {
    // Set default boarding points based on the first boardingPoint in the list
    if (boardingPoints && boardingPoints.length > 0 && !selectedBoardingPoint) {
      // Check if we have all boarding points from backend
      if (allBoardingPoints && allBoardingPoints.length > 0) {
        const defaultPoint = allBoardingPoints.find(p => p.name === boardingPoints[0]) || 
          { id: `${id}-default-boarding`, name: boardingPoints[0], landmark: 'Main Stop' };
        setSelectedBoardingPoint(defaultPoint);
      } else {
        // Create a default boarding point from the string
        setSelectedBoardingPoint({
          id: `${id}-default-boarding`,
          name: boardingPoints[0],
          landmark: 'Main Stop'
        });
      }
    }
    
    // Set default dropping points based on the first droppingPoint in the list
    if (droppingPoints && droppingPoints.length > 0 && !selectedDroppingPoint) {
      // Check if we have all dropping points from backend
      if (allDroppingPoints && allDroppingPoints.length > 0) {
        const defaultPoint = allDroppingPoints.find(p => p.name === droppingPoints[0]) || 
          { id: `${id}-default-dropping`, name: droppingPoints[0], landmark: 'Main Stop' };
        setSelectedDroppingPoint(defaultPoint);
      } else {
        // Create a default dropping point from the string
        setSelectedDroppingPoint({
          id: `${id}-default-dropping`,
          name: droppingPoints[0],
          landmark: 'Main Stop'
        });
      }
    }
  }, [
    id,
    boardingPoints, 
    droppingPoints, 
    allBoardingPoints, 
    allDroppingPoints, 
    selectedBoardingPoint, 
    selectedDroppingPoint
  ]);

  const Header = () => (
    <div className="bg-gradient-to-r from-blue-600 to-blue-700 p-4 text-white w-full">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-1">
          <span className="text-base font-semibold leading-tight">
            {startTime} - {endTime}
          </span>
          <span className="bg-[#FBE822] text-[#1765F3] px-2 py-0.5 rounded-full text-xs font-medium leading-tight">
            ★ {rating}
          </span>
        </div>
        <div>
          <span className="font-semibold leading-tight">{duration}</span>
        </div>
      </div>
      <div className="text-sm mt-1 font-medium">{from} to {to}</div>
    </div>
  );

  const handleSeatClick = (seat: Seat) => {
    if (seat.isOccupied || seat.isDummy) {
      setBookingError('This seat is not available for selection.');
      return;
    }
  
    setBookingError(null); // Clear any previous error
    setSelectedSeat(seat);
    setConfirmStep(true);
  };

  const handleBack = () => {
    setConfirmStep(false);
    setSelectedSeat(null);
  };

  const handleConfirmTicket = () => {
    if (selectedSeat && selectedBoardingPoint && selectedDroppingPoint) {
      onSeatSelect(selectedSeat.id, selectedBoardingPoint, selectedDroppingPoint);
      toast.success(`Selected seat ${selectedSeat.seatName} for booking!`);
      setSelectedSeat(null);
      setConfirmStep(false);
    } else {
      if (!selectedBoardingPoint) {
        setBookingError('Please select a boarding point');
      } else if (!selectedDroppingPoint) {
        setBookingError('Please select a dropping point');
      }
    }
  };

  const isSeatSelected = (seat: Seat) => selectedSeat?.id === seat.id;

  const renderSeatsByCategory = (category: 'Regular' | 'Budget-Friendly' | 'Premium', colorScheme: string) => {
    const categorySeats = seats?.filter(seat => seat.type === category) || [];
    
    if (categorySeats.length === 0) {
      return null;
    }
    
    return (
      <div>
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">{category} Seats</h4>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {categorySeats.map(seat => (
            <button
              key={seat.id}
              onClick={() => seat.isOccupied || seat.isDummy ? null : handleSeatClick(seat)}
              disabled={seat.isOccupied || !!selectedSeat && selectedSeat.id !== seat.id}
              className={`flex items-center justify-between p-2 rounded-lg text-sm transition-colors
                ${seat.isOccupied
                  ? 'bg-gray-100 dark:bg-gray-700 cursor-not-allowed'
                  : isSeatSelected(seat)
                    ? `${colorScheme} border-${colorScheme.replace('bg-', 'border-')}-400 dark:border-${colorScheme.replace('bg-', 'border-')}-600`
                    : `${colorScheme} hover:${colorScheme.replace('bg-', 'hover:bg-')}-300 dark:hover:${colorScheme.replace('bg-', 'hover:bg-')}-800 border border-${colorScheme.replace('bg-', 'border-')}-300 dark:border-${colorScheme.replace('bg-', 'border-')}-700`
                }`}
            >
              <span className="text-gray-600 dark:text-gray-100 leading-tight">{seat.seatName}</span>
              <span className="font-semibold text-gray-600 dark:text-gray-400 leading-tight">₹{seat.price}</span>
            </button>
          ))}
        </div>
      </div>
    );
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
              {selectedPoint?.name || 'Select Location'}
            </p>
            {selectedPoint?.landmark && (
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

  if (confirmStep && selectedSeat) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden h-full">
        <Header />
        <div className="p-4 space-y-4">
          <h4 className="text-lg font-semibold mb-2 text-gray-900 dark:text-gray-100">Confirm Ticket</h4>
          
          {bookingError && (
            <div className="p-3 bg-red-50 dark:bg-red-900/30 rounded-lg text-red-700 dark:text-red-200">
              {bookingError}
            </div>
          )}
          
          <div className="space-y-4">
            <p className="font-medium">
              You have selected seat <span className="font-bold">{selectedSeat.seatName}</span> 
              <span className="text-green-600 dark:text-green-400 font-bold"> (₹{selectedSeat.price})</span>
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <LocationDropdown 
                label="Select Boarding Point" 
                points={allBoardingPoints} 
                selectedPoint={selectedBoardingPoint}
                setSelectedPoint={setSelectedBoardingPoint}
                isOpen={boardingPointDropdownOpen}
                setIsOpen={setBoardingPointDropdownOpen}
                defaultPoint={boardingPoints[0]}
              />
              
              <LocationDropdown 
                label="Select Dropping Point" 
                points={allDroppingPoints} 
                selectedPoint={selectedDroppingPoint}
                setSelectedPoint={setSelectedDroppingPoint}
                isOpen={droppingPointDropdownOpen}
                setIsOpen={setDroppingPointDropdownOpen}
                defaultPoint={droppingPoints[0]}
              />
            </div>
          </div>
          
          <div className="flex gap-4 pt-2">
            <button
              type="button"
              onClick={handleBack}
              className="w-full bg-gray-300 hover:bg-gray-400 text-gray-800 py-2 px-4 rounded-lg transition-colors duration-200"
            >
              Back
            </button>
            <button
              type="button"
              onClick={handleConfirmTicket}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg transition-colors duration-200"
            >
              Confirm Ticket
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden h-full">
      <Header />
      <div className="p-4">
        {bookingError && (
          <div className="mb-2 p-3 bg-red-50 dark:bg-red-900/30 rounded-lg text-red-700 dark:text-red-200">
            {bookingError}
          </div>
        )}
        <div className="flex justify-between items-center mb-2 p-3 bg-gray-100 dark:bg-gray-700 rounded-lg">
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 leading-tight">Pickup Point</h4>
            <p className="text-base text-gray-900 dark:text-gray-100 leading-tight">{boardingPoints[0] || 'N/A'}</p>
          </div>
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 leading-tight">Drop Point</h4>
            <p className="text-base text-gray-900 dark:text-gray-100 leading-tight">{droppingPoints[0] || 'N/A'}</p>
          </div>
        </div>
        <div className="space-y-4">
          {renderSeatsByCategory('Regular', 'bg-blue-200')}
          {renderSeatsByCategory('Regular', 'bg-blue-200') && renderSeatsByCategory('Budget-Friendly', 'bg-green-200') && (
            <hr className="border-gray-200 dark:border-gray-700" />
          )}
          {renderSeatsByCategory('Budget-Friendly', 'bg-green-200')}
          {(renderSeatsByCategory('Regular', 'bg-blue-200') || renderSeatsByCategory('Budget-Friendly', 'bg-green-200')) && 
           renderSeatsByCategory('Premium', 'bg-purple-200') && (
            <hr className="border-gray-200 dark:border-gray-700" />
          )}
          {renderSeatsByCategory('Premium', 'bg-purple-200')}
          
          {seats && seats.length > 0 && (
            <div className="flex justify-end mt-4">
              <button 
                type="button"
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                onClick={() => toast.info("Please select a seat first to proceed with booking.")}
              >
                Book Now
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};