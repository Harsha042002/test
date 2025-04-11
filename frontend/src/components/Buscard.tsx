import React, { useState } from 'react';
import { BusRoute, Seat } from '../types';

interface BusCardProps extends BusRoute {
  onSeatSelect: (seatId: number) => void; // Updated to accept a number
}

export const BusCard: React.FC<BusCardProps> = ({
  rating,
  duration,
  startTime,
  endTime,
  boardingPoints,
  droppingPoints,
  seats,
  onSeatSelect,
}) => {
  const [selectedSeat, setSelectedSeat] = useState<Seat | null>(null);
  const [confirmStep, setConfirmStep] = useState(false);
  const [bookingError, setBookingError] = useState<string | null>(null);

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
    if (selectedSeat) {
      onSeatSelect(Number(selectedSeat.id));
      setSelectedSeat(null);
      setConfirmStep(false);
    }
  };

  const isSeatSelected = (seat: Seat) => selectedSeat?.id === seat.id;

  const renderSeatsByCategory = (category: 'Regular' | 'Budget-Friendly' | 'Premium', colorScheme: string) => {
    return (
      <div>
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">{category} Seats</h4>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {seats
            ?.filter(seat => seat.type === category)
            .map(seat => (
              <button
                key={seat.id}
                onClick={() => seat.isOccupied || seat.isDummy ? null : handleSeatClick(seat)}
                disabled={seat.isOccupied || !!selectedSeat}
                className={`flex items-center justify-between p-2 rounded-lg text-sm transition-colors
                  ${seat.isOccupied
                    ? 'bg-gray-100 dark:bg-gray-700 cursor-not-allowed'
                    : isSeatSelected(seat)
                      ? `${colorScheme} border-${colorScheme}-400 dark:border-${colorScheme}-600`
                      : `${colorScheme} hover:${colorScheme}-300 dark:hover:${colorScheme}-800 border border-${colorScheme}-300 dark:border-${colorScheme}-700`
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

  if (confirmStep && selectedSeat) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden h-full">
        <Header />
        <div className="p-4">
          <h4 className="text-lg font-semibold mb-2 text-gray-900 dark:text-gray-100">Confirm Ticket</h4>
          <p className="mb-4">
            You have selected seat <strong>{selectedSeat.seatName}</strong> (Price: ₹{selectedSeat.price}).
          </p>
          <div className="flex gap-4">
            <button
              onClick={handleBack}
              className="w-full bg-gray-300 hover:bg-gray-400 text-gray-800 py-2 px-4 rounded-lg transition-colors duration-200"
            >
              Back
            </button>
            <button
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
            <p className="text-base text-gray-900 dark:text-gray-100 leading-tight">{boardingPoints[0]}</p>
          </div>
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 leading-tight">Drop Point</h4>
            <p className="text-base text-gray-900 dark:text-gray-100 leading-tight">{droppingPoints[0]}</p>
          </div>
        </div>
        <div className="space-y-4">
          {renderSeatsByCategory('Regular', 'bg-blue-200')}
          <hr className="border-gray-200 dark:border-gray-700" />
          {renderSeatsByCategory('Budget-Friendly', 'bg-green-200')}
          <hr className="border-gray-200 dark:border-gray-700" />
          {renderSeatsByCategory('Premium', 'bg-purple-200')}
        </div>
      </div>
    </div>
  );
};