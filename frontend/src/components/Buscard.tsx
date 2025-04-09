import React, { useState } from 'react';
import { BusRoute, Seat } from '../types';

interface BusCardProps extends BusRoute {
  onSeatSelect: (seatId: string) => void;
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
  // Only one seat is allowed.
  const [selectedSeat, setSelectedSeat] = useState<Seat | null>(null);
  // Track whether we're in the confirmation step.
  const [confirmStep, setConfirmStep] = useState(false);
  const [bookingError] = useState<string | null>(null);

  // Constant header rendered in both steps.
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

  // When a seat is clicked, if none is selected yet, set it and go to confirmation.
  const handleSeatClick = (seat: Seat) => {
    if (!selectedSeat) {
      setSelectedSeat(seat);
      setConfirmStep(true);
    }
  };

  // Option to go back to seat selection.
  const handleBack = () => {
    setConfirmStep(false);
    setSelectedSeat(null);
  };

  // Confirm ticket – here, we simply call onSeatSelect with the selected seat's id.
  const handleConfirmTicket = () => {
    if (selectedSeat) {
      onSeatSelect(selectedSeat.id);
      // Optionally clear selection here.
    }
  };

  // Helper to check if a seat is selected.
  const isSeatSelected = (seat: Seat) => {
    return selectedSeat?.id === seat.id;
  };

  if (confirmStep && selectedSeat) {
    // Confirmation view – always show header on top.
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden h-full">
        <Header />
        <div className="p-4">
          <h4 className="text-lg font-semibold mb-2 text-gray-900 dark:text-gray-100">
            Confirm Ticket
          </h4>
          <p className="mb-4">
            You have selected seat <strong>W{selectedSeat.id}</strong> (Price: ₹{selectedSeat.price}).
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

  // Seat selection view – always show header above the seat options.
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden h-full">
      <Header />
      <div className="p-4">
        {bookingError && (
          <div className="mb-2 p-3 bg-red-50 dark:bg-red-900/30 rounded-lg text-red-700 dark:text-red-200">
            {bookingError}
          </div>
        )}

        {/* (Optional) Summary for selected seat if already chosen */}
        {selectedSeat && (
          <div className="mb-2 p-3 bg-blue-50 dark:bg-blue-900/30 rounded-lg">
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Selected Seat: W{selectedSeat.id} - ₹{selectedSeat.price}
            </p>
          </div>
        )}

        {/* Pickup / Drop Points */}
        <div className="flex justify-between items-center mb-2 p-3 bg-gray-100 dark:bg-gray-700 rounded-lg">
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 leading-tight">
              Pickup Point
            </h4>
            <p className="text-base text-gray-900 dark:text-gray-100 leading-tight">
              {boardingPoints[0]}
            </p>
          </div>
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 leading-tight">
              Drop Point
            </h4>
            <p className="text-base text-gray-900 dark:text-gray-100 leading-tight">
              {droppingPoints[0]}
            </p>
          </div>
        </div>

        {/* Seat Options by Category */}
        <div className="space-y-4">
          {/* Regular Seats */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Regular Seats</h4>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              {seats
                .filter(seat => seat.type === 'Regular')
                .map(seat => (
                  <button
                    key={seat.id}
                    onClick={() => seat.available && handleSeatClick(seat)}
                    disabled={!seat.available || !!selectedSeat}
                    className={`flex items-center justify-between p-2 rounded-lg text-sm transition-colors
                      ${!seat.available 
                        ? 'bg-gray-100 dark:bg-gray-700 cursor-not-allowed'
                        : isSeatSelected(seat)
                          ? 'bg-blue-300 dark:bg-blue-700 border-blue-400 dark:border-blue-600'
                          : 'bg-blue-200 dark:bg-blue-700 hover:bg-blue-300 dark:hover:bg-blue-800 border border-blue-300 dark:border-blue-700'
                      }`}
                  >
                    <span className="text-gray-600 dark:text-gray-100 leading-tight">W{seat.id}</span>
                    <span className="font-semibold text-blue-600 dark:text-blue-400 leading-tight">₹{seat.price}</span>
                  </button>
                ))}
            </div>
          </div>

          <hr className="border-gray-200 dark:border-gray-700" />

          {/* Budget-Friendly Seats */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Budget-Friendly Seats
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              {seats
                .filter(seat => seat.type === 'Budget-Friendly')
                .map(seat => (
                  <button
                    key={seat.id}
                    onClick={() => seat.available && handleSeatClick(seat)}
                    disabled={!seat.available || !!selectedSeat}
                    className={`flex items-center justify-between p-2 rounded-lg text-sm transition-colors
                      ${!seat.available 
                        ? 'bg-gray-100 dark:bg-gray-700 cursor-not-allowed'
                        : isSeatSelected(seat)
                          ? 'bg-green-300 dark:bg-green-700 border-green-400 dark:border-green-600'
                          : 'bg-green-200 dark:bg-green-700 hover:bg-green-300 dark:hover:bg-green-800 border border-green-300 dark:border-green-700'
                      }`}
                  >
                    <span className="text-gray-600 dark:text-gray-100 leading-tight">W{seat.id}</span>
                    <span className="font-semibold text-green-600 dark:text-green-400 leading-tight">₹{seat.price}</span>
                  </button>
                ))}
            </div>
          </div>

          <hr className="border-gray-200 dark:border-gray-700" />

          {/* Premium Seats */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Premium Seats</h4>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              {seats
                .filter(seat => seat.type === 'Premium')
                .map(seat => (
                  <button
                    key={seat.id}
                    onClick={() => seat.available && handleSeatClick(seat)}
                    disabled={!seat.available || !!selectedSeat}
                    className={`flex items-center justify-between p-2 rounded-lg text-sm transition-colors
                      ${!seat.available 
                        ? 'bg-gray-100 dark:bg-gray-700 cursor-not-allowed'
                        : isSeatSelected(seat)
                          ? 'bg-purple-300 dark:bg-purple-700 border-purple-400 dark:border-purple-600'
                          : 'bg-purple-200 dark:bg-purple-700 hover:bg-purple-300 dark:hover:bg-purple-800 border border-purple-300 dark:border-purple-700'
                      }`}
                  >
                    <span className="text-gray-600 dark:text-gray-100 leading-tight">W{seat.id}</span>
                    <span className="font-semibold text-purple-600 dark:text-purple-400 leading-tight">₹{seat.price}</span>
                  </button>
                ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};