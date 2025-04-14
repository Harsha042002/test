import { useState, useEffect } from 'react';
import { Message, BusRoute, Seat } from '../types';

export function useBusRoutes(message: Message): BusRoute[] | null {
  const [busRoutes, setBusRoutes] = useState<BusRoute[] | null>(null);
  
  useEffect(() => {
    // If message already has busRoutes, use those.
    if (message.busRoutes && message.busRoutes.length > 0) {
      console.log("Using existing bus routes:", message.busRoutes.length);
      setBusRoutes(message.busRoutes);
      return;
    }
    
    // Try to extract JSON from message content.
    if (message.content && message.content.includes('```json')) {
      try {
        console.log("Attempting to extract JSON from message content");
        // Extract the JSON part from the message content.
        const jsonStr = message.content.split('```json')[1].split('```')[0];
        const jsonData = JSON.parse(jsonStr.trim());
        console.log("Extracted JSON data:", jsonData);
        
        if (jsonData && jsonData.trips && Array.isArray(jsonData.trips)) {
          // Inline helper function: transform a seat object to preserve all fields including "fare"
          const transformSeat = (seat: any): Seat => ({
            ...seat,
            fare: seat.fare || {}  // Ensure the fare field is preserved, or default to an empty object.
          });
          
          const extractedRoutes: BusRoute[] = jsonData.trips.map((trip: any) => {
            const seats: Seat[] = [];
            
            // If recommendations exist, process seats from them.
            if (trip.recommendations) {
              for (const category in trip.recommendations) {
                const catData = trip.recommendations[category];
                
                // Process window seat if available.
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
                    fare: transformSeat(catData.window).fare,
                    label: `Window ${catData.window.seatNumber}`,
                    available: true,
                    hasStaticFare: true,
                    isDummy: false,
                    type: category === 'premium'
                      ? 'Premium'
                      : category === 'budget_friendly'
                      ? 'Budget-Friendly'
                      : 'Regular'
                  });
                }
                
                // Process aisle seat if available.
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
                    fare: transformSeat(catData.aisle).fare,
                    label: `Aisle ${catData.aisle.seatNumber}`,
                    available: true,
                    hasStaticFare: true,
                    isDummy: false,
                    type: category === 'premium'
                      ? 'Premium'
                      : category === 'budget_friendly'
                      ? 'Budget-Friendly'
                      : 'Regular'
                  });
                }
              }
            }
            
            const busRoute: BusRoute = {
              id: trip.tripId,
              from: trip.from || 'Unknown',
              to: trip.to || 'Unknown',
              rating: parseFloat(trip.rating) || 4.5,
              duration: trip.duration || "Unknown",
              startTime: trip.departureTime || "Unknown",
              endTime: trip.arrivalTime || "Unknown",
              boardingPoints: [trip.boardingPoint || "Default Boarding"],
              droppingPoints: [trip.droppingPoint || "Default Dropping"],
              seats: seats
            };
            
            // Override boarding/dropping points if provided in full arrays.
            if (trip.all_boarding_points) {
              busRoute.boardingPoints = trip.all_boarding_points;
            }
            if (trip.all_dropping_points) {
              busRoute.droppingPoints = trip.all_dropping_points;
            }
            
            // Attach booking info if available.
            if (trip.booking_info) {
              busRoute.bookingInfo = trip.booking_info;
            }
            
            return busRoute;
          });
          
          if (extractedRoutes.length > 0) {
            console.log(`Created ${extractedRoutes.length} bus routes from message content`);
            setBusRoutes(extractedRoutes);
          }
        }
      } catch (error) {
        console.error("Error extracting bus routes from message content:", error);
      }
    }
  }, [message]);
  
  return busRoutes;
}
