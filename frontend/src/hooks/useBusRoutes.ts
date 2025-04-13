import { useState, useEffect } from 'react';
import { Message, BusRoute, Seat } from '../types';

export function useBusRoutes(message: Message) {
  const [busRoutes, setBusRoutes] = useState<BusRoute[] | null>(null);
  
  useEffect(() => {
    // If message already has busRoutes, use those
    if (message.busRoutes && message.busRoutes.length > 0) {
      console.log("Using existing bus routes:", message.busRoutes.length);
      setBusRoutes(message.busRoutes);
      return;
    }
    
    // Try to extract JSON from message content
    if (message.content && message.content.includes('```json')) {
      try {
        console.log("Attempting to extract JSON from message content");
        // Extract the JSON part
        const jsonStr = message.content.split('```json')[1].split('```')[0];
        const jsonData = JSON.parse(jsonStr.trim());
        
        console.log("Extracted JSON data:", jsonData);
        
        if (jsonData && jsonData.trips && Array.isArray(jsonData.trips)) {
          const extractedRoutes: BusRoute[] = [];
          
          for (const trip of jsonData.trips) {
            // Make the condition less strict - we only need basic trip data
            if (trip.tripId && (trip.from || trip.to)) {
              console.log(`Processing trip ${trip.tripId}`);
              
              // Create BusRoute object
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
                seats: []
              };
              
              // Process seats from recommendations
              const seats: Seat[] = [];
              
              // Check if recommendations exist
              if (trip.recommendations) {
                for (const category in trip.recommendations) {
                  const catData = trip.recommendations[category];
                  
                  // Process window seat if available
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
                      fare: { "Base Fare": parseInt(catData.window.price) || 0, GST: 0, Discount: 0 },
                      label: `Window ${catData.window.seatNumber}`,
                      available: true,
                      hasStaticFare: true,
                      isDummy: false,
                      type: category === 'premium' ? 'Premium' :
                            category === 'budget_friendly' ? 'Budget-Friendly' : 'Regular'
                    });
                  }
                  
                  // Process aisle seat if available
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
                      fare: { "Base Fare": parseInt(catData.aisle.price) || 0, GST: 0, Discount: 0 },
                      label: `Aisle ${catData.aisle.seatNumber}`,
                      available: true,
                      hasStaticFare: true,
                      isDummy: false,
                      type: category === 'premium' ? 'Premium' :
                            category === 'budget_friendly' ? 'Budget-Friendly' : 'Regular'
                    });
                  }
                }
              } else {
                // Create at least one seat if recommendations are missing
                console.log(`No recommendations for trip ${trip.tripId}, creating default seat`);
                seats.push({
                  id: `${trip.tripId}-default-1`,
                  x: 0,
                  y: 1,
                  seatName: "1",
                  price: parseInt(trip.price) || 500,
                  totalFare: parseInt(trip.price) || 500,
                  isOccupied: false,
                  availabilityStatus: 'A',
                  isReservedForFemales: false,
                  isReservedForMales: false,
                  fare: { "Base Fare": parseInt(trip.price) || 500, GST: 0, Discount: 0 },
                  label: "Window 1",
                  available: true,
                  hasStaticFare: true,
                  isDummy: false,
                  type: 'Regular'
                });
              }
              
              // Add the processed seats to the bus route
              busRoute.seats = seats;
              
              // Add all boarding points if available
              if (trip.all_boarding_points) {
                busRoute.boardingPoints = trip.all_boarding_points;
              }
              
              // Add all dropping points if available
              if (trip.all_dropping_points) {
                busRoute.droppingPoints = trip.all_dropping_points;
              }
              
              // Add booking info if available
              if (trip.booking_info) {
                busRoute.bookingInfo = trip.booking_info;
              }
              
              // Only add routes that have at least one valid seat
              if (seats.length > 0) {
                extractedRoutes.push(busRoute);
                console.log(`Added bus route ${trip.tripId} with ${seats.length} seats`);
              }
            }
          }
          
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