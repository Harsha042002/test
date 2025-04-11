export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  busRoutes?: BusRoute[];
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  lastUpdated: Date;
}


export interface Seat {
  id: number | string;
  x: number;
  y: number;
  totalFare: number;
  seatName: string;
  isOccupied: boolean;
  availabilityStatus: string; // "A", "M", "F", etc.
  isReservedForFemales: boolean;
  isReservedForMales: boolean;
  fare: {
    "Base Fare": number;
    GST: number;
    Discount: number;
  };
  label: string;
  available: boolean;
  hasStaticFare: boolean;
  isDummy: boolean;
  type: 'Regular' | 'Budget-Friendly' | 'Premium'; // Added type
  price: number; // Added price
}

export interface BusRoute {
  id: string;
  from: string;
  to: string;
  rating: number;
  duration: string;
  startTime: string;
  endTime: string;
  boardingPoints: string[];
  droppingPoints: string[];
  seats: Seat[]; // Use the Seat type here
}