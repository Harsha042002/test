// In types.ts
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  busRoutes?: BusRoute[];
  rawData?: any; // Add this property to store the complete JSON response
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
    "GST": number;
    "Discount": number;
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
  boardingPoints: (string | LocationPoint)[];
  droppingPoints: (string | LocationPoint)[];
  seats: Seat[];
  // Add the bookingInfo property
  bookingInfo?: {
    mobile: string;
    email: string;
    seat_map: Array<{
      passenger_age: number;
      seat_id: number | string;
      passenger_name: string;
      gender: string;
    }>;
    trip_id: number;
    boarding_point_id: number | string;
    dropping_point_id: number | string;
    boarding_point_time: string;
    dropping_point_time: string;
    total_collect_amount: number;
    main_category: number;
    freshcardId: number;
    freshcard: boolean;
    return_url: string;
  };
}

export interface LocationPoint {
  id: number | string;
  name: string;
  landmark?: string;
  time?: string;
  address?: string;
}