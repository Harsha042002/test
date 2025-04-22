// In types.ts
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  busRoutes?: BusRoute[];
  rawData?: any; // Add this property to store the complete JSON response
isLoading?: boolean; // Optional property to indicate loading state
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  lastUpdated: Date;
}

export interface FareDetails {
  "Base Fare": number;
  "GST": number;
  "Discount": number;
}

export interface SeatPosition {
  seatNumber: string;
  price: number | string;
  seat_id: number | string;
  fare?: FareDetails;
}

export interface Seat {
  id?: number | string;
  seat_id?: number | string;
  x?: number;
  y?: number;
  totalFare?: number;
  seatName?: string;
  isOccupied?: boolean;
  availabilityStatus?: string; // "A", "M", "F", etc.
  isReservedForFemales?: boolean;
  isReservedForMales?: boolean;
  fare?: FareDetails;
  label?: string;
  available?: boolean;
  hasStaticFare?: boolean;
  isDummy?: boolean;
  type?: 'Regular' | 'Budget-Friendly' | 'Premium'; // Added type
  price?: number | string; // Added price, can be a string or number
  
  // For nested seat information
  window?: SeatPosition;
  aisle?: SeatPosition;
  
  // Additional properties that might be present in the recommendation object
  seatNumber?: string;
}

export interface BookingInfo {
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
  fare?: FareDetails;
  main_category: number;
  freshcardId: number;
  freshcard: boolean;
  return_url: string;
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
  
  // Properties for recommendations
  recommendations?: {
    reasonable?: {
      window?: Seat;
      aisle?: Seat;
    };
    premium?: {
      window?: Seat;
      aisle?: Seat;
    };
    budget_friendly?: {
      window?: Seat;
      aisle?: Seat;
    };
  };
  
  // Add the bookingInfo property
  bookingInfo?: BookingInfo;
}

export interface LocationPoint {
  id: number | string;
  name: string;
  landmark?: string;
  time?: string;
  latitude?: number;
  longitude?: number;
  address?: string;
}