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
  seats: {
    id: string;
    type: string;
    price: number;
    available: boolean;
    label: string;
  }[];
} 