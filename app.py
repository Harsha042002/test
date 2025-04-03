import os
import json
import aiohttp
import re
import math
import tempfile
import requests
import asyncio
import redis
import time
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from anthropic import Anthropic
from dotenv import load_dotenv
import uuid
import chromadb
import tiktoken
import atexit

load_dotenv()


# Updated Redis configuration using Upstash
REDIS_HOST = 'tops-buffalo-19522.upstash.io'
REDIS_PORT = 6379
REDIS_PASSWORD = 'AUxCAAIjcDFjZGUyYmMyYWI2ZWU0YzE5Yjg5NDBiYWU5ZWE0ZWIxNXAxMA'
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True,
    ssl=True
)

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    await fresh_bus_assistant.init_http_session()
    await fresh_bus_assistant.init_system_prompt()
    yield
    await fresh_bus_assistant.cleanup()

# Initialize external clients and tokenizers
anthropic_client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    timeout=45  # Set a longer timeout for all requests
)
voyage_api_key = os.getenv("VOYAGE_API_KEY")
chroma_client = chromadb.Client()
tokenizer = tiktoken.get_encoding("cl100k_base")  # Claude's encoding

# Request models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    stream: Optional[bool] = True
    gender: Optional[str] = None
    location: Optional[Dict[str, float]] = None

class InitializeRequest(BaseModel):
    system_prompt_path: str = "./system_prompt/qa_prompt.txt"

# Response models
class ConversationSummary(BaseModel):
    conversation_id: str
    session_id: str
    timestamp: str
    user_query: str
    message_count: int

class ConversationDetail(BaseModel):
    conversation_id: str
    session_id: str
    timestamp: str
    messages: List[Dict[str, str]]

# Language codes mapping for Fireworks API
LANGUAGE_CODES = {
    "english": "en",
    "hindi": "hi",
    "telugu": "te",
    "kannada": "kn",
    "tamil": "ta",
    "malayalam": "ml"
}
LANGUAGE_CODES_REVERSE = {v: k for k, v in LANGUAGE_CODES.items()}

#################################
# Redis Conversation Manager
#################################
class RedisConversationManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.conversation_prefix = "fresh_bus:conversation:"
        self.session_index_prefix = "fresh_bus:session_index:"
    
    def save_conversation(self, session_id, messages):
        try:
            if not messages:
                return None
            
            conversation_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Create conversation data
            conversation_data = {
                "conversation_id": conversation_id,
                "session_id": session_id,
                "timestamp": timestamp,
                "messages": json.dumps(messages)
            }
            
            # Save conversation
            key = f"{self.conversation_prefix}{conversation_id}"
            self.redis.hmset(key, conversation_data)
            
            # Add to session index
            self.redis.sadd(f"{self.session_index_prefix}{session_id}", conversation_id)
            
            # Add to global index
            self.redis.zadd("fresh_bus:conversations", {conversation_id: time.time()})
            
            return conversation_id
        except Exception as e:
            print(f"Redis save_conversation error: {e}")
            return None
    
    def get_conversation(self, conversation_id):
        try:
            key = f"{self.conversation_prefix}{conversation_id}"
            conversation = self.redis.hgetall(key)
            if not conversation:
                return None
            
            conversation["messages"] = json.loads(conversation["messages"])
            return conversation
        except Exception as e:
            print(f"Redis get_conversation error: {e}")
            return None
    
    def get_conversations_by_session(self, session_id):
        try:
            conversation_ids = self.redis.smembers(f"{self.session_index_prefix}{session_id}")
            conversations = []
            
            for conv_id in conversation_ids:
                conversation = self.get_conversation(conv_id)
                if conversation:
                    # Create a summary for listing
                    conversations.append({
                        "conversation_id": conversation["conversation_id"],
                        "session_id": conversation["session_id"],
                        "timestamp": conversation["timestamp"],
                        "user_query": conversation["messages"][0]["content"] if conversation["messages"] else "",
                        "message_count": len(conversation["messages"])
                    })
            
            # Sort by timestamp (newest first)
            conversations.sort(key=lambda x: x["timestamp"], reverse=True)
            return conversations
        except Exception as e:
            print(f"Redis get_conversations_by_session error: {e}")
            return []
    
    def get_all_conversations(self, limit=100, offset=0):
        try:
            # Get conversation IDs sorted by time (newest first)
            conversation_ids = self.redis.zrevrange("fresh_bus:conversations", offset, offset + limit - 1)
            conversations = []
            
            for conv_id in conversation_ids:
                conversation = self.get_conversation(conv_id)
                if conversation:
                    # Create a summary for listing
                    conversations.append({
                        "conversation_id": conversation["conversation_id"],
                        "session_id": conversation["session_id"],
                        "timestamp": conversation["timestamp"],
                        "user_query": conversation["messages"][0]["content"] if conversation["messages"] else "",
                        "message_count": len(conversation["messages"])
                    })
            
            return conversations
        except Exception as e:
            print(f"Redis get_all_conversations error: {e}")
            return []
    
    def delete_conversation(self, conversation_id):
        try:
            # Get the session_id first
            key = f"{self.conversation_prefix}{conversation_id}"
            session_id = self.redis.hget(key, "session_id")
            
            if session_id:
                # Remove from session index
                self.redis.srem(f"{self.session_index_prefix}{session_id}", conversation_id)
            
            # Remove from global index
            self.redis.zrem("fresh_bus:conversations", conversation_id)
            
            # Delete the conversation
            self.redis.delete(key)
            
            return True
        except Exception as e:
            print(f"Redis delete_conversation error: {e}")
            return False
    
    def delete_session_conversations(self, session_id):
        try:
            conversation_ids = self.redis.smembers(f"{self.session_index_prefix}{session_id}")
            
            for conv_id in conversation_ids:
                # Remove from global index
                self.redis.zrem("fresh_bus:conversations", conv_id)
                # Delete the conversation
                self.redis.delete(f"{self.conversation_prefix}{conv_id}")
            
            # Delete the session index
            self.redis.delete(f"{self.session_index_prefix}{session_id}")
            
            return len(conversation_ids)
        except Exception as e:
            print(f"Redis delete_session_conversations error: {e}")
            return 0

# Initialize Redis conversation manager
conversation_manager = RedisConversationManager(redis_client)

#################################
# VoyageEmbeddings Class
#################################
class VoyageEmbeddings:
    def __init__(self, api_key):
        self.api_key = api_key
        self.embed_url = "https://api.voyageai.com/v1/embeddings"
        self.session = None
    
    async def init_session(self):
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_connect=10, sock_read=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_embeddings(self, texts):
        if not self.session:
            await self.init_session()
        if isinstance(texts, str):
            texts = [texts]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "voyage-3-large",
            "input": texts
        }
        try:
            async with self.session.post(self.embed_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error from Voyage AI: {error_text}")
                    return None
                result = await response.json()
                return [item["embedding"] for item in result["data"]]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return None

#################################
# VectorDBManager Class
#################################
class VectorDBManager:
    def __init__(self, embeddings_client):
        self.embeddings_client = embeddings_client
        self.collection_cache = {}
    

    
    async def get_or_create_collection(self, collection_name):
        if collection_name in self.collection_cache:
            return self.collection_cache[collection_name]
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except:
            collection = chroma_client.create_collection(name=collection_name)
        self.collection_cache[collection_name] = collection
        return collection
    
    async def _store_eta_data(self, collection_name, eta_data, trip_id=None):
        """Store ETA data in vector database"""
        if not eta_data:
            return
            
        eta_text = "BUS TRACKING INFORMATION:\n"
        
        # Add trip ID if available
        if trip_id:
            eta_text += f"• Trip ID: {trip_id}\n"
        
        # Handle message-only responses
        if "message" in eta_data and eta_data["message"] == "This trip has ended":
            eta_text += "• Status: The trip has already ended. The bus has completed its journey.\n"
        elif "message" in eta_data:
            eta_text += f"• Status: {eta_data['message']}\n"
        else:
            # Format proper ETA data with bus location
            if "currentLocation" in eta_data:
                eta_text += f"• Current Bus Location: {eta_data['currentLocation']}\n"
            
            if "estimatedArrival" in eta_data:
                eta_text += f"• Estimated Arrival Time: {eta_data['estimatedArrival']}\n"
            
            if "delayMinutes" in eta_data:
                eta_text += f"• Delay: {eta_data['delayMinutes']} minutes\n"
                
            if "lastUpdated" in eta_data:
                eta_text += f"• Last Updated: {eta_data['lastUpdated']}\n"
        
        # Store the ETA information in vector database
        await self.add_texts(
            collection_name,
            [eta_text],
            [{"type": "eta_data", "trip_id": str(trip_id) if trip_id else "unknown"}],
            [f"eta_data_{trip_id or 'unknown'}_{collection_name}"]
        )
    
    async def add_texts(self, collection_name, texts, metadatas=None, ids=None):
        collection = await self.get_or_create_collection(collection_name)
        if not texts:
            return
        embeddings = await self.embeddings_client.get_embeddings(texts)
        if not embeddings:
            print(f"Failed to get embeddings for {collection_name}")
            return
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Ensure all metadata values are simple types (str, int, float, bool)
        sanitized_metadatas = []
        for metadata in metadatas:
            sanitized_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (list, set, tuple)):
                    sanitized_metadata[key] = ", ".join(str(item) for item in value)
                elif isinstance(value, dict):
                    sanitized_metadata[key] = json.dumps(value)
                else:
                    sanitized_metadata[key] = str(value) if not isinstance(value, (int, float, bool, str)) else value
            sanitized_metadatas.append(sanitized_metadata)
        
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=sanitized_metadatas,
            ids=ids
        )
    
    async def query_collection(self, collection_name, query_text, n_results=5):
        collection = await self.get_or_create_collection(collection_name)
        query_embedding = await self.embeddings_client.get_embeddings(query_text)
        if not query_embedding:
            print(f"Failed to get embedding for query: {query_text}")
            return []
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return results
    
    async def store_system_prompt(self, prompt_text, collection_name="system_prompts"):
        sections = re.split(r'\n\s*\n', prompt_text)
        texts = []
        metadatas = []
        ids = []
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            section_id = f"system_section_{i}"
            section_title = section.split('\n')[0] if '\n' in section else "Section"
            texts.append(section)
            metadatas.append({
                "type": "system_prompt",
                "section": section_title,
                "section_id": i
            })
            ids.append(section_id)
        await self.add_texts(collection_name, texts, metadatas, ids)
    
    async def get_system_prompt(self, query, collection_name="system_prompts"):
        results = await self.query_collection(collection_name, query, n_results=5)
        if not results or not results["documents"]:
            return "You are a Fresh Bus travel assistant. Provide accurate bus information based on API data only."
        prompt_sections = []
        documents = results["documents"]
        if documents and isinstance(documents[0], list):
            documents = [doc for sublist in documents for doc in sublist]
        elif documents and not isinstance(documents[0], str):
            documents = [str(doc) for doc in documents]
        for doc in documents:
            if isinstance(doc, str):
                prompt_sections.append(doc)
            else:
                prompt_sections.append(str(doc))
        return "\n\n".join(prompt_sections)
    
    async def _store_active_tickets(self, collection_name, tickets):
        """Store active tickets information in vector database"""
        if not tickets:
            return
            
        tickets_text = "ACTIVE TICKETS INFORMATION:\n"
        
        # Add summary of active tickets
        tickets_text += f"• You have {len(tickets)} active tickets for upcoming or ongoing journeys.\n\n"
        
        # Add details for each ticket
        for i, ticket in enumerate(tickets):
            tickets_text += f"Ticket {i+1}:\n"
            tickets_text += f"• Journey: {ticket.get('source', 'Unknown')} to {ticket.get('destination', 'Unknown')}\n"
            
            if "journey_date" in ticket:
                tickets_text += f"• Date: {ticket['journey_date']}\n"
                
            if "trip_id" in ticket:
                tickets_text += f"• Trip ID: {ticket['trip_id']}\n"
            
            # Add ETA information if available
            if "eta_data" in ticket:
                eta_data = ticket["eta_data"]
                
                if "message" in eta_data and eta_data["message"] == "This trip has ended":
                    tickets_text += "• Status: The trip has already ended\n"
                elif "message" in eta_data:
                    tickets_text += f"• Status: {eta_data['message']}\n"
                else:
                    if "currentLocation" in eta_data:
                        tickets_text += f"• Current Location: {eta_data['currentLocation']}\n"
                    if "estimatedArrival" in eta_data:
                        tickets_text += f"• Estimated Arrival: {eta_data['estimatedArrival']}\n"
                    if "delayMinutes" in eta_data:
                        tickets_text += f"• Delay: {eta_data['delayMinutes']} minutes\n"
            
            tickets_text += "\n"
        
        # Store the tickets information in vector database
        await self.add_texts(
            collection_name,
            [tickets_text],
            [{"type": "active_tickets", "count": len(tickets)}],
            [f"active_tickets_{collection_name}"]
        )
    
    async def store_api_data(self, session_id, api_data):
        collection_name = f"api_data_{session_id}"
        if api_data.get("trips"):
            await self._store_trips(collection_name, api_data["trips"], api_data.get("user_direction"))
        if api_data.get("boarding_points"):
            await self._store_boarding_points(collection_name, api_data["boarding_points"], 
                                            api_data.get("nearest_boarding_points"))
        if api_data.get("dropping_points"):
            await self._store_dropping_points(collection_name, api_data["dropping_points"])
        if api_data.get("recommendations"):
            await self._store_seat_recommendations(collection_name, api_data["recommendations"])
        if api_data.get("user_profile"):
            await self._store_user_profile(collection_name, api_data["user_profile"])
    
    async def _store_trips(self, collection_name, trips_data, user_direction=None):
        texts = []
        metadatas = []
        ids = []
        
        # Emphasize trip count
        trip_count = len(trips_data)
        all_trips_summary = f"IMPORTANT: There are EXACTLY {trip_count} unique bus services available. "
        if trip_count == 1:
            all_trips_summary += "There is ONLY ONE bus service available. Do not display or invent multiple options.\n\n"
        all_trips_summary += "Available bus options:\n"
        
        # Extract source and destination from API data
        api_sources = set(trip.get('source', 'Unknown') for trip in trips_data)
        api_destinations = set(trip.get('destination', 'Unknown') for trip in trips_data)
        
        # Log tripids to ensure uniqueness
        tripids = [trip.get('tripid') for trip in trips_data]
        print(f"Storing {trip_count} unique trips with IDs: {tripids}")
        
        # If user_direction is provided, use that instead of API direction
        if user_direction and 'source' in user_direction and 'destination' in user_direction:
            display_source = user_direction['source']
            display_destination = user_direction['destination']
            # Add note about the user's requested direction
            direct_route_text = (
                f"USER REQUESTED DIRECTION: Buses requested FROM {display_source} TO {display_destination}. "
                f"Using available bus data while respecting user's requested direction.\n"
                f"TRIP COUNT: There are EXACTLY {trip_count} unique bus services available."
            )
        else:
            # Use API direction
            display_source = next(iter(api_sources)) if api_sources else "Unknown"
            display_destination = next(iter(api_destinations)) if api_destinations else "Unknown"
            direct_route_text = (
                f"ROUTE INFORMATION: Buses available FROM {display_source} TO {display_destination}.\n"
                f"TRIP COUNT: There are EXACTLY {trip_count} unique bus services available."
            )
        
        texts.append(direct_route_text)
        metadatas.append({
            "type": "route_info", 
            "sources": ", ".join(str(s) for s in api_sources), 
            "destinations": ", ".join(str(d) for d in api_destinations),
            "display_source": display_source,
            "display_destination": display_destination,
            "trip_count": trip_count
        })
        ids.append(f"route_info_{collection_name}")
        
        for i, trip in enumerate(trips_data):
            # Handle time conversion properly
            departure_time = "N/A"
            if "boardingtime" in trip:
                try:
                    dt = datetime.fromisoformat(trip["boardingtime"].replace('Z', '+00:00'))
                    ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
                    departure_time = ist_dt.strftime("%I:%M %p IST")
                except:
                    departure_time = trip.get("boardingtime", "").split("T")[1].split(".")[0] + " UTC"
                    
            arrival_time = "N/A"
            if "droppingtime" in trip:
                try:
                    dt = datetime.fromisoformat(trip["droppingtime"].replace('Z', '+00:00'))
                    ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
                    arrival_time = ist_dt.strftime("%I:%M %p IST")
                except:
                    arrival_time = trip.get("droppingtime", "").split("T")[1].split(".")[0] + " UTC"
            
            # Calculate duration if available
            duration = ""
            if "boardingtime" in trip and "droppingtime" in trip:
                try:
                    boarding_dt = datetime.fromisoformat(trip["boardingtime"].replace('Z', '+00:00'))
                    dropping_dt = datetime.fromisoformat(trip["droppingtime"].replace('Z', '+00:00'))
                    duration_mins = (dropping_dt - boarding_dt).total_seconds() / 60
                    hours = int(duration_mins // 60)
                    mins = int(duration_mins % 60)
                    duration = f"{hours}h {mins}m"
                except:
                    duration = ""
            
            # Use source and destination based on user preference if available
            if user_direction and 'source' in user_direction and 'destination' in user_direction:
                source = user_direction['source']
                destination = user_direction['destination']
            else:
                source = trip.get('source', 'Unknown')
                destination = trip.get('destination', 'Unknown')
            
            # Include tripid in the summary for clarity
            all_trips_summary += (
                f"Option {i+1}: Bus {trip.get('servicenumber', 'Unknown')} (Trip ID: {trip.get('tripid', 'Unknown')}), " +
                f"Route: FROM {source} TO {destination}, " +
                f"departure: {departure_time}, arrival: {arrival_time}, " +
                (f"duration: {duration}, " if duration else "") +
                f"price: ₹{trip.get('fare', 'Unknown')}, " +
                f"bus type: {trip.get('vehicletype', 'Standard')}, " +
                f"available seats: {trip.get('availableseats', 'N/A')}\n"
            )
            
            # Add boarding and dropping points
            boarding_point = trip.get('boardingpointname', 'Unknown boarding point')
            dropping_point = trip.get('droppingpointname', 'Unknown dropping point')
            
            # Swap boarding and dropping points if user direction is opposite of API direction
            if (user_direction and 
                user_direction.get('swap_points', False) and 
                boarding_point and dropping_point):
                boarding_point, dropping_point = dropping_point, boarding_point
                
            all_trips_summary += f"  → Boards at: {boarding_point}\n"
            all_trips_summary += f"  → Drops at: {dropping_point}\n"
        
        texts.append(all_trips_summary)
        metadatas.append({"type": "trips_summary", "count": len(trips_data)})
        ids.append(f"trips_summary_{collection_name}")
        
        await self.add_texts(collection_name, texts, metadatas, ids)
    
    async def _store_boarding_points(self, collection_name, boarding_points, nearest_points=None):
        texts = []
        metadatas = []
        ids = []
        boarding_text = "Available boarding points:\n"
        for i, point in enumerate(boarding_points):
            bp_info = point.get('boardingPoint', {})
            name = bp_info.get('name', 'Unknown')
            landmark = bp_info.get('landmark', '')
            time = bp_info.get('time', '')
            boarding_text += f"• {name}" + (f" (near {landmark})" if landmark else "") + (f", time: {time}" if time else "") + "\n"
        texts.append(boarding_text)
        metadatas.append({"type": "boarding_points", "count": len(boarding_points)})
        ids.append(f"boarding_all_{collection_name}")
        if nearest_points:
            nearest_text = "Boarding points closest to your location:\n"
            for point in nearest_points:
                nearest_text += f"• {point['name']} ({point['distance_km']} km away)"
                if point.get('landmark'):
                    nearest_text += f" - Near {point['landmark']}"
                nearest_text += "\n"
            texts.append(nearest_text)
            metadatas.append({"type": "nearest_boarding", "count": len(nearest_points)})
            ids.append(f"boarding_nearest_{collection_name}")
        await self.add_texts(collection_name, texts, metadatas, ids)
    
    async def _store_dropping_points(self, collection_name, dropping_points):
        dropping_text = "Available dropping points:\n"
        for i, point in enumerate(dropping_points):
            dp_info = point.get('droppingPoint', {})
            name = dp_info.get('name', 'Unknown')
            landmark = dp_info.get('landmark', '')
            time = dp_info.get('time', '')
            dropping_text += f"• {name}" + (f" (near {landmark})" if landmark else "") + (f", time: {time}" if time else "") + "\n"
        await self.add_texts(
            collection_name, 
            [dropping_text], 
            [{"type": "dropping_points", "count": len(dropping_points)}],
            [f"dropping_all_{collection_name}"]
        )
    
    async def _store_seat_recommendations(self, collection_name, recommendations):
        texts = []
        metadatas = []
        ids = []
        summary_text = "Available seat categories:\n"
        for category in recommendations:
            window_count = len(recommendations[category]['window'])
            aisle_count = len(recommendations[category]['aisle'])
            if window_count or aisle_count:
                summary_text += f"• {category}: {window_count} window, {aisle_count} aisle seats available\n"
        texts.append(summary_text)
        metadatas.append({"type": "seat_summary"})
        ids.append(f"seats_summary_{collection_name}")
        for category, positions in recommendations.items():
            for position, seats in positions.items():
                if seats:
                    seat_text = f"Available {position} seats in {category} category:\n"
                    for seat in seats[:5]:
                        seat_text += f"• Seat #{seat['number']} (₹{seat['price']})\n"
                    texts.append(seat_text)
                    metadatas.append({
                        "type": "seat_detail",
                        "category": category,
                        "position": position,
                        "count": len(seats)
                    })
                    ids.append(f"seats_{category}_{position}_{collection_name}")
        await self.add_texts(collection_name, texts, metadatas, ids)
    
    async def _store_user_profile(self, collection_name, profile_data):
        """Store user profile data in vector database"""
        if not profile_data:
            return
            
        profile_text = "USER PROFILE INFORMATION:\n"
        
        # Basic user info
        if profile_data.get("name"):
            profile_text += f"• Name: {profile_data['name']}\n"
        if profile_data.get("email"):
            profile_text += f"• Email: {profile_data['email']}\n"
        if profile_data.get("mobile"):
            profile_text += f"• Mobile: {profile_data['mobile']}\n"
        if profile_data.get("gender"):
            profile_text += f"• Gender: {profile_data['gender']}\n"
            
        # Preferred language
        if profile_data.get("preferredLanguage"):
            profile_text += f"• Preferred Language: {profile_data['preferredLanguage']}\n"
            
        # Add booking history if available
        if profile_data.get("bookingHistory") and len(profile_data["bookingHistory"]) > 0:
            profile_text += "\nBOOKING HISTORY:\n"
            
            for i, booking in enumerate(profile_data["bookingHistory"][:5]):  # Limit to last 5 bookings
                booking_info = f"• Booking {i+1}: "
                
                if booking.get("source") and booking.get("destination"):
                    booking_info += f"{booking['source']} to {booking['destination']}"
                
                if booking.get("travelDate"):
                    try:
                        # Parse and format date
                        date_str = booking["travelDate"]
                        if isinstance(date_str, str):
                            if "T" in date_str:
                                date_str = date_str.split("T")[0]
                            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                            formatted_date = date_obj.strftime("%d %b %Y")
                            booking_info += f" on {formatted_date}"
                    except Exception as e:
                        print(f"Error parsing date: {e}")
                
                if booking.get("seatNumber"):
                    booking_info += f", Seat: {booking['seatNumber']}"
                    
                if booking.get("fare"):
                    booking_info += f", Fare: ₹{booking['fare']}"
                    
                profile_text += booking_info + "\n"
                
        # Add seat preferences based on history
        if profile_data.get("seatPreferences"):
            seat_prefs = profile_data["seatPreferences"]
            profile_text += "\nSEAT PREFERENCES:\n"
            
            if seat_prefs.get("favoriteSeats"):
                profile_text += f"• Favorite seats: {', '.join(map(str, seat_prefs['favoriteSeats']))}\n"
                
            if seat_prefs.get("position"):
                profile_text += f"• Preferred position: {seat_prefs['position']}\n"
                
            if seat_prefs.get("category"):
                profile_text += f"• Preferred category: {seat_prefs['category']}\n"
                
        # Add frequently traveled routes
        if profile_data.get("frequentRoutes") and len(profile_data["frequentRoutes"]) > 0:
            profile_text += "\nFREQUENT ROUTES:\n"
            for i, route in enumerate(profile_data["frequentRoutes"]):
                profile_text += f"• {route}\n"
                
        # Store the profile information in vector database
        await self.add_texts(
            collection_name,
            [profile_text],
            [{"type": "user_profile"}],
            [f"user_profile_{collection_name}"]
        )
    
    def get_token_count(self, text):
        return len(tokenizer.encode(text))

    


#################################
# FreshBusAssistant Class
#################################
class FreshBusAssistant:
    def __init__(self):
        self.system_prompt = ""
        self.load_system_prompt("./system_prompt/qa_prompt.txt")
        self.BASE_URL = "https://api-stage.freshbus.com"
        self.stations = {
            "hyderabad": 3,
            "vijayawada": 5,
            "bangalore": 7,
            "tirupati": 12,
            "chittoor": 130,
            "guntur": 13
        }
        self.seat_categories = {
            "Premium": [5, 8, 9, 12, 13, 16, 17, 20, 24, 25, 28],
            "Reasonable": [1, 2, 3, 4, 6, 7, 10, 11, 14, 15, 18, 19, 21, 22, 23, 26, 27, 29, 30, 31, 32, 37, 40],
            "Low Reasonable": [33, 34, 35, 36, 38, 39],
            "Budget-Friendly": [41, 42, 43, 44]
        }
        self.window_seats = [1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44]
        self.aisle_seats = [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43]
        self.sessions = {}
        self.http_session = None
        self.embeddings_client = VoyageEmbeddings(voyage_api_key)
        self.vector_db = VectorDBManager(self.embeddings_client)
        self.system_prompt_initialized = False

    def load_system_prompt(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
                print(f"Loaded system prompt from {path}")
        except Exception as e:
            print(f"Error loading system prompt: {e}")
            self.system_prompt = "You are a Fresh Bus travel assistant. Provide accurate bus information based on API data only."
    
    async def init_system_prompt(self):
        if not self.system_prompt_initialized:
            await self.embeddings_client.init_session()
            await self.vector_db.store_system_prompt(self.system_prompt)
            self.system_prompt_initialized = True
            print("System prompt stored in vector DB")
    
    async def init_http_session(self):
        if self.http_session is None:
            timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_connect=10, sock_read=10)
            self.http_session = aiohttp.ClientSession(timeout=timeout)
            print("HTTP session initialized with timeout settings")
            await self.embeddings_client.init_session()
    
    async def cleanup(self):
        if self.http_session:
            await self.http_session.close()
            print("HTTP session closed")
        await self.embeddings_client.close_session()
    
    async def fetch_user_profile(self, auth_token):
        """Fetch user profile data from the Fresh Bus API"""
        if not self.http_session:
            await self.init_http_session()
        
        try:
            # Call the profile API endpoint
            headers = {
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json"
            }
            
            url = f"{self.BASE_URL}/profile"
            print(f"Fetching user profile: {url}")
            
            async with self.http_session.get(url, headers=headers) as response:
                if response.status == 200:
                    profile_data = await response.json()
                    print(f"Successfully fetched user profile")
                    return profile_data
                else:
                    print(f"Error fetching user profile: {response.status}")
                    error_text = await response.text()
                    print(f"Error response: {error_text}")
                    return None
        except Exception as e:
            print(f"Exception fetching user profile: {e}")
            return None
    
    async def get_active_tickets_and_status(self, auth_token):
        """Get user's active tickets and their status (future, ongoing, or completed)"""
        if not self.http_session:
            await self.init_http_session()
        
        try:
            # First, get all tickets
            headers = {
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json"
            }
            
            tickets_url = f"{self.BASE_URL}/tickets"
            print(f"Fetching tickets: {tickets_url}")
            
            async with self.http_session.get(tickets_url, headers=headers) as response:
                if response.status != 200:
                    print(f"Error fetching tickets: {response.status}")
                    return None
                
                tickets_data = await response.json()
            
            # Categorize tickets as future, ongoing, or completed
            now = datetime.now()
            future_tickets = []
            ongoing_tickets = []
            completed_tickets = []
            
            for ticket in tickets_data:
                # Parse journey date and time
                try:
                    if "journeyDate" in ticket:
                        journey_date_str = ticket["journeyDate"]
                        
                        # Handle different date formats
                        if "T" in journey_date_str:
                            journey_date = datetime.fromisoformat(journey_date_str.replace('Z', '+00:00'))
                        else:
                            # If only date is provided, add a default time
                            journey_date = datetime.fromisoformat(f"{journey_date_str}T00:00:00+00:00")
                        
                        # Compute time differences
                        time_diff_minutes = (journey_date - now).total_seconds() / 60
                        
                        # Journey is in the future (more than 30 minutes away)
                        if time_diff_minutes > 30:
                            future_tickets.append(ticket)
                        # Journey is ongoing or about to start (within 30 minutes before to 24 hours after)
                        elif time_diff_minutes > -24*60 and time_diff_minutes <= 30:
                            ongoing_tickets.append(ticket)
                        # Journey is completed (more than 24 hours ago)
                        else:
                            completed_tickets.append(ticket)
                except Exception as e:
                    print(f"Error parsing journey date for ticket: {e}")
                    # Include the ticket in ongoing if we can't parse the date
                    ongoing_tickets.append(ticket)
            
            # Get ETA data for ongoing tickets
            for ticket in ongoing_tickets:
                if "tripId" in ticket:
                    trip_id = ticket["tripId"]
                    eta_url = f"https://api.freshbus.com/eta-data?id={trip_id}"
                    
                    try:
                        async with self.http_session.get(eta_url) as eta_response:
                            if eta_response.status == 200:
                                eta_data = await eta_response.json()
                                ticket["eta_data"] = eta_data
                                
                                # If trip has ended, move to completed tickets
                                if eta_data.get("message") == "This trip has ended":
                                    completed_tickets.append(ticket)
                                    ongoing_tickets.remove(ticket)
                            else:
                                print(f"Error fetching ETA data for trip {trip_id}: {eta_response.status}")
                                ticket["eta_data"] = {"message": "Could not fetch ETA data"}
                    except Exception as eta_err:
                        print(f"Exception fetching ETA data: {eta_err}")
                        ticket["eta_data"] = {"message": f"Error: {str(eta_err)}"}
            
            # Get feedback questions for completed tickets
            if completed_tickets:
                try:
                    feedback_url = f"{self.BASE_URL}/tickets/feedbackQuestions"
                    async with self.http_session.get(feedback_url, headers=headers) as fb_response:
                        if fb_response.status == 200:
                            feedback_data = await fb_response.json()
                            
                            # Associate feedback data with completed tickets
                            for ticket in completed_tickets:
                                trip_id = ticket.get("tripId")
                                ticket_feedback = next((fb for fb in feedback_data if fb.get("tripId") == trip_id), None)
                                if ticket_feedback:
                                    ticket["feedback_data"] = ticket_feedback
                                else:
                                    ticket["feedback_data"] = {"message": "No feedback questions available"}
                        else:
                            print(f"Error fetching feedback questions: {fb_response.status}")
                            for ticket in completed_tickets:
                                ticket["feedback_data"] = {"message": "Could not fetch feedback questions"}
                except Exception as fb_err:
                    print(f"Exception fetching feedback questions: {fb_err}")
                    for ticket in completed_tickets:
                        ticket["feedback_data"] = {"message": f"Error: {str(fb_err)}"}
            
            return {
                "future": future_tickets,
                "ongoing": ongoing_tickets,
                "completed": completed_tickets
            }
        except Exception as e:
            print(f"Exception in get_active_tickets_and_status: {e}")
            return None
    
    def trim_text(self, text, token_limit=2000):
        tokens = tokenizer.encode(text)
        if len(tokens) > token_limit:
            return tokenizer.decode(tokens[:token_limit])
        return text
    
    def deduplicate_trips(self, trips_data):
        """Deduplicate trips by their tripid"""
        if not trips_data:
            return []
            
        unique_trips = {}
        for trip in trips_data:
            tripid = trip.get('tripid')
            if tripid and tripid not in unique_trips:
                unique_trips[tripid] = trip
        
        print(f"Deduplicated trips: {len(trips_data)} raw trips -> {len(unique_trips)} unique trips")
        return list(unique_trips.values())

    def enforce_response_format(self, response_text, api_data, context):
        """Enforce the required response format for bus listings."""
        # Check if response contains bus info but not in proper format
        if ("bus" in response_text.lower() and "price" in response_text.lower() and 
            ("seat" in response_text.lower() or "seats" in response_text.lower()) and 
            not re.search(r'>\s*\*🚌', response_text)):
            
            # The response has bus info but incorrect format - create formatted version
            formatted_response = self.generate_formatted_response(api_data, context)
            
            # Add a note explaining the correction
            final_response = "Here are the available buses for your trip:\n\n" + formatted_response
            
            return final_response
        
        # Remove "CORRECTION" text if present - we want to honor user's requested direction
        if "CORRECTION:" in response_text and context.get('user_requested_source') and context.get('user_requested_destination'):
            corrected_text = response_text.replace("CORRECTION: The buses are from", "Here are buses for your trip from")
            corrected_text = corrected_text.replace("not the other way around.", "")
            return corrected_text
        
        # Check for incorrect trip count
        if api_data and api_data.get("trips"):
            actual_trip_count = len(api_data["trips"])
            
            # If there's only one trip but response shows multiple
            if actual_trip_count == 1 and ("Option 2" in response_text or "option 2" in response_text.lower()):
                # Create a corrected response that shows exactly one trip
                corrected_response = self.generate_formatted_response(api_data, context)
                return "NOTICE: There is only ONE bus service available for this route:\n\n" + corrected_response
        
        return response_text
    
    def generate_direct_bus_response(self, api_data, context):
        """
        Generate bus response DIRECTLY from API data without allowing hallucination
        """
        trips = api_data.get("trips", [])
        
        if not trips:
            return "I couldn't find any buses matching your criteria."
        
        # Get source and destination based on context
        source = context.get('user_requested_source') or trips[0].get('source', 'Unknown')
        destination = context.get('user_requested_destination') or trips[0].get('destination', 'Unknown')
        
        # Create header
        trip_count = len(trips)
        header = f"I found {trip_count} bus{'es' if trip_count > 1 else ''} from {source} to {destination} for tomorrow:\n\n"
        
        formatted_buses = []
        for trip in trips:
            # Get basic trip data directly from JSON
            bus_data = {
                'tripid': trip.get('tripid', 'Unknown'),
                'fare': trip.get('fare', 'Unknown'),
                'seats': trip.get('availableseats', 'Unknown'),
                'rating': trip.get('redbusrating', 'Unknown'),
                'boarding': trip.get('boardingpointname', 'Unknown boarding point'),
                'dropping': trip.get('droppingpointname', 'Unknown dropping point'),
                'vehicletype': trip.get('vehicletype', 'AC Seater')
            }
            
            # Format times
            if "boardingtime" in trip:
                try:
                    dt = datetime.fromisoformat(trip["boardingtime"].replace('Z', '+00:00'))
                    ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
                    bus_data['departure_time'] = ist_dt.strftime("%I:%M %p").lstrip('0')
                except:
                    bus_data['departure_time'] = "Unknown"
                    
            if "droppingtime" in trip:
                try:
                    dt = datetime.fromisoformat(trip["droppingtime"].replace('Z', '+00:00'))
                    ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
                    bus_data['arrival_time'] = ist_dt.strftime("%I:%M %p").lstrip('0')
                except:
                    bus_data['arrival_time'] = "Unknown"
            
            # Get seat recommendations
            window_seat = "33"  # Default reasonable values from API
            window_price = "598"
            aisle_seat = "34"
            aisle_price = "598"
            
            if api_data.get("recommendations") and api_data["recommendations"].get("Reasonable"):
                if api_data["recommendations"]["Reasonable"].get("window") and len(api_data["recommendations"]["Reasonable"]["window"]) > 0:
                    window_seat = api_data["recommendations"]["Reasonable"]["window"][0].get("number", "33")
                    window_price = api_data["recommendations"]["Reasonable"]["window"][0].get("price", "598")
                
                if api_data["recommendations"]["Reasonable"].get("aisle") and len(api_data["recommendations"]["Reasonable"]["aisle"]) > 0:
                    aisle_seat = api_data["recommendations"]["Reasonable"]["aisle"][0].get("number", "34")
                    aisle_price = api_data["recommendations"]["Reasonable"]["aisle"][0].get("price", "598")
            
            # Build structured response using verified data only
            formatted_bus = (
                "Verified\n"
                f"₹{bus_data['fare']}\n"
                f"{bus_data['seats']} seats\n"
                f"{bus_data['rating']}/5\n"
                "Boarding\n"
                f"{bus_data['boarding']}\n"
                "Dropping\n"
                f"{bus_data['dropping']}\n"
                "Reasonable Seats\n"
                f"Window {window_seat}\n"
                f"(₹{window_price})\n"
                f"Aisle {aisle_seat}\n"
                f"(₹{aisle_price})\n"
                "Book This Bus"
            )
            
            formatted_buses.append(formatted_bus)
        
        return header + "\n\n".join(formatted_buses)

    def generate_formatted_response(self, api_data, context):
        """Generate properly formatted bus response from API data."""
        if not api_data or not api_data.get("trips"):
            return "I couldn't find any buses matching your criteria."
        
        trips = api_data.get("trips", [])
        trip_count = len(trips)
        print(f"Formatting response for {trip_count} unique trips with IDs: {[t.get('tripid') for t in trips]}")
        
        formatted_buses = []
        
        # Check if we should use user's requested direction instead of API data
        use_user_direction = (context.get('user_requested_source') and 
                             context.get('user_requested_destination'))
        
        user_source = context.get('user_requested_source', '')
        user_dest = context.get('user_requested_destination', '')
        
        for i, trip in enumerate(trips):
            # Format departure and arrival times to IST (+5:30)
            departure_time = "Unknown"
            arrival_time = "Unknown"
            
            if "boardingtime" in trip:
                try:
                    dt = datetime.fromisoformat(trip["boardingtime"].replace('Z', '+00:00'))
                    ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
                    departure_time = ist_dt.strftime("%I:%M %p").lstrip('0')
                except:
                    departure_time = "Unknown"
                
            if "droppingtime" in trip:
                try:
                    dt = datetime.fromisoformat(trip["droppingtime"].replace('Z', '+00:00'))
                    ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
                    arrival_time = ist_dt.strftime("%I:%M %p").lstrip('0')
                except:
                    arrival_time = "Unknown"
            
            # Calculate duration
            duration = "N/A"
            if "boardingtime" in trip and "droppingtime" in trip:
                try:
                    b_dt = datetime.fromisoformat(trip["boardingtime"].replace('Z', '+00:00'))
                    d_dt = datetime.fromisoformat(trip["droppingtime"].replace('Z', '+00:00'))
                    duration_min = (d_dt - b_dt).total_seconds() / 60
                    hours = int(duration_min // 60)
                    minutes = int(duration_min % 60)
                    duration = f"{hours}h {minutes}m"
                except:
                    duration = "N/A"
            
            # Get bus type
            bus_type = trip.get('vehicletype', 'AC Seater(2+2)')
            
            # Get price, seats and rating
            price = trip.get('fare', 'N/A')
            available_seats = trip.get('availableseats', 'N/A')
            rating = trip.get('redbusrating', 'N/A')
            
            # Get source and destination based on user's preference or API data
            source = user_source if use_user_direction else trip.get('source', 'Unknown')
            destination = user_dest if use_user_direction else trip.get('destination', 'Unknown')
            
            # Get boarding point with distance if location available
            boarding_point = trip.get('boardingpointname', 'Unknown boarding point')
            boarding_distance = ""
            
            # Include nearest boarding point info based on user location
            if api_data.get("nearest_boarding_points") and len(api_data["nearest_boarding_points"]) > 0:
                nearest = api_data["nearest_boarding_points"][0]
                boarding_point = nearest.get("name", boarding_point)
                boarding_distance = f" ({nearest.get('distance_km', 0)} km from your location)"
            
            # Get dropping point
            dropping_point = trip.get('droppingpointname', 'Unknown dropping point')
            
            # If user direction is opposite of API direction, swap boarding and dropping points
            if use_user_direction:
                api_source = trip.get('source', '')
                api_dest = trip.get('destination', '')
                if (user_source.lower() != api_source.lower() or 
                    user_dest.lower() != api_dest.lower()):
                    # Swap boarding and dropping points
                    boarding_point, dropping_point = dropping_point, boarding_point
            
            # Get seat recommendations
            reasonable_window = "N/A"
            reasonable_window_price = "N/A"
            reasonable_aisle = "N/A"
            reasonable_aisle_price = "N/A"
            
            premium_window = "N/A"
            premium_window_price = "N/A"
            premium_aisle = "N/A"
            premium_aisle_price = "N/A"
            
            budget_window = "N/A"
            budget_window_price = "N/A"
            budget_aisle = "N/A"
            budget_aisle_price = "N/A"
            
            # Set seat recommendations if available
            if api_data.get("recommendations"):
                rec = api_data["recommendations"]
                
                if rec.get("Reasonable"):
                    if rec["Reasonable"].get("window") and len(rec["Reasonable"]["window"]) > 0:
                        reasonable_window = rec["Reasonable"]["window"][0].get("number", "N/A")
                        reasonable_window_price = rec["Reasonable"]["window"][0].get("price", "N/A")
                    
                    if rec["Reasonable"].get("aisle") and len(rec["Reasonable"]["aisle"]) > 0:
                        reasonable_aisle = rec["Reasonable"]["aisle"][0].get("number", "N/A")
                        reasonable_aisle_price = rec["Reasonable"]["aisle"][0].get("price", "N/A")
                
                if rec.get("Premium"):
                    if rec["Premium"].get("window") and len(rec["Premium"]["window"]) > 0:
                        premium_window = rec["Premium"]["window"][0].get("number", "N/A")
                        premium_window_price = rec["Premium"]["window"][0].get("price", "N/A")
                    
                    if rec["Premium"].get("aisle") and len(rec["Premium"]["aisle"]) > 0:
                        premium_aisle = rec["Premium"]["aisle"][0].get("number", "N/A")
                        premium_aisle_price = rec["Premium"]["aisle"][0].get("price", "N/A")
                
                if rec.get("Budget-Friendly"):
                    if rec["Budget-Friendly"].get("window") and len(rec["Budget-Friendly"]["window"]) > 0:
                        budget_window = rec["Budget-Friendly"]["window"][0].get("number", "N/A")
                        budget_window_price = rec["Budget-Friendly"]["window"][0].get("price", "N/A")
                    
                    if rec["Budget-Friendly"].get("aisle") and len(rec["Budget-Friendly"]["aisle"]) > 0:
                        budget_aisle = rec["Budget-Friendly"]["aisle"][0].get("number", "N/A")
                        budget_aisle_price = rec["Budget-Friendly"]["aisle"][0].get("price", "N/A")
            
            # Add tripid to the listing for debugging
            tripid_note = f"(Trip ID: {trip.get('tripid', 'N/A')})"
            
            # Create formatted bus listing with user's requested direction
            bus_listing = f"> *🚌 {source} to {destination} | {departure_time} - {arrival_time} | {bus_type} | {duration} {tripid_note}  \n"
            bus_listing += f"> Price: ₹{price} | {available_seats} seats | Rating: {rating}/5  \n"
            bus_listing += f"> Boarding: {boarding_point}{boarding_distance}  \n"
            bus_listing += f"> Dropping: {dropping_point}  \n"
            bus_listing += f"> Recommended seats:  \n"
            bus_listing += f"> • **Reasonable**: Window {reasonable_window} (₹{reasonable_window_price}), Aisle {reasonable_aisle} (₹{reasonable_aisle_price})  \n"
            bus_listing += f"> • **Premium**: Window {premium_window} (₹{premium_window_price}), Aisle {premium_aisle} (₹{premium_aisle_price})  \n"
            bus_listing += f"> • **Budget-Friendly**: Window {budget_window} (₹{budget_window_price}), Aisle {budget_aisle} (₹{budget_aisle_price})*"
            
            formatted_buses.append(bus_listing)
        
        # Add a final validation check
        if len(formatted_buses) != trip_count:
            print(f"WARNING: Formatted {len(formatted_buses)} buses but there are {trip_count} unique trips!")
        
        return "\n\n".join(formatted_buses)
    
    def get_or_create_session(self, session_id):
        if not session_id:
            session_id = str(uuid.uuid4())
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "messages": [],
                "last_updated": datetime.now(),
                "context": {
                    "last_source": None,
                    "last_destination": None,
                    "last_date": None,
                    "last_boarding_point": None,
                    "last_dropping_point": None,
                    "ticket_count": 1,
                    "selected_bus": None,
                    "selected_seats": [],
                    "user_location": None,
                    "language": "english",
                    "user_requested_source": None,
                    "user_requested_destination": None,
                    "auth": None,
                    "user": None,
                    "user_profile": None  # Add this new field for user profile
                }
            }
        return self.sessions[session_id], session_id
    
    def extract_locations(self, query):
        query_lower = query.lower()
        
        # Handle "to [destination] from [source]" pattern first - this is common in user requests
        to_from_pattern = r"to\s+([a-z]+)\s+from\s+([a-z]+)"
        match = re.search(to_from_pattern, query_lower)
        if match:
            dst, src = match.groups()  # Note: order is destination first, source second in this pattern
            for city in self.stations.keys():
                if src in city or city in src:
                    src = city
                if dst in city or city in dst:
                    dst = city
            if src in self.stations.keys() and dst in self.stations.keys():
                print(f"Extracted from 'to-from' pattern: FROM {src} TO {dst}")
                return src, dst
        
        # Handle other common patterns
        patterns = [
            r"from\s+([a-z]+)\s+to\s+([a-z]+)",
            r"([a-z]+)\s+to\s+([a-z]+)",
            r"book.*from\s+([a-z]+)\s+to\s+([a-z]+)",
            r"travel.*from\s+([a-z]+)\s+to\s+([a-z]+)"
        ]
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                src, dst = match.groups()
                for city in self.stations.keys():
                    if src in city or city in src:
                        src = city
                    if dst in city or city in dst:
                        dst = city
                if src in self.stations.keys() and dst in self.stations.keys():
                    print(f"Extracted from pattern: FROM {src} TO {dst}")
                    return src, dst
                    
        # Check for cities mentioned
        cities_mentioned = [city for city in self.stations.keys() if city in query_lower]
        if len(cities_mentioned) == 2:
            # If "from" is before the first city, use cities in order mentioned
            if "from" in query_lower and query_lower.find("from") < query_lower.find(cities_mentioned[0]):
                print(f"Extracted from cities mentioned with 'from': FROM {cities_mentioned[0]} TO {cities_mentioned[1]}")
                return cities_mentioned[0], cities_mentioned[1]
            # If "to" is before the first city, swap the order
            elif "to" in query_lower and query_lower.find("to") < query_lower.find(cities_mentioned[0]):
                print(f"Extracted from cities mentioned with 'to': FROM {cities_mentioned[1]} TO {cities_mentioned[0]}")
                return cities_mentioned[1], cities_mentioned[0]
            else:
                # Otherwise assume first city is source, second is destination
                print(f"Extracted from cities mentioned: FROM {cities_mentioned[0]} TO {cities_mentioned[1]}")
                return cities_mentioned[0], cities_mentioned[1]
        elif len(cities_mentioned) == 1:
            # If only one city is mentioned, try to infer direction
            if "to" in query_lower and query_lower.find("to") < query_lower.find(cities_mentioned[0]):
                return None, cities_mentioned[0]
            elif "from" in query_lower and query_lower.find("from") < query_lower.find(cities_mentioned[0]):
                return cities_mentioned[0], None
            else:
                return None, cities_mentioned[0]
                
        return None, None
    
    def extract_ticket_count(self, query):
        query_lower = query.lower()
        patterns = [
            r"(\d+)\s*tickets?",
            r"(\d+)\s*seats?",
            r"(\d+)\s*persons?",
            r"(\d+)\s*people",
            r"(\d+)\s*passengers?",
            r"book\s+(\d+)",
            r"for\s+(\d+)\s+of\s+us",
            r"for\s+(\d+)\s+people"
        ]
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                count = int(match.group(1))
                if 1 <= count <= 6:
                    return count
        if re.search(r"^[1-6]$", query_lower):
            return int(query_lower)
        if any(word in query_lower for word in ["we", "us", "our", "family", "friends", "couple"]):
            if "couple" in query_lower:
                return 2
            if "family" in query_lower:
                return 4
            return 2
        return None
    
    def parse_date(self, query):
        query_lower = query.lower()
        current_date = datetime.now()
        if "tomorrow" in query_lower:
            return (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "day after tomorrow" in query_lower:
            return (current_date + timedelta(days=2)).strftime("%Y-%m-%d")
        elif "today" in query_lower:
            return current_date.strftime("%Y-%m-%d")
        elif "next week" in query_lower:
            return (current_date + timedelta(days=7)).strftime("%Y-%m-%d")
        days = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
        for day, index in days.items():
            if day in query_lower:
                current_weekday = current_date.weekday()
                days_ahead = (index - current_weekday) % 7
                if days_ahead == 0:
                    days_ahead = 7
                return (current_date + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        months = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                  "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
                  "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, 
                  "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}
        for month_name, month_num in months.items():
            pattern1 = fr'(\d+)(?:st|nd|rd|th)?\s+{month_name}'
            pattern2 = fr'{month_name}\s+(\d+)(?:st|nd|rd|th)?'
            for pattern in [pattern1, pattern2]:
                match = re.search(pattern, query_lower)
                if match:
                    day = int(match.group(1))
                    year = current_date.year
                    try:
                        date_obj = datetime(year, month_num, day)
                        if date_obj < current_date:
                            date_obj = datetime(year+1, month_num, day)
                        return date_obj.strftime("%Y-%m-%d")
                    except ValueError:
                        continue
        return (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
    
    def process_user_profile(self, profile_data):
        """Process user profile data to extract useful information"""
        if not profile_data:
            return {}
            
        user_preferences = {
            "favorite_seats": [],
            "recent_routes": [],
            "preferred_bus_types": [],
            "seat_position": None,
            "preferred_category": None
        }
        
        # Extract booking history if available
        if "bookingHistory" in profile_data and isinstance(profile_data["bookingHistory"], list):
            booking_history = profile_data["bookingHistory"]
            seat_counts = {}
            routes = set()
            bus_types = set()
            
            for booking in booking_history:
                # Count seat numbers for favorite seats
                if "seatNumber" in booking and booking["seatNumber"]:
                    try:
                        seat_num = int(booking["seatNumber"])
                        seat_counts[seat_num] = seat_counts.get(seat_num, 0) + 1
                    except:
                        pass
                
                # Extract routes
                if "source" in booking and "destination" in booking:
                    route = f"{booking['source']} to {booking['destination']}"
                    routes.add(route)
                
                # Extract bus types
                if "busType" in booking and booking["busType"]:
                    bus_types.add(booking["busType"])
            
            # Sort seat numbers by frequency
            favorite_seats = sorted(seat_counts.keys(), key=lambda x: seat_counts[x], reverse=True)
            user_preferences["favorite_seats"] = favorite_seats[:3]  # Top 3 favorite seats
            
            # Add recent routes
            user_preferences["recent_routes"] = list(routes)[:5]  # Top 5 routes
            
            # Add preferred bus types
            user_preferences["preferred_bus_types"] = list(bus_types)
            
            # Determine preferred seat position (window/aisle)
            if favorite_seats:
                window_count = sum(1 for seat in favorite_seats[:3] if seat in self.window_seats)
                aisle_count = sum(1 for seat in favorite_seats[:3] if seat in self.aisle_seats)
                
                if window_count > aisle_count:
                    user_preferences["seat_position"] = "window"
                elif aisle_count > window_count:
                    user_preferences["seat_position"] = "aisle"
            
            # Determine preferred category
            if favorite_seats:
                categories_count = {
                    "Premium": 0,
                    "Reasonable": 0,
                    "Budget-Friendly": 0
                }
                
                for seat in favorite_seats[:3]:
                    if seat in self.seat_categories["Premium"]:
                        categories_count["Premium"] += 1
                    elif seat in self.seat_categories["Reasonable"] or seat in self.seat_categories["Low Reasonable"]:
                        categories_count["Reasonable"] += 1
                    elif seat in self.seat_categories["Budget-Friendly"]:
                        categories_count["Budget-Friendly"] += 1
                
                # Find the most frequent category
                preferred_category = max(categories_count.items(), key=lambda x: x[1])[0]
                if categories_count[preferred_category] > 0:
                    user_preferences["preferred_category"] = preferred_category
        
        # Add basic user info
        user_preferences["profile"] = {
            "name": profile_data.get("name", ""),
            "email": profile_data.get("email", ""),
            "mobile": profile_data.get("mobile", ""),
            "gender": profile_data.get("gender", ""),
            "preferredLanguage": profile_data.get("preferredLanguage", "")
        }
        
        return user_preferences
    
    def detect_bus_selection(self, query, trips_data):
        query_lower = query.lower()
        number_pattern = r'^[\s]*(\d+)[\s]*$'
        match = re.search(number_pattern, query_lower)
        if match:
            selection = int(match.group(1))
            if 1 <= selection <= len(trips_data):
                return selection - 1
        time_pattern = r'(\d{1,2})[:\.]?(\d{2})?\s*(am|pm|a\.m\.|p\.m\.)?'
        matches = re.finditer(time_pattern, query_lower)
        for match in matches:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            am_pm = match.group(3)
            if am_pm and 'p' in am_pm.lower() and hour < 12:
                hour += 12
            elif am_pm and 'a' in am_pm.lower() and hour == 12:
                hour = 0
            for i, trip in enumerate(trips_data):
                if "boardingtime" in trip:
                    boarding_time = datetime.fromisoformat(trip["boardingtime"].replace("Z", "+00:00"))
                    boarding_time = boarding_time.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30)))
                    if abs(boarding_time.hour - hour) <= 1:
                        if abs(boarding_time.hour - hour) == 0 and abs(boarding_time.minute - minute) <= 15:
                            return i
                        elif abs(boarding_time.hour - hour) == 1 and abs(60 + boarding_time.minute - minute) <= 15:
                            return i
        if any(term in query_lower for term in ["first", "1st", "early", "earliest"]):
            return 0
        elif any(term in query_lower for term in ["last", "latest", "final"]):
            return len(trips_data) - 1
        return None
    
    def detect_seat_selection(self, query):
        query_lower = query.lower()
        seat_patterns = [
            r'seat\s+(?:number\s+)?(\d+)',
            r'seat\s+#\s*(\d+)',
            r'seat\s+no\.?\s*(\d+)',
            r'seat\s+(\d+)',
            r'book\s+seat\s+(\d+)',
            r'select\s+seat\s+(\d+)',
            r'choose\s+seat\s+(\d+)',
            r'number\s+(\d+)\s+seat',
            r'the\s+(\d+)\s+seat'
        ]
        seats = []
        for pattern in seat_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                seat_num = int(match.group(1))
                if 1 <= seat_num <= 44:
                    seats.append(seat_num)
        window_preference = "window" in query_lower
        aisle_preference = "aisle" in query_lower
        return seats, window_preference, aisle_preference
    
    async def fetch_trips(self, source_id, destination_id, journey_date):
        if not self.http_session:
            await self.init_http_session()
        url = f"{self.BASE_URL}/trips?journey_date={journey_date}&source_id={source_id}&destination_id={destination_id}"
        print(f"Fetching trips: {url}")
        try:
            async with self.http_session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Print raw response for debugging
                    print(f"Raw API response: {json.dumps(data)}")
                    
                    # Return deduplicated trips
                    return self.deduplicate_trips(data)
                else:
                    print(f"Error fetching trips: {response.status}")
                    return []
        except Exception as e:
            print(f"Exception fetching trips: {e}")
            return []
    
    async def fetch_boarding_points(self, trip_id, source_id):
        if not self.http_session:
            await self.init_http_session()
        url = f"{self.BASE_URL}/trips/{trip_id}/boardings/{source_id}"
        print(f"Fetching boarding points: {url}")
        try:
            async with self.http_session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Error fetching boarding points: {response.status}")
                    return []
        except Exception as e:
            print(f"Exception fetching boarding points: {e}")
            return []
    
    async def fetch_dropping_points(self, trip_id):
        if not self.http_session:
            await self.init_http_session()
        url = f"{self.BASE_URL}/trips/{trip_id}/droppings"
        print(f"Fetching dropping points: {url}")
        try:
            async with self.http_session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Error fetching dropping points: {response.status}")
                    return []
        except Exception as e:
            print(f"Exception fetching dropping points: {e}")
            return []
    
    async def fetch_seats(self, trip_id, source_id, destination_id):
        if not self.http_session:
            await self.init_http_session()
        url = f"{self.BASE_URL}/trips/{trip_id}/seats?source_id={source_id}&destination_id={destination_id}"
        print(f"Fetching seats: {url}")
        try:
            async with self.http_session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Error fetching seats: {response.status}")
                    return {}
        except Exception as e:
            print(f"Exception fetching seats: {e}")
            return {}
    
    def find_nearest_boarding_points(self, boarding_points, user_location, max_points=3):
        if not user_location or not boarding_points:
            return []
        user_lat = user_location.get('latitude')
        user_lon = user_location.get('longitude')
        if not user_lat or not user_lon:
            return []
        boarding_with_distance = []
        for bp in boarding_points:
            bp_info = bp.get('boardingPoint', {})
            bp_lat = bp_info.get('latitude')
            bp_lon = bp_info.get('longitude')
            if bp_lat and bp_lon:
                distance = self.calculate_distance(user_lat, user_lon, bp_lat, bp_lon)
                boarding_with_distance.append((bp, distance))
        boarding_with_distance.sort(key=lambda x: x[1])
        return [bp[0] for bp in boarding_with_distance[:max_points]]
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371
        return c * r
    
    def suggest_boarding_point(self, boarding_points, user_location=None, last_boarding_point=None):
        if not boarding_points:
            return None
        if last_boarding_point:
            for point in boarding_points:
                if point.get('boardingPoint', {}).get('name', '').lower() == last_boarding_point.lower():
                    return point.get('boardingPoint', {}).get('name')
        if user_location and user_location.get('latitude') and user_location.get('longitude'):
            nearest_points = self.find_nearest_boarding_points(boarding_points, user_location, max_points=1)
            if nearest_points:
                return nearest_points[0].get('boardingPoint', {}).get('name')
        return boarding_points[0].get('boardingPoint', {}).get('name')
    
    def suggest_dropping_point(self, dropping_points, last_dropping_point=None):
        if not dropping_points:
            return None
        if last_dropping_point:
            for point in dropping_points:
                if point.get('droppingPoint', {}).get('name', '').lower() == last_dropping_point.lower():
                    return point.get('droppingPoint', {}).get('name')
        return dropping_points[0].get('droppingPoint', {}).get('name')
    
    def get_nearest_boarding_points_info(self, boarding_points, user_location, max_points=3):
        if not user_location or not boarding_points:
            return []
        nearest_points = self.find_nearest_boarding_points(boarding_points, user_location, max_points)
        if not nearest_points:
            return []
        nearest_info = []
        for point in nearest_points:
            bp_info = point.get('boardingPoint', {})
            name = bp_info.get('name', 'Unknown')
            landmark = bp_info.get('landmark', '')
            distance = self.calculate_distance(
                user_location.get('latitude'), 
                user_location.get('longitude'),
                bp_info.get('latitude'), 
                bp_info.get('longitude')
            )
            nearest_info.append({
                "name": name,
                "landmark": landmark,
                "distance_km": round(distance, 2)
            })
        return nearest_info
    
    async def fetch_user_preferences(self, mobile, auth_token=None):
        """Fetch user seat preferences and booking history if available"""
        if not self.http_session:
            await self.init_http_session()
        
        try:
            # This is a placeholder for a real API endpoint to fetch user data
            # In a real implementation, you would call the actual Fresh Bus API
            preferences = {
                "favorite_seats": [],
                "recent_routes": [],
                "preferred_bus_types": []
            }
            
            # Simulate fetching user's booking history
            # In a real implementation, you'd call the actual API endpoint
            headers = {}
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
                
            # Example of how to fetch user data with auth token
            # url = f"{self.BASE_URL}/users/{mobile}/bookings"
            # async with self.http_session.get(url, headers=headers) as response:
            #     if response.status == 200:
            #         bookings = await response.json()
            #         # Process booking history
            #         # Extract preferred seats, routes, etc.
            
            # For now, return simulated preferences based on mobile number
            # In a real implementation, this would come from the API
            seat_hash = sum(int(digit) for digit in mobile if digit.isdigit()) % 44 + 1
            preferences["favorite_seats"] = [seat_hash]
            
            # Add some reasonable preferences based on the seat number
            if seat_hash in self.window_seats:
                preferences["seat_position"] = "window"
            else:
                preferences["seat_position"] = "aisle"
                
            # Determine preferred category based on seat number
            if seat_hash in self.seat_categories["Premium"]:
                preferences["preferred_category"] = "Premium"
            elif seat_hash in self.seat_categories["Reasonable"]:
                preferences["preferred_category"] = "Reasonable"
            elif seat_hash in self.seat_categories["Budget-Friendly"]:
                preferences["preferred_category"] = "Budget-Friendly"
            else:
                preferences["preferred_category"] = "Low Reasonable"
                
            return preferences
            
        except Exception as e:
            print(f"Error fetching user preferences: {e}")
            return {}
    
    def get_seat_recommendations(self, seats_data, ticket_count=1, user_preferences=None):
        if not seats_data or 'seats' not in seats_data:
            return {}
        available_seats = [seat for seat in seats_data.get('seats', []) if seat['availabilityStatus'] == 'A']
        categorized_seats = {
            'Premium': {
                'window': [],
                'aisle': []
            },
            'Reasonable': {
                'window': [],
                'aisle': []
            },
            'Budget-Friendly': {
                'window': [],
                'aisle': []
            }
        }
        for seat in available_seats:
            seat_number = int(seat['seatName'])
            seat_price = seat['totalFare']
            category = None
            if seat_number in self.seat_categories['Premium']:
                category = 'Premium'
            elif seat_number in self.seat_categories['Reasonable']:
                category = 'Reasonable'
            elif seat_number in self.seat_categories['Budget-Friendly']:
                category = 'Budget-Friendly'
            elif seat_number in self.seat_categories['Low Reasonable']:
                category = 'Reasonable'
            position = None
            if seat_number in self.window_seats:
                position = 'window'
            elif seat_number in self.aisle_seats:
                position = 'aisle'
            if category and position:
                categorized_seats[category][position].append({
                    'number': seat_number,
                    'price': seat_price
                })
        
        # Consider user preferences for seat ordering
        if user_preferences:
            # If user has favorite seats, prioritize them
            favorite_seats = user_preferences.get('favorite_seats', [])
            preferred_category = user_preferences.get('preferred_category')
            preferred_position = user_preferences.get('seat_position')
            
            # Prioritize seats in user's favorite category and position
            if preferred_category and preferred_category in categorized_seats:
                for position in categorized_seats[preferred_category]:
                    # First sort by price (lowest first)
                    categorized_seats[preferred_category][position].sort(key=lambda x: x['price'])
                    
                    # Then prioritize favorite seats
                    if favorite_seats:
                        categorized_seats[preferred_category][position].sort(
                            key=lambda x: 0 if x['number'] in favorite_seats else 1
                        )
            
            # If user has a preferred position (window/aisle), prioritize that position
            if preferred_position:
                # Make sure to place user's preferred position first when displayed
                for category in categorized_seats:
                    if preferred_position in categorized_seats[category] and favorite_seats:
                        categorized_seats[category][preferred_position].sort(
                            key=lambda x: 0 if x['number'] in favorite_seats else 1
                        )
        else:
            # Without user preferences, just sort by price
            for category in categorized_seats:
                for position in categorized_seats[category]:
                    categorized_seats[category][position].sort(key=lambda x: x['price'])
        
        return categorized_seats
    
    def detect_language(self, text):
        if re.search(r'[\u0C00-\u0C7F]', text):
            return "telugu"
        if re.search(r'[\u0900-\u097F]', text):
            return "hindi"
        if re.search(r'[\u0B80-\u0BFF]', text):
            return "tamil"
        if re.search(r'[\u0C80-\u0CFF]', text):
            return "kannada"
        if re.search(r'[\u0D00-\u0D7F]', text):
            return "malayalam"
        tenglish_words = ["meeku", "naaku", "undi", "ledu", "kavalante", "cheyandi", "telugu"]
        if any(word in text.lower() for word in tenglish_words):
            return "tenglish"
        hinglish_words = ["kya", "hai", "nahi", "aap", "karenge", "chahiye", "dijiye"]
        if any(word in text.lower() for word in hinglish_words):
            return "hinglish"
        kanglish_words = ["naanu", "ninge", "bega", "swalpa", "banni", "guru"]
        if any(word in text.lower() for word in kanglish_words):
            return "kanglish"
        tamlish_words = ["enna", "neenga", "inge", "vanakkam", "romba", "irukku"]
        if any(word in text.lower() for word in tamlish_words):
            return "tamlish"
        return "english"
    
    def _build_direct_context_from_api_data(self, api_data, query, context):
        if not api_data:
            return None
        context_parts = []
        query_lower = query.lower()
        
        # Get user's requested direction
        user_source = context.get('user_requested_source')
        user_dest = context.get('user_requested_destination')
        use_user_direction = (user_source and user_dest)
        
        # Handle trip ended status first (highest priority)
        if api_data.get("eta_data") and api_data["eta_data"].get("message") == "This trip has ended":
            context_parts.append("\nTRIP STATUS: COMPLETED")
            context_parts.append("This trip has already ended. The bus has completed its journey.")
            context_parts.append(f"Trip ID: {api_data.get('trip_id', 'Unknown')}")
            
            # Add feedback info prominently if available
            if api_data.get("feedback_data"):
                feedback = api_data["feedback_data"]
                if isinstance(feedback, list) and len(feedback) > 0:
                    context_parts.append("\nFEEDBACK OPPORTUNITY:")
                    context_parts.append("You can now provide feedback on your completed journey.")
                    feedback_item = feedback[0]  # Use first feedback item
                    if feedback_item.get("questions"):
                        context_parts.append("Please rate your journey experience through the Fresh Bus app:")
                        for i, question in enumerate(feedback_item["questions"]):
                            context_parts.append(f"• {question.get('questionText', 'Rate your journey')}")
                    else:
                        context_parts.append("Please provide feedback on your journey through the Fresh Bus app.")
                elif feedback.get("questions"):
                    context_parts.append("\nFEEDBACK OPPORTUNITY:")
                    context_parts.append("You can now provide feedback on your completed journey.")
                    context_parts.append("Please rate your journey experience through the Fresh Bus app:")
                    for i, question in enumerate(feedback["questions"]):
                        context_parts.append(f"• {question.get('questionText', 'Rate your journey')}")
                else:
                    context_parts.append("\nFEEDBACK:")
                    context_parts.append(feedback.get("message", "You can rate your journey experience in the Fresh Bus app."))
            
            # Return immediately with trip ended status
            return "\n".join(context_parts)
        
        if api_data.get("trips"):
            trips = api_data.get("trips", [])
            trip_count = len(trips)
            
            # Add emphatic trip count information
            context_parts.append(f"IMPORTANT: There are EXACTLY {trip_count} unique bus services available.")
            if trip_count == 1:
                context_parts.append("There is ONLY ONE bus service available. Do not display or invent multiple options.")
            
            # List trip IDs for verification
            tripids = [trip.get('tripid') for trip in trips]
            context_parts.append(f"The unique Trip IDs are: {', '.join(str(tid) for tid in tripids)}")
            
            # Extract source and destination based on either user request or API data
            if trips:
                # Get API direction
                api_sources = set(trip.get('source', 'Unknown') for trip in trips)
                api_destinations = set(trip.get('destination', 'Unknown') for trip in trips)
                
                # Determine which direction to display
                if use_user_direction:
                    display_source = user_source
                    display_destination = user_dest
                    direction_note = f"USER REQUESTED DIRECTION: Buses requested FROM {display_source} TO {display_destination}. Showing available bus data for this route."
                else:
                    display_source = next(iter(api_sources)) if api_sources else "Unknown"
                    display_destination = next(iter(api_destinations)) if api_destinations else "Unknown"
                    direction_note = f"ROUTE INFORMATION: Buses available FROM {display_source} TO {display_destination}."
                
                context_parts.append(direction_note)
            
            context_parts.append("\nAvailable bus options:")
            for i, trip in enumerate(trips):
                # Add tripid to each bus listing to ensure uniqueness
                context_parts.append(f"Bus #{i+1} - Trip ID: {trip.get('tripid', 'Unknown')}")
                
                # Handle time conversion properly
                boarding_time = "N/A"
                if "boardingtime" in trip:
                    try:
                        dt = datetime.fromisoformat(trip["boardingtime"].replace('Z', '+00:00'))
                        ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
                        boarding_time = ist_dt.strftime("%I:%M %p IST")
                    except:
                        boarding_time = trip.get("boardingtime", "").split("T")[1].split(".")[0] + " UTC"
                    
                dropping_time = "N/A"
                if "droppingtime" in trip:
                    try:
                        dt = datetime.fromisoformat(trip["droppingtime"].replace('Z', '+00:00'))
                        ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
                        dropping_time = ist_dt.strftime("%I:%M %p IST")
                    except:
                        dropping_time = trip.get("droppingtime", "").split("T")[1].split(".")[0] + " UTC"
                
                # Calculate duration
                duration = ""
                if "boardingtime" in trip and "droppingtime" in trip:
                    try:
                        boarding_dt = datetime.fromisoformat(trip["boardingtime"].replace('Z', '+00:00'))
                        dropping_dt = datetime.fromisoformat(trip["droppingtime"].replace('Z', '+00:00'))
                        duration_mins = (dropping_dt - boarding_dt).total_seconds() / 60
                        hours = int(duration_mins // 60)
                        mins = int(duration_mins % 60)
                        duration = f"{hours}h {mins}m"
                    except:
                        duration = ""
                
                # Get source and destination based on user's request or API data
                if use_user_direction:
                    source = user_source
                    destination = user_dest
                else:
                    source = trip.get('source', 'Unknown')
                    destination = trip.get('destination', 'Unknown')
                
                # Add bus information with correct direction
                context_parts.append(
                    f"Option {i+1}: Bus {trip.get('servicenumber', 'Unknown')}, " +
                    f"Route: FROM {source} TO {destination}, " +
                    f"departure: {boarding_time}, arrival: {dropping_time}, " +
                    (f"duration: {duration}, " if duration else "") +
                    f"price: ₹{trip.get('fare', 'Unknown')}, " +
                    f"bus type: {trip.get('vehicletype', 'Standard')}, " +
                    f"available seats: {trip.get('availableseats', 'N/A')}"
                )
                
                # Get boarding and dropping points
                boarding_point = trip.get('boardingpointname', 'Unknown boarding point')
                dropping_point = trip.get('droppingpointname', 'Unknown dropping point')
                
                # Swap boarding and dropping points if user direction is opposite of API direction
                if use_user_direction:
                    api_source = trip.get('source', '')
                    api_dest = trip.get('destination', '')
                    if (user_source.lower() != api_source.lower() or 
                        user_dest.lower() != api_dest.lower()):
                        # Swap boarding and dropping points
                        boarding_point, dropping_point = dropping_point, boarding_point
                
                context_parts.append(f"  → Boards at: {boarding_point}")
                context_parts.append(f"  → Drops at: {dropping_point}")
        
        if api_data.get("boarding_points"):
            context_parts.append("\nAvailable boarding points:")
            for point in api_data["boarding_points"]:
                bp_info = point.get('boardingPoint', {})
                name = bp_info.get('name', 'Unknown')
                landmark = bp_info.get('landmark', '')
                time = None
                if 'currentTime' in point:
                    dt = datetime.fromisoformat(point["currentTime"].replace('Z', '+00:00'))
                    ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
                    time = ist_dt.strftime("%I:%M %p")
                context_parts.append(
                    f"• {name}" + (f" (near {landmark})" if landmark else "") + (f", time: {time}" if time else "")
                )
        
        if api_data.get("nearest_boarding_points"):
            context_parts.append("\nNearest boarding points to your location:")
            for point in api_data["nearest_boarding_points"]:
                context_parts.append(
                    f"• {point['name']} ({point['distance_km']} km away)" + 
                    (f" - Near {point['landmark']}" if point.get('landmark') else "")
                )
        
        if api_data.get("dropping_points"):
            context_parts.append("\nAvailable dropping points:")
            for point in api_data["dropping_points"]:
                dp_info = point.get('droppingPoint', {})
                name = dp_info.get('name', 'Unknown')
                landmark = dp_info.get('landmark', '')
                time = None
                if 'currentTime' in point:
                    dt = datetime.fromisoformat(point["currentTime"].replace('Z', '+00:00'))
                    ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
                    time = ist_dt.strftime("%I:%M %p")
                context_parts.append(
                    f"• {name}" + (f" (near {landmark})" if landmark else "") + (f", time: {time}" if time else "")
                )
        
        if api_data.get("recommendations"):
            context_parts.append("\nAvailable seat categories:")
            for category, positions in api_data["recommendations"].items():
                window_count = len(positions['window'])
                aisle_count = len(positions['aisle'])
                if window_count or aisle_count:
                    context_parts.append(f"• {category}: {window_count} window, {aisle_count} aisle seats available")
                    if window_count > 0:
                        seat_examples = [f"#{s['number']} (₹{s['price']})" for s in positions['window'][:3]]
                        context_parts.append(f"  Window seats: {', '.join(seat_examples)}")
                    if aisle_count > 0:
                        seat_examples = [f"#{s['number']} (₹{s['price']})" for s in positions['aisle'][:3]]
                        context_parts.append(f"  Aisle seats: {', '.join(seat_examples)}")
        
        if api_data.get("selected_seats"):
            context_parts.append(f"\nSelected seats: {', '.join(map(str, api_data['selected_seats']))}")
        
        if api_data.get("suggested_boarding"):
            context_parts.append(f"\nSuggested boarding point: {api_data['suggested_boarding']}")
        
        if api_data.get("suggested_dropping"):
            context_parts.append(f"\nSuggested dropping point: {api_data['suggested_dropping']}")
        
        context_parts.append(f"\nNumber of passengers: {api_data.get('ticket_count', 1)}")
        
        # Add time zone conversion reminder
        context_parts.append("\nTIME ZONE REMINDER: All times are converted from UTC to IST by adding 5 hours and 30 minutes.")
        
        # Add note about user requested direction
        if use_user_direction:
            context_parts.append(f"\nDIRECTION NOTE: User requested to book a ticket FROM {user_source} TO {user_dest}. Show this direction in the response.")
        
        # Add user preferences information if available
        if api_data.get("user_preferences"):
            user_prefs = api_data.get("user_preferences")
            context_parts.append("\nUSER PREFERENCES:")
            
            if user_prefs.get("favorite_seats"):
                context_parts.append(f"• Favorite seats: {', '.join(map(str, user_prefs['favorite_seats']))}")
            
            if user_prefs.get("seat_position"):
                context_parts.append(f"• Preferred position: {user_prefs['seat_position']}")
                
            if user_prefs.get("preferred_category"):
                context_parts.append(f"• Preferred category: {user_prefs['preferred_category']}")
                
            if user_prefs.get("recent_routes"):
                context_parts.append(f"• Recent routes: {', '.join(user_prefs['recent_routes'])}")
        
        # Add ETA data if available (but not trip ended, which is handled above)
        if api_data.get("eta_data") and not api_data["eta_data"].get("message") == "This trip has ended":
            context_parts.append("\nBUS TRACKING INFORMATION:")
            eta_data = api_data.get("eta_data")
            
            if "message" in eta_data:
                context_parts.append(f"ETA Status: {eta_data['message']}")
            else:
                # Format proper ETA data with bus location
                if "currentLocation" in eta_data:
                    location = eta_data["currentLocation"]
                    context_parts.append(f"Current Bus Location: {location}")
                
                if "estimatedArrival" in eta_data:
                    eta = eta_data["estimatedArrival"]
                    context_parts.append(f"Estimated Arrival Time: {eta}")
                
                if "delayMinutes" in eta_data:
                    delay = eta_data["delayMinutes"]
                    context_parts.append(f"Delay: {delay} minutes")
            
            context_parts.append(f"Trip ID: {api_data.get('trip_id', 'Unknown')}")
        
        # Add matched ticket information
        if api_data.get("matched_ongoing_ticket"):
            ticket = api_data["matched_ongoing_ticket"]
            context_parts.append("\nONGOING JOURNEY:")
            context_parts.append(f"Journey from {ticket.get('source', 'Unknown')} to {ticket.get('destination', 'Unknown')}")
            context_parts.append(f"Date: {ticket.get('journeyDate', 'Unknown')}")
            context_parts.append(f"Trip ID: {ticket.get('tripId', 'Unknown')}")
            
            # Add ETA data for ongoing journey
            eta_data = ticket.get("eta_data", {})
            if "message" in eta_data and eta_data["message"] == "This trip has ended":
                context_parts.append("Status: The journey has ended.")
            elif "message" in eta_data:
                context_parts.append(f"Status: {eta_data['message']}")
            else:
                if "currentLocation" in eta_data:
                    context_parts.append(f"Current Location: {eta_data['currentLocation']}")
                if "estimatedArrival" in eta_data:
                    context_parts.append(f"Estimated Arrival: {eta_data['estimatedArrival']}")
                if "delayMinutes" in eta_data:
                    context_parts.append(f"Delay: {eta_data['delayMinutes']} minutes")
        
        # Add matched future ticket information
        if api_data.get("matched_future_ticket"):
            ticket = api_data["matched_future_ticket"]
            context_parts.append("\nUPCOMING JOURNEY:")
            context_parts.append(f"Journey from {ticket.get('source', 'Unknown')} to {ticket.get('destination', 'Unknown')}")
            context_parts.append(f"Date: {ticket.get('journeyDate', 'Unknown')}")
            context_parts.append(f"Trip ID: {ticket.get('tripId', 'Unknown')}")
            context_parts.append("Status: This journey is in the future. Live tracking will be available 30 minutes before departure.")
        
        # Add matched completed ticket information with feedback
        if api_data.get("matched_completed_ticket"):
            ticket = api_data["matched_completed_ticket"]
            context_parts.append("\nCOMPLETED JOURNEY:")
            context_parts.append(f"Journey from {ticket.get('source', 'Unknown')} to {ticket.get('destination', 'Unknown')}")
            context_parts.append(f"Date: {ticket.get('journeyDate', 'Unknown')}")
            context_parts.append(f"Trip ID: {ticket.get('tripId', 'Unknown')}")
            context_parts.append("Status: This journey has been completed.")
            
            # Add feedback questions if available
            if api_data.get("feedback_data"):
                feedback = api_data["feedback_data"]
                if "message" in feedback:
                    context_parts.append(f"Feedback: {feedback['message']}")
                else:
                    context_parts.append("\nFEEDBACK QUESTIONS:")
                    if "questions" in feedback:
                        for i, question in enumerate(feedback["questions"]):
                            context_parts.append(f"Question {i+1}: {question.get('questionText', '')}")
                    context_parts.append("\nYou can provide feedback on your journey through the Fresh Bus app or website.")
        
        # Add lists of future/ongoing tickets if available
        if api_data.get("future_tickets") or api_data.get("ongoing_tickets") or api_data.get("completed_tickets"):
            if api_data.get("future_tickets"):
                context_parts.append(f"\nYou have {len(api_data['future_tickets'])} upcoming journeys:")
                for i, ticket in enumerate(api_data['future_tickets'][:3]): # Limit to 3 for brevity
                    context_parts.append(f"{i+1}. From {ticket.get('source', 'Unknown')} to {ticket.get('destination', 'Unknown')} on {ticket.get('journeyDate', 'Unknown')}")
                    context_parts.append(f"   Trip ID: {ticket.get('tripId', 'Unknown')}")
                    
            if api_data.get("ongoing_tickets"):
                context_parts.append(f"\nYou have {len(api_data['ongoing_tickets'])} ongoing/imminent journeys:")
                for i, ticket in enumerate(api_data['ongoing_tickets']):
                    context_parts.append(f"{i+1}. From {ticket.get('source', 'Unknown')} to {ticket.get('destination', 'Unknown')} on {ticket.get('journeyDate', 'Unknown')}")
                    context_parts.append(f"   Trip ID: {ticket.get('tripId', 'Unknown')}")
                    eta_data = ticket.get("eta_data", {})
                    if "currentLocation" in eta_data:
                        context_parts.append(f"   Current Location: {eta_data['currentLocation']}")
            
            if api_data.get("completed_tickets"):
                context_parts.append(f"\nYou have {len(api_data['completed_tickets'])} completed journeys.")
        
        # Add final trip count reminder
        if api_data.get("trips"):
            trip_count = len(api_data["trips"])
            context_parts.append(f"\nFINAL REMINDER: There are EXACTLY {trip_count} unique bus services available. Display exactly {trip_count} bus options, no more and no less.")
        
        return "\n".join(context_parts)
    
    def _generate_fallback_response(self, api_data, query, context):
        """Generate a fallback response when Claude API is unavailable"""
        # Special handling for trip ended
        if api_data.get("eta_data") and api_data["eta_data"].get("message") == "This trip has ended":
            trip_id = api_data.get("trip_id", "Unknown")
            response = f"This trip (ID: {trip_id}) has ended. The bus has completed its journey.\n\n"
            
            # Check for feedback data
            if api_data.get("feedback_data"):
                feedback = api_data["feedback_data"]
                
                # Handle array format
                if isinstance(feedback, list) and len(feedback) > 0:
                    response += "You can provide feedback on your journey through the Fresh Bus app.\n\n"
                    
                    # Use the first feedback item in the array
                    first_feedback = feedback[0]
                    if first_feedback.get("questions"):
                        response += "Feedback questions:\n"
                        for i, question in enumerate(first_feedback["questions"]):
                            response += f"{i+1}. {question.get('questionText', 'Rate your journey')}\n"
                # Handle object format
                elif feedback.get("questions"):
                    response += "You can provide feedback on your journey through the Fresh Bus app.\n\n"
                    response += "Feedback questions:\n"
                    for i, question in enumerate(feedback["questions"]):
                        response += f"{i+1}. {question.get('questionText', 'Rate your journey')}\n"
                elif feedback.get("message"):
                    response += feedback.get("message")
                else:
                    response += "You can provide feedback on your journey through the Fresh Bus app."
            else:
                response += "You can check for feedback options in the Fresh Bus app."
            
            return response
            
        if not api_data or not api_data.get("trips"):
            return "I'm sorry, I couldn't find any buses matching your criteria right now. Please try different dates or routes."
        
        # Use the direct bus response generator for consistent formatting
        return self.generate_direct_bus_response(api_data, context)
    
    async def generate_response(self, query, session_id=None, user_gender=None, user_location=None, detected_language=None):
        session, session_id = self.get_or_create_session(session_id)
        context = session['context']
        
        # Check if user is authenticated
        user_authenticated = False
        user_data = None
        user_mobile = None
        auth_token = None
        
        if context.get('auth') and context.get('user'):
            user_authenticated = True
            user_data = context['user']
            user_mobile = user_data.get('mobile')
            auth_token = context['auth'].get('token')
            print(f"Processing request for authenticated user: {user_mobile}")
            print(f"User profile data: {context.get('user_profile')}")
        
        # Improved location handling
        if user_location:
            print(f"Processing user location: {user_location}")
            
            # Ensure the location format is correct
            if isinstance(user_location, dict) and 'latitude' in user_location and 'longitude' in user_location:
                try:
                    # Validate latitude and longitude values
                    lat = float(user_location['latitude'])
                    lon = float(user_location['longitude'])
                    
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        print(f"Valid location data: Lat {lat}, Lon {lon}")
                        context['user_location'] = user_location
                    else:
                        print(f"Invalid location coordinates: Lat {lat}, Lon {lon}")
                        context['user_location'] = None
                except (ValueError, TypeError) as e:
                    print(f"Invalid location data: {e}")
                    context['user_location'] = None
            else:
                print(f"Invalid location format: {user_location}")
                context['user_location'] = None
            
        if detected_language:
            context['language'] = detected_language
        else:
            context['language'] = self.detect_language(query)
        print(f"Language detected/set: {context['language']}")
        
        # Extract user's requested source and destination from query
        source, destination = self.extract_locations(query)
        if source and destination:
            # Store user's requested direction
            context['user_requested_source'] = source
            context['user_requested_destination'] = destination
            context['last_source'] = source
            context['last_destination'] = destination
            print(f"User requested direction: FROM {source} TO {destination}")
        elif not source and context.get('last_source'):
            source = context['last_source']
        elif not destination and context.get('last_destination'):
            destination = context['last_destination']
        
        ticket_count = self.extract_ticket_count(query)
        if ticket_count:
            context['ticket_count'] = ticket_count
        
        session['messages'].append({"role": "user", "content": query})
        api_data = {}
        
        # Check if user is asking about bus location or tracking
        is_tracking_request = any(phrase in query.lower() for phrase in 
                                ["where is my bus", "track my bus", "bus location", "track bus", 
                                "bus tracking", "where is the bus", "bus eta", "arrival time", 
                                "when will bus arrive", "trip id", "tripid", "trip status"])

        if is_tracking_request:
            print("Bus tracking request detected")
            trip_id = None
            
            # Better trip ID extraction
            trip_id_match = re.search(r'trip\s*id\s*(?:is|:)?\s*(\d+)', query.lower())
            if not trip_id_match:
                # Also check for just numbers (people often just paste the ID)
                trip_id_match = re.search(r'\b(\d{5,})\b', query.lower())
            if trip_id_match:
                trip_id = trip_id_match.group(1)
                print(f"Trip ID found in query: {trip_id}")
            
            # If user is authenticated, check for tickets
            if user_authenticated and auth_token:
                print("Checking user's tickets")
                ticket_status = await self.get_active_tickets_and_status(auth_token)
                
                if ticket_status:
                    # Add the categorized tickets to API data
                    api_data["future_tickets"] = ticket_status.get("future", [])
                    api_data["ongoing_tickets"] = ticket_status.get("ongoing", [])
                    api_data["completed_tickets"] = ticket_status.get("completed", [])
                    
                    # If there's a trip ID, try to find it in tickets
                    if trip_id:
                        # Check ongoing tickets first
                        ongoing_match = next((t for t in ticket_status.get("ongoing", []) 
                                            if t.get("tripId") == trip_id), None)
                        if ongoing_match:
                            api_data["matched_ongoing_ticket"] = ongoing_match
                            api_data["eta_data"] = ongoing_match.get("eta_data", {})
                        
                        # Then check future tickets
                        future_match = next((t for t in ticket_status.get("future", []) 
                                        if t.get("tripId") == trip_id), None)
                        if future_match:
                            api_data["matched_future_ticket"] = future_match
                        
                        # Finally check completed tickets
                        completed_match = next((t for t in ticket_status.get("completed", []) 
                                            if t.get("tripId") == trip_id), None)
                        if completed_match:
                            api_data["matched_completed_ticket"] = completed_match
                            api_data["feedback_data"] = completed_match.get("feedback_data", {})
                    
                    # If no trip ID was provided, check if there's exactly one ongoing ticket
                    elif len(ticket_status.get("ongoing", [])) == 1:
                        ticket = ticket_status["ongoing"][0]
                        trip_id = ticket.get("tripId")
                        api_data["matched_ongoing_ticket"] = ticket
                        api_data["eta_data"] = ticket.get("eta_data", {})
                        print(f"Using trip ID from ongoing ticket: {trip_id}")
                        
                    # If there are no ongoing tickets but exactly one future ticket
                    elif not ticket_status.get("ongoing") and len(ticket_status.get("future", [])) == 1:
                        ticket = ticket_status["future"][0]
                        trip_id = ticket.get("tripId")
                        api_data["matched_future_ticket"] = ticket
                        api_data["trip_id"] = trip_id
                        print(f"Found future ticket with trip ID: {trip_id}")
                        
                    # If there are no ongoing or future tickets but exactly one completed ticket
                    elif not ticket_status.get("ongoing") and not ticket_status.get("future") and len(ticket_status.get("completed", [])) == 1:
                        ticket = ticket_status["completed"][0]
                        trip_id = ticket.get("tripId")
                        api_data["matched_completed_ticket"] = ticket
                        api_data["feedback_data"] = ticket.get("feedback_data", {})
                        print(f"Found completed ticket with trip ID: {trip_id}")
            
            # If we have a trip ID but no ticket match yet, fetch ETA data directly
            if trip_id and not api_data.get("matched_ongoing_ticket") and not api_data.get("matched_future_ticket") and not api_data.get("matched_completed_ticket"):
                print(f"Fetching ETA data for trip ID: {trip_id}")
                try:
                    eta_url = f"https://api.freshbus.com/eta-data?id={trip_id}"
                    
                    async with self.http_session.get(eta_url) as eta_response:
                        if eta_response.status == 200:
                            eta_data = await eta_response.json()
                            api_data["eta_data"] = eta_data
                            api_data["trip_id"] = trip_id
                            print(f"ETA data fetched: {eta_data}")
                            
                            # If trip has ended, immediately get feedback data
                            if eta_data.get("message") == "This trip has ended" and user_authenticated and auth_token:
                                try:
                                    # Use the base URL directly
                                    feedback_url = f"{self.BASE_URL}/tickets/feedbackQuestions"
                                    print(f"Fetching feedback for ended trip: {feedback_url}")
                                    
                                    headers = {
                                        "Authorization": f"Bearer {auth_token}",
                                        "Content-Type": "application/json"
                                    }
                                    
                                    async with self.http_session.get(feedback_url, headers=headers) as fb_response:
                                        print(f"Feedback response status: {fb_response.status}")
                                        
                                        if fb_response.status == 200:
                                            feedback_data = await fb_response.json()
                                            print(f"Feedback data received: {feedback_data}")
                                            
                                            # Use the whole feedback data
                                            api_data["feedback_data"] = feedback_data
                                            
                                            # Also find specific trip feedback if available
                                            trip_feedback = next((fb for fb in feedback_data if str(fb.get("tripId")) == str(trip_id)), None)
                                            if trip_feedback:
                                                api_data["trip_feedback"] = trip_feedback
                                        else:
                                            error_text = await fb_response.text()
                                            print(f"Error fetching feedback: {fb_response.status}, {error_text}")
                                            api_data["feedback_data"] = {"message": f"Could not fetch feedback questions: {error_text}"}
                                except Exception as fb_err:
                                    print(f"Exception fetching feedback: {fb_err}")
                                    api_data["feedback_data"] = {"message": f"Error: {str(fb_err)}"}
                        else:
                            error_text = await eta_response.text()
                            print(f"Error fetching ETA data: {eta_response.status}, {error_text}")
                            api_data["eta_error"] = error_text
                except Exception as eta_err:
                    print(f"Exception fetching ETA data: {eta_err}")
                    api_data["eta_error"] = str(eta_err)
        
        # If it's not a tracking request or we couldn't find any tracking data, proceed with regular bus booking flows
        if source and destination:
            journey_date = self.parse_date(query)
            if journey_date:
                context['last_date'] = journey_date
            
            source_id = self.stations.get(source)
            destination_id = self.stations.get(destination)
            
            if source_id and destination_id:
                # Try to fetch trips with user's requested direction
                trips_data = await self.fetch_trips(source_id, destination_id, journey_date)
                
                # If no trips found, try the reverse direction
                if not trips_data or len(trips_data) == 0:
                    print(f"No trips found FROM {source} TO {destination}, trying reverse direction")
                    trips_data = await self.fetch_trips(destination_id, source_id, journey_date)
                    
                    # If trips found in reverse direction, add a flag to indicate we need to swap
                    if trips_data and len(trips_data) > 0:
                        print(f"Found trips in reverse direction FROM {destination} TO {source}")
                        context['reverse_direction'] = True
                        api_data["user_direction"] = {
                            "source": source,
                            "destination": destination,
                            "swap_points": True  # Flag to swap boarding/dropping points
                        }
                
                # Make sure we're working with deduplicated trips
                trips_data = self.deduplicate_trips(trips_data)
                print(f"Processing {len(trips_data)} unique trips")
                
                api_data["trips"] = trips_data
                
                selected_bus_index = self.detect_bus_selection(query, trips_data)
                if trips_data and len(trips_data) > 0:
                    trip_index = selected_bus_index if selected_bus_index is not None else 0
                    if selected_bus_index is not None:
                        context['selected_bus'] = trips_data[trip_index].get('servicenumber')
                    
                    trip_id = trips_data[trip_index]["tripid"]
                    
                    # Get IDs for API calls based on the direction of the found trips
                    api_source_id = source_id
                    api_destination_id = destination_id
                    
                    # If we're using reverse direction trips, swap the IDs for API calls
                    if context.get('reverse_direction'):
                        api_source_id, api_destination_id = destination_id, source_id
                    
                    # Fetch boarding, dropping and seats concurrently
                    tasks = [
                        self.fetch_boarding_points(trip_id, api_source_id),
                        self.fetch_dropping_points(trip_id),
                        self.fetch_seats(trip_id, api_source_id, api_destination_id)
                    ]
                    boarding_points, dropping_points, seats_data = await asyncio.gather(*tasks)
                    
                    api_data["boarding_points"] = boarding_points
                    user_loc = context.get('user_location')
                    if user_loc:
                        nearest_points = self.get_nearest_boarding_points_info(boarding_points, user_loc, max_points=3)
                        api_data["nearest_boarding_points"] = nearest_points
                    
                    suggested_boarding = self.suggest_boarding_point(boarding_points, user_loc, context.get('last_boarding_point'))
                    api_data["suggested_boarding"] = suggested_boarding
                    
                    api_data["dropping_points"] = dropping_points
                    suggested_dropping = self.suggest_dropping_point(dropping_points, context.get('last_dropping_point'))
                    api_data["suggested_dropping"] = suggested_dropping
                    
                    api_data["seats"] = seats_data
                    
                    # If user is authenticated, fetch their preferences
                    user_preferences = None
                    if user_authenticated and user_mobile:
                        user_preferences = await self.fetch_user_preferences(user_mobile, auth_token)
                        api_data["user_preferences"] = user_preferences
                    
                    # Get seat recommendations with user preferences if available
                    recommendations = self.get_seat_recommendations(
                        seats_data, 
                        context.get('ticket_count', 1),
                        user_preferences
                    )
                    api_data["recommendations"] = recommendations
                    
                    seats, window_pref, aisle_pref = self.detect_seat_selection(query)
                    if seats:
                        context['selected_seats'] = seats
                        api_data["selected_seats"] = seats
                    
                    api_data["window_preference"] = window_pref
                    api_data["aisle_preference"] = aisle_pref
        
        api_data["ticket_count"] = context.get('ticket_count', 1)
        
        # Pass user's requested direction to api_data if needed
        if context.get('user_requested_source') and context.get('user_requested_destination'):
            if not api_data.get("user_direction"):
                api_data["user_direction"] = {
                    "source": context['user_requested_source'],
                    "destination": context['user_requested_destination'],
                    "swap_points": context.get('reverse_direction', False)
                }
        
        # Add user profile to API data if available
        if context.get('user_profile'):
            api_data["user_profile"] = context.get('user_profile')
        
        # Special direct handling for trip ended case
        if api_data.get("eta_data") and api_data["eta_data"].get("message") == "This trip has ended":
            trip_id = api_data.get("trip_id", "Unknown")
            direct_response = f"This trip (ID: {trip_id}) has ended. The bus has completed its journey.\n\n"
            
            # Check for feedback data
            if api_data.get("feedback_data"):
                feedback = api_data["feedback_data"]
                
                # First try trip-specific feedback
                if api_data.get("trip_feedback"):
                    trip_feedback = api_data["trip_feedback"]
                    direct_response += "You can provide feedback on your journey through the Fresh Bus app.\n\n"
                    
                    if trip_feedback.get("questions"):
                        direct_response += "Feedback questions:\n"
                        for i, question in enumerate(trip_feedback["questions"]):
                            direct_response += f"{i+1}. {question.get('questionText', 'Rate your journey')}\n"
                # Then try any feedback data
                elif isinstance(feedback, list) and len(feedback) > 0:
                    direct_response += "You can provide feedback on your journey through the Fresh Bus app.\n\n"
                    direct_response += "Feedback questions:\n"
                    
                    # Show questions from the first feedback item
                    if feedback[0].get("questions"):
                        for i, question in enumerate(feedback[0]["questions"]):
                            direct_response += f"{i+1}. {question.get('questionText', 'Rate your journey')}\n"
                # Handle message format
                elif feedback.get("message"):
                    direct_response += feedback.get("message")
                # Default message
                else:
                    direct_response += "You can provide feedback on your journey through the Fresh Bus app."
            else:
                direct_response += "You can check for feedback options in the Fresh Bus app."
            
            # Save the direct response
            session['messages'].append({"role": "assistant", "content": direct_response})
            
            # Try to save to Redis
            try:
                conversation_id = conversation_manager.save_conversation(session_id, session['messages'])
                print(f"Saved direct trip ended response to Redis: {conversation_id}")
            except Exception as redis_err:
                print(f"Failed to save to Redis: {redis_err}")
                conversation_id = None
            
            # Return the direct response instead of calling Claude
            yield json.dumps({"text": direct_response, "done": False})
            yield json.dumps({
                "text": "", 
                "done": True, 
                "session_id": session_id, 
                "language_style": context.get('language', 'english'),
                "conversation_id": conversation_id
            })
            return
        
        # Check if this is a bus listing request and we have trip data
        is_bus_listing_request = any(kw in query.lower() for kw in ["book", "find", "search", "trip", "bus", "ticket"])
        
        if is_bus_listing_request and api_data and api_data.get("trips"):
            # Use direct response generation for bus listings
            print("Using direct bus response generator to avoid hallucinations")
            direct_response = self.generate_direct_bus_response(api_data, context)
            
            # Save the direct response
            session['messages'].append({"role": "assistant", "content": direct_response})
            
            # Try to save to Redis
            try:
                conversation_id = conversation_manager.save_conversation(session_id, session['messages'])
                print(f"Saved direct response to Redis: {conversation_id}")
            except Exception as redis_err:
                print(f"Failed to save to Redis: {redis_err}")
                conversation_id = None
            
            # Return the direct response instead of calling Claude
            yield json.dumps({"text": direct_response, "done": False})
            yield json.dumps({
                "text": "", 
                "done": True, 
                "session_id": session_id, 
                "language_style": context.get('language', 'english'),
                "conversation_id": conversation_id
            })
            return
        
        # Only call vector_db operations if we have real API data and it's not a direct response
        relevant_context = {"documents": ["I don't have specific information about that."]}
        try:
            if api_data:
                await self.vector_db.store_api_data(session_id, api_data)
                relevant_context = await self.vector_db.query_collection(f"api_data_{session_id}", query, n_results=5)
        except Exception as e:
            print(f"Vector DB operation failed: {e}")
            # Continue with direct context building
        
        # If vector DB returns minimal or no context, build direct context.
        if not relevant_context or not relevant_context.get("documents"):
            direct_context = self._build_direct_context_from_api_data(api_data, query, context)
            if direct_context:
                relevant_context = {"documents": [direct_context]}
            else:
                relevant_context = {"documents": ["I don't have specific information about that."]}
        else:
            # Flatten documents if needed
            documents = relevant_context.get("documents")
            if documents and isinstance(documents[0], list):
                documents = [doc for sublist in documents for doc in sublist]
            relevant_context = {"documents": documents}
        
        # For simplicity, combine available documents as context
        combined_context = "\n\n".join(relevant_context["documents"])
        
        dynamic_system_prompt = ""
        try:
            dynamic_system_prompt = await self.vector_db.get_system_prompt(query)
            dynamic_system_prompt = self.trim_text(dynamic_system_prompt, token_limit=1000)
        except Exception as e:
            print(f"Error getting system prompt: {e}")
            dynamic_system_prompt = "You are a Fresh Bus travel assistant. Provide accurate bus information based on API data only."
        
        system_message = dynamic_system_prompt
        system_message += "\n\nIMPORTANT: You are a bus booking assistant. Use only the information provided below to answer user queries. Always mention specific details from the API data about buses, times, prices, boarding points, etc. when available. Do not give generic responses when specific data is available."
        
        # Add trip count instruction
        if api_data and api_data.get("trips"):
            trip_count = len(api_data["trips"])
            system_message += f"\n\nTRIP COUNT: There are EXACTLY {trip_count} unique bus services available. Display exactly {trip_count} bus options, no more and no less."
            
            if trip_count == 1:
                system_message += " There is ONLY ONE bus service available. Do not display or invent multiple options."
        
        # Add instruction to respect user's requested direction
        if context.get('user_requested_source') and context.get('user_requested_destination'):
            user_source = context['user_requested_source']
            user_dest = context['user_requested_destination']
            system_message += f"\n\nDIRECTION INSTRUCTION: User has requested to book a bus FROM {user_source} TO {user_dest}. Always display this direction in your response, regardless of what the API data shows."
        
        # Add explicit time conversion instruction
        system_message += "\n\nTIME CONVERSION: All times must be converted from UTC to IST by adding 5 hours and 30 minutes."
        
        # Add user authentication info if available
        if user_authenticated:
            system_message += f"\n\n--- USER INFORMATION ---"
            system_message += f"\nAuthenticated user: {user_mobile}"
            
            # Add user preferences if available
            if api_data.get("user_preferences"):
                prefs = api_data["user_preferences"]
                system_message += "\nUser preferences:"
                
                if prefs.get("favorite_seats"):
                    system_message += f"\n- Favorite seats: {', '.join(map(str, prefs['favorite_seats']))}"
                
                if prefs.get("seat_position"):
                    system_message += f"\n- Preferred position: {prefs['seat_position']}"
                    
                if prefs.get("preferred_category"):
                    system_message += f"\n- Preferred category: {prefs['preferred_category']}"
                
                system_message += "\nWhen recommending seats, prioritize those that match the user's preferences."
        
        # Add user profile information if available
        if context.get('user_profile'):
            profile = context.get('user_profile')
            system_message += f"\n\n--- USER PROFILE INFORMATION ---\n"
            
            # Add each field that's available
            for field in ['name', 'mobile', 'email', 'gender', 'age', 'address', 'availableCoins']:
                if profile.get(field):
                    system_message += f"{field.capitalize()}: {profile[field]}\n"
            
            system_message += "\nWhen the user asks about their profile, make sure to include their name and other relevant details from this information."
        
        # Add special handling for bus tracking requests
        if is_tracking_request:
            system_message += "\n\n--- BUS TRACKING REQUEST ---\n"
            system_message += "The user is asking about bus tracking or location information. "
            
            if api_data.get("eta_data"):
                system_message += "I have ETA data available for their query."
                if "message" in api_data["eta_data"] and api_data["eta_data"]["message"] == "This trip has ended":
                    system_message += " The trip has already ended/completed."
            
            if api_data.get("feedback_data"):
                system_message += "\nIMPORTANT: Feedback data is available for this completed trip. "
                system_message += "Make sure to mention feedback options to the user."
            
            # Add special prompt for trip that has ended
            if api_data.get("eta_data") and api_data["eta_data"].get("message") == "This trip has ended":
                system_message += "\n\n--- COMPLETED TRIP ---\n"
                system_message += "This trip (ID: " + api_data.get("trip_id", "Unknown") + ") has ended. "
                system_message += "The bus has completed its journey. "
                system_message += "IMPORTANT: Tell the user clearly that their trip has ended and the journey is complete. "
                system_message += "IMPORTANT: Focus on providing the feedback opportunity information from the context. "
                system_message += "Encourage the user to provide feedback on their journey through the Fresh Bus app. "
                system_message += "DO NOT suggest any real-time tracking as the journey is already complete."
        
        system_message += f"\n\n--- LANGUAGE INFO ---\nDetected language: {context.get('language', 'english')}\n"
        if context.get('language') != 'english':
            system_message += f"IMPORTANT: Respond in {context.get('language')}. If the user used transliteration, respond in the same style.\n"
        
        system_message += "\n\n--- RELEVANT INFORMATION ---\n" + combined_context
        system_message += "\n\n--- CONTEXT ---\n"
        
        essential_context = {
            'source': context.get('user_requested_source') or context.get('last_source'),
            'destination': context.get('user_requested_destination') or context.get('last_destination'),
            'travel_date': context.get('last_date'),
            'ticket_count': context.get('ticket_count', 1),
        }
        
        for key, value in essential_context.items():
            if value:
                system_message += f"{key}: {value}\n"
        
        if context.get('user_location'):
            system_message += f"user_has_location_data: true\n"
        
        # Add final note about respecting user's direction
        if context.get('user_requested_source') and context.get('user_requested_destination'):
            system_message += f"\n\nFINAL DIRECTION NOTE: The user wants to travel FROM {context['user_requested_source']} TO {context['user_requested_destination']}. Never 'correct' this direction in your response."
        
        system_message += "\n\nNEVER add a 'CORRECTION:' note to your response. If a user asks for a specific journey direction, always respect their request."
        
        # Add final reminder about trip count
        if api_data and api_data.get("trips"):
            trip_count = len(api_data["trips"])
            trip_ids = [trip.get('tripid') for trip in api_data["trips"]]
            system_message += f"\n\nFINAL TRIP COUNT REMINDER: There are EXACTLY {trip_count} unique bus services with Trip IDs: {trip_ids}. Do not show more or fewer bus options than this exact number."
        
        # Add final reminder about user profile
        if "profile" in query.lower() and context.get('user_profile'):
            system_message += f"\n\nPROFILE QUERY DETECTED: User is asking about their profile. Make sure to include their complete profile details including name, email, mobile, age, etc. from the USER PROFILE INFORMATION section."
        
        # Add reminders for bus tracking
        if is_tracking_request:
            system_message += "\n\nTRACKING REQUEST REMINDER: The user is asking about bus location/ETA. Focus your response on providing tracking information from the context. If no tracking data is available, suggest providing a Trip ID or logging in to see active tickets."
        
        token_count = self.vector_db.get_token_count(system_message)
        print(f"System message token count: {token_count}")
        recent_messages = session['messages'][-3:]
        
        try:
            # Set a longer timeout and implement retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    with anthropic_client.messages.stream(
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=2000,
                        system=system_message,
                        messages=recent_messages,
                        timeout=45  # Increased timeout
                    ) as stream:
                        response_text = ""
                        for text in stream.text_stream:
                            response_text += text
                            yield json.dumps({"text": text, "done": False})
                        
                        # Remove any "CORRECTION:" text and enforce proper format
                        corrected_response = self.enforce_response_format(response_text, api_data, context)
                        
                        # If the response was modified
                        if corrected_response != response_text:
                            response_text = corrected_response
                            yield json.dumps({"text": corrected_response, "replace": True, "done": False})
                        
                        session['messages'].append({"role": "assistant", "content": response_text})
                        
                        # Save conversation to Redis if available
                        try:
                            conversation_id = conversation_manager.save_conversation(session_id, session['messages'])
                            if conversation_id:
                                print(f"Saved conversation to Redis with ID: {conversation_id}")
                        except Exception as redis_err:
                            print(f"Failed to save to Redis: {redis_err}")
                            conversation_id = None
                        
                        yield json.dumps({
                            "text": "", 
                            "done": True, 
                            "session_id": session_id, 
                            "language_style": context.get('language', 'english'),
                            "conversation_id": conversation_id
                        })
                        
                        # Successful response - exit retry loop
                        break
                        
                except Exception as retry_err:
                    print(f"Claude API attempt {attempt+1}/{max_retries} failed: {str(retry_err)}")
                    if attempt < max_retries - 1:
                        # Wait before retrying with exponential backoff
                        await asyncio.sleep(2 ** attempt)
                    else:
                        # All retries failed, use fallback
                        raise retry_err
                        
        except Exception as e:
            error_msg = f"Error with Claude API: {str(e)}"
            print(error_msg)
            
            # Use fallback response mechanism
            fallback_response = self._generate_fallback_response(api_data, query, context)
            print("Using fallback response mechanism")
            session['messages'].append({"role": "assistant", "content": fallback_response})
            
            # Try to save to Redis even with fallback
            try:
                conversation_id = conversation_manager.save_conversation(session_id, session['messages'])
                if conversation_id:
                    print(f"Saved fallback conversation to Redis with ID: {conversation_id}")
            except Exception as redis_err:
                print(f"Failed to save fallback to Redis: {redis_err}")
                conversation_id = None
            
            # Return the fallback response
            yield json.dumps({"text": fallback_response, "done": False})
            yield json.dumps({
                "text": "", 
                "done": True, 
                "session_id": session_id,
                "language_style": context.get('language', 'english'),
                "conversation_id": conversation_id
            })

# Instantiate the Fresh Bus assistant
fresh_bus_assistant = FreshBusAssistant()

# Create the FastAPI app with lifespan
app = FastAPI(title="Fresh Bus Travel Assistant", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

#################################
# API Endpoints
#################################

@app.get("/trip-feedback/{trip_id}")
async def get_trip_feedback(trip_id: str, request: Request):
    """Get feedback questions for a specific trip"""
    try:
        # Get the auth token from the request headers
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"success": False, "message": "Authentication required"}
            )
        
        auth_token = auth_header.replace("Bearer ", "")
        
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()
        
        # Call the feedback API endpoint
        url = f"{fresh_bus_assistant.BASE_URL}/tickets/feedbackQuestions"
        print(f"Fetching feedback questions for trip {trip_id}: {url}")
        
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }
        
        async with fresh_bus_assistant.http_session.get(url, headers=headers) as response:
            if response.status == 200:
                all_feedback = await response.json()
                
                # Find feedback for this specific trip
                trip_feedback = next((fb for fb in all_feedback if str(fb.get("tripId")) == str(trip_id)), None)
                
                if trip_feedback:
                    return JSONResponse(content=trip_feedback)
                else:
                    # Return all feedback data if no specific match found
                    return JSONResponse(content=all_feedback)
            else:
                error_text = await response.text()
                print(f"Error fetching feedback questions: {response.status}, {error_text}")
                return JSONResponse(
                    status_code=response.status,
                    content={"message": f"Failed to fetch feedback questions: {error_text}"}
                )
    except Exception as e:
        print(f"Exception fetching feedback questions: {e}")
        return JSONResponse(
            status_code=500, 
            content={"message": f"Error fetching feedback questions: {str(e)}"}
        )

@app.get("/eta-data/{trip_id}")
async def get_eta_data(trip_id: str, request: Request):
    """Get ETA data for a specific trip"""
    try:
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()
        
        # Call the ETA API endpoint
        url = f"https://api.freshbus.com/eta-data?id={trip_id}"
        print(f"Fetching ETA data: {url}")
        
        async with fresh_bus_assistant.http_session.get(url) as response:
            if response.status == 200:
                eta_data = await response.json()
                return JSONResponse(content=eta_data)
            else:
                error_text = await response.text()
                print(f"Error fetching ETA data: {response.status}, {error_text}")
                return JSONResponse(
                    status_code=response.status,
                    content={"message": f"Failed to fetch ETA data: {error_text}"}
                )
    except Exception as e:
        print(f"Exception fetching ETA data: {e}")
        return JSONResponse(
            status_code=500, 
            content={"message": f"Error fetching ETA data: {str(e)}"}
        )

@app.get("/user/tickets")
async def get_user_tickets(request: Request):
    """Get all tickets for the authenticated user"""
    try:
        # Get the auth token from the request headers
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"success": False, "message": "Authentication required"}
            )
        
        auth_token = auth_header.replace("Bearer ", "")
        
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()
        
        # Call the tickets API endpoint
        url = f"{fresh_bus_assistant.BASE_URL}/tickets"
        print(f"Fetching user tickets: {url}")
        
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }
        
        async with fresh_bus_assistant.http_session.get(url, headers=headers) as response:
            if response.status == 200:
                tickets_data = await response.json()
                return JSONResponse(content=tickets_data)
            else:
                error_text = await response.text()
                print(f"Error fetching tickets: {response.status}, {error_text}")
                return JSONResponse(
                    status_code=response.status,
                    content={"message": f"Failed to fetch tickets: {error_text}"}
                )
    except Exception as e:
        print(f"Exception fetching tickets: {e}")
        return JSONResponse(
            status_code=500, 
            content={"message": f"Error fetching tickets: {str(e)}"}
        )

@app.get("/tickets/feedbackQuestions")
async def get_feedback_questions(request: Request):
    """Get feedback questions for completed journeys"""
    try:
        # Get the auth token from the request headers
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"success": False, "message": "Authentication required"}
            )
        
        auth_token = auth_header.replace("Bearer ", "")
        
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()
        
        # Call the feedback API endpoint
        url = f"{fresh_bus_assistant.BASE_URL}/tickets/feedbackQuestions"
        print(f"Fetching feedback questions: {url}")
        
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }
        
        async with fresh_bus_assistant.http_session.get(url, headers=headers) as response:
            if response.status == 200:
                feedback_data = await response.json()
                return JSONResponse(content=feedback_data)
            else:
                error_text = await response.text()
                print(f"Error fetching feedback questions: {response.status}, {error_text}")
                return JSONResponse(
                    status_code=response.status,
                    content={"message": f"Failed to fetch feedback questions: {error_text}"}
                )
    except Exception as e:
        print(f"Exception fetching feedback questions: {e}")
        return JSONResponse(
            status_code=500, 
            content={"message": f"Error fetching feedback questions: {str(e)}"}
        )

@app.get("/profile")
async def get_user_profile(request: Request):
    """Get user profile data"""
    try:
        # Get the auth token from the request headers
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"success": False, "message": "Authentication required"}
            )
        
        auth_token = auth_header.replace("Bearer ", "")
        
        # Fetch profile data from Fresh Bus API
        profile_data = await fresh_bus_assistant.fetch_user_profile(auth_token)
        
        if not profile_data:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "Failed to fetch user profile"}
            )
        
        return JSONResponse(content=profile_data)
    except Exception as e:
        print(f"Error getting user profile: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error fetching profile: {str(e)}"}
        )

@app.post("/auth/sendotp")
async def send_otp(request: Request):
    """Proxy endpoint to send OTP to user's mobile number"""
    try:
        body = await request.json()
        mobile = body.get("mobile")
        
        if not mobile or len(mobile) != 10:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Please provide a valid 10-digit mobile number"}
            )
        
        # Forward request to Fresh Bus API
        api_url = f"{fresh_bus_assistant.BASE_URL}/auth/sendotp"
        
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()
        
        async with fresh_bus_assistant.http_session.post(
            api_url,
            json={"mobile": mobile},
            headers={"Content-Type": "application/json"}
        ) as response:
            response_data = await response.json()
            
            # Return same status code and response
            return JSONResponse(
                status_code=response.status,
                content=response_data
            )
    except Exception as e:
        print(f"Error sending OTP: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error sending OTP: {str(e)}"}
        )

@app.post("/auth/verifyotp")
async def verify_otp(request: Request):
    """Proxy endpoint to verify OTP and authenticate user"""
    try:
        body = await request.json()
        mobile = body.get("mobile")
        otp = body.get("otp")
        device_id = body.get("deviceId", "web_client")
        
        if not mobile or not otp:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Mobile number and OTP are required"}
            )
        
        # Forward request to Fresh Bus API
        api_url = f"{fresh_bus_assistant.BASE_URL}/auth/verifyotp"
        
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()
        
        async with fresh_bus_assistant.http_session.post(
            api_url,
            json={"mobile": mobile, "otp": otp, "deviceId": device_id},
            headers={"Content-Type": "application/json"}
        ) as response:
            response_data = await response.json()
            
            # Extract tokens from cookies
            cookies = response.cookies
            access_token = None
            refresh_token = None
            
            for cookie_name, cookie in cookies.items():
                if cookie_name == "access_token":
                    access_token = cookie.value
                elif cookie_name == "refresh_token":
                    refresh_token = cookie.value
            
            # If tokens exist in cookies but not in response, add them
            if access_token and "access_token" not in response_data:
                response_data["access_token"] = access_token
            
            if refresh_token and "refresh_token" not in response_data:
                response_data["refresh_token"] = refresh_token
            
            # Add user details if available
            if "user" not in response_data:
                response_data["user"] = {"mobile": mobile}
            
            # Return the response data
            return JSONResponse(
                status_code=response.status,
                content=response_data
            )
    except Exception as e:
        print(f"Error verifying OTP: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error verifying OTP: {str(e)}"}
        )

@app.post("/update-location")
async def update_location(request: Request):
    try:
        data = await request.json()
        session_id = data.get('session_id')
        location = data.get('location')
        
        if not session_id or not location:
            return JSONResponse(status_code=400, content={"success": False, "error": "Missing session_id or location"})
        
        session, _ = fresh_bus_assistant.get_or_create_session(session_id)
        session['context']['user_location'] = location
        
        print(f"Updated location for session {session_id}: {location}")
        
        return JSONResponse(content={"success": True})
    except Exception as e:
        print(f"Error updating location: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/redis-test")
async def test_redis_connection(request: Request):
    """Test Redis connection and list keys"""
    try:
        # Test Redis connection
        redis_ping = redis_client.ping()
        
        # Get all conversation keys 
        all_keys = redis_client.keys("fresh_bus:conversation:*")
        session_keys = redis_client.keys("fresh_bus:session_index:*")
        
        # Sample content of a conversation if any exist
        sample_conversation = None
        if all_keys:
            sample_key = all_keys[0]
            sample_conversation = redis_client.hgetall(sample_key)
        
        return JSONResponse({
            "redis_connected": redis_ping,
            "conversation_count": len(all_keys),
            "session_count": len(session_keys),
            "keys": [k for k in all_keys[:10]],  # Show first 10 keys
            "session_keys": [k for k in session_keys[:10]],
            "sample_conversation": sample_conversation
        })
    except Exception as e:
        return JSONResponse({
            "redis_connected": False,
            "error": str(e)
        })

@app.post("/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...)):
    """Handle speech-to-text conversion"""
    temp_file = None
    temp_file_path = None
    try:
        contents = await audio.read()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file_path = temp_file.name
        temp_file.write(contents)
        temp_file.close()
        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Speech recognition API key not configured"}
            )
        with open(temp_file_path, "rb") as f:
            detect_response = requests.post(
                "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": f},
                data={
                    "vad_model": "whisperx-pyannet",
                    "alignment_model": "tdnn_ffn",
                    "preprocessing": "dynamic",
                    "temperature": "0.3",
                    "timestamp_granularities": "segment",
                    "audio_window_seconds": "5",
                    "speculation_window_words": "4",
                    "task": "transcribe"
                },
                timeout=30  # Add timeout for request
            )
        if detect_response.status_code != 200:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": f"Speech recognition failed: {detect_response.text}"}
            )
        detect_result = detect_response.json()
        detected_language = detect_result.get("language", "en")
        print(f"Detected language: {detected_language}")
        language_style = LANGUAGE_CODES_REVERSE.get(detected_language, "english")
        with open(temp_file_path, "rb") as f:
            response = requests.post(
                "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": f},
                data={
                    "vad_model": "whisperx-pyannet",
                    "alignment_model": "tdnn_ffn",
                    "preprocessing": "dynamic",
                    "temperature": "0.3",
                    "timestamp_granularities": "segment",
                    "audio_window_seconds": "5",
                    "speculation_window_words": "4",
                    "language": detected_language,
                    "task": "transcribe"
                },
                timeout=30  # Add timeout for request
            )
        if response.status_code != 200:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": f"Speech recognition failed: {response.text}"}
            )
        result = response.json()
        transcribed_text = result.get("text", "")
        return {
            "success": True,
            "text": transcribed_text,
            "language": language_style
        }
    except Exception as e:
        print(f"Error in speech-to-text: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error processing speech: {str(e)}"}
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file_path}: {e}")
                atexit.register(lambda file=temp_file_path: os.remove(file) if os.path.exists(file) else None)

@app.post("/initialize")
async def initialize(request: InitializeRequest):
    """Initialize system prompt"""
    try:
        fresh_bus_assistant.load_system_prompt(request.system_prompt_path)
        await fresh_bus_assistant.init_system_prompt()
        return {"status": "success", "message": "System prompt updated successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.api_route("/query", methods=["GET", "POST"])
async def query(request: Request, response: Response):
    """Handle user query with optional auth token"""
    try:
        if request.method == "GET":
            query_str = request.query_params.get("query")
            session_id = request.query_params.get("session_id", "")
            gender = request.query_params.get("gender")
            if not query_str:
                raise HTTPException(status_code=400, detail="No query provided")
            query_request = QueryRequest(
                query=query_str,
                session_id=session_id,
                gender=gender
            )
        else:
            try:
                body = await request.json()
                # Add debug logging for location data
                if 'location' in body and body['location']:
                    print(f"Location data received: {body['location']}")
                
                # Extract auth data if present
                auth_data = None
                if 'auth' in body:
                    auth_data = body.pop('auth')
                    print(f"Auth data received for user: {auth_data.get('user', {}).get('mobile')}")
                
                query_request = QueryRequest(**body)
                
                # Store auth data in session context if available
                if auth_data and query_request.session_id:
                    session, _ = fresh_bus_assistant.get_or_create_session(query_request.session_id)
                    session['context']['auth'] = auth_data
                    session['context']['user'] = auth_data.get('user', {})
                    # Let the assistant know user is logged in
                    if auth_data.get('user', {}).get('mobile'):
                        print(f"User authenticated with mobile: {auth_data['user']['mobile']}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid request body: {str(e)}")
        
        detected_language = request.headers.get("X-Detected-Language")
        if query_request.location:
            print(f"Location received from client: {query_request.location}")
        
        async def stream_generator():
            async for chunk in fresh_bus_assistant.generate_response(
                query_request.query, 
                query_request.session_id,
                user_gender=query_request.gender,
                user_location=query_request.location,
                detected_language=detected_language
            ):
                yield f"data: {chunk}\n\n"
                try:
                    data = json.loads(chunk)
                    if data.get("done") and data.get("session_id"):
                        response.set_cookie(
                            key="session_id", 
                            value=data["session_id"],
                            max_age=30 * 60,
                            httponly=True,
                            samesite="lax"
                        )
                except Exception as e:
                    print(f"Error parsing response: {e}")
        
        return StreamingResponse(
            stream_generator(), 
            media_type="text/event-stream"
        )
    except Exception as e:
        print(f"Error in query endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/clear-session")
def clear_session(response: Response, session_id: str = None):
    """Clear session"""
    if session_id and session_id in fresh_bus_assistant.sessions:
        del fresh_bus_assistant.sessions[session_id]
    try:
        collection_name = f"api_data_{session_id}"
        if session_id:
            chroma_client.delete_collection(collection_name)
            print(f"Deleted collection {collection_name}")
    except Exception as e:
        print(f"Error deleting collection: {e}")
    response.delete_cookie(key="session_id")
    return {"status": "success", "message": "Session cleared"}

@app.get("/healthcheck")
def healthcheck():
    return {"status": "healthy"}

@app.get("/token-count")
def get_token_count(text: str):
    count = len(tokenizer.encode(text))
    return {"token_count": count}

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

#################################
# Conversation History Endpoints
#################################

@app.get("/conversations", response_model=List[ConversationSummary])
async def get_conversations(
    session_id: str = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get list of conversations, optionally filtered by session_id"""
    try:
        if session_id:
            conversations = conversation_manager.get_conversations_by_session(session_id)
        else:
            conversations = conversation_manager.get_all_conversations(limit, offset)
        return conversations
    except Exception as e:
        print(f"Error getting conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving conversations: {str(e)}")

@app.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str):
    """Get a specific conversation by ID"""
    try:
        conversation = conversation_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail=f"Conversation with ID {conversation_id} not found")
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a specific conversation by ID"""
    try:
        success = conversation_manager.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Conversation with ID {conversation_id} not found")
        return {"status": "success", "message": f"Conversation {conversation_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")

@app.delete("/conversations")
async def delete_all_session_conversations(session_id: str):
    """Delete all conversations for a session"""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id parameter is required")
    try:
        count = conversation_manager.delete_session_conversations(session_id)
        return {"status": "success", "message": f"Deleted {count} conversations for session {session_id}"}
    except Exception as e:
        print(f"Error deleting session conversations for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting conversations: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)