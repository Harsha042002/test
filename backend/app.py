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

# Import the custom AI provider modules
from ai_providers import AIProviderFactory, AIProvider
from ai_cost_tracker import AICostTracker
from config import Config

load_dotenv()

# Initialize Redis client using configuration
redis_client = redis.Redis(**Config.get_redis_config())

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    await fresh_bus_assistant.init_http_session()
    await fresh_bus_assistant.init_system_prompt()
    print(f"Using AI provider: {fresh_bus_assistant.ai_provider.provider_name} - Model: {fresh_bus_assistant.ai_provider.model}")
    yield
    await fresh_bus_assistant.cleanup()

# Initialize external clients and tokenizers
anthropic_client = Anthropic(
    api_key=Config.ANTHROPIC_API_KEY,
    timeout=45  # Set a longer timeout for all requests
)
voyage_api_key = Config.VOYAGE_API_KEY
chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
tokenizer = tiktoken.get_encoding("cl100k_base")  # Claude's encoding

# Request models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    stream: Optional[bool] = True
    gender: Optional[str] = None
    location: Optional[Dict[str, float]] = None
    provider: Optional[str] = None  # Added for model selection
    model: Optional[str] = None     # Added for model selection
    silent: Optional[bool] = False  # Added for silent model switching

class InitializeRequest(BaseModel):
    system_prompt_path: str = "./system_prompt/qa_prompt.txt"

class ModelSelectionRequest(BaseModel):
    provider: str
    model: str

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

# Use language codes from config
LANGUAGE_CODES = Config.LANGUAGE_CODES
LANGUAGE_CODES_REVERSE = Config.LANGUAGE_CODES_REVERSE

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
        query_lower = query.lower()
        
        try:
            # Check for simple queries to prioritize identity and greeting info
            if query_lower in ["hi", "hello", "hey"] or "name" in query_lower or "who" in query_lower:
                # First look for assistant identity sections
                identity_results = await self.query_collection(
                    collection_name, 
                    "assistant identity name", 
                    n_results=2
                )
                
                # If we have identity results, prioritize them
                if identity_results and identity_results.get("documents"):
                    docs = identity_results.get("documents")
                    if isinstance(docs[0], list):
                        docs = [doc for sublist in docs for doc in sublist]
                    
                    # Add standard prompt elements
                    standard_prompt = "You are a Fresh Bus travel assistant. Provide accurate bus information based on API data only."
                    return standard_prompt + "\n\n" + "\n\n".join(docs)
            
            # Regular query processing
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
        except Exception as e:
            print(f"Error getting system prompt: {e}")
            # Return a fallback prompt if there's an error
            return "You are a Fresh Bus travel assistant. Provide accurate bus information based on API data only."
    
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
        if api_data.get("tracking"):
            await self._store_tracking_data(collection_name, api_data["tracking"])
        
    async def _store_tracking_data(self, collection_name, tracking_data):
        """Store tracking data in vector database"""
        if not tracking_data:
            return
            
        tracking_type = tracking_data.get("type", "unknown")
        
        if tracking_type == "active_ticket":
            tracking_text = "BUS TRACKING INFORMATION (ACTIVE TICKET):\n"
            tracking_text += f"• Trip ID: {tracking_data.get('tripId', 'Unknown')}\n"
            tracking_text += f"• Journey: {tracking_data.get('from', 'Unknown')} to {tracking_data.get('to', 'Unknown')}\n"
            tracking_text += f"• Journey Date: {tracking_data.get('journeyDate', 'Unknown')}\n\n"
            
            # Add ETA data if available
            eta_data = tracking_data.get("eta", {})
            if eta_data:
                if "message" in eta_data:
                    tracking_text += f"• Status: {eta_data['message']}\n"
                else:
                    if "currentLocation" in eta_data:
                        tracking_text += f"• Current Bus Location: {eta_data['currentLocation']}\n"
                    if "estimatedArrival" in eta_data:
                        tracking_text += f"• Estimated Arrival: {eta_data['estimatedArrival']}\n"
                    if "delayMinutes" in eta_data:
                        tracking_text += f"• Delay: {eta_data['delayMinutes']} minutes\n"
            
        elif tracking_type == "completed_ticket":
            tracking_text = "BUS TRACKING INFORMATION (COMPLETED JOURNEY):\n"
            tracking_text += f"• Trip ID: {tracking_data.get('tripId', 'Unknown')}\n"
            tracking_text += f"• Completed Journey: {tracking_data.get('from', 'Unknown')} to {tracking_data.get('to', 'Unknown')}\n"
            tracking_text += f"• Journey Date: {tracking_data.get('journeyDate', 'Unknown')}\n"
            tracking_text += "• Status: This journey has been completed.\n\n"
            
            # Add NPS data if available
            nps_data = tracking_data.get("nps", {})
            if nps_data and nps_data.get("status") == "available" and nps_data.get("questions"):
                tracking_text += "FEEDBACK REQUESTED:\n"
                for i, question in enumerate(nps_data["questions"]):
                    if "questionText" in question:
                        tracking_text += f"• Question {i+1}: {question['questionText']}\n"
                        
        elif tracking_type == "future_ticket":
            tracking_text = "BUS TRACKING INFORMATION (FUTURE JOURNEY):\n"
            tracking_text += f"• Trip ID: {tracking_data.get('tripId', 'Unknown')}\n"
            tracking_text += f"• Future Journey: {tracking_data.get('from', 'Unknown')} to {tracking_data.get('to', 'Unknown')}\n"
            tracking_text += f"• Journey Date: {tracking_data.get('journeyDate', 'Unknown')}\n"
            tracking_text += "• Status: This journey is scheduled for the future. Tracking will be available closer to departure time.\n"
            
        else:
            tracking_text = f"BUS TRACKING INFORMATION ({tracking_type.upper()}):\n"
            tracking_text += f"• Message: {tracking_data.get('message', 'No tracking data available')}\n"
        
        # Store the tracking information in vector database
        await self.add_texts(
            collection_name,
            [tracking_text],
            [{"type": "tracking_data", "tracking_type": tracking_type}],
            [f"tracking_data_{tracking_type}_{collection_name}"]
        )
    
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
        self.load_system_prompt(Config.DEFAULT_SYSTEM_PROMPT_PATH)
        self.BASE_URL = Config.BASE_URL
        self.BASE_URL_CUSTOMER = Config.BASE_URL_CUSTOMER
        self.stations = Config.STATIONS
        self.seat_categories = Config.SEAT_CATEGORIES
        self.window_seats = Config.WINDOW_SEATS
        self.aisle_seats = Config.AISLE_SEATS
        self.sessions = {}
        self.http_session = None
        self.embeddings_client = VoyageEmbeddings(voyage_api_key)
        self.vector_db = VectorDBManager(self.embeddings_client)
        self.system_prompt_initialized = False
        
        # Cache for user tickets
        self.user_tickets_cache = {}
        self.user_tickets_cache_expiry = {}
        
        # Initialize AI provider from config
        provider_name = Config.DEFAULT_AI_PROVIDER
        provider_model = Config.DEFAULT_AI_MODEL
        
        # Clean up provider name by removing any comments and whitespace
        if '#' in provider_name:
            provider_name = provider_name.split('#')[0].strip()
            print(f"WARNING: Cleaned up provider name to: {provider_name}")
        
        # Ensure provider name is valid
        valid_providers = ["gemini", "claude"]
        if provider_name.lower() not in valid_providers:
            print(f"WARNING: Unknown provider '{provider_name}', falling back to 'gemini'")
            provider_name = "gemini"
            
        try:
            self.ai_provider = AIProviderFactory.create_provider(
                provider_name=provider_name,
                model=provider_model
            )
        except ValueError as e:
            print(f"Error initializing provider {provider_name}: {e}")
            print("Falling back to Gemini provider")
            self.ai_provider = AIProviderFactory.create_provider(
                provider_name="gemini",
                model="gemini-2.0-flash"
            )
        
        # Initialize cost tracker
        self.cost_tracker = AICostTracker()

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
            # Load identity info from a file if it exists, otherwise use default
            identity_info_path = os.path.join(os.path.dirname(Config.DEFAULT_SYSTEM_PROMPT_PATH), "identity_info.txt")
            identity_info = ""
            
            try:
                if os.path.exists(identity_info_path):
                    with open(identity_info_path, 'r', encoding='utf-8') as f:
                        identity_info = f.read()
                        print(f"Loaded identity info from {identity_info_path}")
                else:
                    # Default identity info
                    identity_info = """
                    
                    ASSISTANT IDENTITY:
                    - Your name is Ṧ.AI (pronounced 'Sai')
                    - You are the Fresh Bus AI travel assistant
                    - When asked about your name or identity, always identify yourself as Ṧ.AI
                    - When asked simple greeting questions like "hi", "hello", respond with a friendly greeting
                    
                    LANGUAGE CAPABILITIES:
                    - You can respond fluently in multiple Indian languages including Telugu, Hindi, Tamil, Kannada, and Malayalam
                    - When a user asks a question in one of these languages, respond in the same language
                    - For transliteration requests, you can convert between scripts while maintaining the same language (e.g., Telugu written in Latin script or 'Tenglish')
                    - Never claim you cannot speak or understand these languages
                    
                    HANDLING NON-BUS QUERIES:
                    - For general questions not related to bus booking, provide brief, helpful responses
                    - For identity questions like "what's your name", always state that you are Ṧ.AI
                    - For greetings like "hi", "hello", respond with a friendly greeting
                    - Only direct users back to bus-related topics for completely off-topic conversations
                    
                    OUTPUT FORMAT FOR BUS LISTINGS:
                    When providing bus information, use this JSON format inside triple backticks:
                    ```json
                    {
                    "trips": [
                        {
                        "busNumber": "123",
                        "price": "499",
                        "seats": "84",
                        "rating": "4.6",
                        "from": "Hyderabad",
                        "to": "Guntur",
                        "boardingPoint": "L B Nagar Metro Station",
                        "droppingPoint": "NTR CIRCLE RTC Bus Stand",
                        "departureTime": "10:00 PM",
                        "arrivalTime": "5:30 AM",
                        "duration": "7h 30m",
                        "tripId": "12345",
                        "busType": "AC Sleeper",
                        "recommendations": {
                            "reasonable": {
                            "window": {"seatNumber": "1", "price": "449"},
                            "aisle": {"seatNumber": "2", "price": "449"}
                            }
                        }
                        }
                    ]
                    }
                    ```
                    
                    When the user wants this information in another language, first provide the JSON, then translate it to the requested language.
                    """
            except Exception as e:
                print(f"Error loading identity info: {e}")
                # Use a minimal identity info if there's an error
                identity_info = """
                ASSISTANT IDENTITY:
                - Your name is Ṧ.AI (pronounced 'Sai')
                - You are the Fresh Bus AI travel assistant
                """
            
            self.system_prompt += identity_info
            
            # Initialize VectorDB with the enhanced system prompt
            await self.embeddings_client.init_session()
            await self.vector_db.store_system_prompt(self.system_prompt)
            self.system_prompt_initialized = True
            print("Enhanced system prompt stored in vector DB")
    
    async def init_http_session(self):
        if self.http_session is None:
            timeout_config = Config.get_http_timeout_config()
            timeout = aiohttp.ClientTimeout(
                total=timeout_config["total"],
                connect=timeout_config["connect"],
                sock_connect=timeout_config["sock_connect"],
                sock_read=timeout_config["sock_read"]
            )
            self.http_session = aiohttp.ClientSession(timeout=timeout)
            print("HTTP session initialized with timeout settings")
            await self.embeddings_client.init_session()
            
            # Initialize AI provider
            await self.ai_provider.initialize()
            print(f"Initialized {self.ai_provider.provider_name} AI provider with model {self.ai_provider.model}")
    
    async def cleanup(self):
        if self.http_session:
            await self.http_session.close()
            print("HTTP session closed")
        await self.embeddings_client.close_session()
        await self.ai_provider.cleanup()
        print(f"Cleaned up {self.ai_provider.provider_name} AI provider")
    
    # Improved language detection
    def detect_language(self, text):
        """Detect language of user input with expanded support for Indian languages"""
        # Check for Telugu script
        if re.search(r'[\u0C00-\u0C7F]', text):
            return "telugu"
        
        # Check for Hindi script
        if re.search(r'[\u0900-\u097F]', text):
            return "hindi"
        
        # Check for Tamil script
        if re.search(r'[\u0B80-\u0BFF]', text):
            return "tamil"
        
        # Check for Kannada script
        if re.search(r'[\u0C80-\u0CFF]', text):
            return "kannada"
        
        # Check for Malayalam script
        if re.search(r'[\u0D00-\u0D7F]', text):
            return "malayalam"
        
        # Check for transliterated Telugu (Tenglish)
        tenglish_words = ["meeku", "naaku", "undi", "ledu", "kavalante", "cheyandi", 
                        "telugu", "nenu", "nenu", "miru", "meeru", "enti", "ala", "ela", 
                        "ooru", "peru", "vastunnanu", "vastunna", "vachanu"]
        
        if any(word in text.lower() for word in tenglish_words):
            return "tenglish"
        
        # Check for transliterated Hindi (Hinglish)
        hinglish_words = ["kya", "hai", "nahi", "aap", "karenge", "chahiye", "dijiye", 
                        "kaise", "kyun", "mujhe", "tumhe", "hum", "tum", "kaun"]
        
        if any(word in text.lower() for word in hinglish_words):
            return "hinglish"
        
        # Default to English
        return "english"
    
    # Method to detect bus tracking queries
    def is_bus_tracking_query(self, query):
        """Detect if a query is asking about bus tracking or ETA"""
        tracking_phrases = [
            "where is my bus", "track my bus", "bus location", "track bus",
            "bus tracking", "where is the bus", "bus eta", "arrival time",
            "when will bus arrive", "trip id", "tripid", "trip status",
            "bus status", "bus position", "live tracking", "live location",
            "bus reached", "bus arrived", "bus coming", "ticket status"
        ]
        
        query_lower = query.lower()
        return any(phrase in query_lower for phrase in tracking_phrases)
    
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
    
    # Updated method to handle user tickets with caching
    async def fetch_user_tickets(self, auth_token, force_refresh=False):
        """Fetch user's tickets and cache them for a short period"""
        if not auth_token:
            return None
            
        # Check cache first (if not forced to refresh)
        cache_key = auth_token[:10]  # Use part of token as key
        current_time = time.time()
        
        if not force_refresh and cache_key in self.user_tickets_cache:
            # Use cache if it's not expired
            if current_time - self.user_tickets_cache_expiry.get(cache_key, 0) < Config.CACHE_EXPIRY_SECONDS:
                return self.user_tickets_cache[cache_key]
        
        if not self.http_session:
            await self.init_http_session()
        
        try:
            # Call the tickets API endpoint
            url = f"{self.BASE_URL_CUSTOMER}/tickets"
            print(f"Fetching user tickets: {url}")
            
            headers = {
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json"
            }
            
            async with self.http_session.get(url, headers=headers) as response:
                if response.status == 200:
                    tickets_data = await response.json()
                    
                    # Cache the result
                    self.user_tickets_cache[cache_key] = tickets_data
                    self.user_tickets_cache_expiry[cache_key] = current_time
                    
                    print(f"Successfully fetched user tickets: {len(tickets_data)} tickets")
                    return tickets_data
                else:
                    print(f"Error fetching user tickets: {response.status}")
                    error_text = await response.text()
                    print(f"Error response: {error_text}")
                    return None
        except Exception as e:
            print(f"Exception fetching user tickets: {e}")
            return None
    
    # Process tickets to find active, completed, and future tickets
    async def process_user_tickets(self, tickets):
        """Process user tickets to categorize them and get ETA data for active ones"""
        if not tickets:
            return {
                "active": [],
                "completed": [],
                "future": []
            }
            
        now = datetime.now()
        active_tickets = []
        completed_tickets = []
        future_tickets = []
        
        for ticket in tickets:
            # Parse journey date
            journey_date = None
            dropping_time = None
            
            try:
                if "journeyDate" in ticket:
                    journey_date_str = ticket["journeyDate"]
                    
                    # Handle different date formats
                    if "T" in journey_date_str:
                        journey_date = datetime.fromisoformat(journey_date_str.replace('Z', '+00:00'))
                    else:
                        # If only date is provided, add a default time
                        journey_date = datetime.fromisoformat(f"{journey_date_str}T00:00:00+00:00")
                
                if "droppingTime" in ticket:
                    dropping_time_str = ticket["droppingTime"]
                    if "T" in dropping_time_str:
                        dropping_time = datetime.fromisoformat(dropping_time_str.replace('Z', '+00:00'))
                    
            except Exception as e:
                print(f"Error parsing journey date for ticket: {e}")
                journey_date = None
            
            # Only process tickets with valid dates
            if journey_date:
                # Convert to local timezone for comparison
                journey_local = journey_date.astimezone(timezone(timedelta(hours=5, minutes=30)))
                now_local = now.astimezone(timezone(timedelta(hours=5, minutes=30)))
                
                # Calculate time difference in minutes
                time_diff_minutes = (journey_local - now_local).total_seconds() / 60
                
                # Fetch ETA data for tickets that might be active
                if -24*60 < time_diff_minutes < 30:
                    # This could be an active ticket (within past 24h or next 30 min)
                    if "tripId" in ticket:
                        try:
                            # Get ETA data
                            eta_data = await self.fetch_eta_data(ticket["tripId"])
                            ticket["eta_data"] = eta_data
                            
                            # If trip has ended based on ETA data
                            if eta_data and eta_data.get("message") == "This trip has ended":
                                completed_tickets.append(ticket)
                            else:
                                active_tickets.append(ticket)
                        except Exception as e:
                            print(f"Error fetching ETA for ticket {ticket.get('tripId')}: {e}")
                            active_tickets.append(ticket)
                    else:
                        active_tickets.append(ticket)
                # Future tickets
                elif time_diff_minutes >= 30:
                    future_tickets.append(ticket)
                # Check if it's completed based on dropping time
                elif dropping_time and dropping_time < now:
                    # Get NPS data for completed tickets
                    if "tripId" in ticket:
                        try:
                            nps_data = await self.fetch_nps_data(ticket["tripId"])
                            ticket["nps_data"] = nps_data
                        except Exception as e:
                            print(f"Error fetching NPS for ticket {ticket.get('tripId')}: {e}")
                    
                    completed_tickets.append(ticket)
                else:
                    # Default to completed if in the past
                    completed_tickets.append(ticket)
        
        # Sort by journey date (newest first)
        active_tickets.sort(key=lambda x: x.get("journeyDate", ""), reverse=True)
        future_tickets.sort(key=lambda x: x.get("journeyDate", ""))
        completed_tickets.sort(key=lambda x: x.get("journeyDate", ""), reverse=True)
        
        return {
            "active": active_tickets,
            "completed": completed_tickets,
            "future": future_tickets
        }
    
    # Method to fetch ETA data
    async def fetch_eta_data(self, trip_id):
        """Fetch ETA data for a trip"""
        if not self.http_session:
            await self.init_http_session()
            
        try:
            # Use the ETA API endpoint
            url = f"https://api.freshbus.com/eta-data?id={trip_id}"
            print(f"Fetching ETA data: {url}")
            
            async with self.http_session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Error fetching ETA data: {response.status}")
                    error_text = await response.text()
                    print(f"Error response: {error_text}")
                    return {"message": "Could not retrieve tracking information"}
        except Exception as e:
            print(f"Exception fetching ETA data: {e}")
            return {"message": f"Error retrieving tracking information: {str(e)}"}
    
    # Method to fetch NPS (feedback) data
    async def fetch_nps_data(self, trip_id, auth_token=None):
        """Fetch NPS/feedback data for a completed trip"""
        if not auth_token:
            return {"message": "Authentication required to fetch feedback data"}
            
        if not self.http_session:
            await self.init_http_session()
            
        try:
            # Use the feedback questions API
            url = f"{self.BASE_URL_CUSTOMER}/tickets/feedbackQuestions"
            print(f"Fetching feedback data: {url}")
            
            headers = {
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json"
            }
            
            async with self.http_session.get(url, headers=headers) as response:
                if response.status == 200:
                    all_feedback = await response.json()
                    
                    # Find feedback for this specific trip
                    if isinstance(all_feedback, list):
                        trip_feedback = next((fb for fb in all_feedback if str(fb.get("tripId")) == str(trip_id)), None)
                        if trip_feedback:
                            return trip_feedback
                    
                    # Return all feedback data if we couldn't find trip-specific feedback
                    return all_feedback
                else:
                    print(f"Error fetching feedback data: {response.status}")
                    error_text = await response.text()
                    print(f"Error response: {error_text}")
                    return {"message": "Could not retrieve feedback information"}
        except Exception as e:
            print(f"Exception fetching feedback data: {e}")
            return {"message": f"Error retrieving feedback information: {str(e)}"}
    
    # Format ETA data as JSON for consistent frontend display
    def format_eta_data_as_json(self, eta_data, trip_id):
        """Format ETA data as JSON for frontend display"""
        eta_json = {
            "tripId": trip_id,
            "status": "unknown"
        }
        
        if not eta_data:
            eta_json["status"] = "unavailable"
            eta_json["message"] = "Tracking information is not available"
            return eta_json
            
        if "message" in eta_data and eta_data["message"] == "This trip has ended":
            eta_json["status"] = "completed"
            eta_json["message"] = "Trip has ended. The bus has completed its journey."
        elif "message" in eta_data:
            eta_json["status"] = "error"
            eta_json["message"] = eta_data["message"]
        else:
            eta_json["status"] = "active"
            
            if "currentLocation" in eta_data:
                eta_json["currentLocation"] = eta_data["currentLocation"]
                
            if "estimatedArrival" in eta_data:
                eta_json["estimatedArrival"] = eta_data["estimatedArrival"]
                
            if "delayMinutes" in eta_data:
                eta_json["delayMinutes"] = eta_data["delayMinutes"]
                
            if "lastUpdated" in eta_data:
                eta_json["lastUpdated"] = eta_data["lastUpdated"]
        
        return eta_json
    
    # Format NPS data as JSON
    def format_nps_data_as_json(self, nps_data, trip_id):
        """Format NPS/feedback data as JSON for frontend display"""
        nps_json = {
            "tripId": trip_id,
            "status": "unknown"
        }
        
        if not nps_data:
            nps_json["status"] = "unavailable"
            nps_json["message"] = "Feedback information is not available"
            return nps_json
            
        if "message" in nps_data:
            nps_json["status"] = "error"
            nps_json["message"] = nps_data["message"]
        elif isinstance(nps_data, list) and len(nps_data) > 0:
            nps_json["status"] = "available"
            
            # Find feedback for this specific trip
            trip_feedback = next((fb for fb in nps_data if str(fb.get("tripId")) == str(trip_id)), None)
            
            if trip_feedback and "questions" in trip_feedback:
                nps_json["questions"] = trip_feedback["questions"]
            elif len(nps_data) > 0 and "questions" in nps_data[0]:
                # Use first feedback item as fallback
                nps_json["questions"] = nps_data[0]["questions"]
        elif "questions" in nps_data:
            nps_json["status"] = "available"
            nps_json["questions"] = nps_data["questions"]
        
        return nps_json
        
    async def get_active_tickets_and_status(self, auth_token):
        """Get user's active tickets and their status (future, ongoing, or completed)"""
        if not self.http_session:
            await self.init_http_session()
        
        try:
            # Get all tickets
            tickets = await self.fetch_user_tickets(auth_token)
            if not tickets:
                return None
                
            # Process into categories
            return await self.process_user_tickets(tickets)
            
        except Exception as e:
            print(f"Exception in get_active_tickets_and_status: {e}")
            return None
    
    def trim_text(self, text, token_limit=None):
        if token_limit is None:
            token_limit = Config.DEFAULT_TOKEN_LIMIT
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
        for idx, trip in enumerate(trips):
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
            
            # Format each bus as a separate section with proper markdown
            formatted_bus = f"> *🚌 {source} to {destination} | {bus_data.get('departure_time', 'Unknown')} - {bus_data.get('arrival_time', 'Unknown')} | {bus_data.get('vehicletype', 'AC Seater')} | {duration}*\n"
            formatted_bus += f"> Price: ₹{bus_data.get('fare', 'Unknown')} | {bus_data.get('seats', 'Unknown')} seats | Rating: {bus_data.get('rating', 'Unknown')}/5\n"
            formatted_bus += f"> Boarding: {bus_data.get('boarding', 'Unknown')}\n"
            formatted_bus += f"> Dropping: {bus_data.get('dropping', 'Unknown')}\n"
            formatted_bus += f"> Recommended seats:\n"
            formatted_bus += f"> • **Reasonable**: Window {window_seat} (₹{window_price}), Aisle {aisle_seat} (₹{aisle_price})"
            
            formatted_buses.append(formatted_bus)
        
        # Join with double newlines to ensure proper separation between buses
        return header + "\n\n".join(formatted_buses)

    # Generate JSON response for bus data
    def generate_json_bus_response(self, api_data, context):
        """Generate JSON bus response from API data"""
        trips = api_data.get("trips", [])
        
        if not trips:
            return {"error": "No buses found matching your criteria."}
        
        # Create JSON output
        bus_json = {"trips": []}
        
        for trip in trips:
            # Format times
            departure_time = "Unknown"
            arrival_time = "Unknown"
            duration = ""
            
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
            
            # Get source and destination
            source = context.get('user_requested_source') or trip.get('source', 'Unknown')
            destination = context.get('user_requested_destination') or trip.get('destination', 'Unknown')
            
            # Get recommendations
            window_seat = "N/A"
            window_price = "N/A"
            aisle_seat = "N/A"
            aisle_price = "N/A"
            
            if api_data.get("recommendations") and api_data["recommendations"].get("Reasonable"):
                if api_data["recommendations"]["Reasonable"].get("window") and len(api_data["recommendations"]["Reasonable"]["window"]) > 0:
                    window_seat = api_data["recommendations"]["Reasonable"]["window"][0].get("number", "N/A")
                    window_price = api_data["recommendations"]["Reasonable"]["window"][0].get("price", "N/A")
                
                if api_data["recommendations"]["Reasonable"].get("aisle") and len(api_data["recommendations"]["Reasonable"]["aisle"]) > 0:
                    aisle_seat = api_data["recommendations"]["Reasonable"]["aisle"][0].get("number", "N/A")
                    aisle_price = api_data["recommendations"]["Reasonable"]["aisle"][0].get("price", "N/A")
            
            # Create bus JSON object
            bus_data = {
                "busNumber": trip.get('servicenumber', 'Unknown'),
                "price": str(trip.get('fare', 'Unknown')),
                "seats": str(trip.get('availableseats', 'N/A')),
                "rating": str(trip.get('redbusrating', 'N/A')),
                "from": source,
                "to": destination,
                "boardingPoint": trip.get('boardingpointname', 'Unknown'),
                "droppingPoint": trip.get('droppingpointname', 'Unknown'),
                "departureTime": departure_time,
                "arrivalTime": arrival_time,
                "duration": duration,
                "tripId": str(trip.get('tripid', 'Unknown')),
                "busType": trip.get('vehicletype', 'Standard'),
                "recommendations": {
                    "reasonable": {
                        "window": {"seatNumber": str(window_seat), "price": str(window_price)},
                        "aisle": {"seatNumber": str(aisle_seat), "price": str(aisle_price)}
                    }
                }
            }
            
            bus_json["trips"].append(bus_data)
        
        return bus_json

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
        
        # Make sure journey_date is not None
        if journey_date is None:
            journey_date = datetime.now().strftime("%Y-%m-%d")
            print(f"Using default date: {journey_date}")
        
        url = f"{self.BASE_URL}/trips?journey_date={journey_date}&source_id={source_id}&destination_id={destination_id}"
        print(f"Fetching trips: {url}")
        
        try:
            async with self.http_session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data:
                        print(f"Successfully fetched {len(data)} trips")
                        return self.deduplicate_trips(data)
                    else:
                        print("API returned empty response")
                        return []
                else:
                    print(f"Error fetching trips: {response.status}")
                    # Properly handle API errors without inventing data
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
        
        # Handle tracking data
        if api_data.get("tracking"):
            tracking_data = api_data["tracking"]
            tracking_type = tracking_data.get("type", "unknown")
            
            if tracking_type == "active_ticket":
                context_parts.append("\nBUS TRACKING INFORMATION (ACTIVE TICKET):")
                context_parts.append(f"• Trip ID: {tracking_data.get('tripId', 'Unknown')}")
                context_parts.append(f"• Journey: {tracking_data.get('from', 'Unknown')} to {tracking_data.get('to', 'Unknown')}")
                context_parts.append(f"• Journey Date: {tracking_data.get('journeyDate', 'Unknown')}")
                
                # Add ETA data if available
                eta_data = tracking_data.get("eta", {})
                if eta_data:
                    if "message" in eta_data:
                        context_parts.append(f"• Status: {eta_data['message']}")
                    else:
                        if "currentLocation" in eta_data:
                            context_parts.append(f"• Current Bus Location: {eta_data['currentLocation']}")
                        if "estimatedArrival" in eta_data:
                            context_parts.append(f"• Estimated Arrival: {eta_data['estimatedArrival']}")
                        if "delayMinutes" in eta_data:
                            context_parts.append(f"• Delay: {eta_data['delayMinutes']} minutes")
            
            elif tracking_type == "completed_ticket":
                context_parts.append("\nBUS TRACKING INFORMATION (COMPLETED JOURNEY):")
                context_parts.append(f"• Trip ID: {tracking_data.get('tripId', 'Unknown')}")
                context_parts.append(f"• Completed Journey: {tracking_data.get('from', 'Unknown')} to {tracking_data.get('to', 'Unknown')}")
                context_parts.append(f"• Journey Date: {tracking_data.get('journeyDate', 'Unknown')}")
                context_parts.append("• Status: This journey has been completed.")
                
                # Add NPS data if available
                nps_data = tracking_data.get("nps", {})
                if nps_data and nps_data.get("status") == "available" and nps_data.get("questions"):
                    context_parts.append("\nFEEDBACK REQUESTED:")
                    for i, question in enumerate(nps_data["questions"]):
                        if "questionText" in question:
                            context_parts.append(f"• Question {i+1}: {question['questionText']}")
            
            elif tracking_type == "future_ticket":
                context_parts.append("\nBUS TRACKING INFORMATION (FUTURE JOURNEY):")
                context_parts.append(f"• Trip ID: {tracking_data.get('tripId', 'Unknown')}")
                context_parts.append(f"• Future Journey: {tracking_data.get('from', 'Unknown')} to {tracking_data.get('to', 'Unknown')}")
                context_parts.append(f"• Journey Date: {tracking_data.get('journeyDate', 'Unknown')}")
                context_parts.append("• Status: This journey is scheduled for the future. Tracking will be available closer to departure time.")
                
            else:
                context_parts.append(f"\nBUS TRACKING INFORMATION ({tracking_type.upper()}):")
                context_parts.append(f"• Message: {tracking_data.get('message', 'No tracking data available')}")
            
            # Return tracking information
            return "\n".join(context_parts)
        
        # Standard bus search results
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
        
        # Continue with standard context building for other API data
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
            
            # Add ETA data if available
            if "eta_data" in ticket:
                eta_data = ticket["eta_data"]
                
                if "message" in eta_data and eta_data["message"] == "This trip has ended":
                    context_parts.append("Status: This trip has already ended.")
                elif "message" in eta_data:
                    context_parts.append(f"Status: {eta_data['message']}")
                else:
                    if "currentLocation" in eta_data:
                        context_parts.append(f"Current Location: {eta_data['currentLocation']}")
                    if "estimatedArrival" in eta_data:
                        context_parts.append(f"Estimated Arrival: {eta_data['estimatedArrival']}")
                    if "delayMinutes" in eta_data:
                        context_parts.append(f"Delay: {eta_data['delayMinutes']} minutes")
        
        if api_data.get("matched_completed_ticket"):
            ticket = api_data["matched_completed_ticket"]
            context_parts.append("\nCOMPLETED JOURNEY:")
            context_parts.append(f"Journey from {ticket.get('source', 'Unknown')} to {ticket.get('destination', 'Unknown')}")
            context_parts.append(f"Date: {ticket.get('journeyDate', 'Unknown')}")
            context_parts.append(f"Trip ID: {ticket.get('tripId', 'Unknown')}")
            context_parts.append("Status: This journey has been completed.")
            
            # Add NPS/feedback data if available
            if "nps_data" in ticket:
                nps_data = ticket["nps_data"]
                
                context_parts.append("\nFEEDBACK QUESTIONS:")
                if isinstance(nps_data, list) and len(nps_data) > 0:
                    feedback = nps_data[0]  # Use first feedback item
                    if "questions" in feedback:
                        for question in feedback["questions"]:
                            context_parts.append(f"• {question.get('questionText', 'Rate your journey')}")
                elif nps_data.get("questions"):
                    for question in nps_data["questions"]:
                        context_parts.append(f"• {question.get('questionText', 'Rate your journey')}")
                else:
                    context_parts.append(f"• {nps_data.get('message', 'Please rate your journey')}")
        
        if api_data.get("matched_upcoming_ticket"):
            ticket = api_data["matched_upcoming_ticket"]
            context_parts.append("\nUPCOMING JOURNEY:")
            context_parts.append(f"Journey from {ticket.get('source', 'Unknown')} to {ticket.get('destination', 'Unknown')}")
            context_parts.append(f"Date: {ticket.get('journeyDate', 'Unknown')}")
            context_parts.append(f"Trip ID: {ticket.get('tripId', 'Unknown')}")
            context_parts.append("Status: This journey is scheduled for the future.")
        
        # Add user profile information if available
        if api_data.get("user_profile"):
            profile = api_data["user_profile"]
            context_parts.append("\nUSER PROFILE INFORMATION:")
            
            # Add basic info
            for field in ['name', 'email', 'mobile', 'gender']:
                if profile.get(field):
                    context_parts.append(f"• {field.capitalize()}: {profile[field]}")
            
            # Add address if available
            if profile.get('address'):
                context_parts.append(f"• Address: {profile['address']}")
                
            # Add loyalty coins if available
            if profile.get('availableCoins'):
                context_parts.append(f"• Available Coins: {profile['availableCoins']}")
                
            # Add preferred language if available
            if profile.get('preferredLanguage'):
                context_parts.append(f"• Preferred Language: {profile['preferredLanguage']}")
        
        # Build final context string
        return "\n".join(context_parts)
    
    def _generate_fallback_response(self, api_data, query, context):
        """Generate a fallback response when the LLM is not available or fails"""
        fallback_parts = []
        query_lower = query.lower()
        
        # Handle greetings
        if query_lower in ["hi", "hello", "hey", "hola", "namaste"]:
            return "Hello! I'm Ṧ.AI, your Fresh Bus travel assistant. How can I help you with your bus travel today?"
        
        # Handle identity questions
        if "name" in query_lower and ("your" in query_lower or "you" in query_lower):
            return "My name is Ṧ.AI (pronounced 'Sai'). I'm your Fresh Bus travel assistant. How can I help you with your bus travel today?"
        
        # Handle what/who are you questions
        if ("what are you" in query_lower or "who are you" in query_lower):
            return "I'm Ṧ.AI, the Fresh Bus travel assistant. I can help you find buses, check schedules, book tickets, and track your journeys. How can I assist you today?"
        
        # Handle the case when no trips are found
        if api_data and api_data.get("no_trips_info"):
            info = api_data["no_trips_info"]
            # Check if travel_date is not None before calling fromisoformat
            if "date" in info and info["date"]:
                return info["message"]
            else:
                # Use a default message if date is missing
                return "I couldn't find any trips matching your criteria."
        
        # Original fallback logic for bus-related queries
        fallback_parts.append("Hello, I'm your Fresh Bus travel assistant.")
        
        # Check for tracking requests first
        is_tracking_request = self.is_bus_tracking_query(query)
        if is_tracking_request:
            if api_data.get("tracking"):
                tracking_data = api_data["tracking"]
                tracking_type = tracking_data.get("type", "unknown")
                
                if tracking_type == "active_ticket":
                    fallback_parts.append("\n🚍 Bus Tracking Information:")
                    fallback_parts.append(f"Your bus from {tracking_data.get('from', 'Unknown')} to {tracking_data.get('to', 'Unknown')} is on the move.")
                    
                    # Add ETA data if available
                    eta_data = tracking_data.get("eta", {})
                    if eta_data:
                        if "currentLocation" in eta_data:
                            fallback_parts.append(f"Current Location: {eta_data['currentLocation']}")
                        if "estimatedArrival" in eta_data:
                            fallback_parts.append(f"Estimated Arrival: {eta_data['estimatedArrival']}")
                        if "delayMinutes" in eta_data and int(eta_data['delayMinutes']) > 0:
                            fallback_parts.append(f"The bus is delayed by {eta_data['delayMinutes']} minutes.")
                        
                elif tracking_type == "completed_ticket":
                    fallback_parts.append("\n✓ This trip has already ended.")
                    fallback_parts.append("The bus has completed its journey.")
                    
                    # Add NPS data if available
                    nps_data = tracking_data.get("nps", {})
                    if nps_data and nps_data.get("status") == "available":
                        fallback_parts.append("\nPlease consider providing feedback on your journey through the Fresh Bus app.")
                        
                elif tracking_type == "future_ticket":
                    fallback_parts.append("\n⏰ Your upcoming trip is scheduled for the future.")
                    fallback_parts.append(f"Journey Date: {tracking_data.get('journeyDate', 'Unknown')}")
                    fallback_parts.append("Tracking will be available closer to departure time.")
                    
                else:
                    fallback_parts.append("\nI couldn't find any active bus trips to track.")
                    fallback_parts.append("Please ensure you're logged in and have an active booking.")
            else:
                fallback_parts.append("\nI couldn't find any active bus trips to track.")
                fallback_parts.append("Please ensure you're logged in and have an active booking.")
                
            return " ".join(fallback_parts)
        
        # Handle trip ended status
        if api_data.get("eta_data") and api_data["eta_data"].get("message") == "This trip has ended":
            fallback_parts.append("\n✓ This trip has already ended.")
            fallback_parts.append("The bus has completed its journey.")
            
            # Add feedback info if available
            if api_data.get("feedback_data"):
                fallback_parts.append("\nPlease consider providing feedback on your journey through the Fresh Bus app.")
                
            return " ".join(fallback_parts)
        
        # Handle direction reversed case
        if api_data.get("direction_reversed") and api_data.get("trips"):
            source = context.get('user_requested_source', '').capitalize()
            destination = context.get('user_requested_destination', '').capitalize()
            
            fallback_parts.append(f"\nI couldn't find buses from {source} to {destination}, but I found buses going in the reverse direction (from {destination} to {source}).")
            fallback_parts.append("Here are the available options:")
            
            trips = api_data.get("trips", [])
            for i, trip in enumerate(trips[:3]):  # Limit to first 3 buses
                fallback_parts.append("\n")
                
                # Format basic trip info (adjust to reflect the user's requested direction)
                boarding_time = "Unknown"
                if "boardingtime" in trip:
                    try:
                        dt = datetime.fromisoformat(trip["boardingtime"].replace('Z', '+00:00'))
                        ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
                        boarding_time = ist_dt.strftime("%I:%M %p")
                    except:
                        boarding_time = "Unknown"
                
                dropping_time = "Unknown"
                if "droppingtime" in trip:
                    try:
                        dt = datetime.fromisoformat(trip["droppingtime"].replace('Z', '+00:00'))
                        ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
                        dropping_time = ist_dt.strftime("%I:%M %p")
                    except:
                        dropping_time = "Unknown"
                
                price = trip.get('fare', 'Unknown')
                available_seats = trip.get('availableseats', 'Unknown')
                
                fallback_parts.append(f"Bus {i+1}: {boarding_time} - {dropping_time} | ₹{price} | {available_seats} seats available")
                # Swap points to match user's requested direction
                fallback_parts.append(f"Boarding: {trip.get('droppingpointname', 'Unknown')}")  # Show dropping as boarding
                fallback_parts.append(f"Dropping: {trip.get('boardingpointname', 'Unknown')}")  # Show boarding as dropping
            
            if len(trips) > 3:
                fallback_parts.append(f"\n...and {len(trips) - 3} more buses available.")
                
            fallback_parts.append("\nWould you like to proceed with one of these options?")
            
            return " ".join(fallback_parts)
        
        # Handle bus search results (normal case)
        if api_data.get("trips"):
            trips = api_data.get("trips", [])
            
            # Get source and destination
            source = context.get('user_requested_source') or trips[0].get('source', 'Unknown')
            destination = context.get('user_requested_destination') or trips[0].get('destination', 'Unknown')
            
            fallback_parts.append(f"\nI found {len(trips)} bus{'es' if len(trips) > 1 else ''} from {source} to {destination}.")
            
            for i, trip in enumerate(trips[:3]):  # Limit to first 3 buses
                fallback_parts.append("\n")
                
                # Format basic trip info
                boarding_time = "Unknown"
                if "boardingtime" in trip:
                    try:
                        dt = datetime.fromisoformat(trip["boardingtime"].replace('Z', '+00:00'))
                        ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
                        boarding_time = ist_dt.strftime("%I:%M %p")
                    except:
                        boarding_time = "Unknown"
                
                dropping_time = "Unknown"
                if "droppingtime" in trip:
                    try:
                        dt = datetime.fromisoformat(trip["droppingtime"].replace('Z', '+00:00'))
                        ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
                        dropping_time = ist_dt.strftime("%I:%M %p")
                    except:
                        dropping_time = "Unknown"
                
                price = trip.get('fare', 'Unknown')
                available_seats = trip.get('availableseats', 'Unknown')
                
                fallback_parts.append(f"Bus {i+1}: {boarding_time} - {dropping_time} | ₹{price} | {available_seats} seats available")
                fallback_parts.append(f"Boarding: {trip.get('boardingpointname', 'Unknown')}")
                fallback_parts.append(f"Dropping: {trip.get('droppingpointname', 'Unknown')}")
            
            if len(trips) > 3:
                fallback_parts.append(f"\n...and {len(trips) - 3} more buses available.")
                
            fallback_parts.append("\nPlease ask for more details about any specific bus or provide more information to refine your search.")
            
            return " ".join(fallback_parts)
        
        # Generic fallback response
        fallback_parts.append("I'm currently having trouble processing your request. Here's what I understand:")
        
        if context.get('user_requested_source') and context.get('user_requested_destination'):
            fallback_parts.append(f"\nYou want to travel from {context['user_requested_source']} to {context['user_requested_destination']}.")
        
        if context.get('last_date'):
            fallback_parts.append(f"Date of travel: {context['last_date']}")
            
        fallback_parts.append("\nPlease try again with a simple request like 'Show buses from [source] to [destination]' or 'Book a ticket to [destination]'.")
        
        return " ".join(fallback_parts)
        
        # Handle trip ended status
        if api_data.get("eta_data") and api_data["eta_data"].get("message") == "This trip has ended":
            fallback_parts.append("\n✓ This trip has already ended.")
            fallback_parts.append("The bus has completed its journey.")
            
            # Add feedback info if available
            if api_data.get("feedback_data"):
                fallback_parts.append("\nPlease consider providing feedback on your journey through the Fresh Bus app.")
                
            return " ".join(fallback_parts)
        
        # Handle bus search results
        if api_data.get("trips"):
            trips = api_data.get("trips", [])
            
            # Get source and destination
            source = context.get('user_requested_source') or trips[0].get('source', 'Unknown')
            destination = context.get('user_requested_destination') or trips[0].get('destination', 'Unknown')
            
            fallback_parts.append(f"\nI found {len(trips)} bus{'es' if len(trips) > 1 else ''} from {source} to {destination}.")
            
            for i, trip in enumerate(trips[:3]):  # Limit to first 3 buses
                fallback_parts.append("\n")
                
                # Format basic trip info
                boarding_time = "Unknown"
                if "boardingtime" in trip:
                    try:
                        dt = datetime.fromisoformat(trip["boardingtime"].replace('Z', '+00:00'))
                        ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
                        boarding_time = ist_dt.strftime("%I:%M %p")
                    except:
                        boarding_time = "Unknown"
                
                dropping_time = "Unknown"
                if "droppingtime" in trip:
                    try:
                        dt = datetime.fromisoformat(trip["droppingtime"].replace('Z', '+00:00'))
                        ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
                        dropping_time = ist_dt.strftime("%I:%M %p")
                    except:
                        dropping_time = "Unknown"
                
                price = trip.get('fare', 'Unknown')
                available_seats = trip.get('availableseats', 'Unknown')
                
                fallback_parts.append(f"Bus {i+1}: {boarding_time} - {dropping_time} | ₹{price} | {available_seats} seats available")
                fallback_parts.append(f"Boarding: {trip.get('boardingpointname', 'Unknown')}")
                fallback_parts.append(f"Dropping: {trip.get('droppingpointname', 'Unknown')}")
            
            if len(trips) > 3:
                fallback_parts.append(f"\n...and {len(trips) - 3} more buses available.")
                
            fallback_parts.append("\nPlease ask for more details about any specific bus or provide more information to refine your search.")
            
            return " ".join(fallback_parts)
        
        # Generic fallback response
        fallback_parts.append("I'm currently having trouble processing your request. Here's what I understand:")
        
        if context.get('user_requested_source') and context.get('user_requested_destination'):
            fallback_parts.append(f"\nYou want to travel from {context['user_requested_source']} to {context['user_requested_destination']}.")
        
        if context.get('last_date'):
            fallback_parts.append(f"Date of travel: {context['last_date']}")
            
        fallback_parts.append("\nPlease try again with a simple request like 'Show buses from [source] to [destination]' or 'Book a ticket to [destination]'.")
        
        return " ".join(fallback_parts)
    
    async def generate_response(self, query, session_id=None, user_gender=None, user_location=None, detected_language=None, provider=None, model=None):
        """Generate a response to a user query"""
        print(f"\n--- New Query: '{query}' ---")
        session, session_id = self.get_or_create_session(session_id)
        context = session['context']
        
        # Determine if this is a simple query that shouldn't use fallback
        is_simple_query = query.lower() in ["hi", "hello", "hey"] or "name" in query.lower() or "who are you" in query.lower()
        print(f"Is simple query: {is_simple_query}")
        
        # Temporary provider switch if requested
        original_provider = None
        if provider and model:
            try:
                # Store original provider to restore later
                original_provider = self.ai_provider
                
                # Create and initialize new provider
                new_provider = AIProviderFactory.create_provider(
                    provider_name=provider,
                    model=model
                )
                await new_provider.initialize()
                self.ai_provider = new_provider
                
                print(f"Temporarily switched to {provider} {model} for this request")
            except Exception as e:
                print(f"Error switching provider: {e}")
                # Continue with current provider if there's an error
        
        # Check if this is a silent model switch request
        if query == "_model_switch_":
            print("Silent model switch detected - not generating a response")
            # Just return a simple acknowledgment
            yield json.dumps({"text": "", "done": True, "session_id": session_id})
            return
        
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
        
        # Store location if provided
        if user_location and user_location.get('latitude') and user_location.get('longitude'):
            context['user_location'] = user_location
            print(f"Updated user location: {user_location}")
        
        # Set gender if provided
        if user_gender:
            context['gender'] = user_gender
            print(f"Set user gender: {user_gender}")
        
        # Set language
        if detected_language:
            context['language'] = detected_language
        else:
            context['language'] = self.detect_language(query)
        print(f"Language detected/set: {context['language']}")
        
        # Extract locations from query
        source, destination = self.extract_locations(query)
        if source and destination:
            context['user_requested_source'] = source
            context['user_requested_destination'] = destination
            print(f"Extracted locations: {source} to {destination}")
        
        # Extract ticket count
        ticket_count = self.extract_ticket_count(query)
        if ticket_count:
            context['ticket_count'] = ticket_count
            print(f"Extracted ticket count: {ticket_count}")
        
        # Extract date
        if not 'last_date' in context or 'tomorrow' in query.lower() or 'today' in query.lower():
            date = self.parse_date(query)
            context['last_date'] = date
            print(f"Extracted date: {date}")
        
        # Add user message to session
        session['messages'].append({"role": "user", "content": query})
        api_data = {}
        
        # Detect seat selection
        seats, window_pref, aisle_pref = self.detect_seat_selection(query)
        if seats:
            context['selected_seats'] = seats
            api_data["selected_seats"] = seats
            print(f"Detected seat selection: {seats}")
        
        # Check if this is a bus tracking request
        is_tracking_request = self.is_bus_tracking_query(query)
        print(f"Is tracking request: {is_tracking_request}")

        # Handle bus tracking requests
        if is_tracking_request:
            print("Bus tracking request detected")
            
            # Only proceed with authenticated users for tracking
            if user_authenticated and auth_token:
                # Fetch user tickets 
                tickets = await self.fetch_user_tickets(auth_token)
                
                if tickets:
                    # Process tickets to get active, completed, future tickets
                    processed_tickets = await self.process_user_tickets(tickets)
                    
                    # Add tickets data to API data
                    api_data["all_tickets"] = tickets
                    api_data["active_tickets"] = processed_tickets["active"]
                    api_data["completed_tickets"] = processed_tickets["completed"]
                    api_data["future_tickets"] = processed_tickets["future"]
                    
                    # Check if we have active tickets
                    if processed_tickets["active"]:
                        # Use the most recent active ticket
                        active_ticket = processed_tickets["active"][0]
                        trip_id = active_ticket.get("tripId")
                        
                        if trip_id:
                            # Format as JSON for consistent frontend display
                            api_data["tracking"] = {
                                "type": "active_ticket",
                                "ticket": active_ticket,
                                "tripId": trip_id,
                                "from": active_ticket.get("source", "Unknown"),
                                "to": active_ticket.get("destination", "Unknown"),
                                "journeyDate": active_ticket.get("journeyDate", "Unknown"),
                                "eta": active_ticket.get("eta_data", {})
                            }
                            
                            # Get ETA data if not already present
                            if not active_ticket.get("eta_data"):
                                eta_data = await self.fetch_eta_data(trip_id)
                                active_ticket["eta_data"] = eta_data
                                api_data["tracking"]["eta"] = eta_data
                    
                    # If no active but completed tickets, look for NPS data
                    elif processed_tickets["completed"]:
                        # Use the most recent completed ticket
                        completed_ticket = processed_tickets["completed"][0]
                        trip_id = completed_ticket.get("tripId")
                        
                        if trip_id:
                            # Get NPS data if not already present
                            if not completed_ticket.get("nps_data"):
                                nps_data = await self.fetch_nps_data(trip_id, auth_token)
                                completed_ticket["nps_data"] = nps_data
                            
                            # Format NPS data as JSON
                            nps_json = self.format_nps_data_as_json(
                                completed_ticket.get("nps_data", {}),
                                trip_id
                            )
                            
                            api_data["tracking"] = {
                                "type": "completed_ticket",
                                "ticket": completed_ticket,
                                "tripId": trip_id,
                                "from": completed_ticket.get("source", "Unknown"),
                                "to": completed_ticket.get("destination", "Unknown"),
                                "journeyDate": completed_ticket.get("journeyDate", "Unknown"),
                                "nps": nps_json
                            }
                    
                    # If only future tickets, show the earliest one
                    elif processed_tickets["future"]:
                        future_ticket = processed_tickets["future"][0]
                        trip_id = future_ticket.get("tripId")
                        
                        if trip_id:
                            api_data["tracking"] = {
                                "type": "future_ticket",
                                "ticket": future_ticket,
                                "tripId": trip_id,
                                "from": future_ticket.get("source", "Unknown"),
                                "to": future_ticket.get("destination", "Unknown"),
                                "journeyDate": future_ticket.get("journeyDate", "Unknown")
                            }
                else:
                    # No tickets found
                    api_data["tracking"] = {
                        "type": "no_tickets",
                        "message": "No tickets found for your account"
                    }
            else:
                # User not authenticated
                api_data["tracking"] = {
                    "type": "not_authenticated",
                    "message": "Please log in to track your bus"
                }
                
        elif user_authenticated and (
            "profile" in query.lower() or 
            "my account" in query.lower() or 
            "my details" in query.lower()
        ):
            # Handle profile requests
            print("User profile request detected")
            
            # If user profile not cached or expired, fetch it
            if not context.get('user_profile'):
                user_profile = await self.fetch_user_profile(auth_token)
                if user_profile:
                    context['user_profile'] = user_profile
                    api_data["user_profile"] = user_profile
                    print("Fetched user profile")
            else:
                api_data["user_profile"] = context.get('user_profile')
                print("Using cached user profile")
        
        # Handle standard bus booking flows
        elif context.get('user_requested_source') and context.get('user_requested_destination'):
            source_id = self.stations.get(context['user_requested_source'])
            destination_id = self.stations.get(context['user_requested_destination'])
            
            if source_id and destination_id:
                # Check if looking for nearby boarding points
                looking_for_boarding = any(phrase in query.lower() for phrase in ["boarding", "pickup", "pick up", "board", "pick-up", "station", "stop", "nearby", "close", "nearest"])
                
                # Check if looking for dropping points
                looking_for_dropping = any(phrase in query.lower() for phrase in ["dropping", "drop", "drop-off", "drop off", "destination"])
                
                # Check if looking for seat selection
                looking_for_seats = any(phrase in query.lower() for phrase in ["seat", "where", "sit", "book", "select", "window", "aisle"])
                
                # Get user's selected trip from context or detect from query
                selected_trip = context.get('selected_bus')
                
                # If user doesn't have a selected trip but we have trips data, try to detect
                if not selected_trip and 'last_trips' in context:
                    trips = context['last_trips']
                    selected_trip_index = self.detect_bus_selection(query, trips)
                    if selected_trip_index is not None:
                        selected_trip = trips[selected_trip_index]
                        context['selected_bus'] = selected_trip
                        print(f"Detected bus selection: {selected_trip_index}")
                
                # Check trip selection or seat selection in query
                if not selected_trip and not api_data.get("trips"):
                    # If no trips yet, fetch them
                    date = context.get('last_date')
                    # Ensure date is not None
                    if date is None:
                        date = self.parse_date(query)
                        context['last_date'] = date
                    trips = await self.fetch_trips(source_id, destination_id, date)
                    
                    if trips:
                        # Process trips as normal
                        api_data["trips"] = trips
                        context['last_trips'] = trips
                        
                        # Add user direction to API data
                        api_data["user_direction"] = {
                            "source": context['user_requested_source'],
                            "destination": context['user_requested_destination'],
                            "swap_points": False  # Never swap points for initial search
                        }
                        
                        # If user has a location, find nearest boarding points
                        if context.get('user_location'):
                            boarding_points = []
                            
                            # Check all trips and aggregate boarding points
                            for trip in trips:
                                trip_boarding_points = await self.fetch_boarding_points(trip.get('tripid'), source_id)
                                boarding_points.extend(trip_boarding_points)
                            
                            if boarding_points:
                                api_data["boarding_points"] = boarding_points
                                
                                # Get nearest boarding points
                                nearest_points = self.get_nearest_boarding_points_info(
                                    boarding_points, 
                                    context['user_location'],
                                    max_points=3
                                )
                                
                                if nearest_points:
                                    api_data["nearest_boarding_points"] = nearest_points
                    else:
                        # Handle case when no trips are found
                        # Try reversing the direction
                        print(f"No trips found from {context['user_requested_source']} to {context['user_requested_destination']}, trying reverse direction")
                        reverse_trips = await self.fetch_trips(destination_id, source_id, date)
                        
                        if reverse_trips:
                            print(f"Found {len(reverse_trips)} trips in reverse direction")
                            
                            # Store the trips with a note about reversed direction
                            api_data["trips"] = reverse_trips
                            context['last_trips'] = reverse_trips
                            api_data["direction_reversed"] = True
                            
                            # Add user direction to API data, with swap_points=True
                            api_data["user_direction"] = {
                                "source": context['user_requested_source'],
                                "destination": context['user_requested_destination'],
                                "swap_points": True  # Swap points since direction is reversed
                            }
                        else:
                            # No trips found in either direction
                            source_name = context['user_requested_source'].capitalize()
                            destination_name = context['user_requested_destination'].capitalize()
                            travel_date = date
                            
                            formatted_date = None
                            if travel_date:
                                try:
                                    formatted_date = datetime.fromisoformat(travel_date).strftime("%A, %B %d, %Y")
                                except:
                                    formatted_date = travel_date  # Use the raw date if parsing fails
                            else:
                                formatted_date = "the selected date"  # Default if no date
                                                        
                            # Provide a helpful message about no trips
                            no_trips_message = f"I'm sorry, I couldn't find any buses from {source_name} to {destination_name} on {formatted_date}. This could be due to:"
                            
                            # Provide a helpful message about no trips
                            no_trips_message = f"I'm sorry, I couldn't find any buses from {source_name} to {destination_name} on {formatted_date}. This could be due to:"
                            no_trips_message += "\n\n1. No available services on this route for the selected date"
                            no_trips_message += "\n2. All buses being fully booked"
                            no_trips_message += "\n3. A temporary issue with our booking system"
                            
                            no_trips_message += "\n\nYou could try:"
                            no_trips_message += f"\n• Checking a different date"
                            no_trips_message += f"\n• Looking for buses to nearby destinations"
                            no_trips_message += f"\n• Checking again in a few minutes if it's a temporary issue"
                            
                            # Set api_data with route information but no trips
                            api_data["no_trips_info"] = {
                                "source": source_name,
                                "destination": destination_name,
                                "date": formatted_date,
                                "message": no_trips_message
                            }
                
                # If user selected a specific trip and is asking about boarding/dropping points
                if selected_trip:
                    trip_id = selected_trip.get('tripid')
                    
                    # Only fetch if not already in context
                    if looking_for_boarding and not api_data.get("boarding_points"):
                        boarding_points = await self.fetch_boarding_points(trip_id, source_id)
                        if boarding_points:
                            api_data["boarding_points"] = boarding_points
                            
                            # If user has location, also get nearest boarding points
                            if context.get('user_location'):
                                nearest_points = self.get_nearest_boarding_points_info(
                                    boarding_points, 
                                    context['user_location'],
                                    max_points=3
                                )
                                
                                if nearest_points:
                                    api_data["nearest_boarding_points"] = nearest_points
                                    
                                # Suggest the nearest boarding point if available
                                suggestion = self.suggest_boarding_point(
                                    boarding_points,
                                    context.get('user_location'),
                                    context.get('last_boarding_point')
                                )
                                
                                if suggestion:
                                    api_data["suggested_boarding"] = suggestion
                                    context['last_boarding_point'] = suggestion
                    
                    # Only fetch if not already in context
                    if looking_for_dropping and not api_data.get("dropping_points"):
                        dropping_points = await self.fetch_dropping_points(trip_id)
                        if dropping_points:
                            api_data["dropping_points"] = dropping_points
                            
                            # Suggest a dropping point
                            suggestion = self.suggest_dropping_point(
                                dropping_points,
                                context.get('last_dropping_point')
                            )
                            
                            if suggestion:
                                api_data["suggested_dropping"] = suggestion
                                context['last_dropping_point'] = suggestion
                    
                    # Only fetch if not already in context
                    if looking_for_seats and not api_data.get("recommendations"):
                        seats_data = await self.fetch_seats(trip_id, source_id, destination_id)
                        if seats_data:
                            # Get user preferences (either from auth or session)
                            user_preferences = None
                            
                            # If user is authenticated and mobile number available
                            if user_authenticated and user_mobile:
                                user_preferences = await self.fetch_user_preferences(user_mobile, auth_token)
                                api_data["user_preferences"] = user_preferences
                            
                            # Get recommendations
                            recommendations = self.get_seat_recommendations(
                                seats_data, 
                                context.get('ticket_count', 1),
                                user_preferences
                            )
                            
                            if recommendations:
                                api_data["recommendations"] = recommendations
                                
                    # Add preferences to api_data even if seats weren't fetched
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
        
        # Only use direct response for real bus queries, not simple ones
        if is_bus_listing_request and api_data and api_data.get("trips") and not is_simple_query:
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
            dynamic_system_prompt = self.trim_text(dynamic_system_prompt, token_limit=Config.SYSTEM_PROMPT_TOKEN_LIMIT)
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
        
        # Special handling for simple queries
        if is_simple_query:
            system_message += "\n\n--- SIMPLE QUERY HANDLING ---\n"
            system_message += "This is a simple greeting or identity question. Respond in a friendly manner."
            
            if "name" in query.lower():
                system_message += "\nRemember to state that your name is Ṧ.AI (pronounced 'Sai') and you are the Fresh Bus travel assistant."
            elif query.lower() in ["hi", "hello", "hey"]:
                system_message += "\nRespond with a friendly greeting and offer to help with bus-related questions."
        
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
        
        # Add special handling for seat selection
        if context.get('selected_seats'):
            seat_nums = ", ".join(map(str, context['selected_seats']))
            system_message += f"\n\nSEAT SELECTION: The user wants to book seat(s) {seat_nums}. Provide specific information about these seats including the price and category if available."
        
        # Track token counts for cost tracking
        input_tokens = 0
        try:
            # Count system message tokens
            input_tokens += await self.ai_provider.count_tokens(system_message)
            
            # Count message history tokens
            for msg in session['messages']:
                input_tokens += await self.ai_provider.count_tokens(msg.get("content", ""))
        except Exception as token_err:
            print(f"Error counting input tokens: {token_err}")
            input_tokens = 0  # Reset if there's an error
        
        print(f"System message token count: {input_tokens}")
        recent_messages = session['messages'][-3:]
        
        try:
            # Generate response using the AI provider abstraction
            response_text = ""
            
            print(f"Sending to {self.ai_provider.provider_name} with system message length: {len(system_message)}")
            print(f"First 100 chars of system message: {system_message[:100]}...")
            
            async for chunk in self.ai_provider.generate_stream(
                prompt=query,
                system_message=system_message,
                messages=recent_messages,
                max_tokens=Config.DEFAULT_TOKEN_LIMIT,
                temperature=0.7
            ):
                if "error" in chunk:
                    error_msg = f"Error with {self.ai_provider.provider_name} API: {chunk['error']}"
                    print(error_msg)
                    raise Exception(error_msg)
                
                if chunk.get("done", False):
                    if "complete_response" in chunk:
                        response_text = chunk["complete_response"]
                        
                        # Process the complete response
                        corrected_response = self.enforce_response_format(response_text, api_data, context)
                        
                        # If the response was modified
                        if corrected_response != response_text:
                            response_text = corrected_response
                            yield json.dumps({"text": corrected_response, "replace": True, "done": False})
                        
                        session['messages'].append({"role": "assistant", "content": response_text})
                        
                        # Track output tokens for cost tracking
                        output_tokens = 0
                        try:
                            output_tokens = await self.ai_provider.count_tokens(response_text)
                        except Exception as token_err:
                            print(f"Error counting output tokens: {token_err}")
                        
                        # Log usage for cost tracking
                        provider_name = self.ai_provider.provider_name
                        model_name = self.ai_provider.model
                        
                        usage_data = self.cost_tracker.log_request(
                            provider=provider_name,
                            model=model_name,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            session_id=session_id or "unknown",
                            query_type="chat"
                        )
                        
                        print(f"Request cost: ${usage_data['total_cost']:.5f} " + 
                            f"({usage_data['input_tokens']} input + {usage_data['output_tokens']} output tokens)")
                        
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
                    else:
                        # Just final done signal without complete text
                        yield json.dumps({
                            "text": "", 
                            "done": True, 
                            "session_id": session_id, 
                            "language_style": context.get('language', 'english')
                        })
                else:
                    # Stream text chunks
                    yield json.dumps({"text": chunk["text"], "done": False})
                    
        except Exception as e:
            error_msg = f"Error with {self.ai_provider.provider_name} API: {str(e)}"
            print(error_msg)
            
            # IMPORTANT: Clear any previous streaming response by sending a replace flag
            yield json.dumps({"text": "", "replace": True, "done": False})
            
            # Handle the case when we have bus data but the AI call failed
            if api_data and api_data.get("trips") and context.get('user_requested_source') and context.get('user_requested_destination'):
                direct_response = self.generate_direct_bus_response(api_data, context)
                print("Using direct bus response generation due to AI API error")
                session['messages'].append({"role": "assistant", "content": direct_response})
                
                # Try to save to Redis
                try:
                    conversation_id = conversation_manager.save_conversation(session_id, session['messages'])
                    if conversation_id:
                        print(f"Saved direct response to Redis: {conversation_id}")
                except Exception as redis_err:
                    print(f"Failed to save to Redis: {redis_err}")
                    conversation_id = None
                
                yield json.dumps({"text": direct_response, "done": False})
                yield json.dumps({
                    "text": "", 
                    "done": True, 
                    "session_id": session_id,
                    "language_style": context.get('language', 'english'),
                    "conversation_id": conversation_id
                })
                return
            
            # Special handling for simple queries
            if is_simple_query:
                # For simple queries that don't need API data, give simple responses
                if "name" in query.lower():
                    simple_response = "My name is Ṧ.AI (pronounced 'Sai'). I'm your Fresh Bus travel assistant. How can I help you today?"
                else:
                    simple_response = "Hello! I'm Ṧ.AI, your Fresh Bus travel assistant. How can I help you with your bus travel today?"
                    
                session['messages'].append({"role": "assistant", "content": simple_response})
                
                # Try to save to Redis
                try:
                    conversation_id = conversation_manager.save_conversation(session_id, session['messages'])
                    if conversation_id:
                        print(f"Saved simple response to Redis: {conversation_id}")
                except Exception as redis_err:
                    print(f"Failed to save to Redis: {redis_err}")
                    conversation_id = None
                    
                yield json.dumps({"text": simple_response, "done": False})
                yield json.dumps({
                    "text": "", 
                    "done": True, 
                    "session_id": session_id,
                    "language_style": context.get('language', 'english'),
                    "conversation_id": conversation_id
                })
            else:
                # Special handling for seat selection
                if context.get('selected_seats'):
                    seat_numbers = ", ".join(map(str, context.get('selected_seats')))
                    source = context.get('user_requested_source', 'your source')
                    destination = context.get('user_requested_destination', 'your destination')
                    fallback_response = f"I understand you want to book seat(s) {seat_numbers} for your journey from {source} to {destination}. "
                    
                    if context.get('selected_bus'):
                        fallback_response += "I found details about your selected bus. "
                        
                        if api_data.get("recommendations"):
                            fallback_response += "To complete your booking, please confirm these seat numbers and proceed through the Fresh Bus app."
                        else:
                            fallback_response += "However, I couldn't fetch the seat availability for this journey. Please try selecting seats directly through the Fresh Bus app."
                    else:
                        fallback_response += "However, I need you to select a bus first before we can book specific seats. Please let me know which bus you'd like to take."
                else:
                    # Use general fallback response
                    fallback_response = self._generate_fallback_response(api_data, query, context)
                    
                print("Using fallback response mechanism due to API error")
                session['messages'].append({"role": "assistant", "content": fallback_response})
                
                # Try to save to Redis even with fallback
                try:
                    conversation_id = conversation_manager.save_conversation(session_id, session['messages'])
                    if conversation_id:
                        print(f"Saved fallback conversation to Redis: {conversation_id}")
                except Exception as redis_err:
                    print(f"Failed to save to Redis: {redis_err}")
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
        finally:
            # Restore original provider if we made a temporary switch
            if original_provider:
                # Clean up temporary provider
                await self.ai_provider.cleanup()
                
                # Restore original
                self.ai_provider = original_provider
                print(f"Restored original provider: {self.ai_provider.provider_name} {self.ai_provider.model}")

# Instantiate the Fresh Bus assistant
fresh_bus_assistant = FreshBusAssistant()

# Create the FastAPI app with lifespan
app = FastAPI(title="Fresh Bus Travel Assistant", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

#################################
# API Endpoints
#################################

@app.get("/api/models")
async def get_available_models():
    """Get list of available AI models."""
    return JSONResponse(content=AIProviderFactory.get_available_providers())

@app.get("/api/current-model")
async def get_current_model():
    """Get the currently active model."""
    return JSONResponse(content={
        "provider": fresh_bus_assistant.ai_provider.provider_name.lower(),
        "model": fresh_bus_assistant.ai_provider.model,
        "display_name": AIProviderFactory.MODEL_DISPLAY_NAMES.get(
            fresh_bus_assistant.ai_provider.model, 
            fresh_bus_assistant.ai_provider.model
        )
    })

@app.post("/api/select-model")
async def select_model(selection: ModelSelectionRequest):
    """Change the AI model being used."""
    try:
        # Create the new provider
        new_provider = AIProviderFactory.create_provider(
            provider_name=selection.provider,
            model=selection.model
        )
        
        # Initialize it
        await new_provider.initialize()
        
        # Clean up old provider
        await fresh_bus_assistant.ai_provider.cleanup()
        
        # Swap provider
        old_provider = fresh_bus_assistant.ai_provider.provider_name
        old_model = fresh_bus_assistant.ai_provider.model
        fresh_bus_assistant.ai_provider = new_provider
        
        # Log the change
        print(f"Model changed from {old_provider} {old_model} to {selection.provider} {selection.model}")
        
        return {
            "success": True, 
            "message": f"Changed to {selection.provider} {selection.model}",
            "provider": selection.provider,
            "model": selection.model,
            "display_name": AIProviderFactory.MODEL_DISPLAY_NAMES.get(selection.model, selection.model)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to change model: {str(e)}"
        )

@app.get("/admin/costs/today")
async def get_today_costs():
    """Get today's AI usage costs."""
    summary = fresh_bus_assistant.cost_tracker.get_daily_summary()
    return JSONResponse(content=summary)

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

@app.get("/user/active-tickets")
async def get_active_tickets(request: Request):
    """Get active tickets for the authenticated user"""
    try:
        # Get the auth token from the request headers
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"success": False, "message": "Authentication required"}
            )
        
        auth_token = auth_header.replace("Bearer ", "")
        
        # Fetch tickets
        tickets = await fresh_bus_assistant.fetch_user_tickets(auth_token, force_refresh=True)
        
        if not tickets:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "No tickets found"}
            )
        
        # Process tickets
        processed_tickets = await fresh_bus_assistant.process_user_tickets(tickets)
        
        return JSONResponse(content=processed_tickets)
    except Exception as e:
        print(f"Error getting active tickets: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error fetching tickets: {str(e)}"}
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

@app.get("/tracking/{trip_id}")
async def get_bus_tracking(trip_id: str, request: Request):
    """Get live tracking for a specific trip"""
    try:
        # Fetch ETA data
        eta_data = await fresh_bus_assistant.fetch_eta_data(trip_id)
        
        if not eta_data:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "No tracking data found"}
            )
        
        # Format ETA data as JSON
        eta_json = fresh_bus_assistant.format_eta_data_as_json(eta_data, trip_id)
        
        # Check if this is a completed trip
        if eta_json["status"] == "completed":
            # Get auth token from headers
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                auth_token = auth_header.replace("Bearer ", "")
                
                # Fetch NPS data for completed trips
                nps_data = await fresh_bus_assistant.fetch_nps_data(trip_id, auth_token)
                
                # Add NPS data to response
                if nps_data:
                    nps_json = fresh_bus_assistant.format_nps_data_as_json(nps_data, trip_id)
                    eta_json["nps"] = nps_json
        
        return JSONResponse(content=eta_json)
    except Exception as e:
        print(f"Error getting bus tracking: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error fetching tracking data: {str(e)}"}
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
        api_key = Config.FIREWORKS_API_KEY
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
            provider = request.query_params.get("provider")
            model = request.query_params.get("model")
            
            if not query_str:
                raise HTTPException(status_code=400, detail="No query provided")
                
            query_request = QueryRequest(
                query=query_str,
                session_id=session_id,
                gender=gender,
                provider=provider,
                model=model
            )
        else:
            try:
                body = await request.json()
                
                # Check if this is a silent model switch request
                is_silent = body.get('silent', False)
                special_command = body.get('query') == "_model_switch_"
                
                if is_silent and special_command:
                    # Handle silent model switch
                    provider = body.get('provider')
                    model = body.get('model')
                    session_id = body.get('session_id')
                    
                    if provider and model:
                        # Create a new provider
                        new_provider = AIProviderFactory.create_provider(
                            provider_name=provider,
                            model=model
                        )
                        
                        # Initialize it
                        await new_provider.initialize()
                        
                        # Cleanup old provider
                        await fresh_bus_assistant.ai_provider.cleanup()
                        
                        # Set the new provider
                        fresh_bus_assistant.ai_provider = new_provider
                        
                        # Store the session's current provider if needed
                        if session_id:
                            session, _ = fresh_bus_assistant.get_or_create_session(session_id)
                            session['provider'] = provider
                            session['model'] = model
                        
                        # Return empty success response
                        return StreamingResponse(
                            iter(["data: {\"success\": true}\n\n"]), 
                            media_type="text/event-stream"
                        )
                
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
                detected_language=detected_language,
                provider=query_request.provider,
                model=query_request.model
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
