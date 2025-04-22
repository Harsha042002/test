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
import datetime
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

def detect_language(text):
    """
    Detect the language of the given text.
    Currently a simple implementation that defaults to english.
    """
    # Add more sophisticated detection if needed
    if text:
        # Very basic detection - could be enhanced with a proper language detection library
        text_lower = text.lower()
        if any(word in text_lower for word in ['हिंदी', 'में', 'का', 'है', 'हूँ', 'नहीं']):
            return 'hindi'
        if any(word in text_lower for word in ['తెలుగు', 'లో', 'కు', 'లేదు', 'ఉంది']):
            return 'telugu'
        # Add more languages as needed
    return 'english'

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
    user_mobile: Optional[str] = None
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
        self.user_index_prefix = "fresh_bus:user_index:"  # Add this line
        self.global_index_key = "fresh_bus:conversations"
    
    def save_conversation(self, session_id, messages, user_id=None):
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
            
            if user_id:
                conversation_data["user_id"] = str(user_id)
            # Save conversation
            key = f"{self.conversation_prefix}{conversation_id}"
            self.redis.hmset(key, conversation_data)
            # Add to session index
            self.redis.sadd(f"{self.session_index_prefix}{session_id}", conversation_id)
            if user_id:
                self.redis.sadd(f"{self.user_index_prefix}{user_id}", conversation_id)
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
    def initialize_ai_provider(self):
        try:
            provider_name = Config.DEFAULT_AI_PROVIDER.lower().strip()
            provider_model = Config.DEFAULT_AI_MODEL
            
            # Validate provider name
            valid_providers = ["gemini", "claude"]
            if provider_name not in valid_providers:
                print(f"Invalid provider '{provider_name}', falling back to gemini")
                provider_name = "gemini"
                provider_model = "gemini-pro"

            self.ai_provider = AIProviderFactory.create_provider(
                provider_name=provider_name,
                model=provider_model
            )
            print(f"Initialized {provider_name} provider with model {provider_model}")
            return True
        except Exception as e:
            print(f"Error initializing AI provider: {e}")
            # Set fallback provider
            self.ai_provider = AIProviderFactory.create_provider(
                provider_name="gemini",
                model="gemini-pro"
            )
            return False

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
        self.initialize_ai_provider()
        
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
    
    def is_valid_route(self, source, destination):
        """
        Check if a route is valid based on the approved routes in the system prompt
        """
        # Define the approved routes based on system prompt
        approved_routes = [
            ("hyderabad", "guntur"),
            ("hyderabad", "vijayawada"),
            ("vijayawada", "hyderabad"),
            ("guntur", "hyderabad"),
            ("bangalore", "tirupati"),
            ("bangalore", "chittoor"),
            ("tirupati", "bangalore"),
            ("chittoor", "bangalore")
        ]
        
        # Normalize source and destination
        source = source.lower() if source else ""
        destination = destination.lower() if destination else ""
        
        # Check if the route is in the approved routes
        return (source, destination) in approved_routes

    async def init_system_prompt(self):
        if not self.system_prompt_initialized:
            try:
                # Load base system prompt (qa_prompt.txt)
                with open(Config.DEFAULT_SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
                    self.system_prompt = f.read()
                    print(f"Loaded base prompt ({len(self.system_prompt)} chars)")

                # Load identity info
                identity_info_path = os.path.join(os.path.dirname(Config.DEFAULT_SYSTEM_PROMPT_PATH), "identity_info.txt")
                if os.path.exists(identity_info_path):
                    with open(identity_info_path, 'r', encoding='utf-8') as f:
                        identity_info = f.read()
                        print(f"Loaded identity info ({len(identity_info)} chars)")
                        self.system_prompt = f"{self.system_prompt}\n\n{identity_info}"

                # Validate system prompt content
                if "FreshBus" not in self.system_prompt or "Ṧ.AI" not in self.system_prompt:
                    raise ValueError("System prompt appears incomplete!")

                # Initialize VectorDB
                await self.embeddings_client.init_session()
                await self.vector_db.store_system_prompt(self.system_prompt)
                self.system_prompt_initialized = True
                print("System prompt initialization complete")
                return True
            except Exception as e:
                print(f"Error initializing system prompt: {e}")
                return False
    
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
    
    
    
    def is_simple_query(self, query):
        """Detect if this is a simple greeting or identity question"""
        query_lower = query.lower().strip()
        simple_greetings = ["hi", "hello", "hey", "hola", "namaste", "greetings"]
        
        # Check for simple greetings
        if query_lower in simple_greetings or query_lower.startswith("hi ") or query_lower.startswith("hello "):
            return True
        
        # Check for identity/name questions
        if "name" in query_lower and ("your" in query_lower or "you" in query_lower):
            return True
        
        # Check for "who are you" type questions
        if "who are you" in query_lower or "what are you" in query_lower:
            return True
        
        return False
    
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
    
    async def handle_authenticated_tracking_request(self, access_token):
        """Handle authenticated tracking request by fetching user's first ticket and providing tracking info."""
        try:
            # Fetch user's tickets using their access token
            tickets_url = f"{self.BASE_URL_CUSTOMER}/tickets"
            print(f"Fetching user tickets from: {tickets_url}")
            
            async with self.http_session.get(
                tickets_url,
                headers={"Authorization": f"Bearer {access_token}"}
            ) as tickets_response:
                if tickets_response.status != 200:
                    error_text = await tickets_response.text()
                    print(f"Error fetching tickets: {tickets_response.status}")
                    print(f"Error response: {error_text}")
                    return "I'm having trouble accessing your ticket information at the moment. Please try again in a few minutes or contact Fresh Bus support if the issue persists."
                
                tickets_data = await tickets_response.json()
                print(f"Found {len(tickets_data)} tickets")
                
                if not tickets_data:
                    return "I don't see any bus bookings in your account. If you've recently booked a ticket, it might take a moment to appear in your account. Please check back in a few minutes."
                
                # Take the first ticket from the response
                current_ticket = tickets_data[0]
                
                # Extract tracking URL
                tracking_url = current_ticket.get("trackingUrlNew")
                if not tracking_url:
                    tracking_url = current_ticket.get("trackingUrl")
                
                # Get destination time to check if trip is completed
                is_completed = False
                if isinstance(current_ticket.get("destination"), dict) and current_ticket["destination"].get("time"):
                    dest_time_str = current_ticket["destination"]["time"]
                    try:
                        # Parse ISO format date string to datetime
                        dest_time = datetime.datetime.fromisoformat(dest_time_str.replace('Z', '+00:00'))
                        
                        # Convert to local time for comparison
                        dest_time = dest_time.replace(tzinfo=datetime.timezone.utc).astimezone(tz=None)
                        
                        # Check if the trip is completed
                        current_time = datetime.datetime.now()
                        is_completed = dest_time < current_time
                    except Exception as e:
                        print(f"Error parsing date {dest_time_str}: {e}")
                
                # Get basic ticket information
                from_location = current_ticket.get("source", {}).get("name", "your pickup point")
                to_location = current_ticket.get("destination", {}).get("name", "your destination")
                departure_time = self.format_datetime(current_ticket.get("source", {}).get("time", ""))
                arrival_time = self.format_datetime(current_ticket.get("destination", {}).get("time", ""))
                booking_id = current_ticket.get("bookingid", "")
                
                # If trip is completed, show NPS prompt
                if is_completed:
                    response = f"I found your recent journey from {from_location} to {to_location}.\n\n"
                    response += f"This trip completed on {arrival_time}.\n\n"
                    response += "How was your experience with Fresh Bus? We'd love to hear your feedback to improve our service."
                    return response
                
                # Otherwise, provide tracking info for active trip
                response = f"I found your ticket from {from_location} to {to_location}!\n\n"
                response += f"📍 Trip: {from_location} to {to_location}\n"
                response += f"🕒 Departure: {departure_time}\n"
                response += f"🏁 Estimated Arrival: {arrival_time}\n"
                if booking_id:
                    response += f"🎫 Booking ID: {booking_id}\n\n"
                
                if tracking_url:
                    response += f"Track your bus in real-time here: {tracking_url}\n\n"
                    response += "This link will show you the live location of your bus on a map."
                else:
                    response += "Real-time tracking isn't available for this journey. Please check the Fresh Bus app for more details."
                
                return response
                    
        except Exception as e:
            print(f"Error in handle_authenticated_tracking_request: {e}")
            import traceback
            traceback.print_exc()
            return "I encountered an error while getting your bus location. Please check the Fresh Bus app for the most current tracking information."
        
    def format_datetime(self, datetime_str):
        """Format ISO datetime string to readable format."""
        if not datetime_str:
            return "Not available"
        
        try:
            dt = datetime.datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            return dt.strftime("%d %b %Y, %I:%M %p")
        except Exception:
            return datetime_str
    
    async def fetch_user_profile(self, access_token: str, refresh_token: Optional[str] = None):
        """
        Fetch the user's profile from the customer API gateway.
        Sends access_token and refresh_token as cookies, plus the Bearer header.
        """
        if not self.http_session:
            await self.init_http_session()
    
        url = f"{self.BASE_URL_CUSTOMER}/profile"
        
        # Setup headers with Bearer token
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
        # Setup cookies with both tokens
        cookies = {
            "access_token": access_token
        }
        if refresh_token:
            cookies["refresh_token"] = refresh_token
            headers["refresh_token"] = refresh_token  # Also send as header just in case
    
        print(f"Fetching user profile from {url}")
        print(f"Using cookies: {list(cookies.keys())}")
        print(f"Using headers: {list(headers.keys())}")
    
        try:
            async with self.http_session.get(
                url,
                headers=headers,
                cookies=cookies
            ) as response:
                if response.status == 200:
                    profile_data = await response.json()
                    print(f"Successfully fetched profile for user: {profile_data.get('name', 'Unknown')}")
                    return profile_data
                else:
                    error_text = await response.text()
                    print(f"❌ Error fetching user profile: {response.status} — {error_text}")
                    if response.status == 401 and refresh_token:
                        print("Token expired, will attempt refresh")
                    return None
        except Exception as e:
            print(f"Exception in fetch_user_profile: {e}")
            return None
            await self.init_http_session()

        url = f"{self.BASE_URL_CUSTOMER}/profile"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        # send both tokens as cookies
        cookies = {"access_token": access_token}
        if refresh_token:
            cookies["refresh_token"] = refresh_token

        print(f"Fetching user profile: {url} with cookies {list(cookies.keys())}")
        async with self.http_session.get(url, headers=headers, cookies=cookies) as resp:
            if resp.status == 200:
                data = await resp.json()
                print("✅ Successfully fetched user profile")
                return data
            else:
                err = await resp.text()
                print(f"❌ Error fetching user profile: {resp.status} — {err}")
                return None



    
    # Updated method to handle user tickets with caching
    async def fetch_user_tickets(self, access_token):
        """Fetch user tickets with proper authentication"""
        if not self.http_session:
            await self.init_http_session()
            
        try:
            # Use the tickets API endpoint with authentication
            url = f"{self.base_url_customer}/tickets"
            print(f"Fetching user tickets: {url}")
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            async with self.http_session.get(url, headers=headers) as response:
                if response.status == 200:
                    tickets = await response.json()
                    print(f"Successfully fetched {len(tickets)} tickets")
                    return tickets
                else:
                    print(f"Error fetching tickets: {response.status}")
                    error_text = await response.text()
                    print(f"Error response: {error_text}")
                    return []
        except Exception as e:
            print(f"Exception fetching tickets: {e}")
            return []

    async def process_user_tickets(self, tickets):
        """Process user tickets into active, completed, and future categories"""
        if not tickets:
            return {"active": [], "completed": [], "future": []}
        
        now = datetime.now()
        active_tickets = []
        completed_tickets = []
        future_tickets = []
        
        try:
            for ticket in tickets:
                # Extract journey date from the ticket
                journey_date_str = ticket.get("date")
                
                if not journey_date_str:
                    continue
                    
                try:
                    journey_date = datetime.fromisoformat(journey_date_str.replace('Z', '+00:00'))
                    
                    # Add a field for easy comparison
                    ticket["journey_datetime"] = journey_date
                    
                    # Get status from ticket
                    status = ticket.get("status", "").lower()
                    
                    # Categorize based on status or date
                    if status == "completed":
                        completed_tickets.append(ticket)
                    elif journey_date > now:
                        future_tickets.append(ticket)
                    else:
                        active_tickets.append(ticket)
                        
                except Exception as date_error:
                    print(f"Error parsing date {journey_date_str}: {date_error}")
                    continue
            
            # Sort each list by date
            active_tickets.sort(key=lambda x: x["journey_datetime"], reverse=False)
            completed_tickets.sort(key=lambda x: x["journey_datetime"], reverse=True)
            future_tickets.sort(key=lambda x: x["journey_datetime"], reverse=False)
            
            print(f"Processed tickets: {len(active_tickets)} active, {len(completed_tickets)} completed, {len(future_tickets)} future")
            
            return {
                "active": active_tickets,
                "completed": completed_tickets,
                "future": future_tickets
            }
        except Exception as e:
            print(f"Error processing tickets: {e}")
            return {"active": [], "completed": [], "future": []}

    async def fetch_eta_data(self, trip_id):
        """Fetch ETA data for a trip"""
        if not self.http_session:
            await self.init_http_session()
            
        try:
            # Use the ETA API endpoint
            url = f"{self.base_url}/eta-data?id={trip_id}"
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

    def is_bus_tracking_query(self, query):
        """Check if the query is related to bus tracking"""
        tracking_keywords = [
            "where is my bus", "track my bus", "bus location", "bus status",
            "journey status", "my trip", "my bus", "eta", "arrival time",
            "when will my bus arrive", "bus tracking", "track bus",
            "bus position", "locate my bus", "find my bus", "current location"
        ]
        
        query_lower = query.lower()
        
        for keyword in tracking_keywords:
            if keyword in query_lower:
                return True
                
        return False

    async def handle_unauthenticated_tracking_request(self):
        """Generate a helpful response for tracking requests without authentication"""
        tracking_response = (
            "I need to see your active tickets to track your bus. "
            "Please log in to your Fresh Bus account first, then I can show you your bus location, "
            "estimated arrival time, and journey status.\n\n"
            "After logging in, you can ask me 'Where is my bus?' again, and I'll provide real-time tracking information."
        )
        return tracking_response
    
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
    async def fetch_nps_data(self, trip_id, access_token=None):
        """Fetch NPS/feedback data for a completed trip"""
        if not access_token:
            return {"message": "Authentication required to fetch feedback data"}
            
        if not self.http_session:
            await self.init_http_session()
            
        try:
            # Use the feedback questions API
            url = f"{self.BASE_URL_CUSTOMER}/tickets/feedbackQuestions"
            print(f"Fetching feedback data: {url}")
            
            headers = {
                "Authorization": f"Bearer {access_token}",
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
        
    async def get_active_tickets_and_status(self, access_token):
        """Get user's active tickets and their status (future, ongoing, or completed)"""
        if not self.http_session:
            await self.init_http_session()
        
        try:
            # Get all tickets
            tickets = await self.fetch_user_tickets(access_token)
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
    
    async def generate_direct_bus_response(self, api_data: dict, context: dict, *, return_json: bool = False):
        import json
        from datetime import datetime, timezone, timedelta

        # ── invalid-route / no-trips ─────────────────────────
        if api_data.get("invalid_route"):
            bad = api_data["invalid_route"]
            msg = (
                f"No buses from {bad['requested_source'].title()} to "
                f"{bad['requested_destination'].title()}. "
                "Available routes:\n"
                + "\n".join(f"• {r}" for r in bad["valid_routes"])
            )
            return {"error":"invalid_route","message":msg} if return_json else msg

        trips = api_data.get("trips", [])
        if not trips:
            msg = "I couldn’t find any buses that match your search."
            return {"error":"no_trips","message":msg} if return_json else msg

        # ── compute display date ─────────────────────────────
        raw = context.get("last_date")
        if raw:
            try:
                dt = datetime.fromisoformat(raw.replace("Z","+00:00"))
                date_label = dt.strftime("%d %b %Y"); iso_date = dt.date().isoformat()
            except:
                date_label = iso_date = raw
        else:
            now_ist = datetime.now(timezone(timedelta(hours=5,minutes=30)))
            tmw = now_ist + timedelta(days=1)
            date_label = tmw.strftime("%d %b %Y"); iso_date = tmw.date().isoformat()

        # ── header ───────────────────────────────────────────
        src = context.get("user_requested_source", trips[0]["source"])
        dst = context.get("user_requested_destination", trips[0]["destination"])
        header = f"I found {len(trips)} bus{'es' if len(trips)>1 else ''} from {src} to {dst} on {date_label}:\n\n"

        # ── IST formatter ───────────────────────────────────
        def ist(ts):
            try:
                d = datetime.fromisoformat(ts.replace("Z","+00:00"))
                return d.astimezone(timezone(timedelta(hours=5,minutes=30))).strftime("%I:%M %p").lstrip("0")
            except:
                return "—"

        # ── detect API recs ─────────────────────────────────
        def has_api_recs(r):
            return any(
                r.get(cat,{}).get(pos)
                for cat in ("Premium","Reasonable","Budget-Friendly")
                for pos in ("window","aisle")
            )

        api_recs = api_data.get("recommendations", {})
        trips_text, trips_json = [], []

        for t in trips:
            dep = ist(t["boardingtime"]); arr = ist(t["droppingtime"])
            # duration
            dur = ""
            try:
                a = datetime.fromisoformat(t["boardingtime"].replace("Z","+00:00"))
                b = datetime.fromisoformat(t["droppingtime"].replace("Z","+00:00"))
                m = int((b-a).total_seconds()//60); dur = f"{m//60}h {m%60}m"
            except:
                pass

            # ── build recs blob ─────────────────────────────
            if has_api_recs(api_recs):
                recs = api_recs
            else:
                sd = await self.fetch_seats(t["tripid"],
                                            context.get("source_id"),
                                            context.get("destination_id")) or {}
                recs = {c:{"window":[],"aisle":[]} for c in ("Premium","Reasonable","Budget-Friendly")}
                for s in sd.get("seats", []):
                    num   = s.get("seatName") or s.get("number")
                    sid   = s.get("seat_id")  or s.get("seatId")
                    base  = s.get("baseFare", 0)
                    gst   = s.get("gst", 0)
                    disc  = s.get("discount", 0)
                    gross = base + gst + disc
                    cat   = s.get("category","Reasonable")
                    pos   = "window" if num in self.window_seats else "aisle"
                    if num is not None and sid:
                        recs.setdefault(cat,{"window":[],"aisle":[]})
                        recs[cat][pos].append({
                            "seatNumber": str(num),
                            "seatId":     sid,
                            "price":      str(gross),
                            "fare": {
                                "Base Fare": base,
                                "GST":        gst,
                                "Discount":   disc
                            }
                        })

            # ── bullets & JSON recs ────────────────────────
            bullets, json_recs = [], {}
            for disp, cat in [("Premium","Premium"),
                              ("Reasonable","Reasonable"),
                              ("Budget‑Friendly","Budget-Friendly")]:
                wins = recs.get(cat,{}).get("window", [])
                ails = recs.get(cat,{}).get("aisle",  [])
                key  = disp.lower().replace("‑","_").replace("-","_")
                json_recs[key] = {"window": {}, "aisle": {}}

                parts = []
                if wins:
                    w  = wins[0]
                    sn = w.get("seatNumber") or w.get("number","")
                    sp = w.get("price")      or ""
                    parts.append(f"Window {sn} (₹{sp})")
                    json_recs[key]["window"] = w
                if ails:
                    a  = ails[0]
                    sn = a.get("seatNumber") or a.get("number","")
                    sp = a.get("price")      or ""
                    parts.append(f"Aisle {sn} (₹{sp})")
                    json_recs[key]["aisle"]  = a

                if parts:
                    bullets.append(f"• **{disp}**: {', '.join(parts)}")

            # ── assemble text ─────────────────────────────
            trips_text.append(
                f"🚌 {src} → {dst} | {dep} - {arr} | {t.get('vehicletype','')} | {dur} "
                f"Price: ₹{t['fare']} | {t['availableseats']} seats | Rating {t['redbusrating']}/5\n"
                f"Boarding Points: {api_data.get('boarding_points',[])}\n"
                f"Dropping Points: {api_data.get('dropping_points',[])}\n"
                f"Recommended seats: {'  '.join(bullets)}"
            )

            # ── assemble JSON ─────────────────────────────
            trips_json.append({
                "journeyDate":     iso_date,
                "busNumber":       t["servicenumber"],
                "price":           str(t["fare"]),
                "seats":           str(t["availableseats"]),
                "rating":          str(t["redbusrating"]),
                "from":            src,
                "to":              dst,
                "boardingPoints":  api_data.get("boarding_points", []),
                "droppingPoints":  api_data.get("dropping_points", []),
                "departureTime":   dep,
                "arrivalTime":     arr,
                "duration":        dur,
                "tripId":          str(t["tripid"]),
                "busType":         t["vehicletype"],
                "recommendations": json_recs
            })

        if return_json:
            return {"trips": trips_json}

        return (
            header
            + "\n\n".join(trips_text)
            + "\n```json\n"
            + json.dumps({"trips": trips_json}, indent=2)
            + "\n```"
        )


    def generate_direct_bus_response_sync(self, api_data, context, return_json=False):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.generate_direct_bus_response(api_data, context, return_json=return_json)
        )

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
        
        for idx, trip in enumerate(trips):
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
            
            # Process available seats directly from the API data or fetch them
            available_seats_data = []
            seat_data = api_data.get("seats", [])
            
            # If seat data is not present in the main API response, fetch it
            if not api_data.get("recommendations") and trip.get('tripid'):
                trip_id = trip.get('tripid')
                source_id = context.get('source_id') or context.get('user_requested_source') or '3'
                destination_id = context.get('destination_id') or context.get('user_requested_destination') or '13'
                
                # Convert source/destination names to IDs if needed
                if not str(source_id).isdigit():
                    source_id = self.stations.get(str(source_id).lower(), '3')
                if not str(destination_id).isdigit():
                    destination_id = self.stations.get(str(destination_id).lower(), '13')
                
                # Create a non-async HTTP request since we're in a synchronous method
                import requests
                try:
                    seats_url = f"{self.BASE_URL}/trips/{trip_id}/seats?source_id={source_id}&destination_id={destination_id}"
                    print(f"Fetching seats from: {seats_url}")
                    
                    response = requests.get(seats_url, timeout=10)
                    
                    if response.status_code == 200:
                        seat_data = response.json()
                        if "seats" in seat_data:
                            for seat in seat_data["seats"]:
                                if (not seat.get('isOccupied', False) and 
                                    seat.get('availabilityStatus', '') in ['A', 'M', 'F']):
                                    try:
                                        seat_number = int(seat.get('seatName', '0'))
                                        available_seats_data.append({
                                            'number': seat_number,
                                            'price': seat.get('totalFare', 0)
                                        })
                                    except (ValueError, TypeError):
                                        # Skip invalid seat numbers
                                        pass
                except Exception as e:
                    print(f"Error fetching seat data: {e}")
            
            # Create seat recommendations based on price tiers if we have seat data
            recommendations = {
                "Reasonable": {"window": [], "aisle": []},
                "Premium": {"window": [], "aisle": []},
                "Budget-Friendly": {"window": [], "aisle": []}
            }
            
            if available_seats_data:
                # Split seats into window and aisle first
                window_seats = []
                aisle_seats = []
                
                for seat in available_seats_data:
                    seat_number = seat['number']
                    
                    # Determine if it's window or aisle based on the CONFIG
                    if seat_number in Config.WINDOW_SEATS:
                        window_seats.append(seat)
                    elif seat_number in Config.AISLE_SEATS:
                        aisle_seats.append(seat)
                
                # Sort both lists by price
                window_seats.sort(key=lambda s: s['price'])
                aisle_seats.sort(key=lambda s: s['price'])
                
                # If we have both window and aisle seats, process them
                if window_seats and aisle_seats:
                    # Get min and max prices
                    all_prices = [s['price'] for s in window_seats + aisle_seats]
                    min_price = min(all_prices)
                    max_price = max(all_prices)
                    price_range = max_price - min_price
                    
                    # Define price thresholds
                    budget_max = min_price + (price_range * 0.33) if price_range > 0 else min_price + 50
                    premium_min = max_price - (price_range * 0.33) if price_range > 0 else max_price - 50
                    
                    # Categorize window seats
                    for seat in window_seats:
                        if seat['price'] <= budget_max:
                            recommendations["Budget-Friendly"]["window"].append(seat)
                        elif seat['price'] >= premium_min:
                            recommendations["Premium"]["window"].append(seat)
                        else:
                            recommendations["Reasonable"]["window"].append(seat)
                    
                    # Categorize aisle seats
                    for seat in aisle_seats:
                        if seat['price'] <= budget_max:
                            recommendations["Budget-Friendly"]["aisle"].append(seat)
                        elif seat['price'] >= premium_min:
                            recommendations["Premium"]["aisle"].append(seat)
                        else:
                            recommendations["Reasonable"]["aisle"].append(seat)
                    
                    # Ensure we have at least one seat of each type in each category if possible
                    # If a category is missing either window or aisle, try to add one from adjacent categories
                    for category in ["Budget-Friendly", "Reasonable", "Premium"]:
                        # If we're missing window seats in this category
                        if not recommendations[category]["window"] and category == "Budget-Friendly":
                            # Try to get the cheapest window seat from Reasonable
                            if recommendations["Reasonable"]["window"]:
                                recommendations[category]["window"].append(recommendations["Reasonable"]["window"][0])
                        
                        # If we're missing aisle seats in this category
                        if not recommendations[category]["aisle"] and category == "Budget-Friendly":
                            # Try to get the cheapest aisle seat from Reasonable
                            if recommendations["Reasonable"]["aisle"]:
                                recommendations[category]["aisle"].append(recommendations["Reasonable"]["aisle"][0])
            
            # Add tripid to the listing for debugging
            tripid_note = f"(Trip ID: {trip.get('tripid', 'N/A')})"
            
            # Create formatted bus listing with user's requested direction
            bus_listing = f"> *🚌 {source} to {destination} | {departure_time} - {arrival_time} | {bus_type} | {duration} {tripid_note}*  \n"
            bus_listing += f"> Price: ₹{price} | {available_seats} seats | Rating: {rating}/5  \n"
            bus_listing += f"> Boarding: {boarding_point}{boarding_distance}  \n"
            bus_listing += f"> Dropping: {dropping_point}  \n"
            bus_listing += f"> Recommended seats:  \n"

            # List of standard categories we always want to display (in order)
            standard_categories = ["Reasonable", "Premium", "Budget-Friendly"]

            # Check if we have recommendations from the API data first
            api_recommendations = api_data.get("recommendations", {})

            # Process each standard category
            for i, category in enumerate(standard_categories):
                bus_listing += f"> • **{category}**: "
                
                window_info = ""
                aisle_info = ""
                has_recommendations = False
                
                # First try to use recommendations from API data
                if api_recommendations and api_recommendations.get(category):
                    # Get window seat recommendation if available
                    if (api_recommendations[category].get("window") and 
                        len(api_recommendations[category]["window"]) > 0):
                        window_seat = api_recommendations[category]["window"][0].get("number", "")
                        window_price = api_recommendations[category]["window"][0].get("price", "")
                        if window_seat and window_price:
                            window_info = f"Window {window_seat} (₹{window_price})"
                            has_recommendations = True
                    
                    # Get aisle seat recommendation if available
                    if (api_recommendations[category].get("aisle") and 
                        len(api_recommendations[category]["aisle"]) > 0):
                        aisle_seat = api_recommendations[category]["aisle"][0].get("number", "")
                        aisle_price = api_recommendations[category]["aisle"][0].get("price", "")
                        if aisle_seat and aisle_price:
                            aisle_info = f"Aisle {aisle_seat} (₹{aisle_price})"
                            has_recommendations = True
                
                # If we don't have recommendations from API, use our dynamically created ones
                elif recommendations[category]["window"] or recommendations[category]["aisle"]:
                    if recommendations[category]["window"] and len(recommendations[category]["window"]) > 0:
                        window_seat = recommendations[category]["window"][0]["number"]
                        window_price = recommendations[category]["window"][0]["price"]
                        window_info = f"Window {window_seat} (₹{window_price})"
                        has_recommendations = True
                    
                    if recommendations[category]["aisle"] and len(recommendations[category]["aisle"]) > 0:
                        aisle_seat = recommendations[category]["aisle"][0]["number"]
                        aisle_price = recommendations[category]["aisle"][0]["price"]
                        aisle_info = f"Aisle {aisle_seat} (₹{aisle_price})"
                        has_recommendations = True
                
                # Add the seat information if available, otherwise indicate no data
                if has_recommendations:
                    if window_info:
                        bus_listing += window_info
                        if aisle_info:
                            bus_listing += ", "
                    if aisle_info:
                        bus_listing += aisle_info
                else:
                    bus_listing += "No seats available"
                
                # Add line break (but ensure proper formatting for markdown)
                if i < len(standard_categories) - 1:
                    bus_listing += "  \n"
            
            formatted_buses.append(bus_listing)
        
        # Add a final validation check
        if len(formatted_buses) != trip_count:
            print(f"WARNING: Formatted {len(formatted_buses)} buses but there are {trip_count} unique trips!")
        
        return "\n\n".join(formatted_buses)
    
    def get_or_create_session(self, session_id):
        try:
            # Create a new session ID if none provided
            if not session_id:
                session_id = str(uuid.uuid4())
                print(f"Created new session ID: {session_id}")
            
            # Check if session exists
            if session_id not in self.sessions:
                print(f"Creating new session with ID: {session_id}")
                # Create a new session with all required fields
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
                        "user_profile": None
                    }
                }
            else:
                print(f"Using existing session with ID: {session_id}")
                # Ensure existing session has all required fields
                if "context" not in self.sessions[session_id] or self.sessions[session_id]["context"] is None:
                    print(f"Fixing missing context in session {session_id}")
                    self.sessions[session_id]["context"] = {}
                
                if "messages" not in self.sessions[session_id] or self.sessions[session_id]["messages"] is None:
                    print(f"Fixing missing messages in session {session_id}")
                    self.sessions[session_id]["messages"] = []
            
            return self.sessions[session_id], session_id
        except Exception as e:
            print(f"Error in get_or_create_session: {e}")
            # Create a fallback session
            fallback_session_id = str(uuid.uuid4())
            fallback_session = {
                "messages": [],
                "last_updated": datetime.now(),
                "context": {}
            }
            self.sessions[fallback_session_id] = fallback_session
            return fallback_session, fallback_session_id
    
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
                # Find best matching station keys
                best_src_match = None
                best_dst_match = None
                # Find exact or partial matches
                for city in self.stations.keys():
                    if src == city or src in city or city in src:
                        if best_src_match is None or len(city) < len(best_src_match):
                            best_src_match = city
                    if dst == city or dst in city or city in dst:
                        if best_dst_match is None or len(city) < len(best_dst_match):
                            best_dst_match = city
                
                if best_src_match and best_dst_match:
                    print(f"Extracted from pattern: FROM {best_src_match} TO {best_dst_match}")
                    return best_src_match, best_dst_match
                    
        # Check for cities mentioned
        cities_mentioned = []
        for city in self.stations.keys():
            if city in query_lower:
                cities_mentioned.append(city)
        
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
        """Fetch trips with proper error handling."""
        if not self.http_session:
            await self.init_http_session()
        
        # Make sure journey_date is not None
        if journey_date is None:
            journey_date = datetime.now().strftime("%Y-%m-%d")
            print(f"Using default date: {journey_date}")
        
        # Debug
        print(f"Fetching trips for route: source_id={source_id}, destination_id={destination_id}, date={journey_date}")
        
        url = f"{self.BASE_URL}/trips?journey_date={journey_date}&source_id={source_id}&destination_id={destination_id}"
        print(f"Fetching trips from URL: {url}")
        
        try:
            headers = {"Content-Type": "application/json"}
            async with self.http_session.get(url, headers=headers) as response:
                print(f"Got response status: {response.status}")
                if response.status == 200:
                    response_text = await response.text()
                    print(f"Got response: {response_text[:200]}...")  # Print first 200 chars
                    
                    try:
                        data = json.loads(response_text)
                        if data:
                            print(f"Successfully fetched {len(data)} trips")
                            return self.deduplicate_trips(data)
                        else:
                            print("API returned empty response (no trips found)")
                            return []
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        print(f"Response content: {response_text[:500]}")
                        return []
                else:
                    error_text = await response.text()
                    print(f"Error fetching trips: {response.status}, response: {error_text}")
                    return []
        except Exception as e:
            print(f"Exception fetching trips: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def fetch_boarding_points(self, trip_id, source_id):
        """Fetch boarding points for a specific trip."""
        if not self.http_session:
            await self.init_http_session()
        
        url = f"{self.BASE_URL}/trips/{trip_id}/boardings/{source_id}"
        print(f"Fetching boarding points: {url}")
        
        try:
            async with self.http_session.get(url) as response:
                if response.status == 200:
                    boarding_points = await response.json()
                    print(f"Successfully fetched {len(boarding_points)} boarding points")
                    return boarding_points
                else:
                    print(f"Error fetching boarding points: {response.status}")
                    error_text = await response.text()
                    print(f"Error response: {error_text}")
                    return []
        except Exception as e:
            print(f"Exception fetching boarding points: {e}")
            return []
    
    async def fetch_dropping_points(self, trip_id, destination_id=None):
        """Fetch dropping points for a specific trip, filtered by destination if provided."""
        if not self.http_session:
            await self.init_http_session()
        
        url = f"{self.BASE_URL}/trips/{trip_id}/droppings"
        print(f"Fetching dropping points: {url}")
        
        try:
            async with self.http_session.get(url) as response:
                if response.status == 200:
                    all_dropping_points = await response.json()
                    
                    # Filter dropping points by destination_id if provided
                    if destination_id and all_dropping_points:
                        # Try to convert to int if it's a string
                        if isinstance(destination_id, str) and destination_id.isdigit():
                            destination_id = int(destination_id)
                        
                        filtered_dropping_points = [
                            point for point in all_dropping_points 
                            if point.get('droppingPoint', {}).get('stationId') == destination_id
                        ]
                        
                        # Only return filtered points if we found some, otherwise return all
                        if filtered_dropping_points:
                            print(f"Filtered dropping points for destination {destination_id}: {len(filtered_dropping_points)} points")
                            return filtered_dropping_points
                        else:
                            print(f"No dropping points found for destination {destination_id}, returning all {len(all_dropping_points)} points")
                    
                    return all_dropping_points
                else:
                    print(f"Error fetching dropping points: {response.status}")
                    error_text = await response.text()
                    print(f"Error response: {error_text}")
                    return []
        except Exception as e:
            print(f"Exception fetching dropping points: {e}")
            return []
    
    async def fetch_seats(self, trip_id, source_id, destination_id):
        """Fetch seat availability for a specific trip."""
        if not self.http_session:
            await self.init_http_session()
        
        url = f"{self.BASE_URL}/trips/{trip_id}/seats?source_id={source_id}&destination_id={destination_id}"
        print(f"Fetching seats from: {url}")
        
        try:
            async with self.http_session.get(url) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        print(f"Received seat data with keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dictionary'}")
                        
                        # Return the data as-is
                        return data
                    except Exception as json_error:
                        print(f"Error parsing seat JSON: {json_error}")
                        error_text = await response.text()
                        print(f"Raw response: {error_text[:500]}")  # First 500 chars for debugging
                        return {"seats": []}
                else:
                    print(f"Error fetching seats: Status {response.status}")
                    try:
                        error_text = await response.text()
                        print(f"Error response: {error_text[:500]}")  # First 500 chars for debugging
                    except:
                        print("Could not read error response text")
                    return {"seats": []}
        except Exception as e:
            print(f"Exception fetching seats: {e}")
            import traceback
            traceback.print_exc()
            return {"seats": []}
    
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
    
    async def fetch_user_preferences(self, mobile, access_token=None):
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
            if access_token:
                headers["Authorization"] = f"Bearer {access_token}"
                
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
    
    async def get_seat_recommendations(self, *args):
        """
        Build seat recommendations grouped by price tiers.

        Call patterns
        --------------
        1. get_seat_recommendations(trip_id, source_id, destination_id)
        – the three IDs may be str **or** int.

        2. get_seat_recommendations(seats_data_dict, ticket_count=?, user_preferences=?)
        – pass a seats‑payload you already fetched.

        Returns
        -------
        {
            "Premium":         {"window": [ {...}, ... ], "aisle": [ {...}, ... ]},
            "Reasonable":      {"window": [ {...}, ... ], "aisle": [ {...}, ... ]},
            "Budget-Friendly": {"window": [ {...}, ... ], "aisle": [ {...}, ... ]}
        }
        """
        # ---------------------------------------------------------------
        # Empty containers
        # ---------------------------------------------------------------
        categories = {
            "Premium":         {"window": [], "aisle": []},
            "Reasonable":      {"window": [], "aisle": []},
            "Budget-Friendly": {"window": [], "aisle": []},
        }

        try:
            # -----------------------------------------------------------
            # Detect which signature we received
            # -----------------------------------------------------------
            seat_data = None

            # ── pattern 1: (trip_id, source_id, destination_id) ─────────
            if len(args) == 3 and isinstance(args[0], (str, int)):
                trip_id, source_id, destination_id = map(str, args)
                print(
                    f"Fetching seat recommendations for trip {trip_id} "
                    f"from {source_id} to {destination_id}"
                )
                seat_data = await self.fetch_seats(trip_id, source_id, destination_id)

            # ── pattern 2: first argument is a seats‑payload dict ───────
            elif args and isinstance(args[0], dict):
                seat_data = args[0]

            # Unsupported signature – bail out early
            else:
                print(
                    f"get_seat_recommendations called with unsupported args: "
                    f"{[type(a) for a in args]}"
                )
                return categories

            # -----------------------------------------------------------
            # Validate / locate the "seats" list
            # -----------------------------------------------------------
            if not seat_data or not isinstance(seat_data, dict):
                print("Seat payload is empty or not a dict")
                return categories

            if not isinstance(seat_data.get("seats"), list):
                # Try to salvage if some other key contains the list
                for k, v in seat_data.items():
                    if isinstance(v, list) and k.lower().find("seat") >= 0:
                        seat_data["seats"] = v
                        break

            seats_array = seat_data.get("seats", [])
            if not isinstance(seats_array, list) or not seats_array:
                print("No seat list found in payload")
                return categories

            # -----------------------------------------------------------
            # Build list of available seats with number + price
            # -----------------------------------------------------------
            available_seats = []
            for seat in seats_array:
                if not isinstance(seat, dict):
                    continue

                occupied = (
                    seat.get("isOccupied", False)
                    or seat.get("occupied", False)
                    or seat.get("status", "").lower() == "occupied"
                    or seat.get("available", True) is False
                )
                status_ok = seat.get("availabilityStatus", "A") in {
                    "A", "M", "F", "Available", "available"
                }
                if occupied or not status_ok:
                    continue

                # Seat number (first field that parses to int)
                seat_num = None
                for key in ("seatName", "name", "number", "seatNumber", "seat_number", "id"):
                    if key in seat and seat[key]:
                        try:
                            seat_num = int(str(seat[key]).strip())
                            break
                        except (ValueError, TypeError):
                            continue
                if not seat_num or seat_num <= 0:
                    continue

                # Price (first field that converts to float)
                price_val = None
                for key in (
                    "totalFare", "fare", "price", "seatFare",
                    "seat_fare", "cost", "amount"
                ):
                    if key not in seat or not seat[key]:
                        continue
                    raw = seat[key]
                    try:
                        if isinstance(raw, dict):
                            # dig for numeric inside dict
                            for sub in ("amount", "value", "total", "fare"):
                                if sub in raw and raw[sub]:
                                    price_val = float(raw[sub])
                                    break
                        else:
                            price_val = float(raw)
                    except (ValueError, TypeError):
                        pass
                    if price_val is not None:
                        break
                if price_val is None:
                    continue

                available_seats.append(
                    {
                        "number": seat_num,
                        "price": price_val,
                        "seat_id": seat.get("id", seat_num),
                        "fare": seat.get("fare", price_val),
                    }
                )

            if not available_seats:
                return categories

            # -----------------------------------------------------------
            # Determine thresholds
            # -----------------------------------------------------------
            prices = [s["price"] for s in available_seats]
            min_price, max_price = min(prices), max(prices)
            price_range = max_price - min_price
            budget_thr  = min_price + price_range * 0.33 if price_range else min_price + 50
            premium_thr = max_price - price_range * 0.33 if price_range else max_price - 50

            # -----------------------------------------------------------
            # Split by window / aisle and price buckets
            # -----------------------------------------------------------
            window, aisle = [], []
            for seat in available_seats:
                if seat["number"] in self.window_seats:
                    window.append(seat)
                elif seat["number"] in self.aisle_seats:
                    aisle.append(seat)

            window.sort(key=lambda s: s["price"])
            aisle.sort(key=lambda s: s["price"])

            def bucket(seat):
                if seat["price"] <= budget_thr:
                    return "Budget-Friendly"
                if seat["price"] >= premium_thr:
                    return "Premium"
                return "Reasonable"

            used = set()
            for seat in window + aisle:
                if seat["number"] in used:
                    continue
                used.add(seat["number"])
                pos = "window" if seat in window else "aisle"
                categories[bucket(seat)][pos].append(seat)

            # -----------------------------------------------------------
            # Trim to three per list
            # -----------------------------------------------------------
            for cat in categories.values():
                cat["window"] = cat["window"][:3]
                cat["aisle"]  = cat["aisle"][:3]

            return categories

        except Exception as err:
            print(f"Error in get_seat_recommendations: {err}")
            import traceback
            traceback.print_exc()
            return categories

    
    def _build_direct_context_from_api_data(self, api_data, query, context):
        if not api_data:
            return None
        context_parts = []
        query_lower = query.lower()
        
        # Get user's requested direction
        user_source = context.get('user_requested_source')
        user_dest = context.get('user_requested_destination')
        use_user_direction = (user_source and user_dest)
        
        # Handle invalid route first
        if api_data.get("invalid_route"):
            invalid_route = api_data["invalid_route"]
            source = invalid_route["requested_source"].capitalize()
            destination = invalid_route["requested_destination"].capitalize()
            
            context_parts.append(f"\nINVALID ROUTE REQUESTED: {source} to {destination}")
            context_parts.append("We don't currently operate bus services on this route.")
            context_parts.append("\nWe offer the following routes:")
            for route in invalid_route["valid_routes"]:
                context_parts.append(f"• {route}")
            context_parts.append("\nWould you like to book a ticket for one of these routes instead?")
            context_parts.append("You can also check the Fresh Bus website/app for updates on other routes.")
            
            # Return immediately to avoid processing other data
            return "\n".join(context_parts)
        
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
        
        # Handle invalid route first
        if api_data and api_data.get("invalid_route"):
            invalid_route = api_data["invalid_route"]
            source = invalid_route["requested_source"].capitalize()
            destination = invalid_route["requested_destination"].capitalize()
            
            fallback_parts.append(f"I'm sorry, but we don't currently operate bus services from {source} to {destination}.")
            fallback_parts.append("\nWe offer the following routes:")
            for route in invalid_route["valid_routes"]:
                fallback_parts.append(f"• {route}")
            fallback_parts.append("\nWould you like to book a ticket for one of these routes instead?")
            fallback_parts.append("Or you can check the Fresh Bus website/app for updates on other routes.")
            
            return " ".join(fallback_parts)
        
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
        
        # Generic fallback based on what we have in context
        if context.get('user_requested_source') and context.get('user_requested_destination'):
            source = context['user_requested_source']
            destination = context['user_requested_destination']
            date = context.get('last_date', 'tomorrow')
            
            if isinstance(date, str) and "T" in date:
                # Format ISO date string
                try:
                    dt = datetime.fromisoformat(date.replace('Z', '+00:00'))
                    date = dt.strftime("%A, %B %d")
                except:
                    # Keep as is if parsing fails
                    date = date.split("T")[0]
            
            if api_data and api_data.get("trips"):
                trips = api_data["trips"]
                trip_count = len(trips)
                
                return f"I found {trip_count} bus{'es' if trip_count > 1 else ''} from {source} to {destination} for {date}. You can view the details and choose your preferred bus."
            else:
                return f"I'm searching for buses from {source} to {destination} for {date}. Please provide any additional preferences you might have for your journey."
        
        # Generic fallback response
        fallback_parts.append("I'm currently having trouble processing your request. Here's what I understand:")
        
        if context.get('user_requested_source') and context.get('user_requested_destination'):
            fallback_parts.append(f"\nYou want to travel from {context['user_requested_source']} to {context['user_requested_destination']}.")
        
        if context.get('last_date'):
            fallback_parts.append(f"Date of travel: {context['last_date']}")
            
        fallback_parts.append("\nPlease try again with a simple request like 'Show buses from [source] to [destination]' or 'Book a ticket to [destination]'.")
        
        return " ".join(fallback_parts)
    
    async def fetch_all_boarding_dropping_points(self, trip_id, source_id, destination_id=None):
        """Fetch all boarding and dropping points for a trip, filtered by destination"""
        if not self.http_session:
            await self.init_http_session()
        
        # Initialize results
        result = {
            "boarding_points": [],
            "dropping_points": []
        }
        
        # Fetch boarding points
        try:
            boarding_url = f"{self.BASE_URL}/trips/{trip_id}/boardings/{source_id}"
            print(f"Fetching all boarding points: {boarding_url}")
            
            async with self.http_session.get(boarding_url) as response:
                if response.status == 200:
                    boarding_data = await response.json()
                    # Format boarding points into a simplified structure
                    for point in boarding_data:
                        bp_info = point.get('boardingPoint', {})
                        formatted_point = {
                            "name": bp_info.get('name', 'Unknown'),
                            "landmark": bp_info.get('landmark', ''),
                            "time": point.get('time', ''),
                            "latitude": bp_info.get('latitude'),
                            "longitude": bp_info.get('longitude'),
                            "address": bp_info.get('address', ''),
                            "stationId": bp_info.get('stationId')  # Add station ID for filtering
                        }
                        result["boarding_points"].append(formatted_point)
                else:
                    print(f"Failed to fetch boarding points: Status {response.status}")
        except Exception as e:
            print(f"Error fetching boarding points: {e}")
        
        # Fetch dropping points with destination filtering
        try:
            dropping_url = f"{self.BASE_URL}/trips/{trip_id}/droppings"
            print(f"Fetching all dropping points: {dropping_url}")
            
            async with self.http_session.get(dropping_url) as response:
                if response.status == 200:
                    dropping_data = await response.json()
                    
                    # Format dropping points into a simplified structure
                    for point in dropping_data:
                        dp_info = point.get('droppingPoint', {})
                        
                        # Only include dropping points that match the destination station ID if provided
                        if destination_id and dp_info.get('stationId') != int(destination_id):
                            continue
                            
                        formatted_point = {
                            "name": dp_info.get('name', 'Unknown'),
                            "landmark": dp_info.get('landmark', ''),
                            "time": point.get('time', ''),
                            "latitude": dp_info.get('latitude'),
                            "longitude": dp_info.get('longitude'),
                            "address": dp_info.get('address', ''),
                            "stationId": dp_info.get('stationId')  # Add station ID for filtering
                        }
                        result["dropping_points"].append(formatted_point)
                    
                    if destination_id:
                        print(f"Filtered dropping points for destination {destination_id}: {len(result['dropping_points'])} points")
                else:
                    print(f"Failed to fetch dropping points: Status {response.status}")
        except Exception as e:
            print(f"Error fetching dropping points: {e}")
        
        return result
    
    async def generate_response(self, query, session_id=None, user_gender=None, user_location=None, detected_language=None, provider=None, model=None, **kwargs):
        """Generate a response to a user query"""
        print(f"\n--- New Query: '{query}' ---")
        
        # Initialize refresh_token to None at the beginning
        refresh_token = None
        
        # Create a valid session first - fixed to handle None properly
        try:
            session, session_id = self.get_or_create_session(session_id)
        except Exception as e:
            print(f"Error creating session: {e}")
            # Create a fallback session if there's an error
            session = {
                "messages": [],
                "last_updated": datetime.now(),
                "context": {}
            }
            session_id = str(uuid.uuid4())
            print(f"Created fallback session ID: {session_id}")
        
        # CRITICAL FIX: Make sure session is not None
        if not session:
            print("Session is None, creating a new one")
            session = {
                "messages": [],
                "last_updated": datetime.now(),
                "context": {}
            }
        
        # Make sure context exists
        if "context" not in session:
            print("Adding missing context to session")
            session["context"] = {}
        elif session["context"] is None:
            print("Context is None, initializing as empty dict")
            session["context"] = {}
        
        context = session["context"]
        
        # Get user mobile safely - this was causing the error
        user_mobile = None
        if isinstance(context.get('user'), dict):
            user_mobile = context['user'].get('mobile')
        
        # Print session info for debugging
        print(f"Session ID: {session_id}")
        if session_id in self.sessions:
            print(f"Current session messages count: {len(session.get('messages', []))}")
            messages_to_show = min(5, len(session.get('messages', [])))
            for i, msg in enumerate(session.get('messages', [])[-messages_to_show:]):  # Show last 5 messages
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    print(f"Message {i}: {msg['role']} - {msg['content'][:30]}...")
        
        # Determine if this is a simple query that shouldn't use fallback
        is_simple_query = self.is_simple_query(query)
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
        access_token = None
        user_id = None
        
        # Verify token validity and refresh if needed
        # ——————————————————————————————
        # VERIFY / REFRESH TOKENS
        # ——————————————————————————————

        # 1. If request contained no refresh_token, fall back to session
        session_auth = session.get("context", {}).get("auth", {}) or {}
        if not refresh_token and session_auth.get("refresh_token"):
            refresh_token = session_auth["refresh_token"]
            print(f"Found refresh token in session: {refresh_token[:10]}...")

        token_valid = False
        if access_token:
            try:
                # Pass BOTH tokens into our helper now
                profile_data = await fresh_bus_assistant.fetch_user_profile(
                    access_token,
                    refresh_token
                )
                if profile_data:
                    print("✅ Token verification successful!")
                    token_valid = True

                    # Populate user context
                    if not user_id and profile_data.get("id"):
                        user_id = profile_data["id"]
                    session["context"]["user"] = {
                        "id": profile_data.get("id"),
                        "name": profile_data.get("name"),
                        "mobile": profile_data.get("mobile"),
                        "email": profile_data.get("email")
                    }
                else:
                    print("❌ Token verification failed, will attempt refresh")
                    token_valid = False
            except Exception as e:
                print(f"Error verifying token: {e}")
                token_valid = False

        # If token is invalid and we have a refresh token, try to refresh
        if not token_valid and refresh_token:
            print("Token is invalid. Attempting to refresh...")
            try:
                # Call the refresh token API
                refresh_url = f"{fresh_bus_assistant.BASE_URL}/auth/refresh"
                
                async with fresh_bus_assistant.http_session.post(
                    refresh_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {refresh_token}"
                    },
                    json={"refresh_token": refresh_token}
                ) as refresh_response:
                    if refresh_response.status == 200:
                        refresh_data = await refresh_response.json()
                        new_access_token = refresh_data.get("access_token")
                        
                        if new_access_token:
                            print(f"Token refreshed successfully! New token: {new_access_token[:10]}...")
                            access_token = new_access_token
                            token_valid = True
                            
                            # Update session with new token
                            session["context"]["auth"]["token"] = new_access_token
                            if refresh_data.get("refresh_token"):
                                session["context"]["auth"]["refresh_token"] = refresh_data.get("refresh_token")
                    else:
                        print(f"Failed to refresh token: {refresh_response.status}")
            except Exception as refresh_error:
                print(f"Error refreshing token: {refresh_error}")
                
        if token_valid and session.get("context", {}).get("auth", {}).get("token"):
            user_authenticated = True
            user_data = context['user']
            user_mobile = user_data.get('mobile')
            user_id = user_data.get('id')  # Get user ID from context if available
            access_token = context['auth'].get('token')
            print(f"Processing request for authenticated user: {user_mobile} (ID: {user_id})")
        
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
        if "messages" not in session or session["messages"] is None:
            session["messages"] = []
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
        # Handle bus tracking requests
        if is_tracking_request:
            print("Bus tracking request detected")
            
            # Only proceed with authenticated users for tracking
            if user_authenticated and access_token:
                # Fetch user tickets 
                tickets = await self.fetch_user_tickets(access_token)
                
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
                            # Get tracking URL
                            tracking_url = active_ticket.get("trackingUrlNew")
                            
                            # Format as JSON for consistent frontend display
                            api_data["tracking"] = {
                                "type": "active_ticket",
                                "ticket": active_ticket,
                                "tripId": trip_id,
                                "from": active_ticket.get("source", {}).get("name", "Unknown"),
                                "to": active_ticket.get("destination", {}).get("name", "Unknown"),
                                "journeyDate": active_ticket.get("date", "Unknown"),
                                "trackingUrl": tracking_url,
                                "eta": active_ticket.get("eta_data", {})
                            }
                            
                            # Get ETA data if not already present
                            if not active_ticket.get("eta_data"):
                                eta_data = await self.fetch_eta_data(trip_id)
                                active_ticket["eta_data"] = eta_data
                                api_data["tracking"]["eta"] = eta_data
                            
                            # Create a direct tracking response
                            source_name = active_ticket.get("source", {}).get("name", "Unknown")
                            dest_name = active_ticket.get("destination", {}).get("name", "Unknown")
                            journey_date = active_ticket.get("date", "Unknown")
                            
                            direct_response = f"Your bus from {source_name} to {dest_name} is currently en route.\n\n"
                            direct_response += f"You can track your bus in real-time here: {tracking_url}\n\n"
                            
                            # Add ETA information if available
                            if eta_data and eta_data.get("message") != "This trip has ended":
                                if eta_data.get("eta"):
                                    direct_response += f"Estimated arrival time: {eta_data.get('eta')}\n"
                                if eta_data.get("distance"):
                                    direct_response += f"Current distance: {eta_data.get('distance')}\n"
                            elif eta_data and eta_data.get("message") == "This trip has ended":
                                direct_response = f"Your journey from {source_name} to {dest_name} has been completed.\n\n"
                                direct_response += "The bus has already reached its destination. Thank you for traveling with Fresh Bus!"
                            else:
                                direct_response += "Real-time ETA information will be available when your bus starts its journey."
                            
                            # Save the direct response
                            session['messages'].append({"role": "assistant", "content": direct_response})
                            
                            # Try to save to Redis with user ID
                            try:
                                user_mobile = None
                                if context.get('user') and isinstance(context.get('user'), dict):
                                    user_mobile = context['user'].get('mobile')
                                
                                conversation_id = conversation_manager.save_conversation(
                                    session_id, 
                                    session['messages'],
                                    user_id=user_mobile
                                )
                                print(f"Saved tracking response to Redis: {conversation_id}, user mobile: {user_mobile}")
                            except Exception as redis_err:
                                print(f"Failed to save to Redis: {redis_err}")
                                conversation_id = None
                            
                            # Return the direct response
                            yield json.dumps({"text": direct_response, "done": False})
                            yield json.dumps({
                                "text": "", 
                                "done": True, 
                                "session_id": session_id,
                                "language_style": context.get('language', 'english'),
                                "conversation_id": conversation_id,
                                "json_data": api_data
                            })
                            return
                    
                    # If no active but completed tickets, show most recent
                    elif processed_tickets["completed"]:
                        completed_ticket = processed_tickets["completed"][0]
                        source_name = completed_ticket.get("source", {}).get("name", "Unknown")
                        dest_name = completed_ticket.get("destination", {}).get("name", "Unknown")
                        journey_date_str = completed_ticket.get("date", "Unknown")
                        
                        direct_response = f"You don't have any active trips right now.\n\n"
                        direct_response += f"Your most recent completed journey was from {source_name} to {dest_name}"
                        
                        try:
                            journey_date = datetime.fromisoformat(journey_date_str.replace('Z', '+00:00'))
                            formatted_date = journey_date.strftime("%d %b %Y")
                            direct_response += f" on {formatted_date}."
                        except:
                            direct_response += "."
                        
                        # Save the direct response
                        session['messages'].append({"role": "assistant", "content": direct_response})
                        
                        # Try to save to Redis with user ID
                        try:
                            user_mobile = None
                            if context.get('user') and isinstance(context.get('user'), dict):
                                user_mobile = context['user'].get('mobile')
                            
                            conversation_id = conversation_manager.save_conversation(
                                session_id, 
                                session['messages'],
                                user_id=user_mobile
                            )
                            print(f"Saved completed journey response to Redis: {conversation_id}, user mobile: {user_mobile}")
                        except Exception as redis_err:
                            print(f"Failed to save to Redis: {redis_err}")
                            conversation_id = None
                        
                        yield json.dumps({"text": direct_response, "done": False})
                        yield json.dumps({
                            "text": "", 
                            "done": True, 
                            "session_id": session_id,
                            "language_style": context.get('language', 'english'),
                            "conversation_id": conversation_id,
                            "json_data": api_data
                        })
                        return
                    
                    # If only future tickets, show earliest one
                    elif processed_tickets["future"]:
                        future_ticket = processed_tickets["future"][0]
                        source_name = future_ticket.get("source", {}).get("name", "Unknown")
                        dest_name = future_ticket.get("destination", {}).get("name", "Unknown")
                        journey_date_str = future_ticket.get("date", "Unknown")
                        
                        direct_response = f"You don't have any active trips right now.\n\n"
                        direct_response += f"Your next upcoming journey is from {source_name} to {dest_name}"
                        
                        try:
                            journey_date = datetime.fromisoformat(journey_date_str.replace('Z', '+00:00'))
                            formatted_date = journey_date.strftime("%d %b %Y at %I:%M %p")
                            direct_response += f" on {formatted_date}."
                        except:
                            direct_response += "."
                        
                        direct_response += "\n\nYou'll be able to track your bus once your journey begins."
                        
                        # Save the direct response
                        session['messages'].append({"role": "assistant", "content": direct_response})
                        
                        # Try to save to Redis with user ID
                        try:
                            user_mobile = None
                            if context.get('user') and isinstance(context.get('user'), dict):
                                user_mobile = context['user'].get('mobile')
                            
                            conversation_id = conversation_manager.save_conversation(
                                session_id, 
                                session['messages'],
                                user_id=user_mobile
                            )
                            print(f"Saved future journey response to Redis: {conversation_id}, user mobile: {user_mobile}")
                        except Exception as redis_err:
                            print(f"Failed to save to Redis: {redis_err}")
                            conversation_id = None
                        
                        yield json.dumps({"text": direct_response, "done": False})
                        yield json.dumps({
                            "text": "", 
                            "done": True, 
                            "session_id": session_id,
                            "language_style": context.get('language', 'english'),
                            "conversation_id": conversation_id,
                            "json_data": api_data
                        })
                        return
                    
                    # No tickets found
                    else:
                        direct_response = "You don't have any active or upcoming tickets right now.\n\n"
                        direct_response += "When you book a ticket with Fresh Bus, you'll be able to track your bus in real-time by asking 'Where is my bus?'"
                        
                        # Save the direct response
                        session['messages'].append({"role": "assistant", "content": direct_response})
                        
                        # Try to save to Redis with user ID
                        try:
                            user_mobile = None
                            if context.get('user') and isinstance(context.get('user'), dict):
                                user_mobile = context['user'].get('mobile')
                            
                            conversation_id = conversation_manager.save_conversation(
                                session_id, 
                                session['messages'],
                                user_id=user_mobile
                            )
                            print(f"Saved no tickets response to Redis: {conversation_id}, user mobile: {user_mobile}")
                        except Exception as redis_err:
                            print(f"Failed to save to Redis: {redis_err}")
                            conversation_id = None
                        
                        yield json.dumps({"text": direct_response, "done": False})
                        yield json.dumps({
                            "text": "", 
                            "done": True, 
                            "session_id": session_id,
                            "language_style": context.get('language', 'english'),
                            "conversation_id": conversation_id,
                            "json_data": api_data
                        })
                        return
                else:
                    # No tickets found
                    direct_response = "I couldn't find any tickets associated with your account.\n\n"
                    direct_response += "When you book a ticket with Fresh Bus, you'll be able to track your bus in real-time by asking 'Where is my bus?'"
                    
                    # Save the direct response
                    session['messages'].append({"role": "assistant", "content": direct_response})
                    
                    # Try to save to Redis with user ID
                    try:
                        user_mobile = None
                        if context.get('user') and isinstance(context.get('user'), dict):
                            user_mobile = context['user'].get('mobile')
                        
                        conversation_id = conversation_manager.save_conversation(
                            session_id, 
                            session['messages'],
                            user_id=user_mobile
                        )
                        print(f"Saved no tickets found response to Redis: {conversation_id}, user mobile: {user_mobile}")
                    except Exception as redis_err:
                        print(f"Failed to save to Redis: {redis_err}")
                        conversation_id = None
                    
                    yield json.dumps({"text": direct_response, "done": False})
                    yield json.dumps({
                        "text": "", 
                        "done": True, 
                        "session_id": session_id,
                        "language_style": context.get('language', 'english'),
                        "conversation_id": conversation_id,
                        "json_data": api_data
                    })
                    return
            else:
                # User not authenticated - create an authentication prompt
                direct_response = "Please log in to track your bus. I need access to your ticket information to provide real-time tracking updates."
                
                # Save the direct response
                session['messages'].append({"role": "assistant", "content": direct_response})
                
                # Try to save to Redis, though likely no user ID
                try:
                    conversation_id = conversation_manager.save_conversation(
                        session_id, 
                        session['messages'],
                        user_id=None
                    )
                    print(f"Saved auth prompt to Redis: {conversation_id}")
                except Exception as redis_err:
                    print(f"Failed to save to Redis: {redis_err}")
                    conversation_id = None
                
                yield json.dumps({"text": direct_response, "done": False})
                yield json.dumps({
                    "text": "", 
                    "done": True, 
                    "session_id": session_id,
                    "language_style": context.get('language', 'english'),
                    "conversation_id": conversation_id,
                    "json_data": {
                        "tracking": {
                            "type": "not_authenticated",
                            "message": "Please log in to track your bus"
                        }
                    }
                })
                return
        
        # Check if this is a bus listing request and we have trip data
        is_bus_listing_request = any(kw in query.lower() for kw in ["book", "find", "search", "trip", "bus", "ticket"])
        
        # If we have source and destination, try to fetch trips
        if context.get('user_requested_source') and context.get('user_requested_destination'):
            source = context['user_requested_source']
            destination = context['user_requested_destination']
            travel_date = context.get('last_date', datetime.now().strftime("%Y-%m-%d"))
            
            # Convert source and destination names to IDs
            source_id = self.stations.get(source.lower(), source)
            destination_id = self.stations.get(destination.lower(), destination)
            
            print(f"Fetching trips for route: {source} to {destination}")
            
            # Store source and destination IDs for later use
            context['source_id'] = source_id
            context['destination_id'] = destination_id
            
            # Only fetch trips if we have valid IDs
            if source_id and destination_id:
                print(f"Using source_id={source_id}, destination_id={destination_id}")
                
                # Check if the route is valid first
                if not self.is_valid_route(source, destination):
                    print(f"Invalid route detected: {source} to {destination}")
                    
                    # List valid routes for the response
                    valid_routes = [
                        "Hyderabad to Guntur",
                        "Hyderabad to Vijayawada",
                        "Vijayawada to Hyderabad",
                        "Guntur to Hyderabad",
                        "Bangalore to Tirupati",
                        "Bangalore to Chittoor",
                        "Tirupati to Bangalore",
                        "Chittoor to Bangalore"
                    ]
                    
                    # Build invalid route response
                    api_data["invalid_route"] = {
                        "requested_source": source,
                        "requested_destination": destination,
                        "valid_routes": valid_routes
                    }
                else:
                    # Route is valid, fetch trips
                    trips = await self.fetch_trips(source_id, destination_id, travel_date)
                    
                    if trips:
                        print(f"Found {len(trips)} trips for {source} to {destination}")
                        api_data["trips"] = self.deduplicate_trips(trips)
                        
                        # Check if user is authenticated to fetch preferences
                        if user_authenticated and user_mobile:
                            user_preferences = await self.fetch_user_preferences(user_mobile, access_token)
                            if user_preferences:
                                api_data["user_preferences"] = user_preferences
                                
                        # If there's only one trip, also fetch boarding, dropping points, and seats
                        if len(trips) == 1:
                            trip_id = trips[0]['tripid']
                            api_data["selected_bus"] = trip_id
                            
                            # Fetch boarding and dropping points
                            boarding_points = await self.fetch_boarding_points(trip_id, source_id)
                            if boarding_points:
                                api_data["boarding_points"] = boarding_points
                                
                                # If user has location, find nearest boarding points
                                if context.get('user_location'):
                                    nearest_points = self.get_nearest_boarding_points_info(
                                        boarding_points, 
                                        context['user_location'],
                                        max_points=3
                                    )
                                    if nearest_points:
                                        api_data["nearest_boarding_points"] = nearest_points
                                
                                # Suggest a boarding point based on context
                                suggested_boarding = self.suggest_boarding_point(
                                    boarding_points,
                                    context.get('user_location'),
                                    context.get('last_boarding_point')
                                )
                                if suggested_boarding:
                                    api_data["suggested_boarding"] = suggested_boarding
                                    
                            # Fetch dropping points filtered by destination if available
                            dropping_points = await self.fetch_dropping_points(trip_id, destination_id)
                            if dropping_points:
                                api_data["dropping_points"] = dropping_points
                                
                                # Suggest a dropping point based on context
                                suggested_dropping = self.suggest_dropping_point(
                                    dropping_points,
                                    context.get('last_dropping_point')
                                )
                                if suggested_dropping:
                                    api_data["suggested_dropping"] = suggested_dropping
                                    
                            # Fetch seats and get recommendations
                            seat_recommendations = await self.get_seat_recommendations(trip_id, source_id, destination_id)
                            if seat_recommendations:
                                api_data["recommendations"] = seat_recommendations
                        
                        # If user has selected a bus from multiple options
                        elif context.get('selected_bus') and len(trips) > 1:
                            selected_trip = next((t for t in trips if str(t['tripid']) == str(context['selected_bus'])), None)
                            if selected_trip:
                                trip_id = selected_trip['tripid']
                                
                                # Same fetching logic as above for the selected bus
                                boarding_points = await self.fetch_boarding_points(trip_id, source_id)
                                if boarding_points:
                                    api_data["boarding_points"] = boarding_points
                                    
                                    if context.get('user_location'):
                                        nearest_points = self.get_nearest_boarding_points_info(
                                            boarding_points, 
                                            context['user_location'],
                                            max_points=3
                                        )
                                        if nearest_points:
                                            api_data["nearest_boarding_points"] = nearest_points
                                    
                                    suggested_boarding = self.suggest_boarding_point(
                                        boarding_points,
                                        context.get('user_location'),
                                        context.get('last_boarding_point')
                                    )
                                    if suggested_boarding:
                                        api_data["suggested_boarding"] = suggested_boarding
                                        
                                dropping_points = await self.fetch_dropping_points(trip_id, destination_id)
                                if dropping_points:
                                    api_data["dropping_points"] = dropping_points
                                    
                                    suggested_dropping = self.suggest_dropping_point(
                                        dropping_points,
                                        context.get('last_dropping_point')
                                    )
                                    if suggested_dropping:
                                        api_data["suggested_dropping"] = suggested_dropping
                                        
                                seat_recommendations = await self.get_seat_recommendations(trip_id, source_id, destination_id)
                                if seat_recommendations:
                                    api_data["recommendations"] = seat_recommendations
                    else:
                        # No trips found for the route
                        api_data["no_trips_info"] = {
                            "source": source,
                            "destination": destination,
                            "date": travel_date,
                            "message": f"No buses found from {source} to {destination} for the selected date."
                        }
                        print(f"No trips found for {source} to {destination} on {travel_date}")
                        
                        # Check if we can find trips in the opposite direction
                        print(f"Checking reverse direction: {destination} to {source}")
                        reverse_source_id = destination_id
                        reverse_destination_id = source_id
                        reverse_trips = await self.fetch_trips(reverse_source_id, reverse_destination_id, travel_date)
                        
                        if reverse_trips:
                            print(f"Found {len(reverse_trips)} trips in the reverse direction")
                            api_data["direction_reversed"] = True
                            api_data["trips"] = self.deduplicate_trips(reverse_trips)
                            api_data["reverse_direction"] = {
                                "source": destination,
                                "destination": source
                            }

        # Handle invalid route with direct response
        if api_data.get("invalid_route"):
            invalid_route = api_data["invalid_route"]
            source = invalid_route["requested_source"].capitalize()
            destination = invalid_route["requested_destination"].capitalize()
            
            direct_response = f"I'm sorry, but we don't currently operate bus services from {source} to {destination}.\n\n"
            direct_response += "We offer the following routes:\n"
            for route in invalid_route["valid_routes"]:
                direct_response += f"• {route}\n"
            direct_response += "\nWould you like to book a ticket for one of these routes instead? "
            direct_response += "Or you can check the Fresh Bus website/app for updates on other routes."
            
            # Save the direct response
            session['messages'].append({"role": "assistant", "content": direct_response})
            
            # Try to save to Redis with user ID
            try:
                # Extract user ID from auth context, user_profile, or previously set value
                if not user_id and context.get('auth') and context.get('user'):
                    user_data = context['user']
                    user_id = user_data.get('id') or user_data.get('mobile')
                
                # Also check if we have ID in user_profile
                if not user_id and context.get('user_profile'):
                    user_id = context['user_profile'].get('id')
                    
                conversation_id = conversation_manager.save_conversation(
                    session_id, 
                    session['messages'],
                    user_id=user_mobile
                )
                print(f"Saved invalid route response to Redis: {conversation_id}, user: {user_id}")
            except Exception as redis_err:
                print(f"Failed to save to Redis: {redis_err}")
                conversation_id = None
            
            # Return the direct response instead of calling the AI model
            yield json.dumps({"text": direct_response, "done": False})
            yield json.dumps({
                "text": "", 
                "done": True, 
                "session_id": session_id, 
                "language_style": context.get('language', 'english'),
                "conversation_id": conversation_id
            })
            return
        
        # Only use direct response for real bus queries, not simple ones
        if is_bus_listing_request and api_data and api_data.get("trips") and not is_simple_query:
            # Use direct response generation for bus listings
            print("Using direct bus response generator to avoid hallucinations")
            
            # Use await with the async functions
            direct_response = await self.generate_direct_bus_response(api_data, context, return_json=False)
            
            # Get JSON response as well
            json_response = await self.generate_direct_bus_response(api_data, context, return_json=True)
            
            # Save the direct response
            session['messages'].append({"role": "assistant", "content": direct_response})
            
            # Try to save to Redis with user ID
            try:
                # Extract user ID from auth context, user_profile, or previously set value
                if not user_id and context.get('auth') and context.get('user'):
                    user_data = context['user']
                    user_id = user_data.get('id') or user_data.get('mobile')
                
                # Also check if we have ID in user_profile
                if not user_id and context.get('user_profile'):
                    user_id = context['user_profile'].get('id')
                    
                conversation_id = conversation_manager.save_conversation(
                    session_id, 
                    session['messages'],
                    user_id=user_mobile
                )
                print(f"Saved direct response to Redis: {conversation_id}, user: {user_id}")
            except Exception as redis_err:
                print(f"Failed to save to Redis: {redis_err}")
                conversation_id = None
            
            # Add JSON data formatted as code block to the end of the response
            json_formatted = "\n\n```json\n" + json.dumps(json_response, indent=2) + "\n```"
            complete_response = direct_response + json_formatted
            
            # Return the direct response with JSON included
            yield json.dumps({"text": complete_response, "done": False})
            yield json.dumps({
                "text": "", 
                "done": True, 
                "session_id": session_id, 
                "language_style": context.get('language', 'english'),
                "conversation_id": conversation_id,
                "json_data": json_response  # Include JSON data separately
            })
            return
        
        # Continue with the rest of the method...
        # (Remaining code unchanged)
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
            system_message += f"\nAuthenticated user: {user_mobile}" + (f" (ID: {user_id})" if user_id else "")
            
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
        
        # Get complete message history for the session
        complete_message_history = session['messages']
        
        # If conversation gets too long, we can optionally limit it while still keeping more context
        max_history_length = 50  # Adjust this value based on your needs
        if len(complete_message_history) > max_history_length:
            # Keep first message (system context) and most recent messages
            complete_message_history = [
                complete_message_history[0],  # System message
                *complete_message_history[-(max_history_length-1):]  # Most recent messages
            ]
            # Keep first message for context and the most recent messages
            first_message = complete_message_history[0:1]
            recent_messages = complete_message_history[-(max_history_length-1):]
            complete_message_history = first_message + recent_messages
            
        print(f"Using {len(complete_message_history)} messages from conversation history")
        
        try:
            # Generate response using the AI provider abstraction with full history
            response_text = ""
            
            print(f"Sending to {self.ai_provider.provider_name} with system message length: {len(system_message)}")
            print(f"First 100 chars of system message: {system_message[:100]}...")
            
            async for chunk in self.ai_provider.generate_stream(
                prompt=query,
                system_message=system_message,
                messages=complete_message_history,  # Use full history instead of recent_messages
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
                        
                        # Save conversation to Redis with user ID if available
                        try:
                            # Use user_id from various sources in priority order
                            redis_user_id = None
                            
                            # 1. First try user_id we've already determined
                            if user_id:
                                redis_user_id = user_id
                            # 2. Try auth context
                            elif context.get('auth') and context.get('user'):
                                user_data = context['user']
                                redis_user_id = user_data.get('id') or user_data.get('mobile')
                            # 3. Try user profile
                            elif context.get('user_profile'):
                                redis_user_id = context['user_profile'].get('id')
                                
                            conversation_id = conversation_manager.save_conversation(
                                session_id, 
                                session['messages'],
                                user_id=user_mobile
                            )
                            if conversation_id:
                                print(f"Saved conversation to Redis with ID: {conversation_id}, user: {redis_user_id}")
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
                # Use await with the async functions
                direct_response = await self.generate_direct_bus_response(api_data, context, return_json=False)
                # Get JSON response too
                json_response = await self.generate_direct_bus_response(api_data, context, return_json=True)
                
                print("Using direct bus response generation due to AI API error")
                session['messages'].append({"role": "assistant", "content": direct_response})
                
                # Try to save to Redis with user ID
                try:
                    # Use user_id from various sources in priority order
                    redis_user_id = None
                    
                    # 1. First try user_id we've already determined
                    if user_id:
                        redis_user_id = user_id
                    # 2. Try auth context
                    elif context.get('auth') and context.get('user'):
                        user_data = context['user']
                        redis_user_id = user_data.get('id') or user_data.get('mobile')
                    # 3. Try user profile
                    elif context.get('user_profile'):
                        redis_user_id = context['user_profile'].get('id')
                            
                    conversation_id = conversation_manager.save_conversation(
                        session_id, 
                        session['messages'],
                        user_id=user_mobile
                    )
                    if conversation_id:
                        print(f"Saved direct response to Redis: {conversation_id}, user: {redis_user_id}")
                except Exception as redis_err:
                    print(f"Failed to save to Redis: {redis_err}")
                    conversation_id = None
                
                # Add JSON data formatted as code block to the end of the response
                json_formatted = "\n\n```json\n" + json.dumps(json_response, indent=2) + "\n```"
                complete_response = direct_response + json_formatted
                
                yield json.dumps({"text": complete_response, "done": False})
                yield json.dumps({
                    "text": "", 
                    "done": True, 
                    "session_id": session_id,
                    "language_style": context.get('language', 'english'),
                    "conversation_id": conversation_id,
                    "json_data": json_response  # Include JSON data separately
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
                
                # Try to save to Redis with user ID
                try:
                    # Use user_id from various sources in priority order
                    redis_user_id = None
                    
                    # 1. First try user_id we've already determined
                    if user_id:
                        redis_user_id = user_id
                    # 2. Try auth context
                    elif context.get('auth') and context.get('user'):
                        user_data = context['user']
                        redis_user_id = user_data.get('id') or user_data.get('mobile')
                    # 3. Try user profile
                    elif context.get('user_profile'):
                        redis_user_id = context['user_profile'].get('id')
                            
                    conversation_id = conversation_manager.save_conversation(
                        session_id, 
                        session['messages'],
                        user_id=user_mobile
                    )
                    if conversation_id:
                        print(f"Saved simple response to Redis: {conversation_id}, user: {redis_user_id}")
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
                
                # Try to save to Redis with user ID
                try:
                    # Use user_id from various sources in priority order
                    redis_user_id = None
                    
                    # 1. First try user_id we've already determined
                    if user_id:
                        redis_user_id = user_id
                    # 2. Try auth context
                    elif context.get('auth') and context.get('user'):
                        user_data = context['user']
                        redis_user_id = user_data.get('id') or user_data.get('mobile')
                    # 3. Try user profile
                    elif context.get('user_profile'):
                        redis_user_id = context['user_profile'].get('id')
                            
                    conversation_id = conversation_manager.save_conversation(
                        session_id, 
                        session['messages'],
                        user_id=user_mobile
                    )
                    if conversation_id:
                        print(f"Saved fallback conversation to Redis: {conversation_id}, user: {redis_user_id}")
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

# Add CORS middleware with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Authorization", "X-Access-Token", "access_token", "refresh_token", "Set-Cookie"]
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

#################################
# API Endpoints
#################################

def setup_middleware(app_instance):
    @app_instance.middleware("http")
    async def authenticate_user(request: Request, call_next):
        """Middleware to authenticate user and add user data to request"""
        # 1. Extract tokens
        auth_header = request.headers.get("Authorization", "")
        access_token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else None

        # Try cookies next
        if not access_token:
            access_token = request.cookies.get("access_token")

        # Refresh token from cookies or headers
        refresh_token = request.cookies.get("refresh_token") \
            or request.headers.get("X-Refresh-Token") \
            or request.headers.get("refresh_token")

        # 2. If we have an access token, fetch profile
        if access_token:
            try:
                profile = await fresh_bus_assistant.fetch_user_profile(
                    access_token,
                    refresh_token
                )
                if profile:
                    request.state.user = profile
                    request.state.token = access_token
            except Exception as e:
                print(f"Error authenticating user: {e}")

        # 3. Continue with the request
        response = await call_next(request)

        # Ensure CORS exposes our custom headers
        if any(h in response.headers for h in ("access_token", "refresh_token")):
            response.headers["Access-Control-Expose-Headers"] = "access_token, refresh_token"

        return response

# Then call this function with app
setup_middleware(app)

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

# --------------------------------------------------------------------
#  Ticket‑blocking endpoint (replace the whole function with this one)
# --------------------------------------------------------------------
# from fastapi import HTTPException   # already imported near top – keep only once
# from fastapi.responses import JSONResponse


import re
from fastapi import HTTPException
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Helper 1 – coerce anything that "looks like digits" to an int
# ---------------------------------------------------------------------------
def _to_int(value, field_name: str) -> int:
    """
    Convert *value* to int.

    • If it is already an int  → return it untouched.
    • If it is a string that contains digits anywhere
      (e.g. "32997‑boarding") → grab the first run of digits.
    • Otherwise raise 400.
    """
    if isinstance(value, int):
        return value

    if isinstance(value, str):
        m = re.search(r"\d+", value)
        if m:
            return int(m.group())

    raise HTTPException(
        status_code=400,
        detail=f"Bad {field_name} value: must contain digits"
    )


# ---------------------------------------------------------------------------
# Helper 2 – turn tokens like "32997‑10‑w" into the *real* seat.id (e.g. 698951)
# NEW IMPLEMENTATION: Since we don't have a reliable way to map from seat tokens
# to actual seat IDs, we'll just handle them properly in the frontend.
# ---------------------------------------------------------------------------
async def _resolve_seat_token(
    *,
    seat_token: str,
    trip_id: int,
    boarding_point_id: int,
    dropping_point_id: int,
    session,  # aiohttp.ClientSession
    base_url_customer: str
) -> int:
    """
    If *seat_token* is already all‑digits → just return int(token).

    For non-numeric tokens, we'll extract the seat ID using the first 
    pattern of digits we find.
    """
    # Handle direct numeric seat IDs
    if str(seat_token).isdigit():
        return int(seat_token)

    # For patterns like "32997-10-a", extract just the first numeric part
    m = re.search(r"\d+", str(seat_token))
    if m:
        # This will give us a basic numeric ID - better than nothing
        numeric_id = int(m.group())
        print(f"Extracted numeric part {numeric_id} from seat token {seat_token}")
        
        # If this is the trip ID format (like in "32997-10-a"), try to get a real seat ID
        if "-" in str(seat_token):
            parts = str(seat_token).split("-")
            if len(parts) >= 2 and parts[0].isdigit():
                # In this case, we need to get the real seat IDs from the API
                # This requires that proper boarding_point_id and dropping_point_id be provided
                # Format the URL for the seats endpoint
                url = (
                    f"{base_url_customer}/trips/{trip_id}/seats"
                    f"?source_id={boarding_point_id}"
                    f"&destination_id={dropping_point_id}"
                )
                print(f"Fetching seat map from: {url}")
                
                try:
                    async with session.get(url) as r:
                        if r.status != 200:
                            text = await r.text()
                            print(f"Error fetching seat map: {r.status} - {text}")
                            # Fall back to numeric_id instead of failing completely
                            return numeric_id
                        
                        seat_data = await r.json()
                        print(f"Received seat data keys: {seat_data.keys() if isinstance(seat_data, dict) else 'list'}")
                        
                        # Extract seats from response (could be under 'seats' key or direct list)
                        seats = []
                        if isinstance(seat_data, dict) and "seats" in seat_data:
                            seats = seat_data["seats"]
                        elif isinstance(seat_data, list):
                            seats = seat_data
                            
                        seat_name = parts[1]  # The second part like "10" in "32997-10-a"
                        
                        # Try to find matching seat
                        for seat in seats:
                            seat_name_value = str(seat.get("seatName", seat.get("name", "")))
                            if seat_name_value == seat_name:
                                seat_id = seat.get("id")
                                if seat_id:
                                    print(f"Found matching seat: {seat_name} → ID: {seat_id}")
                                    return int(seat_id)
                        
                        print(f"No matching seat found for {seat_name}, falling back to {numeric_id}")
                except Exception as e:
                    print(f"Error in seat resolution: {e}")
                    # Fall back to numeric_id
        
        return numeric_id

    raise HTTPException(
        status_code=400,
        detail=f"Bad seat_id value: {seat_token} - must contain digits"
    )


# ---------------------------------------------------------------------------
# Helper function to resolve boarding/dropping point IDs 
# ---------------------------------------------------------------------------
async def _resolve_point_id(
    *,
    point_token: str,
    trip_id: int,
    is_boarding: bool,
    session,  # aiohttp.ClientSession
    base_url_customer: str
) -> int:
    """
    Resolve boarding or dropping point tokens to actual IDs.
    For patterns like "32997-boarding", extract just the numeric part.
    """
    # If it's already a number, just return it
    if str(point_token).isdigit():
        return int(point_token)
    
    # Try to extract numeric part
    m = re.search(r"\d+", str(point_token))
    if m:
        # If we have a format like "32997-boarding", try to get the real point ID
        # by fetching from the API
        endpoint = "boardings" if is_boarding else "droppings"
        url = f"{base_url_customer}/trips/{trip_id}/{endpoint}"
        print(f"Fetching {'boarding' if is_boarding else 'dropping'} points from: {url}")
        
        try:
            async with session.get(url) as r:
                if r.status != 200:
                    text = await r.text()
                    print(f"Error fetching points: {r.status} - {text}")
                    # Fall back to numeric part of token
                    return int(m.group())
                
                points_data = await r.json()
                
                # For simple case, return first point's ID
                if isinstance(points_data, list) and points_data:
                    point_id = points_data[0].get("id")
                    if point_id:
                        print(f"Using first {'boarding' if is_boarding else 'dropping'} point: {point_id}")
                        return int(point_id)
        except Exception as e:
            print(f"Error resolving point ID: {e}")
    
        # Fall back to numeric part of token
        return int(m.group())
    
    raise HTTPException(
        status_code=400,
        detail=f"Bad {'boarding' if is_boarding else 'dropping'}_point_id value: {point_token}"
    )


# ---------------------------------------------------------------------------
# The **fixed** /tickets/block route
# ---------------------------------------------------------------------------
# ──────────────────────────────────────────────────────────────────────────────
#  /tickets/block  – CREATE A PAYMENT ORDER
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/tickets/block")
async def block_ticket(request: Request):
    """Create a payment order for booking tickets."""
    try:
        body = await request.json()
        req_id = str(uuid.uuid4())[:8]
        print(f"[{req_id}] Received ticket block request → {body}")

        # Authentication check
        access = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not access:
            return JSONResponse(
                status_code=401, 
                content={"success": False, "message": "Authentication required"}
            )

        # Validate required fields
        required_fields = ["mobile", "email", "trip_id", "seat_map", 
                          "boarding_point_id", "dropping_point_id"]
        missing_fields = [field for field in required_fields if field not in body]
        if missing_fields:
            return JSONResponse(
                status_code=400, 
                content={"success": False, "message": f"Missing required fields: {', '.join(missing_fields)}"}
            )

        # Prepare the upstream request
        upstream = {
            "mobile": body["mobile"],
            "email": body["email"],
            "seat_map": body["seat_map"],
            "trip_id": body["trip_id"],
            "boarding_point_id": body["boarding_point_id"],
            "dropping_point_id": body["dropping_point_id"],
            "boarding_point_time": body.get("boarding_point_time"),
            "dropping_point_time": body.get("dropping_point_time"),
            "total_collect_amount": body.get("total_collect_amount", 0),
            "main_category": body.get("main_category", 1),
            "freshcard": body.get("freshcard", False),
            "freshcardId": body.get("freshcardId") if body.get("freshcard", False) else 1,
            "return_url": body.get("return_url")
        }
        
        # Remove None values 
        upstream = {k: v for k, v in upstream.items() if v is not None}

        print(f"[{req_id}] Prepared payload → {upstream}")

        # Forward to Fresh Bus
        headers = {
            "Authorization": f"Bearer {access}",
            "Content-Type": "application/json",
        }
        book_url = f"{fresh_bus_assistant.BASE_URL_CUSTOMER}/tickets/block"
        
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()
            
        async with fresh_bus_assistant.http_session.post(
            book_url, 
            json=upstream,
            headers=headers
        ) as response:
            status_code = response.status
            try:
                resp_json = await response.json()
            except:
                resp_text = await response.text()
                resp_json = {"error": f"Failed to parse response: {resp_text[:200]}..."}
                
            print(f"[{req_id}] Upstream status {status_code}, body: {resp_json}")

            if status_code in (200, 201):
                return JSONResponse(
                    status_code=status_code, 
                    content={
                        "success": True,
                        "payment_details": resp_json
                    }
                )
            return JSONResponse(
                status_code=status_code,
                content={
                    "success": False, 
                    "message": resp_json.get("message", "Booking failed"),
                    "details": resp_json
                }
            )

    except Exception as e:
        print(f"[{req_id}] UNHANDLED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500, 
            content={"success": False, "message": str(e)}
        )


# In app.py

@app.route("/profile", methods=["GET"])
async def get_user_profile(request: Request):
    """Fetch user profile from the customer service"""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        return JSONResponse(
            status_code=401, 
            content={"error": "No authorization token provided"}
        )
    
    try:
        # Forward request to customer service
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()
        
        base_url_customer = fresh_bus_assistant.BASE_URL_CUSTOMER
        
        async with fresh_bus_assistant.http_session.get(
            f"{base_url_customer}/profile",
            headers={"Authorization": f"Bearer {token}"}
        ) as response:
            if response.status == 200:
                profile_data = await response.json()
                
                # Extract mobile number from profile
                mobile = profile_data.get('mobile')
                
                # Print user login details in terminal with clear formatting
                print("\n" + "="*50)
                print("                USER LOGIN DETECTED                ")
                print("="*50)
                print(f"Mobile:     {mobile}")
                print(f"Name:       {profile_data.get('name', 'N/A')}")
                print(f"Email:      {profile_data.get('email', 'N/A')}")
                print(f"Gender:     {profile_data.get('gender', 'N/A')}")
                print(f"Login Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*50)
                
                # Get or create the session
                session_id = request.query_params.get("session_id")
                print(f"Session ID from query params: {session_id}")
                
                session, session_id = fresh_bus_assistant.get_or_create_session(session_id)
                
                # Update the session context with user profile
                session['context']['user_profile'] = profile_data
                
                # CRITICAL: Store user data with mobile in user context
                session['context']['user'] = {
                    'mobile': mobile,  # Key change: store mobile instead of ID
                    'name': profile_data.get('name'),
                    'email': profile_data.get('email'),
                    'gender': profile_data.get('gender')
                }
                
                # Save context to Redis
                conversation_manager.save_context(session_id, session['context'])
                
                print(f"⚠️ Session ID: {session_id}")
                print(f"⚠️ User mobile {mobile} stored in session context")
                print(f"⚠️ Context saved to Redis")
                print("="*50 + "\n")
                
                # Additionally, store user profile separately in Redis (using mobile as key)
                conversation_manager.update_user_profile(mobile, profile_data, ttl_seconds=86400*7)  # 7 days TTL
                
                # Add session ID to response for frontend
                response_data = profile_data.copy()
                response_data["session_id"] = session_id
                
                return JSONResponse(content=response_data)
            else:
                error_text = await response.text()
                print("\n=== Login Failed ===")
                print(f"Error: {error_text}")
                print("=" * 25)
                return JSONResponse(
                    status_code=response.status,
                    content={"error": f"Failed to fetch profile: {error_text}"}
                )
    except Exception as e:
        print("\n=== Login Error ===")
        print(f"Error: {str(e)}")
        print("=" * 25)
        import traceback
        traceback.print_exc()  # Print the full stack trace
        return JSONResponse(
            status_code=500,
            content={"error": f"Error fetching profile: {str(e)}"}
        )

@app.get("/user/conversations")
async def get_user_conversations(
    request: Request,
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0)
):
    """Get conversations for the authenticated user"""
    try:
        # 1. Access token
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"error": "Authentication required"})
        access_token = auth.replace("Bearer ", "")

        # 2. Refresh token
        refresh_token = (
            request.cookies.get("refresh_token")
            or request.headers.get("X-Refresh-Token")
            or request.headers.get("refresh_token")
        )

        # 3. Ensure HTTP session
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()

        # 4. Fetch profile (now hits BASE_URL_CUSTOMER/profile)
        profile = await fresh_bus_assistant.fetch_user_profile(access_token, refresh_token)
        if not profile:
            return JSONResponse(status_code=404, content={"error": "User profile not found"})

        # 5. Determine user_id
        user_id = str(profile.get("id") or profile.get("mobile") or "")
        if not user_id:
            return JSONResponse(status_code=400, content={"error": "User ID not found in profile"})

        # 6. Fetch and return
        convos = conversation_manager.get_conversations_by_user(user_id, limit, offset)
        return JSONResponse(content=convos)

    except Exception as e:
        print(f"Error getting user conversations: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/user/conversations")
async def delete_user_conversations(request: Request):
    """Delete all conversations for the authenticated user"""
    try:
        # 1. Access token
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"error": "Authentication required"})
        access_token = auth.replace("Bearer ", "")

        # 2. Refresh token
        refresh_token = (
            request.cookies.get("refresh_token")
            or request.headers.get("X-Refresh-Token")
            or request.headers.get("refresh_token")
        )

        # 3. Ensure HTTP session
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()

        # 4. Fetch profile to get user ID
        profile = await fresh_bus_assistant.fetch_user_profile(access_token, refresh_token)
        if not profile:
            return JSONResponse(status_code=404, content={"error": "User profile not found"})

        user_id = str(profile.get("id") or profile.get("mobile") or "")
        if not user_id:
            return JSONResponse(status_code=400, content={"error": "User ID not found in profile"})

        # 5. Delete and respond
        count = conversation_manager.delete_user_conversations(user_id)
        return JSONResponse(content={
            "status": "success",
            "message": f"Deleted {count} conversations"
        })

    except Exception as e:
        print(f"Error deleting user conversations: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})



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
        
        access_token = auth_header.replace("Bearer ", "")
        
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()
        
        # Call the feedback API endpoint
        url = f"{fresh_bus_assistant.BASE_URL}/tickets/feedbackQuestions"
        print(f"Fetching feedback questions for trip {trip_id}: {url}")
        
        headers = {
            "Authorization": f"Bearer {access_token}",
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
        
        access_token = auth_header.replace("Bearer ", "")
        
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()
        
        # Call the tickets API endpoint
        url = f"{fresh_bus_assistant.BASE_URL}/tickets"
        print(f"Fetching user tickets: {url}")
        
        headers = {
            "Authorization": f"Bearer {access_token}",
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
        
        access_token = auth_header.replace("Bearer ", "")
        
        # Fetch tickets
        tickets = await fresh_bus_assistant.fetch_user_tickets(access_token, force_refresh=True)
        
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
        
        access_token = auth_header.replace("Bearer ", "")
        
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()
        
        # Call the feedback API endpoint
        url = f"{fresh_bus_assistant.BASE_URL}/tickets/feedbackQuestions"
        print(f"Fetching feedback questions: {url}")
        
        headers = {
            "Authorization": f"Bearer {access_token}",
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
                access_token = auth_header.replace("Bearer ", "")
                
                # Fetch NPS data for completed trips
                nps_data = await fresh_bus_assistant.fetch_nps_data(trip_id, access_token)
                
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
    """Get user profile data from backend api"""
    # Debug request headers
    print("REQUEST HEADERS:")
    for header_name, header_value in request.headers.items():
        print(f"  {header_name}: {header_value}")
    
    # Try to get auth token from Authorization header first
    auth_token = None
    auth_header = request.headers.get("Authorization")
    
    if auth_header and auth_header.startswith("Bearer "):
        auth_token = auth_header.replace("Bearer ", "")
    
    # If not found in header, try cookies
    if not auth_token:
        cookies = request.cookies
        auth_token = cookies.get("access_token")
    
    if not auth_token:
        print("Authorization token missing")
        return JSONResponse(
            status_code=401,
            content={"error": "Authorization token required"}
        )
    
    print(f"Extracted token: {auth_token[:10]}...")  # Only print first 10 chars
    
    session_id = request.query_params.get("session_id")
    print(f"Session ID from query params: {session_id}")
    
    try:
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()
        
        # Forward the request to the backend
        profile_url = f"{fresh_bus_assistant.BASE_URL_CUSTOMER}/profile"
        
        print(f"Forwarding request to: {profile_url}")
        print(f"Using headers: Authorization: Bearer {auth_token[:10]}...")
        
        async with fresh_bus_assistant.http_session.get(
            profile_url,
            headers={"Authorization": f"Bearer {auth_token}"}
        ) as response:
            if response.status == 200:
                profile_data = await response.json()
                print("Successfully fetched user profile")
                print("\n==================================================")
                print("                USER LOGIN DETECTED                ")
                print("==================================================")
                print(f"Mobile:     {profile_data.get('mobile', 'Unknown')}")
                print(f"Name:       {profile_data.get('name', 'Unknown')}")
                print(f"Email:      {profile_data.get('email', 'Unknown')}")
                print(f"Gender:     {profile_data.get('gender', 'Unknown')}")
                print(f"Login Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("==================================================")
                
                # Store profile in session if session_id provided
                if session_id:
                    try:
                        session, _ = fresh_bus_assistant.get_or_create_session(session_id)
                        
                        if "context" not in session:
                            session["context"] = {}
                            
                        session["context"]["user_profile"] = profile_data
                        
                        # Extract and store basic user info too
                        mobile = profile_data.get("mobile")
                        if mobile:
                            session["context"]["user"] = {
                                "mobile": mobile,
                                "name": profile_data.get("name", ""),
                                "email": profile_data.get("email", ""),
                                "gender": profile_data.get("gender", "")
                            }
                            
                            # Store auth token
                            session["context"]["auth"] = {
                                "token": auth_token
                            }
                            
                            # Save to Redis - catching any errors
                            try:
                                # Use save_context from the conversation manager
                                key = f"{conversation_manager.context_prefix}{session_id}"
                                # Convert complex objects to strings for Redis
                                serialized_context = {}
                                for k, v in session["context"].items():
                                    if isinstance(v, dict):
                                        serialized_context[k] = json.dumps(v)
                                    elif isinstance(v, (list, tuple)):
                                        serialized_context[k] = json.dumps(v)
                                    else:
                                        serialized_context[k] = str(v)
                                
                                # Delete existing key first to avoid partial updates
                                conversation_manager.redis.delete(key)
                                
                                # Set all values at once
                                if serialized_context:
                                    conversation_manager.redis.hmset(key, serialized_context)
                                    
                                # Set expiration for context data (7 days)
                                conversation_manager.redis.expire(key, 7 * 86400)
                                
                                print(f"Successfully saved context to Redis for session {session_id}")
                            except Exception as redis_err:
                                print(f"Warning: Failed to save context to Redis: {redis_err}")
                                import traceback
                                traceback.print_exc()
                            
                            # Also store user profile separately for future reference
                            try:
                                # Store profile directly
                                profile_key = f"fresh_bus:user_profile:{mobile}"
                                profile_json = json.dumps(profile_data, default=str)
                                conversation_manager.redis.setex(profile_key, 7 * 86400, profile_json)
                                
                                print(f"Successfully saved user profile to Redis for mobile {mobile}")
                            except Exception as redis_err:
                                print(f"Warning: Failed to save user profile to Redis: {redis_err}")
                                import traceback
                                traceback.print_exc()
                    except Exception as session_err:
                        print(f"Warning: Error updating session with profile data: {session_err}")
                        print("\n=== Login Error ===")
                        print(f"Error: {session_err}")
                        print("=========================")
                        import traceback
                        traceback.print_exc()
                        
                # Add session_id to response
                response_data = profile_data.copy()
                if session_id:
                    response_data["session_id"] = session_id
                
                return JSONResponse(content=response_data)
            else:
                error_text = await response.text()
                print(f"Error fetching user profile: {response.status}")
                print(f"Error response: {error_text}")
                
                print("\n=== Login Failed ===")
                print(f"Error: {error_text}")
                print("=========================")
                
                return JSONResponse(
                    status_code=response.status,
                    content={"error": f"Failed to fetch profile: {error_text}"}
                )
    except Exception as e:
        print(f"Exception fetching user profile: {e}")
        
        print("\n=== Login Error ===")
        print(f"Error: {str(e)}")
        print("=========================")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={"error": f"Error fetching profile: {str(e)}"}
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
    """Verify OTP and authenticate user"""
    try:
        body = await request.json()
        mobile = body.get("mobile")
        otp = body.get("otp")
        device_id = body.get("deviceId", "web_client")
        session_id = body.get("session_id")
        
        if not mobile or not otp:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Mobile number and OTP are required"}
            )
        
        print(f"Verifying OTP for mobile: {mobile}, OTP: {otp}, Device ID: {device_id}")
        
        # Forward request to Fresh Bus API
        api_url = f"{fresh_bus_assistant.BASE_URL}/auth/verifyotp"
        print(f"Forwarding OTP verification to {api_url}")
        
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()
        
        # Create headers for the request
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Create the request body
        request_body = {
            "mobile": mobile,
            "otp": otp,
            "deviceId": device_id
        }
        
        # Make the request
        async with fresh_bus_assistant.http_session.post(
            api_url,
            json=request_body,
            headers=headers
        ) as response:
            # Log response info
            print(f"OTP verification response status: {response.status}")
            
            # Get the raw response text first
            response_text = await response.text()
            print(f"Raw response text: {response_text}")
            
            # Parse response data
            try:
                response_data = json.loads(response_text)
                print(f"Parsed JSON response: {json.dumps(response_data, indent=2)}")
            except json.JSONDecodeError:
                print(f"Failed to parse response as JSON")
                response_data = {"success": False, "message": "Invalid response from server"}
            
            # Extract token from response data
            access_token = response_data.get("access_token")
            refresh_token = response_data.get("refresh_token")
            
            # Check response headers for tokens
            auth_header = response.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                access_token = auth_header.replace("Bearer ", "")
                print(f"Found access token in Authorization header: {access_token[:10]}...")
            
            # Extract cookies from response
            cookies = response.cookies
            for cookie_name, cookie in cookies.items():
                print(f"Cookie: {cookie_name} = {cookie.value[:10]}..." if len(cookie.value) > 10 else f"Cookie: {cookie_name} = {cookie.value}")
                if cookie_name == "access_token" and not access_token:
                    access_token = cookie.value
                    print(f"Found access token in cookies: {access_token[:10]}...")
                elif cookie_name == "refresh_token" and not refresh_token:
                    refresh_token = cookie.value
                    print(f"Found refresh token in cookies: {refresh_token[:10]}...")
            
            # If verification was successful
            if response.status in [200, 201]:
                # Add tokens to response data if they exist but aren't in the data
                if access_token and "access_token" not in response_data:
                    response_data["access_token"] = access_token
                    print(f"Added access token to response data")
                
                if refresh_token and "refresh_token" not in response_data:
                    response_data["refresh_token"] = refresh_token
                    print(f"Added refresh token to response data")
                
                # Get or create session
                session, new_session_id = fresh_bus_assistant.get_or_create_session(session_id)
                
                # Store user info in session
                if "context" not in session:
                    session["context"] = {}
                
                # Use token from response data if available
                final_access_token = response_data.get("access_token", access_token)
                final_refresh_token = response_data.get("refresh_token", refresh_token)
                
                if final_access_token:
                    print(f"Final access token: {final_access_token[:10]}...")
                else:
                    print("WARNING: No access token found!")

                if final_access_token:
                    print(f"final_refresh_token: {final_refresh_token[:10]}...")
                else:
                    print("WARNING: No refresh token found!")
                
                session["context"]["user"] = {
                    "mobile": mobile,
                    "name": response_data.get("name", ""),
                    "email": response_data.get("email", ""),
                    "id": response_data.get("id", "")
                }
                
                # Store auth info in session
                session["context"]["auth"] = {
                    "token": final_access_token,
                    "refresh_token": final_refresh_token
                }
                
                # Add session_id to response
                response_data["session_id"] = new_session_id
                
                print(f"=== USER AUTHENTICATED ===")
                print(f"Mobile: {mobile}")
                print(f"Session ID: {new_session_id}")
                print(f"Access Token: {final_access_token[:10]}..." if final_access_token else "None")
                print(f"=========================")
                
                # Create response with data in body
                response_obj = JSONResponse(content=response_data)
                
                # Add Authorization header to response
                if final_access_token:
                    response_obj.headers["Authorization"] = f"Bearer {final_access_token}"
                    response_obj.headers["X-Access-Token"] = final_access_token
                    response_obj.headers["access_token"] = final_access_token
                
                if final_refresh_token:
                    response_obj.headers["refresh_token"] = final_refresh_token
                
                # Set response headers for CORS
                response_obj.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
                response_obj.headers["Access-Control-Allow-Credentials"] = "true"
                response_obj.headers["Access-Control-Expose-Headers"] = "Authorization, X-Access-Token, access_token, refresh_token, Set-Cookie"
                
                # Set cookies properly for browser storage
                if final_access_token:
                    # Get cookie attributes if available from original response
                    max_age_access = 900  # Default to 15 minutes
                    path_access = "/"
                    same_site_access = "lax"
                    http_only_access = True
                    
                    # Check if we have cookie info from the original response
                    access_cookie = next((c for c in cookies.values() if c.key == "access_token"), None)
                    if access_cookie:
                        if "max-age" in access_cookie:
                            max_age_access = int(access_cookie["max-age"])
                        if "path" in access_cookie:
                            path_access = access_cookie["path"]
                        if "samesite" in access_cookie:
                            same_site_access = access_cookie["samesite"]
                        if "httponly" in access_cookie:
                            http_only_access = bool(access_cookie["httponly"])
                    
                    # Set the access token cookie
                    response_obj.set_cookie(
                        key="access_token",
                        value=final_access_token,
                        max_age=max_age_access,
                        path=path_access,
                        samesite=same_site_access,
                        httponly=http_only_access,
                        secure=False  # Set to True in production with HTTPS
                    )
                    print(f"Set access_token cookie with max_age={max_age_access}")
                
                if final_refresh_token:
                    # Get cookie attributes if available from original response
                    max_age_refresh = 604800  # Default to 7 days
                    path_refresh = "/"
                    same_site_refresh = "lax"
                    http_only_refresh = True
                    
                    # Check if we have cookie info from the original response
                    refresh_cookie = next((c for c in cookies.values() if c.key == "refresh_token"), None)
                    if refresh_cookie:
                        if "max-age" in refresh_cookie:
                            max_age_refresh = int(refresh_cookie["max-age"])
                        if "path" in refresh_cookie:
                            path_refresh = refresh_cookie["path"]
                        if "samesite" in refresh_cookie:
                            same_site_refresh = refresh_cookie["samesite"]
                        if "httponly" in refresh_cookie:
                            http_only_refresh = bool(refresh_cookie["httponly"])
                    
                    # Set the refresh token cookie
                    response_obj.set_cookie(
                        key="refresh_token",
                        value=final_refresh_token,
                        max_age=max_age_refresh,
                        path=path_refresh,
                        samesite=same_site_refresh,
                        httponly=http_only_refresh,
                        secure=False  # Set to True in production with HTTPS
                    )
                    print(f"Set refresh_token cookie with max_age={max_age_refresh}")
                
                return response_obj
            
            # If verification failed
            return JSONResponse(
                status_code=response.status,
                content=response_data
            )
                
    except Exception as e:
        print(f"Error in verify_otp: {e}")
        import traceback
        traceback.print_exc()
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


@app.api_route("/query", methods=["POST"])
async def query(request: Request):
    try:
        body = await request.json()
        query = body.get("query", "")
        session_id = body.get("session_id")
        user_id = body.get("user_id")
        provider = body.get("provider")
        model = body.get("model")
        user_gender = body.get("gender")
        user_location = body.get("location")
        detected_language = body.get("language")
        
        print(f"--- New Query: '{query}' ---")
        
        # Get access token from multiple sources (prioritize cookies)
        access_token = None
        # Initialize refresh_token variable
        refresh_token = None
        
        # 1. Check cookies first (most reliable source)
        if "access_token" in request.cookies:
            access_token = request.cookies.get("access_token")
            print(f"Found access token in cookies: {access_token[:10]}...")
        
        # 2. Check Authorization header
        if not access_token:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer ") and auth_header != "Bearer undefined":
                access_token = auth_header.replace("Bearer ", "")
                print(f"Found access token in Authorization header: {access_token[:10]}...")
        
        # 3. Check custom headers
        if not access_token:
            access_token = request.headers.get("X-Access-Token") or request.headers.get("access_token")
            if access_token:
                print(f"Found access token in custom header: {access_token[:10]}...")
        
        # 4. Check request body
        if not access_token and "access_token" in body:
            access_token = body.get("access_token")
            if access_token:
                print(f"Found access token in request body: {access_token[:10]}...")
        
        # Get refresh token from multiple sources
        # 1. Check cookies for refresh token
        if "refresh_token" in request.cookies:
            refresh_token = request.cookies.get("refresh_token")
            print(f"Found refresh token in cookies: {refresh_token[:10]}...")
        
        # 2. Check headers for refresh token
        if not refresh_token:
            refresh_header = request.headers.get("refresh_token") or request.headers.get("X-Refresh-Token")
            if refresh_header:
                refresh_token = refresh_header
                print(f"Found refresh token in header: {refresh_token[:10]}...")
        
        # 3. Check request body for refresh token
        if not refresh_token and "refresh_token" in body:
            refresh_token = body.get("refresh_token")
            if refresh_token:
                print(f"Found refresh token in request body: {refresh_token[:10]}...")
        
        # Get or create session early so we can use it for token storage
        session, session_id = fresh_bus_assistant.get_or_create_session(session_id)
        print(f"Using {'existing' if 'messages' in session else 'new'} session with ID: {session_id}")
        
        # Print session info for debugging
        if session_id and session_id in fresh_bus_assistant.sessions:
            print(f"Current session messages count: {len(fresh_bus_assistant.sessions[session_id]['messages'])}")
            for i, msg in enumerate(fresh_bus_assistant.sessions[session_id]['messages'][-2:]):  # Show last 2 messages
                preview = msg.get('content', '')[:30] + '...' if len(msg.get('content', '')) > 30 else msg.get('content', '')
                print(f"Message {i}: {msg.get('role')} - {preview}")
        
        # 5. Check session data for access token
        if not access_token:
            if session.get("context") and session["context"].get("auth") and session["context"]["auth"].get("token"):
                access_token = session["context"]["auth"]["token"]
                print(f"Found access token in session: {access_token[:10]}...")
        
        # Initialize context and auth objects if they don't exist or are None
        if "context" not in session:
            session["context"] = {}
        
        if "auth" not in session["context"] or session["context"]["auth"] is None:
            session["context"]["auth"] = {}  # Reset to empty dict if it's None
            print("Initialized auth object in session")
        
        # Check session for refresh token
        if not refresh_token and session.get("context") and session["context"].get("auth") and session["context"]["auth"].get("refresh_token"):
            refresh_token = session["context"]["auth"]["refresh_token"]
            print(f"Found refresh token in session: {refresh_token[:10]}...")
        
        # Log if no token found
        if not access_token:
            print("No access token found in request")
        
        # Verify token validity and refresh if needed
        # new verification using our cookie‑aware helper
        token_valid = False
        if access_token:
            try:
                # Pass BOTH tokens to fetch_user_profile
                profile_data = await fresh_bus_assistant.fetch_user_profile(access_token, refresh_token)
                if profile_data:
                    print("✅ Token verification successful!")
                    token_valid = True
                    user_authenticated = True

                    # Update session with profile data
                    if "context" not in session:
                        session["context"] = {}
                    
                    # Store complete user data
                    session["context"]["user"] = {
                        "id": profile_data.get("id"),
                        "name": profile_data.get("name"),
                        "mobile": profile_data.get("mobile"),
                        "email": profile_data.get("email"),
                        "gender": profile_data.get("gender"),
                        "profile": profile_data  # Store complete profile
                    }

                    # Store auth tokens
                    session["context"]["auth"] = {
                        "token": access_token,
                        "refresh_token": refresh_token
                    }

                    print(f"Updated session with user profile data for {profile_data.get('name')}")
                else:
                    print("❌ Token verification failed, attempting refresh")
                    if refresh_token:
                        # Implement refresh logic here if needed
                        pass
            except Exception as e:
                print(f"Error verifying token: {e}")
                token_valid = False
        
        # If token is invalid and we have a refresh token, try to refresh
        if not token_valid and refresh_token:
            print("Token is invalid. Attempting to refresh...")
            try:
                # Call the refresh token API
                refresh_url = f"{fresh_bus_assistant.BASE_URL}/auth/refresh"
                
                async with fresh_bus_assistant.http_session.post(
                    refresh_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {refresh_token}"
                    },
                    json={"refresh_token": refresh_token}
                ) as refresh_response:
                    if refresh_response.status == 200:
                        refresh_data = await refresh_response.json()
                        new_access_token = refresh_data.get("access_token")
                        
                        if new_access_token:
                            print(f"Token refreshed successfully! New token: {new_access_token[:10]}...")
                            access_token = new_access_token
                            token_valid = True
                            
                            # Update session with new token
                            session["context"]["auth"]["token"] = new_access_token
                            if refresh_data.get("refresh_token"):
                                session["context"]["auth"]["refresh_token"] = refresh_data.get("refresh_token")
                    else:
                        print(f"Failed to refresh token: {refresh_response.status}")
            except Exception as refresh_error:
                print(f"Error refreshing token: {refresh_error}")

        # Validate query
        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Query is required"}
            )
        
        # Update token in session if we have one
        if access_token:
            # Store or update the access token
            session["context"]["auth"]["token"] = access_token
            
            # Store refresh token if available
            if refresh_token:
                session["context"]["auth"]["refresh_token"] = refresh_token
            
            print(f"Updated auth tokens in session")
        
        # Determine if this is a simple query
        is_simple_query = query.lower() in ["hi", "hello", "hey"] or "name" in query.lower() or "who are you" in query.lower()
        print(f"Is simple query: {is_simple_query}")
        
        # Check if this is a tracking request
        is_tracking_request = fresh_bus_assistant.is_bus_tracking_query(query)
        print(f"Is tracking request: {is_tracking_request}")
        
        # Detect language
        language = detected_language or fresh_bus_assistant.detect_language(query)
        print(f"Language detected/set: {language}")
        
        # Extract locations if present in query
        source, destination = fresh_bus_assistant.extract_locations(query)
        if source and destination:
            print(f"Extracted locations: {source} to {destination}")
            # Add to logs for debugging
            if source and destination:
                print(f"Extracted from 'to-from' pattern: FROM {source} TO {destination}")
        
        # Special handling for tracking requests
        if is_tracking_request:
            print("Handling bus tracking request")
            
            # Check if user is authenticated with a valid token
            user_authenticated = token_valid
            
            if user_authenticated:
                print(f"User is authenticated, fetching ticket data with token: {access_token[:10]}..." if access_token else "Token not available")
                # Add user message to session
                if "messages" not in session:
                    session["messages"] = []
                session["messages"].append({"role": "user", "content": query})
                
                # Use the authenticated flow to get user's tickets and bus location
                tracking_response = await fresh_bus_assistant.handle_authenticated_tracking_request(access_token)
                
                # Add response to session
                session["messages"].append({"role": "assistant", "content": tracking_response})
            else:
                print("User is not authenticated, providing login instructions")
                tracking_response = "I need to see your active tickets to track your bus. Please log in to your Fresh Bus account first, then I can show you your bus location, estimated arrival time, and journey status.\n\nAfter logging in, you can ask me 'Where is my bus?' again, and I'll provide real-time tracking information."
                
                # Add message to session
                if "messages" not in session:
                    session["messages"] = []
                session["messages"].append({"role": "user", "content": query})
                session["messages"].append({"role": "assistant", "content": tracking_response})
            
            # Save to Redis if possible
            try:
                conversation_id = conversation_manager.save_conversation(
                    session_id, 
                    session["messages"],
                    user_id=user_id
                )
                print(f"Saved tracking response to Redis: {conversation_id}")
            except Exception as redis_err:
                print(f"Failed to save to Redis: {redis_err}")
                conversation_id = None
                
            # Stream the response
            async def tracking_stream():
                # Send the text chunk
                yield f"data: {json.dumps({'text': tracking_response, 'done': False})}\n\n"
                
                # Send the completion chunk with metadata
                yield f"data: {json.dumps({
                    'text': '',
                    'done': True,
                    'session_id': session_id,
                    'language_style': 'english',
                    'conversation_id': conversation_id,
                    'access_token': access_token  # Include token in the final chunk
                })}\n\n"
                
            response = StreamingResponse(
                tracking_stream(),
                media_type="text/event-stream",
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
            
            # Add auth headers if available
            if access_token:
                response.headers["Authorization"] = f"Bearer {access_token}"
                response.headers["X-Access-Token"] = access_token
                response.headers["access_token"] = access_token
                
                # Also set cookie for access token
                response.set_cookie(
                    key="access_token",
                    value=access_token,
                    httponly=True,
                    secure=False,  # Set to True in production
                    samesite="lax",
                    path="/",
                    max_age=900  # 15 minutes
                )
            
            # Set refresh token cookie if available
            if refresh_token:
                response.set_cookie(
                    key="refresh_token",
                    value=refresh_token,
                    httponly=True,
                    secure=False,  # Set to True in production
                    samesite="lax",
                    path="/",
                    max_age=604800  # 7 days
                )
            
            # Add CORS headers
            response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Expose-Headers"] = "Authorization, X-Access-Token, access_token, refresh_token, Set-Cookie"
            
            return response

        # For simple greetings
        if is_simple_query:
            print("Handling simple greeting")
            # Create a simple greeting response
            if query.lower() in ["hi", "hello", "hey"]:
                greeting_text = "Hello! I'm Ṧ.AI, your Fresh Bus travel assistant. How can I help you today?"
            elif "name" in query.lower():
                greeting_text = "My name is Ṧ.AI (pronounced as 'Sai'). I'm your Fresh Bus travel assistant. How can I help you today?"
            else:
                greeting_text = "Hello! I'm Ṧ.AI, your Fresh Bus travel assistant. How can I help you with your bus travel needs?"
            
            # Add messages to the session
            if "messages" not in session:
                session["messages"] = []
            
            session["messages"].append({"role": "user", "content": query})
            session["messages"].append({"role": "assistant", "content": greeting_text})
            
            # Save to Redis if possible
            try:
                user_mobile = None
                if session.get("context") and isinstance(session["context"].get("user"), dict):
                    user_mobile = session["context"]["user"].get("mobile")
                
                conversation_id = conversation_manager.save_conversation(
                    session_id, 
                    session["messages"],
                    user_id=user_mobile or user_id
                )
                print(f"Saved greeting response to Redis: {conversation_id}, user mobile: {user_mobile}")
            except Exception as redis_err:
                print(f"Failed to save greeting to Redis: {redis_err}")
                conversation_id = None

            # Stream the response as SSE
            async def simple_stream():
                # Send the text chunk
                yield f"data: {json.dumps({'text': greeting_text, 'done': False})}\n\n"
                
                # Send the completion chunk with metadata
                yield f"data: {json.dumps({
                    'text': '', 
                    'done': True, 
                    'session_id': session_id,
                    'language_style': 'english',
                    'conversation_id': conversation_id,
                    'access_token': access_token  # Include token in the final chunk
                })}\n\n"
                
            response = StreamingResponse(
                simple_stream(),
                media_type="text/event-stream",
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
            
            # Add auth headers if available
            if access_token:
                response.headers["Authorization"] = f"Bearer {access_token}"
                response.headers["X-Access-Token"] = access_token
                response.headers["access_token"] = access_token
                
                # Also set cookie for access token
                response.set_cookie(
                    key="access_token",
                    value=access_token,
                    httponly=True,
                    secure=False,  # Set to True in production
                    samesite="lax",
                    path="/",
                    max_age=900  # 15 minutes
                )
            
            # Set refresh token cookie if available
            if refresh_token:
                response.set_cookie(
                    key="refresh_token",
                    value=refresh_token,
                    httponly=True,
                    secure=False,  # Set to True in production
                    samesite="lax",
                    path="/",
                    max_age=604800  # 7 days
                )
            
            # Add CORS headers
            response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Expose-Headers"] = "Authorization, X-Access-Token, access_token, refresh_token, Set-Cookie"
            
            return response

        # Normal response flow for other queries
        # Build parameters object for generate_response
        params = {
            "query": query,
            "session_id": session_id,
            "provider": provider,
            "model": model
        }
        
        # Add user location if available
        if user_location:
            params["user_location"] = user_location
        
        # Add gender if available
        if user_gender:
            params["user_gender"] = user_gender
        
        # Add detected language if available
        if detected_language:
            params["detected_language"] = detected_language

        # Pass the access token to generate_response
        if access_token:
            params["access_token"] = access_token

        # Generate response
        response_generator = fresh_bus_assistant.generate_response(**params)

        async def stream_response():
            try:
                async for chunk in response_generator:
                    # Debug the type of chunk
                    if isinstance(chunk, dict):
                        chunk_preview = json.dumps(chunk)[:100] + "..." if len(json.dumps(chunk)) > 100 else json.dumps(chunk)
                        print(f"OUTPUT CHUNK: {chunk_preview}")
                    else:
                        chunk_preview = str(chunk)[:100] + "..." if len(str(chunk)) > 100 else str(chunk)
                        print(f"OUTPUT CHUNK: {chunk_preview}")
                    
                    # Parse the chunk if it's a string
                    if isinstance(chunk, str):
                        try:
                            chunk = json.loads(chunk)
                        except json.JSONDecodeError:
                            # If it's not valid JSON, wrap it
                            chunk = {"text": chunk, "done": False}
                    
                    # Always ensure chunk is a dictionary with required fields
                    if not isinstance(chunk, dict):
                        chunk = {"text": str(chunk), "done": False}
                    
                    if "error" in chunk:
                        yield f"data: {json.dumps({'error': chunk['error'], 'done': True})}\n\n"
                        return
                    
                    # If this is the final chunk with done=True
                    if chunk.get("done"):
                        # Ensure we preserve json_data if it exists
                        if "json_data" in chunk:
                            # Make sure json_data field is properly passed to the frontend
                            final_chunk = {
                                "text": "", 
                                "done": True,
                                "session_id": chunk.get("session_id", session_id),
                                "language_style": chunk.get("language_style", "english"),
                                "conversation_id": chunk.get("conversation_id"),
                                "json_data": chunk["json_data"]  # Keep the JSON data
                            }
                            
                            # Include access token if available
                            if access_token:
                                final_chunk["access_token"] = access_token
                                
                            # Associate conversation with user if needed
                            if user_id and final_chunk.get("conversation_id"):
                                try:
                                    # Associate the conversation with the user
                                    conversation_manager.redis.sadd(
                                        f"{conversation_manager.user_index_prefix}{user_id}", 
                                        final_chunk.get("conversation_id")
                                    )
                                    # Add user_id to the conversation data
                                    conversation_key = f"{conversation_manager.conversation_prefix}{final_chunk.get('conversation_id')}"
                                    conversation_manager.redis.hset(conversation_key, "user_id", user_id)
                                    print(f"Associated conversation {final_chunk.get('conversation_id')} with user {user_id}")
                                except Exception as e:
                                    print(f"Error associating conversation with user: {e}")
                            
                            # Send the final chunk with json_data
                            print(f"Sending final chunk with json_data: {json.dumps(final_chunk)[:100]}...")
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                        else:
                            # Standard final chunk without json_data
                            chunk["access_token"] = access_token if access_token else None
                            yield f"data: {json.dumps(chunk)}\n\n"
                    else:
                        # Send normal text chunk
                        yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as e:
                error_msg = f"Error in stream_response: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'error': error_msg, 'done': True})}\n\n"

        # Create response with headers
        response = StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )
        
        # Add auth headers if available
        if access_token:
            response.headers["Authorization"] = f"Bearer {access_token}"
            response.headers["X-Access-Token"] = access_token
            response.headers["access_token"] = access_token
            
            # Set cookie for access token
            response.set_cookie(
                key="access_token",
                value=access_token,
                httponly=True,
                secure=False,  # Set to True in production
                samesite="lax",
                path="/",
                max_age=900  # 15 minutes
            )
        
        # Set refresh token cookie if available
        if refresh_token:
            response.set_cookie(
                key="refresh_token",
                value=refresh_token,
                httponly=True,
                secure=False,  # Set to True in production
                samesite="lax",
                path="/",
                max_age=604800  # 7 days
            )
        
        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Expose-Headers"] = "Authorization, X-Access-Token, access_token, refresh_token, Set-Cookie"
        
        return response
        
    except Exception as e:
        print(f"Error in query: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    
@app.get("/auth/status")
async def auth_status(request: Request):
    """Check authentication status and return user info if authenticated"""
    # Get token from request
    auth_header = request.headers.get("Authorization")
    access_token = None
    
    if auth_header and auth_header.startswith("Bearer ") and auth_header != "Bearer undefined":
        access_token = auth_header.replace("Bearer ", "")
    
    # Check custom header
    if not access_token:
        access_token = request.headers.get("X-Access-Token") or request.headers.get("access_token")
    
    # Check session
    session_id = request.query_params.get("session_id")
    if not access_token and session_id:
        session, _ = fresh_bus_assistant.get_or_create_session(session_id)
        if session.get("context") and session["context"].get("auth"):
            access_token = session["context"]["auth"].get("token")
    
    # If we have a token, verify it
    if access_token:
        try:
            profile_url = f"{fresh_bus_assistant.BASE_URL_CUSTOMER}/profile"
            
            async with fresh_bus_assistant.http_session.get(
                profile_url,
                headers={"Authorization": f"Bearer {access_token}"}
            ) as profile_response:
                if profile_response.status == 200:
                    profile_data = await profile_response.json()
                    
                    # Create response
                    response = JSONResponse(content={
                        "authenticated": True,
                        "user": profile_data,
                        "token": access_token
                    })
                    
                    # Add auth headers
                    response.headers["Authorization"] = f"Bearer {access_token}"
                    response.headers["X-Access-Token"] = access_token
                    response.headers["access_token"] = access_token
                    
                    # CORS headers
                    response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
                    response.headers["Access-Control-Allow-Credentials"] = "true"
                    response.headers["Access-Control-Expose-Headers"] = "Authorization, X-Access-Token, access_token"
                    
                    return response
                else:
                    return JSONResponse(content={
                        "authenticated": False,
                        "message": "Invalid or expired token"
                    })
        except Exception as e:
            print(f"Error verifying token: {e}")
            return JSONResponse(content={
                "authenticated": False,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "authenticated": False,
        "message": "No authentication token found"
    })

@app.get("/user/conversations")
async def get_user_conversations(
    request: Request,
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0)
):
    """Get conversations for the authenticated user"""
    try:
        # 1. Extract access token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"error": "Authentication required"})
        access_token = auth_header.replace("Bearer ", "")

        # 2. Extract refresh token
        refresh_token = None
        if "refresh_token" in request.cookies:
            refresh_token = request.cookies.get("refresh_token")
        elif request.headers.get("X-Refresh-Token"):
            refresh_token = request.headers.get("X-Refresh-Token")
        elif request.headers.get("refresh_token"):
            refresh_token = request.headers.get("refresh_token")

        # 3. Ensure HTTP session
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()

        # 4. Fetch profile
        profile_data = await fresh_bus_assistant.fetch_user_profile(access_token, refresh_token)
        if not profile_data:
            return JSONResponse(status_code=404, content={"error": "User profile not found"})

        # 5. Determine user_id
        user_id = str(profile_data.get("id", "") or profile_data.get("mobile", ""))
        if not user_id:
            return JSONResponse(status_code=400, content={"error": "User ID not found in profile"})

        print(f"Fetching conversations for user ID: {user_id}")

        # 6. Retrieve and return
        conversations = conversation_manager.get_conversations_by_user(user_id, limit, offset)
        return JSONResponse(content=conversations)

    except Exception as e:
        print(f"Error getting user conversations: {e}")
        return JSONResponse(status_code=500, content={"error": f"Error retrieving conversations: {str(e)}"})

    
@app.delete("/user/conversations")
async def delete_user_conversations(request: Request):
    """Delete all conversations for the authenticated user"""
    try:
        # 1. Extract access token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"error": "Authentication required"})
        access_token = auth_header.replace("Bearer ", "")

        # 2. Extract refresh token
        refresh_token = None
        if "refresh_token" in request.cookies:
            refresh_token = request.cookies.get("refresh_token")
        elif request.headers.get("X-Refresh-Token"):
            refresh_token = request.headers.get("X-Refresh-Token")
        elif request.headers.get("refresh_token"):
            refresh_token = request.headers.get("refresh_token")

        # 3. Ensure HTTP session exists
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()

        # 4. Fetch profile using both tokens
        profile_data = await fresh_bus_assistant.fetch_user_profile(access_token, refresh_token)
        if not profile_data:
            return JSONResponse(status_code=404, content={"error": "User profile not found"})

        # 5. Determine user_id
        user_id = str(profile_data.get("id") or profile_data.get("mobile") or "")
        if not user_id:
            return JSONResponse(status_code=400, content={"error": "User ID not found in profile"})

        # 6. Delete conversations
        count = conversation_manager.delete_user_conversations(user_id)
        return JSONResponse(content={"status": "success", "message": f"Deleted {count} conversations"})

    except Exception as e:
        print(f"Error deleting user conversations: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error deleting conversations: {str(e)}"}
        )

@app.get("/trips/{trip_id}/boarding-points/{source_id}")
async def get_boarding_points(trip_id: str, source_id: str):
    """Get boarding points for a specific trip"""
    try:
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()
            
        boarding_points = await fresh_bus_assistant.fetch_boarding_points(trip_id, source_id)
        
        if not boarding_points:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "No boarding points found"}
            )
            
        return JSONResponse(content=boarding_points)
    except Exception as e:
        print(f"Error fetching boarding points: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error fetching boarding points: {str(e)}"}
        )

@app.get("/trips/{trip_id}/dropping-points")
async def get_dropping_points(trip_id: str):
    """Get dropping points for a specific trip"""
    try:
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()
            
        dropping_points = await fresh_bus_assistant.fetch_dropping_points(trip_id)
        
        if not dropping_points:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "No dropping points found"}
            )
            
        return JSONResponse(content=dropping_points)
    except Exception as e:
        print(f"Error fetching dropping points: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error fetching dropping points: {str(e)}"}
        )

@app.get("/trips/{trip_id}/seats")
async def get_seats(trip_id: str, source_id: str, destination_id: str):
    """Get seats for a specific trip"""
    try:
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()
            
        seats_data = await fresh_bus_assistant.fetch_seats(trip_id, source_id, destination_id)
        
        if not seats_data:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "No seat data found"}
            )
            
        return JSONResponse(content=seats_data)
    except Exception as e:
        print(f"Error fetching seats: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error fetching seats: {str(e)}"}
        )
    

    
@app.post("/book-ticket")
async def book_ticket(request: Request):
    """Book a ticket with passenger details"""
    try:
        # Get the request body
        booking_data = await request.json()
        
        # Check if all required fields are present
        required_fields = ["mobile", "email", "trip_id", "boarding_point_id", 
                          "dropping_point_id", "seat_map"]
        
        for field in required_fields:
            if field not in booking_data:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": f"Missing required field: {field}"}
                )
        
        # Get the auth token from the request headers
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"success": False, "message": "Authentication required"}
            )
        
        access_token = auth_header.replace("Bearer ", "")
        
        # Initialize HTTP session if needed
        if not fresh_bus_assistant.http_session:
            await fresh_bus_assistant.init_http_session()
        
        # Prepare the booking request
        booking_payload = {
            "mobile": booking_data["mobile"],
            "email": booking_data["email"],
            "seat_map": booking_data["seat_map"],
            "trip_id": booking_data["trip_id"],
            "boarding_point_id": booking_data["boarding_point_id"],
            "dropping_point_id": booking_data["dropping_point_id"],
            "total_collect_amount": booking_data.get("total_collect_amount", 0),
            "freshcard": booking_data.get("freshcard", False),
            "return_url": booking_data.get("return_url", "https://freshbus.com/booking-confirmation")
        }
        
        # Add optional fields if present
        if "boarding_point_time" in booking_data:
            booking_payload["boarding_point_time"] = booking_data["boarding_point_time"]
        
        if "dropping_point_time" in booking_data:
            booking_payload["dropping_point_time"] = booking_data["dropping_point_time"]
        
        if "main_category" in booking_data:
            booking_payload["main_category"] = booking_data["main_category"]
        
        if "freshcardId" in booking_data and booking_data["freshcard"]:
            booking_payload["freshcardId"] = booking_data["freshcardId"]
        
        # Make the booking API call
        booking_url = f"{fresh_bus_assistant.BASE_URL}/bookings"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        async with fresh_bus_assistant.http_session.post(
            booking_url, 
            json=booking_payload,
            headers=headers
        ) as response:
            response_data = await response.json()
            
            if response.status == 200:
                # Booking successful
                return JSONResponse(
                    content={
                        "success": True,
                        "message": "Ticket booked successfully",
                        "booking_details": response_data
                    }
                )
            else:
                # Booking failed
                error_message = response_data.get("message", "Unknown error")
                return JSONResponse(
                    status_code=response.status,
                    content={"success": False, "message": f"Booking failed: {error_message}"}
                )
                
    except Exception as e:
        print(f"Error booking ticket: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error booking ticket: {str(e)}"}
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