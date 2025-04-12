# redis_manager.py

import redis
import json
import uuid
import time
from datetime import datetime, timezone, timedelta # Added timezone
from typing import Dict, List, Any, Optional
from config import Config  # Import Config for Redis settings and TTL

# --- Active Session Management ---

class RedisConversationManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.conversation_prefix = "fresh_bus:conversation:"
        self.session_index_prefix = "fresh_bus:session_index:"
        self.user_index_prefix = "fresh_bus:user_index:"  # User indices prefix
        self.message_prefix = "fresh_bus:messages:"
        self.context_prefix = "fresh_bus:context:"
        self.session_ttl_seconds = 86400 * 7  # 7 days by default
        self.global_index_key = "fresh_bus:conversations"  # Sorted set for all conversations
        
        print("RedisConversationManager initialized.")
    
    def save_conversation(self, session_id, messages, user_id=None):
        """Save conversation with optional user_id tagging"""
        try:
            if not messages:
                return None
            
            conversation_id = str(uuid.uuid4())
            # Use UTC timestamp for consistency
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Create conversation data
            conversation_data = {
                "conversation_id": conversation_id,
                "session_id": session_id,
                "timestamp": timestamp,
            }
            
            # Add user_id if provided
            if user_id:
                conversation_data["user_id"] = str(user_id)
            
            try:
                conversation_data["messages"] = json.dumps(messages)
            except TypeError as e:
                print(f"Error serializing messages for archival (session {session_id}): {e}")
                # Try serializing with default=str for difficult types
                try:
                    conversation_data["messages"] = json.dumps(messages, default=str)
                except Exception as json_err:
                    print(f"Failed to serialize messages even with default=str: {json_err}")
                    return None
            
            # Save conversation
            key = f"{self.conversation_prefix}{conversation_id}"
            pipe = self.redis.pipeline()
            pipe.hmset(key, conversation_data)
            
            # Add to session index
            pipe.sadd(f"{self.session_index_prefix}{session_id}", conversation_id)
            
            # Add to user index if user_id is provided
            if user_id:
                pipe.sadd(f"{self.user_index_prefix}{user_id}", conversation_id)
            
            # Add to global index with timestamp as score for sorting
            timestamp_unix = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp()
            pipe.zadd(self.global_index_key, {conversation_id: timestamp_unix})
            
            pipe.execute()
            
            print(f"Archived conversation {conversation_id} for session {session_id}" + 
                  (f" and user {user_id}" if user_id else ""))
            return conversation_id
        except Exception as e:
            print(f"Redis save_conversation error: {e}")
            return None
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific archived conversation by its ID."""
        key = f"{self.conversation_prefix}{conversation_id}"
        try:
            conversation_data = self.redis.hgetall(key)
            if not conversation_data:
                print(f"Archived conversation {conversation_id} not found.")
                return None

            # Convert bytes to strings if needed
            conversation_data = {k.decode('utf-8') if isinstance(k, bytes) else k: 
                               v.decode('utf-8') if isinstance(v, bytes) else v 
                               for k, v in conversation_data.items()}

            try:
                # Handle potential missing 'messages' key gracefully
                messages_str = conversation_data.get("messages", "[]")
                conversation_data["messages"] = json.loads(messages_str)
            except json.JSONDecodeError as json_err:
                print(f"Error decoding messages for archived conversation {conversation_id}: {json_err}")
                conversation_data["messages"] = [{"role": "system", "content": f"Error loading messages: {json_err}"}]

            # Ensure essential keys exist
            conversation_data.setdefault("conversation_id", conversation_id)
            conversation_data.setdefault("session_id", "unknown")
            conversation_data.setdefault("timestamp", "unknown")

            return conversation_data
        except redis.RedisError as e:
            print(f"Redis error getting conversation {conversation_id}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error getting conversation {conversation_id}: {e}")
            return None
    
    def get_conversations_by_user(self, user_id, limit=50, offset=0):
        """Retrieves all conversations for a specific user"""
        try:
            if not user_id:
                return []
                
            user_index_key = f"{self.user_index_prefix}{user_id}"
            conversation_ids = self.redis.smembers(user_index_key)
            
            if not conversation_ids:
                print(f"No conversations found for user {user_id}")
                return []
            
            # Convert bytes to strings
            conversation_ids_list = [cid.decode('utf-8') if isinstance(cid, bytes) else cid for cid in conversation_ids]
            
            # Sort by timestamp by getting all conversations first
            all_conversations = []
            for conv_id in conversation_ids_list:
                conversation = self.get_conversation(conv_id)
                if conversation:
                    all_conversations.append(conversation)
            
            # Sort by timestamp (newest first)
            all_conversations.sort(key=lambda x: x.get("timestamp", "0"), reverse=True)
            
            # Apply limit and offset
            paginated_conversations = all_conversations[offset:offset+limit]
            
            # Create summaries for the response
            conversations = []
            for conversation in paginated_conversations:
                first_user_query = ""
                if conversation.get("messages"):
                    user_msgs = [m.get("content", "") for m in conversation["messages"] if m.get("role") == "user"]
                    if user_msgs:
                        first_user_query = user_msgs[0][:100] + ('...' if len(user_msgs[0]) > 100 else '')
                
                # Create a summary with necessary data
                conversations.append({
                    "conversation_id": conversation.get("conversation_id"),
                    "session_id": conversation.get("session_id"),
                    "timestamp": conversation.get("timestamp"),
                    "user_query": first_user_query,
                    "message_count": len(conversation.get("messages", [])),
                    "messages": conversation.get("messages")  # Include full messages for client usage
                })
            
            return conversations
        except Exception as e:
            print(f"Redis get_conversations_by_user error: {e}")
            return []
            
    def delete_user_conversations(self, user_id):
        """Deletes all conversations for a specific user"""
        try:
            if not user_id:
                return 0
                
            user_index_key = f"{self.user_index_prefix}{user_id}"
            conversation_ids = self.redis.smembers(user_index_key)
            
            if not conversation_ids:
                return 0
            
            # Convert bytes to strings if needed
            conversation_ids = [cid.decode('utf-8') if isinstance(cid, bytes) else cid for cid in conversation_ids]
                
            # Delete each conversation
            deleted_count = 0
            pipe = self.redis.pipeline()
            
            for conv_id in conversation_ids:
                # Get the session_id first
                key = f"{self.conversation_prefix}{conv_id}"
                session_id_bytes = self.redis.hget(key, "session_id")
                
                if session_id_bytes:
                    session_id = session_id_bytes.decode('utf-8') if isinstance(session_id_bytes, bytes) else session_id_bytes
                    # Remove from session index
                    pipe.srem(f"{self.session_index_prefix}{session_id}", conv_id)
                
                # Remove from global index
                pipe.zrem(self.global_index_key, conv_id)
                
                # Delete the conversation
                pipe.delete(key)
                deleted_count += 1
            
            # Delete the user index
            pipe.delete(user_index_key)
            
            # Execute all commands
            pipe.execute()
            
            print(f"Deleted {deleted_count} conversations for user {user_id}")
            return deleted_count
        except Exception as e:
            print(f"Redis delete_user_conversations error: {e}")
            return 0
            
    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieves the list of messages for a given session ID."""
        key = f"{self.message_prefix}{session_id}"
        messages = []
        try:
            raw_messages = self.redis.lrange(key, 0, -1)
            if raw_messages:
                self.redis.expire(key, self.session_ttl_seconds) # Refresh TTL
                for i, msg_str in enumerate(raw_messages):
                    try:
                        msg_decoded = msg_str.decode('utf-8') if isinstance(msg_str, bytes) else msg_str
                        message = json.loads(msg_decoded)
                        # Ensure message has required fields
                        if isinstance(message, dict) and "role" in message and "content" in message:
                            messages.append(message)
                    except json.JSONDecodeError as json_err:
                        print(f"Error decoding message at index {i}: {json_err}")
                        continue
            return messages
        except redis.RedisError as e:
            print(f"Redis error getting messages: {e}")
            return []

    def add_message(self, session_id: str, message: Dict[str, str], user_id=None):
        """Adds a message to the session's history with optional user association."""
        key = f"{self.message_prefix}{session_id}"
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            print(f"Invalid message format: {message}")
            return
            
        try:
            # Add timestamp to message
            message["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            # Add user_id to message if provided
            if user_id:
                message["user_id"] = str(user_id)
                
            message_json = json.dumps(message)
            self.redis.rpush(key, message_json)
            self.redis.expire(key, self.session_ttl_seconds)
            
            # If this completes a conversation (user message followed by assistant message),
            # consider archiving it
            if message["role"] == "assistant":
                messages = self.get_messages(session_id)
                if len(messages) >= 2:
                    # Save the conversation with user_id if available
                    self.save_conversation(session_id, messages, user_id)
        except Exception as e:
            print(f"Error adding message: {e}")

    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Retrieves the context dictionary for a given session ID."""
        key = f"{self.context_prefix}{session_id}"
        try:
            raw_context = self.redis.get(key)
            if raw_context:
                self.redis.expire(key, self.session_ttl_seconds) # Refresh TTL
                try:
                    context_str = raw_context.decode('utf-8') if isinstance(raw_context, bytes) else raw_context
                    return json.loads(context_str)
                except json.JSONDecodeError as json_err:
                    print(f"Error decoding context for {session_id} from key {key}: {json_err} - Data: '{str(raw_context)[:100]}...'")
                    return self._get_default_context()
            else:
                return self._get_default_context()
        except redis.RedisError as e:
            print(f"Redis error getting context for {session_id}: {e}")
            return self._get_default_context()
        except Exception as e:
            print(f"Unexpected error getting context for {session_id}: {e}")
            return self._get_default_context()

    def save_context(self, session_id: str, context: Dict[str, Any]):
        """Saves the context dictionary for a session ID with TTL."""
        key = f"{self.context_prefix}{session_id}"
        if not isinstance(context, dict):
            print(f"Error: Invalid context format provided for session {session_id}: {type(context)}")
            return
        try:
            # Attempt to serialize with default=str for common non-serializable types like datetime
            context_json = json.dumps(context, default=str)
            self.redis.setex(key, self.session_ttl_seconds, context_json)
        except redis.RedisError as e:
            print(f"Redis error saving context for {session_id}: {e}")
        except TypeError as e:
            # Log keys and types more robustly if default=str didn't help
            unserializable_info = {}
            for k, v in context.items():
                try:
                    json.dumps({k: v}, default=str)
                except TypeError:
                    unserializable_info[k] = f"{type(v).__name__}: {str(v)[:50]}..." # Show type and partial value
            print(f"Serialization error saving context for {session_id}: {e}")
            print(f"Keys with potentially unserializable types: {unserializable_info}")
        except Exception as e:
            print(f"Unexpected error saving context for {session_id}: {e}")

    def delete_session(self, session_id: str) -> bool:
        """Deletes all Redis data associated with a session ID."""
        message_key = f"{self.message_prefix}{session_id}"
        context_key = f"{self.context_prefix}{session_id}"
        session_index_key = f"{self.session_index_prefix}{session_id}"
        deleted_count = 0
        try:
            # Get all conversation IDs for this session
            conversation_ids = self.redis.smembers(session_index_key)
            
            pipe = self.redis.pipeline()
            
            # Delete message and context data
            pipe.unlink(message_key, context_key)
            
            # Delete each conversation and remove from indices
            if conversation_ids:
                for conv_id_bytes in conversation_ids:
                    conv_id = conv_id_bytes.decode('utf-8') if isinstance(conv_id_bytes, bytes) else conv_id_bytes
                    key = f"{self.conversation_prefix}{conv_id}"
                    
                    # Check if this conversation has a user_id to remove from user index
                    user_id_bytes = self.redis.hget(key, "user_id")
                    if user_id_bytes:
                        user_id = user_id_bytes.decode('utf-8') if isinstance(user_id_bytes, bytes) else user_id_bytes
                        pipe.srem(f"{self.user_index_prefix}{user_id}", conv_id)
                    
                    # Remove from global index
                    pipe.zrem(self.global_index_key, conv_id)
                    
                    # Delete the conversation
                    pipe.delete(key)
            
            # Delete the session index itself
            pipe.delete(session_index_key)
            
            # Execute all commands
            results = pipe.execute()
            
            # Sum up deletion results
            deleted_count = sum(1 for res in results if res)
            
            print(f"Deleted {deleted_count} Redis keys for session {session_id}")
            return deleted_count > 0
        except redis.RedisError as e:
            print(f"Redis error deleting session {session_id}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error deleting session {session_id}: {e}")
            return False

    def _get_default_context(self) -> Dict[str, Any]:
        """Returns the standard default context structure."""
        return {
            "last_source": None, "last_destination": None, "last_date": None,
            "last_boarding_point": None, "last_dropping_point": None,
            "ticket_count": 1, "selected_bus": None, "selected_seats": [],
            "user_location": None, "language": "english",
            "user_requested_source": None, "user_requested_destination": None,
            "auth": None, "user": None, "user_profile": None,
            "last_trips": [], "reverse_direction": False, "gender": None,
        }

    def get_conversations_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves summaries of conversations for a given session ID."""
        session_index_key = f"{self.session_index_prefix}{session_id}"
        conversations = []
        try:
            conversation_ids = self.redis.smembers(session_index_key)
            if not conversation_ids:
                return []

            for conv_id_bytes in conversation_ids:
                conv_id = conv_id_bytes.decode('utf-8') if isinstance(conv_id_bytes, bytes) else conv_id_bytes
                conversation = self.get_conversation(conv_id)
                if conversation:
                    first_user_query = ""
                    if conversation.get("messages"):
                         user_msgs = [m.get("content", "") for m in conversation["messages"] if m.get("role") == "user"]
                         if user_msgs:
                              first_user_query = user_msgs[0][:100] + ('...' if len(user_msgs[0]) > 100 else '')

                    summary = {
                        "conversation_id": conversation.get("conversation_id", conv_id),
                        "session_id": conversation.get("session_id", session_id),
                        "timestamp": conversation.get("timestamp", "unknown"),
                        "user_query": first_user_query,
                        "message_count": len(conversation.get("messages", [])),
                        "messages": conversation.get("messages")
                    }
                    conversations.append(summary)

            # Sort conversations by timestamp (newest first)
            conversations.sort(key=lambda x: x.get("timestamp", "0"), reverse=True)
            return conversations
        except redis.RedisError as e:
            print(f"Redis error getting conversations for session {session_id}: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error getting conversations for session {session_id}: {e}")
            return []

    def get_all_conversations(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieves summaries of all conversations, sorted by time (newest first)."""
        conversations = []
        try:
            # Get conversation IDs sorted by score (timestamp) in reverse (newest first)
            # ZREVRANGE returns list of bytes
            conversation_ids_bytes = self.redis.zrevrange(self.global_index_key, offset, offset + limit - 1)
            if not conversation_ids_bytes:
                return []

            for conv_id_bytes in conversation_ids_bytes:
                conv_id = conv_id_bytes.decode('utf-8') if isinstance(conv_id_bytes, bytes) else conv_id_bytes
                conversation = self.get_conversation(conv_id)
                if conversation:
                    first_user_query = ""
                    if conversation.get("messages"):
                         user_msgs = [m.get("content", "") for m in conversation["messages"] if m.get("role") == "user"]
                         if user_msgs:
                              first_user_query = user_msgs[0][:100] + ('...' if len(user_msgs[0]) > 100 else '')

                    summary = {
                        "conversation_id": conversation.get("conversation_id", conv_id),
                        "session_id": conversation.get("session_id", "unknown"),
                        "timestamp": conversation.get("timestamp", "unknown"),
                        "user_query": first_user_query,
                        "message_count": len(conversation.get("messages", [])),
                    }
                    conversations.append(summary)
            return conversations
        except redis.RedisError as e:
            print(f"Redis error getting all conversations: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error getting all conversations: {e}")
            return []

    def delete_conversation(self, conversation_id: str) -> bool:
        """Deletes a specific conversation by ID from all indices."""
        key = f"{self.conversation_prefix}{conversation_id}"
        pipe = self.redis.pipeline()
        deleted = False
        try:
            # Get session_id and user_id first to remove from indices
            conv_data = self.redis.hgetall(key)
            if not conv_data:
                print(f"Conversation {conversation_id} not found.")
                return False
                
            # Convert bytes to strings
            conv_data = {k.decode('utf-8') if isinstance(k, bytes) else k: 
                         v.decode('utf-8') if isinstance(v, bytes) else v 
                         for k, v in conv_data.items()}
                
            # Remove from session index
            if "session_id" in conv_data:
                session_index_key = f"{self.session_index_prefix}{conv_data['session_id']}"
                pipe.srem(session_index_key, conversation_id)
                
            # Remove from user index
            if "user_id" in conv_data:
                user_index_key = f"{self.user_index_prefix}{conv_data['user_id']}"
                pipe.srem(user_index_key, conversation_id)

            # Remove from global sorted set
            pipe.zrem(self.global_index_key, conversation_id)
            # Delete the main conversation hash data
            pipe.unlink(key) # Use unlink for potentially faster deletion

            results = pipe.execute()
            # Check results to verify deletion
            deleted = any(res > 0 for res in results)
            if deleted:
                print(f"Deleted archived conversation {conversation_id}")
            else:
                print(f"Conversation {conversation_id} not found or already deleted.")
            return deleted
        except redis.RedisError as e:
            print(f"Redis error deleting conversation {conversation_id}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error deleting conversation {conversation_id}: {e}")
            return False

    def delete_session_conversations(self, session_id: str) -> int:
        """Deletes all archived conversations associated with a session ID."""
        session_index_key = f"{self.session_index_prefix}{session_id}"
        deleted_count = 0
        try:
            conversation_ids_bytes = self.redis.smembers(session_index_key)
            if not conversation_ids_bytes:
                print(f"No archived conversations found for session {session_id}")
                return 0

            conv_ids_to_delete = [cid.decode('utf-8') if isinstance(cid, bytes) else cid for cid in conversation_ids_bytes]
            
            # For each conversation, get user_id to clean up user indices too
            for conv_id in conv_ids_to_delete:
                key = f"{self.conversation_prefix}{conv_id}"
                user_id_bytes = self.redis.hget(key, "user_id")
                if user_id_bytes:
                    user_id = user_id_bytes.decode('utf-8') if isinstance(user_id_bytes, bytes) else user_id_bytes
                    # Remove from user index
                    self.redis.srem(f"{self.user_index_prefix}{user_id}", conv_id)
            
            keys_to_delete = [f"{self.conversation_prefix}{cid}" for cid in conv_ids_to_delete]

            pipe = self.redis.pipeline()
            # Remove from global index
            if conv_ids_to_delete:
                pipe.zrem(self.global_index_key, *conv_ids_to_delete)
            # Delete individual conversation hashes
            if keys_to_delete:
                pipe.unlink(*keys_to_delete) # Use unlink
            # Delete the session index itself
            pipe.unlink(session_index_key) # Use unlink

            results = pipe.execute()
            # Calculate deleted count
            deleted_count = sum(res for res in results if isinstance(res, int) and res > 0)
            print(f"Deleted {deleted_count} archived conversations for session {session_id}")
            return deleted_count
        except redis.RedisError as e:
            print(f"Redis error deleting session conversations for {session_id}: {e}")
            return 0
        except Exception as e:
            print(f"Unexpected error deleting session conversations for {session_id}: {e}")
            return 0
            
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any], ttl_seconds: int = 86400) -> bool:
        """
        Stores or updates user profile data in Redis.
        
        Args:
            user_id: The unique identifier for the user
            profile_data: Dictionary containing user profile information
            ttl_seconds: Time to live for the profile data in seconds (default: 1 day)
            
        Returns:
            bool: Success status
        """
        try:
            profile_key = f"fresh_bus:user_profile:{user_id}"
            
            # Serialize the profile data
            profile_json = json.dumps(profile_data, default=str)
            
            # Store with expiration
            self.redis.setex(profile_key, ttl_seconds, profile_json)
            
            print(f"Updated profile for user {user_id}")
            return True
        except Exception as e:
            print(f"Error updating user profile for {user_id}: {e}")
            return False
            
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves user profile data from Redis.
        
        Args:
            user_id: The unique identifier for the user
            
        Returns:
            Optional[Dict]: The user profile data or None if not found
        """
        try:
            profile_key = f"fresh_bus:user_profile:{user_id}"
            profile_data = self.redis.get(profile_key)
            
            if not profile_data:
                return None
                
            # Decode bytes if needed
            profile_str = profile_data.decode('utf-8') if isinstance(profile_data, bytes) else profile_data
            profile = json.loads(profile_str)
            
            # Refresh TTL
            self.redis.expire(profile_key, 86400)
            
            return profile
        except Exception as e:
            print(f"Error retrieving user profile for {user_id}: {e}")
            return None