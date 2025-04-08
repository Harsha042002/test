# ai_providers.py

import os
import json
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, AsyncGenerator, Optional
import tiktoken
import logging
from anthropic import Anthropic
# Update this line in your ai_providers.py
import google.generativeai as genai

# Import Google Generative AI
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_providers")

class AIProvider(ABC):
    """Abstract base class for AI model providers."""
    
    @abstractmethod
    async def initialize(self):
        """Initialize the provider client."""
        pass
    
    @abstractmethod
    async def generate_stream(self, 
                             prompt: str, 
                             system_message: str, 
                             messages: List[Dict[str, str]], 
                             max_tokens: int = 2000,
                             temperature: float = 0.7) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response from the AI model."""
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in the provided text."""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Clean up resources."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the provider."""
        pass
    
    @property
    @abstractmethod
    def model(self) -> str:
        """Return the model name."""
        pass


class ClaudeProvider(AIProvider):
    """Claude AI provider implementation."""
    
    def __init__(self, api_key: str = None, model: str = "claude-3-5-sonnet-20240620"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        self._model = model
        self.client = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Claude's encoding
    
    async def initialize(self):
        """Initialize the Anthropic client."""
        self.client = Anthropic(api_key=self.api_key)
        logger.info(f"Initialized Claude provider with model: {self._model}")
    
    async def generate_stream(self, 
                             prompt: str, 
                             system_message: str, 
                             messages: List[Dict[str, str]], 
                             max_tokens: int = 2000,
                             temperature: float = 0.7) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response from Claude."""
        if not self.client:
            await self.initialize()
        
        try:
            with self.client.messages.stream(
                model=self._model,
                max_tokens=max_tokens,
                system=system_message,
                messages=messages,
                temperature=temperature
            ) as stream:
                response_text = ""
                for text in stream.text_stream:
                    response_text += text
                    yield {"text": text, "done": False}
                
                # Final yield with full response and done flag
                yield {"text": "", "done": True, "complete_response": response_text}
                
        except Exception as e:
            logger.error(f"Error generating Claude response: {str(e)}")
            yield {"error": str(e), "done": True}
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens using Claude's tokenizer."""
        return len(self.tokenizer.encode(text))
    
    async def cleanup(self):
        """Clean up resources."""
        # Currently no cleanup needed for Claude client
        pass
    
    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "Claude"
    
    @property
    def model(self) -> str:
        """Return the model name."""
        return self._model


class GeminiProvider(AIProvider):
    """Google Gemini AI provider implementation."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-pro"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required")
        self._model = model
        self.client = None
        # Simple token estimator for Gemini (approximation)
        self.token_multiplier = 0.25  # Rough estimate: 4 chars = 1 token
    
    async def initialize(self):
        """Initialize the Gemini client."""
        # Updated initialization for newer versions of the Google Generative AI client
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model_name=self._model)
        logger.info(f"Initialized Gemini provider with model: {self._model}")
    
    async def generate_stream(self, 
                             prompt: str, 
                             system_message: str, 
                             messages: List[Dict[str, str]], 
                             max_tokens: int = 2000,
                             temperature: float = 0.7) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response from Gemini."""
        if not self.client:
            await self.initialize()
        
        try:
            # Convert messages to Gemini format
            import google.generativeai as genai
            
            # Create content array for the conversation
            gemini_content = []
            
            # Add system message as first user message if provided
            if system_message:
                gemini_content.append({
                    "role": "user",
                    "parts": [{"text": system_message}]
                })
                gemini_content.append({
                    "role": "model",
                    "parts": [{"text": "I'll help you with that."}]
                })
            
            # Add past messages
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                gemini_content.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })
            
            # Add current prompt as the last user message
            gemini_content.append({
                "role": "user",
                "parts": [{"text": prompt}]
            })
            
            # Create a wrapper to run the synchronous Gemini API in a thread pool
            loop = asyncio.get_event_loop()
            
            # Set up safety settings and generation config
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
                "top_k": 64
            }
            
            # Stream the response
            stream = await loop.run_in_executor(
                None,
                lambda: self.client.generate_content(
                    contents=gemini_content,
                    generation_config=generation_config,
                    stream=True
                )
            )
            
            # Process streaming response
            full_response = ""
            async for chunk in self._aiter_stream(stream):
                if hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                    full_response += chunk_text
                    yield {"text": chunk_text, "done": False}
            
            # Final yield with done flag
            yield {"text": "", "done": True, "complete_response": full_response}
            
        except Exception as e:
            logger.error(f"Error generating Gemini response: {str(e)}")
            yield {"error": str(e), "done": True}
    
    async def _aiter_stream(self, stream):
        """Helper method to convert synchronous iterator to async iterator."""
        loop = asyncio.get_event_loop()
        for chunk in stream:
            yield await loop.run_in_executor(None, lambda: chunk)
    
    async def count_tokens(self, text: str) -> int:
        """Approximate token count for Gemini."""
        # This is a rough approximation, as Gemini doesn't expose its tokenizer
        return int(len(text) * self.token_multiplier)
    
    async def cleanup(self):
        """Clean up resources."""
        # Currently no cleanup needed for Gemini client
        pass
    
    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "Gemini"
    
    @property
    def model(self) -> str:
        """Return the model name."""
        return self._model


class AIProviderFactory:
    """Factory for creating AI providers."""
    
    # Available models by provider - simplified version with just the two models
    AVAILABLE_MODELS = {
        "claude": ["claude-3-5-sonnet-20240620"],
        "gemini": ["gemini-2.0-flash"]
    }
    
    # User-friendly names for models
    MODEL_DISPLAY_NAMES = {
        "claude-3-5-sonnet-20240620": "Claude 3.5 Sonnet",
        "gemini-2.0-flash": "Gemini 2.0 Flash"
    }
    
    # Relative cost indicators (1-10 scale)
    MODEL_COST_RATINGS = {
        "claude-3-5-sonnet-20240620": 6,
        "gemini-2.0-flash": 2
    }
    
    @staticmethod
    def create_provider(provider_name: str, api_key: Optional[str] = None, model: Optional[str] = None) -> AIProvider:
        """Create and return an AI provider based on name."""
        provider_name = provider_name.lower()
        
        # Validate provider name
        if provider_name not in ["claude", "gemini"]:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        # Use default model if none specified
        if not model:
            model = AIProviderFactory.AVAILABLE_MODELS[provider_name][0]
        
        # Validate model name
        if model not in AIProviderFactory.AVAILABLE_MODELS[provider_name]:
            raise ValueError(f"Unknown model '{model}' for provider '{provider_name}'")
        
        # Create appropriate provider
        if provider_name == "claude":
            return ClaudeProvider(api_key=api_key, model=model)
        elif provider_name == "gemini":
            return GeminiProvider(api_key=api_key, model=model)
    
    @staticmethod
    def get_available_providers():
        """Get list of available providers with their models."""
        providers = []
        
        for provider_name, models in AIProviderFactory.AVAILABLE_MODELS.items():
            provider_models = []
            
            for model in models:
                provider_models.append({
                    "id": model,
                    "name": AIProviderFactory.MODEL_DISPLAY_NAMES.get(model, model),
                    "cost_rating": AIProviderFactory.MODEL_COST_RATINGS.get(model, 5)
                })
            
            providers.append({
                "id": provider_name,
                "name": provider_name.capitalize(),
                "models": provider_models
            })
        
        return providers