import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class Config:
    """Configuration class to centralize all settings"""
    
    # Load environment variables
    load_dotenv()
    
    # API endpoints
    BASE_URL = os.getenv("BASE_URL", "https://api-stage.freshbus.com")
    BASE_URL_CUSTOMER = os.getenv("BASE_URL_CUSTOMER", "https://api-stage.freshbus.com")
    
    # Redis configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "tops-buffalo-19522.upstash.io")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "AUxCAAIjcDFjZGUyYmMyYWI2ZWU0YzE5Yjg5NDBiYWU5ZWE0ZWIxNXAxMA")
    REDIS_SSL = os.getenv("REDIS_SSL", "True").lower() in ("true", "1", "t", "yes")
    
    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
    FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    
    # AI Provider settings
    DEFAULT_AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini")
    DEFAULT_AI_MODEL = os.getenv("AI_PROVIDER_MODEL", "gemini-2.0-flash")
    
    # System prompt path
    DEFAULT_SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "./system_prompt/qa_prompt.txt")
    
    # Chroma DB settings
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    # Station IDs mapping
    STATIONS = {
        "hyderabad": 3,
        "vijayawada": 5,
        "bangalore": 7,
        "tirupati": 12,
        "chittoor": 130,
        "guntur": 13
    }
    
    # Seat categories
    SEAT_CATEGORIES = {
        "Premium": [5, 8, 9, 12, 13, 16, 17, 20, 24, 25, 28],
        "Reasonable": [1, 2, 3, 4, 6, 7, 10, 11, 14, 15, 18, 19, 21, 22, 23, 26, 27, 29, 30, 31, 32, 37, 40],
        "Low Reasonable": [33, 34, 35, 36, 38, 39],
        "Budget-Friendly": [41, 42, 43, 44]
    }
    
    # Seat positions
    WINDOW_SEATS = [1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44]
    AISLE_SEATS = [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43]
    
    # Language codes mapping for Fireworks API
    LANGUAGE_CODES = {
        "english": "en",
        "hindi": "hi",
        "telugu": "te",
        "kannada": "kn",
        "tamil": "ta",
        "malayalam": "ml",
        "tenglish": "te",
        "hinglish": "hi",
        "tanglish": "ta",
        "kanglish": "kn"
    }
    
    LANGUAGE_CODES_REVERSE = {v: k for k, v in LANGUAGE_CODES.items()}
    
    # HTTP client settings
    HTTP_TIMEOUT_TOTAL = int(os.getenv("HTTP_TIMEOUT_TOTAL", "30"))
    HTTP_TIMEOUT_CONNECT = int(os.getenv("HTTP_TIMEOUT_CONNECT", "10"))
    HTTP_TIMEOUT_SOCK_CONNECT = int(os.getenv("HTTP_TIMEOUT_SOCK_CONNECT", "10"))
    HTTP_TIMEOUT_SOCK_READ = int(os.getenv("HTTP_TIMEOUT_SOCK_READ", "10"))
    
    # Token limits
    DEFAULT_TOKEN_LIMIT = int(os.getenv("DEFAULT_TOKEN_LIMIT", "2000"))
    SYSTEM_PROMPT_TOKEN_LIMIT = int(os.getenv("SYSTEM_PROMPT_TOKEN_LIMIT", "3000"))
    
    # Cache settings
    CACHE_EXPIRY_SECONDS = int(os.getenv("CACHE_EXPIRY_SECONDS", "300"))  # 5 minutes
    
    @classmethod
    def get_redis_config(cls) -> Dict[str, Any]:
        """Get Redis configuration as a dictionary"""
        return {
            "host": cls.REDIS_HOST,
            "port": cls.REDIS_PORT,
            "password": cls.REDIS_PASSWORD,
            "decode_responses": True,
            "ssl": cls.REDIS_SSL
        }
    
    @classmethod
    def get_http_timeout_config(cls) -> Dict[str, int]:
        """Get HTTP timeout configuration as a dictionary"""
        return {
            "total": cls.HTTP_TIMEOUT_TOTAL,
            "connect": cls.HTTP_TIMEOUT_CONNECT,
            "sock_connect": cls.HTTP_TIMEOUT_SOCK_CONNECT,
            "sock_read": cls.HTTP_TIMEOUT_SOCK_READ
        }
