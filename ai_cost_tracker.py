# ai_cost_tracker.py

import time
import json
import os
from datetime import datetime
from typing import Dict, Any

class AICostTracker:
    """Track and report AI usage costs."""
    
    # Approximate cost per 1000 tokens (as of April 2025)
    PROVIDER_COSTS = {
        "claude": {
            "claude-3-5-sonnet-20240620": {
                "input": 0.003,
                "output": 0.015
            }
        },
        "gemini": {
            "gemini-2.0-flash": {
                "input": 0.0005,
                "output": 0.0015
            }
        }
    }
    
    def __init__(self, log_path: str = "./ai_usage_logs"):
        self.log_path = log_path
        self.usage_data = []
        
        # Create log directory if it doesn't exist
        os.makedirs(log_path, exist_ok=True)
        
        # Load existing data for today
        self.today_file = os.path.join(
            self.log_path, 
            f"usage_{datetime.now().strftime('%Y-%m-%d')}.json"
        )
        self._load_today_data()
    
    def _load_today_data(self):
        """Load existing usage data for today."""
        try:
            if os.path.exists(self.today_file):
                with open(self.today_file, 'r') as f:
                    self.usage_data = json.load(f)
        except Exception as e:
            print(f"Error loading today's usage data: {e}")
    
    def _save_today_data(self):
        """Save current usage data to file."""
        try:
            with open(self.today_file, 'w') as f:
                json.dump(self.usage_data, f, indent=2)
        except Exception as e:
            print(f"Error saving today's usage data: {e}")
    
    def log_request(self, provider: str, model: str, input_tokens: int, output_tokens: int, 
                   session_id: str, query_type: str = "chat"):
        """Log a request and its token usage."""
        timestamp = datetime.now().isoformat()
        
        # Calculate cost
        input_cost = 0
        output_cost = 0
        
        try:
            costs = self.PROVIDER_COSTS.get(provider.lower(), {}).get(model, None)
            if costs:
                input_cost = (input_tokens / 1000) * costs["input"]
                output_cost = (output_tokens / 1000) * costs["output"]
        except Exception:
            # Use default costs if provider/model not found
            input_cost = (input_tokens / 1000) * 0.001
            output_cost = (output_tokens / 1000) * 0.002
            
        total_cost = input_cost + output_cost
        
        usage_entry = {
            "timestamp": timestamp,
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "session_id": session_id,
            "query_type": query_type
        }
        
        self.usage_data.append(usage_entry)
        self._save_today_data()
        
        return usage_entry
    
    def get_daily_summary(self):
        """Get usage summary for today."""
        if not self.usage_data:
            return {
                "date": datetime.now().strftime('%Y-%m-%d'),
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0,
                "providers": {}
            }
        
        summary = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "total_requests": len(self.usage_data),
            "total_tokens": sum(entry["total_tokens"] for entry in self.usage_data),
            "total_cost": sum(entry["total_cost"] for entry in self.usage_data),
            "providers": {}
        }
        
        # Aggregate by provider
        for entry in self.usage_data:
            provider = entry["provider"]
            if provider not in summary["providers"]:
                summary["providers"][provider] = {
                    "requests": 0,
                    "total_tokens": 0,
                    "total_cost": 0,
                    "models": {}
                }
            
            provider_data = summary["providers"][provider]
            provider_data["requests"] += 1
            provider_data["total_tokens"] += entry["total_tokens"]
            provider_data["total_cost"] += entry["total_cost"]
            
            # Aggregate by model
            model = entry["model"]
            if model not in provider_data["models"]:
                provider_data["models"][model] = {
                    "requests": 0,
                    "total_tokens": 0,
                    "total_cost": 0
                }
            
            model_data = provider_data["models"][model]
            model_data["requests"] += 1
            model_data["total_tokens"] += entry["total_tokens"]
            model_data["total_cost"] += entry["total_cost"]
        
        return summary
from typing import Dict, Any, List, Optional

class AICostTracker:
    """
    Tracks usage and costs for different AI providers
    """
    
    def __init__(self, log_file: str = "ai_usage_logs.json"):
        self.log_file = log_file
        self.cost_per_1k_tokens = {
            # Claude models
            "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            
            # Gemini models
            "gemini-2.0-flash": {"input": 0.00035, "output": 0.00035},
            "gemini-2.0-pro": {"input": 0.0007, "output": 0.0007},
            "gemini-1.5-pro": {"input": 0.0007, "output": 0.0007},
            "gemini-1.5-flash": {"input": 0.00035, "output": 0.00035}
        }
        
        # Load existing logs if available
        self.logs = self._load_logs()
    
    def _load_logs(self) -> Dict[str, Any]:
        """Load existing logs from file"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading logs: {e}")
                return {"requests": [], "daily_summaries": {}}
        return {"requests": [], "daily_summaries": {}}
    
    def _save_logs(self) -> None:
        """Save logs to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
        except Exception as e:
            print(f"Error saving logs: {e}")
    
    def log_request(
        self, 
        provider: str, 
        model: str, 
        input_tokens: int, 
        output_tokens: int,
        session_id: str,
        query_type: str = "chat"
    ) -> Dict[str, Any]:
        """
        Log a request and calculate its cost
        
        Args:
            provider: The AI provider (e.g., "claude", "gemini")
            model: The model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            session_id: User session ID
            query_type: Type of query (e.g., "chat", "embedding")
            
        Returns:
            Dict with usage data including costs
        """
        # Calculate costs
        input_cost = 0
        output_cost = 0
        
        if model in self.cost_per_1k_tokens:
            input_cost = (input_tokens / 1000) * self.cost_per_1k_tokens[model]["input"]
            output_cost = (output_tokens / 1000) * self.cost_per_1k_tokens[model]["output"]
        
        total_cost = input_cost + output_cost
        
        # Create log entry
        timestamp = datetime.now().isoformat()
        today = date.today().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "date": today,
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "session_id": session_id,
            "query_type": query_type
        }
        
        # Add to logs
        self.logs["requests"].append(log_entry)
        
        # Update daily summary
        if today not in self.logs["daily_summaries"]:
            self.logs["daily_summaries"][today] = {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0,
                "models": {}
            }
        
        daily = self.logs["daily_summaries"][today]
        daily["total_requests"] += 1
        daily["total_tokens"] += (input_tokens + output_tokens)
        daily["total_cost"] += total_cost
        
        # Update model-specific stats
        model_key = f"{provider}:{model}"
        if model_key not in daily["models"]:
            daily["models"][model_key] = {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0
            }
        
        model_stats = daily["models"][model_key]
        model_stats["requests"] += 1
        model_stats["input_tokens"] += input_tokens
        model_stats["output_tokens"] += output_tokens
        model_stats["cost"] += total_cost
        
        # Save logs
        self._save_logs()
        
        return log_entry
    
    def get_daily_summary(self, day: Optional[str] = None) -> Dict[str, Any]:
        """
        Get usage summary for a specific day
        
        Args:
            day: ISO format date string (YYYY-MM-DD), defaults to today
            
        Returns:
            Dict with usage summary
        """
        if day is None:
            day = date.today().isoformat()
            
        if day in self.logs["daily_summaries"]:
            return self.logs["daily_summaries"][day]
        
        return {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0,
            "models": {}
        }
    
    def get_date_range_summary(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get usage summary for a date range
        
        Args:
            start_date: ISO format date string (YYYY-MM-DD)
            end_date: ISO format date string (YYYY-MM-DD)
            
        Returns:
            Dict with usage summary
        """
        summary = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0,
            "models": {}
        }
        
        for day, daily in self.logs["daily_summaries"].items():
            if start_date <= day <= end_date:
                summary["total_requests"] += daily["total_requests"]
                summary["total_tokens"] += daily["total_tokens"]
                summary["total_cost"] += daily["total_cost"]
                
                # Merge model stats
                for model_key, model_stats in daily["models"].items():
                    if model_key not in summary["models"]:
                        summary["models"][model_key] = {
                            "requests": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "cost": 0
                        }
                    
                    summary["models"][model_key]["requests"] += model_stats["requests"]
                    summary["models"][model_key]["input_tokens"] += model_stats["input_tokens"]
                    summary["models"][model_key]["output_tokens"] += model_stats["output_tokens"]
                    summary["models"][model_key]["cost"] += model_stats["cost"]
        
        return summary
