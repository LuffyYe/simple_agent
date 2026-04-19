"""
Configuration Management
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass
class Config:
    """Agent Configuration"""
    
    # LLM Configuration
    llm_model: str = os.getenv("LLM_MODEL_ID", "gpt-4o-mini")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    
    # Search Configuration
    search_api_key: str = os.getenv("SEARCH_API_KEY", "")
    search_provider: str = os.getenv("SEARCH_PROVIDER", "serpapi")
    
    # Knowledge Base Configuration
    knowledge_base_path: str = os.getenv("KNOWLEDGE_BASE_PATH", "./knowledge_base")
    
    # Agent Configuration
    max_steps: int = int(os.getenv("MAX_STEPS", "5"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.3"))
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        return cls()
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.llm_api_key:
            print("Warning: LLM_API_KEY is not set")
            return False
        return True