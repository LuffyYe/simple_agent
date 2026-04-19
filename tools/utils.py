"""
Utility Functions Module
"""

import os
import json
import re
import hashlib
from typing import Dict, List, Optional
from datetime import datetime

QUERY_ALIAS_EXPANSIONS = {
    r"\bnda\b": "non disclosure agreement",
    r"\bndas\b": "non disclosure agreements",
    r"\bdoctor note\b": "doctor's note",
    r"\bdoctor note[s]?\b": "doctor's note",
    r"\bvacation\b": "paid leave vacation",
    r"\bpto\b": "paid leave",
    r"\bremote work\b": "remote flexible work",
    r"\bwork from home\b": "remote flexible work",
    r"\bwfh\b": "remote flexible work",
    r"\bcoding style\b": "coding style guide",
    r"\bstyle guide\b": "coding style guide",
    r"\bsecurity incident[s]?\b": "security incidents",
}

# ==================== File Operation Utilities ====================

def read_markdown_file(filepath: str) -> str:
    """Read content from a Markdown file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Failed to read file {filepath}: {e}")
        return ""

def read_json_file(filepath: str) -> Optional[Dict]:
    """Read content from a JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to read JSON file {filepath}: {e}")
        return None

def write_json_file(filepath: str, data: Dict, indent: int = 2) -> bool:
    """Write data to a JSON file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Failed to write JSON file {filepath}: {e}")
        return False

def list_markdown_files(directory: str) -> List[str]:
    """List all Markdown files in a directory"""
    md_files = []
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.md', '.markdown')):
                    md_files.append(os.path.join(root, file))
    except Exception as e:
        print(f"Failed to list files in {directory}: {e}")
    return md_files

def ensure_directory(path: str) -> str:
    """Ensure directory exists and return absolute path"""
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

# ==================== Text Processing Utilities ====================

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

def strip_markdown(text: str) -> str:
    """Remove common Markdown syntax while preserving readable text."""
    if not text:
        return ""

    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]*)`', r'\1', text)
    text = re.sub(r'!\[[^\]]*\]\([^)]+\)', ' ', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'^[>\-\*\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'[*_~]', '', text)
    return text

def normalize_query_text(text: str) -> str:
    """Normalize user queries for routing and lexical retrieval."""
    if not text:
        return ""

    normalized = text
    normalized = strip_markdown(normalized)
    normalized = normalized.lower()

    for pattern, replacement in QUERY_ALIAS_EXPANSIONS.items():
        normalized = re.sub(pattern, replacement, normalized)

    normalized = re.sub(r"[^a-z0-9\s'/]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()

def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON object from text"""
    try:
        # Try direct parsing
        return json.loads(text)
    except:
        pass
    
    try:
        # Try extracting JSON block
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    try:
        # Try extracting JSON from code block
        code_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if code_match:
            return json.loads(code_match.group(1))
    except:
        pass
    
    return None

def format_response(response: str, source: str, metadata: Optional[Dict] = None) -> str:
    """Format response with source information"""
    result = response
    
    # Add source label
    source_labels = {
        "rag": "Company Internal Knowledge Base",
        "search": "Web Search",
        "intrinsic": "AI Model Intrinsic Knowledge"
    }
    label = source_labels.get(source, source)
    
    result += f"\n\n >>>Source: {label}"
    
    # Add metadata if available
    if metadata:
        if metadata.get("confidence"):
            result += f" (Confidence: {metadata['confidence']:.0%})"
        if metadata.get("documents"):
            result += f"\n >>>Reference Documents: {', '.join(metadata['documents'])}"
    
    return result

# ==================== Hash and ID Utilities ====================

def generate_hash(text: str, length: int = 8) -> str:
    """Generate hash value for text"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:length]

def generate_document_id(content: str, filename: Optional[str] = None) -> str:
    """Generate document ID"""
    if filename:
        base = f"{filename}:{generate_hash(content)}"
    else:
        base = generate_hash(content)
    return f"doc_{base}"

# ==================== Time Utilities ====================

def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_date() -> str:
    """Get current date string"""
    return datetime.now().strftime("%Y-%m-%d")

# ==================== Logging Utilities ====================

class Logger:
    """Simple logging utility"""
    
    def __init__(self, name: str = "Agent", log_file: Optional[str] = None):
        self.name = name
        self.log_file = log_file
        
        if log_file:
            ensure_directory(os.path.dirname(log_file))
    
    def _format(self, level: str, message: str) -> str:
        return f"[{get_timestamp()}] [{self.name}] [{level}] {message}"
    
    def info(self, message: str):
        log = self._format("INFO", message)
        # print(f"[INFO]  {message}")
        self._write(log)
    
    def warning(self, message: str):
        log = self._format("WARNING", message)
        print(f"[WARNING]  {message}")
        self._write(log)
    
    def error(self, message: str):
        log = self._format("ERROR", message)
        print(f"[ERROR] {message}")
        self._write(log)
    
    def success(self, message: str):
        log = self._format("SUCCESS", message)
        # print(f"[SUCCESS] {message}")
        self._write(log)
    
    def _write(self, log: str):
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log + '\n')
            except:
                pass

# ==================== Configuration Utilities ====================

def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    """Load environment variables from .env file"""
    env_vars = {}
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return env_vars

def get_env(key: str, default: str = "") -> str:
    """Get environment variable value"""
    return os.getenv(key, default)

# ==================== Conversation History Utilities ====================

class ConversationHistory:
    """Conversation history management"""
    
    def __init__(self, max_turns: int = 10):
        self.history: List[Dict] = []
        self.max_turns = max_turns
    
    def add(self, user: str, assistant: str, metadata: Optional[Dict] = None):
        """Add a conversation record"""
        entry = {
            "timestamp": get_timestamp(),
            "user": user,
            "assistant": assistant,
            "metadata": metadata or {}
        }
        self.history.append(entry)
        
        # Limit history length
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
    
    def get_context(self, turns: int = 3) -> str:
        """Get last N turns as context"""
        recent = self.history[-turns:]
        context_parts = []
        for entry in recent:
            context_parts.append(f"User: {entry['user']}")
            context_parts.append(f"Assistant: {entry['assistant']}")
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear conversation history"""
        self.history = []
    
    def to_dict(self) -> Dict:
        """Export history as dictionary"""
        return {
            "history": self.history,
            "count": len(self.history)
        }
    
    def save(self, filepath: str):
        """Save history to file"""
        write_json_file(filepath, self.to_dict())
    
    def load(self, filepath: str):
        """Load history from file"""
        data = read_json_file(filepath)
        if data:
            self.history = data.get("history", [])

# ==================== Performance Monitoring Utilities ====================

class PerformanceMonitor:
    """Performance monitoring utility"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.counts: Dict[str, int] = {}
    
    def record(self, metric_name: str, value: float):
        """Record a metric value"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
            self.counts[metric_name] = 0
        self.metrics[metric_name].append(value)
        self.counts[metric_name] += 1
    
    def get_stats(self, metric_name: str) -> Dict:
        """Get statistics for a metric"""
        if metric_name not in self.metrics:
            return {"error": "Metric not found"}
        
        values = self.metrics[metric_name]
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "last": values[-1] if values else None
        }
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all metrics"""
        return {name: self.get_stats(name) for name in self.metrics}
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {}
        self.counts = {}
