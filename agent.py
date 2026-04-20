"""
Company Assistant Agent - Built on hello-agents framework

Capabilities:
1. Intelligent routing: automatically select best knowledge source (RAG / search / intrinsic)
2. Safety filtering: block harmful or policy-violating queries
3. Clarification: proactively ask for clarification on ambiguous queries
4. Conversation memory: maintain context and provide statistics
"""

import os
import re
import time
from typing import Optional, Dict

# hello-agents framework imports
from hello_agents import HelloAgentsLLM, Config

# Local tool imports
from tools.router_tool import RouterTool
from tools.safety_tool import SafetyTool
from tools.search_tool import SearchTool
from tools.simple_rag import SimpleRAGTool

# Utility imports
from tools.utils import (
    Logger,
    ConversationHistory,
    PerformanceMonitor,
    format_response,
    extract_json_from_text,
    list_markdown_files,
    ensure_directory,
    get_timestamp
)


class CompanyAssistantAgent:
    """
    Company Assistant Agent
    
    Built on hello-agents with extended capabilities:
    - Intelligent query routing
    - Safety filtering
    - Clarification handling
    - Multi-source knowledge support (RAG / search / intrinsic knowledge)
    """
    
    def __init__(
        self,
        name: str = "CompanyAssistant",
        llm: Optional[HelloAgentsLLM] = None,
        config: Optional[Config] = None,
        knowledge_base_path: str = "./knowledge_base",
        enable_logging: bool = True,
        log_dir: str = "./logs"
    ):
        """
        Initialize CompanyAssistantAgent
        
        Args:
            name: Agent name
            llm: LLM instance
            config: Configuration object
            knowledge_base_path: Path to knowledge base
            enable_logging: Whether to enable logging
            log_dir: Directory for log files
        """

        # Basic configuration
        self.name = name
        self.llm = llm
        self.config = config
        self.knowledge_base_path = ensure_directory(knowledge_base_path)
        
        # Initialize tools
        self._init_tools()
        
        # Initialize utility components
        self._init_utils(enable_logging, log_dir)
        
        # Initialize RAG
        self._init_rag()
        
        # Print initialization info
        # self._print_init_info()
    
    def _init_tools(self):
        """Initialize all tools"""
        # print(" [INIT] Initializing tools...")
        
        # 1. Safety tool
        self.safety_tool = SafetyTool()
        
        # 2. Router tool
        self.router_tool = RouterTool(llm=self.llm)
        
        # 3. Search tool (if API is configured)
        search_api_key = os.getenv("SEARCH_API_KEY", "")
        self.search_tool = SearchTool(
            api_provider=os.getenv("SEARCH_PROVIDER", "serpapi"),
            api_key=search_api_key
        )
        if not search_api_key:
            print("[SEARCH] SearchTool registered in unavailable mode (missing SEARCH_API_KEY)")
        
    
    def _init_utils(self, enable_logging: bool, log_dir: str):
        """Initialize utility components"""
        
        # Logger
        if enable_logging:
            ensure_directory(log_dir)
            log_file = os.path.join(log_dir, f"agent_{get_timestamp().replace(':', '-')}.log")
            self.logger = Logger(name=self.name, log_file=log_file)
            self.logger.info(f"Agent initialized, log file: {log_file}")
        else:
            self.logger = Logger(name=self.name)
        
        # Conversation history
        self.history = ConversationHistory(max_turns=10)
        
        # Performance monitor
        self.monitor = PerformanceMonitor()
    
    def _init_rag(self):
        """Initialize RAG tool"""
        # print(" [RAG] Initializing RAG tool...")
        
        try:
            # Check knowledge base files
            md_files = list_markdown_files(self.knowledge_base_path)
            if not md_files:
                print(f"[RAG] Knowledge base directory is empty: {self.knowledge_base_path}")
                print(" [RAG] Please add Markdown files to the knowledge base directory")
            
            # Create RAG tool using SimpleRAGTool
            self.rag_tool = SimpleRAGTool(
                knowledge_base_path=self.knowledge_base_path
            )
            self.router_tool.set_retrieval_probe(self.rag_tool.probe)
            # print(f"[RAG] Knowledge base path: {self.knowledge_base_path}")
            # print(f" [RAG] Found {len(md_files)} document(s)")
            
        except ImportError as e:
            print(f"[RAG] RAGTool import failed: {e}")
            print(" [RAG] Please ensure hello-agents is properly installed")
            self.rag_tool = None
        except Exception as e:
            print(f"[RAG] RAG initialization failed: {e}")
            self.rag_tool = None
    
    def _print_init_info(self):
        """Print initialization completion info"""
        print(f"\n [INIT] {self.name} initialization complete")
        print(f" [INIT] Knowledge base: {self.knowledge_base_path}")
        print()
    
    def run(self, input_text: str, **kwargs) -> str:
        """
        Main entry point for processing user queries
        
        Flow:
        1. Safety check
        2. Clarification check
        3. Routing decision
        4. Query execution
        5. Response formatting
        6. History and performance logging
        
        Args:
            input_text: User input
            **kwargs: Additional parameters
        
        Returns:
            Response string
        """
        start_time = time.time()
        
        # Log the query
        self.logger.info(f"Received query: {input_text}")
        # print(f"\n[PROCESS] Processing query: {input_text}")
        
        try:
            # ========== Step 1: Safety Check ==========
            safety_result = self._check_safety(input_text)
            if safety_result.get("blocked", False):
                response = safety_result.get("message", "Query rejected")
                self.logger.warning(f"Query rejected: {safety_result.get('category')}")
                self._finalize(start_time, input_text, response, "blocked", safety_result)
                return response
            
            # ========== Step 2: Clarification Check ==========
            clarification = self._check_clarification(input_text)
            if clarification:
                self.logger.info(f"Clarification needed: {clarification}")
                self._finalize(start_time, input_text, clarification, "clarification")
                return clarification
            
            # ========== Step 3: Routing Decision ==========
            context = self.history.get_context(turns=3)
            route_result = self._route_query(input_text, context)
            source = route_result.get("source", "intrinsic")
            reason = route_result.get("reason", "")
            
            # print(f"[ROUTE] Routing decision: {source} (confidence: {confidence:.0%})")
            # print(f"[ROUTE] Reason: {reason}")
            self.logger.info(f"Routing decision: {source}, reason: {reason}")
            
            # ========== Step 4: Execute Query ==========
            response = self._execute_query(input_text, route_result)
            
            # ========== Step 5: Format Response ==========
            metadata = {
                "reason": reason,
                **route_result.get("metadata", {})
            }
            formatted_response = format_response(response, source, metadata)
            
            # ========== Step 6: Log History and Performance ==========
            self._finalize(start_time, input_text, formatted_response, source, route_result)
            
            return formatted_response
            
        except Exception as e:
            error_msg = f"[ERROR] Error processing query: {str(e)}"
            self.logger.error(f"Error: {str(e)}")
            print(f"   {error_msg}")
            return error_msg
    
    def _check_safety(self, query: str) -> Dict:
        """
        Safety check
        
        Uses SafetyTool to check if query is safe
        """
        # print("[SAFETY] Safety check...")
        
        # Use safety tool
        safety_result = self.safety_tool.run({"text": query})
        
        if not safety_result.get("safe", True):
            return {
                "blocked": True,
                "message": safety_result.get("message", "Query rejected"),
                "category": safety_result.get("category", "unknown")
            }
        
        # Optional: Secondary check using LLM
        if os.getenv("ENABLE_LLM_SAFETY_CHECK") and os.getenv("ENABLE_LLM_SAFETY_CHECK").lower() == "true":
            llm_safety = self._llm_safety_check(query)
            if llm_safety.get("blocked", False):
                return llm_safety
        
        # print("[SAFETY] Safety check passed")
        return {"blocked": False}
    
    def _llm_safety_check(self, query: str) -> Dict:
        """Perform safety check using LLM"""
        try:
            prompt = f"""
            You are a safety classifier.

            Your task is to determine whether the following user query is safe and appropriate.

            Rules:
            - If the query involves illegal, harmful, or dangerous activities → mark as unsafe
            - If the query is normal → mark as safe
            - Be strict and conservative

            Return ONLY valid JSON. Do not include any extra text.

            Format:
            {{"safe": true/false, "reason": "short explanation", "category": "category or null"}}

            Query:
            \"\"\"{query}\"\"\"
            """
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages)
            
            result = extract_json_from_text(response)
            if result and not result.get("safe", True):
                return {
                    "blocked": True,
                    "message": f"[SAFETY] Query rejected: {result.get('reason', 'Violates safety policy')}",
                    "category": result.get("category", "policy_violation")
                }
        except Exception as e:
            self.logger.warning(f"[SAFETY] LLM safety check failed: {e}")
        
        return {"blocked": False}
    
    def _check_clarification(self, query: str) -> Optional[str]:
        """
        Check if clarification is needed - Fixed version
        """
        # print("[CLARIFICATION] Clarification check...")
        
        CONCRETE_NOUNS = {
            "vacation", "holiday", "leave", "pto", "reimbursement",
            "salary", "bonus", "insurance", "benefit", "policy",
            "training", "promotion", "review", "performance",
            "remote work", "work remotely", "telecommute",
            "work from home", "wfh", "hybrid work", "hybrid",
            "onsite", "office work", "flexible hours", "flextime",
            "api", "sdk", "code", "repository", "database",
            "server", "endpoint", "token", "key", "password",
            "job", "position", "role", "visa", "product",
            "service", "price", "cost", "deadline", "schedule"
        }
        
        AMBIGUOUS_SINGLE_WORDS = {
            "what", "how", "why", "when", "where", "who",
            "this", "that", "it", "these", "those",
            "help", "info", "information"
        }
        
        PRONOUNS = {'it', 'this', 'that', 'these', 'those', 'they', 'them'}
        
        AMBIGUOUS_PATTERNS = [
            r'^how do i\b',
            r'^how can i\b',
            r'^what should i\b',
            r'^tell me\b',
            r'^i want to\b',
            r'^is it\s+(allowed|possible)\s*\?$',
            r'^can i\b\s*\?$',
            r'^do i\b',
            r'^how does it\b',
            r'^what is it\b',
        ]
        
        def _has_concrete_information(q: str) -> bool:
            q_lower = q.lower()
            
            for noun in CONCRETE_NOUNS:
                if noun in q_lower:
                    return True
            
            concrete_patterns = [
                r'\bwork\s+(remotely|from home|hybrid|onsite)\b',
                r'\bapply\s+for\s+\w+',
                r'\bsubmit\s+\w+',
                r'\brequest\s+\w+',
            ]
            
            for pattern in concrete_patterns:
                if re.search(pattern, q_lower):
                    return True
            
            return False
        
        def _is_ambiguous(q: str) -> tuple[bool, str]:
            q_stripped = q.strip().lower()
            words = q_stripped.split()
            
            if not q_stripped:
                return True, "empty_query"
            
            if len(words) <= 2:
                return True, "too_short"
            
            if q_stripped in AMBIGUOUS_SINGLE_WORDS:
                return True, "ambiguous_single_word"
            
            has_concrete = _has_concrete_information(q_stripped)
            
            if has_concrete:
                return False, "has_concrete_info"
            
            for pattern in AMBIGUOUS_PATTERNS:
                if re.match(pattern, q_stripped):
                    return True, "ambiguous_pattern"
            
            has_pronoun = any(re.search(rf'\b{p}\b', q_stripped) for p in PRONOUNS)
            if has_pronoun and len(words) <= 6:
                return True, "pronoun_without_context"
            
            return False, "clear"
        
        query_stripped = query.strip()
        is_ambiguous, reason = _is_ambiguous(query_stripped)
        
        if is_ambiguous:
            # print(f"[CLARIFICATION] Clarification needed: {reason}")
            return self._generate_clarification_message(reason, query_stripped)
        
        if len(query_stripped) < 15 and not _has_concrete_information(query_stripped):
            llm_result = self._llm_check_clarification(query)
            if llm_result:
                # print(f"[CLARIFICATION] Clarification needed: LLM judgment")
                return llm_result
        
        # print(f"[CLARIFICATION] Query is clear")
        return None
    
    def _generate_clarification_message(self, reason: str, query: str) -> str:
        """Generate clarification message based on reason"""
        query_lower = query.lower()
        
        if reason == "too_short" or reason == "ambiguous_single_word":
            return "Your query is too brief. Please provide more details about what you'd like to know."
        
        if reason == "context_required_verb_without_noun":
            if "apply" in query_lower:
                return "What would you like to apply for? Please specify, e.g., apply for vacation, apply for reimbursement, apply for a position."
            if "allowed" in query_lower or "permitted" in query_lower:
                return "What would you like to know is allowed? Please specify the action or item."
            if "get" in query_lower:
                return "What would you like to get? Please specify the information or resource you need."
            return "Your query is missing key information. Please specify what you're referring to."
        
        if reason == "pronoun_without_context":
            return "Your query contains a pronoun without clear reference. Please specify what you're referring to."
        
        if reason == "ambiguous_pattern":
            return "Your query is not specific enough. Please provide more context or details."
        
        return "Your query is unclear. Please provide more context or specific information."
    
    def _llm_check_clarification(self, query: str) -> Optional[str]:
        """Check if clarification is needed using LLM"""
        try:
            prompt = f"""
            You are a query clarity checker.

            Determine whether the user query is clear enough to answer.

            Rules:
            - If the query lacks context or is ambiguous → ask a clarification question
            - If the query is clear → return CLEAR

            Output format (strict):
            - If unclear: CLARIFY: <your question>
            - If clear: CLEAR

            Query:
            \"\"\"{query}\"\"\"
            """
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages).strip()
            
            if response.startswith("CLARIFY:"):
                clarification = response.replace("CLARIFY:", "").strip()
                return f"Your query is not clear enough. {clarification}"
        except Exception as e:
            self.logger.warning(f"[CLARIFICATION] LLM clarification check failed: {e}")
        
        return None
    
    def _route_query(self, query: str, context: str = "") -> Dict:
        """
        Routing decision
        
        Uses RouterTool to determine the best knowledge source
        """
        # print(f"[ROUTING] Routing decision...")
        
        # Use router tool
        route_result = self.router_tool.run({
            "query": query,
            "context": context
        })
        
        return route_result
    
    def _execute_query(self, query: str, route: Dict) -> str:
        """
        Execute query based on routing decision
        """
        source = route.get("source", "intrinsic")
        
        if source == "rag":
            return self._query_rag(query, route)
        elif source == "search":
            return self._query_search(query, route)
        else:
            return self._query_intrinsic(query, route)
    
    def _query_rag(self, query: str, route: Dict) -> str:
        """Query knowledge base using RAG"""
        # print(f"[RAG] Querying knowledge base...")
        
        if not self.rag_tool:
            self.logger.error("RAG tool unavailable")
            return "I cannot answer this question because the local knowledge base is unavailable."
        
        try:
            # Execute RAG retrieval
            rag_result = self.rag_tool.run({
                "action": "search",
                "query": query,
                "limit": 3
            })
            
            if rag_result.get("status") != "ok" or not rag_result.get("matches"):
                self.logger.warning(f"RAG retrieval did not find enough evidence: {rag_result.get('status')}")
                return "I cannot answer this question based on the available documentation."

            matches = rag_result["matches"]
            documents = sorted({match["filename"] for match in matches})
            route.setdefault("metadata", {})["documents"] = documents
            route["metadata"]["sections"] = [match["heading_path"] for match in matches]

            context_blocks = []
            for match in matches:
                context_blocks.append(
                    f"[Document: {match['filename']}]\n"
                    f"[Section: {match['heading_path']}]\n"
                    f"{match['content']}"
                )
            rag_context = "\n\n---\n\n".join(context_blocks)
            
            # Build enhanced prompt
            prompt = f"""
            You are a company assistant answering from internal documentation.

            You MUST follow these rules strictly:

            1. ONLY use the information provided in the context below
            2. DO NOT add any external knowledge
            3. DO NOT infer or guess missing information
            4. If the answer is not clearly stated in the context, say:
               "I cannot answer this question based on the available documentation."
            5. Keep the answer concise and professional
            6. Cite the relevant document or section in a short phrase when helpful

            Context:
            \"\"\"{rag_context}\"\"\"

            Question:
            \"\"\"{query}\"\"\"

            Answer:
            """
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages)
            
            # print(f"[RAG] Knowledge base query successful")
            return response
            
        except Exception as e:
            self.logger.error(f"RAG query failed: {e}")
            print(f" [RAG] RAG query failed: {e}")
            return f"[RAG] Knowledge base query failed: {str(e)}"
    
    def _query_search(self, query: str, route: Dict) -> str:
        """Query using web search"""
        # print(f"[SEARCH] Web search...")
        
        if not self.search_tool:
            self.logger.warning("Search tool unavailable")
            print(" [SEARCH] Search tool unavailable")
            return "Live web search is unavailable because SEARCH_API_KEY is not configured."
        
        try:
            # Execute search
            search_result = self.search_tool.run({
                "query": query,
                "limit": 3
            })

            if (
                not search_result
                or search_result.startswith("Live web search is unavailable")
                or search_result.startswith("Live web search failed")
                or search_result.startswith("Unsupported search provider")
            ):
                self.logger.warning(f"Search unavailable or failed: {search_result}")
                return search_result
            
            # Build enhanced prompt
            prompt = f"""
            You are a helpful assistant.

            Answer the question based ONLY on the search results below.

            Rules:
            - Do not add external knowledge
            - If the information is insufficient, say so
            - Keep the answer concise

            Search Results:
            \"\"\"{search_result}\"\"\"

            Question:
            \"\"\"{query}\"\"\"

            Answer:
            """
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages)
            
            # print(f"[SEARCH] Search query successful")
            return response
            
        except Exception as e:
            self.logger.error(f"Search query failed: {e}")
            print(f" [SEARCH] Search query failed: {e}")
            return f"Live web search failed: {str(e)}"
    
    def _query_intrinsic(self, query: str, route: Dict, note: str = "") -> str:
        """Query using model's intrinsic knowledge"""
        # print(f"[INTRINSIC] Using intrinsic knowledge...")
        
        prompt = f"""
        Answer the following question clearly and accurately.

        Question:
        \"\"\"{query}\"\"\"

        Answer:
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages)
        
        # print(f"[INTRINSIC] Intrinsic knowledge query complete")
        return response
    
    def _finalize(
        self, 
        start_time: float, 
        query: str, 
        response: str, 
        source: str,
        metadata: Optional[Dict] = None
    ):
        """Finalize query processing, log history and performance"""
        elapsed = time.time() - start_time
        
        # Record performance
        self.monitor.record("response_time", elapsed)
        self.monitor.record(f"source_{source}", 1)
        
        # Save conversation history
        self.history.add(query, response, {
            "source": source,
            "response_time": elapsed,
            "metadata": metadata or {}
        })
        
        # Log
        self.logger.info(f"Query completed, elapsed: {elapsed:.2f}s, source: {source}")
        
        # print(f"[FINALIZE] Elapsed: {elapsed:.2f}s")
    
    # ==================== Helper Methods ====================
    
    def get_stats(self) -> Dict:
        """
        Get agent statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            "name": self.name,
            "conversation_count": len(self.history.history),
            "knowledge_base": self.knowledge_base_path,
            "performance": self.monitor.get_all_stats(),
            "recent_history": self.history.to_dict()
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.history.clear()
        self.logger.info("Conversation history cleared")
        print(f"[FINALIZE] Conversation history cleared")
    
    def export_history(self, filepath: str) -> bool:
        """
        Export conversation history to file
        
        Args:
            filepath: Export file path
        
        Returns:
            Whether successful
        """
        try:
            self.history.save(filepath)
            self.logger.info(f"Conversation history exported: {filepath}")
            print(f"[FINALIZE] Conversation history exported: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export history: {e}")
            print(f" [FINALIZE] Export failed: {e}")
            return False
