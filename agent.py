"""
Company Assistant Agent - Built on hello-agents framework

Capabilities:
1. Intelligent routing: automatically select best knowledge source (RAG / search / intrinsic)
2. Safety filtering: block harmful or policy-violating queries
3. Clarification: proactively ask for clarification on ambiguous queries
4. Conversation memory: maintain context and provide statistics
"""


import os, re, time
from typing import Optional, Dict

from hello_agents import HelloAgentsLLM, Config
from tools.router_tool import RouterTool
from tools.safety_tool import SafetyTool
from tools.search_tool import SearchTool
from tools.simple_rag import SimpleRAGTool

from tools.utils import (
    Logger, ConversationHistory, PerformanceMonitor,
    format_response, extract_json_from_text,
    list_markdown_files, ensure_directory, get_timestamp
)


class CompanyAssistantAgent:
    def __init__(
        self,
        name="CompanyAssistant",
        llm: Optional[HelloAgentsLLM] = None,
        config: Optional[Config] = None,
        knowledge_base_path="./knowledge_base",
        enable_logging=True,
        log_dir="./logs"
    ):
        self.name, self.llm, self.config = name, llm, config
        self.kb_path = ensure_directory(knowledge_base_path)

        # tools
        self.safety = SafetyTool()
        self.router = RouterTool(llm=llm)
        self.search = SearchTool(
            api_provider=os.getenv("SEARCH_PROVIDER", "serpapi"),
            api_key=os.getenv("SEARCH_API_KEY", "")
        )
        self.rag = None

        # utils
        log_file = os.path.join(log_dir, f"agent_{get_timestamp().replace(':','-')}.log")
        self.logger = Logger(name=name, log_file=log_file) if enable_logging else Logger(name=name)
        self.history = ConversationHistory(max_turns=10)
        self.monitor = PerformanceMonitor()

        self._init_rag()

    def _init_rag(self) -> None:
        try:
            if not list_markdown_files(self.kb_path):
                print(f"[RAG] Empty KB: {self.kb_path}")
            self.rag = SimpleRAGTool(self.kb_path)
            self.router.set_retrieval_probe(self.rag.probe)
        except Exception as e:
            print(f"[RAG] init failed: {e}")
            self.rag = None

    def run(self, query: str) -> str:
        """Process a user query through safety, routing, execution, and formatting."""

        start_time = time.time()
        self.logger.info(f"Received query: {query}")

        try:
            # ---------- Step 1: Safety Check ----------
            safety_result = self._check_safety(query)
            if safety_result.get("blocked"):
                message = safety_result.get("message", "Query rejected")
                return self._finalize_response(start_time, query, message, "blocked", safety_result)

            # ---------- Step 2: Clarification Check ----------
            clarification_message = self._check_clarification(query)
            if clarification_message:
                return self._finalize_response(start_time, query, clarification_message, "clarification")

            # ---------- Step 3: Routing ----------
            context = self.history.get_context(3)
            route_result = self.router.run({
                "query": query,
                "context": context
            })

            source = route_result.get("source", "intrinsic")
            reason = route_result.get("reason", "")
            self.logger.info(f"Routing decision: {source} | Reason: {reason}")

            # ---------- Step 4: Execute ----------
            if source == "rag":
                response = self._rag(query, route_result)
            elif source == "search":
                response = self._search(query)
            else:
                response = self._intrinsic(query)

            # ---------- Step 5: Format ----------
            metadata = {
                "reason": reason,
                **route_result.get("metadata", {})
            }
            formatted_response = format_response(response, source, metadata)

            # ---------- Step 6: Finalize ----------
            return self._finalize_response(
                start_time,
                query,
                formatted_response,
                source,
                route_result
            )

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return f"[ERROR] Error processing query: {e}"
        
    def _check_safety(self, query: str) -> Dict:
        """Check whether the query violates safety rules."""

        safety_result = self.safety.run({"text": query})
        if not safety_result.get("safe", True):
            return {
                "blocked": True,
                "message": safety_result.get("message", "Query rejected"),
                "category": safety_result.get("category")
            }

        if os.getenv("ENABLE_LLM_SAFETY_CHECK", "").lower() == "true":
            try:
                prompt = f"""
                Return JSON in the format:
                {{"safe": true/false, "reason": "", "category": ""}}

                Query:
                \"\"\"{query}\"\"\"
                """

                llm_response = self.llm.invoke([
                    {"role": "user", "content": prompt}
                ])

                parsed_result = extract_json_from_text(llm_response)

                if parsed_result and not parsed_result.get("safe", True):
                    return {
                        "blocked": True,
                        "message": f"[SAFETY] {parsed_result.get('reason', 'Policy violation')}",
                        "category": parsed_result.get("category")
                    }

            except Exception as e:
                self.logger.warning(f"LLM safety check failed: {e}")

        return {"blocked": False}

    def _check_clarification(self, query: str) -> Optional[str]:
        """Determine whether the query is ambiguous and needs clarification."""

        normalized_query = query.strip().lower()
        words = normalized_query.split()

        if not normalized_query or len(words) <= 2:
            return "Your query is too brief. Please provide more details."

        if re.match(r"^(how|what|why|when|where|who)\b$", normalized_query):
            return "Your query is too vague. Please provide more details."

        has_pronoun = re.search(r"\b(it|this|that|they)\b", normalized_query)
        if has_pronoun and len(words) <= 6:
            return "Your query contains a pronoun without clear reference."

        if len(normalized_query) < 15:
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

                llm_response = self.llm.invoke([{"role": "user", "content": prompt}]).strip()

                if llm_response.startswith("CLARIFY:"):
                    clarification = llm_response.replace("CLARIFY:", "").strip()
                    return f"Your query is not clear enough. {clarification}"

            except Exception as e:
                self.logger.warning(f"LLM clarification check failed: {e}")

        return None

    def _rag(self, query: str, route: Dict) -> str:
        """Answer a query using the local knowledge base (RAG)."""

        if self.rag is None:
            return "I cannot answer this question because the local knowledge base is unavailable."

        rag_result = self.rag.run({
            "query": query,
            "limit": 3
        })

        if rag_result.get("status") != "ok":
            return "I cannot answer this question based on the available documentation."

        matches = rag_result.get("matches", [])

        metadata = route.setdefault("metadata", {})
        metadata["documents"] = list({match["filename"] for match in matches})
        metadata["sections"] = [match["heading_path"] for match in matches]

        context_parts = []
        for match in matches:
            block = (
                f"[Document: {match['filename']}]\n"
                f"[Section: {match['heading_path']}]\n"
                f"{match['content']}"
            )
            context_parts.append(block)

        context_text = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a company assistant answering from internal documentation.

        Rules:
        - Only use the provided context
        - Do not add external knowledge
        - If the answer is not clearly stated, say you cannot answer

        Context:
        \"\"\"{context_text}\"\"\"

        Question:
        \"\"\"{query}\"\"\"

        Answer:"""

        response = self.llm.invoke([{"role": "user", "content": prompt}])

        return response

    def _search(self, query: str) -> str:
        """Answer a query using live web search results."""

        if self.search is None:
            return "Live web search is unavailable because SEARCH_API_KEY is not configured."

        search_result = self.search.run({
            "query": query,
            "limit": 3
        })

        if not search_result or search_result.startswith("Live web search"):
            return search_result

        prompt = f"""
        You are a helpful assistant.

        Answer ONLY using the search results below.

        Search Results:
        \"\"\"{search_result}\"\"\"

        Question:
        \"\"\"{query}\"\"\"

        Answer:
        """

        response = self.llm.invoke([{"role": "user", "content": prompt}])

        return response

    def _intrinsic(self, query: str) -> str:
        """Answer a query using the model's internal knowledge."""

        prompt = f"""
        Answer the following question clearly and concisely.

        Question:
        \"\"\"{query}\"\"\"

        Answer:
        """

        response = self.llm.invoke([{"role": "user", "content": prompt}])

        return response

    def _finalize_response(self, t0: float, query: str, response: str, src: str, meta: Optional[Dict]=None) -> str:
        """Finalize query processing, log history and performance"""
        dt = time.time() - t0
        self.monitor.record("response_time", dt)
        self.monitor.record(f"source_{src}", 1)
        self.history.add(query, response, {"source": src, "response_time": dt, "metadata": meta or {}})
        self.logger.info(f"Done {src} {dt:.2f}s")
        return response

    def get_stats(self) -> Dict:
        """Get agent statistics and returns statistics dictionary"""
        return {
            "name": self.name,
            "conversation_count": len(self.history.history),
            "knowledge_base": self.kb_path,
            "performance": self.monitor.get_all_stats(),
            "recent_history": self.history.to_dict()
        }

    def clear_history(self) -> None:
        """Clear conversation history"""
        self.history.clear()
        self.logger.info("History cleared")

    def export_history(self, path) -> None:
        """Export conversation history to file"""
        try:
            self.history.save(path)
            return True
        except:
            return False
