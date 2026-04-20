"""
Query Routing Tool - Decide which knowledge source to use
"""

from typing import Callable, Dict, List, Optional
import re

from tools.utils import normalize_query_text
# from utils import normalize_query_text

class RouterTool:
    """
    Deterministic query router with lightweight retrieval probing.

    Routing order:
    1. Explicit live-web intent -> search
    2. Explicit internal/company intent -> rag
    3. Retrieval probe -> rag when KB evidence is strong
    4. Weak general knowledge patterns -> intrinsic
    """

    def __init__(
        self,
        name: str = "router",
        description: str = "Route query to the best knowledge source",
        llm=None,
        company_keywords: Optional[List[str]] = None,
        retrieval_probe: Optional[Callable[[str], Dict]] = None,
    ):
        self.name = name
        self.description = description
        self.llm = llm
        self.retrieval_probe = retrieval_probe

        self.company_keywords = company_keywords or [
            "zuru melon",
            "company",
            "internal",
            "policy",
            "policies",
            "procedure",
            "procedures",
            "guideline",
            "guidelines",
            "mission statement",
            "benefits",
            "employee",
            "employees",
            "contractor",
            "leave request",
            "paid leave",
            "vacation",
            "sick leave",
            "doctor's note",
            "remote work",
            "coding style guide",
            "naming conventions",
            "confidentiality",
            "non disclosure agreement",
            "client data",
            "security incident",
            "ethics committee",
            "ethics review board",
            "ai ethics",
            "hr portal",
            "data protection officer",
        ]
        self.internal_aliases = {
            "how many paid vacation days": "paid leave vacation days",
            "leave approval": "leave request approval",
            "vacation approval": "leave request approval",
            "doctor note": "doctor's note",
            "remote": "remote work",
            "work remotely": "remote work",
            "style guide": "coding style guide",
        }
        self.live_web_patterns = [
            r"\btoday\b",
            r"\btomorrow\b",
            r"\blatest\b",
            r"\bcurrent\b",
            r"\bmost recent\b",
            r"\breal[- ]?time\b",
            r"\bheadline[s]?\b",
            r"\bnews\b",
            r"\bweather\b",
            r"\bforecast\b",
            r"\bclosing price\b",
            r"\bstock\b",
            r"\bexchange rate\b",
        ]
        self.external_patterns = [
            r"\bofficial website\b",
            r"\bwikipedia\b",
            r"\breport\b",
        ]
        self.general_patterns = [
            r"\bcapital of\b",
            r"\bwhich ocean\b",
            r"\bexplain\b",
            r"\bexample\b",
            r"\bsyntax\b",
            r"\balgorithm\b",
            r"\bmath\b",
            r"\bscience\b",
            r"\bhistory\b",
            r"\bwho is\b",
            r"\bwhat is\b",
            r"\bhow to\b",
        ]

    def set_retrieval_probe(self, retrieval_probe: Callable[[str], Dict]):
        """Attach a retrieval probe after the RAG tool is initialized."""
        self.retrieval_probe = retrieval_probe

    def run(self, params: Dict) -> Dict:
        """
        Execute routing decision.

        Returns a routing payload:
            {
                "source": "rag" | "search" | "intrinsic",
                "reason": str,
                "metadata": dict
            }
        """
        query = params.get("query", "")
        if not query:
            return {
                "source": "intrinsic",
                "reason": "Empty query, defaulting to intrinsic knowledge",
                "metadata": {"normalized_query": ""},
            }

        normalized_query = normalize_query_text(query)
        live_web_result = self._route_live_web(normalized_query)
        if live_web_result:
            return live_web_result

        internal_result = self._route_internal(query, normalized_query)
        if internal_result:
            return internal_result
        
        intrinsic_result = self._route_intrinsic(normalized_query)
        if intrinsic_result:
            return intrinsic_result

        probe_result = self._probe_knowledge_base(query)
        if probe_result:
            return {
                "source": "rag",
                "reason": "Knowledge-base probe found strong supporting evidence",
                "metadata": {
                    "normalized_query": normalized_query,
                    "intent_tags": ["retrieval_probe"],
                    "retrieval_probe": probe_result["metadata"],
                },
            }


        return {
            "source": "intrinsic",
            "reason": "No strong internal or live-web signals detected",
            "metadata": {
                "normalized_query": normalized_query,
                "intent_tags": ["fallback_intrinsic"],
            },
        }

    def _route_live_web(self, normalized_query: str) -> Optional[Dict]:
        matches = [
            pattern
            for pattern in self.live_web_patterns + self.external_patterns
            if re.search(pattern, normalized_query)
        ]
        if not matches:
            return None

        return {
            "source": "search",
            "reason": "Detected explicit live-web or time-sensitive intent",
            "metadata": {
                "normalized_query": normalized_query,
                "intent_tags": ["live_web"],
                "matched_patterns": matches,
            },
        }

    def _route_internal(self, original_query: str, normalized_query: str) -> Optional[Dict]:
        enriched_query = normalized_query
        for source, target in self.internal_aliases.items():
            if source in enriched_query:
                enriched_query = f"{enriched_query} {target}"

        matched_keywords = [
            keyword for keyword in self.company_keywords if keyword in enriched_query
        ]

        if len(matched_keywords) >= 2:
            return {
                "source": "rag",
                "reason": "Detected multiple internal-document signals",
                "metadata": {
                    "normalized_query": normalized_query,
                    "intent_tags": ["internal_docs"],
                    "matched_keywords": matched_keywords,
                },
            }

        explicit_company_signals = [
            "zuru melon",
            "company",
            "internal",
            "policy",
            "coding style guide",
        ]
        if any(signal in enriched_query for signal in explicit_company_signals):
            return {
                "source": "rag",
                "reason": "Detected explicit company documentation intent",
                "metadata": {
                    "normalized_query": normalized_query,
                    "intent_tags": ["internal_docs"],
                    "matched_keywords": matched_keywords,
                },
            }

        if "python" in normalized_query and any(
            keyword in normalized_query
            for keyword in ["style guide", "coding style", "naming", "variable", "class"]
        ):
            return {
                "source": "rag",
                "reason": "Python question references the internal coding style guide",
                "metadata": {
                    "normalized_query": normalized_query,
                    "intent_tags": ["internal_docs", "coding_style"],
                },
            }

        return None

    def _probe_knowledge_base(self, query: str) -> Optional[Dict]:
        if not self.retrieval_probe:
            return None

        try:
            probe = self.retrieval_probe(query)
        except Exception:
            return None

        if not probe or probe.get("status") != "ok":
            return None

        matches = probe.get("matches", [])
        if not matches:
            return None

        top_match = matches[0]
        top_score = float(top_match.get("score", 0.0))
        if top_score < 0.2:
            return None

        return {
            "metadata": {
                "filename": top_match.get("filename"),
                "heading_path": top_match.get("heading_path"),
                "score": top_score,
            },
        }

    def _route_intrinsic(self, normalized_query: str) -> Optional[Dict]:
        matched_patterns = [
            pattern for pattern in self.general_patterns if re.search(pattern, normalized_query)
        ]
        if not matched_patterns:
            return None

        return {
            "source": "intrinsic",
            "reason": "Detected a general-knowledge style query without internal or live-web signals",
            "metadata": {
                "normalized_query": normalized_query,
                "intent_tags": ["general_knowledge"],
                "matched_patterns": matched_patterns,
            },
        }

