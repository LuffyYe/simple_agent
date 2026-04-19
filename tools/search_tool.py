"""
Web search tool with fail-closed behavior when live search is unavailable.
"""

from typing import Dict, Optional
import os

import requests


class SearchTool:
    """Web search wrapper for SerpAPI."""

    def __init__(
        self,
        name: str = "search",
        description: str = "Web search for live or external information",
        api_provider: str = "serpapi",
        api_key: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.api_provider = api_provider
        self.api_key = api_key or os.getenv("SEARCH_API_KEY")

        if not self.api_key:
            print("[WARNING] Live web search is unavailable because SEARCH_API_KEY is not configured.")

    def run(self, params: Dict) -> str:
        """Execute web search."""
        query = params.get("query", "")
        limit = params.get("limit", 3)

        if not query:
            return "Search query cannot be empty."

        if not self.api_key:
            return "Live web search is unavailable because SEARCH_API_KEY is not configured."

        # print(f"[SEARCH] Searching the web: {query}")

        try:
            if self.api_provider == "serpapi":
                return self._search_serpapi(query, limit)
            return f"Unsupported search provider: {self.api_provider}"
        except Exception as exc:
            print(f"[FAILED] Search failed: {exc}")
            return f"Live web search failed: {exc}"

    def _search_serpapi(self, query: str, limit: int) -> str:
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": limit,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        organic_results = data.get("organic_results", [])[:limit]
        for index, result in enumerate(organic_results, 1):
            results.append(
                f"{index}. {result.get('title', '')}\n"
                f"   {result.get('snippet', '')}\n"
                f"   URL: {result.get('link', '')}"
            )

        return "\n\n".join(results) if results else "No live web results were returned."
