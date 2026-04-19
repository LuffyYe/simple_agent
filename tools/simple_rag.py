"""
Simple hybrid RAG for the local knowledge base.

Uses heading-aware chunking plus BM25 and TF-IDF reranking.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tools.utils import clean_text, normalize_query_text, strip_markdown


class SimpleRAGTool:
    """
    Local knowledge-base retriever with structured results.

    The retriever is optimized for a small Markdown knowledge base:
    - split by headings
    - chunk within sections
    - lexical retrieval with BM25 + TF-IDF reranking
    """

    def __init__(
        self,
        name: str = "rag",
        description: str = "Structured knowledge-base retrieval tool",
        knowledge_base_path: str = "./knowledge_base",
    ):
        self.name = name
        self.description = description
        self.knowledge_base_path = knowledge_base_path
        self.documents: List[Dict] = []
        self.all_chunks: List[Dict] = []
        self.bm25: Optional[BM25Okapi] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        self._load_knowledge_base()

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer with stop word removal."""
        tokens = re.findall(r"\b[a-z0-9']+\b", normalize_query_text(text))
        filtered = [
            token for token in tokens
            if token not in ENGLISH_STOP_WORDS and len(token) > 1
        ]
        return filtered or [token for token in tokens if len(token) > 1]

    def _clean_heading(self, heading: str) -> str:
        """Clean heading text for better matching."""
        cleaned = strip_markdown(heading)
        cleaned = cleaned.replace(":", " ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _clean_content(self, text: str) -> str:
        """Clean content text for better retrieval and display."""
        text = strip_markdown(text)
        text = re.sub(r"^\s*[-*+]\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)
        return clean_text(text)

    def _split_markdown_sections(self, filename: str, text: str) -> List[Dict]:
        """Split a Markdown document into sections based on headings, with fallback to whole text."""
        fallback_title = Path(filename).stem.replace("_", " ")
        heading_stack: List[str] = [fallback_title]
        sections: List[Dict] = []
        buffer: List[str] = []
        in_code_block = False

        def flush_buffer():
            """Flush the current buffer into a section if it has content."""
            content = self._clean_content("\n".join(buffer))
            if not content:
                return
            sections.append(
                {
                    "heading_path": " / ".join(part for part in heading_stack if part),
                    "content": content,
                }
            )

        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                buffer.append(line)
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
            if heading_match:
                flush_buffer()
                buffer = []

                level = len(heading_match.group(1))
                heading = self._clean_heading(heading_match.group(2))
                if not heading:
                    continue

                if level == 1:
                    heading_stack = [heading]
                else:
                    heading_stack = heading_stack[: level - 1]
                    heading_stack.append(heading)
                continue

            if re.match(r"^\s*---+\s*$", line):
                continue

            buffer.append(line)

        flush_buffer()

        if not sections and text.strip():
            sections.append(
                {
                    "heading_path": fallback_title,
                    "content": self._clean_content(text),
                }
            )
        return sections

    def _chunk_section(
        self,
        filename: str,
        filepath: str,
        section: Dict,
        chunk_size: int = 600,
        overlap: int = 60,
    ) -> List[Dict]:
        """Chunk a section into smaller pieces for better retrieval granularity, with some overlap for context preservation."""
        words = section["content"].split()
        if not words:
            return []

        chunks: List[Dict] = []
        step = max(1, chunk_size - overlap)
        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size]
            if not chunk_words:
                continue

            content = " ".join(chunk_words)
            heading_path = section["heading_path"]
            search_text = f"{heading_path} {content}"
            chunks.append(
                {
                    "filename": filename,
                    "filepath": filepath,
                    "heading_path": heading_path,
                    "content": content,
                    "search_text": search_text,
                    "normalized_search_text": normalize_query_text(search_text),
                    "chunk_start": start,
                }
            )

            if start + chunk_size >= len(words):
                break

        return chunks

    def _load_knowledge_base(self):
        """Load and preprocess the knowledge base from the specified directory, then build the search index."""
        # print(f"[LOAD] Loading knowledge base from: {self.knowledge_base_path}")

        if not os.path.exists(self.knowledge_base_path):
            print("[WARNING] Knowledge base directory does not exist.")
            return

        for root, _, files in os.walk(self.knowledge_base_path):
            for file in files:
                if not file.endswith((".md", ".markdown", ".txt")):
                    continue

                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as handle:
                        content = handle.read()

                    sections = self._split_markdown_sections(file, content)
                    chunks: List[Dict] = []
                    for section in sections:
                        chunks.extend(self._chunk_section(file, filepath, section))

                    self.documents.append(
                        {
                            "id": f"{file}_{len(self.documents)}",
                            "filename": file,
                            "filepath": filepath,
                            "content": content,
                            "sections": sections,
                            "chunks": chunks,
                        }
                    )
                    # print(f"[LOAD] Loaded: {file} ({len(chunks)} chunks)")
                except Exception as exc:
                    print(f"[WARNING] Failed to load {file}: {exc}")

        # print(f"[LOAD] Successfully loaded {len(self.documents)} documents.")
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild the BM25 and TF-IDF indices after loading or updating the knowledge base."""
        self.all_chunks = []
        for document in self.documents:
            self.all_chunks.extend(document["chunks"])

        if not self.all_chunks:
            self.bm25 = None
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        tokenized_corpus = [self._tokenize(chunk["normalized_search_text"]) for chunk in self.all_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(
            [chunk["normalized_search_text"] for chunk in self.all_chunks]
        )
        # print(f"[INDEX] Search index built with {len(self.all_chunks)} total chunks.")

    def _normalize_scores(self, scores) -> np.ndarray:
        """Normalize an array of scores to the range [0, 1] by dividing by the maximum score, with clipping to handle edge cases."""
        array = np.asarray(scores, dtype=float)
        if array.size == 0:
            return array

        array = np.clip(array, 0, None)
        max_score = float(array.max())
        if max_score <= 0:
            return np.zeros_like(array)
        return array / max_score

    def _score_chunks(self, normalized_query: str) -> List[Dict]:
        """Score all chunks against the normalized query using a combination of BM25 and TF-IDF cosine similarity, with additional boosts for phrase matches and heading relevance."""
        tokenized_query = self._tokenize(normalized_query)
        if not tokenized_query or not self.bm25 or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        bm25_scores = self._normalize_scores(self.bm25.get_scores(tokenized_query))
        tfidf_query = self.vectorizer.transform([normalized_query])
        tfidf_scores = self._normalize_scores(cosine_similarity(tfidf_query, self.tfidf_matrix)[0])

        scored_results: List[Dict] = []
        for index, chunk in enumerate(self.all_chunks):
            phrase_hits = 0
            for term in set(tokenized_query):
                if len(term) > 3 and term in chunk["normalized_search_text"]:
                    phrase_hits += 1

            phrase_boost = min(0.15, phrase_hits * 0.02)
            heading_overlap = len(set(tokenized_query) & set(self._tokenize(chunk["heading_path"])))
            heading_boost = min(0.12, heading_overlap * 0.03)
            filename_lower = chunk["filename"].lower()
            file_boost = 0.0
            if "policy" in normalized_query and "policies" in filename_lower:
                file_boost += 0.08
            if "guide" in normalized_query and "guide" in filename_lower:
                file_boost += 0.05

            score = float(
                0.65 * bm25_scores[index]
                + 0.35 * tfidf_scores[index]
                + phrase_boost
                + heading_boost
                + file_boost
            )
            scored_results.append(
                {
                    **chunk,
                    "score": round(score, 4),
                }
            )

        scored_results.sort(key=lambda item: item["score"], reverse=True)
        return scored_results

    def _select_matches(
        self,
        scored_results: List[Dict],
        limit: int,
        min_score: float = 0.18,
    ) -> List[Dict]:
        """Select the top matching chunks based on score, with a minimum score threshold and deduplication by section to ensure diverse results."""
        if not scored_results:
            return []

        top_score = scored_results[0]["score"]
        minimum_relative_score = max(min_score, top_score * 0.55)
        selected: List[Dict] = []
        seen_sections = set()

        for candidate in scored_results:
            if candidate["score"] < minimum_relative_score:
                continue

            section_key = (candidate["filename"], candidate["heading_path"])
            if section_key in seen_sections:
                continue

            seen_sections.add(section_key)
            selected.append(
                {
                    "filename": candidate["filename"],
                    "filepath": candidate["filepath"],
                    "heading_path": candidate["heading_path"],
                    "score": candidate["score"],
                    "content": candidate["content"],
                }
            )

            if len(selected) >= limit:
                break
    

        return selected

    def search(self, query: str, limit: int = 3, min_score: float = 0.18) -> Dict:
        """Search the knowledge base for relevant sections based on the query, returning structured results with metadata and content."""
        normalized_query = normalize_query_text(query)
        if not normalized_query:
            return {
                "status": "empty_query",
                "normalized_query": "",
                "matches": [],
            }

        if not self.bm25 or not self.all_chunks:
            return {
                "status": "empty_kb",
                "normalized_query": normalized_query,
                "matches": [],
            }

        scored_results = self._score_chunks(normalized_query)
        matches = self._select_matches(scored_results, limit=limit, min_score=min_score)
        if not matches:
            return {
                "status": "no_match",
                "normalized_query": normalized_query,
                "matches": [],
            }

        return {
            "status": "ok",
            "normalized_query": normalized_query,
            "matches": matches,
        }

    def probe(self, query: str) -> Dict:
        """Cheap routing probe against the KB."""
        return self.search(query, limit=1, min_score=0.3)

    def run(self, params: Dict):
        action = params.get("action", "search")

        if action == "search":
            return self.search(
                query=params.get("query", ""),
                limit=params.get("limit", 3),
                min_score=params.get("min_score", 0.18),
            )
        if action == "add_text":
            return self._add_text(params)
        return {"status": "error", "message": f"Unsupported action: {action}", "matches": []}

    def _add_text(self, params: Dict) -> Dict:
        """Add a new text document to the knowledge base, with automatic section splitting and chunking, then rebuild the search index to include the new content."""
        text = params.get("text", "")
        document_id = params.get("document_id", f"doc_{len(self.documents)}")

        if not text:
            return {"status": "error", "message": "Text cannot be empty."}

        sections = self._split_markdown_sections(document_id, text)
        chunks: List[Dict] = []
        for section in sections:
            chunks.extend(self._chunk_section(document_id, "memory", section))

        self.documents.append(
            {
                "id": document_id,
                "filename": document_id,
                "filepath": "memory",
                "content": text,
                "sections": sections,
                "chunks": chunks,
            }
        )
        self._rebuild_index()
        return {"status": "ok", "message": f"Successfully added document: {document_id}"}
