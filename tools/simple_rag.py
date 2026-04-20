"""
Compact hybrid RAG (BM25 + TF-IDF) for local Markdown knowledge base.
"""

import os, re
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
    def __init__(self, knowledge_base_path: str = "./knowledge_base"):
        self.kb_path = knowledge_base_path
        self.chunks: List[Dict] = []
        self.bm25: Optional[BM25Okapi] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf = None
        self._load()

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer with stop word removal."""
        tokens = re.findall(r"\b[a-z0-9']+\b", normalize_query_text(text))
        tokens = [t for t in tokens if len(t) > 1]
        return [t for t in tokens if t not in ENGLISH_STOP_WORDS] or tokens

    def _clean(self, text: str) -> str:
        text = strip_markdown(text)
        text = re.sub(r"^\s*[-*+]\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)
        return clean_text(text)

    def _split_md(self, filename: str, text: str) -> List[Dict]:
        """Split a Markdown document into sections based on headings, with fallback to whole text."""
        title = Path(filename).stem.replace("_", " ")
        stack, buf, sections = [title], [], []
        in_code = False

        def flush():
            """Flush the current buffer into a section if it has content."""
            content = self._clean("\n".join(buf))
            if content:
                sections.append({
                    "heading": " / ".join(stack),
                    "content": content
                })

        for line in text.splitlines():
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                buf.append(line); continue

            m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
            if m:
                flush(); buf = []
                level = len(m.group(1))
                h = strip_markdown(m.group(2)).strip()
                if not h: continue
                stack = stack[:level-1] + [h]
                continue

            if not re.match(r"^\s*---+\s*$", line):
                buf.append(line)

        flush()
        return sections or [{"heading": title, "content": self._clean(text)}]

    def _chunk(self, file: str, path: str, section: Dict, size: int=600, overlap: int=60) -> List[Dict]:
        """Chunk a section into smaller pieces for better retrieval granularity, with some overlap for context preservation."""
        words = section["content"].split()
        step = max(1, size - overlap)
        out = []

        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i+size])
            if not chunk: continue
            search = f"{section['heading']} {chunk}"
            out.append({
                "filename": file,
                "filepath": path,
                "heading": section["heading"],
                "content": chunk,
                "norm": normalize_query_text(search),
            })
            if i + size >= len(words): break
        return out

    def _load(self):
        """Load and preprocess the knowledge base from the specified directory, then build the search index."""
        if not os.path.exists(self.kb_path):
            return

        for root, _, files in os.walk(self.kb_path):
            for f in files:
                if not f.endswith((".md", ".txt", ".markdown")): continue
                p = os.path.join(root, f)
                try:
                    text = open(p, encoding="utf-8").read()
                    for sec in self._split_md(f, text):
                        self.chunks += self._chunk(f, p, sec)
                except:
                    pass

        self._build_index()

    def _build_index(self):
        """Build the BM25 and TF-IDF indices after loading or updating the knowledge base."""
        if not self.chunks:
            return

        corpus = [c["norm"] for c in self.chunks]
        tokenized = [self._tokenize(x) for x in corpus]

        self.bm25 = BM25Okapi(tokenized)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.tfidf = self.vectorizer.fit_transform(corpus)

    def _norm(self, x: np.ndarray[float]) -> np.ndarray:
        """Normalize an array of scores to the range [0, 1] by dividing by the maximum score, with clipping to handle edge cases."""
        x = np.clip(np.asarray(x, float), 0, None)
        return x / x.max() if x.size and x.max() > 0 else np.zeros_like(x)

    def _score(self, q: str) -> List[Dict]:
        """Score all chunks against the normalized query using a combination of BM25 and TF-IDF cosine similarity, with additional boosts for phrase matches and heading relevance."""
        toks = self._tokenize(q)
        if not toks or not self.bm25:
            return []

        bm = self._norm(self.bm25.get_scores(toks))
        tf = self._norm(cosine_similarity(self.vectorizer.transform([q]), self.tfidf)[0])

        results = []
        for i, c in enumerate(self.chunks):
            hit = sum(1 for t in set(toks) if len(t) > 3 and t in c["norm"])
            phrase = min(0.15, hit * 0.02)

            head_overlap = len(set(toks) & set(self._tokenize(c["heading"])))
            head_boost = min(0.12, head_overlap * 0.03)

            file_boost = 0
            fn = c["filename"].lower()
            if "policy" in q and "policies" in fn: file_boost += 0.08
            if "guide" in q and "guide" in fn: file_boost += 0.05

            score = 0.65 * bm[i] + 0.35 * tf[i] + phrase + head_boost + file_boost

            results.append({**c, "score": round(score, 4)})

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def _select(self, scored: List[Dict], k: int, min_score: float) -> List[Dict]:
        """Select the top matching chunks based on score, with a minimum score threshold and deduplication by section to ensure diverse results."""
        if not scored: return []

        top = scored[0]["score"]
        thresh = max(min_score, top * 0.55)

        out, seen = [], set()
        for c in scored:
            if c["score"] < thresh: continue
            key = (c["filename"], c["heading"])
            if key in seen: continue

            seen.add(key)
            out.append({
                "filename": c["filename"],
                "filepath": c["filepath"],
                "heading_path": c["heading"],
                "score": c["score"],
                "content": c["content"],
            })
            if len(out) >= k: break
        return out

    def search(self, query: str, limit: int=3, min_score: float=0.18) -> Dict:
        """Search the knowledge base for relevant sections based on the query, returning structured results with metadata and content."""
        q = normalize_query_text(query)
        if not q:
            return {"status": "empty_query", "normalized_query": "", "matches": []}
        if not self.bm25:
            return {"status": "empty_kb", "normalized_query": q, "matches": []}

        scored = self._score(q)
        matches = self._select(scored, limit, min_score)

        return {
            "status": "ok" if matches else "no_match",
            "normalized_query": q,
            "matches": matches,
        }

    def probe(self, query: str):
        """Cheap routing probe against the KB."""
        return self.search(query, limit=1, min_score=0.2)

    def run(self, params: Dict):
        return self.search(
            params.get("query", ""),
            params.get("limit", 3),
            params.get("min_score", 0.18),
        )
