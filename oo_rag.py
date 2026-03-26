"""
oo_rag.py — RAG Knowledge Base for Self-Improvement
=====================================================
Local vector store for the sovereign's experiences, errors,
solutions, and web findings. Uses sentence embeddings + cosine
similarity for retrieval. JSON-backed, no external dependencies
beyond numpy.

Knowledge categories:
  error       — Failures + root cause + fix
  solution    — Successful approaches
  fact        — Web-fetched information
  code        — Working code snippets
  observation — Self-reflection insights

Usage:
  rag = KnowledgeBase()
  rag.store("Fixed NaN by reducing RLF depth", "solution", {"depth": 1})
  results = rag.retrieve("NaN in hidden state", top_k=3)
"""
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class KnowledgeEntry:
    """A single knowledge entry in the RAG store."""

    entry_id: str
    text: str
    category: str    # error, solution, fact, code, observation
    metadata: dict = field(default_factory=dict)
    embedding: list = field(default_factory=list)
    created_at: float = 0.0
    access_count: int = 0
    relevance_score: float = 1.0   # decays over time


class SimpleEmbedder:
    """Lightweight text embedder using character n-gram hashing.

    No external model needed — uses deterministic hash-based
    embeddings. Not state-of-the-art but sufficient for a
    self-contained system with no API dependencies.
    """

    def __init__(self, dim: int = 256, ngram_range: tuple = (2, 4)) -> None:
        """Initialize embedder.

        Args:
            dim: embedding dimension
            ngram_range: (min, max) character n-gram sizes
        """
        self.dim = dim
        self.ngram_range = ngram_range

    def embed(self, text: str) -> list[float]:
        """Embed text as a dense vector via character n-gram hashing.

        Args:
            text: input text

        Returns:
            list of floats (unit-normalized)
        """
        vec = [0.0] * self.dim
        text_lower = text.lower().strip()

        if not text_lower:
            return vec

        # Generate character n-grams
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(text_lower) - n + 1):
                gram = text_lower[i:i + n]
                # Hash to dimension index
                h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                idx = h % self.dim
                # Sign from second hash
                sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
                vec[idx] += sign

        # Add word-level features
        words = text_lower.split()
        for word in words:
            h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 2.0  # Stronger signal for whole words

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec

    def similarity(self, a: list[float], b: list[float]) -> float:
        """Cosine similarity between two embeddings.

        Args:
            a: first embedding
            b: second embedding

        Returns:
            similarity score in [-1, 1]
        """
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        return dot  # Already L2-normalized


class KnowledgeBase:
    """JSON-backed vector knowledge base for sovereign self-improvement."""

    CATEGORIES = {"error", "solution", "fact", "code", "observation"}

    def __init__(
        self,
        db_path: str = "sovereign_knowledge.json",
        embed_dim: int = 256,
    ) -> None:
        """Initialize or load knowledge base.

        Args:
            db_path: path to JSON database file
            embed_dim: embedding dimension
        """
        self.db_path = db_path
        self.embedder = SimpleEmbedder(dim=embed_dim)
        self.entries: list[KnowledgeEntry] = []
        self._load()

    def _load(self) -> None:
        """Load knowledge base from disk."""
        if os.path.exists(self.db_path):
            with open(self.db_path) as f:
                data = json.load(f)
            for item in data.get("entries", []):
                self.entries.append(KnowledgeEntry(**item))

    def _save(self) -> None:
        """Persist knowledge base to disk."""
        data = {
            "version": 1,
            "count": len(self.entries),
            "updated_at": time.time(),
            "entries": [asdict(e) for e in self.entries],
        }
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2)

    def store(
        self,
        text: str,
        category: str = "observation",
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a new knowledge entry.

        Args:
            text: knowledge text
            category: one of CATEGORIES
            metadata: optional structured data

        Returns:
            entry_id
        """
        if category not in self.CATEGORIES:
            category = "observation"

        # Dedup: don't store if very similar entry exists
        embedding = self.embedder.embed(text)
        for existing in self.entries:
            if existing.embedding:
                sim = self.embedder.similarity(embedding, existing.embedding)
                if sim > 0.95:
                    # Update existing instead of duplicating
                    existing.access_count += 1
                    existing.metadata.update(metadata or {})
                    self._save()
                    return existing.entry_id

        entry_id = hashlib.sha256(
            f"{text}{time.time()}".encode()
        ).hexdigest()[:16]

        entry = KnowledgeEntry(
            entry_id=entry_id,
            text=text,
            category=category,
            metadata=metadata or {},
            embedding=embedding,
            created_at=time.time(),
        )
        self.entries.append(entry)
        self._save()
        return entry_id

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        min_similarity: float = 0.1,
    ) -> list[dict]:
        """Retrieve relevant knowledge entries.

        Args:
            query: search query
            top_k: number of results
            category: filter by category
            min_similarity: minimum similarity threshold

        Returns:
            list of dicts with 'entry_id', 'text', 'category',
            'similarity', 'metadata'
        """
        if not self.entries:
            return []

        query_emb = self.embedder.embed(query)

        scored = []
        for entry in self.entries:
            if category and entry.category != category:
                continue
            if not entry.embedding:
                continue

            sim = self.embedder.similarity(query_emb, entry.embedding)

            # Recency boost: entries from last hour get +0.1
            age_hours = (time.time() - entry.created_at) / 3600
            recency_boost = 0.1 if age_hours < 1 else (0.05 if age_hours < 24 else 0)

            # Frequency boost
            freq_boost = min(0.05, entry.access_count * 0.01)

            final_score = sim + recency_boost + freq_boost

            if sim >= min_similarity:
                scored.append((final_score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, entry in scored[:top_k]:
            entry.access_count += 1
            results.append({
                "entry_id": entry.entry_id,
                "text": entry.text,
                "category": entry.category,
                "similarity": score,
                "metadata": entry.metadata,
            })

        if results:
            self._save()

        return results

    def store_error(self, error: str, root_cause: str, fix: str) -> str:
        """Store an error with root cause and fix.

        Args:
            error: error description
            root_cause: what caused the error
            fix: how it was fixed

        Returns:
            entry_id
        """
        text = f"ERROR: {error}\nCAUSE: {root_cause}\nFIX: {fix}"
        return self.store(text, "error", {
            "error": error, "root_cause": root_cause, "fix": fix
        })

    def store_solution(self, problem: str, solution: str, result: str = "") -> str:
        """Store a successful solution.

        Args:
            problem: what was the problem
            solution: how it was solved
            result: outcome

        Returns:
            entry_id
        """
        text = f"PROBLEM: {problem}\nSOLUTION: {solution}\nRESULT: {result}"
        return self.store(text, "solution", {
            "problem": problem, "solution": solution, "result": result
        })

    def store_code(self, description: str, code: str, lang: str = "python") -> str:
        """Store working code.

        Args:
            description: what the code does
            code: source code
            lang: programming language

        Returns:
            entry_id
        """
        text = f"CODE ({lang}): {description}\n```{lang}\n{code}\n```"
        return self.store(text, "code", {
            "description": description, "code": code, "lang": lang
        })

    def store_fact(self, topic: str, content: str, source: str = "") -> str:
        """Store a fact from web research.

        Args:
            topic: topic/title
            content: fact content
            source: source URL

        Returns:
            entry_id
        """
        text = f"FACT: {topic}\n{content}"
        return self.store(text, "fact", {
            "topic": topic, "source": source
        })

    def stats(self) -> dict:
        """Return knowledge base statistics.

        Returns:
            dict with counts by category and total
        """
        by_cat = {}
        for e in self.entries:
            by_cat[e.category] = by_cat.get(e.category, 0) + 1
        return {
            "total": len(self.entries),
            "by_category": by_cat,
            "db_size_kb": os.path.getsize(self.db_path) / 1024 if os.path.exists(self.db_path) else 0,
        }

    def clear(self) -> None:
        """Clear all entries."""
        self.entries = []
        self._save()


def run_test() -> None:
    """Self-test: exercise store, retrieve, similarity."""
    print("\n  oo_rag self-test")
    print("  " + "=" * 40)

    test_db = "/tmp/sovereign_kb_test.json"
    if os.path.exists(test_db):
        os.remove(test_db)

    kb = KnowledgeBase(db_path=test_db)

    # Test 1: Store entries
    id1 = kb.store_error(
        "NaN in hidden state during loop 5",
        "RLF depth too high for input length",
        "Reduced RLF depth to 4, NaN disappeared"
    )
    assert id1
    print(f"  ✓ Store error: {id1}")

    id2 = kb.store_solution(
        "Model outputs 'sembly' for all queries",
        "LoRA weights not loading correctly, need strict=False",
        "Accuracy restored to 100%"
    )
    print(f"  ✓ Store solution: {id2}")

    id3 = kb.store_code(
        "Fibonacci generator",
        "def fib(n): a,b=0,1\n for _ in range(n): a,b=b,a+b\n return a",
        "python"
    )
    print(f"  ✓ Store code: {id3}")

    id4 = kb.store_fact(
        "Mamba2 SSM architecture",
        "Mamba2 uses SSD (Structured State Space Duality) with O(N) inference",
        "https://arxiv.org/abs/2405.21060"
    )
    print(f"  ✓ Store fact: {id4}")

    kb.store("Gate μ=1.003 is optimal for lifeline stability", "observation")
    print("  ✓ Store observation")

    # Test 2: Retrieve
    results = kb.retrieve("NaN error hidden state")
    assert len(results) > 0
    assert results[0]["category"] == "error"
    print(f"  ✓ Retrieve 'NaN': top={results[0]['similarity']:.3f} "
          f"cat={results[0]['category']}")

    results = kb.retrieve("model output wrong answer")
    assert len(results) > 0
    print(f"  ✓ Retrieve 'wrong answer': top={results[0]['similarity']:.3f} "
          f"cat={results[0]['category']}")

    # Test 3: Category filter
    results = kb.retrieve("architecture", category="fact")
    assert all(r["category"] == "fact" for r in results)
    print(f"  ✓ Category filter: {len(results)} facts")

    # Test 4: Deduplication
    old_count = len(kb.entries)
    kb.store_error(
        "NaN in hidden state during loop 5",
        "RLF depth too high for input length",
        "Reduced RLF depth to 4, NaN disappeared"
    )
    assert len(kb.entries) == old_count, "Duplicate should not be added"
    print("  ✓ Deduplication: duplicate skipped")

    # Test 5: Stats
    stats = kb.stats()
    assert stats["total"] == 5
    print(f"  ✓ Stats: {stats['total']} entries, {stats['db_size_kb']:.1f}KB")

    # Test 6: Persistence
    kb2 = KnowledgeBase(db_path=test_db)
    assert len(kb2.entries) == 5
    print(f"  ✓ Persistence: reloaded {len(kb2.entries)} entries")

    # Cleanup
    os.remove(test_db)
    print("  ✓ Cleanup: OK")

    print(f"\n  All tests passed.\n")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        run_test()
    else:
        print("Usage: python oo_rag.py --test")
