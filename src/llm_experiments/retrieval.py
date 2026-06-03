"""Small BM25 retriever for few-shot prompt examples."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, examples: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.examples = examples
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(example["text"]) for example in examples]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freqs: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freqs.update(set(tokens))
        self.n_docs = len(self.doc_tokens)

    def score(self, query: str) -> list[float]:
        query_terms = tokenize(query)
        scores = [0.0 for _ in self.examples]
        if not query_terms or not self.examples:
            return scores

        for term in query_terms:
            df = self.doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            for index, term_freq in enumerate(self.term_freqs):
                freq = term_freq.get(term, 0)
                if freq == 0:
                    continue
                length = self.doc_lengths[index] or 1
                denom = freq + self.k1 * (1 - self.b + self.b * length / self.avgdl)
                scores[index] += idf * freq * (self.k1 + 1) / denom
        return scores

    def top_k(self, query: str, k: int) -> list[dict[str, Any]]:
        scores = self.score(query)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (item[1], self.examples[item[0]]["example_id"]),
            reverse=True,
        )
        return [self.examples[index] for index, score in ranked[:k] if score > 0]


class FewShotRetriever:
    def __init__(self, examples: list[dict[str, Any]], section_aware: bool = True):
        self.examples = examples
        self.section_aware = section_aware
        self.global_index = BM25Index(examples)
        self.section_indexes: dict[str, BM25Index] = {}
        if section_aware:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for example in examples:
                grouped[example.get("section", "")].append(example)
            self.section_indexes = {
                section: BM25Index(rows) for section, rows in grouped.items()
            }

    def retrieve(self, example: dict[str, Any], k: int) -> list[dict[str, Any]]:
        if k <= 0:
            return []
        query = example["text"]
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()

        if self.section_aware:
            index = self.section_indexes.get(example.get("section", ""))
            if index is not None:
                for candidate in index.top_k(query, k):
                    selected.append(candidate)
                    seen.add(candidate["example_id"])

        if len(selected) < k:
            for candidate in self.global_index.top_k(query, k * 3):
                if candidate["example_id"] in seen:
                    continue
                selected.append(candidate)
                seen.add(candidate["example_id"])
                if len(selected) >= k:
                    break
        return selected[:k]
