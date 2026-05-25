from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from mcq_data import MCQExample, read_jsonl


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "which",
    "who",
    "with",
}


def tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS]


def phrase_windows(tokens: List[str], min_len: int = 3, max_len: int = 8) -> List[str]:
    phrases = []
    max_len = min(max_len, len(tokens))
    for size in range(max_len, min_len - 1, -1):
        for start in range(0, len(tokens) - size + 1):
            phrases.append(" ".join(tokens[start : start + size]))
    return phrases


class TrainingLookup:
    def __init__(self, data_path: str = "data/mcqs_v2_5k_each.jsonl", max_rows: int = 100000) -> None:
        path = Path(data_path)
        if not path.exists():
            path = Path("data/mcqs_extended_10k_each.jsonl")
        if not path.exists():
            path = Path("data/mcqs.jsonl")
        self.examples = [example for example in read_jsonl(path) if example.split == "train"][:max_rows]
        self.phrase_index: Dict[str, List[int]] = {}
        self.example_tokens: List[set[str]] = []
        self.inverted_index: Dict[str, List[int]] = defaultdict(list)
        for index, example in enumerate(self.examples):
            tokens = set(tokenize(example.question))
            self.example_tokens.append(tokens)
            for token in tokens:
                self.inverted_index[token].append(index)
            seen = set()
            for phrase in phrase_windows(tokenize(example.question)):
                if phrase in seen:
                    continue
                seen.add(phrase)
                self.phrase_index.setdefault(phrase, []).append(index)

    def find(self, question: str) -> Optional[dict]:
        query_tokens = tokenize(question)
        if len(query_tokens) < 3:
            return None
        for phrase in phrase_windows(query_tokens):
            candidates = self.phrase_index.get(phrase)
            if not candidates:
                continue
            example = self.examples[candidates[0]]
            return match_payload(example, match_type="phrase", matched_phrase=phrase, score=1.0)

        return self.find_content_match(query_tokens)

    def find_content_match(self, query_tokens: List[str]) -> Optional[dict]:
        query_set = set(query_tokens)
        if not query_set:
            return None
        candidate_counts: Counter[int] = Counter()
        for token in query_set:
            for index in self.inverted_index.get(token, []):
                candidate_counts[index] += 1

        best_index = None
        best_score = 0.0
        best_overlap: List[str] = []
        for index, overlap_count in candidate_counts.most_common(1000):
            example_tokens = self.example_tokens[index]
            if not example_tokens:
                continue
            union = len(query_set | example_tokens)
            score = overlap_count / union
            if score > best_score:
                best_index = index
                best_score = score
                best_overlap = sorted(query_set & example_tokens)

        if best_index is None or best_score < 0.10 or len(best_overlap) < 2:
            return None
        example = self.examples[best_index]
        return match_payload(
            example,
            match_type="content",
            matched_phrase=", ".join(best_overlap[:8]),
            score=best_score,
        )


def match_payload(example: MCQExample, match_type: str, matched_phrase: str, score: float) -> dict:
    return {
        "match_type": match_type,
        "matched_phrase": matched_phrase,
        "score": score,
        "source": example.source,
        "id": example.id,
        "question_excerpt": excerpt(example.question, matched_phrase),
        "answer": example.answer,
        "answer_text": example.choices.get(example.answer, ""),
    }


def excerpt(text: str, phrase: str, radius: int = 90) -> str:
    lowered = text.lower()
    phrase_text = phrase.replace(" ", " ")
    pattern = re.compile(re.escape(phrase_text).replace(r"\ ", r"\s+"), re.IGNORECASE)
    match = pattern.search(lowered)
    if not match:
        tokens = phrase.split()
        match = re.search(re.escape(tokens[0]), lowered, re.IGNORECASE) if tokens else None
    if not match:
        return trim(text, radius * 2)
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix


def trim(text: str, max_len: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3].rstrip() + "..."
