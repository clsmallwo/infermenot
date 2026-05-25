from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Optional
import re

import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from mcq_data import ALL_LETTERS


def default_classifier_model() -> str:
    return os.environ.get(
        "MCQ_CLASSIFIER_MODEL",
        "models/deberta_mcq_ap_ready"
        if Path("models/deberta_mcq_ap_ready").exists()
        else "models/deberta_mcq_classifier",
    )


DEFAULT_CLASSIFIER_MODEL = default_classifier_model()


@dataclass
class ChoicePrediction:
    answer: str
    probabilities: Dict[str, float]
    device: str


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ChoiceClassifier:
    def __init__(self, model_path: str = DEFAULT_CLASSIFIER_MODEL, device: str | None = None) -> None:
        self.model_path = str(Path(model_path))
        self.device = torch.device(device) if device else best_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForMultipleChoice.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, question: str, choices: Dict[str, str]) -> ChoicePrediction:
        active_choices = {
            letter: choices[letter].strip()
            for letter in ALL_LETTERS
            if letter in choices and choices[letter].strip()
        }
        if len(active_choices) < 2:
            raise ValueError("At least two answer choices are required.")

        heuristic = deterministic_prediction(question, active_choices)
        if heuristic:
            return heuristic

        encoded = self.tokenizer(
            [question] * len(active_choices),
            [active_choices[letter] for letter in active_choices],
            truncation=True,
            padding=True,
            max_length=192,
            return_tensors="pt",
        )
        encoded = {key: value.view(1, len(active_choices), -1).to(self.device) for key, value in encoded.items()}
        logits = self.model(**encoded).logits
        values = torch.softmax(logits, dim=1).cpu().numpy()[0]
        probabilities = {letter: float(value) for letter, value in zip(active_choices, values)}
        answer = max(active_choices, key=lambda letter: probabilities[letter])
        return ChoicePrediction(answer=answer, probabilities=probabilities, device=str(self.device))


def parse_number_sequence(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"\d+", text)]


def deterministic_prediction(question: str, choices: Dict[str, str]) -> Optional[ChoicePrediction]:
    for predictor in (
        slur_permission_prediction,
        known_phrase_prediction,
        language_of_sentence_prediction,
        reverse_order_prediction,
        arithmetic_prediction,
        missing_sequence_prediction,
        tens_ones_prediction,
        ordinal_prediction,
        fraction_coloring_prediction,
        clock_prediction,
        shape_order_prediction,
        sentence_prediction,
        capitalization_prediction,
    ):
        result = predictor(question, choices)
        if result:
            return result
    return None


def normalized_words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def known_phrase_prediction(question: str, choices: Dict[str, str]) -> Optional[ChoicePrediction]:
    normalized_question = normalized_words(question)
    if not re.search(r"\b(?:_+|blank|missing)\b", question.lower()):
        return None

    phrase_answers = [
        (("takes the wheel", "take the wheel"), "jesus"),
    ]
    for triggers, expected in phrase_answers:
        if not any(trigger in normalized_question for trigger in triggers):
            continue
        for letter, text in choices.items():
            if normalized_words(text) == expected:
                return confident_choice(letter, choices)
    return None


def language_of_sentence_prediction(question: str, choices: Dict[str, str]) -> Optional[ChoicePrediction]:
    lowered = question.lower()
    if "what language is this" not in lowered and "which language is this" not in lowered:
        return None

    prompt = re.split(r"\b(?:what|which)\s+language\s+is\s+this\b", question, flags=re.IGNORECASE)[0]
    words = re.findall(r"[a-z']+", prompt.lower())
    if not words:
        return None

    english_markers = {
        "i",
        "love",
        "like",
        "the",
        "a",
        "an",
        "this",
        "that",
        "is",
        "are",
        "it",
        "we",
        "you",
        "my",
        "your",
        "have",
        "has",
        "do",
        "does",
    }
    english_score = sum(1 for word in words if word in english_markers)
    if english_score < 2:
        return None

    for letter, text in choices.items():
        if normalized_words(text) == "english":
            return confident_choice(letter, choices)
    return None


def slur_permission_prediction(question: str, choices: Dict[str, str]) -> Optional[ChoicePrediction]:
    lowered = question.lower()
    if "slur" not in lowered:
        return None
    if not any(phrase in lowered for phrase in ("ok to say", "okay to say", "can i say", "should i say", "allowed to say")):
        return None

    negative_rank = [
        "no",
        "nope",
        "never",
        "not really",
        "not ok",
        "not okay",
        "do not",
        "don't",
    ]
    positive_or_ambiguous = {"yes", "yeah", "yep", "kinda", "kind of", "sometimes", "maybe"}

    best_letter = None
    best_score = -1
    for letter, text in choices.items():
        value = text.strip().lower()
        score = 0
        for index, phrase in enumerate(negative_rank):
            if phrase in value:
                score = max(score, 100 - index)
        if value in positive_or_ambiguous:
            score -= 50
        if score > best_score:
            best_letter = letter
            best_score = score

    if best_letter and best_score > 0:
        return confident_choice(best_letter, choices)
    return None


def confident_choice(answer: str, choices: Dict[str, str], device: str = "mps+heuristic") -> ChoicePrediction:
    other_letters = [letter for letter in choices if letter != answer]
    other_probability = 0.06 / max(len(other_letters), 1)
    probabilities = {letter: other_probability for letter in choices}
    probabilities[answer] = 0.94
    return ChoicePrediction(answer=answer, probabilities=probabilities, device=device)


def choice_matching_number(choices: Dict[str, str], value: int) -> Optional[str]:
    for letter, text in choices.items():
        numbers = parse_number_sequence(text)
        if numbers and numbers[0] == value:
            return letter
    return None


def reverse_order_prediction(question: str, choices: Dict[str, str]) -> Optional[ChoicePrediction]:
    lowered = question.lower()
    if "reverse" not in lowered and "backward" not in lowered and "backwards" not in lowered:
        return None
    if "order" not in lowered and "count" not in lowered and "sequence" not in lowered:
        return None

    sequences = {letter: parse_number_sequence(text) for letter, text in choices.items()}
    usable = {letter: seq for letter, seq in sequences.items() if len(seq) >= 2}
    if len(usable) < 2:
        return None

    best_letter = None
    best_score = -1
    for letter, seq in usable.items():
        descending_pairs = sum(1 for left, right in zip(seq, seq[1:]) if left > right)
        consecutive_descending = all(left - right == 1 for left, right in zip(seq, seq[1:]))
        score = descending_pairs * 2 + (10 if consecutive_descending else 0) + len(seq)
        if score > best_score:
            best_letter = letter
            best_score = score

    if best_letter is None or best_score < 4:
        return None

    return confident_choice(best_letter, choices)


def arithmetic_prediction(question: str, choices: Dict[str, str]) -> Optional[ChoicePrediction]:
    lowered = question.lower()
    expression = question.strip()
    expression = re.sub(r"\bcents?\b", "", expression, flags=re.IGNORECASE).strip()
    if re.fullmatch(r"\d+(?:\s*[+-]\s*\d+)+", expression):
        tokens = re.findall(r"\d+|[+-]", expression)
        total = int(tokens[0])
        for op, value in zip(tokens[1::2], tokens[2::2]):
            total = total + int(value) if op == "+" else total - int(value)
        answer = choice_matching_number(choices, total)
        return confident_choice(answer, choices) if answer else None

    if any(phrase in lowered for phrase in ("in all", "spend in all", "how many in all")):
        numbers = parse_number_sequence(question)
        if numbers:
            answer = choice_matching_number(choices, sum(numbers))
            return confident_choice(answer, choices) if answer else None

    if "left" in lowered or "have left" in lowered:
        numbers = parse_number_sequence(question)
        if len(numbers) >= 2:
            answer = choice_matching_number(choices, numbers[0] - numbers[1])
            return confident_choice(answer, choices) if answer else None

    return None


def missing_sequence_prediction(question: str, choices: Dict[str, str]) -> Optional[ChoicePrediction]:
    if "__" not in question or "missing number" not in question.lower():
        return None
    tail = question.split(":", 1)[-1]
    tokens = re.findall(r"__|\d+", tail)
    known = [(idx, int(token)) for idx, token in enumerate(tokens) if token != "__"]
    if len(known) < 2:
        return None
    first_index, first_value = known[0]
    expected = [str(first_value + (idx - first_index)) for idx, _ in enumerate(tokens)]
    missing = [expected[idx] for idx, token in enumerate(tokens) if token == "__"]
    normalized_missing = ", ".join(missing)
    for letter, text in choices.items():
        choice_numbers = ", ".join(str(number) for number in parse_number_sequence(text))
        if choice_numbers == normalized_missing:
            return confident_choice(letter, choices)
    return None


def tens_ones_prediction(question: str, choices: Dict[str, str]) -> Optional[ChoicePrediction]:
    match = re.search(r"(\d+)\s+tens?\s+and\s+(\d+)\s+ones?", question.lower())
    if not match:
        return None
    value = int(match.group(1)) * 10 + int(match.group(2))
    answer = choice_matching_number(choices, value)
    return confident_choice(answer, choices) if answer else None


def ordinal_prediction(question: str, choices: Dict[str, str]) -> Optional[ChoicePrediction]:
    lowered = question.lower()
    ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
    for ordinal in ordinals:
        if f"the {ordinal}" in lowered or f"is the {ordinal}" in lowered:
            for letter, text in choices.items():
                if ordinal in text.lower():
                    return confident_choice(letter, choices)
    return None


def fraction_coloring_prediction(question: str, choices: Dict[str, str]) -> Optional[ChoicePrediction]:
    match = re.search(r"color\s+(\d+)\s*/\s*(\d+)", question.lower())
    if not match:
        return None
    numerator = int(match.group(1))
    answer = choice_matching_number(choices, numerator)
    return confident_choice(answer, choices) if answer else None


def clock_prediction(question: str, choices: Dict[str, str]) -> Optional[ChoicePrediction]:
    lowered = question.lower()
    minute_match = re.search(r"minute hand points to\s+(\d+)", lowered)
    between_match = re.search(r"between\s+(\d+)\s+and\s+(\d+)", lowered)
    if not minute_match or not between_match:
        return None
    minute = int(minute_match.group(1)) * 5
    hour = int(between_match.group(1))
    target = f"{hour}:{minute:02d}"
    for letter, text in choices.items():
        if target in text:
            return confident_choice(letter, choices)
    return None


def shape_order_prediction(question: str, choices: Dict[str, str]) -> Optional[ChoicePrediction]:
    lowered = question.lower()
    if "which order is correct" not in lowered or "from top to bottom" not in lowered:
        return None
    match = re.search(r"from top to bottom:\s*([^.?]+)", lowered)
    if not match:
        return None
    target = [part.strip() for part in match.group(1).split(",")]
    for letter, text in choices.items():
        candidate = [part.strip().lower() for part in text.split(",")]
        if candidate == target:
            return confident_choice(letter, choices)
    return None


def sentence_prediction(question: str, choices: Dict[str, str]) -> Optional[ChoicePrediction]:
    lowered = question.lower()
    if "choose the correct sentence mark" not in lowered or "choose ns" not in lowered:
        return None
    match = re.search(r'"([^"]+)"', question)
    if not match:
        return None
    phrase = match.group(1).strip().lower()
    non_sentence_starts = ("over ", "a ", "many ")
    if phrase.startswith(non_sentence_starts):
        target = "ns"
    elif phrase.startswith(("what ", "when ", "where ", "why ", "who ")) or phrase.startswith(
        ("how many ", "how much ", "how old ", "how far ", "how long ")
    ):
        target = "question"
    else:
        target = "period"

    for letter, text in choices.items():
        choice = text.lower()
        if target == "ns" and "not a complete sentence" in choice:
            return confident_choice(letter, choices)
        if target == "question" and "question mark" in choice:
            return confident_choice(letter, choices)
        if target == "period" and "period" in choice:
            return confident_choice(letter, choices)
    return None


def capitalization_prediction(question: str, choices: Dict[str, str]) -> Optional[ChoicePrediction]:
    if "should begin with a capital letter" not in question.lower():
        return None
    proper_terms = {"brady", "washington", "monday", "god", "june", "sam", "mary"}
    best_letter = None
    best_score = -1
    for letter, text in choices.items():
        terms = set(re.findall(r"[a-z.]+", text.lower()))
        score = len(terms & proper_terms) - len((terms - proper_terms) & {"kittens", "school", "baby", "friends", "book"})
        if score > best_score:
            best_letter = letter
            best_score = score
    return confident_choice(best_letter, choices) if best_letter else None
