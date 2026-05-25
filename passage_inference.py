from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from mcq_data import ALL_LETTERS


DEFAULT_PASSAGE_MODEL = os.environ.get(
    "MCQ_PASSAGE_MODEL",
    "potsawee/longformer-large-4096-answering-race",
)


@dataclass
class PassagePrediction:
    answer: str
    probabilities: Dict[str, float]
    scores: Dict[str, float]
    device: str


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_passage_question(text: str) -> tuple[str, str]:
    normalized = text.strip()
    marker = "\nQuestion: "
    if normalized.startswith("Passage: ") and marker in normalized:
        passage, question = normalized.split(marker, 1)
        return passage.removeprefix("Passage: ").strip(), question.strip()
    return "", normalized


class LongformerRaceMCQ:
    def __init__(self, model_name: str = DEFAULT_PASSAGE_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = torch.device(device) if device else best_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMultipleChoice.from_pretrained(
            model_name,
            low_cpu_mem_usage=False,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, question: str, choices: Dict[str, str]) -> PassagePrediction:
        active_choices = {
            letter: choices[letter].strip()
            for letter in ALL_LETTERS
            if letter in choices and choices[letter].strip()
        }
        if len(active_choices) < 2:
            raise ValueError("At least two answer choices are required.")

        passage, prompt_question = split_passage_question(question)
        bos = self.tokenizer.bos_token or self.tokenizer.sep_token or ""
        context_question = f"{passage} {bos} {prompt_question}".strip()
        letters = list(active_choices)
        encoded = self.tokenizer(
            [context_question] * len(letters),
            [active_choices[letter] for letter in letters],
            max_length=4096,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        batch = {key: value.unsqueeze(0).to(self.device) for key, value in encoded.items()}
        logits = self.model(**batch).logits[0]
        probabilities_tensor = torch.softmax(logits, dim=0).cpu()
        scores = {letter: float(score) for letter, score in zip(letters, logits.cpu())}
        probabilities = {
            letter: float(probability)
            for letter, probability in zip(letters, probabilities_tensor)
        }
        answer = max(probabilities, key=probabilities.get)
        return PassagePrediction(
            answer=answer,
            probabilities=probabilities,
            scores=scores,
            device=str(self.device),
        )
