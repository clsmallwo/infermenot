from __future__ import annotations

import re
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from mcq_data import ALL_LETTERS, MCQExample


def default_transformer_model() -> str:
    return os.environ.get(
        "MCQ_TRANSFORMER_MODEL",
        "models/unifiedqa_t5_large_hardened"
        if Path("models/unifiedqa_t5_large_hardened").exists()
        else "models/unifiedqa_t5_large"
        if Path("models/unifiedqa_t5_large").exists()
        else "allenai/unifiedqa-t5-base",
    )


DEFAULT_TRANSFORMER_MODEL = default_transformer_model()


@dataclass
class Prediction:
    answer: str
    generated: str
    scores: Dict[str, float]


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def format_prompt(question: str, choices: Dict[str, str]) -> str:
    letters = [letter for letter in ALL_LETTERS if letter in choices]
    options = " \n ".join(f"({letter}) {choices[letter]}" for letter in letters)
    return f"{question} \n {options}"


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def map_generated_to_letter(generated: str, choices: Dict[str, str]) -> Tuple[str, Dict[str, float]]:
    text = generated.strip().lower()
    letters = [letter for letter in ALL_LETTERS if letter in choices]
    letter_pattern = "".join(letter.lower() for letter in letters)
    letter_match = re.search(rf"\b([{letter_pattern}])\b", text) if letter_pattern else None
    if letter_match:
        letter = letter_match.group(1).upper()
        return letter, {choice: 1.0 if choice == letter else 0.0 for choice in letters}

    scores: Dict[str, float] = {}
    generated_tokens = _tokens(text)
    for letter in letters:
        choice = choices[letter].strip().lower()
        choice_tokens = _tokens(choice)
        if text == choice:
            scores[letter] = 1.0
        elif choice and (text in choice or choice in text):
            scores[letter] = 0.95
        else:
            scores[letter] = len(generated_tokens & choice_tokens) / max(len(choice_tokens), 1)
    best = max(letters, key=lambda letter: scores[letter])
    return best, scores


class UnifiedQAMCQPiper:
    def __init__(self, model_name: str = DEFAULT_TRANSFORMER_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = torch.device(device) if device else best_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, examples: Iterable[MCQExample], batch_size: int = 8) -> List[Prediction]:
        rows = list(examples)
        predictions: List[Prediction] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            prompts = [format_prompt(example.question, example.choices) for example in batch]
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            generated_ids = self.model.generate(**encoded, max_new_tokens=16)
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for example, generated in zip(batch, generated_texts):
                answer, scores = map_generated_to_letter(generated, example.choices)
                predictions.append(Prediction(answer=answer, generated=generated, scores=scores))
        return predictions

    @torch.no_grad()
    def score_choices_batch(self, examples: Iterable[MCQExample], batch_size: int = 8) -> List[Prediction]:
        rows = list(examples)
        predictions: List[Prediction] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            flat_prompts: list[str] = []
            flat_targets: list[str] = []
            flat_letters: list[str] = []
            offsets: list[tuple[int, int]] = []
            for example in batch:
                letters = [letter for letter in ALL_LETTERS if letter in example.choices]
                offsets.append((len(flat_letters), len(letters)))
                prompt = format_prompt(example.question, example.choices)
                for letter in letters:
                    flat_prompts.append(prompt)
                    flat_targets.append(example.choices[letter])
                    flat_letters.append(letter)

            encoded = self.tokenizer(
                flat_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            labels = self.tokenizer(
                text_target=flat_targets,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )["input_ids"].to(self.device)
            labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
            decoder_input_ids = self.model._shift_right(labels)
            outputs = self.model(**encoded, decoder_input_ids=decoder_input_ids)
            token_losses = torch.nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(labels.size())
            lengths = labels.ne(-100).sum(dim=1).clamp_min(1)
            choice_scores = (-(token_losses.sum(dim=1) / lengths)).cpu().tolist()

            for offset, count in offsets:
                letters = flat_letters[offset : offset + count]
                scores = {
                    letter: float(score)
                    for letter, score in zip(letters, choice_scores[offset : offset + count])
                }
                best = max(scores, key=scores.get)
                predictions.append(Prediction(answer=best, generated=flat_targets[offset + letters.index(best)], scores=scores))
        return predictions

    def predict(self, question: str, choices: Dict[str, str]) -> Prediction:
        example = MCQExample(
            id="manual",
            source="manual",
            split="manual",
            question=question,
            choices=choices,
            answer="A",
        )
        return self.score_choices_batch([example], batch_size=1)[0]
