from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import torch
from sklearn.metrics import accuracy_score

from choice_inference import ChoiceClassifier, DEFAULT_CLASSIFIER_MODEL
from mcq_data import MCQExample, read_jsonl
from train_choice_classifier import ChoiceDataset, choice_letters, make_loader


AP_MMLU_TAGS = {
    "astronomy",
    "college biology",
    "college chemistry",
    "college mathematics",
    "college physics",
    "conceptual physics",
    "econometrics",
    "formal logic",
    "high school biology",
    "high school chemistry",
    "high school computer science",
    "high school european history",
    "high school geography",
    "high school government and politics",
    "high school macroeconomics",
    "high school mathematics",
    "high school microeconomics",
    "high school physics",
    "high school psychology",
    "high school statistics",
    "high school us history",
    "high school world history",
}


def bracket_tag(question: str) -> Optional[str]:
    match = re.match(r"\[([^\]]+)\]", question)
    return match.group(1) if match else None


def ap_category(example: MCQExample) -> Optional[str]:
    tag = bracket_tag(example.question)
    if example.source == "cais/mmlu/all" and tag in AP_MMLU_TAGS:
        return f"MMLU / {tag}"
    if example.source == "notefill/ck12-tqa-instruction":
        return "CK-12 science"
    if example.source == "allenai/ai2_arc/ARC-Challenge":
        return "ARC Challenge science"
    if example.source == "allenai/qasc":
        return "QASC science"
    if example.source == "allenai/sciq":
        return "SciQ science"
    if example.source == "derek-thomas/ScienceQA":
        if tag and any(grade in tag for grade in ("grade6", "grade7", "grade8", "grade9", "grade10", "grade11", "grade12")):
            return "ScienceQA upper-grade"
    return None


def ap_examples(examples: Iterable[MCQExample]) -> list[MCQExample]:
    return [example for example in examples if ap_category(example)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AP-style held-out MCQs by source and subject.")
    parser.add_argument("--data", default="data/mcqs_ap_ready.jsonl")
    parser.add_argument("--model", default=DEFAULT_CLASSIFIER_MODEL)
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="models/ap_readiness_metrics.json")
    args = parser.parse_args()

    rows = ap_examples(example for example in read_jsonl(Path(args.data)) if example.split == args.split)
    if args.limit:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.limit]

    classifier = ChoiceClassifier(args.model)
    dataset = ChoiceDataset(rows, classifier.tokenizer, args.max_length)
    loader = make_loader(dataset, args.batch_size, shuffle=False, seed=args.seed, num_workers=0)
    truth: list[str] = []
    predictions: list[str] = []
    by_category = defaultdict(lambda: {"true": [], "pred": []})

    classifier.model.eval()
    seen = 0
    with torch.no_grad():
        for batch in loader:
            batch_examples = [rows[index] for index in batch.pop("example_indices")]
            batch = {key: value.to(classifier.device) for key, value in batch.items()}
            logits = classifier.model(**batch).logits
            predicted_indices = logits.argmax(dim=1).cpu().tolist()
            for example, predicted_index in zip(batch_examples, predicted_indices):
                letters = choice_letters(example)
                predicted = letters[predicted_index]
                truth.append(example.answer)
                predictions.append(predicted)
                category = ap_category(example) or "AP-style"
                by_category[category]["true"].append(example.answer)
                by_category[category]["pred"].append(predicted)
                seen += 1
            if seen == len(batch_examples) or seen % 100 == 0 or seen == len(rows):
                print(f"evaluated {seen}/{len(rows)}", flush=True)

    metrics = {
        "model": args.model,
        "data": args.data,
        "split": args.split,
        "questions": len(rows),
        "accuracy": accuracy_score(truth, predictions) if rows else 0.0,
        "category_accuracy": {
            category: {
                "questions": len(values["true"]),
                "accuracy": accuracy_score(values["true"], values["pred"]),
            }
            for category, values in sorted(by_category.items())
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
