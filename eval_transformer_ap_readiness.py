from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score

from eval_ap_readiness import ap_category, ap_examples
from mcq_data import read_jsonl
from transformer_mcq import DEFAULT_TRANSFORMER_MODEL, UnifiedQAMCQPiper


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a seq2seq QA model on AP-style held-out MCQs.")
    parser.add_argument("--data", default="data/mcqs_ap_ready.jsonl")
    parser.add_argument("--model", default=DEFAULT_TRANSFORMER_MODEL)
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--score-mode", choices=("generate", "choice-likelihood"), default="generate")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="models/transformer_ap_readiness_metrics.json")
    args = parser.parse_args()

    rows = ap_examples(example for example in read_jsonl(Path(args.data)) if example.split == args.split)
    if args.limit:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.limit]

    scorer = UnifiedQAMCQPiper(args.model)
    predictions = []
    by_category = defaultdict(lambda: {"true": [], "pred": []})
    seen = 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        if args.score_mode == "choice-likelihood":
            batch_predictions = scorer.score_choices_batch(batch, batch_size=args.batch_size)
        else:
            batch_predictions = scorer.predict_batch(batch, batch_size=args.batch_size)
        predictions.extend(batch_predictions)
        for example, prediction in zip(batch, batch_predictions):
            category = ap_category(example) or "AP-style"
            by_category[category]["true"].append(example.answer)
            by_category[category]["pred"].append(prediction.answer)
        seen += len(batch)
        if seen % 100 == 0 or seen == len(rows):
            print(f"evaluated {seen}/{len(rows)}", flush=True)

    truth = [example.answer for example in rows]
    predicted = [prediction.answer for prediction in predictions]
    metrics = {
        "model": args.model,
        "data": args.data,
        "split": args.split,
        "questions": len(rows),
        "score_mode": args.score_mode,
        "accuracy": accuracy_score(truth, predicted) if rows else 0.0,
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
