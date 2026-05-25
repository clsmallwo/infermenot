from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score

from mcq_data import read_jsonl
from transformer_mcq import DEFAULT_TRANSFORMER_MODEL, UnifiedQAMCQPiper


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a UnifiedQA transformer MCQ scorer.")
    parser.add_argument("--data", default="data/mcqs.jsonl")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--model", default=DEFAULT_TRANSFORMER_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for quick checks.")
    parser.add_argument("--out", default="models/transformer_metrics.json")
    args = parser.parse_args()

    examples = [example for example in read_jsonl(Path(args.data)) if example.split == args.split]
    if args.limit:
        examples = examples[: args.limit]

    scorer = UnifiedQAMCQPiper(args.model)
    predictions = scorer.predict_batch(examples, batch_size=args.batch_size)
    y_true = [example.answer for example in examples]
    y_pred = [prediction.answer for prediction in predictions]

    by_source = defaultdict(lambda: {"true": [], "pred": []})
    for example, prediction in zip(examples, predictions):
        by_source[example.source]["true"].append(example.answer)
        by_source[example.source]["pred"].append(prediction.answer)

    metrics = {
        "model": args.model,
        "split": args.split,
        "questions": len(examples),
        "accuracy": accuracy_score(y_true, y_pred),
        "source_accuracy": {
            source: accuracy_score(values["true"], values["pred"])
            for source, values in sorted(by_source.items())
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
