from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score

from mcq_data import read_jsonl
from passage_inference import DEFAULT_PASSAGE_MODEL, LongformerRaceMCQ


def race_level(example_id: str) -> str:
    lowered = example_id.lower()
    if "high" in lowered:
        return "high"
    if "middle" in lowered:
        return "middle"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate English passage-based RACE MCQs.")
    parser.add_argument("--data", default="data/mcqs_ap_ready.jsonl")
    parser.add_argument("--model", default=DEFAULT_PASSAGE_MODEL)
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="models/longformer_race_english_passage_metrics.json")
    args = parser.parse_args()

    rows = [
        example
        for example in read_jsonl(Path(args.data))
        if example.split == args.split and example.source == "ehovy/race/all"
    ]
    if args.limit:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.limit]

    classifier = LongformerRaceMCQ(args.model)
    truth: list[str] = []
    predictions: list[str] = []
    by_level = defaultdict(lambda: {"true": [], "pred": []})

    for index, example in enumerate(rows, 1):
        result = classifier.predict(example.question, example.choices)
        truth.append(example.answer)
        predictions.append(result.answer)
        level = race_level(example.id)
        by_level[level]["true"].append(example.answer)
        by_level[level]["pred"].append(result.answer)
        if index % 100 == 0 or index == len(rows):
            print(
                f"evaluated {index}/{len(rows)} accuracy={accuracy_score(truth, predictions):.3f}",
                flush=True,
            )

    metrics = {
        "model": args.model,
        "data": args.data,
        "source": "ehovy/race/all",
        "split": args.split,
        "questions": len(rows),
        "accuracy": accuracy_score(truth, predictions) if rows else 0.0,
        "level_accuracy": {
            level: {
                "questions": len(values["true"]),
                "accuracy": accuracy_score(values["true"], values["pred"]) if values["true"] else 0.0,
            }
            for level, values in sorted(by_level.items())
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
