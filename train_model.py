from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from mcq_data import LETTERS, MCQExample, read_jsonl


def pair_text(example: MCQExample, letter: str) -> str:
    return f"Question: {example.question}\nAnswer candidate: {example.choices[letter]}"


def make_pairs(examples: List[MCQExample]) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    for example in examples:
        for letter in LETTERS:
            texts.append(pair_text(example, letter))
            labels.append(1 if letter == example.answer else 0)
    return texts, labels


def grouped_accuracy(model: Pipeline, examples: List[MCQExample]) -> float:
    y_true = []
    y_pred = []
    for example in examples:
        scores = model.predict_proba([pair_text(example, letter) for letter in LETTERS])[:, 1]
        predicted = LETTERS[int(np.argmax(scores))]
        y_true.append(example.answer)
        y_pred.append(predicted)
    return accuracy_score(y_true, y_pred)


def source_breakdown(model: Pipeline, examples: List[MCQExample]) -> Dict[str, float]:
    by_source: Dict[str, List[MCQExample]] = defaultdict(list)
    for example in examples:
        by_source[example.source].append(example)
    return {source: grouped_accuracy(model, rows) for source, rows in sorted(by_source.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight MCQ answer-choice inference model.")
    parser.add_argument("--data", default="data/mcqs.jsonl", help="Normalized JSONL data path.")
    parser.add_argument("--model-out", default="models/mcq_inference.joblib", help="Model artifact path.")
    parser.add_argument("--metrics-out", default="models/metrics.json", help="Metrics JSON path.")
    args = parser.parse_args()

    examples = read_jsonl(Path(args.data))
    train = [example for example in examples if example.split == "train"]
    validation = [example for example in examples if example.split == "validation"]
    test = [example for example in examples if example.split == "test"]

    x_train, y_train = make_pairs(train)
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=2.0,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="liblinear",
                    random_state=7,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    x_val, y_val = make_pairs(validation)
    pair_predictions = model.predict(x_val)
    metrics = {
        "train_questions": len(train),
        "validation_questions": len(validation),
        "test_questions": len(test),
        "validation_grouped_accuracy": grouped_accuracy(model, validation),
        "test_grouped_accuracy": grouped_accuracy(model, test),
        "validation_pair_report": classification_report(y_val, pair_predictions, output_dict=True),
        "validation_source_accuracy": source_breakdown(model, validation),
        "test_source_accuracy": source_breakdown(model, test),
    }

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Validation grouped accuracy: {metrics['validation_grouped_accuracy']:.3f}")
    print(f"Test grouped accuracy:       {metrics['test_grouped_accuracy']:.3f}")


if __name__ == "__main__":
    main()
