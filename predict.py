from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from mcq_data import LETTERS


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick the most likely answer choice for a new A/B/C/D MCQ.")
    parser.add_argument("--model", default="models/mcq_inference.joblib", help="Trained model path.")
    parser.add_argument("--question", required=True, help="Question text.")
    parser.add_argument("--a", required=True, help="Choice A.")
    parser.add_argument("--b", required=True, help="Choice B.")
    parser.add_argument("--c", required=True, help="Choice C.")
    parser.add_argument("--d", required=True, help="Choice D.")
    args = parser.parse_args()

    model = joblib.load(Path(args.model))
    choices = {"A": args.a, "B": args.b, "C": args.c, "D": args.d}
    pair_texts = [f"Question: {args.question}\nAnswer candidate: {choices[letter]}" for letter in LETTERS]
    raw_scores = model.predict_proba(pair_texts)[:, 1]
    probabilities = np.exp(raw_scores) / np.exp(raw_scores).sum()
    best = LETTERS[int(np.argmax(raw_scores))]

    print(f"Prediction: {best}")
    for letter, probability, score in zip(LETTERS, probabilities, raw_scores):
        print(f"{letter}: probability={probability:.3f} score={score:.3f} choice={choices[letter]}")


if __name__ == "__main__":
    main()
