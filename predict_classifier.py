from __future__ import annotations

import argparse

from choice_inference import ChoiceClassifier, DEFAULT_CLASSIFIER_MODEL
from mcq_data import ALL_LETTERS


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick an answer with the supervised choice classifier.")
    parser.add_argument("--model", default=DEFAULT_CLASSIFIER_MODEL)
    parser.add_argument("--question", required=True)
    for letter in ALL_LETTERS[:8]:
        parser.add_argument(f"--{letter.lower()}", default="", help=f"Choice {letter}.")
    args = parser.parse_args()

    choices = {
        letter: getattr(args, letter.lower())
        for letter in ALL_LETTERS[:8]
        if getattr(args, letter.lower()).strip()
    }
    result = ChoiceClassifier(args.model).predict(args.question, choices)
    print(f"Prediction: {result.answer}")
    print(f"Device: {result.device}")
    for letter in choices:
        probability = result.probabilities[letter]
        print(f"{letter}: probability={probability:.3f} choice={choices[letter]}")


if __name__ == "__main__":
    main()
