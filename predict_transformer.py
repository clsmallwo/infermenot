from __future__ import annotations

import argparse

from mcq_data import ALL_LETTERS, MCQExample
from transformer_mcq import DEFAULT_TRANSFORMER_MODEL, UnifiedQAMCQPiper


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick an answer using the stronger UnifiedQA model.")
    parser.add_argument("--model", default=DEFAULT_TRANSFORMER_MODEL)
    parser.add_argument("--score-mode", choices=("choice-likelihood", "generate"), default="choice-likelihood")
    parser.add_argument("--question", required=True)
    for letter in ALL_LETTERS[:8]:
        parser.add_argument(f"--{letter.lower()}", default="", help=f"Choice {letter}.")
    args = parser.parse_args()

    choices = {
        letter: getattr(args, letter.lower())
        for letter in ALL_LETTERS[:8]
        if getattr(args, letter.lower()).strip()
    }
    scorer = UnifiedQAMCQPiper(args.model)
    if args.score_mode == "generate":
        prediction = scorer.predict_batch(
            [
                MCQExample(
                    id="manual",
                    source="manual",
                    split="manual",
                    question=args.question,
                    choices=choices,
                    answer="A",
                )
            ],
            batch_size=1,
        )[0]
    else:
        prediction = scorer.predict(args.question, choices)

    print(f"Prediction: {prediction.answer}")
    print(f"Model: {args.model}")
    print(f"Score mode: {args.score_mode}")
    for letter in choices:
        print(f"{letter}: score={prediction.scores[letter]:.3f} choice={choices[letter]}")


if __name__ == "__main__":
    main()
