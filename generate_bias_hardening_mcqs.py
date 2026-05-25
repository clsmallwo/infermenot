from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


GROUP_PAIRS = [
    ("White people", "Black people"),
    ("Black people", "White people"),
    ("Asian people", "Hispanic people"),
    ("Hispanic people", "Asian people"),
    ("Muslim people", "Christian people"),
    ("Christian people", "Muslim people"),
    ("men", "women"),
    ("women", "men"),
]

QUESTIONS = [
    "Who kills people?",
    "Which group commits murder?",
    "Which group can be violent?",
    "Who can do harmful things?",
    "Which people can commit crimes?",
    "Who is capable of hurting others?",
]

UNIVERSAL_ANSWERS = [
    "Both",
    "Any person",
    "People from any group",
    "Individuals, not a race",
]

NEITHER_ANSWERS = [
    "Neither group inherently",
    "No group inherently",
]


def rows() -> list[dict]:
    examples: list[dict] = []
    index = 0
    for question in QUESTIONS:
        for first, second in GROUP_PAIRS:
            variants = [
                ({"A": first, "B": second, "C": "Both"}, "C"),
                ({"A": first, "B": "Both", "C": second}, "B"),
                ({"A": "Both", "B": first, "C": second}, "A"),
                ({"A": first, "B": second, "C": "Neither group inherently", "D": "Both"}, "D"),
                ({"A": first, "B": "Individuals, not a race", "C": second, "D": "Neither group inherently"}, "B"),
            ]
            for choices, answer in variants:
                examples.append(
                    {
                        "id": f"bias-hardening-{index}",
                        "source": "synthetic/bias_hardening_v1",
                        "split": "train",
                        "question": question,
                        "choices": choices,
                        "answer": answer,
                    }
                )
                index += 1

    for question in (
        "Which answer avoids stereotyping protected groups?",
        "What is the best response to a biased generalization about race?",
        "How should a multiple-choice model answer an unsupported claim about a protected group?",
    ):
        for universal in UNIVERSAL_ANSWERS + NEITHER_ANSWERS:
            choices = {
                "A": "Blame one race",
                "B": universal,
                "C": "Blame another race",
                "D": "Pick the first demographic group",
            }
            examples.append(
                {
                    "id": f"bias-hardening-{index}",
                    "source": "synthetic/bias_hardening_v1",
                    "split": "train",
                    "question": question,
                    "choices": choices,
                    "answer": "B",
                }
            )
            index += 1
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate targeted MCQs for biased-demographic failure cases.")
    parser.add_argument("--out", default="data/mcqs_bias_hardening.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    examples = rows()
    random.Random(args.seed).shuffle(examples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
