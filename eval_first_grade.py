from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

import requests


@dataclass
class ExamItem:
    section: str
    num: int
    question: str
    choices: Dict[str, str]
    answer: str


def build_exam() -> List[ExamItem]:
    items: List[ExamItem] = []

    def add(section: str, num: int, question: str, choices: Dict[str, str], answer: str) -> None:
        items.append(ExamItem(section, num, question, choices, answer))

    english = [
        (1, "Over the hill and across the field", "NS"),
        (2, "Dogs bark", "S."),
        (3, "What time is it", "S?"),
        (4, "A happy boy", "NS"),
        (5, "Come to my house", "S."),
        (6, "Many children at the park", "NS"),
        (7, "How good God is", "S."),
        (8, "The sky is blue", "S."),
    ]
    sentence_choices = {
        "A": "S with period (S.)",
        "B": "S with question mark (S?)",
        "C": "NS, not a complete sentence",
        "D": "S with exclamation point (S!)",
    }
    key = {"S.": "A", "S?": "B", "NS": "C", "S!": "D"}
    for num, phrase, answer in english:
        add(
            "English",
            num,
            f'If the group of words is a sentence, choose the correct sentence mark. If it is not a sentence, choose NS: "{phrase}"',
            sentence_choices,
            key[answer],
        )

    add(
        "English",
        9,
        "Which words from this list should begin with a capital letter? fr. brady kittens washington school baby monday god friends june sam book mary",
        {
            "A": "fr. brady, washington, monday, god, june, sam, mary",
            "B": "kittens, school, baby, friends, book",
            "C": "fr. brady, kittens, school, baby, friends, book",
            "D": "washington, monday, june only",
        },
        "A",
    )

    math_items = [
        (1, "8 + 7", {"A": "13", "B": "14", "C": "15", "D": "16"}, "C"),
        (2, "4 + 9", {"A": "11", "B": "12", "C": "13", "D": "14"}, "C"),
        (3, "16 - 9", {"A": "5", "B": "6", "C": "7", "D": "8"}, "C"),
        (4, "14 - 6", {"A": "6", "B": "7", "C": "8", "D": "9"}, "C"),
        (5, "5 + 3 + 2", {"A": "8", "B": "9", "C": "10", "D": "11"}, "C"),
        (6, "4 + 2 + 7", {"A": "11", "B": "12", "C": "13", "D": "14"}, "C"),
        (7, "John bought a pencil for 6 cents and an eraser for 5 cents. How much did he spend in all?", {"A": "10 cents", "B": "11 cents", "C": "12 cents", "D": "13 cents"}, "B"),
        (8, "Mary had 15 cents. She bought a holy card for 10 cents. How much does she have left?", {"A": "3 cents", "B": "4 cents", "C": "5 cents", "D": "6 cents"}, "C"),
        (9, "How many objects are shown as 2 tens and 6 ones?", {"A": "24", "B": "25", "C": "26", "D": "27"}, "C"),
        (10, "If there are ten balls in a row, which ball is the third ball?", {"A": "first", "B": "second", "C": "third", "D": "fourth"}, "C"),
        (11, "17 - 5", {"A": "10", "B": "11", "C": "12", "D": "13"}, "C"),
        (12, "15 - 8", {"A": "6", "B": "7", "C": "8", "D": "9"}, "B"),
        (13, "6 + 7", {"A": "11", "B": "12", "C": "13", "D": "14"}, "C"),
        (14, "9 + 3", {"A": "10", "B": "11", "C": "12", "D": "13"}, "C"),
        (15, "What time is shown when the minute hand points to 6 and the hour hand is halfway between 9 and 10?", {"A": "9:00", "B": "9:30", "C": "10:30", "D": "6:45"}, "B"),
        (16, "Write the missing numbers: 66, 67, __, __, __, __, __, 73", {"A": "68, 69, 70, 71, 72", "B": "67, 68, 69, 70, 71", "C": "69, 70, 71, 72, 73", "D": "68, 70, 71, 72, 74"}, "A"),
        (17, "Write the missing numbers: __, 136, __, __, 139, 140, __", {"A": "134, 137, 138, 141", "B": "135, 137, 138, 141", "C": "135, 136, 137, 141", "D": "135, 137, 139, 141"}, "B"),
        (18, "Write the missing numbers: 408, __, __, 411, __, __, 414", {"A": "409, 410, 412, 413", "B": "407, 409, 412, 413", "C": "409, 411, 412, 413", "D": "410, 411, 412, 413"}, "A"),
        (19, "38 cents + 27 cents", {"A": "55 cents", "B": "60 cents", "C": "65 cents", "D": "75 cents"}, "C"),
        (20, "35 + 87", {"A": "112", "B": "122", "C": "132", "D": "147"}, "B"),
        (21, "26 + 84 + 37", {"A": "137", "B": "147", "C": "157", "D": "167"}, "B"),
        (22, "62 - 7", {"A": "45", "B": "55", "C": "57", "D": "69"}, "B"),
        (23, "43 - 6", {"A": "35", "B": "36", "C": "37", "D": "38"}, "C"),
        (24, "86 - 9", {"A": "67", "B": "76", "C": "77", "D": "78"}, "C"),
        (25, "25 cents - 8 cents", {"A": "15 cents", "B": "16 cents", "C": "17 cents", "D": "18 cents"}, "C"),
        (26, "Sam had 4 cars. Chris had 7 cars. Matt had 12 cars. How many in all?", {"A": "21", "B": "22", "C": "23", "D": "24"}, "C"),
        (27, "Match the shapes from top to bottom: rectangle, triangle, square, circle. Which order is correct?", {"A": "rectangle, triangle, square, circle", "B": "circle, square, triangle, rectangle", "C": "triangle, rectangle, circle, square", "D": "square, circle, rectangle, triangle"}, "A"),
        (28, "The instruction says: Color 1/3 of a circle split into 3 equal parts. How many parts should be colored?", {"A": "1 part", "B": "2 parts", "C": "3 parts", "D": "4 parts"}, "A"),
        (29, "The instruction says: Color 3/4 of a square split into 4 equal parts. How many parts should be colored?", {"A": "1 part", "B": "2 parts", "C": "3 parts", "D": "4 parts"}, "C"),
        (30, "The instruction says: Color 1/2 of a circle split into 2 equal parts. How many parts should be colored?", {"A": "0 parts", "B": "1 part", "C": "2 parts", "D": "3 parts"}, "B"),
    ]
    for num, question, choices, answer in math_items:
        add("Math", num, question, choices, answer)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local first-grade objective benchmark through the web API.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/predict")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    results = []
    for item in build_exam():
        response = requests.post(args.url, json={"question": item.question, "choices": item.choices}, timeout=30)
        response.raise_for_status()
        data = response.json()
        prediction = data["answer"]
        results.append(
            {
                **item.__dict__,
                "prediction": prediction,
                "correct": prediction == item.answer,
                "probabilities": data["probabilities"],
                "device": data["device"],
            }
        )

    correct = sum(row["correct"] for row in results)
    print(f"Score: {correct}/{len(results)} = {correct / len(results):.1%}")
    for section in sorted({row["section"] for row in results}):
        rows = [row for row in results if row["section"] == section]
        section_correct = sum(row["correct"] for row in rows)
        print(f"{section}: {section_correct}/{len(rows)} = {section_correct / len(rows):.1%}")

    print("\nMissed items:")
    for row in results:
        if not row["correct"]:
            print(
                f"{row['section']} {row['num']}: predicted {row['prediction']} "
                f"({row['choices'][row['prediction']]}) | correct {row['answer']} ({row['choices'][row['answer']]})"
            )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
