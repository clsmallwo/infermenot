from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from datasets import Image, load_dataset


LETTERS = ("A", "B", "C", "D")
ALL_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class MCQExample:
    id: str
    source: str
    split: str
    question: str
    choices: Dict[str, str]
    answer: str


def stable_shuffle(items: List[str], seed_text: str) -> List[str]:
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    copied = list(items)
    rng.shuffle(copied)
    return copied


def normalize_arc(config: str, split: str) -> Iterable[MCQExample]:
    ds = load_dataset("allenai/ai2_arc", config, split=split)
    source = f"allenai/ai2_arc/{config}"
    for row in ds:
        labels = [str(label).upper() for label in row["choices"]["label"]]
        texts = [str(text).strip() for text in row["choices"]["text"]]
        if labels != list(LETTERS) or len(texts) != 4:
            continue
        answer = str(row["answerKey"]).upper()
        if answer not in LETTERS:
            continue
        yield MCQExample(
            id=str(row["id"]),
            source=source,
            split=split,
            question=str(row["question"]).strip(),
            choices=dict(zip(LETTERS, texts)),
            answer=answer,
        )


def normalize_openbookqa(split: str) -> Iterable[MCQExample]:
    ds = load_dataset("allenai/openbookqa", "main", split=split)
    for row in ds:
        labels = [str(label).upper() for label in row["choices"]["label"]]
        texts = [str(text).strip() for text in row["choices"]["text"]]
        if labels != list(LETTERS) or len(texts) != 4:
            continue
        answer = str(row["answerKey"]).upper()
        if answer not in LETTERS:
            continue
        yield MCQExample(
            id=str(row["id"]),
            source="allenai/openbookqa/main",
            split=split,
            question=str(row["question_stem"]).strip(),
            choices=dict(zip(LETTERS, texts)),
            answer=answer,
        )


def normalize_sciq(split: str) -> Iterable[MCQExample]:
    ds = load_dataset("allenai/sciq", split=split)
    for index, row in enumerate(ds):
        raw_choices = [
            str(row["correct_answer"]).strip(),
            str(row["distractor1"]).strip(),
            str(row["distractor2"]).strip(),
            str(row["distractor3"]).strip(),
        ]
        shuffled = stable_shuffle(raw_choices, f"sciq-{split}-{index}-{row['question']}")
        choices = dict(zip(LETTERS, shuffled))
        answer = next(letter for letter, text in choices.items() if text == row["correct_answer"].strip())
        yield MCQExample(
            id=f"sciq-{split}-{index}",
            source="allenai/sciq",
            split=split,
            question=str(row["question"]).strip(),
            choices=choices,
            answer=answer,
        )


def normalize_mmlu(split: str) -> Iterable[MCQExample]:
    source_split = {"train": "auxiliary_train", "validation": "validation", "test": "test"}[split]
    ds = load_dataset("cais/mmlu", "all", split=source_split)
    for index, row in enumerate(ds):
        choices = [str(choice).strip() for choice in row["choices"]]
        answer_index = int(row["answer"])
        if len(choices) != 4 or answer_index not in range(4):
            continue
        subject = str(row.get("subject") or "").replace("_", " ").strip()
        question = str(row["question"]).strip()
        if subject:
            question = f"[{subject}] {question}"
        yield MCQExample(
            id=f"mmlu-{source_split}-{index}",
            source="cais/mmlu/all",
            split=split,
            question=question,
            choices=dict(zip(LETTERS, choices)),
            answer=LETTERS[answer_index],
        )


def normalize_race(split: str) -> Iterable[MCQExample]:
    ds = load_dataset("ehovy/race", "all", split=split)
    for index, row in enumerate(ds):
        choices = [str(choice).strip() for choice in row["options"]]
        answer = str(row["answer"]).upper()
        if len(choices) != 4 or answer not in LETTERS:
            continue
        article = " ".join(str(row["article"]).split())
        question = str(row["question"]).strip()
        yield MCQExample(
            id=f"race-{split}-{index}-{row['example_id']}",
            source="ehovy/race/all",
            split=split,
            question=f"Passage: {article}\nQuestion: {question}",
            choices=dict(zip(LETTERS, choices)),
            answer=answer,
        )


def normalize_medmcqa(split: str) -> Iterable[MCQExample]:
    ds = load_dataset("openlifescienceai/medmcqa", split=split)
    for row in ds:
        answer_index = int(row["cop"])
        if answer_index not in range(4):
            continue
        question = str(row["question"]).strip()
        subject = str(row.get("subject_name") or "").strip()
        topic = str(row.get("topic_name") or "").strip()
        context = " / ".join(part for part in (subject, topic) if part)
        if context:
            question = f"[{context}] {question}"
        yield MCQExample(
            id=str(row["id"]),
            source="openlifescienceai/medmcqa",
            split=split,
            question=question,
            choices={
                "A": str(row["opa"]).strip(),
                "B": str(row["opb"]).strip(),
                "C": str(row["opc"]).strip(),
                "D": str(row["opd"]).strip(),
            },
            answer=LETTERS[answer_index],
        )


def normalize_commonsenseqa(split: str) -> Iterable[MCQExample]:
    ds = load_dataset("tau/commonsense_qa", split=split)
    for row in ds:
        labels = [str(label).upper() for label in row["choices"]["label"]]
        texts = [str(text).strip() for text in row["choices"]["text"]]
        answer = str(row.get("answerKey", "")).upper()
        if not labels or len(labels) != len(texts) or answer not in labels:
            continue
        yield MCQExample(
            id=str(row["id"]),
            source="tau/commonsense_qa",
            split=split,
            question=str(row["question"]).strip(),
            choices=dict(zip(labels, texts)),
            answer=answer,
        )


def normalize_qasc(split: str) -> Iterable[MCQExample]:
    ds = load_dataset("allenai/qasc", split=split)
    for row in ds:
        labels = [str(label).upper() for label in row["choices"]["label"]]
        texts = [str(text).strip() for text in row["choices"]["text"]]
        answer = str(row["answerKey"]).upper()
        if not labels or len(labels) != len(texts) or answer not in labels:
            continue
        question = str(row["question"]).strip()
        fact1 = str(row.get("fact1") or "").strip()
        fact2 = str(row.get("fact2") or "").strip()
        facts = " ".join(part for part in (fact1, fact2) if part)
        if facts:
            question = f"{question}\nFacts: {facts}"
        yield MCQExample(
            id=str(row["id"]),
            source="allenai/qasc",
            split=split,
            question=question,
            choices=dict(zip(labels, texts)),
            answer=answer,
        )


def normalize_hellaswag(split: str) -> Iterable[MCQExample]:
    ds = load_dataset("Rowan/hellaswag", split=split)
    for row in ds:
        endings = [str(ending).strip() for ending in row["endings"]]
        if len(endings) != 4:
            continue
        label = str(row.get("label", ""))
        if not label.isdigit() or int(label) not in range(4):
            continue
        context = " ".join(str(row["ctx"]).split())
        activity = str(row.get("activity_label") or "").strip()
        question = f"Choose the most plausible ending. Context: {context}"
        if activity:
            question = f"[{activity}] {question}"
        yield MCQExample(
            id=f"hellaswag-{split}-{row['ind']}",
            source="Rowan/hellaswag",
            split=split,
            question=question,
            choices=dict(zip(LETTERS, endings)),
            answer=LETTERS[int(label)],
        )


def normalize_scienceqa(split: str) -> Iterable[MCQExample]:
    ds = load_dataset("derek-thomas/ScienceQA", split=split).cast_column("image", Image(decode=False))
    for index, row in enumerate(ds):
        if row.get("task") != "closed choice":
            continue
        if row.get("image") is not None:
            continue
        choices = [str(choice).strip() for choice in row["choices"]]
        answer_index = int(row["answer"])
        if len(choices) < 2 or len(choices) > len(ALL_LETTERS) or answer_index not in range(len(choices)):
            continue
        question = str(row["question"]).strip()
        hint = str(row.get("hint") or "").strip()
        context = " / ".join(
            part
            for part in (
                str(row.get("grade") or "").strip(),
                str(row.get("subject") or "").strip(),
                str(row.get("topic") or "").strip(),
                str(row.get("category") or "").strip(),
            )
            if part
        )
        if hint:
            question = f"Context: {hint}\nQuestion: {question}"
        if context:
            question = f"[ScienceQA / {context}] {question}"
        letters = ALL_LETTERS[: len(choices)]
        yield MCQExample(
            id=f"scienceqa-{split}-{index}",
            source="derek-thomas/ScienceQA",
            split=split,
            question=question,
            choices=dict(zip(letters, choices)),
            answer=letters[answer_index],
        )


def normalize_ck12_tqa(split: str) -> Iterable[MCQExample]:
    ds = load_dataset("notefill/ck12-tqa-instruction", split=split)
    for row in ds:
        if row.get("has_diagram"):
            continue
        choices = [str(choice).strip() for choice in row.get("options") or []]
        labels = [str(label).upper().strip() for label in row.get("option_labels") or []]
        answer = str(row.get("output") or "").upper().strip()
        if len(choices) < 2 or len(choices) > len(ALL_LETTERS) or len(choices) != len(labels):
            continue
        if labels != list(ALL_LETTERS[: len(labels)]) or answer not in labels:
            continue
        question = str(row.get("input") or "").split("\n\nOptions:", 1)[0].strip()
        lesson = str(row.get("lesson_name") or "").strip()
        if lesson:
            question = f"[CK-12 / {lesson}] {question}"
        yield MCQExample(
            id=str(row["id"]),
            source="notefill/ck12-tqa-instruction",
            split=split,
            question=question,
            choices=dict(zip(labels, choices)),
            answer=answer,
        )


def take_at_most(examples: Iterable[MCQExample], limit: Optional[int]) -> Iterator[MCQExample]:
    if limit is None or limit <= 0:
        yield from examples
        return
    for index, example in enumerate(examples):
        if index >= limit:
            break
        yield example


def sample_at_most(examples: Iterable[MCQExample], limit: Optional[int], seed_text: str) -> Iterator[MCQExample]:
    if limit is None or limit <= 0:
        yield from examples
        return
    buffered = list(examples)
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    rng.shuffle(buffered)
    yield from buffered[:limit]


def iter_public_mcqs(
    splits: Iterable[str],
    include_extended: bool = False,
    include_v2: bool = False,
    include_ap_sources: bool = False,
    extended_train_limit_per_source: Optional[int] = None,
) -> Iterable[MCQExample]:
    for split in splits:
        for config in ("ARC-Easy", "ARC-Challenge"):
            yield from normalize_arc(config, split)
        yield from normalize_openbookqa(split)
        yield from normalize_sciq(split)
        if include_extended:
            limit = extended_train_limit_per_source if split == "train" else None
            yield from sample_at_most(normalize_mmlu(split), limit, f"mmlu-{split}")
            yield from sample_at_most(normalize_race(split), limit, f"race-{split}")
            yield from sample_at_most(normalize_medmcqa(split), limit, f"medmcqa-{split}")
        if include_v2:
            limit = extended_train_limit_per_source if split == "train" else None
            yield from sample_at_most(normalize_commonsenseqa(split), limit, f"commonsenseqa-{split}")
            yield from sample_at_most(normalize_qasc(split), limit, f"qasc-{split}")
            yield from sample_at_most(normalize_hellaswag(split), limit, f"hellaswag-{split}")
        if include_ap_sources:
            yield from normalize_scienceqa(split)
            yield from normalize_ck12_tqa(split)


def write_jsonl(path: Path, examples: Iterable[MCQExample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[MCQExample]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            examples.append(MCQExample(**payload))
    return examples
