from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from mcq_data import ALL_LETTERS, MCQExample


INSTRUCTION_RE = re.compile(r"^Answer the following multiple choice question\..*?\n\n", re.DOTALL)
FIRST_CHOICE_RE = re.compile(r"(?m)^\s*A\s*[:.)]\s+")
CHOICE_LABEL_RE = re.compile(r"(?m)^\s*([A-Z])\s*[:.)]\s+")


def normalize_answer(raw_answer) -> Optional[str]:
    text = str(raw_answer or "").upper().strip()
    if text in ALL_LETTERS:
        return text
    match = re.search(r"\b([A-Z])\b", text)
    return match.group(1) if match else None


def extract_choices(raw_options) -> Dict[str, str]:
    if raw_options is None:
        return {}
    entries = raw_options if isinstance(raw_options, list) else [raw_options]
    choices: Dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for letter, raw_text in entry.items():
            normalized_letter = str(letter).upper().strip()
            if normalized_letter not in ALL_LETTERS or raw_text is None:
                continue
            text = str(raw_text).strip()
            if text:
                choices[normalized_letter] = text
    return {letter: choices[letter] for letter in ALL_LETTERS if letter in choices}


def task_content_from_archive(raw_archive: bytes) -> tuple[str, dict, dict]:
    with tarfile.open(fileobj=io.BytesIO(raw_archive), mode="r:gz") as archive:
        instruction_file = archive.extractfile("instruction.md")
        verifier_file = archive.extractfile("tests/verifier_data.json")
        metadata_file = archive.extractfile("metadata.json")
        instruction = instruction_file.read().decode("utf-8") if instruction_file else ""
        verifier = json.loads(verifier_file.read().decode("utf-8")) if verifier_file else {}
        metadata = json.loads(metadata_file.read().decode("utf-8")) if metadata_file else {}
    return instruction, verifier, metadata


def strip_task_wrapper(content: str) -> str:
    marker = "\n---\n\n"
    if marker in content:
        content = content.split(marker, 1)[1]
    return content.strip()


def extract_choices_from_content(content: str) -> Dict[str, str]:
    matches = list(CHOICE_LABEL_RE.finditer(content))
    choices: Dict[str, str] = {}
    for index, match in enumerate(matches):
        letter = match.group(1)
        if letter not in ALL_LETTERS:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        text = content[start:end].strip()
        if text:
            choices[letter] = " ".join(text.split())
    return {letter: choices[letter] for letter in ALL_LETTERS if letter in choices}


def extract_prompt_content(row: dict) -> str:
    params = row.get("responses_create_params") or {}
    if isinstance(params, dict):
        for message in params.get("input") or []:
            if isinstance(message, dict) and message.get("content"):
                return str(message["content"]).strip()
    return str(row.get("problem") or row.get("question") or "").strip()


def clean_question(content: str) -> str:
    question = INSTRUCTION_RE.sub("", strip_task_wrapper(content), count=1)
    choice_match = FIRST_CHOICE_RE.search(question)
    if choice_match:
        question = question[: choice_match.start()].strip()
    return question or content.strip()


def split_for_key(key: str, validation_ratio: float, test_ratio: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + validation_ratio:
        return "validation"
    return "train"


def iter_parquet_rows(path: Path, batch_size: int) -> Iterator[dict]:
    parquet = pq.ParquetFile(path)
    expected_columns = ["responses_create_params", "expected_answer", "uuid", "options", "path", "task_binary"]
    columns = [column for column in expected_columns if column in parquet.schema.names]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        yield from batch.to_pylist()


def normalize_rows(
    rows: Iterable[dict],
    source: str,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> Iterator[MCQExample]:
    for index, row in enumerate(rows):
        metadata = {}
        if row.get("task_binary"):
            try:
                content, verifier, metadata = task_content_from_archive(row["task_binary"])
            except (tarfile.TarError, KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            choices = extract_choices_from_content(strip_task_wrapper(content))
            answer = normalize_answer(verifier.get("expected_answer"))
        else:
            content = extract_prompt_content(row)
            choices = extract_choices(row.get("options")) or extract_choices_from_content(content)
            answer = normalize_answer(row.get("expected_answer"))
        if not answer or answer not in choices or not 2 <= len(choices) <= len(ALL_LETTERS):
            continue
        question = clean_question(content)
        if not question or any(not choice for choice in choices.values()):
            continue
        row_id = str(row.get("uuid") or metadata.get("source_uuid") or row.get("path") or f"row-{index}")
        yield MCQExample(
            id=f"{source}:{row_id}",
            source=source,
            split=split_for_key(row_id, validation_ratio, test_ratio, seed),
            question=question,
            choices=choices,
            answer=answer,
        )


def source_size(info) -> int:
    return sum((sibling.size or 0) for sibling in info.siblings or [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize a NeMo Gym MCQA parquet dataset.")
    parser.add_argument("--repo", default="laion/nemotron-gym-knowledge-mcqa")
    parser.add_argument("--filename", default="tasks.parquet")
    parser.add_argument("--out", default="data/nemotron_gym_knowledge_mcqa.jsonl")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--test-ratio", type=float, default=0.02)
    parser.add_argument("--read-batch-size", type=int, default=2048)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.validation_ratio < 0 or args.test_ratio < 0 or args.validation_ratio + args.test_ratio >= 1:
        raise ValueError("validation/test ratios must be non-negative and sum to less than 1.")

    api = HfApi()
    info = api.dataset_info(args.repo, files_metadata=True)
    parquet_path = Path(hf_hub_download(repo_id=args.repo, filename=args.filename, repo_type="dataset"))

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    choice_counts = Counter()
    total = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in normalize_rows(
            iter_parquet_rows(parquet_path, args.read_batch_size),
            source=args.repo,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        ):
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
            counts[example.split] += 1
            choice_counts[len(example.choices)] += 1
            total += 1
            if args.log_every and total % args.log_every == 0:
                print(f"normalized={total:,}", flush=True)
            if args.max_examples and total >= args.max_examples:
                break

    manifest_path = Path(args.manifest_out) if args.manifest_out else output_path.with_suffix(".manifest.json")
    manifest = {
        "path": str(output_path),
        "source_repo": args.repo,
        "source_filename": args.filename,
        "source_last_modified": info.lastModified.isoformat() if info.lastModified else None,
        "source_size_bytes": source_size(info),
        "source_sha": info.sha,
        "total_examples": total,
        "splits": dict(sorted(counts.items())),
        "choice_counts": dict(sorted((str(key), value) for key, value in choice_counts.items())),
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {total:,} normalized MCQs to {output_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
