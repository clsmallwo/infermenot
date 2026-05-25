from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModelForMultipleChoice, AutoTokenizer, get_linear_schedule_with_warmup

from mcq_data import MCQExample, read_jsonl


def choice_letters(example: MCQExample) -> List[str]:
    return sorted(example.choices)


def approximate_example_cost(example: MCQExample) -> int:
    return len(example.question) + max((len(choice) for choice in example.choices.values()), default=0)


class ChoiceDataset(Dataset):
    def __init__(self, examples: List[MCQExample], tokenizer, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return index, self.examples[index]

    def collate(self, rows):
        indices = [index for index, _ in rows]
        examples = [example for _, example in rows]
        first = []
        second = []
        labels = []
        letters = choice_letters(examples[0])
        for example in examples:
            example_letters = choice_letters(example)
            if example_letters != letters:
                raise ValueError("Grouped batches must contain the same choice labels.")
            for letter in letters:
                first.append(example.question)
                second.append(example.choices[letter])
            labels.append(letters.index(example.answer))

        encoded = self.tokenizer(
            first,
            second,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_size = len(examples)
        encoded = {key: value.view(batch_size, len(letters), -1) for key, value in encoded.items()}
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        encoded["example_indices"] = indices
        return encoded


class GroupedChoiceBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        examples: List[MCQExample],
        batch_size: int,
        shuffle: bool,
        seed: int,
        length_bucketing: bool,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.length_bucketing = length_bucketing
        self.lengths = [approximate_example_cost(example) for example in examples]
        groups: Dict[tuple[str, ...], List[int]] = defaultdict(list)
        for index, example in enumerate(examples):
            groups[tuple(choice_letters(example))].append(index)
        self.groups = list(groups.values())

    def _order_indices(self, indices: List[int], rng: random.Random) -> List[int]:
        ordered = list(indices)
        if self.shuffle:
            rng.shuffle(ordered)
        if self.length_bucketing:
            window = max(self.batch_size * 32, self.batch_size)
            bucketed: List[int] = []
            for start in range(0, len(ordered), window):
                block = ordered[start : start + window]
                block.sort(key=lambda index: self.lengths[index], reverse=True)
                bucketed.extend(block)
            ordered = bucketed
        return ordered

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        batches: List[List[int]] = []
        for group in self.groups:
            indices = self._order_indices(group, rng)
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start : start + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return sum((len(group) + self.batch_size - 1) // self.batch_size for group in self.groups)


def make_loader(
    dataset: ChoiceDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    length_bucketing: bool = True,
    device: Optional[torch.device] = None,
) -> DataLoader:
    sampler = GroupedChoiceBatchSampler(
        dataset.examples,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        length_bucketing=length_bucketing,
    )
    kwargs = {
        "batch_sampler": sampler,
        "collate_fn": dataset.collate,
        "num_workers": num_workers,
        "pin_memory": bool(device and device.type == "cuda"),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_amp_dtype(mode: str, device: torch.device) -> Optional[torch.dtype]:
    if mode == "off" or device.type == "cpu":
        return None
    if mode == "fp16":
        return torch.float16
    if mode == "bf16":
        return torch.bfloat16
    if mode == "auto" and device.type in {"cuda", "mps"}:
        return torch.float16
    return None


def autocast_context(device: torch.device, dtype: Optional[torch.dtype]):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def move_batch_to_device(batch, device: torch.device):
    non_blocking = device.type == "cuda"
    return {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}


def parameter_summary(model) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def freeze_lower_layers(model, layers: int) -> None:
    if layers <= 0:
        return
    base = getattr(model, "deberta", None)
    if base is None:
        return
    for parameter in base.embeddings.parameters():
        parameter.requires_grad = False
    encoder_layers = getattr(base.encoder, "layer", [])
    for layer in list(encoder_layers)[:layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def choice_count_summary(examples: List[MCQExample]) -> Dict[int, int]:
    summary: Dict[int, int] = defaultdict(int)
    for example in examples:
        summary[len(example.choices)] += 1
    return dict(summary)


def source_count_summary(examples: List[MCQExample]) -> Dict[str, int]:
    return dict(sorted(Counter(example.source for example in examples).items()))


def deterministic_limit(examples: List[MCQExample], limit: int, seed: int) -> List[MCQExample]:
    if not limit:
        return examples
    rows = list(examples)
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def deterministic_limit_per_source(examples: List[MCQExample], limit: int, seed: int) -> List[MCQExample]:
    if not limit:
        return examples
    by_source: Dict[str, List[MCQExample]] = defaultdict(list)
    for example in examples:
        by_source[example.source].append(example)
    limited: List[MCQExample] = []
    for offset, source in enumerate(sorted(by_source)):
        rows = by_source[source]
        random.Random(seed + offset).shuffle(rows)
        limited.extend(rows[:limit])
    return limited


def filter_examples(
    examples: List[MCQExample],
    max_choices: int,
    max_question_chars: int,
    max_choice_chars: int,
) -> List[MCQExample]:
    if not max_choices and not max_question_chars and not max_choice_chars:
        return examples
    filtered = []
    for example in examples:
        if max_choices and len(example.choices) > max_choices:
            continue
        if max_question_chars and len(example.question) > max_question_chars:
            continue
        if max_choice_chars and any(len(choice) > max_choice_chars for choice in example.choices.values()):
            continue
        filtered.append(example)
    return filtered


@torch.no_grad()
def evaluate(model, loader, device, examples: List[MCQExample], amp_dtype: Optional[torch.dtype] = None):
    model.eval()
    predictions = []
    truth = []
    grouped_examples = []
    for batch in loader:
        batch_examples = [examples[index] for index in batch.pop("example_indices")]
        batch = move_batch_to_device(batch, device)
        with autocast_context(device, amp_dtype):
            logits = model(**batch).logits
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        for example in batch_examples:
            letters = choice_letters(example)
            truth.append(letters.index(example.answer))
            grouped_examples.append(example)

    by_source = defaultdict(lambda: {"true": [], "pred": []})
    for example, true, pred in zip(grouped_examples, truth, predictions):
        by_source[example.source]["true"].append(true)
        by_source[example.source]["pred"].append(pred)
    model.train()
    return {
        "accuracy": accuracy_score(truth, predictions),
        "source_accuracy": {
            source: accuracy_score(values["true"], values["pred"])
            for source, values in sorted(by_source.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised variable-choice transformer classifier.")
    parser.add_argument("--data", nargs="+", default=["data/mcqs.jsonl"])
    parser.add_argument("--base-model", default="microsoft/deberta-v3-small")
    parser.add_argument("--out", default="models/deberta_mcq_classifier")
    parser.add_argument("--metrics-out", default="models/deberta_mcq_metrics.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-train-per-source", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--freeze-lower-layers", type=int, default=0)
    parser.add_argument("--eval-limit-per-source", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--mixed-precision", choices=("auto", "off", "fp16", "bf16"), default="auto")
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--length-bucketing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-choices", type=int, default=0)
    parser.add_argument("--max-question-chars", type=int, default=0)
    parser.add_argument("--max-choice-chars", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.matmul_precision)

    examples: List[MCQExample] = []
    for data_path in args.data:
        examples.extend(read_jsonl(Path(data_path)))
    examples = filter_examples(
        examples,
        max_choices=args.max_choices,
        max_question_chars=args.max_question_chars,
        max_choice_chars=args.max_choice_chars,
    )
    train = [example for example in examples if example.split == "train"]
    validation = [example for example in examples if example.split == "validation"]
    test = [example for example in examples if example.split == "test"]
    random.shuffle(train)
    if args.max_train_per_source:
        counts = defaultdict(int)
        balanced = []
        for example in train:
            if counts[example.source] >= args.max_train_per_source:
                continue
            balanced.append(example)
            counts[example.source] += 1
        train = balanced
    if args.max_train:
        train = train[: args.max_train]
    validation = deterministic_limit_per_source(validation, args.eval_limit_per_source, args.seed + 1000)
    test = deterministic_limit_per_source(test, args.eval_limit_per_source, args.seed + 2000)
    if args.eval_limit:
        validation = deterministic_limit(validation, args.eval_limit, args.seed + 3000)
        test = deterministic_limit(test, args.eval_limit, args.seed + 4000)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForMultipleChoice.from_pretrained(args.base_model)
    freeze_lower_layers(model, args.freeze_lower_layers)
    device = best_device()
    model.to(device)
    amp_dtype = resolve_amp_dtype(args.mixed_precision, device)
    eval_batch_size = args.eval_batch_size or args.batch_size

    train_dataset = ChoiceDataset(train, tokenizer, args.max_length)
    validation_dataset = ChoiceDataset(validation, tokenizer, args.max_length)
    test_dataset = ChoiceDataset(test, tokenizer, args.max_length)
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
        length_bucketing=args.length_bucketing,
        device=device,
    )
    validation_loader = make_loader(
        validation_dataset,
        eval_batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
        length_bucketing=args.length_bucketing,
        device=device,
    )
    test_loader = make_loader(
        test_dataset,
        eval_batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
        length_bucketing=args.length_bucketing,
        device=device,
    )
    optimizer_kwargs = {
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    fused_adamw = bool(args.fused_adamw and device.type == "cuda")
    if fused_adamw:
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), **optimizer_kwargs)

    total_steps = args.epochs * len(train_loader)
    total_optimizer_steps = math.ceil(total_steps / args.gradient_accumulation_steps)
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    optimizer_steps = 0
    step = 0
    parameters = parameter_summary(model)
    use_scaler = amp_dtype == torch.float16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None
    print(
        "Training "
        f"train={len(train):,} validation={len(validation):,} test={len(test):,} "
        f"steps={total_steps:,} optimizer_steps={total_optimizer_steps:,} "
        f"device={device.type} amp={str(amp_dtype).replace('torch.', '') if amp_dtype else 'off'} "
        f"trainable_params={parameters['trainable']:,}/{parameters['total']:,}",
        flush=True,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        running = 0.0
        for batch in train_loader:
            step += 1
            batch.pop("example_indices", None)
            batch = move_batch_to_device(batch, device)
            with autocast_context(device, amp_dtype):
                output = model(**batch)
                loss = output.loss / args.gradient_accumulation_steps
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == total_steps:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            running += float(output.loss.detach().cpu())
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                denom = 1 if step == 1 else min(args.log_every, step)
                print(f"epoch={epoch + 1} step={step}/{total_steps} loss={running / denom:.4f}", flush=True)
                running = 0.0

    metrics = {
        "base_model": args.base_model,
        "data": args.data,
        "training_args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "max_train": args.max_train,
            "max_train_per_source": args.max_train_per_source,
            "eval_limit": args.eval_limit,
            "eval_limit_per_source": args.eval_limit_per_source,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "freeze_lower_layers": args.freeze_lower_layers,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "mixed_precision": args.mixed_precision,
            "resolved_amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else "off",
            "matmul_precision": args.matmul_precision,
            "length_bucketing": args.length_bucketing,
            "fused_adamw": fused_adamw,
            "eval_batch_size": eval_batch_size,
            "log_every": args.log_every,
            "max_choices": args.max_choices,
            "max_question_chars": args.max_question_chars,
            "max_choice_chars": args.max_choice_chars,
            "seed": args.seed,
        },
        "device": device.type,
        "parameters": parameters,
        "train_questions": len(train),
        "validation_questions": len(validation),
        "test_questions": len(test),
        "choice_counts": dict(sorted((str(key), value) for key, value in choice_count_summary(train).items())),
        "train_source_counts": source_count_summary(train),
        "validation_source_counts": source_count_summary(validation),
        "test_source_counts": source_count_summary(test),
        "optimizer_steps": optimizer_steps,
        "frozen_lower_layers": args.freeze_lower_layers,
        "validation": evaluate(model, validation_loader, device, validation, amp_dtype),
        "test": evaluate(model, test_loader, device, test, amp_dtype),
    }

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved classifier to {out_path}")


if __name__ == "__main__":
    main()
