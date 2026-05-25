from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from mcq_data import MCQExample, read_jsonl
from transformer_mcq import format_prompt


class MCQSeq2SeqDataset(Dataset):
    def __init__(self, examples: List[MCQExample], tokenizer, max_input_length: int, max_target_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        return {
            "prompt": format_prompt(example.question, example.choices),
            "target": example.choices[example.answer],
        }

    def collate(self, rows):
        encoded = self.tokenizer(
            [row["prompt"] for row in rows],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )
        labels = self.tokenizer(
            [row["target"] for row in rows],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
        ).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        encoded["labels"] = labels
        return encoded


def best_training_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune UnifiedQA on the normalized public MCQ training data.")
    parser.add_argument("--data", nargs="+", default=["data/mcqs.jsonl"])
    parser.add_argument("--base-model", default="allenai/unifiedqa-t5-small")
    parser.add_argument("--out", default="models/unifiedqa_mcq_finetuned")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=32)
    parser.add_argument("--max-train", type=int, default=0, help="Optional cap for quick experiments.")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples: List[MCQExample] = []
    for data_path in args.data:
        examples.extend(example for example in read_jsonl(Path(data_path)) if example.split == "train")
    random.shuffle(examples)
    if args.max_train:
        examples = examples[: args.max_train]

    device = best_training_device()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model).to(device)
    if args.freeze_encoder:
        for parameter in model.get_encoder().parameters():
            parameter.requires_grad = False
    model.train()

    dataset = MCQSeq2SeqDataset(
        examples,
        tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    total_steps = args.epochs * len(loader)
    optimizer_steps = 0
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in loader:
            step += 1
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if step % args.gradient_accumulation_steps == 0 or step == total_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            running_loss += float(output.loss.detach().cpu())
            if step == 1 or step % 100 == 0 or step == total_steps:
                avg_loss = running_loss / min(100, step if step < 100 else 100)
                print(
                    f"epoch={epoch + 1} step={step}/{total_steps} "
                    f"optimizer_steps={optimizer_steps} loss={avg_loss:.4f}",
                    flush=True,
                )
                running_loss = 0.0

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    metrics = {
        "base_model": args.base_model,
        "data": args.data,
        "train_questions": len(examples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "freeze_encoder": args.freeze_encoder,
        "optimizer_steps": optimizer_steps,
        "seed": args.seed,
    }
    (out_path / "finetune_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved fine-tuned model to {out_path}")


if __name__ == "__main__":
    main()
