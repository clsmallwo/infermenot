from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from mcq_data import iter_public_mcqs, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize public English MCQ datasets.")
    parser.add_argument("--out", default="data/mcqs.jsonl", help="Output JSONL path.")
    parser.add_argument("--extended", action="store_true", help="Include larger public four-choice datasets.")
    parser.add_argument("--v2", action="store_true", help="Include variable-choice public datasets.")
    parser.add_argument(
        "--ap-sources",
        action="store_true",
        help="Include open textbook and science benchmark sources useful for AP-style preparation.",
    )
    parser.add_argument(
        "--extended-train-limit-per-source",
        type=int,
        default=0,
        help="Optional cap per extended source for train split only. 0 means no cap.",
    )
    args = parser.parse_args()

    output_path = Path(args.out)
    counts = Counter()

    def counted_examples():
        limit = args.extended_train_limit_per_source or None
        for example in iter_public_mcqs(
            ("train", "validation", "test"),
            include_extended=args.extended,
            include_v2=args.v2,
            include_ap_sources=args.ap_sources,
            extended_train_limit_per_source=limit,
        ):
            counts[(example.source, example.split)] += 1
            yield example

    total = write_jsonl(output_path, counted_examples())
    print(f"Wrote {total:,} normalized MCQs to {output_path}")
    for (source, split), count in sorted(counts.items()):
        print(f"{source:32s} {split:10s} {count:6d}")


if __name__ == "__main__":
    main()
