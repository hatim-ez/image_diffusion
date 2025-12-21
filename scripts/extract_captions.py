#!/usr/bin/env python3
"""
Extract captions from metadata shards for tokenizer training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from data.text_cleaning import clean_caption


def iter_metadata(path: Path) -> Iterator[dict]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        with path.open("r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    elif suffix in {".tsv", ".csv"}:
        import csv

        dialect = "excel-tab" if suffix == ".tsv" else "excel"
        with path.open("r") as handle:
            reader = csv.DictReader(handle, dialect=dialect)
            for row in reader:
                yield row
    elif suffix == ".parquet":
        import pyarrow.parquet as pq

        file = pq.ParquetFile(path)
        for batch in file.iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                yield row
    else:
        raise ValueError(f"Unsupported metadata format: {path.suffix}")


def extract_captions(metadata: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in iter_metadata(metadata):
            caption = record.get("TEXT") or record.get("text") or record.get("caption", "")
            caption = clean_caption(caption)
            if not caption:
                continue
            handle.write(caption + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract captions from metadata")
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata file (parquet/csv/jsonl)")
    parser.add_argument("--output", type=Path, required=True, help="Destination text file")
    args = parser.parse_args()
    extract_captions(args.metadata, args.output)


if __name__ == "__main__":
    main()
