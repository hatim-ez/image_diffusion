#!/usr/bin/env python3
"""
Filter LAION metadata and build WebDataset shards.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Dict, Iterator, Optional

import requests
from PIL import Image, ImageOps
from tqdm import tqdm
import webdataset as wds

from data.text_cleaning import clean_caption


def iter_metadata(path: Path) -> Iterator[Dict]:
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

        dialect = "excel-tab" if path.suffix == ".tsv" else "excel"
        with path.open("r") as handle:
            reader = csv.DictReader(handle, dialect=dialect)
            for row in reader:
                yield row
    elif suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                yield row
    else:
        raise ValueError(f"Unsupported metadata format: {path.suffix}")


def is_safe(record: Dict) -> bool:
    flag = str(record.get("NSFW", record.get("nsfw", ""))).lower()
    return flag not in {"nsfw", "unsafe", "porn", "sexual"} and flag != "1"


def download_image(url: str, timeout: int = 15) -> Optional[bytes]:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content
    except Exception:
        return None


def process_image(image_bytes: bytes, size: int) -> Optional[bytes]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            img = ImageOps.fit(img, (size, size), method=Image.BICUBIC)
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=95)
            return output.getvalue()
    except Exception:
        return None


def build_webdataset(
    metadata_path: Path,
    output_dir: Path,
    shard_size: int,
    image_size: int,
    max_text_len: int,
    limit: Optional[int] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = wds.ShardWriter(str(output_dir / "%06d.tar"), maxcount=shard_size)
    kept = 0
    total = 0
    try:
        for idx, record in enumerate(tqdm(iter_metadata(metadata_path), desc="Filtering")):
            if limit and idx >= limit:
                break
            total += 1
            if not is_safe(record):
                continue
            url = record.get("URL") or record.get("url")
            caption = record.get("TEXT") or record.get("text") or record.get("caption", "")
            caption = clean_caption(caption)
            if not url or not caption:
                continue
            caption = caption[:max_text_len]
            image_bytes = download_image(url)
            if not image_bytes:
                continue
            image_bytes = process_image(image_bytes, image_size)
            if not image_bytes:
                continue
            sample = {
                "__key__": f"sample-{idx:09d}",
                "jpg": image_bytes,
                "txt": caption,
            }
            writer.write(sample)
            kept += 1
    finally:
        writer.close()
    print(f"Kept {kept} / {total} entries")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LAION WebDataset shards")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-text-len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    build_webdataset(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        image_size=args.image_size,
        max_text_len=args.max_text_len,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
