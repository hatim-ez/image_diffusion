#!/usr/bin/env python3
"""
Download LAION metadata shards to disk for subsequent filtering.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path
from typing import List

import requests
from tqdm import tqdm


def fetch(url: str, output: Path, chunk_size: int = 1 << 20) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                handle.write(chunk)


def main(urls: List[str], output_dir: Path, workers: int = 4) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for url in urls:
            filename = url.split("/")[-1]
            target = output_dir / filename
            futures.append(executor.submit(fetch, url, target))
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading"):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LAION metadata shards")
    parser.add_argument("--url-file", type=Path, required=True, help="Text file with one URL per line")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    urls = [
        line.strip()
        for line in args.url_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    main(urls, args.output_dir, args.workers)
