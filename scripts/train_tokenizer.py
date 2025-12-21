#!/usr/bin/env python3
"""
Train SentencePiece tokenizer on caption corpus.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.tokenizer import train_tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tokenizer from captions")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()
    train_tokenizer(args.corpus, args.output_dir, args.vocab_size)
