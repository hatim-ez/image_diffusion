"""
SentencePiece tokenizer utilities for caption processing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import sentencepiece as spm


SPECIAL_TOKENS = {
    "<pad>": 0,
    "<s>": 1,
    "</s>": 2,
    "<unk>": 3,
    "<null>": 4,
}


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path, max_length: int = 77) -> None:
        self.model_path = Path(model_path)
        self.processor = spm.SentencePieceProcessor(str(self.model_path))
        self.max_length = max_length
        self.pad_id = SPECIAL_TOKENS["<pad>"]
        self.null_prompt_id = SPECIAL_TOKENS["<null>"]
        self.vocab_size = self.processor.get_piece_size()

    def encode(self, text: str) -> List[int]:
        ids = self.processor.encode(text.lower(), out_type=int)
        ids = ids[: self.max_length]
        padding_needed = self.max_length - len(ids)
        if padding_needed > 0:
            ids = ids + [self.pad_id] * padding_needed
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.processor.decode(ids)

    def hash(self) -> str:
        contents = self.model_path.read_bytes()
        return hashlib.sha1(contents).hexdigest()


def train_tokenizer(
    corpus_path: str | Path,
    output_dir: str | Path,
    vocab_size: int = 32000,
    model_type: str = "bpe",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "tokenizer.model"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(output_dir / "tokenizer"),
        vocab_size=vocab_size,
        model_type=model_type,
        bos_id=SPECIAL_TOKENS["<s>"],
        eos_id=SPECIAL_TOKENS["</s>"],
        pad_id=SPECIAL_TOKENS["<pad>"],
        unk_id=SPECIAL_TOKENS["<unk>"],
        user_defined_symbols="<null>",
    )
    meta = {
        "vocab_size": vocab_size,
        "model_type": model_type,
        "special_tokens": SPECIAL_TOKENS,
    }
    with (output_dir / "tokenizer_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    return model_path
