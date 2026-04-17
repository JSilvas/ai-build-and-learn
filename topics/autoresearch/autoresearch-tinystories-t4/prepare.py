"""
prepare.py — Data preparation for AutoResearch (T4 / TinyStories adaptation).

Adapted from Karpathy's AutoResearch prepare.py.
Original used climbmix-400b (400B tokens, H100-scale).
This version uses roneneldan/TinyStories (~2GB, suitable for T4 5-minute runs).

Responsibilities:
  - Download TinyStories from HuggingFace
  - Tokenize using a simple character-level tokenizer
  - Write train.bin and val.bin to the data/ directory
  - Write tokenizer metadata (vocab_size) to data/meta.json

Run once before starting the agent loop:
    python prepare.py

Static file — AutoResearch never modifies this.
"""

import json
import os
import struct
import numpy as np
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATASET_NAME = "roneneldan/TinyStories"
VAL_SPLIT_RATIO = 0.1        # 10% of data for validation
CHUNK_SIZE = 10_000          # encode this many stories at a time to avoid OOM


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _build_tokenizer(dataset) -> tuple[dict, dict]:
    """
    Build a character-level tokenizer by scanning a sample of stories.

    Scans the first 50,000 stories to collect the character vocabulary —
    sufficient for TinyStories which uses simple ASCII English.

    Returns:
        stoi : str -> int mapping
        itos : int -> str mapping
    """
    print("Scanning vocabulary from first 50,000 stories...")
    chars = set()
    for i, text in enumerate(dataset["text"]):
        chars.update(text)
        if i >= 50_000:
            break
    chars = sorted(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def _encode(text: str, stoi: dict) -> bytes:
    """
    Encode a string to packed uint16 bytes, skipping unknown characters.

    Returns raw bytes ready to write directly to a binary file.
    """
    ids = [stoi[ch] for ch in text if ch in stoi]
    return struct.pack(f"{len(ids)}H", *ids)


# ── Main ──────────────────────────────────────────────────────────────────────

def prepare() -> None:
    """
    Download TinyStories, tokenize, and write train.bin / val.bin.

    Encodes in chunks of CHUNK_SIZE stories to avoid loading all token IDs
    into memory at once. Writes directly to binary files as it goes.

    Output files:
        data/train.bin  — uint16 binary array of training token IDs
        data/val.bin    — uint16 binary array of validation token IDs
        data/meta.json  — tokenizer metadata (vocab_size, stoi, itos)
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading TinyStories from HuggingFace...")
    dataset = load_dataset(DATASET_NAME, split="train")

    texts = dataset["text"]
    total = len(texts)
    split_idx = int(total * (1 - VAL_SPLIT_RATIO))

    print(f"Total stories: {total:,} | Train: {split_idx:,} | Val: {total - split_idx:,}")

    # Build tokenizer from a sample (avoids loading all text into memory)
    stoi, itos = _build_tokenizer(dataset)
    actual_vocab_size = len(stoi)
    print(f"Vocabulary size: {actual_vocab_size}")

    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")
    meta_path = os.path.join(DATA_DIR, "meta.json")

    # Encode and write training split in chunks
    print("Encoding training split...")
    train_tokens = 0
    with open(train_path, "wb") as f:
        for start in range(0, split_idx, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, split_idx)
            for text in texts[start:end]:
                chunk_bytes = _encode(text, stoi)
                f.write(chunk_bytes)
                train_tokens += len(chunk_bytes) // 2
            print(f"  train: {end:,}/{split_idx:,} stories encoded ({train_tokens:,} tokens)", end="\r")
    print()

    # Encode and write validation split in chunks
    print("Encoding validation split...")
    val_tokens = 0
    with open(val_path, "wb") as f:
        for start in range(split_idx, total, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, total)
            for text in texts[start:end]:
                chunk_bytes = _encode(text, stoi)
                f.write(chunk_bytes)
                val_tokens += len(chunk_bytes) // 2
            print(f"  val: {end - split_idx:,}/{total - split_idx:,} stories encoded", end="\r")
    print()

    # Write metadata
    meta = {
        "vocab_size": actual_vocab_size,
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "dataset": DATASET_NAME,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done.")
    print(f"  train.bin : {train_tokens:,} tokens → {train_path}")
    print(f"  val.bin   : {val_tokens:,} tokens → {val_path}")
    print(f"  meta.json : vocab_size={actual_vocab_size} → {meta_path}")


if __name__ == "__main__":
    prepare()