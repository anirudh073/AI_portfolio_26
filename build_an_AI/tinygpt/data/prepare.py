import pickle
import sys
from pathlib import Path

import numpy as np

# This block makes the project root importable when the script is executed as `python data/prepare.py`.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import GPTConfig, resolve_path


# This block loads the raw Lovecraft corpus and builds the character vocabulary.
def main() -> None:
    config = GPTConfig()
    input_path = resolve_path(config.input_file)
    meta_path = resolve_path(config.meta_file)
    train_path = resolve_path(config.train_bin)
    val_path = resolve_path(config.val_bin)

    # This block reads the full text corpus without modifying the source file.
    text = input_path.read_text(encoding="utf-8")

    # This block derives deterministic character-to-id and id-to-character mappings.
    chars = sorted(set(text))
    stoi = {ch: idx for idx, ch in enumerate(chars)}
    itos = {idx: ch for idx, ch in enumerate(chars)}
    vocab_size = len(chars)

    # This block saves the vocabulary metadata needed by training and downstream tooling.
    meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
    with meta_path.open("wb") as handle:
        pickle.dump(meta, handle)

    # This block splits the corpus into training and validation segments.
    split_idx = int(len(text) * config.train_split)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # This block encodes characters into integer ids for efficient memmap storage.
    train_ids = np.array([stoi[ch] for ch in train_text], dtype=np.uint16)
    val_ids = np.array([stoi[ch] for ch in val_text], dtype=np.uint16)

    # This block writes the training ids to a raw uint16 memmap file.
    train_memmap = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=train_ids.shape)
    train_memmap[:] = train_ids
    train_memmap.flush()

    # This block writes the validation ids to a raw uint16 memmap file.
    val_memmap = np.memmap(val_path, dtype=np.uint16, mode="w+", shape=val_ids.shape)
    val_memmap[:] = val_ids
    val_memmap.flush()

    # This block prints the dataset statistics requested by the project spec.
    print(f"vocab size: {vocab_size}")
    print(f"train tokens: {len(train_ids)}")
    print(f"val tokens: {len(val_ids)}")


if __name__ == "__main__":
    main()
