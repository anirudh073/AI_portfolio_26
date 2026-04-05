import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

# This block makes the project root importable when the script is executed as `python data/prepare.py`.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import GPTConfig, resolve_path


# This helper writes a uint16 token array into a raw memmap file on disk.
def write_memmap(path: Path, token_ids: np.ndarray) -> None:
    memmap = np.memmap(path, dtype=np.uint16, mode="w+", shape=token_ids.shape)
    memmap[:] = token_ids
    memmap.flush()


# This block trains a SentencePiece BPE tokenizer and writes the tokenized train/val shards.
def main() -> None:
    config = GPTConfig()
    input_path = resolve_path(config.input_file)
    train_path = resolve_path(config.train_bin)
    val_path = resolve_path(config.val_bin)
    spm_model_path = resolve_path(config.spm_model)
    spm_vocab_path = spm_model_path.with_suffix(".vocab")

    # This block checks that the Shakespeare corpus exists and tells the user how to link it in.
    if not input_path.exists():
        raise FileNotFoundError(
            "Expected Shakespeare corpus at data/shakespeare.txt.\n"
            "Create it with:\n"
            "ln -s ../tinygpt/data/shakespeare.txt data/shakespeare.txt"
        )

    # This block reads the raw corpus that will be used both for tokenizer training and encoding.
    text = input_path.read_text(encoding="utf-8")

    # This block trains a BPE sentencepiece model and saves `bard.model` and `bard.vocab`.
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(spm_model_path.with_suffix("")),
        vocab_size=4000,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    # This block reloads the trained tokenizer so the full corpus can be encoded into token ids.
    sp = spm.SentencePieceProcessor(model_file=str(spm_model_path))
    token_ids = np.array(sp.encode_as_ids(text), dtype=np.uint16)

    # This block splits the encoded token stream into train and validation partitions.
    split_idx = int(len(token_ids) * config.train_split)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    # This block writes the token ids into raw memmap binaries for training.
    write_memmap(train_path, train_ids)
    write_memmap(val_path, val_ids)

    # This block reports tokenizer and dataset statistics, including token compression vs characters.
    compression_ratio = len(token_ids) / len(text)
    print(f"vocab size: {sp.get_piece_size()}")
    print(f"train tokens: {len(train_ids)}")
    print(f"val tokens: {len(val_ids)}")
    print(f"compression ratio vs characters: {compression_ratio:.4f} tokens/char")
    print(f"saved tokenizer model: {spm_model_path}")
    print(f"saved tokenizer vocab: {spm_vocab_path}")


if __name__ == "__main__":
    main()
