from dataclasses import asdict, dataclass
from pathlib import Path


# This dataclass centralizes model, data, training, and sampling hyperparameters.
@dataclass
class GPTConfig:
    vocab_size: int = 0
    context_len: int = 256
    n_embd: int = 512
    n_head: int = 8
    n_layer: int = 12
    ffn_dim: int = 2048
    dropout: float = 0.2

    train_split: float = 0.9
    batch_size: int = 32
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 100
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "cuda"
    dtype: str = "float16"
    compile: bool = True

    temperature: float = 0.6
    top_k: int = 40
    default_prompt: str = "To be, or not to be"

    input_file: str = "data/shakespeare.txt"
    train_bin: str = "data/train.bin"
    val_bin: str = "data/val.bin"
    spm_model: str = "data/bard.model"
    out_dir: str = "out"
    checkpoint_file: str = "out/ckpt.pt"
    loss_curve_file: str = "out/loss_curve.png"

    def to_dict(self) -> dict:
        return asdict(self)


# This helper returns the project root so every script can build relative paths consistently.
def project_root() -> Path:
    return Path(__file__).resolve().parent


# This helper resolves a config path from the project root to an absolute filesystem path.
def resolve_path(relative_path: str) -> Path:
    return project_root() / relative_path
