import math
from contextlib import nullcontext

import matplotlib
import numpy as np
import sentencepiece as spm
import torch

from config import GPTConfig, resolve_path
from model import GPT

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# This helper samples random subsequences from a memmapped token array.
def get_batch(data: np.memmap, config: GPTConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - config.context_len - 1
    if max_start <= 0:
        raise ValueError("Dataset is too small for the configured context length")

    # This block chooses random start positions and slices out aligned input and target sequences.
    starts = torch.randint(0, max_start + 1, (config.batch_size,))
    x = np.stack([data[start : start + config.context_len] for start in starts.tolist()])
    y = np.stack([data[start + 1 : start + 1 + config.context_len] for start in starts.tolist()])

    # This block moves the batch to the target device as integer token ids.
    x_tensor = torch.from_numpy(x.astype(np.int64)).to(device)
    y_tensor = torch.from_numpy(y.astype(np.int64)).to(device)
    return x_tensor, y_tensor


# This helper computes a cosine-decayed learning rate between the configured bounds.
def get_learning_rate(iteration: int, config: GPTConfig) -> float:
    if iteration >= config.max_iters:
        return config.min_learning_rate

    # This block interpolates between the start and minimum learning rates with cosine decay.
    decay_ratio = iteration / config.max_iters
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)


# This helper estimates mean train and validation loss over several random batches.
@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_data: np.memmap,
    val_data: np.memmap,
    config: GPTConfig,
    device: torch.device,
    autocast_context,
) -> dict:
    losses = {}
    model.eval()

    # This block runs repeated random evaluations on both dataset splits.
    for split_name, split_data in (("train", train_data), ("val", val_data)):
        split_losses = torch.zeros(config.eval_iters)
        for eval_idx in range(config.eval_iters):
            xb, yb = get_batch(split_data, config, device)
            with autocast_context():
                _, loss = model(xb, yb)
            split_losses[eval_idx] = loss.detach().float().cpu()
        losses[split_name] = split_losses.mean().item()

    model.train()
    return losses


# This helper builds parameter groups so weight decay applies only to matrix weights.
def configure_optimizer(model: GPT, config: GPTConfig) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []

    # This block separates parameters that should and should not receive weight decay.
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    # This block constructs AdamW with separate decay settings for each parameter group.
    optimizer_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optimizer_groups, lr=config.learning_rate, betas=(0.9, 0.95))


# This helper plots the recorded training and validation losses to disk.
def save_loss_curve(log_history: list[dict], config: GPTConfig, best_iter: int | None = None) -> None:
    if not log_history:
        return

    out_dir = resolve_path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # This block turns the logged metrics into a simple training curve figure.
    iters = [entry["iter"] for entry in log_history]
    train_losses = [entry["train_loss"] for entry in log_history]
    val_losses = [entry["val_loss"] for entry in log_history]

    plt.figure(figsize=(8, 5))
    plt.plot(iters, train_losses, label="train")
    plt.plot(iters, val_losses, label="val")

    # This block marks the iteration where the best checkpoint was saved.
    if best_iter is not None:
        plt.axvline(x=best_iter, color="green", linestyle="--", alpha=0.7, label=f"best ckpt (iter {best_iter})")

    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("BardGPT Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(resolve_path(config.loss_curve_file))
    plt.close()


# This helper returns the correct AMP autocast context for the selected device.
def build_autocast_context(device: torch.device, config: GPTConfig):
    if device.type == "cuda" and config.dtype == "float16":
        return lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext


# This block runs end-to-end training, evaluation, checkpointing, and loss plotting.
def main() -> None:
    config = GPTConfig()
    device_name = "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    device = torch.device(device_name)

    # This block loads the trained sentencepiece tokenizer so vocab size can be derived at runtime.
    sp = spm.SentencePieceProcessor(model_file=str(resolve_path(config.spm_model)))
    runtime_config = GPTConfig(**{**config.to_dict(), "vocab_size": sp.get_piece_size()})

    # This block opens the raw token files as memory-mapped arrays for efficient random access.
    train_data = np.memmap(resolve_path(runtime_config.train_bin), dtype=np.uint16, mode="r")
    val_data = np.memmap(resolve_path(runtime_config.val_bin), dtype=np.uint16, mode="r")

    # This block builds the model, reports its size, and optionally compiles it for speed.
    model = GPT(runtime_config).to(device)
    print(f"parameter count: {model.get_num_params():,}")
    train_model = model
    if runtime_config.compile and hasattr(torch, "compile"):
        try:
            train_model = torch.compile(model)
        except Exception as exc:
            print(f"torch.compile unavailable, continuing without compilation: {exc}")

    # This block prepares optimization and mixed-precision training utilities.
    optimizer = configure_optimizer(model, runtime_config)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and runtime_config.dtype == "float16")
    autocast_context = build_autocast_context(device, runtime_config)

    log_history = []
    best_val_loss = float("inf")
    best_iter = 0
    out_dir = resolve_path(runtime_config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # This block reads the serialized tokenizer bytes so the checkpoint is self-contained.
    spm_model_path = resolve_path(runtime_config.spm_model)
    spm_model_bytes = spm_model_path.read_bytes()

    # This loop performs gradient-based language model training with periodic evaluation.
    for iteration in range(runtime_config.max_iters):
        lr = get_learning_rate(iteration, runtime_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # This block evaluates train and validation loss on schedule and records the results.
        if iteration % runtime_config.eval_interval == 0 or iteration == runtime_config.max_iters - 1:
            losses = estimate_loss(
                train_model,
                train_data,
                val_data,
                runtime_config,
                device,
                autocast_context,
            )
            log_history.append(
                {
                    "iter": iteration,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                }
            )

            # This block saves the checkpoint whenever validation loss improves.
            improved = ""
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_iter = iteration
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "config": runtime_config.to_dict(),
                    "spm_model": runtime_config.spm_model,
                    "spm_model_bytes": spm_model_bytes,
                    "log_history": log_history,
                }
                torch.save(checkpoint, resolve_path(runtime_config.checkpoint_file))
                improved = " <- best, saved"

            print(
                f"iter {iteration:4d} | lr {lr:.6f} | "
                f"train loss {losses['train']:.4f} | val loss {losses['val']:.4f}{improved}"
            )

        # This block draws the next random batch and runs a forward/backward optimization step.
        xb, yb = get_batch(train_data, runtime_config, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            _, loss = train_model(xb, yb)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), runtime_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

    # This block saves the final loss curve with the best checkpoint iteration marked.
    save_loss_curve(log_history, runtime_config, best_iter=best_iter)
    print(f"best val loss {best_val_loss:.4f} at iter {best_iter}")
    print(f"saved checkpoint to {resolve_path(runtime_config.checkpoint_file)}")
    print(f"saved loss curve to {resolve_path(runtime_config.loss_curve_file)}")


if __name__ == "__main__":
    main()
