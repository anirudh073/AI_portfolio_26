import argparse

import torch

from config import GPTConfig, resolve_path
from model import GPT


# This helper loads the saved checkpoint and rebuilds the model with the stored config and vocab.
def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[GPT, GPTConfig, dict]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, checkpoint["vocab"]


# This helper converts a prompt string into token ids using the saved character vocabulary.
def encode_prompt(prompt: str, stoi: dict[str, int]) -> list[int]:
    missing = sorted(set(ch for ch in prompt if ch not in stoi))
    if missing:
        missing_text = ", ".join(repr(ch) for ch in missing)
        raise ValueError(f"Prompt contains characters not seen during training: {missing_text}")
    return [stoi[ch] for ch in prompt]


# This helper samples new tokens autoregressively with temperature scaling and top-k filtering.
@torch.no_grad()
def generate_text(
    model: GPT,
    prompt_ids: list[int],
    length: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> torch.Tensor:
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # This loop repeatedly predicts the next token from the most recent context window.
    for _ in range(length):
        idx_cond = idx[:, -model.config.context_len :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        # This block scales the logits by temperature before sampling.
        temp = max(temperature, 1e-5)
        logits = logits / temp

        # This block keeps only the top-k logits when top-k filtering is enabled.
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            cutoff = values[:, [-1]]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))

        # This block samples a next token and appends it to the running sequence.
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

    return idx[0]


# This block parses CLI arguments, loads the checkpoint, and prints generated text.
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained character-level GPT.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text to seed generation.")
    parser.add_argument("--length", type=int, default=500, help="Number of new characters to sample.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k cutoff for sampling.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_path(GPTConfig().checkpoint_file)
    model, config, vocab = load_checkpoint(checkpoint_path, device)

    # This block chooses the prompt, encodes it, samples new text, and decodes the result.
    prompt = args.prompt if args.prompt is not None else config.default_prompt
    prompt_ids = encode_prompt(prompt, vocab["stoi"])
    output_ids = generate_text(
        model=model,
        prompt_ids=prompt_ids,
        length=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    generated_text = "".join(vocab["itos"][token_id] for token_id in output_ids.tolist())
    print(generated_text)


if __name__ == "__main__":
    main()
