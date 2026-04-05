"""
BrainBlast™ Inference Server
Serves BardGPT and TinyGPT via HTTP so the portfolio chat UI can talk to them.

Usage:
    python serve.py

Then expose it publicly with ngrok:
    ngrok http 8787

Set the ngrok URL as INFERENCE_BASE_URL in Netlify environment variables.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
BARDGPT_DIR = ROOT / "build_an_AI" / "bardgpt"
TINYGPT_DIR = ROOT / "build_an_AI" / "tinygpt"

# ---------------------------------------------------------------------------
# Import bardgpt modules without polluting the global module namespace
# ---------------------------------------------------------------------------
for _m in ["config", "model", "generate"]:
    sys.modules.pop(_m, None)
sys.path.insert(0, str(BARDGPT_DIR))

import config as _bard_cfg
import generate as _bard_gen

bard_load_checkpoint = _bard_gen.load_checkpoint
bard_encode_prompt   = _bard_gen.encode_prompt
bard_generate_text   = _bard_gen.generate_text
BardConfig           = _bard_cfg.GPTConfig
bard_resolve_path    = _bard_cfg.resolve_path

sys.path.pop(0)

# ---------------------------------------------------------------------------
# Import tinygpt modules
# ---------------------------------------------------------------------------
for _m in ["config", "model", "generate"]:
    sys.modules.pop(_m, None)
sys.path.insert(0, str(TINYGPT_DIR))

import config as _tiny_cfg
import generate as _tiny_gen

tiny_load_checkpoint = _tiny_gen.load_checkpoint
tiny_encode_prompt   = _tiny_gen.encode_prompt
tiny_generate_text   = _tiny_gen.generate_text
TinyConfig           = _tiny_cfg.GPTConfig
tiny_resolve_path    = _tiny_cfg.resolve_path

sys.path.pop(0)
for _m in ["config", "model", "generate"]:
    sys.modules.pop(_m, None)

# ---------------------------------------------------------------------------
# Now safe to import everything else
# ---------------------------------------------------------------------------
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

PORT = 8787
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Load models at startup (slow once, fast forever)
# ---------------------------------------------------------------------------
print("Loading BardGPT...")
_bard_ckpt = bard_resolve_path(BardConfig().checkpoint_file)
bard_model, bard_config, bard_sp = bard_load_checkpoint(str(_bard_ckpt), device)
print(f"  BardGPT ready — {bard_model.get_num_params():,} params")

print("Loading TinyGPT...")
_tiny_ckpt = tiny_resolve_path(TinyConfig().checkpoint_file)
tiny_model, tiny_config, tiny_vocab = tiny_load_checkpoint(str(_tiny_ckpt), device)
print(f"  TinyGPT ready — {tiny_model.get_num_params():,} params")

print(f"\nBrainBlast™ inference server running on http://localhost:{PORT}\n")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MAX_TOKENS_CAP = 300

def _parse_params(data: dict) -> tuple[str, float, int, int]:
    prompt      = str(data.get("prompt", "")).strip() or "To be, or not to be"
    temperature = float(data.get("temperature", 0.8))
    temperature = max(0.1, min(temperature, 2.0))
    max_tokens  = int(data.get("max_tokens", 100))
    max_tokens  = max(10, min(max_tokens, MAX_TOKENS_CAP))
    top_k       = int(data.get("top_k", 40))
    return prompt, temperature, max_tokens, top_k

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/generate/bardgpt", methods=["POST", "OPTIONS"])
def generate_bardgpt():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True) or {}
    prompt, temperature, max_tokens, top_k = _parse_params(data)

    try:
        prompt_ids = bard_encode_prompt(prompt, bard_sp)
    except Exception as e:
        return jsonify({"error": f"Tokenization failed: {e}"}), 400

    output_ids = bard_generate_text(
        model=bard_model,
        prompt_ids=prompt_ids,
        length=max_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=float(data.get("repetition_penalty", 1.3)),
        device=device,
    )

    # Decode full output (prompt + generated); skip prompt tokens
    full_text      = bard_sp.decode_ids(output_ids.tolist())
    prompt_decoded = bard_sp.decode_ids(prompt_ids)
    generated      = full_text[len(prompt_decoded):]

    return jsonify({"output": generated.strip()})


@app.route("/generate/tinygpt", methods=["POST", "OPTIONS"])
def generate_tinygpt():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True) or {}
    prompt, temperature, max_tokens, top_k = _parse_params(data)

    stoi = tiny_vocab["stoi"]
    itos = tiny_vocab["itos"]

    try:
        prompt_ids = tiny_encode_prompt(prompt, stoi)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    output_ids = tiny_generate_text(
        model=tiny_model,
        prompt_ids=prompt_ids,
        length=max_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )

    all_tokens = output_ids.tolist()
    generated  = "".join(itos[t] for t in all_tokens[len(prompt_ids):])

    return jsonify({"output": generated.strip()})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models": ["bardgpt", "tinygpt"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
