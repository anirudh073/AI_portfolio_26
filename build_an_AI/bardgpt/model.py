import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


# This class implements causal multi-head self-attention for autoregressive decoding.
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # This projection creates queries, keys, and values in a single matrix multiply.
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)

        # This projection mixes the attended heads back into the model width.
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        # This dropout regularizes attention probabilities and residual projections.
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # This causal mask prevents tokens from attending to future positions.
        mask = torch.tril(torch.ones(config.context_len, config.context_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_len, config.context_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape

        # This block projects the input into query, key, and value tensors and splits them into heads.
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(channels, dim=2)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # This block computes scaled dot-product attention with a causal mask.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # This block combines the attended values and projects the result back to model width.
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        y = self.out_proj(y)
        return self.resid_dropout(y)


# This class implements the feed-forward sublayer used inside each transformer block.
class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        # This MLP expands the hidden size, applies GELU, and projects back down.
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# This class combines layer norm, causal attention, and feed-forward layers with residual connections.
class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        # This layer norm prepares activations for the attention sublayer.
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # This attention module mixes information from earlier positions in the sequence.
        self.attn = CausalSelfAttention(config)

        # This layer norm prepares activations for the MLP sublayer.
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # This feed-forward network adds non-linear channel mixing.
        self.ffwd = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This residual path applies pre-norm attention.
        x = x + self.attn(self.ln_1(x))

        # This residual path applies pre-norm feed-forward processing.
        x = x + self.ffwd(self.ln_2(x))
        return x


# This class implements the full GPT language model for character-level next-token prediction.
class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be set before constructing GPT")

        self.config = config

        # This embedding table maps character ids into continuous token vectors.
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)

        # This embedding table encodes absolute positions within the context window.
        self.position_embedding_table = nn.Embedding(config.context_len, config.n_embd)

        # This dropout regularizes the summed token and position embeddings.
        self.dropout = nn.Dropout(config.dropout)

        # This stack of transformer blocks performs iterative sequence processing.
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        # This final layer norm stabilizes the decoder output before the logits projection.
        self.ln_f = nn.LayerNorm(config.n_embd)

        # This linear head projects hidden states into vocabulary logits.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # This weight tying shares parameters between the input embedding and output projection.
        self.lm_head.weight = self.token_embedding_table.weight

        # This initialization matches the small-GPT style used for stable training from scratch.
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        # This block initializes linear and embedding weights with a small normal distribution.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        # This helper returns the total number of trainable parameters.
        return sum(param.numel() for param in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = idx.shape
        if seq_len > self.config.context_len:
            raise ValueError("Input sequence length exceeds configured context length")

        # This block creates token and position embeddings and combines them into the model input.
        token_embeddings = self.token_embedding_table(idx)
        positions = torch.arange(seq_len, device=idx.device)
        position_embeddings = self.position_embedding_table(positions)
        x = self.dropout(token_embeddings + position_embeddings)

        # This block applies the full transformer stack followed by the final layer norm.
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # This block projects hidden states into vocabulary logits for next-character prediction.
        logits = self.lm_head(x)

        # This block computes the language modeling loss when supervision targets are provided.
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
