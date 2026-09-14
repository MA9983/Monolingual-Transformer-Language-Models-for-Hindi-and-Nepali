"""Top-level decoder-only (GPT-style) Transformer language model (Phase 2,
section 2.1). Assembled entirely from the primitives in embeddings.py,
attention.py, and block.py -- no nn.Transformer*, no HuggingFace model
classes, no pre-built attention block.

BONUS ABLATION (optional, positional embeddings removed): this is a copy of
hindi/model/gpt.py, trained on Hindi's own data/tokenizer, used only to
compare against the standard Hindi model in the optional bonus ablation
(see report/phase3_report.md's bonus section). The only actual code
difference from hindi/model/ is in attention.py, where the RoPE rotation
has been deleted entirely -- gpt.py, block.py, and embeddings.py here are
otherwise the same architecture. Kept in a separate bonus/ folder (not
inside hindi/) so it's never confused with the main graded Hindi model,
per the assignment's note that this ablation is separate from the main
graded project.
"""

import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from embeddings import InputEmbedding  # noqa: E402
from block import TransformerBlock  # noqa: E402


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, context_length,
                 embed_dropout=0.1, attn_dropout=0.1, resid_dropout=0.1, tie_embeddings=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.tie_embeddings = tie_embeddings

        self.input_embedding = InputEmbedding(vocab_size, d_model, embed_dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, context_length, attn_dropout, resid_dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Output head: linear projection from d_model to vocab_size,
        # producing next-token logits. Tying it to the input token
        # embedding matrix (output_head.weight = token_embedding.weight)
        # saves vocab_size * d_model parameters -- at vocab_size=10,000 and
        # d_model=512 that is 5.12M parameters, roughly a fifth of this
        # model's ~25M-parameter budget, freed up for actual attention/FFN
        # capacity instead of a second, separately-learned vocab-sized
        # matrix. It is also a reasonable inductive bias: the same
        # embedding space is used both to *read* a token in and to *score*
        # candidate tokens out.
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.output_head.weight = self.input_embedding.token_embedding.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None, return_attn=False):
        """token_ids: (B, T) int64 token ids, T <= context_length.

        Returns logits (B, T, vocab_size). If `targets` (B, T) is given,
        also returns the causal-LM cross-entropy loss: the logits at
        position t are compared against the token at position t+1 (the
        standard next-token-prediction objective), averaged over all
        positions.

        If `return_attn=True`, also returns a list of per-layer attention
        weight tensors (each (B, n_heads, T, T)) for attention analysis --
        adds real overhead (materializing every layer's full attention
        matrix), so only used in the dedicated analysis pass, never during
        training or routine generation.
        """
        if token_ids.size(1) > self.context_length:
            raise ValueError(
                f"sequence length {token_ids.size(1)} exceeds context_length={self.context_length}"
            )

        x = self.input_embedding(token_ids)
        attn_weights_per_layer = [] if return_attn else None
        for block in self.blocks:
            if return_attn:
                x, attn_weights = block(x, return_attn=True)
                attn_weights_per_layer.append(attn_weights)
            else:
                x = block(x)
        x = self.final_norm(x)
        logits = self.output_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Shift so logits[:, t] predicts targets[:, t] == the token at
            # position t+1 of the original sequence (the caller is
            # responsible for passing targets already shifted by one).
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-100
            )

        if return_attn:
            return logits, loss, attn_weights_per_layer
        return logits, loss

    def num_parameters(self, exclude_tied_duplicate=True):
        """Exact trainable parameter count, for the required Phase 2
        report. When embeddings are tied, output_head.weight and
        token_embedding.weight are the literal same tensor, so naively
        summing every parameter's numel() would double-count it;
        `exclude_tied_duplicate=True` (default) counts it once, matching
        what the optimizer actually treats as one set of weights.
        """
        if exclude_tied_duplicate and self.tie_embeddings:
            seen = set()
            total = 0
            for p in self.parameters():
                if p.requires_grad:
                    key = p.data_ptr()
                    if key in seen:
                        continue
                    seen.add(key)
                    total += p.numel()
            return total
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=1.0, greedy=False):
        """Autoregressive generation from a prefix. token_ids: (B, T).
        greedy=True ignores temperature and always takes the argmax
        (used for the required greedy-decoding condition); otherwise
        samples from the temperature-scaled softmax distribution.
        """
        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.context_length:]
            logits, _ = self(context)
            next_logits = logits[:, -1, :]  # (B, vocab_size), last position only
            if greedy:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids
