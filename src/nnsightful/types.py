from abc import abstractmethod
from enum import Enum
from typing import Literal

from pydantic import BaseModel


class ToolData(BaseModel):
    """Abstract base for nnsightful tool output data."""

    @abstractmethod
    def display(self, **kwargs):
        """Display a visualization of the data."""
        ...


class LogitLensMeta(BaseModel):
    version: int = 2
    timestamp: str
    model: str


class LogitLensData(ToolData):
    meta: LogitLensMeta
    layers: list[int]
    input: list[str]  # Input tokens as strings (always dense, all tokens)
    positions: list[int] | None = None  # Computed position indices; None = all
    tracked: list[dict[str, list[float]]]  # Per-position: token -> trajectory
    topk: list[list[list[str]]]  # [layer][position] -> list of selected tokens
    entropy: list[list[float]] | None = None  # Optional: [layer][position] -> entropy

    def display(self, **kwargs):
        from nnsightful.viz import display_logit_lens

        return display_logit_lens(self, **kwargs)


class ActivationPatchingData(ToolData):
    lines: list[list[float]]  # [token][layer] probabilities
    ranks: list[list[int]]  # [token][layer] ranks
    prob_diffs: list[list[float]]  # [token][layer] prob diffs
    tokenLabels: list[str]  # Token text labels for each line

    def display(self, tokens: list[int] | None = None, return_html: bool = False, **kwargs):
        from nnsightful.viz import display_activation_patching

        data = self.model_dump()
        n = len(self.lines)
        selected = tokens if tokens is not None else list(range(min(2, n)))
        selected = [i for i in selected if i < n]
        options = kwargs.pop("options", {}) or {}
        options["selectedTokens"] = selected
        return display_activation_patching(
            data, options=options, return_html=return_html, **kwargs
        )


class ArchKind(str, Enum):
    GPT2 = "gpt2"
    LLAMA = "llama"
    GPTJ = "gptj"


class ForwardPassArch(BaseModel):
    kind: ArchKind
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_model: int
    d_head: int
    vocab_size: int
    positional_kind: Literal["absolute", "rope"]
    has_fused_qkv: bool
    tie_word_embeddings: bool


class ForwardPassMeta(BaseModel):
    version: int = 1
    model: str
    arch: ForwardPassArch


class TopKLogits(BaseModel):
    token_ids: list[int]
    tokens: list[str]
    logits: list[float]
    probs: list[float]


class AttentionPayload(BaseModel):
    # Shape: [n_heads][S][S]. Raw Q·Kᵀ / sqrt(d_head) post-RoPE, pre-mask.
    # The causal-masked variant and post-softmax probs are derived by the client
    # from `scores` — see frontend deriveAttention.ts. Halves the wire payload
    # because each was the same O(L·H·S²) size as `scores`.
    scores: list[list[list[float]]]


class LayerPositionPayload(BaseModel):
    resid_pre: list[list[float]]
    ln1_out: list[list[float]]
    q: list[list[list[float]]]
    k: list[list[list[float]]]
    v: list[list[list[float]]]
    attn_out: list[list[float]]
    resid_mid: list[list[float]]
    ln2_out: list[list[float]]
    mlp_out: list[list[float]]
    resid_post: list[list[float]]


class LayerPayload(BaseModel):
    attention: AttentionPayload
    per_position: LayerPositionPayload


class ForwardPassData(ToolData):
    meta: ForwardPassMeta
    input_token_ids: list[int]
    input_tokens: list[str]
    positions: list[int]
    tok_embed: list[list[float]] | None = None
    pos_embed: list[list[float]] | None = None
    input_embed: list[list[float]] | None = None
    layers: list[LayerPayload]
    ln_final_out: list[list[float]]
    topk_per_position: list[TopKLogits]
    next_token: TopKLogits

    def display(self, **kwargs):
        raise NotImplementedError(
            "ForwardPassData has no Python-side visualization; render via the "
            "transformer-explainer Svelte app at http://localhost:5173/transformer-explainer/"
        )
