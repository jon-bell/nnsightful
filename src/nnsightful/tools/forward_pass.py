from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import torch

from ..types import (
    ForwardPassArch,
    ForwardPassData,
    ForwardPassMeta,
)
from ._base import Tool

if TYPE_CHECKING:
    from nnterp import StandardizedTransformer


_SUPPORTED_LLAMA_TYPES = {"llama", "mistral", "qwen2"}
# Per-arch rotary_dim default (in dims of the head). None means full head_dim.
_GPTJ_DEFAULT_ROPE_THETA = 10000.0


def _detect_arch(model: "StandardizedTransformer") -> dict[str, Any]:
    """Return a plain dict describing the model architecture.

    Returns a dict (not a Pydantic model) so the result can flow through
    nnsight's trace + NDIF serializer without dragging in pydantic_core,
    which NDIF's remote worker rejects ("Module pydantic_core._pydantic_core
    is not whitelisted"). to_data_obj reconstructs the typed
    ForwardPassArch on the way out, where pydantic is fine.
    """
    cfg = model._model.config
    model_type = (getattr(cfg, "model_type", "") or "").lower()
    if model_type == "gpt2":
        kind = "gpt2"
        has_fused_qkv = True
        positional_kind: Literal["absolute", "rope"] = "absolute"
    elif model_type == "gptj":
        kind = "gptj"
        has_fused_qkv = False
        positional_kind = "rope"
    elif model_type in _SUPPORTED_LLAMA_TYPES:
        kind = "llama"
        has_fused_qkv = False
        positional_kind = "rope"
    else:
        raise NotImplementedError(
            f"forward_pass: unsupported model_type {model_type!r}; supported: gpt2, gptj, llama, mistral, qwen2"
        )

    n_heads = int(getattr(cfg, "num_attention_heads", 0) or getattr(cfg, "n_head", 0))
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads) or n_heads)
    d_model = int(getattr(cfg, "hidden_size", 0) or getattr(cfg, "n_embd", 0))
    d_head = int(getattr(cfg, "head_dim", 0) or (d_model // n_heads if n_heads else 0))
    vocab_size = int(getattr(cfg, "vocab_size", 0))
    tie_word_embeddings = bool(getattr(cfg, "tie_word_embeddings", False))

    return {
        "kind": kind,
        "n_layers": int(model.num_layers),
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "d_model": d_model,
        "d_head": d_head,
        "vocab_size": vocab_size,
        "positional_kind": positional_kind,
        "has_fused_qkv": has_fused_qkv,
        "tie_word_embeddings": tie_word_embeddings,
    }


def _resolve_positions(positions: Any, seq_len: int) -> list[int]:
    if positions == "all":
        return list(range(seq_len))
    sel: set[int] = set()
    for p in positions:
        idx = p if p >= 0 else seq_len + p
        if 0 <= idx < seq_len:
            sel.add(idx)
    return sorted(sel)


def _get_rope_theta(cfg) -> float:
    theta = getattr(cfg, "rope_theta", None)
    if theta is not None:
        return float(theta)
    rope_params = getattr(cfg, "rope_parameters", None)
    if isinstance(rope_params, dict):
        return float(rope_params.get("rope_theta", 10000.0))
    return 10000.0


class ForwardPassTool(Tool):
    """Capture per-layer activations for transformer-explainer visualization.

    For each layer, returns:
      - dense attention scores / masked scores / softmax probs over all positions
      - per-position residual streams, layernorm outputs, Q/K/V, attention output, MLP output

    Q/K/V are always returned split (GPT-2's fused c_attn is de-fused). For
    rope-based architectures (Llama), scores are computed with post-RoPE Q/K to
    match the model's true attention pattern.
    """

    def _run(
        self,
        model: "StandardizedTransformer",
        prompt: str,
        *args,
        positions: Any = (-1,),
        top_k: int = 10,
        remote: bool = False,
        backend=None,
        non_blocking: bool = False,
        raw: bool = False,
        **kwargs,
    ) -> dict[str, Any] | str:
        # _detect_arch returns a plain dict (not a Pydantic model) so the
        # trace context never closes over any pydantic objects. NDIF
        # serializes the trace frame via dill — anything referencing
        # pydantic_core triggers their module allowlist.
        arch_dict = _detect_arch(model)
        cfg = model._model.config
        L: int = arch_dict["n_layers"]
        H: int = arch_dict["n_heads"]
        Hk: int = arch_dict["n_kv_heads"]
        D: int = arch_dict["d_model"]
        Dh: int = arch_dict["d_head"]
        is_gpt2: bool = arch_dict["kind"] == "gpt2"
        is_gptj: bool = arch_dict["kind"] == "gptj"
        is_llama: bool = arch_dict["kind"] == "llama"
        is_rope: bool = arch_dict["positional_kind"] == "rope"
        is_absolute: bool = arch_dict["positional_kind"] == "absolute"
        # GPT-J: partial RoPE — only the first `rotary_dim` of each head is rotated.
        # Llama and friends rotate the full head_dim.
        rotary_dim: int = int(getattr(cfg, "rotary_dim", 0)) if is_gptj else Dh

        token_ids = model.tokenizer.encode(prompt)
        S = len(token_ids)
        sel = _resolve_positions(positions, S)
        if not sel:
            sel = [S - 1]

        input_tokens = [model.tokenizer.decode([t]) for t in token_ids]

        rope_theta = _get_rope_theta(cfg) if is_rope else None

        def _round4_list(t: torch.Tensor):
            return torch.round(t, decimals=4).tolist()

        def _format(traced: dict[str, Any]) -> dict[str, Any]:
            layers_out = []
            for layer_d in traced["layers"]:
                attn = layer_d["attention"]
                per_pos = layer_d["per_position"]
                layers_out.append(
                    {
                        "attention": {
                            "scores": _round4_list(attn["scores"]),
                            "scores_masked": _round4_list(attn["scores_masked"]),
                            "probs": _round4_list(attn["probs"]),
                        },
                        "per_position": {
                            "resid_pre": _round4_list(per_pos["resid_pre"]),
                            "ln1_out": _round4_list(per_pos["ln1_out"]),
                            "q": _round4_list(per_pos["q"]),
                            "k": _round4_list(per_pos["k"]),
                            "v": _round4_list(per_pos["v"]),
                            "attn_out": _round4_list(per_pos["attn_out"]),
                            "resid_mid": _round4_list(per_pos["resid_mid"]),
                            "ln2_out": _round4_list(per_pos["ln2_out"]),
                            "mlp_out": _round4_list(per_pos["mlp_out"]),
                            "resid_post": _round4_list(per_pos["resid_post"]),
                        },
                    }
                )

            logits_full = traced["logits"]
            probs_full = torch.softmax(logits_full.float(), dim=-1)

            topk_per_position = []
            for pos in sel:
                row_logits = logits_full[pos].float()
                row_probs = probs_full[pos]
                top_vals, top_idx = torch.topk(row_probs, k=min(top_k, row_probs.shape[-1]))
                ids = top_idx.tolist()
                topk_per_position.append(
                    {
                        "token_ids": ids,
                        "tokens": [model.tokenizer.decode([tid]) for tid in ids],
                        "logits": _round4_list(row_logits.gather(0, top_idx)),
                        "probs": _round4_list(top_vals),
                    }
                )

            next_token = topk_per_position[-1]

            return {
                "meta": {
                    "version": 1,
                    "model": model.repo_id,
                    "arch": arch_dict,
                },
                "input_token_ids": list(token_ids),
                "input_tokens": input_tokens,
                "positions": list(sel),
                "tok_embed": _round4_list(traced["tok_embed"]),
                "pos_embed": (
                    _round4_list(traced["pos_embed"])
                    if traced["pos_embed"] is not None
                    else None
                ),
                "input_embed": _round4_list(traced["input_embed"]),
                "layers": layers_out,
                "ln_final_out": _round4_list(traced["ln_final_out"]),
                "topk_per_position": topk_per_position,
                "next_token": next_token,
            }

        with torch.no_grad():
            with model.trace(prompt, remote=remote, backend=backend) as tracer:
                tok_e_full = model.embed_tokens.output[0]  # [S, d]
                tok_e_sel = tok_e_full[sel]

                if is_absolute:
                    # GPT-2: read wpe weight matrix directly (lookup table)
                    wpe_w = model._model.transformer.wpe.weight  # [n_positions, d]
                    pos_e_full = wpe_w[:S]
                    pos_e_sel_t: torch.Tensor | None = pos_e_full[sel]
                    input_e_full = tok_e_full + pos_e_full
                else:
                    pos_e_sel_t = None
                    input_e_full = tok_e_full
                input_e_sel = input_e_full[sel]

                # Precompute cos/sin for RoPE inside trace
                if is_rope:
                    # GPT-J rotates only the first `rotary_dim` of each head.
                    # Llama rotates the entire head_dim.
                    rope_dim_eff = rotary_dim if is_gptj else Dh
                    inv_freq = 1.0 / (
                        rope_theta
                        ** (
                            torch.arange(
                                0, rope_dim_eff, 2, dtype=torch.float, device=tok_e_full.device
                            )
                            / rope_dim_eff
                        )
                    )
                    pos_ids = torch.arange(S, dtype=torch.float, device=tok_e_full.device)
                    freqs = torch.outer(pos_ids, inv_freq)  # [S, rope_dim_eff/2]
                    if is_gptj:
                        # GPT-J: emb = repeat-interleave by 2 so adjacent dims
                        # share each frequency. Matches rotate_every_two() below.
                        cos = freqs.cos().repeat_interleave(2, dim=-1).to(tok_e_full.dtype)
                        sin = freqs.sin().repeat_interleave(2, dim=-1).to(tok_e_full.dtype)
                    else:
                        # Llama: emb = cat([freqs, freqs]) so first half and second
                        # half each carry the same frequency block. Matches rotate_half().
                        emb = torch.cat([freqs, freqs], dim=-1)  # [S, head_dim]
                        cos = emb.cos().to(tok_e_full.dtype)
                        sin = emb.sin().to(tok_e_full.dtype)

                layers_traced: list[dict[str, Any]] = []
                for i in range(L):
                    resid_pre = model.layers_input[i][0]  # [S, d]

                    if is_gpt2:
                        ln1_out = model.layers[i].ln_1.output[0]
                        qkv = model.layers[i].self_attn.c_attn.output[0]  # [S, 3*d]
                        q = qkv[:, :D].view(S, H, Dh)
                        k = qkv[:, D : 2 * D].view(S, H, Dh)
                        v = qkv[:, 2 * D :].view(S, H, Dh)
                        attn_out = model.layers[i].self_attn.c_proj.output[0]
                        ln2_out = model.layers[i].ln_2.output[0]
                    elif is_gptj:
                        # GPT-J: single ln_1, parallel residual, out_proj (not o_proj),
                        # split q/k/v with no GQA. No ln_2 — reuse ln_1 for the
                        # ln2_out slot so the payload shape matches GPT-2/Llama.
                        ln1_out = model.layers[i].ln_1.output[0]
                        sa = model.layers[i].self_attn
                        q = sa.q_proj.output[0].view(S, H, Dh)
                        k = sa.k_proj.output[0].view(S, H, Dh)
                        v = sa.v_proj.output[0].view(S, H, Dh)
                        attn_out = sa.out_proj.output[0]
                        ln2_out = ln1_out
                    else:
                        # Llama / Mistral / Qwen2: input_layernorm, q/k/v, o_proj,
                        # post_attention_layernorm. GQA possible (Hk < H).
                        ln1_out = model.layers[i].input_layernorm.output[0]
                        sa = model.layers[i].self_attn
                        q = sa.q_proj.output[0].view(S, H, Dh)
                        k = sa.k_proj.output[0].view(S, Hk, Dh)
                        v = sa.v_proj.output[0].view(S, Hk, Dh)
                        attn_out = sa.o_proj.output[0]
                        ln2_out = model.layers[i].post_attention_layernorm.output[0]

                    if is_rope:
                        # NDIF hosts these models with device_map="auto", so each
                        # layer can live on a different GPU. cos/sin were built
                        # once from tok_e_full.device (cuda:0); the layer's q/k
                        # may live on cuda:1+. Move on demand so the multiply
                        # actually works.
                        cos_b = cos.to(q.device).unsqueeze(1)  # [S, 1, rope_dim_eff]
                        sin_b = sin.to(q.device).unsqueeze(1)

                        if is_gptj:
                            # Partial RoPE with rotate_every_two: rotate the first
                            # `rotary_dim` of each head, pass the rest through unchanged.
                            def _rotate_every_two(x: torch.Tensor) -> torch.Tensor:
                                x1 = x[..., 0::2]
                                x2 = x[..., 1::2]
                                return torch.stack([-x2, x1], dim=-1).flatten(-2)

                            q_head = q[..., :rotary_dim]
                            q_pass = q[..., rotary_dim:]
                            k_head = k[..., :rotary_dim]
                            k_pass = k[..., rotary_dim:]
                            q_head = (q_head * cos_b) + (_rotate_every_two(q_head) * sin_b)
                            k_head = (k_head * cos_b) + (_rotate_every_two(k_head) * sin_b)
                            q_rot = torch.cat([q_head, q_pass], dim=-1)
                            k_rot = torch.cat([k_head, k_pass], dim=-1)
                        else:
                            # Llama-style full RoPE with rotate_half.
                            def _rotate_half(x: torch.Tensor) -> torch.Tensor:
                                half = x.shape[-1] // 2
                                x1 = x[..., :half]
                                x2 = x[..., half:]
                                return torch.cat([-x2, x1], dim=-1)

                            q_rot = (q * cos_b) + (_rotate_half(q) * sin_b)
                            k_rot = (k * cos_b) + (_rotate_half(k) * sin_b)
                    else:
                        q_rot, k_rot = q, k

                    # GQA: repeat K to match number of Q heads for score computation
                    if Hk != H:
                        repeat = H // Hk
                        k_for_scores = k_rot.repeat_interleave(repeat, dim=1)
                    else:
                        k_for_scores = k_rot

                    q_t = q_rot.transpose(0, 1)  # [H, S, Dh]
                    k_t = k_for_scores.transpose(0, 1)  # [H, S, Dh]
                    scores = torch.matmul(q_t, k_t.transpose(-1, -2)) / math.sqrt(Dh)

                    mask = torch.triu(
                        torch.ones(S, S, device=scores.device, dtype=torch.bool),
                        diagonal=1,
                    )
                    scores_masked_neg_inf = scores.masked_fill(mask, float("-inf"))
                    probs_attn = torch.softmax(scores_masked_neg_inf, dim=-1)
                    # For JSON: replace -inf with 0.0; UI applies causal mask from shape.
                    scores_masked_json = scores.masked_fill(mask, 0.0)

                    resid_mid = resid_pre + attn_out
                    mlp_out = model.mlps_output[i][0]
                    resid_post = model.layers_output[i][0]

                    layers_traced.append(
                        {
                            "attention": {
                                "scores": scores.float().cpu(),
                                "scores_masked": scores_masked_json.float().cpu(),
                                "probs": probs_attn.float().cpu(),
                            },
                            "per_position": {
                                "resid_pre": resid_pre[sel].float().cpu(),
                                "ln1_out": ln1_out[sel].float().cpu(),
                                "q": q[sel].float().cpu(),
                                "k": k[sel].float().cpu(),
                                "v": v[sel].float().cpu(),
                                "attn_out": attn_out[sel].float().cpu(),
                                "resid_mid": resid_mid[sel].float().cpu(),
                                "ln2_out": ln2_out[sel].float().cpu(),
                                "mlp_out": mlp_out[sel].float().cpu(),
                                "resid_post": resid_post[sel].float().cpu(),
                            },
                        }
                    )

                ln_final_out_full = model.ln_final.output[0]
                ln_final_out_sel = ln_final_out_full[sel].float().cpu()
                logits_full = model.logits[0].float().cpu()  # [S, V]

                traced = {
                    "tok_embed": tok_e_sel.float().cpu(),
                    "pos_embed": (
                        pos_e_sel_t.float().cpu() if pos_e_sel_t is not None else None
                    ),
                    "input_embed": input_e_sel.float().cpu(),
                    "ln_final_out": ln_final_out_sel,
                    "logits": logits_full,
                    "layers": layers_traced,
                }

                if raw:
                    results: Any = traced
                else:
                    results = _format(traced)

                results.save()

        if remote and non_blocking:
            return backend.job_id

        return results

    @staticmethod
    def to_data_obj(**kwargs) -> ForwardPassData:
        meta = kwargs.pop("meta")
        arch_dict = meta["arch"]
        meta["arch"] = ForwardPassArch(**arch_dict)
        kwargs["meta"] = ForwardPassMeta(**meta)
        return ForwardPassData(**kwargs)


forward_pass = ForwardPassTool()
