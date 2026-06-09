"""End-to-end tests for nnsightful.tools.forward_pass."""

from __future__ import annotations

import json
import math

import pytest

from nnsightful.tools.forward_pass import forward_pass
from nnsightful.types import ArchKind, ForwardPassData


PROMPT = "The capital of France is"


class TestForwardPassEndToEnd:
    """Integration tests that run forward_pass on a real model."""

    def test_basic_output_structure(self, model):
        data = forward_pass(model, PROMPT)
        assert isinstance(data, ForwardPassData)
        assert data.meta.version == 1
        assert data.meta.model != ""
        assert data.meta.arch.n_layers == model.num_layers

    def test_arch_detection_gpt2(self, model):
        if "gpt2" not in model.repo_id.lower():
            pytest.skip("arch detection test specific to gpt2")
        data = forward_pass(model, PROMPT)
        arch = data.meta.arch
        assert arch.kind == ArchKind.GPT2
        assert arch.has_fused_qkv is True
        assert arch.positional_kind == "absolute"
        assert arch.n_heads == arch.n_kv_heads  # MHA, not GQA
        assert arch.vocab_size == 50257
        assert arch.d_model == 768
        assert arch.d_head == 64

    def test_default_positions_is_last(self, model):
        data = forward_pass(model, PROMPT)
        S = len(model.tokenizer.encode(PROMPT))
        assert data.positions == [S - 1]
        assert len(data.layers[0].per_position.resid_pre) == 1
        assert len(data.layers[0].per_position.q) == 1

    def test_positions_all(self, model):
        data = forward_pass(model, PROMPT, positions="all")
        S = len(model.tokenizer.encode(PROMPT))
        assert data.positions == list(range(S))
        assert len(data.layers[0].per_position.resid_pre) == S
        assert len(data.layers[0].per_position.q) == S

    def test_positions_explicit(self, model):
        data = forward_pass(model, PROMPT, positions=[0, -1])
        S = len(model.tokenizer.encode(PROMPT))
        assert data.positions == [0, S - 1]
        assert len(data.layers[0].per_position.resid_pre) == 2

    def test_layer_count_matches_model(self, model):
        data = forward_pass(model, PROMPT)
        assert len(data.layers) == model.num_layers

    def test_attention_shape(self, model):
        data = forward_pass(model, PROMPT)
        S = len(data.input_tokens)
        H = data.meta.arch.n_heads
        attn = data.layers[0].attention
        assert len(attn.scores) == H
        assert len(attn.scores[0]) == S
        assert len(attn.scores[0][0]) == S
        assert len(attn.scores_masked) == H
        assert len(attn.probs) == H

    def test_attention_probs_sum_to_one(self, model):
        """Each row of probs must sum to ~1 (post-softmax) for unmasked positions."""
        data = forward_pass(model, PROMPT)
        S = len(data.input_tokens)
        attn = data.layers[0].attention
        # query position = last → all keys unmasked
        last_row = attn.probs[0][S - 1]
        assert math.isclose(sum(last_row), 1.0, abs_tol=0.01)

    def test_per_position_shapes(self, model):
        data = forward_pass(model, PROMPT)
        D = data.meta.arch.d_model
        H = data.meta.arch.n_heads
        Hk = data.meta.arch.n_kv_heads
        Dh = data.meta.arch.d_head
        pp = data.layers[0].per_position
        # |positions|=1 by default
        assert len(pp.resid_pre) == 1
        assert len(pp.resid_pre[0]) == D
        assert len(pp.q) == 1
        assert len(pp.q[0]) == H
        assert len(pp.q[0][0]) == Dh
        assert len(pp.k) == 1
        assert len(pp.k[0]) == Hk
        assert len(pp.v[0]) == Hk

    def test_topk_default(self, model):
        data = forward_pass(model, PROMPT)
        assert len(data.next_token.token_ids) == 10
        assert len(data.next_token.tokens) == 10
        assert len(data.next_token.probs) == 10
        # probs should be non-increasing
        assert all(
            data.next_token.probs[i] >= data.next_token.probs[i + 1]
            for i in range(len(data.next_token.probs) - 1)
        )

    def test_topk_custom_k(self, model):
        data = forward_pass(model, PROMPT, top_k=3)
        assert len(data.next_token.token_ids) == 3
        assert len(data.topk_per_position) == 1
        assert len(data.topk_per_position[0].token_ids) == 3

    def test_embedding_shapes_for_absolute(self, model):
        if "gpt2" not in model.repo_id.lower():
            pytest.skip("absolute pos emb assertions specific to gpt2")
        data = forward_pass(model, PROMPT)
        D = data.meta.arch.d_model
        assert data.tok_embed is not None
        assert data.pos_embed is not None
        assert data.input_embed is not None
        assert len(data.tok_embed) == 1
        assert len(data.tok_embed[0]) == D
        assert len(data.pos_embed) == 1
        assert len(data.pos_embed[0]) == D

    def test_input_tokens_match_tokenizer(self, model):
        data = forward_pass(model, PROMPT)
        expected_ids = model.tokenizer.encode(PROMPT)
        assert data.input_token_ids == list(expected_ids)
        assert len(data.input_tokens) == len(expected_ids)

    def test_json_round_trip(self, model):
        data = forward_pass(model, PROMPT)
        blob = data.model_dump_json()
        rehydrated = ForwardPassData.model_validate_json(blob)
        assert rehydrated.meta.arch.n_layers == data.meta.arch.n_layers
        assert len(rehydrated.layers) == len(data.layers)
        assert rehydrated.next_token.token_ids == data.next_token.token_ids
