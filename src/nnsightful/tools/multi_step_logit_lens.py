from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from ..types import LogitLensData, LogitLensMeta, MultiStepLogitLensData
from ._base import Tool

if TYPE_CHECKING:
    from nnterp import StandardizedTransformer


# Format logic mirrors logit_lens._run's inner `format` closure. Duplicated
# rather than extracted to avoid any risk of changing the existing tool's
# output. Keep these in sync if the single-step formatter ever changes.
def _format_step(
    logits: torch.Tensor,
    model: "StandardizedTransformer",
    input_token_ids: list[int],
    top_k: int = 5,
    include_entropy: bool = True,
) -> dict[str, Any]:
    input_tokens = [str(model.tokenizer.decode(tid)) for tid in input_token_ids]
    layers = list(range(model.num_layers))
    positions = list(range(len(input_tokens)))

    if include_entropy:
        log_p = torch.nn.functional.log_softmax(logits, dim=-1)
        p = log_p.exp()
        entropy = torch.round(-(p * log_p).sum(dim=-1), decimals=3).tolist()
    else:
        entropy = None

    probs = torch.nn.functional.softmax(logits, dim=-1)
    logits.to("cpu")

    _, top_indices = torch.topk(probs, k=top_k, dim=-1)

    topks = [
        [
            model.tokenizer.batch_decode(torch.tensor(pos).unsqueeze(dim=1))
            for pos in layer
        ]
        for layer in top_indices.tolist()
    ]

    unique_indices = [
        torch.unique(top_indices[:, pi, :].flatten(), sorted=False).tolist()
        for pi in range(top_indices.shape[1])
    ]
    probs_perm = probs.permute(1, 2, 0)
    trajectories = [
        {
            model.tokenizer.decode(token): torch.round(
                probs_perm[pos_idx][token], decimals=3
            ).tolist()
            for token in pos
        }
        for pos_idx, pos in enumerate(unique_indices)
    ]

    return {
        "meta": {"version": 2, "timestamp": "3h", "model": model.repo_id},
        "layers": layers,
        "input": input_tokens,
        "tracked": trajectories,
        "topk": topks,
        "entropy": entropy,
        "positions": positions,
    }


class MultiStepLogitLensTool(Tool):
    """Multi-step logit lens: run the lens N times, chaining argmax predictions.

    The loop is on token ids, not decoded strings, so we never re-tokenize an
    appended string -- BPE tokenization is not context-free and that would
    drift token boundaries.

    Always blocking per step. If `remote=True`, each step still polls to
    completion before the next step starts; callers wanting fire-and-forget
    should run the whole loop on a background thread.
    """

    def _run(
        self,
        model: "StandardizedTransformer",
        prompt: str,
        *args,
        n_steps: int = 1,
        top_k: int = 5,
        include_entropy: bool = True,
        remote: bool = False,
        backend: Optional[Any] = None,
        non_blocking: bool = False,
        on_step: Optional[Callable[[int, LogitLensData], None]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        assert n_steps >= 1, "n_steps must be >= 1"

        ids: list[int] = list(model.tokenizer.encode(prompt))

        steps: list[dict[str, Any]] = []
        generated_token_ids: list[int] = []
        generated_tokens: list[str] = []

        for i in range(n_steps):
            with torch.no_grad():
                with model.trace(ids, remote=remote, backend=backend):
                    all_logits = []
                    for l_idx in range(model.num_layers):
                        all_logits.append(
                            model.project_on_vocab(model.layers_output[l_idx])
                        )
                    all_logits = torch.cat(all_logits, dim=0)

                    # Argmax on final layer at the last position -- chosen on
                    # the tensor, not by decoding top-k strings.
                    next_id_proxy = all_logits[-1, -1, :].argmax(dim=-1).save()
                    saved_logits = all_logits.save()

            next_id = int(next_id_proxy.item())
            step_dict = _format_step(
                saved_logits,
                model,
                ids,
                top_k=top_k,
                include_entropy=include_entropy,
            )

            steps.append(step_dict)
            generated_token_ids.append(next_id)
            generated_tokens.append(str(model.tokenizer.decode(next_id)))

            if on_step is not None:
                meta = LogitLensMeta(**step_dict["meta"])
                on_step(
                    i,
                    LogitLensData(
                        meta=meta,
                        layers=step_dict["layers"],
                        input=step_dict["input"],
                        positions=step_dict["positions"],
                        tracked=step_dict["tracked"],
                        topk=step_dict["topk"],
                        entropy=step_dict["entropy"],
                    ),
                )

            ids.append(next_id)

        return {
            "meta": {"version": 2, "timestamp": "3h", "model": model.repo_id},
            "steps": steps,
            "generated_token_ids": generated_token_ids,
            "generated_tokens": generated_tokens,
        }

    @staticmethod
    def to_data_obj(**kwargs) -> MultiStepLogitLensData:
        meta_dict = kwargs["meta"]
        meta = LogitLensMeta(**meta_dict)

        step_objs: list[LogitLensData] = []
        for step in kwargs["steps"]:
            step_meta = LogitLensMeta(**step["meta"])
            step_objs.append(
                LogitLensData(
                    meta=step_meta,
                    layers=step["layers"],
                    input=step["input"],
                    positions=step.get("positions"),
                    tracked=step["tracked"],
                    topk=step["topk"],
                    entropy=step.get("entropy"),
                )
            )

        return MultiStepLogitLensData(
            meta=meta,
            steps=step_objs,
            generated_token_ids=kwargs["generated_token_ids"],
            generated_tokens=kwargs["generated_tokens"],
        )


multi_step_logit_lens = MultiStepLogitLensTool()
