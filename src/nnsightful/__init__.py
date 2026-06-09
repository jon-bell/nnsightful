"""nnsightful: a higher-level extension of NNsight for mechanistic interpretability."""

from nnsightful.tools._base import Tool
from nnsightful.tools._utils import IndexSpec, resolve_indices
from nnsightful.tools.activation_patching import activation_patching
from nnsightful.tools.forward_pass import forward_pass
from nnsightful.tools.logit_lens import logit_lens
from nnsightful.types import (
    ActivationPatchingData,
    ArchKind,
    AttentionPayload,
    ForwardPassArch,
    ForwardPassData,
    ForwardPassMeta,
    LayerPayload,
    LayerPositionPayload,
    LogitLensData,
    LogitLensMeta,
    ToolData,
    TopKLogits,
)
from nnsightful.viz import display_activation_patching, display_line_plot, display_logit_lens

__all__ = [
    "activation_patching",
    "forward_pass",
    "logit_lens",
    "Tool",
    "ToolData",
    "IndexSpec",
    "resolve_indices",
    "ActivationPatchingData",
    "LogitLensData",
    "LogitLensMeta",
    "ArchKind",
    "AttentionPayload",
    "ForwardPassArch",
    "ForwardPassData",
    "ForwardPassMeta",
    "LayerPayload",
    "LayerPositionPayload",
    "TopKLogits",
    "display_activation_patching",
    "display_line_plot",
    "display_logit_lens",
]
