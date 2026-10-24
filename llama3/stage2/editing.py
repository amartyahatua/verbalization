"""
Activation editing for Stage 2: set a direction to a known value by
projecting out the activation's current component along that
direction, then adding back alpha * direction.

    h_new = h - (h . d_hat) * d_hat + alpha * d_hat
          = h + (alpha - h . d_hat) * d_hat

Applied at every sequence position, not just the last one -- during
generation you want the concept "clamped" throughout, not just on one
token. Because register_forward_hook fires on every forward call,
this also applies correctly to each newly generated token under
KV-cached generation (each post-prompt call only contains the new
token's position, which still gets edited).
"""

import torch

# --- make stage1/ importable from here, so this can still be run/imported
# directly (e.g. `python stage2_elicit.py`) without needing `-m` invocation
# or a manually-set PYTHONPATH ---
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))

from model_utils import get_transformer_layers


def set_direction_value_hook(direction: torch.Tensor, alpha: float):
    """direction: shape [d_model], any norm (will be unit-normalized here)."""
    d_hat = direction / direction.norm()

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        d = d_hat.to(h.device, h.dtype)
        coeff = torch.einsum("bsd,d->bs", h, d)              # [batch, seq] current component along d
        h_new = h + (alpha - coeff).unsqueeze(-1) * d         # set that component to alpha
        if isinstance(out, tuple):
            return (h_new,) + out[1:]
        return h_new

    return hook


def apply_direction_editing(model, layer: int, direction: torch.Tensor, alpha: float):
    """Registers the editing hook; caller is responsible for calling .remove() on the returned handle."""
    return get_transformer_layers(model)[layer].register_forward_hook(
        set_direction_value_hook(direction, alpha)
    )
