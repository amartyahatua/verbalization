"""
Random direction generation: fully random and random-within-SAE-
decoder-span, norm-matched to the empirical norms of the other three
direction classes (J-space, SAE, probe) so norm alone can't explain
any downstream verbalizability differences.
"""

import numpy as np
import torch
from typing import List, Dict


def random_direction(dim: int, target_norm: float) -> torch.Tensor:
    v = torch.randn(dim)
    v = v / v.norm()
    return v * target_norm


def random_in_decoder_span(W_dec: torch.Tensor, target_norm: float) -> torch.Tensor:
    """
    A random unit combination of SAE decoder columns, projected into
    residual-stream space. This gives a "random but structurally
    plausible" direction, as distinct from a fully random one.
    """
    n_features = W_dec.shape[0]
    coeffs = torch.randn(n_features)
    coeffs = coeffs / coeffs.norm()
    v = coeffs @ W_dec  # [d_model]
    v = v / v.norm()
    return v * target_norm


def build_random_directions(
    dim: int,
    W_dec: torch.Tensor,
    reference_norms: List[float],
    n_fully_random: int,
    n_in_span: int,
) -> Dict[str, torch.Tensor]:
    """Norm-match to the empirical distribution of reference direction norms."""
    norms = np.random.choice(reference_norms, size=n_fully_random + n_in_span, replace=True)

    out = {}
    for i in range(n_fully_random):
        out[f"random::{i}"] = random_direction(dim, float(norms[i]))
    for i in range(n_in_span):
        out[f"random_in_span::{i}"] = random_in_decoder_span(W_dec, float(norms[n_fully_random + i]))
    return out
