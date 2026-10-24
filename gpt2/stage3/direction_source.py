"""
Where the (pair, layer) direction comes from: the J-lens bank, or a random baseline.

The random-direction reporter answers "would ANY clamped direction, trained with the same recipe,
produce an equally good-looking reporter?" -- the negative control for the claim that a trained
reporter reads the J-lens direction. It is built to be as comparable as possible to the real one:

  * same (pair, layer) combinations as the J-lens run (a combination is only used if the bank
    has BOTH J-lens vectors for it),
  * same dimensionality, unit norm (the bipolar J-lens direction is unit norm),
  * same sign convention: +alpha means the pair's first label, -alpha its second,
  * deterministic given (seed, x, y, layer), so Stage 5 can regenerate exactly the directions
    Stage 4 trained on from split.json alone (no tensor needs to be shipped).
"""

import zlib
from typing import Optional

import torch

from decorrelation import build_bipolar_direction


def _random_unit(dim: int, tag: str, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(zlib.crc32(f"{seed}|{tag}".encode()) & 0x7FFFFFFF)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def pair_direction(vectors, x: str, y: str, layer: int, source: str = "jspace", seed: int = 0) -> Optional[torch.Tensor]:
    """Bipolar direction for pair (x, y) at `layer`, or None if this combination isn't in the bank."""
    key_x, key_y = f"jspace::{layer}::{x}", f"jspace::{layer}::{y}"
    if key_x not in vectors or key_y not in vectors:
        return None
    if source == "jspace":
        return build_bipolar_direction(vectors[key_x], vectors[key_y])
    if source == "random":
        return _random_unit(vectors[key_x].shape[0], f"{x}|{y}|{layer}", seed)
    raise ValueError(f"Unknown direction source {source!r} (expected 'jspace' or 'random').")
