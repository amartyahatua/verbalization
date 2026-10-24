"""
Shared protocol + statistics helpers (camera-ready revision).

Everything here is deliberately dependency-light (stdlib only, torch
imported lazily) so any stage can import it from stage1/ without
pulling in model code.

Why this module exists: the reviewers could not reconstruct several
reported percentages as k/n for any integer k (e.g. 41.7% at n=8),
because the scoring protocol -- generations per item, decoding
settings, which reports were excluded -- lived only in CLI defaults and
the reader's head. Every stage now (a) records exact k/n for every cell
it reports and (b) writes a `protocol.json` sidecar next to its results
with the decoding settings, prompts, seed, model and code version.
"""

import json
import math
import os
import random
import subprocess
from typing import Callable, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: Optional[int]) -> None:
    """Seed python, numpy (if present) and torch (if present, incl. CUDA). No-op for None."""
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit() -> Optional[str]:
    """Short hash of the code that produced a result (None outside a git checkout)."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Protocol sidecar
# ---------------------------------------------------------------------------

def decoding_settings(n_trials: int, max_new_tokens: int, temperature: float = 0.8,
                      do_sample: bool = True, use_chat_template: bool = True) -> Dict:
    """The generation settings elicitation.elicit_reports / mechanism._generate actually use."""
    return {
        "generations_per_item": n_trials,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "use_chat_template": use_chat_template,
        "scoring": "case-insensitive whole-word match of the target label; a report scores 0 if it is a "
                   "degenerate repetition loop or names >=2 distinct candidate labels (shotgun listing)",
    }


def write_protocol(out_dir: str, stage: str, **fields) -> str:
    """Write `<out_dir>/protocol.json` (sidecar -- several stages' result files are bare lists, so the
    protocol cannot be embedded without breaking Stage 6's loaders). Returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    record = {"stage": stage, "code_version": git_commit(), **fields}
    path = os.path.join(out_dir, "protocol.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return path


def model_protocol(cfg, device: str) -> Dict:
    return {"model_name": cfg.model_name, "dtype": cfg.dtype, "device": device,
            "elicitation_prompt": cfg.stage2_prompt}


# ---------------------------------------------------------------------------
# Counts and intervals
# ---------------------------------------------------------------------------

def count_cell(k: int, n: int) -> Dict:
    """A reported rate always travels with its numerator/denominator and a Wilson interval."""
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "wilson95": [lo, hi]}


def wilson_interval(k: int, n: int, z: float = 1.959964):
    """Wilson score interval for a binomial proportion. (None, None) when n == 0."""
    if n <= 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, ci: float = 0.95, seed: int = 0,
                 stat: Optional[Callable[[Sequence[float]], float]] = None) -> Dict:
    """
    Percentile bootstrap over `values` (resampled with replacement). Default statistic is the mean.
    What is being resampled is the CALLER's choice and should be stated alongside the interval
    (generations within a cell, entries across pairs/layers, or seeds) -- they answer different
    questions: sampling noise, coverage of the item set, and training variance respectively.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "n_boot": n_boot}
    stat = stat or (lambda xs: sum(xs) / len(xs))
    rng = random.Random(seed)
    n = len(vals)
    boots = sorted(stat([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo_i = int(((1 - ci) / 2) * n_boot)
    hi_i = min(n_boot - 1, int((1 - (1 - ci) / 2) * n_boot))
    return {"mean": stat(vals), "ci_low": boots[lo_i], "ci_high": boots[hi_i], "n": n, "n_boot": n_boot}


# ---------------------------------------------------------------------------
# Activation scale (alpha is only meaningful relative to the layer's residual norm)
# ---------------------------------------------------------------------------

def load_activation_norms(path: Optional[str]) -> Dict[int, float]:
    """Load `activation_norms.json` written by stage1/activation_scale.py ({layer: mean ||h||})."""
    if not path or not os.path.exists(path):
        return {}
    raw = json.load(open(path))
    return {int(k): float(v["mean_norm"]) for k, v in raw["per_layer"].items()}


def alpha_over_norm(alpha: float, layer: int, norms: Dict[int, float]) -> Optional[float]:
    """|alpha| / (typical residual-stream norm at `layer`), or None if that layer wasn't measured."""
    n = norms.get(layer)
    return (abs(alpha) / n) if n else None
