"""
Stage 1 configuration: model, SAE, and direction-extraction settings.

Defaults to GPT-2-small (124M) as the primary model — the smallest
option with well-established, easy-to-load public SAEs — with
Qwen2-0.5B-Instruct as the replication model, giving a cross-
architecture check at similarly small scale.

CAVEAT worth taking seriously at this size: neither of these is a
strong instruction-follower (GPT-2-small has no instruct variant at
all; Qwen2-0.5B is usable but weak). Stage 2/3/7 all depend on the
model producing a coherent free-text self-report, and small
base/lightly-tuned models may simply not do this reliably regardless
of what's happening internally. Run a quick manual pilot — a handful
of verbalizability-elicitation prompts — before investing in the full
Stage 1 pipeline at this size, so you're not building on a model with
nothing usable to report.

NOTE: switching model family changes the SAE `sae_id` naming
convention too (GPT-2-small-res-jb uses "blocks.{L}.hook_resid_pre";
the Qwen2-0.5B SAE below uses "blocks.{L}.hook_resid_post") — update
the `sae_id` construction in `sae_features.py::load_sae` to match
whichever release you're using.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # --- Model ---
    model_name: str = "gpt2"                                      # primary model — 124M params, 12 layers, d_model=768
    replication_model_name: str = "Qwen/Qwen2-0.5B-Instruct"       # replication model — ~494M params, cross-architecture check
    device: str = "cuda"                                            # falls back to "cpu" automatically if unavailable
    dtype: str = "bfloat16"

    # --- SAE (via sae_lens; release/id naming follows SAELens conventions) ---
    sae_release: str = "gpt2-small-res-jb"                          # Joseph Bloom's GPT-2-small SAEs, residual stream
    sae_hook_point: str = "hook_resid_pre"                          # this release's valid ids use hook_resid_pre, not _post
    replication_sae_release: str = "chanind/sae-qwen2-0.5b-res"     # trained on base Qwen2-0.5B; applying to
                                                                     # the "-Instruct" variant is an approximation,
                                                                     # same caveat as any base-SAE-on-tuned-model setup
    replication_sae_hook_point: str = "hook_resid_post"             # this release uses hook_resid_post instead — check
                                                                     # SAELens' error message for any new release's valid ids
    sae_layers: List[int] = field(default_factory=lambda: [2, 4, 6, 8, 10])  # gpt2-small has 12 layers total (0-11)

    # --- Direction extraction counts (per layer, unless noted) ---
    n_sae_features_per_layer: int = 40                    # stratified by activation-frequency decile
    n_probe_attributes: int = 20                           # number of CounterFact relations used as probes
    n_random_directions_per_layer: int = 40                # fully random
    n_random_in_span_per_layer: int = 40                   # random within SAE decoder span

    # --- J-lens ---
    # NOTE: this loop is unbatched (one forward+backward pass per context per
    # (layer, token) pair, no padding/batching implemented). With 5 layers x
    # 7 tokens x n_contexts, total passes = 35 x n_contexts. 300 here keeps a
    # first end-to-end run to a reasonable time on CPU; once you've confirmed
    # the pipeline completes, raise this back toward ~2000 for your real run
    # (J-lens directions get noisier with fewer contexts to average over).
    jlens_n_contexts: int = 300
    jlens_target_tokens: List[str] = field(default_factory=lambda: [
        "Paris", "Rome", "dog", "happy", "sad", "France", "Italy",
        # extend with whatever concept vocabulary Stage 2/3 will actually test
    ])

    # --- Causal efficacy scoring (steering) ---
    steer_alphas: List[float] = field(default_factory=lambda: [4.0, 8.0, 12.0])
    causal_eval_n_contexts: int = 200

    # --- Output ---
    output_dir: str = "direction_bank"


CFG = Config()
