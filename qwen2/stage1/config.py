"""
Stage 1 configuration: model, SAE, and direction-extraction settings.

Defaults to Qwen2-0.5B-Instruct as the primary model, with GPT-2-small
as the replication model (cross-architecture check, and this project's
original primary, so results are directly comparable to everything
collected before this switch).

WHY Qwen2-0.5B-Instruct over GPT-2-small: instruction-tuned, so it
engages with the elicitation prompt as a question rather than treating
it as an arbitrary text continuation -- confirmed in this project's
own Stage 2 alpha-sweep pilot, where Qwen2-0.5B-Instruct showed real
qualitative engagement (explicit answers, confident commitment at
moderate alpha) that GPT-2-small never did across dozens of trials.
Still small enough (0.5B) to iterate quickly.

use_chat_template=True (see model_utils.py::format_prompt_for_chat)
is applied automatically wherever an elicitation-style prompt is
generated or trained on (Stages 2-7) -- this model is instruction-
tuned and expects properly formatted chat turns, not a raw
completion. Stage 1's own context-processing (J-lens extraction,
CounterFact probes, background-context steering) deliberately stays
plain-text regardless of model, since that's not a conversational
interaction.

NOTE on stage2_alpha: 8.0 (this project's GPT-2-small value) is
almost certainly wrong for this model. This project's own alpha-sweep
pilot on Qwen2-0.5B-Instruct found real engagement around alpha 3-4
and gibberish collapse by alpha 8 -- set to 3.0 below as a grounded
starting point, not a guess, but this was tested at one layer with
J-space directions specifically; re-run alpha_sweep.py once you have
a real bank before trusting it for a full run.

NOTE on sae_release: I could not fully verify this specific SAELens
release string against a live install (no network access in my
environment) -- the underlying model/hook-point ARE confirmed
(chanind/sae-qwen2-0.5b-res, hook_resid_post, both verified via
search and already used successfully in this project's Stage 2
alpha-sweep --recompute runs). If SAE.from_pretrained() fails on this
string, SAELens's own error message will list the valid ids for
whatever's actually in that repo -- same pattern that caught and
fixed the GPT-2 hook_resid_pre/_post bug earlier in this project.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # --- Model ---
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"          # primary model — ~494M params, 24 layers, d_model=896
    replication_model_name: str = "gpt2"                   # replication model — this project's original primary
    device: str = "cuda"                                    # falls back to "cpu" automatically if unavailable
    dtype: str = "bfloat16"
    use_chat_template: bool = True                          # applied automatically in Stages 2-7's elicitation
                                                             # prompts; NOT applied to Stage 1's context-processing

    # --- SAE (via sae_lens; release/id naming follows SAELens conventions) ---
    sae_release: str = "chanind/sae-qwen2-0.5b-res"         # trained on base Qwen2-0.5B; applying to the
                                                             # "-Instruct" variant is an approximation, same
                                                             # caveat as any base-SAE-on-tuned-model setup
    sae_hook_point: str = "hook_resid_post"
    replication_sae_release: str = "gpt2-small-res-jb"      # Joseph Bloom's GPT-2-small SAEs, residual stream
    replication_sae_hook_point: str = "hook_resid_pre"      # this release's valid ids use hook_resid_pre, not _post
    sae_layers: List[int] = field(default_factory=lambda: [4, 8, 12, 16, 20])  # qwen2-0.5b has 24 layers total (0-23)

    # --- Direction extraction counts (per layer, unless noted) ---
    n_sae_features_per_layer: int = 40                    # stratified by activation-frequency decile
    n_probe_attributes: int = 20                           # number of CounterFact relations used as probes
    n_random_directions_per_layer: int = 40                # fully random
    n_random_in_span_per_layer: int = 40                   # random within SAE decoder span

    # --- J-lens ---
    # NOTE: this loop is unbatched (one forward+backward pass per context per
    # (layer, token) pair, no padding/batching implemented). With 5 layers x
    # 7 tokens x n_contexts, total passes = 35 x n_contexts. 300 here keeps a
    # first end-to-end run to a reasonable time; once you've confirmed the
    # pipeline completes, raise this back toward ~2000 for your real run
    # (J-lens directions get noisier with fewer contexts to average over).
    jlens_n_contexts: int = 300
    jlens_target_tokens: List[str] = field(default_factory=lambda: [
        "Paris", "Rome", "dog", "cat", "happy", "sad", "France", "Italy",
        # "cat" added specifically so DEFAULT_PAIRS (stage3_run.py) can use 8 distinct
        # labels across 4 non-overlapping pairs -- the original 7-token list forced
        # "happy" to repeat across two pairs (dog/happy AND sad/happy), giving it 2x
        # the training-gradient frequency of every other label during Stage 4, which
        # a real training run showed causes the adapter to mode-collapse onto "happy"
        # as a default output regardless of the actual clamped direction.
    ])

    # --- Causal efficacy scoring (steering) ---
    steer_alphas: List[float] = field(default_factory=lambda: [4.0, 8.0, 12.0])
    causal_eval_n_contexts: int = 200

    # --- Output ---
    output_dir: str = "direction_bank"

    # --- Stage 2: verbalizability elicitation ---
    stage2_prompt: str = "Question: What concept is active in your thinking right now?\nAnswer:"
    stage2_alpha: float = 6.0                 # grounded in this project's own Qwen2 alpha-sweep pilot -- see
                                               # module docstring. Re-verify with alpha_sweep.py before trusting.
    stage2_n_trials: int = 5                  # generations per direction, for a match RATE not a single 0/1
    stage2_max_new_tokens: int = 30
    stage2_temperature: float = 0.8
    stage2_covariate_n_contexts: int = 50     # contexts used for jacobian_magnitude estimation (cheaper than Stage 1's)
    stage2_output_dir: str = "stage2_results"


CFG = Config()
