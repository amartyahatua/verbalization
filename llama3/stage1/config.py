"""
Stage 1 configuration: model, SAE, and direction-extraction settings.

Defaults to Llama-3.1-8B-Instruct as the primary model, extending this
project's Qwen2-0.5B-Instruct / GPT-2-small pipeline to a genuinely
larger, instruction-tuned model. The entire codebase past this file is
architecture-agnostic already (model_utils.py::get_transformer_layers
handles the Llama-family `model.model.layers` path, and Stage 4/7's
LoRA target_modules already default to the q_proj/k_proj/v_proj/o_proj
naming Llama shares with Qwen2) -- this config is the only file that
should need model-specific values for this extension.

WHY Llama-3.1-8B-Instruct (moved from Llama-3-8B-Instruct -- see NOTE
on sae_release below for why): it's ~16x larger than Qwen2-0.5B-
Instruct (8B vs 0.5B params, 32 vs 24 layers, d_model=4096 vs 896)
while staying open-weight and runnable on a single high-memory GPU,
which makes it a direct test of whether the original project's
central finding -- correlated training producing a convincing but
non-causal verbalization channel (DFI = -1.0 under decorrelation) --
is a small-model artifact or persists at real scale. It's
instruction-tuned, so it inherits the same "engages with the
elicitation prompt as a question" property that motivated the
original move away from GPT-2-small. Architecturally Llama-3.1-8B is
identical to Llama-3-8B (same 32 layers, d_model=4096, tokenizer) --
Meta's 3.1 release only changed pretraining data/context length -- so
none of the reasoning above changes by picking 3.1 over 3.0.

use_chat_template=True (see model_utils.py::format_prompt_for_chat) is
applied automatically wherever an elicitation-style prompt is
generated or trained on (Stages 2-7) -- Llama-3.1-Instruct ships a
chat template (tokenizer.apply_chat_template resolves to its
<|start_header_id|>/<|end_header_id|> turn format, same as 3.0) so
this needs no special handling. Stage 1's own context-processing
(J-lens extraction, CounterFact probes, background-context steering)
deliberately stays plain-text regardless of model, same as before.

NOTE on stage2_alpha and steer_alphas: UNVERIFIED PLACEHOLDERS, not a
grounded value the way the Qwen2 config's 6.0 was. That number was
only trustworthy because it came out of a real alpha_sweep.py pilot
run on Qwen2 itself. J-lens/probe/random directions are unit-
normalized (see jlens.py), so the alpha needed to actually shift
behavior without collapsing into gibberish tracks the model's typical
residual-stream activation norm at that layer -- and that norm does
not carry over from a 0.5B/d_model=896 model to an 8B/d_model=4096
one. The values below are starting points for a sweep, scaled up
roughly with d_model as a first guess, nothing more. Before trusting
ANY downstream stage, run:
    python alpha_sweep.py --layer 19 --alphas 4 8 12 16 24 32 48
and read the actual generations, exactly the workflow the original
project used to move from GPT-2's 8.0 to Qwen2's 6.0 -- then update
stage2_alpha (and, if causal-efficacy tagging looks off in Stage 1,
steer_alphas) before running anything for real.

NOTE on sae_release: this was originally set to Llama Scope's
"llama_scope_lxr_32x" (OpenMOSS/fnlp, trained on base Llama-3.1-8B),
which turned out to be broken in practice: SAE.from_pretrained() kept
raising ValueError for every configured layer. Checked against a live
SAELens 6.49.1 install (sae_lens.loading.pretrained_saes_directory) --
the release exists, but its sae_id scheme is "l{layer}r_32x"
(e.g. "l6r_32x"), not the "blocks.{layer}.{hook_point}" scheme
sae_features.py's load_sae() was constructing. See sae_features.py's
docstring for the id-scheme fix.

That still left the base-vs-Instruct approximation problem this note
originally flagged (llama_scope is trained on the BASE model; applying
it to -Instruct activations is only an approximation). Checking the
same live directory for an SAE trained directly on an Instruct
checkpoint surfaced two options:
  - "llama-3-8b-it-res-jh" (Juliushanhanhan/llama-3-8b-it-res) is
    trained on the *exact* meta-llama/Meta-Llama-3-8B-Instruct model
    this config used to point at -- but it ships exactly ONE SAE, at
    layer 25, which can't support this pipeline's 5-7-layer
    depth-spread design.
  - "llama-3.1-8b-instruct-andyrdt" (andyrdt/saes-llama-3.1-8b-instruct)
    is trained on meta-llama/Llama-3.1-8B-Instruct and ships 7 SAEs at
    evenly-spaced layers [3, 7, 11, 15, 19, 23, 27] (sae_id scheme:
    "resid_post_layer_{layer}_trainer_1"). This is why model_name
    moved from Llama-3-8B-Instruct to Llama-3.1-8B-Instruct: it's the
    only option with both (a) enough layer coverage for this design
    and (b) an SAE trained on the Instruct checkpoint directly, which
    is strictly better evidence than the base-model-SAE approximation
    every other stage of this project (including Qwen2) has had to
    accept. sae_layers below was updated to exactly the 7 layers this
    release covers, rather than an arbitrary subset, since sae_lens
    will otherwise just fail to load whatever isn't in that list.

Re-verify both of these against your installed sae_lens version before
a real run -- pretrained-SAE directories get reorganized across
releases (this is what broke the original hook_point-based scheme):
    python -c "
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    d = get_pretrained_saes_directory()
    info = d['llama-3.1-8b-instruct-andyrdt']
    print(info.repo_id, info.saes_map)
    "
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Config:
    # --- Model ---
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"    # primary model — ~8B params, 32 layers, d_model=4096
                                                             # (moved from Meta-Llama-3-8B-Instruct — see NOTE on
                                                             # sae_release in the module docstring for why)
    replication_model_name: str = "Qwen/Qwen2-0.5B-Instruct"  # replication model — this project's prior primary,
                                                                # now the cross-scale check instead of cross-architecture
    device: str = "cuda"                                    # falls back to "cpu" automatically if unavailable
                                                              # (cpu is impractical at this size — expect to need cuda)
    dtype: str = "bfloat16"
    use_chat_template: bool = True                          # applied automatically in Stages 2-7's elicitation
                                                             # prompts; NOT applied to Stage 1's context-processing

    # --- SAE (via sae_lens; release/id naming follows SAELens conventions) ---
    sae_release: str = "llama-3.1-8b-instruct-andyrdt"      # andyrdt/saes-llama-3.1-8b-instruct — trained directly
                                                             # on the Instruct checkpoint (not a base-model
                                                             # approximation). VERIFIED against a live sae_lens
                                                             # 6.49.1 install — see module docstring.
    sae_id_template: str = "resid_post_layer_{layer}_trainer_1"  # this release's sae_id scheme; NOT the
                                                             # "blocks.{layer}.{hook_point}" scheme the old
                                                             # llama_scope release used (see sae_features.py)
    replication_sae_release: str = "chanind/sae-qwen2-0.5b-res"  # STALE/BROKEN, currently unused: this key does
                                                             # not exist anywhere in sae_lens 6.49.1's pretrained
                                                             # directory (checked by repo_id and by model name —
                                                             # no Qwen2-0.5B SAE release currently exists there at
                                                             # all). Harmless today since nothing in the codebase
                                                             # reads this field yet, but don't wire up a real
                                                             # replication run against it without re-checking
                                                             # sae_lens's directory for a working release first.
    replication_sae_id_template: str = "blocks.{layer}.hook_resid_post"  # same caveat as above — unverified/unused
    sae_layers: List[int] = field(default_factory=lambda: [3, 7, 11, 15, 19, 23, 27])  # exactly the 7 layers
                                                             # llama-3.1-8b-instruct-andyrdt ships SAEs for (of 32
                                                             # layers total, 0-31) — ~9/22/34/47/59/72/84%-depth
                                                             # spread, a finer-grained version of the Qwen2 config's
                                                             # [4,8,12,16,20]. Picking any layer NOT in this list
                                                             # will simply fail to load for this release.

    # --- Direction extraction counts (per layer, unless noted) ---
    # Unchanged from the Qwen2 config — these govern dataset size / stratification
    # policy, not anything that scales with model architecture.
    n_sae_features_per_layer: int = 40                    # stratified by activation-frequency decile
    n_probe_attributes: int = 20                           # number of CounterFact relations used as probes
    n_random_directions_per_layer: int = 40                # fully random
    n_random_in_span_per_layer: int = 40                   # random within SAE decoder span

    # --- J-lens ---
    # NOTE: this loop is unbatched (one forward+backward pass per context per
    # (layer, token) pair, no padding/batching implemented). With 7 layers x
    # 8 tokens x n_contexts, total passes = 56 x n_contexts (up from 40 x
    # n_contexts under the old 5-layer sae_layers). At 8B params each pass is
    # far more expensive than Qwen2's 0.5B — budget accordingly (a smaller
    # jlens_n_contexts for your first end-to-end pass is reasonable here even
    # beyond the usual dry_run/full-run split; see running_command).
    jlens_n_contexts: int = 300
    jlens_target_tokens: List[str] = field(default_factory=lambda: [
        "Paris", "Rome", "dog", "cat", "happy", "sad", "France", "Italy",
        # Same 8-token, 4-non-overlapping-pair set as the Qwen2 config, for the same
        # reason: a repeated label (e.g. "happy" in both (dog,happy) and (sad,happy))
        # doubles its training-gradient frequency in Stage 4 and risks mode collapse.
        # Llama-3.x's tokenizer (unchanged between 3.0 and 3.1 -- same 128k-vocab
        # tokenizer.json) differs from Qwen2's, so double-check each of these
        # still encodes with a meaningful first token (jlens.py takes
        # tokenizer.encode(tok, add_special_tokens=False)[0]) before a real run.
    ])

    # --- Causal efficacy scoring (steering) ---
    steer_alphas: List[float] = field(default_factory=lambda: [8.0, 16.0, 24.0])  # UNVERIFIED — see module
                                                             # docstring; scaled up from Qwen2's [4,8,12] as a
                                                             # starting guess for d_model=4096's larger activation norms
    causal_eval_n_contexts: int = 200

    # --- Output ---
    output_dir: str = "direction_bank"

    # --- Stage 2: verbalizability elicitation ---
    stage2_prompt: str = "Question: What concept is active in your thinking right now?\nAnswer:"
    stage2_alphas: Dict[int, float] = field(default_factory=lambda: {
        3:  1.5,   # layers 3 & 7: coherence ceiling only -- no real target-concept content ever
        7:  1.5,   # appeared at any tested alpha (0.5-2 fine-grained sweeps on both layers) before
                   # collapse. Picked the highest still-coherent value; this is likely a genuine
                   # shallow-layer finding (too little abstraction this early to verbalize a
                   # literal concept), not a mistuned alpha -- see alpha_sweep.py output from
                   # 2026-08 calibration pass before assuming it's fixable with more sweeping.
        19: 6.0,   # clean, non-collapsed target mentions confirmed at alpha=6 (top of the range
                   # swept -- true ceiling may be a bit higher, untested above 6).
        23: 6.0,   # same sweet spot as 19/27, confirmed clean through alpha=7 with no collapse.
        27: 6.5,   # sweet spot 6-7; collapse (token-repetition / word-gluing) confirmed starting
                   # at alpha=8 -- see is_degenerate_repetition in scoring.py, which the earlier
                   # alpha=12 default was silently defeating (glued repeats like "catcatcatcat"
                   # have no whitespace, so a naive whole-word repetition check never saw them).
    })                 # Replaces the old flat `stage2_alpha` float -- one global alpha was wrong
                       # for every layer in different directions (too strong for 3/7, and the
                       # 8-24 causal-efficacy steer_alphas below are probably ALSO in the
                       # collapse regime at these layers -- worth a similar sweep-and-recheck of
                       # tag_causal_efficacy's output before trusting metadata.json's
                       # causal_efficacy values as final). CFG.sae_layers changes -> this dict
                       # needs a matching entry, or stage2_elicit.py raises KeyError rather than
                       # silently defaulting -- intentional, so a new layer never runs unsweept.
    stage2_n_trials: int = 5                  # generations per direction, for a match RATE not a single 0/1
    stage2_max_new_tokens: int = 30
    stage2_temperature: float = 0.8
    stage2_covariate_n_contexts: int = 50     # contexts used for jacobian_magnitude estimation (cheaper than Stage 1's)
    stage2_output_dir: str = "stage2_results"


CFG = Config()