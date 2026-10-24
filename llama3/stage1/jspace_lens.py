"""
J-lens (Jacobian lens) direction extraction -- backed by Anthropic's own
reference implementation (anthropics/jacobian-lens, importable as
`jlens`) rather than the hand-rolled per-context backward-pass
approximation the old stage1/jlens.py contained.

FILE RENAMED FROM jlens.py -- READ THIS FIRST: the previous version of
this file was itself named `jlens.py` and lived in this same directory,
which shadowed the pip-installed `jlens` package: running
`python build_direction_bank.py` from this directory puts this
directory at the front of sys.path, so `import jlens` from *inside*
that old file resolved to itself (a partially-initialized self-import),
not to the real library -- hence `AttributeError: module 'jlens' has no
attribute 'from_hf'`. Aliasing the import (`import jlens as
jacobian_lens`) does NOT fix this, because the collision happens at
resolution time, before the alias is applied. The only real fix is to
stop giving the local file the same name as the package, hence this
file is `jspace_lens.py`, not `jlens.py`.

THE ONE CALL SITE THAT NEEDS TO CHANGE because of the rename --
build_direction_bank.py's import line:
    from jspace_lens import build_jspace_directions
becomes:
    from jspace_lens import build_jspace_directions
That's it. Nothing else in build_direction_bank.py changes; the
function name, signature, and return type are all identical to before.

WHY THE SWITCH (unchanged rationale from the original replacement): the
previous module said so itself -- "a best-effort reimplementation from
the technique's public description, not a guaranteed replication" --
because it approximated J_l by backpropagating a single target token's
log-probability through one context at a time and averaging the
resulting per-context gradients. The real jlens package instead fits
the full averaged Jacobian J_l = E[dh_final/dh_l] via batched
vector-Jacobian products over a corpus (jlens.fitting.fit), which is
the actual method the paper and Anthropic's own repo define, not an
approximation of it -- results from this module can now be described as
J-space directions rather than "J-space-inspired." It also only needs
to be fit ONCE per model (not once per (layer, token) pair): the
function below does a single jlens.fit() call across every requested
layer at once, then derives each (layer, token) direction from the
fitted dense Jacobian with one matrix-vector product -- replacing what
used to be `len(layers) * len(target_tokens) * len(contexts)`
forward+backward passes with a cost that does NOT scale with
len(layers) or len(target_tokens) at all (see build_jspace_directions'
docstring).

INSTALL:
    pip install git+https://github.com/anthropics/jacobian-lens

VERIFIED against the actual upstream source (jlens/__init__.py,
jlens/hf.py, jlens/fitting.py) as of this writing:
  - `jlens.from_hf(hf_model, tokenizer, *, layout=None, text_module=None,
    compile=False, force_bos=True) -> HFLensModel` IS exported at the
    package top level (it's in jlens/__init__.py's __all__) -- the
    AttributeError you hit was the shadowing bug above, not a missing
    upstream function.
  - `jlens.fit(model, prompts, *, source_layers=None, target_layer=None,
    dim_batch=8, max_seq_len=128, skip_first=16, checkpoint_path=None,
    checkpoint_every=1, resume=True) -> JacobianLens` matches what's
    called below.
  - `JacobianLens.jacobians` is a `dict[int, torch.Tensor]`, exactly as
    this module assumes.
Still worth re-checking against whatever version pip actually resolves
for you (see "BEFORE TRUSTING THIS FOR A REAL RUN" below) -- this was
verified against the main branch, not a pinned release tag.

DERIVING A DIRECTION FROM A FITTED LENS: the upstream package fits and
applies the lens as a full per-layer operator (any activation in ->
ranked vocab out); it does NOT expose a "the direction that makes
token X more likely" helper, because that's not what the paper's main
use case (reading out an existing activation) needs. But the Jacobian
is stored as a plain dense tensor (`lens.jacobians[layer]`, shape
[d_model, d_model], mapping a layer-`layer` residual into the FINAL
residual basis -- apply() composes it with the model's unembedding
afterward). Since
    logit_X = unembed_row_X @ (J_l @ h) = (J_l.T @ unembed_row_X) @ h,
the direction in layer-`layer` space that increases token X's logit to
first order is just (J_l.T @ unembed_row_X) -- a single matvec, done in
_direction_for_token() below. This is derived from the exposed
Jacobian, not an upstream library feature -- if a future jlens release
ships an equivalent helper, prefer that over this function.

BEFORE TRUSTING THIS FOR A REAL RUN -- same spirit as this project's
existing sae_release/sae_id_template verification habit (see
config.py, sae_features.py): don't assume the API described above is
exactly what your installed version exposes. Check:
    python -c "
    import jlens
    print(jlens.__file__)
    print(jlens.fit, jlens.from_hf)
    "
and after your first real fit() call, sanity-check
`sorted(lens.jacobians.keys()) == sorted(layers)` and
`lens.jacobians[layers[0]].shape == (model.config.hidden_size,) * 2`
before handing directions downstream.

VALIDATE (same caveat the old module carried, now pointed at a genuine
implementation instead of an approximation of one): a correct METHOD
does not guarantee your particular model/corpus/token set reproduces
the paper's qualitative findings. Still check for the low-variance,
middle-layer-concentrated J-space signature before trusting downstream
(Stage 2+) results.
"""

import torch
from typing import Dict, List, Optional, Tuple

# Safe now that this file is no longer named jlens.py -- `import jlens`
# unambiguously resolves to the pip-installed package. Kept under an
# alias purely for readability (so every upstream call below reads
# `jacobian_lens.*`, visually distinct from this project's own names).
import jlens as jacobian_lens


def _unembed_row(model, token_id: int) -> torch.Tensor:
    """
    The row of the model's output-embedding (unembedding) matrix for
    `token_id`, as a plain [d_model] float32 CPU tensor.

    Uses get_output_embeddings() rather than assuming `model.lm_head`
    -- that attribute name isn't universal across HF architectures,
    whereas get_output_embeddings() is the HF-wide, architecture-
    agnostic accessor. Same spirit as model_utils.get_transformer_layers()
    existing so nothing in this codebase hardcodes one architecture's
    attribute path.
    """
    W_U = model.get_output_embeddings().weight  # [vocab, d_model]
    return W_U[token_id].detach().float().cpu()


def _direction_for_token(lens, model, layer: int, token_id: int) -> torch.Tensor:
    """
    The layer-`layer` residual-space direction that increases token
    `token_id`'s logit to first order under the fitted lens:
    J_l.T @ unembed_row(token_id). See module docstring for the
    derivation. Unit-normalized to match this project's existing
    convention (probes.py, random_directions.py, and the old jlens.py)
    for comparability across direction classes under a fixed steering
    alpha in causal_efficacy.py.
    """
    J_l = lens.jacobians[layer].detach().float().cpu()  # [d_model, d_model]
    row = _unembed_row(model, token_id)                  # [d_model]
    direction = J_l.T @ row
    norm = direction.norm()
    if norm == 0:
        raise ValueError(
            f"Zero-norm direction for layer={layer}, token_id={token_id} -- "
            f"either the fitted Jacobian for this layer is degenerate, or "
            f"token_id doesn't index a real row of the unembedding matrix. "
            f"Check tokenizer.decode([{token_id}]) and lens.jacobians[{layer}]."
        )
    return direction / norm


def build_jspace_directions(
    model,
    tokenizer,
    layers: List[int],
    target_tokens: List[str],
    contexts: List[str],
    device: str = "cuda",
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = 16,
    checkpoint_path: Optional[str] = None,
) -> Dict[Tuple[int, str], torch.Tensor]:
    """
    Fit the real Jacobian lens once on `contexts` (across every layer
    in `layers` simultaneously), then derive a unit direction for every
    (layer, target token) pair from the fitted matrices.

    Same function name, signature (plus new keyword-only tuning params,
    all defaulted), and return type as the old per-(layer,token)
    implementation -- build_direction_bank.py's existing call site,
        build_jspace_directions(model, tokenizer, CFG.sae_layers,
                                 CFG.jlens_target_tokens, contexts, device)
    needs no changes beyond the import line (see module docstring).

    COST SHAPE IS DIFFERENT FROM THE OLD MODULE -- read before running
    on the full pipeline:
    fit() cost is ceil(d_model / dim_batch) backward passes PER PROMPT,
    and that cost is shared across every layer in `layers` and every
    token in `target_tokens` (deriving each direction afterward is one
    cheap matvec, not a forward+backward pass). It does NOT scale with
    len(layers) * len(target_tokens) the way the old per-token loop did
    -- the old module's "56 x n_contexts passes" framing (config.py's
    J-lens note -- still stale there, see the note this project should
    update) no longer applies at all; think in terms of
    ceil(d_model / dim_batch) x n_contexts instead. At Llama-3.1-8B's
    d_model=4096, the default dim_batch=8 is 512 backward passes per
    prompt -- raise dim_batch (GPU memory permitting) before running
    this on the full jlens_n_contexts=300 rather than assuming the old
    per-context timing estimates carry over.

    `max_seq_len`, `skip_first`: passed through to jacobian_lens.fit;
    defaults match the paper/upstream repo's own defaults, not this
    project's context truncation in data.py (contexts arrive here
    already truncated to 512 characters -- skip_first=16 and
    max_seq_len=128 apply to the re-tokenized sequence afterward,
    independent of that character-level truncation. Early positions are
    excluded upstream as attention-sink-dominated, and the final
    position is excluded as having no next-token target -- see
    jacobian-lens's own fitting.py for why).

    `checkpoint_path`: optional, forwarded to jacobian_lens.fit as-is.
    Not set by default -- pass one explicitly (e.g. under CFG.output_dir)
    if you want a resumable fit for a long run; unlike the rest of this
    codebase's checkpointing conventions, this defers entirely to
    upstream's own checkpoint/resume behavior rather than reimplementing
    it here.
    """
    lens_model = jacobian_lens.from_hf(model, tokenizer)

    hidden_size = getattr(model.config, "hidden_size", None)
    passes_per_prompt = f"~{-(-hidden_size // dim_batch)}" if hidden_size else "an unknown number of"
    print(
        f"Fitting Jacobian lens on {len(contexts)} contexts, "
        f"source_layers={layers}, dim_batch={dim_batch} "
        f"({passes_per_prompt} backward passes/prompt, shared across all "
        f"{len(layers)} layers and all {len(target_tokens)} target tokens)...",
        flush=True,
    )
    fit_kwargs = dict(
        source_layers=layers,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first=skip_first,
    )
    if checkpoint_path is not None:
        fit_kwargs["checkpoint_path"] = checkpoint_path

    lens = jacobian_lens.fit(lens_model, contexts, **fit_kwargs)

    fitted_layers = sorted(lens.jacobians.keys())
    if fitted_layers != sorted(layers):
        print(
            f"  WARNING: requested source_layers={sorted(layers)} but the fitted "
            f"lens only has jacobians for {fitted_layers} -- check for a version "
            f"mismatch against the API this module assumes (see module docstring's "
            f"'BEFORE TRUSTING THIS' section) before continuing.",
            flush=True,
        )
    print("Lens fit complete. Deriving per-token directions...", flush=True)

    directions = {}
    combos = [(layer, tok) for layer in layers for tok in target_tokens]
    for i, (layer, tok) in enumerate(combos):
        tok_id = tokenizer.encode(tok, add_special_tokens=False)[0]
        directions[(layer, tok)] = _direction_for_token(lens, model, layer, tok_id)
        print(f"  [{i + 1}/{len(combos)}] done: layer={layer}, token='{tok}'", flush=True)

    return directions