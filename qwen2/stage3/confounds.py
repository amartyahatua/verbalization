"""
Stage 3 confound controls.

(b) co-activated correlates: the bipolar direction d_XY might just BE
(or strongly overlap with) the input-token embeddings for X/Y, or
whatever SAE features happen to co-occur with it -- in which case any
apparent causal effect could be mediated through those, not d_XY
itself. Gram-Schmidt orthogonalizes d_XY against both before use.

(c) unembedding triviality: if a direction's causal effect on the
report is no better than what you'd get by just reading off its
closest vocabulary token via the unembedding matrix (skipping all
intermediate computation), that's evidence the "report" isn't
reflecting genuine processing -- just the direction's raw logit-lens
alignment. logit_lens_report() computes that trivial baseline
directly, for comparison against the real elicited reports.
"""

import torch
from typing import Dict, List, Optional

# --- make stage1/ importable from here ---
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))

from model_utils import get_transformer_layers


def gram_schmidt_orthogonalize(direction: torch.Tensor, confounds: List[torch.Tensor]) -> torch.Tensor:
    """
    Project `direction` to be orthogonal to the ENTIRE SPAN of
    `confounds`, then re-normalize.

    Built via an orthonormal basis (QR) of the confound vectors
    rather than naive sequential subtraction: subtracting each raw
    confound's projection one at a time only guarantees exact
    orthogonality to the last one subtracted, since arbitrary
    confound vectors aren't orthogonal to each other -- subtracting
    the second can reintroduce a small component along the first.
    Projecting onto an orthonormal basis of their span avoids that.
    """
    d = direction.clone().float()
    if not confounds:
        return d / d.norm()

    device = d.device
    C = torch.stack([c.float().to(device) for c in confounds], dim=1)  # [d_model, n_confounds]
    Q, _ = torch.linalg.qr(C)                                 # orthonormal basis for span(confounds)
    d_orth = d - Q @ (Q.T @ d)

    norm = d_orth.norm()
    if norm < 1e-6:
        raise ValueError(
            "Direction collapsed to ~zero after orthogonalization -- it was almost "
            "entirely explained by the confound vectors. This is itself a finding "
            "worth reporting, not just an error to work around."
        )
    return d_orth / norm


def get_input_embedding_vectors(model, tokenizer, tokens: List[str]) -> List[torch.Tensor]:
    """Input-embedding (not unembedding) rows for the given tokens -- the confound-(b) 'is this
    direction just the word embedding' check."""
    W_E = model.get_input_embeddings().weight.detach().float()
    vecs = []
    for tok in tokens:
        tok_id = tokenizer.encode(tok, add_special_tokens=False)[0]
        vecs.append(W_E[tok_id])
    return vecs


def get_top_cofiring_sae_features(
    direction: torch.Tensor, sae_vectors: Dict[str, torch.Tensor], k: int = 5
) -> List[torch.Tensor]:
    """
    Top-k SAE decoder vectors (from this layer's slice of the Stage 1
    bank) by cosine similarity to `direction` -- a cheap static proxy
    for "features that tend to co-activate with this direction",
    since it doesn't require live activation data to compute.
    """
    d_hat = direction.float() / direction.float().norm()
    scored = []
    for key, vec in sae_vectors.items():
        v = vec.float().to(d_hat.device)
        cos = torch.dot(d_hat, v / v.norm().clamp_min(1e-8)).abs().item()
        scored.append((cos, vec))
    scored.sort(key=lambda x: -x[0])
    return [vec.to(d_hat.device) for _, vec in scored[:k]]


def logit_lens_report(model, tokenizer, direction: torch.Tensor, k: int = 3) -> List[str]:
    """
    Trivial baseline: project `direction` straight through the
    unembedding matrix (skipping all intermediate layers/computation)
    and return the top-k closest vocabulary tokens by cosine
    similarity. This is what confound (c) compares the real elicited
    reports against.
    """
    W_U = model.get_output_embeddings().weight.detach().float()
    d_hat = direction.float().to(W_U.device)
    d_hat = d_hat / d_hat.norm()
    row_norms = W_U.norm(dim=1).clamp_min(1e-8)
    cos_sims = (W_U @ d_hat) / row_norms
    top_ids = torch.topk(cos_sims, k=k).indices.tolist()
    return [tokenizer.decode([i]).strip() for i in top_ids]


# ---------------------------------------------------------------------------
# Output-pathway (unembedding) control -- camera-ready revision
# ---------------------------------------------------------------------------
#
# Reviewer concern: J-lens directions are gradients of a target token's log-prob, so by
# construction they carry a large component along that token's unembedding row
# (unembedding_alignment ~0.57 for J-lens vs ~0.11-0.20 for the control classes). A clamped
# J-lens direction could therefore raise the logit of the label directly -- "verbalization"
# with no read-out of an internal state. The logit-lens report above (confound (c)) only
# COMPARES against that trivial route; it doesn't remove it. The functions below remove it:
# project the target tokens' output readout vectors out of the direction, THEN run the same
# clamping experiment. If direction-following survives, it can't be coming from the direct
# logit path through those tokens; if it vanishes, the "positive" results were that path.
#
# What counts as the "output readout" of token t:
#   * the raw unembedding row            W_U[t]
#   * the same row seen through the final norm's learned scale   gamma * W_U[t]
#     (logits = W_U @ (gamma * h / rms(h)), so gamma * W_U[t] is the direction in the
#     pre-norm residual stream that actually moves logit t)
# and for each token both the bare and the leading-space variants ("Paris" / " Paris"), since
# the report's first token is usually the space-prefixed one while J-lens extraction uses the
# bare token (jlens.py) and training targets use the space-prefixed one (lora_training.py).

def _first_token_ids(tokenizer, token: str) -> List[int]:
    ids = []
    for variant in (token, " " + token):
        enc = tokenizer.encode(variant, add_special_tokens=False)
        if enc and enc[0] not in ids:
            ids.append(enc[0])
    return ids


def _final_norm_scale(model) -> Optional[torch.Tensor]:
    """gamma of the final (pre-unembedding) norm, or None if this architecture isn't recognised."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for path in (("model", "norm"), ("transformer", "ln_f"), ("gpt_neox", "final_layer_norm")):
        obj = base
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        w = getattr(obj, "weight", None)
        if w is not None:
            return w.detach().float()
    return None


def unembedding_vectors(model, tokenizer, tokens: List[str], fold_final_norm: bool = True) -> List[torch.Tensor]:
    """Output-readout vectors (see block comment above) for every token in `tokens`."""
    W_U = model.get_output_embeddings().weight.detach()      # NOT .float() on the whole matrix: for an 8B model that
    gamma = _final_norm_scale(model) if fold_final_norm else None   # is a ~2 GB copy per call. Cast only the rows used.
    seen, vecs = set(), []
    for tok in tokens:
        for tid in _first_token_ids(tokenizer, tok):
            if tid in seen:
                continue
            seen.add(tid)
            row = W_U[tid].float()
            vecs.append(row)
            if gamma is not None and gamma.shape == row.shape:
                vecs.append(row * gamma.to(row.device))
    return vecs


def project_out_span(direction: torch.Tensor, vectors: List[torch.Tensor], rank_tol: float = 1e-4) -> torch.Tensor:
    """
    Remove from `direction` its component in span(vectors); re-normalise. Uses an SVD-truncated basis
    (singular values below rank_tol * largest are dropped) so near-duplicate vectors -- here the raw and
    norm-scaled rows of one token -- can't add arbitrary extra orthonormal columns to the projector.
    Raises ValueError if (almost) nothing is left, which is itself a reportable result.
    """
    d = direction.detach().clone().float()
    if not vectors:
        return d / d.norm()
    C = torch.stack([v.float().to(d.device) for v in vectors], dim=1)       # [d_model, n]
    U, S, _ = torch.linalg.svd(C, full_matrices=False)
    keep = S > rank_tol * S.max()
    Q = U[:, keep]
    d_proj = d - Q @ (Q.T @ d)
    remaining = d_proj.norm() / d.norm()
    if remaining < 0.05:
        raise ValueError(
            f"Direction collapsed after projecting out the output readout (only {remaining:.3f} of its norm "
            f"is left) -- it is almost entirely the unembedding component. That is a finding, not a bug."
        )
    return d_proj / d_proj.norm()


def project_out_unembedding(direction: torch.Tensor, model, tokenizer, tokens: List[str],
                            fold_final_norm: bool = True):
    """
    Returns (projected_unit_direction, diagnostics). `tokens` = the labels whose logits the direction
    must not be able to move directly (the pair being tested, or every candidate label in play).
    diagnostics records how much was removed, and the cosine to each label's raw unembedding row before
    and after (the same quantity as the `unembedding_alignment` covariate, so the numbers are comparable
    with the paper's Table of covariates).
    """
    vecs = unembedding_vectors(model, tokenizer, tokens, fold_final_norm)
    d = direction.detach().float()
    d_hat = d / d.norm()
    d_proj = project_out_span(d, vecs)

    W_U = model.get_output_embeddings().weight.detach()

    def cos_to_rows(vec):
        out = {}
        v = (vec / vec.norm()).to(W_U.device).float()
        for tok in tokens:
            tid = tokenizer.encode(tok, add_special_tokens=False)[0]
            out[tok] = torch.nn.functional.cosine_similarity(v, W_U[tid].float(), dim=0).item()
        return out

    return d_proj, {
        "tokens": list(tokens),
        "n_readout_vectors": len(vecs),
        "fold_final_norm": fold_final_norm,
        "cosine_original_vs_projected": torch.dot(d_hat.cpu(), d_proj.cpu()).item(),
        "unembedding_alignment_before": cos_to_rows(d_hat),
        "unembedding_alignment_after": cos_to_rows(d_proj),
    }


def steering_efficacy_comparison(model, tokenizer, layer: int, original: torch.Tensor,
                                 projected: torch.Tensor, contexts: List[str], alphas: List[float],
                                 device: str) -> dict:
    """
    Does the projected direction still DO anything? Mean next-token KL under additive steering
    (the same behavioral_change_score used to tag the Stage 1 bank) for the original and projected
    direction, plus a norm-matched random direction as the floor. If the projected direction's
    efficacy collapses to the random floor, a null result on it says little about introspection --
    it would only mean the direction was inert once the unembedding part was removed.
    """
    from causal_efficacy import behavioral_change_score
    g = torch.Generator().manual_seed(1234 + layer)
    rand = torch.randn(original.shape[0], generator=g)
    rand = rand / rand.norm()
    score = lambda d: behavioral_change_score(model, tokenizer, layer, d.to(device), contexts, alphas, device)
    return {"original": score(original), "projected": score(projected), "random_floor": score(rand),
            "alphas": list(alphas), "n_contexts": len(contexts)}
