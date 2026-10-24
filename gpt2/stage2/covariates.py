"""
Covariates for the Stage 2 mixed-effects regression:

  - jacobian_magnitude: how strongly a first-order perturbation along
    `direction` at this layer affects the model's near-term output --
    the magnitude of the directional derivative of the top-predicted
    token's log-probability with respect to `direction`, averaged
    over contexts. Reuses jlens.py's gradient-hook pattern, but
    against a *given* direction instead of computing a new one from a
    target token.
  - unembedding_alignment: cosine similarity between `direction` and
    the unembedding row of its associated token, when one is known
    (e.g. a J-space direction's own target token); otherwise the max
    cosine similarity with ANY vocabulary row (how strongly this
    direction points toward *some* specific token).
  - workspace_participation: fraction of `direction`'s norm that lies
    within the subspace spanned by that layer's J-space directions.
"""

import torch
from typing import Dict, List, Optional

# --- make stage1/ importable from here (see editing.py for rationale) ---
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))

from model_utils import get_transformer_layers


def jacobian_magnitude(
    model,
    tokenizer,
    layer: int,
    direction: torch.Tensor,
    contexts: List[str],
    device: str,
) -> float:
    d_hat = (direction / direction.norm()).to(device)
    total, n = 0.0, 0

    for text in contexts:
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        captured = {}

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h.retain_grad()
            captured["h"] = h
            return out

        handle = get_transformer_layers(model)[layer].register_forward_hook(hook)
        try:
            logits = model(**inputs).logits[0, -1]
            top_token = torch.argmax(logits).item()
            logprob = torch.log_softmax(logits, dim=-1)[top_token]
            model.zero_grad(set_to_none=True)
            logprob.backward()
            grad = captured["h"].grad[0, -1].detach().float()
            total += torch.dot(grad, d_hat.float()).abs().item()
            n += 1
        finally:
            handle.remove()

    return total / max(n, 1)


def unembedding_alignment(
    model,
    direction: torch.Tensor,
    tokenizer=None,
    target_token: Optional[str] = None,
) -> float:
    W_U = model.get_output_embeddings().weight.detach().float()  # [vocab, d_model]
    d_hat = (direction / direction.norm()).float().to(W_U.device)

    if target_token is not None and tokenizer is not None:
        tok_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
        row = W_U[tok_id]
        return torch.nn.functional.cosine_similarity(d_hat, row, dim=0).item()

    row_norms = W_U.norm(dim=1).clamp_min(1e-8)
    cos_sims = (W_U @ d_hat) / row_norms
    return cos_sims.max().item()


def build_jspace_basis(bank: Dict, layer: int) -> Optional[torch.Tensor]:
    """Orthonormal basis (QR) for the span of this layer's J-space directions."""
    vecs = [v["vector"] for v in bank.values() if v["class"] == "jspace" and v["layer"] == layer]
    if not vecs:
        return None
    M = torch.stack(vecs, dim=1).float()  # [d_model, n_jspace_dirs]
    Q, _ = torch.linalg.qr(M)
    return Q  # [d_model, rank], orthonormal columns


def workspace_participation(direction: torch.Tensor, jspace_basis: Optional[torch.Tensor]) -> float:
    if jspace_basis is None:
        return 0.0
    d = direction.float()
    coeffs = jspace_basis.T @ d
    proj = jspace_basis @ coeffs
    return (proj.norm() / d.norm().clamp_min(1e-8)).item()
