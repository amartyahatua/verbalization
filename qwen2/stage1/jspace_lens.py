"""
Approximate J-lens (Jacobian lens) direction extraction.

Implements the technique as described in the J-space paper (Gurnee,
Sofroniew et al. 2026): for a given layer and target token, compute
the linearized (Jacobian) effect of the residual-stream activation on
the model's log-probability of producing that token, averaged over
many contexts.

NOTE: this is a best-effort reimplementation from the technique's
public description, not a guaranteed replication — the original
paper's exact averaging scheme (e.g. which future positions count,
how many layers the Jacobian is taken through) isn't fully public.
Validate against known signatures (J-space should end up low-variance
and concentrated in middle layers) before trusting downstream results,
and expect to revisit this module once/if more methodological detail
becomes available.
"""

import torch
from typing import List, Dict, Tuple

from model_utils import get_transformer_layers


def compute_jlens_direction(
    model,
    tokenizer,
    layer: int,
    target_token_id: int,
    contexts: List[str],
    device: str = "cuda",
    verbose_every: int = 0,
) -> torch.Tensor:
    """
    Average, over `contexts`, the gradient of the target token's final-
    position log-probability with respect to the layer-`layer`
    residual-stream activation at that same position.

    Returns an unnormalized direction vector; callers should normalize
    or norm-match it downstream as needed.

    `verbose_every`: if > 0, print progress every N contexts — this
    loop does one full forward+backward pass per context with no
    batching, so on CPU it can take a while; without this you can't
    tell "slow" from "stuck".
    """
    accum = None
    n = 0

    for i, text in enumerate(contexts):
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
            logits = model(**inputs).logits  # [1, seq, vocab]
            target_logprob = torch.log_softmax(logits[0, -1], dim=-1)[target_token_id]
            model.zero_grad(set_to_none=True)
            target_logprob.backward()

            grad = captured["h"].grad[0, -1].detach()  # [hidden]
            accum = grad if accum is None else accum + grad
            n += 1
        finally:
            handle.remove()

        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"      ...{i + 1}/{len(contexts)} contexts processed for this direction", flush=True)

    if n == 0:
        raise ValueError("No contexts produced a usable gradient.")
    direction = (accum / n).float().cpu()
    return direction / direction.norm()  # unit-normalize to match probes.py's convention —
    # raw gradient magnitude is otherwise tiny and incomparable to other direction classes
    # when steering with a fixed alpha (see causal_efficacy.py)


def build_jspace_directions(
    model,
    tokenizer,
    layers: List[int],
    target_tokens: List[str],
    contexts: List[str],
    device: str = "cuda",
) -> Dict[Tuple[int, str], torch.Tensor]:
    """Compute J-lens directions for every (layer, target token) pair."""
    directions = {}
    combos = [(layer, tok) for layer in layers for tok in target_tokens]
    print(
        f"Computing J-lens directions: {len(combos)} (layer, token) pairs x "
        f"{len(contexts)} contexts each = {len(combos) * len(contexts)} forward+backward passes total.",
        flush=True,
    )
    for i, (layer, tok) in enumerate(combos):
        tok_id = tokenizer.encode(tok, add_special_tokens=False)[0]
        directions[(layer, tok)] = compute_jlens_direction(
            model, tokenizer, layer, tok_id, contexts, device=device, verbose_every=100,
        )
        print(f"  [{i + 1}/{len(combos)}] done: layer={layer}, token='{tok}'", flush=True)
    return directions
