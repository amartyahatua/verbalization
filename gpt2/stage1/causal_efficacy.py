"""
Causal efficacy scoring: does steering a direction actually change
model behavior? This is the key covariate tagged onto every direction
in the bank (Section 5.1) — it's what "the model actually computes
this" gets operationalized as, and it's what later lets Stage 5.1's
"decodable but causally inert" probe directions be identified (a
probe with a high classification accuracy but near-zero causal
efficacy is exactly that case).
"""

import torch
from typing import List, Dict

from model_utils import get_transformer_layers


def steer_hook(direction: torch.Tensor, alpha: float):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + alpha * direction.to(h.device, h.dtype)
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h
    return hook


@torch.no_grad()
def behavioral_change_score(
    model,
    tokenizer,
    layer: int,
    direction: torch.Tensor,
    contexts: List[str],
    alphas: List[float],
    device: str,
) -> float:
    """
    Average, over contexts and steering strengths, the KL divergence
    between the model's next-token distribution with vs. without the
    direction added at `layer`. Higher = more causally efficacious.
    """
    total_kl, n = 0.0, 0

    for text in contexts:
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        base_logits = model(**inputs).logits[0, -1]
        base_logp = torch.log_softmax(base_logits, dim=-1)

        for alpha in alphas:
            handle = get_transformer_layers(model)[layer].register_forward_hook(
                steer_hook(direction, alpha)
            )
            try:
                steered_logits = model(**inputs).logits[0, -1]
            finally:
                handle.remove()
            steered_logp = torch.log_softmax(steered_logits, dim=-1)

            kl = torch.sum(torch.exp(base_logp) * (base_logp - steered_logp))
            total_kl += kl.item()
            n += 1

    return total_kl / max(n, 1)


def tag_causal_efficacy(
    model,
    tokenizer,
    directions: Dict,
    contexts: List[str],
    alphas: List[float],
    device: str,
) -> Dict:
    """
    `directions`: {key: {"layer": int, "vector": Tensor, "class": str, ...}}
    Adds a "causal_efficacy" field to each entry in place, and returns it.
    """
    for key, entry in directions.items():
        score = behavioral_change_score(
            model, tokenizer, entry["layer"], entry["vector"],
            contexts, alphas, device,
        )
        entry["causal_efficacy"] = score
    return directions
