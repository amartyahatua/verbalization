"""
Stage 5 mechanism check (H4).

target_token_jacobian_alignment: unlike Stage 2's jacobian_magnitude
(which measures sensitivity of the TOP predicted token on generic
background text), this measures the directional derivative of the
TARGET LABEL's logprob specifically, at the actual elicitation
prompt -- the precise thing training was trying to increase. Compared
before vs. after training via peft_model.disable_adapter(), which was
verified to correctly reflect the trained contribution (confirmed
against a real trained adapter before building this: enabled vs.
disabled outputs were identical pre-training, and measurably
different -- max diff 0.79 -- after 20 training steps).

ablation_test runs FOUR conditions; their exact definitions (also in
llama3/PROTOCOL.md, because reviewers asked for them to be stated once, precisely):

  baseline          input text present, direction clamped to +/-alpha.
  direction ablation  input text present, direction clamped to alpha = 0: the hook is ACTIVE, the activation's
                    current component along the direction is projected out and set to exactly zero
                    (apply_direction_editing(..., alpha=0.0)) -- an active null intervention on that component.
  input ablation    the input-content sentence/passage is replaced by the bare elicitation prompt (no
                    concept-bearing text at all) while the direction stays clamped at +/-alpha.
  no-hook control   input text present, the hook is entirely ABSENT (apply_hook=False): no projection, no
                    clamp. Distinct from direction ablation: a model that reacts to "something was edited
                    here" regardless of its value would pass a direction-ablation check but not this one.
"""

import torch
from typing import List, Optional

# --- make stage2/ and stage3/ importable from here ---
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage3"))

from elicitation import elicit_reports
from scoring import exact_match_score, score_reports
from decorrelation import build_input_content
from input_provider import get_input_provider
from generalization import generate_over_texts

# --- make stage1/ importable (model_utils) ---
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))
from model_utils import get_transformer_layers


def target_token_jacobian_alignment(
    model, tokenizer, layer: int, direction: torch.Tensor, prompt: str, target_label: str, device: str,
) -> float:
    d_hat = (direction / direction.norm()).to(device)
    target_id = tokenizer.encode(" " + target_label.strip(), add_special_tokens=False)[0]

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    captured = {}

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h.requires_grad_(True)  # force grad-tracking from this point forward -- inside
        # peft_model.disable_adapter(), every parameter is frozen (base model frozen by
        # PEFT, LoRA contribution bypassed), so nothing upstream has requires_grad=True
        # and h would otherwise never be part of a tracked graph at all.
        h.retain_grad()
        captured["h"] = h
        return out

    handle = get_transformer_layers(model)[layer].register_forward_hook(hook)
    try:
        logits = model(**inputs).logits[0, -1]
        logprob = torch.log_softmax(logits, dim=-1)[target_id]
        model.zero_grad(set_to_none=True)
        logprob.backward()
        grad = captured["h"].grad[0, -1].detach().float()
        return torch.dot(grad, d_hat.float()).item()
    finally:
        handle.remove()


def jacobian_alignment_pre_post(peft_model, tokenizer, layer: int, direction: torch.Tensor,
                                 prompt: str, target_label: str, device: str) -> dict:
    after = target_token_jacobian_alignment(peft_model, tokenizer, layer, direction, prompt, target_label, device)
    with peft_model.disable_adapter():
        before = target_token_jacobian_alignment(peft_model, tokenizer, layer, direction, prompt, target_label, device)
    return {"before_training": before, "after_training": after, "delta": after - before}


def ablation_test(
    peft_model, tokenizer, layer: int, direction: torch.Tensor, alpha: float,
    x_concept: str, elicitation_prompt: str, device: str, n_trials: int = 3, max_new_tokens: int = 10,
    provider=None, candidate_labels=None,
) -> dict:
    """
    Four-condition ablation (definitions in the module docstring). With the default TemplateInputs provider
    this is the original test plus the no-hook control and exact counts; with a provider that has several
    held-out texts, every text gets `n_trials` generations and reports are pooled. The input-ablated cell has
    no input text, so it is given n_trials * n_texts generations from the bare prompt to keep n comparable.
    candidate_labels=None reproduces the original scoring (no shotgun-listing check) -- see PROTOCOL.md.
    """
    provider = provider or get_input_provider("template")
    texts = provider.eval_texts(x_concept)

    def run(direction_alpha, with_input=True, apply_hook=True):
        if with_input:
            reports = generate_over_texts(peft_model, tokenizer, layer, direction, direction_alpha, texts,
                                          elicitation_prompt, device, n_trials, max_new_tokens, apply_hook=apply_hook)
        else:  # no concept-bearing input at all
            reports = elicit_reports(peft_model, tokenizer, layer, direction, direction_alpha, elicitation_prompt,
                                     device, n_trials=n_trials * len(texts), max_new_tokens=max_new_tokens,
                                     apply_hook=apply_hook)
        counts = score_reports(reports, x_concept, candidate_labels=candidate_labels)
        return counts["rate"], reports, counts

    baseline_rate, baseline_reports, baseline_counts = run(alpha)
    ablate_direction_rate, ablate_direction_reports, ablate_direction_counts = run(0.0)
    ablate_input_rate, ablate_input_reports, ablate_input_counts = run(alpha, with_input=False)
    no_hook_rate, no_hook_reports, no_hook_counts = run(alpha, apply_hook=False)

    def lower(a, b):
        return a is not None and b is not None and a < b

    return {
        "x_concept": x_concept, "layer": layer, "alpha": alpha, "n_input_texts": len(texts),
        "baseline": {"match_rate": baseline_rate, "counts": baseline_counts, "reports": baseline_reports},
        "ablate_direction": {"match_rate": ablate_direction_rate, "counts": ablate_direction_counts,
                             "reports": ablate_direction_reports},
        "ablate_input": {"match_rate": ablate_input_rate, "counts": ablate_input_counts, "reports": ablate_input_reports},
        "no_hook_control": {"match_rate": no_hook_rate, "counts": no_hook_counts, "reports": no_hook_reports},
        "direction_necessary": lower(ablate_direction_rate, baseline_rate),
        "input_necessary": lower(ablate_input_rate, baseline_rate),
        "hook_presence_alone_sufficient": (baseline_rate is not None and no_hook_rate is not None
                                            and no_hook_rate >= baseline_rate),
    }
