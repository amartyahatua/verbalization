"""
Stage 5's ablation_test builds its prompts via decorrelation.py's
build_input_content("A note nearby reads: '{concept}'.") -- wrong fit
here, since Stage 7's "input" is a realistic scenario passage, not a
sentence naming the concept. This reimplements the ablation logic
against covert_cognition.py's scenario-based prompt builder.

DELIBERATELY SELF-CONTAINED: generation is implemented locally here
(_generate, below) rather than importing elicit_reports from
stage2/elicitation.py for the no-hook control. That control needs an
"apply_hook" option elicit_reports didn't originally have; adding it
there meant this stage's correctness depended on TWO folders
(stage7/ and stage2/) being updated together, which caused real
confusion in practice -- a partial folder replacement left stage7/
current but stage2/elicitation.py stale, and the mismatch wasn't
obvious until a NameError surfaced mid-run. Keeping generation local
to this file means Stage 7 works correctly as long as stage7/ itself
is current, regardless of what state any other stage's folder is in.
"""

import torch
from typing import List, Optional

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage2"))
from model_utils import get_transformer_layers
from editing import apply_direction_editing
from scoring import exact_match_score, score_reports

from covert_cognition import build_cell_prompt, ELICITATION_PROMPT, LABEL_INJECTION


def _generate(
    model, tokenizer, layer: int, direction: torch.Tensor, alpha: float, prompt: str, device: str,
    n_trials: int = 3, max_new_tokens: int = 10, temperature: float = 0.8, apply_hook: bool = True,
) -> List[str]:
    """
    Local generation helper with an explicit apply_hook switch.
    apply_hook=False skips the editing hook entirely -- genuinely no
    intervention, distinct from clamping to alpha=0.0, which still
    actively forces that component to exactly zero. A model reacting
    to "something was clamped here" as a detectable event, independent
    of its value, would pass an alpha=0 check but not this one.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    eos_id = getattr(tokenizer, "eos_token_id", None)

    reports = []
    for _ in range(n_trials):
        handle = apply_direction_editing(model, layer, direction, alpha) if apply_hook else None
        try:
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                    pad_token_id=eos_id, eos_token_id=eos_id,
                )
        finally:
            if handle is not None:
                handle.remove()
        new_tokens = out_ids[0][prompt_len:]
        reports.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return reports


def ablation_test(peft_model, tokenizer, layer, direction, alpha: float, target_label: str,
                   device: str, n_trials: int = 3, max_new_tokens: int = 10, split: str = "train") -> dict:
    """
    split="train" tests on the exact sentences trained on -- prone to
    memorization masking whatever the direction's real causal role
    is, since a memorized literal-text-to-label mapping doesn't need
    the direction at all. split="heldout" tests on phrasings never
    seen during training, which is the regime that actually matters
    for validating a decorrelation-test result: if DFI was measured
    on held-out text, the mechanism check needs to probe held-out
    text too, or it's answering a different question than the one
    that produced the interesting result.
    """
    normal_prompt = build_cell_prompt(target_label, split=split)
    ablated_input_prompt = ELICITATION_PROMPT                          # no scenario passage at all
    sign = 1.0 if target_label == LABEL_INJECTION else -1.0

    def rate(direction_alpha, prompt, apply_hook=True):
        reports = _generate(peft_model, tokenizer, layer, direction, direction_alpha, prompt, device,
                             n_trials=n_trials, max_new_tokens=max_new_tokens, apply_hook=apply_hook)
        counts = score_reports(reports, target_label)     # same scoring as exact_match_score, plus exact k/n
        return counts["rate"], reports, counts

    baseline_rate, baseline_reports, baseline_counts = rate(sign * alpha, normal_prompt)
    ablate_direction_rate, ablate_direction_reports, ablate_direction_counts = rate(0.0, normal_prompt)
    ablate_input_rate, ablate_input_reports, ablate_input_counts = rate(sign * alpha, ablated_input_prompt)
    no_hook_rate, no_hook_reports, no_hook_counts = rate(sign * alpha, normal_prompt, apply_hook=False)

    return {
        "target_label": target_label, "layer": layer, "alpha": sign * alpha, "split": split,
        "baseline": {"match_rate": baseline_rate, "counts": baseline_counts, "reports": baseline_reports},
        "ablate_direction": {"match_rate": ablate_direction_rate, "counts": ablate_direction_counts, "reports": ablate_direction_reports},
        "ablate_input": {"match_rate": ablate_input_rate, "counts": ablate_input_counts, "reports": ablate_input_reports},
        "no_hook_control": {"match_rate": no_hook_rate, "counts": no_hook_counts, "reports": no_hook_reports},
        "direction_necessary": (baseline_rate is not None and ablate_direction_rate is not None
                                 and ablate_direction_rate < baseline_rate),
        "input_necessary": (baseline_rate is not None and ablate_input_rate is not None
                             and ablate_input_rate < baseline_rate),
        "hook_presence_alone_sufficient": (baseline_rate is not None and no_hook_rate is not None
                                            and no_hook_rate >= baseline_rate),
    }
