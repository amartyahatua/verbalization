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
from model_utils import get_transformer_layers, format_prompt_for_chat
from editing import apply_direction_editing
from scoring import exact_match_score, score_reports

from covert_cognition import build_cell_prompt, all_cell_prompts, ELICITATION_PROMPT, LABEL_INJECTION


def _generate(
    model, tokenizer, layer: int, direction: torch.Tensor, alpha: float, prompt: str, device: str,
    n_trials: int = 3, max_new_tokens: int = 10, temperature: float = 0.8, apply_hook: bool = True,
    use_chat_template: bool = True,
) -> List[str]:
    """
    Local generation helper with an explicit apply_hook switch.
    apply_hook=False skips the editing hook entirely -- genuinely no
    intervention, distinct from clamping to alpha=0.0, which still
    actively forces that component to exactly zero. A model reacting
    to "something was clamped here" as a detectable event, independent
    of its value, would pass an alpha=0 check but not this one.
    """
    prompt = format_prompt_for_chat(tokenizer, prompt, use_chat_template)
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
    # EVERY phrasing in the split is used (n_trials generations each, pooled), not just the first one.
    normal_prompts = all_cell_prompts(target_label, split)
    ablated_input_prompts = [ELICITATION_PROMPT]                       # no scenario passage at all
    sign = 1.0 if target_label == LABEL_INJECTION else -1.0
    n_total_gens = n_trials * len(normal_prompts)

    def rate(direction_alpha, prompts, apply_hook=True):
        # keep n equal across the four conditions: the no-passage prompt gets the same total generations
        per_prompt = n_total_gens // len(prompts)
        reports = []
        for prompt in prompts:
            reports += _generate(peft_model, tokenizer, layer, direction, direction_alpha, prompt, device,
                                 n_trials=per_prompt, max_new_tokens=max_new_tokens, apply_hook=apply_hook)
        counts = score_reports(reports, target_label)     # same scoring as exact_match_score, plus exact k/n
        return counts["rate"], reports, counts

    baseline_rate, baseline_reports, baseline_counts = rate(sign * alpha, normal_prompts)
    ablate_direction_rate, ablate_direction_reports, ablate_direction_counts = rate(0.0, normal_prompts)
    ablate_input_rate, ablate_input_reports, ablate_input_counts = rate(sign * alpha, ablated_input_prompts)
    no_hook_rate, no_hook_reports, no_hook_counts = rate(sign * alpha, normal_prompts, apply_hook=False)

    return {
        "target_label": target_label, "layer": layer, "alpha": sign * alpha, "split": split,
        "n_phrasings": len(normal_prompts), "n_trials_per_phrasing": n_trials,
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


def _label_first_token_ids(tokenizer, label: str):
    ids = set()
    for variant in (label, " " + label, label.capitalize(), " " + label.capitalize()):
        enc = tokenizer.encode(variant, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return sorted(ids)


def _label_position_logits(peft_model, tokenizer, inputs: dict, max_skip: int = 3):
    """
    Next-token logits at the position where the REPORT is written. Stage 4/7 training targets are tokenized with
    add_special_tokens=True, so the adapters learn to emit <|begin_of_text|> first and the label second (the report
    decoder drops special tokens, which hides this). If the argmax next token is a special token other than EOS, feed it
    back in (greedy, up to `max_skip` times) and read the logits after it. Returns (logits, skipped_token_strings).
    The caller's editing hook is active for every forward pass here, as it is during generation.
    """
    special = set(getattr(tokenizer, "all_special_ids", []) or []) - {getattr(tokenizer, "eos_token_id", None)}
    ids = inputs["input_ids"]
    mask = inputs.get("attention_mask")
    if mask is None:
        mask = torch.ones_like(ids)
    skipped = []
    for _ in range(max_skip + 1):
        lg = peft_model(input_ids=ids, attention_mask=mask).logits[0, -1].float()
        top = int(lg.argmax())
        if top in special and len(skipped) < max_skip:
            skipped.append(tokenizer.decode([top]))
            ids = torch.cat([ids, torch.tensor([[top]], device=ids.device)], dim=1)
            mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=mask.device)], dim=1)
            continue
        return lg, skipped
    return lg, skipped


def signed_alpha_sweep(peft_model, tokenizer, layer, directions: dict, base_alpha: float, multipliers, prompts: List[str],
                       device: str, label_pos: str, label_neg: str, use_chat_template: bool = True) -> dict:
    """
    Dose-response check that does NOT depend on which label the adapter says by default.

    For each direction in `directions` ({name: unit vector}), clamp the signed value m * base_alpha for each m in
    `multipliers`, and read the NEXT-TOKEN logit difference  logit(label_pos) - logit(label_neg)  (log-sum-exp over
    the bare / space-prefixed / capitalised first tokens of each label) on every prompt. A reporter that reads the
    direction shows a clearly positive slope of that difference against m; a default bias only shifts the intercept.
    Slope is least-squares per prompt (so prompts are the unit of replication), then averaged.
    """
    pos_ids, neg_ids = _label_first_token_ids(tokenizer, label_pos), _label_first_token_ids(tokenizer, label_neg)
    ms = [float(m) for m in multipliers]
    mc = sum(ms) / len(ms)
    denom = sum((m - mc) ** 2 for m in ms)
    out = {"multipliers": ms, "base_alpha": base_alpha, "label_pos": label_pos, "label_neg": label_neg,
           "n_prompts": len(prompts), "directions": {}}
    for name, direction in directions.items():
        per_prompt = []                                  # [prompt][multiplier] -> logit diff
        for prompt in prompts:
            text = format_prompt_for_chat(tokenizer, prompt, use_chat_template)
            inputs = {k: v.to(device) for k, v in tokenizer(text, return_tensors="pt").items()}
            row = []
            for m in ms:
                handle = apply_direction_editing(peft_model, layer, direction, m * base_alpha)
                try:
                    with torch.no_grad():
                        logits, _ = _label_position_logits(peft_model, tokenizer, inputs)
                finally:
                    handle.remove()
                lp = torch.logsumexp(logits[pos_ids], 0) - torch.logsumexp(logits[neg_ids], 0)
                row.append(float(lp))
            per_prompt.append(row)
        if name == "real":
            # debug: what is the actual argmax next token (and the label-token logits) at m = -1 / +1 on one injection-text
            # and one benign-text prompt? Lets us check that the label token ids we sum over are the ones the adapter emits.
            dbg = []
            for pi in (0, len(prompts) - 1):
                text = format_prompt_for_chat(tokenizer, prompts[pi], use_chat_template)
                inputs = {k: v.to(device) for k, v in tokenizer(text, return_tensors="pt").items()}
                for m in (-1.0, 1.0):
                    handle = apply_direction_editing(peft_model, layer, direction, m * base_alpha)
                    try:
                        with torch.no_grad():
                            lg, skipped = _label_position_logits(peft_model, tokenizer, inputs)
                    finally:
                        handle.remove()
                    top = torch.topk(lg, 5)
                    dbg.append({"prompt_index": pi, "multiplier": m, "skipped_special_tokens": skipped,
                                "top5": [(tokenizer.decode([int(i)]), round(float(v), 2)) for v, i in zip(top.values, top.indices)],
                                "label_pos_logits": {tokenizer.decode([i]): round(float(lg[i]), 2) for i in pos_ids},
                                "label_neg_logits": {tokenizer.decode([i]): round(float(lg[i]), 2) for i in neg_ids}})
            out["first_token_debug"] = dbg
        slopes = [sum((m - mc) * (y - sum(row) / len(row)) for m, y in zip(ms, row)) / denom for row in per_prompt]
        mean_by_m = [sum(row[j] for row in per_prompt) / len(per_prompt) for j in range(len(ms))]
        out["directions"][name] = {"logit_diff_by_multiplier": mean_by_m, "per_prompt": per_prompt,
                                    "slope_per_prompt": slopes, "slope_mean": sum(slopes) / len(slopes)}
    return out
