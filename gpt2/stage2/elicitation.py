"""
Stage 2 elicitation: clamp a direction to alpha, generate a free-text
report with a fixed elicitation prompt, repeated n_trials times so a
report-match RATE (not just one binary outcome) can be computed per
direction -- which is what the mixed-effects regression needs, since
"direction identity as random effect" implies repeated trials per
direction.
"""

import torch
from typing import List
from editing import apply_direction_editing


def elicit_reports(
    model,
    tokenizer,
    layer: int,
    direction: torch.Tensor,
    alpha: float,
    prompt: str,
    device: str,
    n_trials: int = 5,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    apply_hook: bool = True,
) -> List[str]:
    """
    apply_hook=False skips the editing hook entirely (no intervention at all) -- distinct from
    alpha=0.0, which still actively projects the direction's component out and sets it to zero.
    Used by Stage 5's no-hook control; defaults to True so every existing call is unchanged.
    (No chat template in this folder: GPT-2 is a base model, the prompt is fed as-is.)
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
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=eos_id,
                )
        finally:
            if handle is not None:
                handle.remove()
        new_tokens = out_ids[0][prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        reports.append(text.strip())
    return reports
