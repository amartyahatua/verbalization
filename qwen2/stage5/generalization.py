"""
Stage 5 generalization tests.

"Unseen directions", "unseen coefficients", and "unseen positions" are
all instances of the SAME underlying question -- does the trained
adapter's correlated-condition mapping hold at combinations it never
saw during training -- so they share one function
(evaluate_correlated), called with different (pairs, layers, alpha)
combinations from stage5_run.py:
  - unseen directions  = held_out_pairs  x  train_layers  x  train_alpha
  - unseen positions    = train_pairs     x  held_out_layers x  train_alpha
  - unseen coefficients = train_pairs     x  train_layers    x  alt_alphas

The decorrelated-cell test reuses Stage 3's decorrelation.py directly
(same bipolar direction construction, same injection-then-report
prompt, same direction-following index) -- the only difference is the
model passed in now has the trained adapter active. This is
deliberate: Stage 3 established what DFI looks like with NO training;
Stage 5 asks whether training changes that, without training ever
having seen conflict cells.
"""

import torch
from typing import List, Tuple

# --- make stage2/ and stage3/ importable from here ---
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage3"))

from elicitation import elicit_reports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))
from scoring import exact_match_score, score_reports
from decorrelation import (build_bipolar_direction, build_input_content, build_cell_prompt,
                           direction_following_index, dfi_bootstrap)
from protocol import alpha_over_norm


def generate_over_texts(model, tokenizer, layer, direction, alpha, texts: List[str], elicitation_prompt: str,
                        device: str, n_trials: int, max_new_tokens: int, apply_hook: bool = True) -> List[str]:
    """`n_trials` generations for EACH input text, pooled in text order."""
    reports = []
    for text in texts:
        reports += elicit_reports(model, tokenizer, layer, direction, alpha, f"{text}\n{elicitation_prompt}", device,
                                  n_trials=n_trials, max_new_tokens=max_new_tokens, apply_hook=apply_hook)
    return reports


def evaluate_correlated(
    model, tokenizer, vectors, pairs: List[Tuple[str, str]], layers: List[int],
    alpha: float, elicitation_prompt: str, device: str, n_trials: int = 3, max_new_tokens: int = 10,
    candidate_labels: List[str] = None, activation_norms=None,
) -> List[dict]:
    """
    candidate_labels: the full set of labels the trained adapter could
    plausibly recite (e.g. every pair it was trained on) -- passed to
    exact_match_score's shotgun-listing check. Defaults to just the
    labels in `pairs` if not given; pass the broader set explicitly
    when evaluating a trained adapter that might hedge across pairs
    outside the ones being tested here specifically.
    """
    if candidate_labels is None:
        candidate_labels = list({label for pair in pairs for label in pair})

    results = []
    for (x, y) in pairs:
        for layer in layers:
            key_x, key_y = f"jspace::{layer}::{x}", f"jspace::{layer}::{y}"
            if key_x not in vectors or key_y not in vectors:
                continue
            d_xy = build_bipolar_direction(vectors[key_x], vectors[key_y])

            for label, sign in ((x, 1.0), (y, -1.0)):
                prompt = f"{build_input_content(label)}\n{elicitation_prompt}"
                reports = elicit_reports(model, tokenizer, layer, d_xy, sign * alpha, prompt, device,
                                          n_trials=n_trials, max_new_tokens=max_new_tokens)
                counts = score_reports(reports, label, candidate_labels=candidate_labels)
                results.append({
                    "pair": [x, y], "layer": layer, "alpha": sign * alpha, "target_label": label,
                    "alpha_over_resid_norm": alpha_over_norm(alpha, layer, activation_norms or {}),
                    "match_rate": counts["rate"], "counts": counts, "reports": reports,
                })
    return results


def evaluate_decorrelated(
    model, tokenizer, vectors, pairs: List[Tuple[str, str]], layers: List[int],
    alpha: float, elicitation_prompt: str, device: str, n_trials: int = 3, max_new_tokens: int = 10,
    candidate_labels: List[str] = None, activation_norms=None, n_boot: int = 2000, boot_seed: int = 0,
) -> List[dict]:
    if candidate_labels is None:
        candidate_labels = list({label for pair in pairs for label in pair})

    results = []
    for (x, y) in pairs:
        for layer in layers:
            key_x, key_y = f"jspace::{layer}::{x}", f"jspace::{layer}::{y}"
            if key_x not in vectors or key_y not in vectors:
                continue
            d_xy = build_bipolar_direction(vectors[key_x], vectors[key_y])

            prompt_input_y = build_cell_prompt(y, elicitation_prompt)  # direction=X, input=Y
            prompt_input_x = build_cell_prompt(x, elicitation_prompt)  # direction=Y, input=X

            reports_dirX_inputY = elicit_reports(model, tokenizer, layer, d_xy, alpha, prompt_input_y, device,
                                                  n_trials=n_trials, max_new_tokens=max_new_tokens)
            reports_dirY_inputX = elicit_reports(model, tokenizer, layer, d_xy, -alpha, prompt_input_x, device,
                                                  n_trials=n_trials, max_new_tokens=max_new_tokens)

            dfi = direction_following_index(reports_dirX_inputY, reports_dirY_inputX, x, y, all_labels=candidate_labels)
            results.append({
                "pair": [x, y], "layer": layer, "alpha": alpha,
                "alpha_over_resid_norm": alpha_over_norm(alpha, layer, activation_norms or {}),
                "reports_dirX_inputY": reports_dirX_inputY, "reports_dirY_inputX": reports_dirY_inputX,
                **dfi,
                "dfi_bootstrap": dfi_bootstrap(reports_dirX_inputY, reports_dirY_inputX, x, y,
                                                all_labels=candidate_labels, n_boot=n_boot, seed=boot_seed),
            })
    return results
