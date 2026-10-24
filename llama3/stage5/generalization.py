"""
Stage 5 generalization tests.

"Unseen directions", "unseen coefficients", and "unseen layers" are
all instances of the SAME underlying question -- does the trained
adapter's correlated-condition mapping hold at combinations it never
saw during training -- so they share one function
(evaluate_correlated), called with different (pairs, layers, alpha)
combinations from stage5_run.py:
  - unseen directions  = held_out_pairs  x  train_layers  x  train_alpha
  - unseen layers       = train_pairs     x  held_out_layers x  train_alpha
  - unseen coefficients = train_pairs     x  train_layers    x  alt_alphas
(The old name "unseen positions" is kept as a results key for compatibility, but what is varied is the
LAYER the direction is clamped at; token position is never varied independently of the layer.)

The decorrelated-cell test reuses Stage 3's decorrelation.py directly
(same bipolar direction construction, same injection-then-report
prompt, same direction-following index) -- the only difference is the
model passed in now has the trained adapter active. This is
deliberate: Stage 3 established what DFI looks like with NO training;
Stage 5 asks whether training changes that, without training ever
having seen conflict cells.

Camera-ready additions (all default to the previous behaviour):
  * `provider`   which input texts accompany the clamp (stage3/input_provider.py). With the `varied`
                 provider these are HELD-OUT passages that never name the concept; every text gets
                 `n_trials` generations and the reports are pooled, so n = n_texts * n_trials.
  * `direction_source` / `direction_seed`   J-lens bank vs the seeded random-direction baseline; must
                 match what Stage 4 trained on (stage5_run.py reads both from split.json).
  * every row carries exact k/n (`counts`), and every decorrelated row a generation-level DFI bootstrap.
"""

import torch
from typing import List, Tuple

# --- make stage2/ and stage3/ importable from here ---
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage3"))

from elicitation import elicit_reports
from scoring import score_reports
from decorrelation import direction_following_index, dfi_bootstrap
from input_provider import get_input_provider
from direction_source import pair_direction
from protocol import alpha_over_norm


def _resolve_alpha(alphas, layer):
    """
    alphas: EITHER a per-layer dict (CFG.stage2_alphas, as stored in Stage 4's split.json) OR a single float
    (args.alt_alphas' entries in stage5_run.py, or an old-format split.json). Same convention as
    stage4_train.build_training_examples -- a single alpha across a multi-layer loop is wrong in general
    (layers 3/7 collapse above ~2, layers 19-27 stay coherent to ~6-7).
    """
    if isinstance(alphas, dict):
        try:
            return alphas[layer]
        except KeyError:
            raise KeyError(
                f"No calibrated alpha entry for layer {layer!r} in the alphas dict "
                f"(from split.json or CFG.stage2_alphas). If CFG.sae_layers gained a "
                f"new layer, run alpha_sweep.py for it and add the entry."
            )
    return alphas


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
    alphas, elicitation_prompt: str, device: str, n_trials: int = 3, max_new_tokens: int = 10,
    candidate_labels: List[str] = None, provider=None, direction_source: str = "jspace",
    direction_seed: int = 0, activation_norms=None,
) -> List[dict]:
    """
    alphas: per-layer dict OR a single float -- see _resolve_alpha.

    candidate_labels: the full set of labels the trained adapter could
    plausibly recite (e.g. every pair it was trained on) -- passed to
    exact_match_score's shotgun-listing check. Defaults to just the
    labels in `pairs` if not given; pass the broader set explicitly
    when evaluating a trained adapter that might hedge across pairs
    outside the ones being tested here specifically.
    """
    provider = provider or get_input_provider("template")
    if candidate_labels is None:
        candidate_labels = list({label for pair in pairs for label in pair})

    results = []
    for (x, y) in pairs:
        for layer in layers:
            d_xy = pair_direction(vectors, x, y, layer, direction_source, direction_seed)
            if d_xy is None:
                continue
            alpha = _resolve_alpha(alphas, layer)

            for label, sign in ((x, 1.0), (y, -1.0)):
                texts = provider.eval_texts(label)
                reports = generate_over_texts(model, tokenizer, layer, d_xy, sign * alpha, texts,
                                              elicitation_prompt, device, n_trials, max_new_tokens)
                counts = score_reports(reports, label, candidate_labels=candidate_labels)
                results.append({
                    "pair": [x, y], "layer": layer, "alpha": sign * alpha, "target_label": label,
                    "alpha_over_resid_norm": alpha_over_norm(alpha, layer, activation_norms or {}),
                    "match_rate": counts["rate"], "counts": counts, "n_input_texts": len(texts),
                    "reports": reports,
                })
    return results


def evaluate_decorrelated(
    model, tokenizer, vectors, pairs: List[Tuple[str, str]], layers: List[int],
    alphas, elicitation_prompt: str, device: str, n_trials: int = 3, max_new_tokens: int = 10,
    candidate_labels: List[str] = None, provider=None, direction_source: str = "jspace",
    direction_seed: int = 0, activation_norms=None, n_boot: int = 2000, boot_seed: int = 0,
) -> List[dict]:
    """alphas: per-layer dict OR a single float -- see _resolve_alpha."""
    provider = provider or get_input_provider("template")
    if candidate_labels is None:
        candidate_labels = list({label for pair in pairs for label in pair})

    results = []
    for (x, y) in pairs:
        for layer in layers:
            d_xy = pair_direction(vectors, x, y, layer, direction_source, direction_seed)
            if d_xy is None:
                continue
            alpha = _resolve_alpha(alphas, layer)

            texts_y, texts_x = provider.eval_texts(y), provider.eval_texts(x)
            # direction=X, input=Y   and   direction=Y, input=X
            reports_dirX_inputY = generate_over_texts(model, tokenizer, layer, d_xy, alpha, texts_y,
                                                      elicitation_prompt, device, n_trials, max_new_tokens)
            reports_dirY_inputX = generate_over_texts(model, tokenizer, layer, d_xy, -alpha, texts_x,
                                                      elicitation_prompt, device, n_trials, max_new_tokens)

            dfi = direction_following_index(reports_dirX_inputY, reports_dirY_inputX, x, y, all_labels=candidate_labels)
            results.append({
                "pair": [x, y], "layer": layer, "alpha": alpha,
                "alpha_over_resid_norm": alpha_over_norm(alpha, layer, activation_norms or {}),
                "n_input_texts_per_cell": len(texts_x),
                "reports_dirX_inputY": reports_dirX_inputY, "reports_dirY_inputX": reports_dirY_inputX,
                **dfi,
                "dfi_bootstrap": dfi_bootstrap(reports_dirX_inputY, reports_dirY_inputX, x, y,
                                                all_labels=candidate_labels, n_boot=n_boot, seed=boot_seed),
            })
    return results
