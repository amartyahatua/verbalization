"""
Stage 6 metrics: aggregates the raw per-entry data already produced
by Stages 3-5 into the five summary metrics from the plan. Nothing
here touches a model -- it's pure aggregation over saved JSON, so the
"testing" for this module is running it against real Stage 3/4/5
output rather than a tiny model architecture check.

Handles both Stage 3's decorrelated-cell format (nested under "raw"/
"orthogonalized") and Stage 5's (flat, same field names otherwise) --
both were built from the same decorrelation.py functions, so their
per-entry schema is identical apart from that one layer of nesting.
"""

import os
import sys
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))
from protocol import bootstrap_ci, wilson_interval


def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _flatten_stage3(stage3_results: List[dict], condition: str = "raw") -> List[dict]:
    """Stage 3 nests each entry under raw/orthogonalized; pull one condition out flat."""
    # Entries that lack the requested condition are skipped (e.g. 'unembed_projected' for a pair whose direction
    # collapsed on projection -- that is reported in the entry's own `unembed_diagnostics`, not silently averaged in).
    skip_keys = ("raw", "orthogonalized", "unembed_projected")
    return [{**{k: v for k, v in e.items() if k not in skip_keys}, **e[condition]}
            for e in stage3_results if condition in e]


def accuracy_vs_direction_and_input(decorrelated_entries: List[dict]) -> Dict:
    """
    Metric 1: report-match accuracy vs. direction, and vs. input,
    scored separately -- averaged over both conflict cells in each
    entry (direction=X/input=Y and direction=Y/input=X).
    """
    dir_scores, input_scores = [], []
    for e in decorrelated_entries:
        dir_scores += [e.get("p_dirX_matches_X"), e.get("p_dirY_matches_Y")]
        input_scores += [e.get("p_dirX_matches_Y_input"), e.get("p_dirY_matches_X_input")]
    out = {
        "accuracy_vs_direction": _mean(dir_scores),
        "accuracy_vs_input": _mean(input_scores),
        "n_entries": len(decorrelated_entries),
    }
    # Pooled exact counts (entries produced by the camera-ready code carry them). The means above are means of
    # per-entry rates; these are the integers behind them.
    pooled = {"direction": [0, 0], "input": [0, 0]}
    have_counts = False
    for e in decorrelated_entries:
        c = e.get("counts")
        if not c:
            continue
        have_counts = True
        for key, bucket in (("dirX_matches_X", "direction"), ("dirY_matches_Y", "direction"),
                            ("dirX_matches_Y_input", "input"), ("dirY_matches_X_input", "input")):
            pooled[bucket][0] += c[key]["k"]
            pooled[bucket][1] += c[key]["n"]
    if have_counts:
        out["pooled_counts"] = {b: {"k": k, "n": n, "wilson95": list(wilson_interval(k, n))}
                                for b, (k, n) in pooled.items()}
    return out


def direction_following_index_summary(decorrelated_entries: List[dict]) -> Dict:
    """Metric 2: direction-following index, aggregated."""
    vals = [e.get("dfi_overall") for e in decorrelated_entries]
    vals = [v for v in vals if v is not None]
    boot = bootstrap_ci(vals, n_boot=2000, seed=0)
    return {
        "dfi_mean": _mean(vals),
        "dfi_min": min(vals) if vals else None,
        "dfi_max": max(vals) if vals else None,
        "n_entries": len(vals),
        # 95% percentile interval from resampling ENTRIES (pair x layer cells). It reflects which cells were
        # sampled, not training-seed variance -- for that, run several adapter seeds and use aggregate_seeds.py.
        "dfi_ci95_over_entries": [boot["ci_low"], boot["ci_high"]],
        "n_generations": sum(e.get("n_generations", 0) for e in decorrelated_entries),
    }


def generalization_gap(correlated_entries: List[dict], decorrelated_entries: List[dict]) -> Dict:
    """
    Metric 3: correlated-train -> decorrelated-test generalization gap.
    correlated_entries: anything with a "match_rate" field (Stage 4's
    training_eval.json, or any of Stage 5's evaluate_correlated outputs).
    The gap is the drop from "accuracy under training-like (agreeing)
    conditions" to "accuracy tracking the DIRECTION specifically once
    decorrelated" -- a large gap is exactly the shortcut-learning
    signature: strong in-sample-looking performance that isn't
    actually about the direction at all.
    """
    correlated_accuracy = _mean([e.get("match_rate") for e in correlated_entries])
    decorrelated_direction_accuracy = accuracy_vs_direction_and_input(decorrelated_entries)["accuracy_vs_direction"]
    gap = (None if correlated_accuracy is None or decorrelated_direction_accuracy is None
           else correlated_accuracy - decorrelated_direction_accuracy)
    return {
        "correlated_accuracy": correlated_accuracy,
        "decorrelated_direction_accuracy": decorrelated_direction_accuracy,
        "gap": gap,
    }


def jacobian_alignment_change_summary(mechanism_check_entries: List[dict]) -> Dict:
    """Metric 4: pre/post-training Jacobian alignment change."""
    before = [e["jacobian_alignment"]["before_training"] for e in mechanism_check_entries]
    after = [e["jacobian_alignment"]["after_training"] for e in mechanism_check_entries]
    delta = [e["jacobian_alignment"]["delta"] for e in mechanism_check_entries]
    n_increased = sum(1 for d in delta if d is not None and d > 0)
    return {
        "before_mean": _mean(before), "after_mean": _mean(after), "delta_mean": _mean(delta),
        "n_increased": n_increased, "n_total": len(delta),
    }


def ablation_summary(mechanism_check_entries: List[dict]) -> Dict:
    """Metric 5: report accuracy under direction-ablation vs. input-ablation."""
    baseline = [e["ablation"]["baseline"]["match_rate"] for e in mechanism_check_entries]
    ablate_dir = [e["ablation"]["ablate_direction"]["match_rate"] for e in mechanism_check_entries]
    ablate_inp = [e["ablation"]["ablate_input"]["match_rate"] for e in mechanism_check_entries]
    n_dir_necessary = sum(1 for e in mechanism_check_entries if e["ablation"]["direction_necessary"])
    n_input_necessary = sum(1 for e in mechanism_check_entries if e["ablation"]["input_necessary"])
    n = len(mechanism_check_entries)
    no_hook = [e["ablation"].get("no_hook_control", {}).get("match_rate") for e in mechanism_check_entries]

    def pooled(cond):
        k = nn = 0
        for e in mechanism_check_entries:
            c = e["ablation"].get(cond, {}).get("counts")
            if c:
                k += c["k"]
                nn += c["n"]
        return {"k": k, "n": nn} if nn else None

    return {
        "no_hook_control_mean": _mean(no_hook),
        "pooled_counts": {c: pooled(c) for c in ("baseline", "ablate_direction", "ablate_input", "no_hook_control")},
        "baseline_mean": _mean(baseline),
        "ablate_direction_mean": _mean(ablate_dir),
        "ablate_input_mean": _mean(ablate_inp),
        "direction_necessary_fraction": (n_dir_necessary / n) if n else None,
        "input_necessary_fraction": (n_input_necessary / n) if n else None,
        "n_total": n,
    }


def compute_all_metrics(stage3_results: List[dict], stage4_training_eval: List[dict], stage5_results: dict,
                         stage3_condition: str = "raw") -> Dict:
    """
    Ties everything together: pre-training numbers come from Stage 3
    (base model, no adapter); post-training numbers come from Stage 5
    (trained adapter). This pre/post split is itself the point --
    Stage 6 exists to make that comparison legible in one place.
    """
    stage3_flat = _flatten_stage3(stage3_results, stage3_condition)
    stage5_decorrelated = stage5_results["decorrelated"]

    return {
        "pre_training": {
            "accuracy_vs_direction_and_input": accuracy_vs_direction_and_input(stage3_flat),
            "direction_following_index": direction_following_index_summary(stage3_flat),
        },
        "post_training": {
            "accuracy_vs_direction_and_input": accuracy_vs_direction_and_input(stage5_decorrelated),
            "direction_following_index": direction_following_index_summary(stage5_decorrelated),
        },
        "generalization_gap": generalization_gap(stage4_training_eval, stage5_decorrelated),
        "jacobian_alignment_change": jacobian_alignment_change_summary(stage5_results["mechanism_check"]),
        "ablation": ablation_summary(stage5_results["mechanism_check"]),
    }
