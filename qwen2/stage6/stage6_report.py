"""
Stage 6 entry point: loads Stage 3's decorrelated results (pre-
training baseline), Stage 4's training_eval.json (in-sample
correlated accuracy), and Stage 5's results (post-training
generalization + mechanism check), computes all five metrics, and
saves both a JSON and a markdown table ready to drop into the paper.

Deliberately does NOT generate a preregistration -- see the project
README for why: preregistration only does its job before you've seen
results, and by the time this stage runs you've already fully
analyzed Stages 3-5. Writing "predictions" now would be post-hoc
rationalization labeled as something it isn't. If/when a genuine
confirmatory follow-up run happens (more directions, a different or
larger model), that's the point to write a real preregistration --
happy to draft one at that point.

Usage:
    python stage6_report.py \
        --stage3 ../stage3/stage3_results/stage3_results.json \
        --stage4_eval ../stage4/stage4_adapter/training_eval.json \
        --stage5 ../stage5/stage5_results/stage5_results.json
"""

import argparse
import json
import os

from metrics import compute_all_metrics


def format_pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def format_num(x, decimals=4):
    return "n/a" if x is None else f"{x:.{decimals}f}"


def format_kn(c):
    """'k/n' from a {'k','n'} dict, or '-' when the results predate exact counts."""
    return "-" if not c else f"{c['k']}/{c['n']}"


def _row(label, m):
    acc = m["accuracy_vs_direction_and_input"]
    dfi = m["direction_following_index"]
    pc = acc.get("pooled_counts", {})
    ci = dfi.get("dfi_ci95_over_entries") or [None, None]
    return (f"| {label} | {format_pct(acc['accuracy_vs_direction'])} ({format_kn(pc.get('direction'))}) "
            f"| {format_pct(acc['accuracy_vs_input'])} ({format_kn(pc.get('input'))}) "
            f"| {format_num(dfi['dfi_mean'], 2)} | [{format_num(ci[0], 2)}, {format_num(ci[1], 2)}] "
            f"| [{format_num(dfi['dfi_min'], 2)}, {format_num(dfi['dfi_max'], 2)}] "
            f"| {acc['n_entries']} | {dfi.get('n_generations', 0)} |")


def to_markdown(m: dict) -> str:
    pre = m["pre_training"]
    post = m["post_training"]
    gap = m["generalization_gap"]
    jac = m["jacobian_alignment_change"]
    ab = m["ablation"]

    lines = []
    lines.append("## Stage 6 metrics summary\n")

    lines.append("### 1-2. Accuracy vs. direction / vs. input, and direction-following index\n")
    lines.append("| | vs. direction (k/n) | vs. input (k/n) | DFI (mean) | DFI 95% CI over entries | DFI (range) | entries | generations |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append(_row("Pre-training (Stage 3, base model)", pre))
    lines.append(_row("Post-training (Stage 5, trained adapter)", post))

    lines.append("\n### 3. Correlated-train -> decorrelated-test generalization gap\n")
    lines.append(f"- Correlated (in-sample, training-like) accuracy: **{format_pct(gap['correlated_accuracy'])}**")
    lines.append(f"- Decorrelated direction-tracking accuracy: **{format_pct(gap['decorrelated_direction_accuracy'])}**")
    lines.append(f"- Gap: **{format_pct(gap['gap'])}** (large gap = shortcut-learning signature: "
                  f"strong in-sample performance not actually explained by the direction)")

    lines.append("\n### 4. Pre/post-training Jacobian alignment change\n")
    lines.append(f"- Before training: {format_num(jac['before_mean'])}")
    lines.append(f"- After training: {format_num(jac['after_mean'])}")
    lines.append(f"- Mean delta: {format_num(jac['delta_mean'])}")
    lines.append(f"- Increased in {jac['n_increased']}/{jac['n_total']} tested combinations")

    lines.append("\n### 5. Report accuracy under direction-ablation vs. input-ablation\n")
    lines.append("| condition | match rate |")
    lines.append("|---|---|")
    pc = ab.get("pooled_counts", {})
    lines.append(f"| baseline (no ablation) | {format_pct(ab['baseline_mean'])} ({format_kn(pc.get('baseline'))}) |")
    lines.append(f"| direction ablated (alpha=0, hook active) | {format_pct(ab['ablate_direction_mean'])} ({format_kn(pc.get('ablate_direction'))}) |")
    lines.append(f"| input ablated (bare elicitation prompt) | {format_pct(ab['ablate_input_mean'])} ({format_kn(pc.get('ablate_input'))}) |")
    lines.append(f"| no-hook control (hook absent) | {format_pct(ab.get('no_hook_control_mean'))} ({format_kn(pc.get('no_hook_control'))}) |")
    lines.append(f"\n- Direction necessary in {format_pct(ab['direction_necessary_fraction'])} of tested combinations")
    lines.append(f"- Input necessary in {format_pct(ab['input_necessary_fraction'])} of tested combinations")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3", default="../stage3/stage3_results/stage3_results.json")
    parser.add_argument("--stage4_eval", default="../stage4/stage4_adapter/training_eval.json")
    parser.add_argument("--stage5", default="../stage5/stage5_results/stage5_results.json")
    parser.add_argument("--stage3_condition", choices=["raw", "orthogonalized", "unembed_projected"], default="raw",
                         help="Which Stage 3 confound-(b) condition to use as the pre-training baseline.")
    parser.add_argument("--out_dir", default="stage6_report")
    args = parser.parse_args()

    stage3 = json.load(open(args.stage3))
    stage4_eval = json.load(open(args.stage4_eval))
    stage5 = json.load(open(args.stage5))

    metrics = compute_all_metrics(stage3, stage4_eval, stage5, stage3_condition=args.stage3_condition)
    markdown = to_markdown(metrics)

    print(markdown)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "stage6_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out_dir, "stage6_summary.md"), "w") as f:
        f.write(markdown)
    print(f"\n\nSaved stage6_metrics.json and stage6_summary.md to {args.out_dir}/")


if __name__ == "__main__":
    main()
