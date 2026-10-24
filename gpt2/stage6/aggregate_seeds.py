"""
Aggregate results across adapter seeds (camera-ready statistics).

One LoRA adapter is one draw of the training procedure. Reviewers asked for three seeds and intervals on DFI,
because several cells are tiny and a single adapter can't distinguish "the method fails" from "this run failed".
Run Stage 4 + Stage 5 (or Stage 7) once per --seed, then:

    python aggregate_seeds.py --kind stage5 \
        runs/seed0/stage5_results.json runs/seed1/stage5_results.json runs/seed2/stage5_results.json
    python aggregate_seeds.py --kind stage7 \
        runs/s0/stage7_results.json runs/s1/stage7_results.json runs/s2/stage7_results.json

What is reported, and what each interval means:
  * per-seed values (always show these -- with 3 seeds they ARE the result),
  * mean, SD and range across seeds,
  * a hierarchical percentile bootstrap (resample seeds, then entries within the chosen seed). With only a few
    seeds that interval is optimistic about between-seed variance; treat the range as the honest summary.
"""

import argparse
import json
import os
import random
import statistics
import sys



def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def hierarchical_bootstrap(per_seed_entries, n_boot=5000, seed=0):
    """per_seed_entries: list (seeds) of lists (entry values). Resample seeds, then entries within each."""
    rng = random.Random(seed)
    data = [[v for v in entries if v is not None] for entries in per_seed_entries]
    data = [d for d in data if d]
    if not data:
        return None
    boots = []
    for _ in range(n_boot):
        picked = [data[rng.randrange(len(data))] for _ in range(len(data))]
        means = [sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for d in picked]
        boots.append(sum(means) / len(means))
    boots.sort()
    return [boots[int(0.025 * n_boot)], boots[min(n_boot - 1, int(0.975 * n_boot))]]


def summarize(name, per_seed_entries):
    per_seed = [_mean(e) for e in per_seed_entries]
    vals = [v for v in per_seed if v is not None]
    if not vals:
        return {"name": name, "per_seed": per_seed, "n_seeds": 0}
    return {
        "name": name, "per_seed": per_seed, "n_seeds": len(vals),
        "mean": sum(vals) / len(vals),
        "sd": statistics.stdev(vals) if len(vals) > 1 else None,
        "min": min(vals), "max": max(vals),
        "hierarchical_bootstrap_ci95": hierarchical_bootstrap(per_seed_entries),
        "n_entries_per_seed": [len([x for x in e if x is not None]) for e in per_seed_entries],
    }


def stage5_metrics(runs):
    out = []
    for section in ("unseen_directions", "unseen_positions", "unseen_coefficients"):
        out.append(summarize(f"{section} match_rate", [[e.get("match_rate") for e in r.get(section, [])] for r in runs]))
    out.append(summarize("decorrelated DFI", [[e.get("dfi_overall") for e in r["decorrelated"]] for r in runs]))
    out.append(summarize("decorrelated: P(report=direction)",
                         [[v for e in r["decorrelated"] for v in (e.get("p_dirX_matches_X"), e.get("p_dirY_matches_Y"))]
                          for r in runs]))
    out.append(summarize("decorrelated: P(report=input)",
                         [[v for e in r["decorrelated"] for v in (e.get("p_dirX_matches_Y_input"), e.get("p_dirY_matches_X_input"))]
                          for r in runs]))
    for cond in ("baseline", "ablate_direction", "ablate_input", "no_hook_control"):
        out.append(summarize(f"ablation {cond} match_rate",
                             [[(e["ablation"].get(cond) or {}).get("match_rate") for e in r["mechanism_check"]] for r in runs]))
    return out


def stage7_metrics(runs):
    out = [summarize("capstone DFI (held-out phrasings)", [[r["decorrelation"].get("dfi_overall")] for r in runs]),
           summarize("capstone P(report=direction)",
                     [[r["decorrelation"].get("p_dirX_matches_X"), r["decorrelation"].get("p_dirY_matches_Y")] for r in runs]),
           summarize("capstone P(report=input)",
                     [[r["decorrelation"].get("p_dirX_matches_Y_input"), r["decorrelation"].get("p_dirY_matches_X_input")]
                      for r in runs])]
    for tag in ("ablation_injection_heldout", "ablation_safe_heldout"):
        for cond in ("baseline", "ablate_direction", "ablate_input", "no_hook_control"):
            out.append(summarize(f"{tag} {cond}", [[r[tag][cond]["match_rate"]] for r in runs]))
    return out


def fmt(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def to_markdown(rows, files):
    lines = [f"## Seed aggregate ({len(files)} runs)\n", "Runs: " + ", ".join(files) + "\n",
             "| metric | per-seed | mean | SD | range | hier. bootstrap 95% |", "|---|---|---|---|---|---|"]
    for r in rows:
        per = ", ".join(fmt(v) for v in r["per_seed"])
        ci = r.get("hierarchical_bootstrap_ci95")
        lines.append(f"| {r['name']} | {per} | {fmt(r.get('mean'))} | {fmt(r.get('sd'))} | "
                     f"[{fmt(r.get('min'))}, {fmt(r.get('max'))}] | "
                     f"{'n/a' if not ci else f'[{fmt(ci[0])}, {fmt(ci[1])}]'} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="One results JSON per seed.")
    parser.add_argument("--kind", choices=["stage5", "stage7"], required=True)
    parser.add_argument("--out_dir", default="seed_aggregate")
    args = parser.parse_args()

    runs = [json.load(open(f)) for f in args.files]
    rows = stage5_metrics(runs) if args.kind == "stage5" else stage7_metrics(runs)
    md = to_markdown(rows, args.files)
    print(md)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{args.kind}_seed_aggregate.json"), "w") as f:
        json.dump({"files": args.files, "rows": rows}, f, indent=2)
    with open(os.path.join(args.out_dir, f"{args.kind}_seed_aggregate.md"), "w") as f:
        f.write(md)
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
