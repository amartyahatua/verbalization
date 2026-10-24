"""
Targeted diagnostic: sweep alpha across a small set of values, on just
the J-space directions (cheap — 7 directions x n_alphas, not the full
835), to check whether ANY clamping strength produces reports that
actually differ by target concept. Complements diagnose_stage2.py
(which checks report text + activation scale at the alpha you already
ran); this checks whether a *different* alpha would help at all.

Also supports overriding the prompt and/or the model, for testing
whether a completion-style prompt or an instruction-tuned model
changes the picture -- see --prompt / --model_name / --recompute below.

Usage:
    # Default: sweep alpha on the existing bank's J-space directions
    python alpha_sweep.py --bank_dir ../stage1/direction_bank --layer 2

    # Try a different prompt on the SAME model (bank directions still valid)
    python alpha_sweep.py --layer 2 --prompt "The concept currently active is:"

    # Try a DIFFERENT model -- requires --recompute, since J-space directions
    # are tied to the model's hidden size and aren't transferable
    python alpha_sweep.py --layer 12 --model_name "Qwen/Qwen2-0.5B-Instruct" --recompute
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))
from config import CFG
from build_direction_bank import load_real_model
from jspace_lens import build_jspace_directions

from elicitation import elicit_reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank_dir", default="../stage1/direction_bank",
                         help="Load J-space directions from this existing bank. Ignored if --recompute is set.")
    parser.add_argument("--layer", type=int, default=7)  # must be one of CFG.sae_layers ([3,7,11,15,19,23,27])
                                                            # to have bank entries -- 11 is the ~34%-depth layer,
                                                            # closest match to the old config's layer-12 default
    parser.add_argument("--alphas", type=float, nargs="+", default=[4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0])
    parser.add_argument("--n_trials", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--prompt", default=None,
                         help="Override CFG.stage2_prompt, e.g. a completion-style or few-shot prompt.")
    parser.add_argument("--model_name", default=None,
                         help="Override CFG.model_name, e.g. to test an instruction-tuned model instead. "
                              "NOTE: a different model has a different hidden size, so directions loaded "
                              "from --bank_dir won't be compatible -- pass --recompute too when using this.")
    parser.add_argument("--recompute", action="store_true",
                         help="Recompute J-space directions fresh for the current model instead of loading "
                              "--bank_dir. Required whenever --model_name differs from the bank's original model.")
    parser.add_argument("--target_tokens", nargs="+", default=None,
                         help="Target concepts to sweep when --recompute is set. Defaults to CFG.jlens_target_tokens.")
    parser.add_argument("--n_contexts", type=int, default=20,
                         help="Contexts used for J-lens direction recomputation, only used with --recompute.")
    args = parser.parse_args()

    device = CFG.device if torch.cuda.is_available() else "cpu"
    model_name = args.model_name or CFG.model_name
    prompt = args.prompt or CFG.stage2_prompt

    if args.model_name and args.model_name != CFG.model_name and not args.recompute:
        print(f"WARNING: --model_name={args.model_name} differs from CFG.model_name={CFG.model_name}, "
              f"but --recompute wasn't passed. Directions loaded from --bank_dir were built for a "
              f"different model's hidden size and will almost certainly error or be meaningless here. "
              f"Pass --recompute too.\n")

    model, tokenizer = load_real_model(model_name, device, CFG.dtype)

    if args.recompute:
        target_tokens = args.target_tokens or CFG.jlens_target_tokens
        from data import load_context_corpus
        print(f"Recomputing J-space directions for {model_name} at layer {args.layer}, "
              f"targets={target_tokens}, using {args.n_contexts} contexts...")
        contexts = load_context_corpus(args.n_contexts)
        jspace = build_jspace_directions(model, tokenizer, [args.layer], target_tokens, contexts, device)
        directions_by_key = {f"jspace::{layer}::{tok}": vec for (layer, tok), vec in jspace.items()}
    else:
        vectors = torch.load(os.path.join(args.bank_dir, "vectors.pt"))
        metadata = json.load(open(os.path.join(args.bank_dir, "metadata.json")))
        directions_by_key = {
            k: vectors[k] for k, m in metadata.items()
            if m["class"] == "jspace" and m["layer"] == args.layer
        }

    print(f"\nSweeping {len(directions_by_key)} J-space directions at layer {args.layer} on {model_name}")
    print(f"prompt={prompt!r}")
    print(f"alphas={args.alphas}\n")

    for key, direction in directions_by_key.items():
        target = key.split("::")[-1]
        print(f"=== {key} (target concept: '{target}') ===")
        for alpha in args.alphas:
            reports = elicit_reports(
                model, tokenizer, args.layer, direction, alpha, prompt, device,
                n_trials=args.n_trials, max_new_tokens=args.max_new_tokens,
            )
            hit = any(target.lower() in r.lower() for r in reports)
            marker = "  <-- mentions target!" if hit else ""
            print(f"  alpha={alpha:6.1f}{marker}")
            for r in reports:
                print(f"    {r!r}")
        print()


if __name__ == "__main__":
    main()
