"""
Diagnostic for match_rate=0: prints actual elicited report text for a
few J-space directions, and checks whether alpha is proportionate to
the model's actual activation scale at those layers.

Usage: python diagnose_stage2.py --results stage2_results/elicitation_results.json
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))
from config import CFG
from build_direction_bank import load_real_model
from data import load_context_corpus
from model_utils import get_transformer_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="stage2_results/elicitation_results.json")
    parser.add_argument("--n_samples", type=int, default=5)
    args = parser.parse_args()

    results = json.load(open(args.results))
    jspace = [r for r in results if r["class"] == "jspace"][: args.n_samples]

    print("=" * 70)
    print("SAMPLE REPORT TEXT (raw, unfiltered)")
    print("=" * 70)
    for r in jspace:
        print(f"\n--- {r['direction_key']} (target concept: '{r['target_concept']}') ---")
        for trial in r["trials"][:2]:
            print(f"  trial {trial['trial']}: {trial['report']!r}")

    print()
    print("=" * 70)
    print("ACTIVATION SCALE CHECK: is alpha proportionate?")
    print("=" * 70)
    device = CFG.device if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_real_model(CFG.model_name, device, CFG.dtype)
    contexts = load_context_corpus(5)

    for layer in sorted({r["layer"] for r in jspace}):
        norms = []
        captured = {}

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["h"] = h.detach()
            return out

        handle = get_transformer_layers(model)[layer].register_forward_hook(hook)
        try:
            for text in contexts:
                inputs = tokenizer(text, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    model(**inputs)
                norms.append(captured["h"][0, -1].float().norm().item())
        finally:
            handle.remove()

        avg_norm = sum(norms) / len(norms)
        layer_alpha = CFG.stage2_alphas.get(layer)  # this diagnostic already loops per-layer,
                                                      # so a per-layer dict lookup is the natural fit --
                                                      # .get() rather than [] since a results file could
                                                      # in principle reference a layer no longer calibrated
        alpha_str = f"{layer_alpha}" if layer_alpha is not None else "NOT CALIBRATED for this layer"
        print(f"layer {layer}: avg residual-stream norm = {avg_norm:.2f}   "
              f"(current stage2_alphas[{layer}] = {alpha_str})")
        if layer_alpha is not None and avg_norm > layer_alpha * 3:
            print(f"  -> alpha looks small relative to this layer's typical scale; "
                  f"consider raising it toward ~{avg_norm:.0f}")


if __name__ == "__main__":
    main()