"""
Measure the natural scale of the residual stream at each layer, so a
clamp value alpha can be reported as a fraction of the activation norm
rather than as a bare number.

Directions are unit-normalised, so "alpha = 12" means very different
things at layer 3 and layer 27 of an 8B model (residual norms grow with
depth). The camera-ready tables should report alpha / mean||h|| per
layer; stage3_run.py, stage5_run.py and stage7_run.py read the JSON this
script writes (--activation_norms) and record that ratio in their
results.

Usage:
    python activation_scale.py                       # real model, FineWeb contexts
    python activation_scale.py --dry_run             # tiny GPT-2, no downloads
    python activation_scale.py --layers 3 7 11 15 19 23 27 --n_contexts 50
"""

import argparse
import json
import os
import statistics

import torch

from config import CFG
from model_utils import get_transformer_layers


@torch.no_grad()
def residual_norm_stats(model, tokenizer, layers, contexts, device, use_chat_template=False):
    """
    Per layer: mean/median/std of ||h|| over every token position of every context
    (positions are pooled; the first token is excluded because the attention-sink position
    has a much larger norm in most LLMs and would dominate the mean).
    """
    per_layer = {l: [] for l in layers}
    handles = []

    def make_hook(l):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            norms = h[0].float().norm(dim=-1)            # [seq]
            per_layer[l].extend(norms[1:].tolist() if norms.shape[0] > 1 else norms.tolist())
        return hook

    blocks = get_transformer_layers(model)
    for l in layers:
        handles.append(blocks[l].register_forward_hook(make_hook(l)))
    try:
        for text in contexts:
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    out = {}
    for l in layers:
        vals = per_layer[l]
        out[str(l)] = {
            "mean_norm": sum(vals) / len(vals),
            "median_norm": statistics.median(vals),
            "std_norm": statistics.pstdev(vals),
            "n_positions": len(vals),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Defaults to CFG.sae_layers.")
    parser.add_argument("--n_contexts", type=int, default=50)
    parser.add_argument("--out", default="activation_norms.json")
    args = parser.parse_args()

    device = CFG.device if torch.cuda.is_available() else "cpu"
    if args.dry_run:
        from transformers import GPT2LMHeadModel, GPT2Config
        from tests.mock_model import MockGenerativeTokenizer
        model = GPT2LMHeadModel(GPT2Config(n_layer=5, n_embd=32, n_head=2, vocab_size=200)).to(device).eval()
        tokenizer = MockGenerativeTokenizer(vocab_size=200)
        layers = args.layers or [2, 4]
        contexts = [f"Sample background context number {i}, long enough to tokenize." for i in range(6)]
    else:
        from build_direction_bank import load_real_model
        from data import load_context_corpus
        model, tokenizer = load_real_model(CFG.model_name, device, CFG.dtype)
        layers = args.layers or CFG.sae_layers
        contexts = load_context_corpus(args.n_contexts)

    stats = residual_norm_stats(model, tokenizer, layers, contexts, device)
    record = {"model_name": CFG.model_name if not args.dry_run else "dry_run_gpt2", "per_layer": stats}
    out_path = args.out if not args.dry_run else "activation_norms_dryrun.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    for l in layers:
        print(f"  layer {l}: mean ||h|| = {stats[str(l)]['mean_norm']:.2f}")
    print(f"Saved {out_path}. Pass it to stage3/5/7 with --activation_norms to log alpha/||h||.")


if __name__ == "__main__":
    main()
