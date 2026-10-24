"""
Stage 1 entry point: build the full direction bank (J-space, SAE,
probe, and random directions), tag every entry with a causal-efficacy
score, and save it to disk for use in Stage 2 (verbalizability
elicitation) and Stage 3 (decorrelation test).

Usage:
    python build_direction_bank.py               # full run, real model + real data
    python build_direction_bank.py --dry_run      # sanity-check the pipeline with a
                                                    # tiny mock model, no downloads,
                                                    # no GPU required
"""

import argparse
import json
import os

import torch

from config import CFG
from jspace_lens import build_jspace_directions
from sae_features import stratified_feature_sample, get_sae_directions
from probes import build_probe_directions
from random_directions import build_random_directions
from causal_efficacy import tag_causal_efficacy


def load_real_model(model_name: str, device: str, torch_dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    model.eval()
    return model, tokenizer


def build_bank(model, tokenizer, sae_by_layer: dict, device: str, contexts, counterfact) -> dict:
    bank = {}

    # 1. J-space directions (native positive controls)
    jspace = build_jspace_directions(
        model, tokenizer, CFG.sae_layers, CFG.jlens_target_tokens, contexts, device
    )
    for (layer, tok), vec in jspace.items():
        bank[f"jspace::{layer}::{tok}"] = {"layer": layer, "vector": vec, "class": "jspace"}

    # 2. SAE feature directions, stratified by abstraction
    for layer in CFG.sae_layers:
        sae = sae_by_layer[layer]
        feat_idx = stratified_feature_sample(sae, CFG.n_sae_features_per_layer)
        for fid, vec in get_sae_directions(sae, feat_idx).items():
            bank[f"sae::{layer}::{fid}"] = {"layer": layer, "vector": vec, "class": "sae"}

    # 3. Probe directions (known attributes; some causally inert by design)
    probes = build_probe_directions(
        model, tokenizer, counterfact, CFG.sae_layers, device, CFG.n_probe_attributes
    )
    for (layer, name), vec in probes.items():
        bank[f"probe::{layer}::{name}"] = {"layer": layer, "vector": vec, "class": "probe"}

    # 4. Random directions, norm-matched to the classes collected so far
    ref_norms = [v["vector"].norm().item() for v in bank.values()] or [1.0]
    for layer in CFG.sae_layers:
        W_dec = sae_by_layer[layer].W_dec.detach().cpu()
        rand = build_random_directions(
            W_dec.shape[1], W_dec, ref_norms,
            CFG.n_random_directions_per_layer, CFG.n_random_in_span_per_layer,
        )
        for name, vec in rand.items():
            cls = "random" if name.startswith("random::") else "random_in_span"
            bank[f"{cls}::{layer}::{name}"] = {"layer": layer, "vector": vec, "class": cls}

    # 5. Tag every direction with a causal-efficacy score
    eval_contexts = contexts[: CFG.causal_eval_n_contexts]
    bank = tag_causal_efficacy(model, tokenizer, bank, eval_contexts, CFG.steer_alphas, device)

    return bank


def save_bank(bank: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    vectors = {k: v["vector"] for k, v in bank.items()}
    torch.save(vectors, os.path.join(out_dir, "vectors.pt"))

    metadata = {
        k: {"layer": v["layer"], "class": v["class"], "causal_efficacy": v.get("causal_efficacy")}
        for k, v in bank.items()
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(bank)} directions to {out_dir}/ (vectors.pt + metadata.json)")


def _apply_dry_run_overrides():
    """Shrink everything so the dry run finishes in seconds, not hours."""
    CFG.jlens_n_contexts = 20
    CFG.jlens_target_tokens = CFG.jlens_target_tokens[:2]
    CFG.sae_layers = CFG.sae_layers[:2]
    CFG.n_sae_features_per_layer = 5
    CFG.n_probe_attributes = 2
    CFG.n_random_directions_per_layer = 5
    CFG.n_random_in_span_per_layer = 5
    CFG.causal_eval_n_contexts = 5
    CFG.steer_alphas = [4.0]
    CFG.output_dir = "direction_bank_dryrun"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Run the full pipeline against a tiny mock model instead of downloading real weights.",
    )
    args = parser.parse_args()

    device = CFG.device if torch.cuda.is_available() else "cpu"

    if args.dry_run:
        _apply_dry_run_overrides()
        from tests.mock_model import build_mock_model_and_sae, mock_contexts_and_counterfact
        model, tokenizer, sae_by_layer = build_mock_model_and_sae(CFG.sae_layers, device)
        contexts, counterfact = mock_contexts_and_counterfact(n_contexts=CFG.jlens_n_contexts)
    else:
        from data import load_context_corpus, load_counterfact
        from sae_features import load_sae

        model, tokenizer = load_real_model(CFG.model_name, device, CFG.dtype)

        sae_by_layer = {}
        failed_layers = []
        for l in CFG.sae_layers:
            try:
                sae_by_layer[l] = load_sae(CFG.sae_release, l, device, sae_id_template=CFG.sae_id_template)
                print(f"  Loaded SAE for layer {l}")
            except Exception as e:
                failed_layers.append(l)
                # Print the actual message, not just the exception type -- a bare
                # "(ValueError)" or "(ModuleNotFoundError)" with no text tells you
                # THAT it failed but not WHY, which is what made this exact release/
                # sae_id_template mismatch slow to track down originally.
                print(f"  WARNING: no SAE available for layer {l} in release {CFG.sae_release!r} "
                      f"({type(e).__name__}: {str(e)[:300]}) -- skipping this layer. Not every "
                      f"layer is necessarily covered by a given SAE release; this is common, not "
                      f"a bug. But if EVERY layer fails with the same error, check CFG.sae_id_template "
                      f"against the release's actual sae_id scheme (see sae_features.py docstring).")

        if failed_layers:
            CFG.sae_layers = [l for l in CFG.sae_layers if l not in failed_layers]
            print(f"Continuing with layers that have an SAE: {CFG.sae_layers}")
        if not sae_by_layer:
            raise RuntimeError(
                f"No SAE could be loaded for release={CFG.sae_release!r} at ANY configured "
                f"layer. Check CFG.sae_release and CFG.sae_id_template are correct for this "
                f"release (see sae_features.py and the warnings printed above for the real error)."
            )

        print(f"Streaming {CFG.jlens_n_contexts} contexts from FineWeb (sample-10BT)...")
        contexts = load_context_corpus(CFG.jlens_n_contexts)
        print(f"Got {len(contexts)} contexts.")
        counterfact = load_counterfact()

    bank = build_bank(model, tokenizer, sae_by_layer, device, contexts, counterfact)
    save_bank(bank, CFG.output_dir)


if __name__ == "__main__":
    main()
