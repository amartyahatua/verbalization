"""
Offline tests for the camera-ready additions (protocol counts, bootstrap, unembedding projection,
input providers, direction sources, seed aggregation). No downloads, no GPU:

    cd llama3 && python -m pytest tests -q
"""
import math
import os
import sys

import pytest
import torch

from scoring import exact_match_score, score_reports
from decorrelation import direction_following_index, dfi_bootstrap, build_bipolar_direction
from protocol import wilson_interval, bootstrap_ci, count_cell, alpha_over_norm, seed_everything
from confounds import project_out_span, project_out_unembedding, unembedding_vectors
from input_provider import (get_input_provider, TemplateInputs, VariedInputs, leaks_label,
                            TRAINING_INPUT_TEMPLATES, _VARIED)
from direction_source import pair_direction
from decorrelation import build_input_content


# ---------------------------------------------------------------- counts
def test_score_reports_matches_exact_match_and_counts():
    reports = ["Paris", "It is Rome.", "Paris Paris Paris Paris", "Paris and Rome", "nothing", "paris"]
    labels = ["Paris", "Rome"]
    c = score_reports(reports, "Paris", candidate_labels=labels)
    expected = [exact_match_score(r, "Paris", candidate_labels=labels) for r in reports]
    assert c["k"] == sum(expected) == 2           # "Paris" and "paris"
    assert c["n"] == c["n_total"] == 6
    assert c["n_degenerate"] == 1                  # the repetition loop
    assert c["n_multi_label"] == 1                 # "Paris and Rome"
    assert c["rate"] == pytest.approx(2 / 6)


def test_score_reports_no_target_is_unscorable():
    c = score_reports(["a", "b"], None)
    assert c["n"] == 0 and c["rate"] is None and c["n_total"] == 2


def test_dfi_counts_are_integers_behind_every_rate():
    # cell1 (direction=X, input=Y): 3 gens, 2 say X, 1 says Y.  cell2: 3 gens, all say Y.
    c1 = ["x", "x", "y"]
    c2 = ["y", "y", "y"]
    d = direction_following_index(c1, c2, "x", "y")
    assert d["counts"]["dirX_matches_X"]["k"] == 2 and d["counts"]["dirX_matches_X"]["n"] == 3
    assert d["counts"]["dirX_matches_Y_input"]["k"] == 1
    assert d["counts"]["dirY_matches_Y"]["k"] == 3
    assert d["counts"]["dirY_matches_X_input"]["k"] == 0
    assert d["n_generations"] == 6
    for key, cnt in (("p_dirX_matches_X", "dirX_matches_X"), ("p_dirY_matches_Y", "dirY_matches_Y")):
        assert d[key] == pytest.approx(d["counts"][cnt]["k"] / d["counts"][cnt]["n"])
    # DFI definition: 0.5 * [(2/3 - 1/3) + (3/3 - 0)]
    assert d["dfi_overall"] == pytest.approx(0.5 * ((2 / 3 - 1 / 3) + (1.0 - 0.0)))


def test_dfi_bootstrap_extremes_and_agreement_with_point_estimate():
    follows_direction = dfi_bootstrap(["x"] * 4, ["y"] * 4, "x", "y", n_boot=200)
    assert follows_direction["mean"] == pytest.approx(1.0) and follows_direction["ci_low"] == pytest.approx(1.0)
    follows_input = dfi_bootstrap(["y"] * 4, ["x"] * 4, "x", "y", n_boot=200)
    assert follows_input["mean"] == pytest.approx(-1.0) and follows_input["ci_high"] == pytest.approx(-1.0)
    mixed_c1, mixed_c2 = ["x", "y", "x", "y", "x"], ["y", "x", "y", "y", "x"]
    b = dfi_bootstrap(mixed_c1, mixed_c2, "x", "y", n_boot=500, seed=3)
    assert b["mean"] == pytest.approx(direction_following_index(mixed_c1, mixed_c2, "x", "y")["dfi_overall"])
    assert b["ci_low"] <= b["mean"] <= b["ci_high"]
    assert b["ci_high"] - b["ci_low"] > 0.2      # 5 generations per cell cannot give a tight interval


def test_dfi_is_difference_of_direction_and_input_accuracy():
    """Reviewer point: DFI == Acc_vs_direction - Acc_vs_input (averaged over the two conflict cells)."""
    c1, c2 = ["x", "x", "y", "z"], ["y", "x", "x", "y"]
    d = direction_following_index(c1, c2, "x", "y")
    acc_dir = 0.5 * (d["p_dirX_matches_X"] + d["p_dirY_matches_Y"])
    acc_inp = 0.5 * (d["p_dirX_matches_Y_input"] + d["p_dirY_matches_X_input"])
    assert d["dfi_overall"] == pytest.approx(acc_dir - acc_inp)


# ---------------------------------------------------------------- intervals
def test_wilson_interval_known_values():
    lo, hi = wilson_interval(5, 12)
    assert 0.0 < lo < 5 / 12 < hi < 1.0
    assert wilson_interval(0, 0) == (None, None)
    lo, hi = wilson_interval(0, 3)
    assert lo == 0.0 and hi > 0.5            # 0/3 is compatible with a large true rate
    assert count_cell(5, 12)["rate"] == pytest.approx(5 / 12)


def test_reported_percentages_that_are_not_k_over_n_are_detectable():
    """The reviewer's arithmetic: some percentages are not k/n for any integer k at the stated n."""
    def reachable(pct, n):
        return any(abs(100 * k / n - pct) < 0.05 for k in range(n + 1))
    assert reachable(37.5, 8) and not reachable(41.7, 8)
    assert not reachable(84.4, 30) and not reachable(35.8, 20)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    vals = [0.0, 0.5, 1.0, 0.25, 0.75, 0.5]
    a, b = bootstrap_ci(vals, seed=1), bootstrap_ci(vals, seed=1)
    assert a == b and a["ci_low"] <= a["mean"] <= a["ci_high"]
    assert bootstrap_ci([], seed=1)["mean"] is None


def test_alpha_over_norm():
    assert alpha_over_norm(12.0, 19, {19: 60.0}) == pytest.approx(0.2)
    assert alpha_over_norm(-12.0, 19, {19: 60.0}) == pytest.approx(0.2)
    assert alpha_over_norm(12.0, 3, {19: 60.0}) is None


# ---------------------------------------------------------------- unembedding projection
def test_project_out_span_is_orthogonal_and_keeps_the_rest():
    g = torch.Generator().manual_seed(0)
    d = torch.randn(64, generator=g)
    vecs = [torch.randn(64, generator=g) for _ in range(5)]
    p = project_out_span(d, vecs)
    assert p.norm() == pytest.approx(1.0, abs=1e-5)
    for v in vecs:
        assert torch.dot(p, v / v.norm()).abs() < 1e-5
    # the part of d outside the span is untouched (up to normalisation)
    Q, _ = torch.linalg.qr(torch.stack(vecs, 1))
    expected = d - Q @ (Q.T @ d)
    assert torch.allclose(p, expected / expected.norm(), atol=1e-5)


def test_project_out_span_duplicates_do_not_remove_extra_directions():
    """Near-duplicate readout vectors (raw row and its norm-scaled version) must not widen the projector."""
    g = torch.Generator().manual_seed(1)
    d = torch.randn(32, generator=g)
    v = torch.randn(32, generator=g)
    single = project_out_span(d, [v])
    duplicated = project_out_span(d, [v, v.clone(), 2 * v])
    assert torch.allclose(single, duplicated, atol=1e-4)


def test_project_out_span_raises_when_direction_is_the_readout():
    v = torch.randn(32)
    with pytest.raises(ValueError):
        project_out_span(v * 3, [v])


def _tiny_model():
    from transformers import GPT2LMHeadModel, GPT2Config
    sys_path_ok = __import__("sys").path
    from tests.mock_model import MockGenerativeTokenizer  # noqa: F401  (stage1 is on sys.path via conftest)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=3, n_embd=32, n_head=2, vocab_size=200)).eval()
    return model, MockGenerativeTokenizer(vocab_size=200)


def test_project_out_unembedding_zeroes_alignment_with_label_rows():
    model, tok = _tiny_model()
    d = torch.randn(32)
    proj, diag = project_out_unembedding(d, model, tok, ["Paris", "Rome"], fold_final_norm=True)
    assert proj.norm() == pytest.approx(1.0, abs=1e-5)
    for t in ("Paris", "Rome"):
        assert abs(diag["unembedding_alignment_after"][t]) < 1e-5
    # and with the final-norm-scaled rows too: logit effect of the direction on those tokens is gone
    for v in unembedding_vectors(model, tok, ["Paris", "Rome"], fold_final_norm=True):
        assert torch.dot(proj, v / v.norm()).abs() < 1e-4


def test_project_out_unembedding_pure_readout_direction_collapses():
    model, tok = _tiny_model()
    tid = tok.encode("Paris", add_special_tokens=False)[0]
    pure = model.get_output_embeddings().weight[tid].detach().clone()
    with pytest.raises(ValueError):
        project_out_unembedding(pure, model, tok, ["Paris"], fold_final_norm=False)


def test_projection_removes_direct_logit_effect_of_label():
    """The causal claim behind the control: stepping along the raw readout direction raises the label's logit,
    stepping along the projected direction does not (through the unembedding alone)."""
    model, tok = _tiny_model()
    tid = tok.encode("Paris", add_special_tokens=False)[0]
    W = model.get_output_embeddings().weight.detach()
    d = 3 * W[tid] / W[tid].norm() + 0.3 * torch.randn(32)      # mostly readout, plus noise
    proj, _ = project_out_unembedding(d, model, tok, ["Paris"], fold_final_norm=False)
    raw_gain = (W[tid] @ (d / d.norm())).item()
    proj_gain = (W[tid] @ proj).item()
    assert raw_gain > 0.5 * W[tid].norm().item()                 # raw direction moves the label's logit a lot ...
    assert abs(proj_gain) < 1e-4 * W[tid].norm().item()          # ... the projected one does not


# ---------------------------------------------------------------- inputs / directions
def test_varied_inputs_never_name_their_label_and_are_disjoint():
    labels = sorted(_VARIED)
    assert set(labels) >= {"Paris", "Rome", "dog", "cat", "sad", "happy", "France", "Italy"}
    p = VariedInputs()
    for concept in labels:
        train, held = p.train_texts(concept), p.eval_texts(concept)
        assert len(train) >= 3 and len(held) >= 1
        assert not set(train) & set(held)
        for text in train + held:
            assert leaks_label(text, labels) == [], f"{concept!r} passage names a label: {text!r}"


def test_template_provider_reproduces_original_inputs():
    p = TemplateInputs()
    assert p.eval_texts("Paris") == [build_input_content("Paris")]
    assert p.train_texts("Rome") == [t.format(concept="Rome") for t in TRAINING_INPUT_TEMPLATES]
    assert p.heldout_is_unseen is False and VariedInputs().heldout_is_unseen is True
    with pytest.raises(ValueError):
        get_input_provider("nope")


def test_pair_direction_sources():
    vectors = {f"jspace::5::{t}": torch.randn(16) for t in ("a", "b")}
    j = pair_direction(vectors, "a", "b", 5, "jspace")
    assert torch.allclose(j, build_bipolar_direction(vectors["jspace::5::a"], vectors["jspace::5::b"]))
    r1, r2 = pair_direction(vectors, "a", "b", 5, "random", seed=7), pair_direction(vectors, "a", "b", 5, "random", seed=7)
    assert torch.equal(r1, r2) and r1.norm() == pytest.approx(1.0, abs=1e-5)
    assert not torch.allclose(r1, pair_direction(vectors, "a", "b", 5, "random", seed=8))
    assert not torch.allclose(r1, j)
    assert pair_direction(vectors, "a", "c", 5, "jspace") is None and pair_direction(vectors, "a", "b", 6, "random") is None
    with pytest.raises(ValueError):
        pair_direction(vectors, "a", "b", 5, "zeros")


def test_default_training_examples_unchanged_by_refactor():
    """Template provider + jspace source must reproduce the original example list exactly (order included)."""
    import stage4_train as s4
    vectors = {f"jspace::{l}::{t}": torch.randn(16) for l in (2, 4) for t in ("Paris", "Rome", "dog", "cat")}
    pairs, layers, alpha, prompt = [("Paris", "Rome"), ("dog", "cat")], [2, 4], 12.0, "Q?\nA:"
    new = s4.build_training_examples(vectors, pairs, layers, alpha, prompt)

    old = []
    for (x, y) in pairs:
        for layer in layers:
            d = build_bipolar_direction(vectors[f"jspace::{layer}::{x}"], vectors[f"jspace::{layer}::{y}"])
            for template in TRAINING_INPUT_TEMPLATES:
                old.append((layer, alpha, f"{template.format(concept=x)}\n{prompt}", x, d))
                old.append((layer, -alpha, f"{template.format(concept=y)}\n{prompt}", y, d))
    assert len(new) == len(old) == 2 * 2 * 4 * 2
    for n, o in zip(new, old):
        assert (n["layer"], n["alpha"], n["prompt"], n["target_label"]) == o[:4]
        assert torch.equal(n["direction"], o[4])


def test_stratified_eval_examples_cover_every_label_and_layer():
    import stage4_train as s4
    vectors = {f"jspace::{l}::{t}": torch.randn(16) for l in (2, 4) for t in ("Paris", "Rome", "dog", "cat")}
    ex = s4.build_training_examples(vectors, [("Paris", "Rome"), ("dog", "cat")], [2, 4], 12.0, "Q?\nA:")
    picked = s4.stratified_eval_examples(ex, max_eval=12)
    assert {(e["target_label"], e["layer"]) for e in picked} == {(t, l) for t in ("Paris", "Rome", "dog", "cat") for l in (2, 4)}
    assert len({e["target_label"] for e in ex[:6]}) < 4        # the old `examples[:6]` slice missed labels
    assert s4.stratified_eval_examples(ex[:5], max_eval=12) == ex[:5]


def test_per_layer_alpha_dict_is_resolved_per_training_example():
    import stage4_train as s4
    from generalization import _resolve_alpha
    vectors = {f"jspace::{l}::{t}": torch.randn(16) for l in (3, 19) for t in ("Paris", "Rome")}
    ex = s4.build_training_examples(vectors, [("Paris", "Rome")], [3, 19], {3: 1.5, 19: 6.0}, "Q?\nA:")
    assert {(e["layer"], abs(e["alpha"])) for e in ex} == {(3, 1.5), (19, 6.0)}
    assert all(e["alpha"] > 0 for e in ex if e["target_label"] == "Paris") and all(e["alpha"] < 0 for e in ex if e["target_label"] == "Rome")
    assert _resolve_alpha({3: 1.5}, 3) == 1.5 and _resolve_alpha(6.0, 19) == 6.0
    with pytest.raises(KeyError):
        _resolve_alpha({3: 1.5}, 19)
    with pytest.raises(KeyError):
        s4.build_training_examples(vectors, [("Paris", "Rome")], [3, 19], {3: 1.5}, "Q?\nA:")


def test_score_reports_uses_current_scoring_including_glued_mentions():
    c = score_reports(["thinking aboutFrance, the country", "France", "catcatcatcat"], "France")
    assert c["k"] == 2 and c["n"] == 3


# ---------------------------------------------------------------- seeds
def test_seed_everything_is_reproducible():
    seed_everything(5); a = torch.randn(3)
    seed_everything(5); b = torch.randn(3)
    assert torch.equal(a, b)


def test_aggregate_seeds_summary():
    import aggregate_seeds as ag
    s = ag.summarize("dfi", [[-1.0, -1.0], [-1.0, -0.5], [-1.0, -1.0, -1.0]])
    assert s["n_seeds"] == 3 and s["min"] == pytest.approx(-1.0) and s["max"] == pytest.approx(-0.75)
    lo, hi = s["hierarchical_bootstrap_ci95"]
    assert lo <= s["mean"] <= hi
    assert ag.summarize("empty", [[None]])["n_seeds"] == 0


# ---------------------------------------------------------------- Stage 5 paths on a tiny PEFT model
def test_stage5_functions_with_varied_provider_and_random_directions():
    from peft import get_peft_model, LoraConfig
    from generalization import evaluate_correlated, evaluate_decorrelated
    from mechanism_check import ablation_test

    base, tok = _tiny_model()
    peft_model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"]))
    vectors = {f"jspace::{l}::{t}": torch.randn(32) for l in (1, 2) for t in ("Paris", "Rome")}
    provider = get_input_provider("varied")
    n_trials, n_texts = 2, len(provider.eval_texts("Paris"))
    common = dict(provider=provider, direction_source="random", direction_seed=4)

    corr = evaluate_correlated(peft_model, tok, vectors, [("Paris", "Rome")], [1, 2], 8.0, "Q?\nA:", "cpu",
                               n_trials=n_trials, max_new_tokens=3, **common)
    assert len(corr) == 4 and all(r["counts"]["n"] == n_trials * n_texts for r in corr)

    dec = evaluate_decorrelated(peft_model, tok, vectors, [("Paris", "Rome")], [1], 8.0, "Q?\nA:", "cpu",
                                n_trials=n_trials, max_new_tokens=3, n_boot=50, **common)[0]
    assert dec["counts"]["dirX_matches_X"]["n"] == n_trials * n_texts and dec["n_generations"] == 2 * n_trials * n_texts
    assert dec["dfi_bootstrap"]["ci_low"] <= dec["dfi_bootstrap"]["ci_high"]

    d = pair_direction(vectors, "Paris", "Rome", 1, "random", 4)
    ab = ablation_test(peft_model, tok, 1, d, 8.0, "Paris", "Q?\nA:", "cpu", n_trials=n_trials, max_new_tokens=3,
                       provider=provider)
    for cond in ("baseline", "ablate_direction", "ablate_input", "no_hook_control"):
        assert ab[cond]["counts"]["n"] == n_trials * n_texts      # input-ablated cell gets matched n
    assert "hook_presence_alone_sufficient" in ab


def test_no_hook_control_really_skips_the_hook():
    """apply_hook=False must leave the forward pass untouched, whereas alpha=0 still edits it."""
    from elicitation import elicit_reports
    base, tok = _tiny_model()
    d = torch.randn(32)
    torch.manual_seed(0); no_hook = elicit_reports(base, tok, 1, d, 50.0, "Q?\nA:", "cpu", n_trials=1, max_new_tokens=6, apply_hook=False)
    torch.manual_seed(0); plain = elicit_reports(base, tok, 1, d, 0.0, "Q?\nA:", "cpu", n_trials=1, max_new_tokens=6, apply_hook=False)
    torch.manual_seed(0); clamped0 = elicit_reports(base, tok, 1, d, 0.0, "Q?\nA:", "cpu", n_trials=1, max_new_tokens=6, apply_hook=True)
    assert no_hook == plain                     # alpha is irrelevant when the hook is absent
    from model_utils import get_transformer_layers
    assert not get_transformer_layers(base)[1]._forward_hooks      # and nothing is left registered
    del clamped0


def test_stage6_skips_entries_whose_projected_condition_collapsed():
    from metrics import _flatten_stage3, direction_following_index_summary, accuracy_vs_direction_and_input
    base = {"layer": 3, "x": "a", "y": "b", "alpha": 1.0}
    cell = direction_following_index(["a", "a"], ["b", "b"], "a", "b")
    ok = {**base, "raw": cell, "unembed_projected": cell}
    collapsed = {**base, "raw": cell, "unembed_diagnostics": {"collapsed": True}}
    flat = _flatten_stage3([ok, collapsed], "unembed_projected")
    assert len(flat) == 1                                   # collapsed entry dropped, not crashed on
    assert len(_flatten_stage3([ok, collapsed], "raw")) == 2
    s = direction_following_index_summary(flat)
    assert s["dfi_mean"] == pytest.approx(1.0) and s["n_generations"] == 4
    assert accuracy_vs_direction_and_input(flat)["pooled_counts"]["direction"] == {"k": 4, "n": 4, "wilson95": pytest.approx(
        accuracy_vs_direction_and_input(flat)["pooled_counts"]["direction"]["wilson95"])}


def test_stage7_heldout_phrasings_are_disjoint_from_train_and_never_name_a_label():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage7"))
    import covert_cognition as cc
    for label in (cc.LABEL_INJECTION, cc.LABEL_SAFE):
        train, held = cc.scenario_texts(label, "train"), cc.scenario_texts(label, "heldout")
        assert 5 <= len(held) <= 6 and not set(train) & set(held)
        assert len(cc.all_cell_prompts(label, "heldout")) == len(held)
        for t in train + held:
            assert "injection" not in t.lower() and "safe" not in t.lower().split()
