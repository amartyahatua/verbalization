# verbalizable-access

Two stages so far, each in its own directory:

- **`stage1/`** — builds the direction bank (J-space, SAE, probe, random directions), each tagged with a causal-efficacy score.
- **`stage2/`** — Native verbalizability measurement (H1): clamps each direction, elicits free-text reports, scores them, computes regression covariates, fits the mixed-effects model.

`stage2/` imports several `stage1/` modules (`config`, `model_utils`, `data`, `build_direction_bank`). This is handled with a small `sys.path` shim at the top of the three files that need it (`editing.py`, `covariates.py`, `stage2_elicit.py`) — it inserts `../stage1` relative to the file's own location, so both stages still run as plain scripts (`python stage2_elicit.py`) without needing `-m` invocation, a manually-set `PYTHONPATH`, or `pip install -e .`. If you turn this into a proper installable package later, that shim is the thing to remove in favor of real package-relative imports.

## Stage 1

### Files

| File | Purpose |
|---|---|
| `config.py` | Model / SAE / extraction settings — edit this first |
| `data.py` | FineWeb contexts + CounterFact triples (lazy-imports `datasets`) |
| `jspace_lens.py` | J-lens direction extraction (best-effort reimplementation — see docstring caveat) |
| `sae_features.py` | Loads a pretrained SAE via `sae_lens`, samples decoder-column directions stratified by abstraction |
| `probes.py` | Mean-difference and logistic-regression probes on CounterFact-style pos/neg pairs |
| `random_directions.py` | Fully-random and random-in-SAE-decoder-span directions, norm-matched to the other classes |
| `causal_efficacy.py` | Steers each direction and scores the KL shift in next-token distribution |
| `model_utils.py` | Architecture-agnostic transformer-layer lookup (GPT-2 / Llama-family / GPT-NeoX) |
| `build_direction_bank.py` | Orchestrates all of the above and saves the bank |
| `tests/mock_model.py` | Tiny fake model/tokenizer/SAE for offline pipeline validation |

### Run

```bash
cd stage1
pip install -r ../requirements.txt
python build_direction_bank.py --dry_run     # sanity-check, no downloads, no GPU
python build_direction_bank.py               # full run
```

Output: `direction_bank/vectors.pt` + `direction_bank/metadata.json`.

### Known caveats to revisit

- **`jspace_lens.py`** is a best-effort reimplementation of the J-lens technique from its public description — validate the extracted directions show the expected signature (concentrated in middle layers) before trusting them as positive controls.
- **`data.py`**'s context loader uses FineWeb's `sample-10BT` config, not the unconfigured full corpus — the latter resolves to tens of thousands of shard files and can stall on streaming-shuffle.
- **`data.py`**'s CounterFact field names assume the `NeelNanda/counterfact-tracing` schema — check/adjust if you use a different release.
- SAE feature stratification falls back to uniform random sampling unless you supply per-feature activation frequencies.

## Stage 2

### Files

| File | Purpose |
|---|---|
| `editing.py` | Activation editing: project out current component, set to a known value `alpha` |
| `elicitation.py` | Runs the fixed elicitation prompt `n_trials` times per direction with the edit active |
| `scoring.py` | Automated exact-match scoring + blind human-judging export/merge (trial-level granularity) |
| `covariates.py` | `jacobian_magnitude`, `unembedding_alignment`, `workspace_participation` |
| `stage2_elicit.py` | Orchestrates all of the above across the Stage 1 bank, saves results |
| `regression.py` | Fits `report_match ~ jacobian_magnitude + unembedding_alignment + workspace_participation`, direction identity as random effect |

### Run

```bash
cd stage2
python stage2_elicit.py --dry_run             # tiny real GPT-2 architecture, no downloads
python stage2_elicit.py --bank_dir ../stage1/direction_bank   # full run against your Stage 1 bank
python regression.py --results stage2_results/elicitation_results.json
```

If you've run blind human judging: fill in the `judge_score` column of `stage2_results/blind_judging.csv`, then:

```bash
python regression.py --results stage2_results/elicitation_results.json \
    --judged_csv stage2_results/blind_judging.csv \
    --answer_key stage2_results/blind_judging_answer_key.json
```

### Known caveats to revisit

- **Compute cost**: a full run across all 835 Stage 1 directions is ~4,175 generations plus ~42,000 gradient passes for covariates. Shrink `stage2_n_trials` / `stage2_covariate_n_contexts` in `config.py` for a first pass.
- **Exact-match only has a well-defined target for J-space directions** (named after their target token). SAE/probe/random directions rely on human judging via the blind-judging CSV — their `exact_match_rate` will be `None` until judged scores are merged in.
- **`fit_mixed_effects`** treats the binary outcome as continuous (a "linear probability model") — simple and robust, but expect `ConvergenceWarning`s even with real signal present (verified against synthetic data with known ground truth). `fit_binomial_glmm` is the statistically stricter alternative.
- **Model capability floor**: GPT-2-small isn't instruction-tuned; the elicitation prompt asks it to introspect. Worth piloting ~10-20 directions and reading the actual report text before committing to a full run.
