# verbalizable-access — Stage 1

Stage 1 of the *Verbalizable Representations Beyond J-Space* project: build a
**direction bank** covering the four direction classes from Section 5.1
(J-space, SAE features, probe directions, random directions), each tagged
with a **causal-efficacy score**. This bank is the input to Stage 2
(verbalizability elicitation) and Stage 3 (the decorrelation test).

## Files

| File | Purpose |
|---|---|
| `config.py` | Model / SAE / extraction settings — edit this first |
| `data.py` | FineWeb contexts + CounterFact triples (lazy-imports `datasets`) |
| `jlens.py` | J-lens direction extraction (best-effort reimplementation — see docstring caveat) |
| `sae_features.py` | Loads a pretrained SAE via `sae_lens`, samples decoder-column directions stratified by abstraction |
| `probes.py` | Mean-difference and logistic-regression probes on CounterFact-style pos/neg pairs |
| `random_directions.py` | Fully-random and random-in-SAE-decoder-span directions, norm-matched to the other classes |
| `causal_efficacy.py` | Steers each direction and scores the KL shift in next-token distribution |
| `build_direction_bank.py` | Orchestrates all of the above and saves the bank |
| `tests/mock_model.py` | Tiny fake model/tokenizer/SAE for offline pipeline validation |

## Setup

```bash
pip install -r requirements.txt
```

Edit `config.py`: set `model_name` (primary) / `replication_model_name`,
`sae_release`, `sae_layers`, and `jlens_target_tokens` to match the concept
vocabulary you'll actually test in Stage 2/3.

## Run

```bash
# Sanity-check the pipeline first — no downloads, no GPU, runs in seconds
python build_direction_bank.py --dry_run

# Full run against real weights + real data
python build_direction_bank.py
```

Output: `direction_bank/vectors.pt` (dict of direction vectors) and
`direction_bank/metadata.json` (layer, class, causal-efficacy score per
direction).

## Known caveats to revisit

- **`jlens.py`** is a best-effort reimplementation of the J-lens technique
  from its public description — the original paper's exact averaging
  scheme isn't fully published. Validate that your extracted J-space
  directions actually show the expected signature (low-variance,
  concentrated in middle layers) before trusting them as positive controls.
- **`data.py`**'s context loader uses FineWeb's `sample-10BT` config, not the
  unconfigured full corpus — the latter resolves to tens of thousands of
  shard files and can stall for a long time on streaming-shuffle. If it's
  still slow for you, try lowering the shuffle `buffer_size` further (or to
  0) in `load_context_corpus`.
- **`data.py`**'s CounterFact field names assume the
  `NeelNanda/counterfact-tracing` schema — check/adjust if you use a
  different release.
- SAE feature stratification falls back to uniform random sampling unless
  you supply per-feature activation frequencies (not wired up yet — pull
  these from SAELens/Neuronpedia stats if you want true decile stratification).
