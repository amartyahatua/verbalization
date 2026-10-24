"""
Data loading for Stage 1.

- FineWeb sample: generic natural-language contexts, used both to
  compute J-lens directions (averaging over many contexts is the
  whole point of the technique) and to provide a background
  distribution for causal-efficacy scoring.
- CounterFact: subject-relation-object triples with paired
  counterfactual subjects (e.g. Paris/Rome - capital of). Used to fit
  probe directions for known attributes here in Stage 1, and reused
  in Stage 3 to build the decorrelation conflict design (direction
  X/Y vs. input X/Y).

`datasets` is imported lazily inside each function so this module —
and anything that imports it — stays importable in the dry-run test
without requiring the package or network access.
"""

from typing import List, Dict, Optional


def load_context_corpus(n: int, seed: int = 0, progress_every: int = 200) -> List[str]:
    """
    Load `n` short natural-language passages to use as the background
    context distribution for J-lens averaging and causal-efficacy
    scoring.

    Uses FineWeb's "sample-10BT" config, NOT the full corpus.
    `load_dataset("HuggingFaceFW/fineweb", ...)` with no `name=`
    resolves to the entire multi-terabyte dataset (tens of thousands
    of shard files) — that's what causes a long stall on "Resolving
    data files" / streaming-shuffle. The 10B-token sample config is
    still large but orders of magnitude fewer shards and streams much
    faster. Buffer size is also kept small (fast first yield, still
    reasonably mixed) rather than the 10k-row buffer used before.
    """
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=500)

    out = []
    for i, row in enumerate(ds):
        text = row["text"].strip()
        if len(text) > 50:
            out.append(text[:512])  # truncate to a manageable context length
        if len(out) >= n:
            break
        if progress_every and i % progress_every == 0:
            print(f"  ...streamed {i} rows, kept {len(out)}/{n} contexts so far", flush=True)
    return out


def load_counterfact(n: Optional[int] = None) -> List[Dict]:
    """
    Load CounterFact (Meng et al., 2022) subject-relation-object triples.

    NOTE: field names below match the commonly-used
    "NeelNanda/counterfact-tracing" HF release; adjust the key lookups
    if you use a different CounterFact release with different schema.
    """
    from datasets import load_dataset

    ds = load_dataset("NeelNanda/counterfact-tracing", split="train")
    records = []
    skipped = 0
    for row in ds:
        relation = row.get("relation_prefix", row.get("template", ""))
        if not relation or not str(relation).strip():
            skipped += 1
            continue
        records.append({
            "subject": row.get("subject"),
            "relation": relation,
            "object_true": row.get("target_true", row.get("object_true", "")),
            "object_new": row.get("target_new", row.get("object_new", "")),
        })
        if n and len(records) >= n:
            break

    if skipped:
        print(f"  (skipped {skipped} CounterFact rows with an empty/missing relation field)")
    return records
