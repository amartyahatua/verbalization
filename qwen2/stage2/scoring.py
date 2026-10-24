"""
Scoring for Stage 2: automated exact-match scoring, plus utilities to
export elicited reports for blinded human judging and merge the
returned judgments back in.

Exact-match is a coarse, high-precision/low-recall automated signal
(catches only literal mentions of the target word) -- it's a cheap
check, not a replacement for human judging, which is why both are
called for in the plan. Both are tracked at trial-level granularity
(direction_key, trial_index), not aggregated per-direction, so a
human judge's per-report call doesn't silently overwrite a sibling
trial's score for the same direction.
"""

import csv
import json
import random
import re
from typing import List, Dict, Optional, Tuple


def is_degenerate_repetition(text: str, min_repeats: int = 3) -> bool:
    """
    True if `text` looks like a degenerate repetition loop (e.g. "Paris
    Paris Paris Paris...") rather than a genuine report -- caught via
    (a) the same word repeating min_repeats+ times consecutively, or
    (b) one word dominating more than half the text even if not
    strictly consecutive. Both patterns trivially satisfy a naive
    whole-word match without being a coherent report.
    """
    words = [w.strip(".,!?\"'") .lower() for w in text.strip().split()]
    if len(words) < min_repeats:
        return False

    max_run, run = 1, 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    if max_run >= min_repeats:
        return True

    from collections import Counter
    most_common_count = Counter(words).most_common(1)[0][1]
    return most_common_count / len(words) > 0.5


def mentions_multiple_labels(text: str, candidate_labels: List[str], min_distinct: int = 2) -> bool:
    """
    True if `text` mentions min_distinct or more DISTINCT labels from
    `candidate_labels` as whole words -- a shotgun-listing shortcut
    distinct from is_degenerate_repetition: instead of repeating one
    word, the model recites several different trained labels at once
    (e.g. "dog sad happy Rome Paris"), which trivially satisfies
    whole-word matching against whichever target happens to be
    checked, without genuinely committing to reporting any one of
    them specifically. Caught only when the caller knows the full set
    of labels in play (e.g. all target concepts trained/tested in a
    given run) -- exact_match_score alone can't detect this, since it
    only ever sees one target at a time.
    """
    seen = set()
    for label in candidate_labels:
        pattern = r"\b" + re.escape(label.strip()) + r"\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            seen.add(label.lower())
    return len(seen) >= min_distinct


def exact_match_score(
    report: str, target_concept: Optional[str], candidate_labels: Optional[List[str]] = None,
) -> Optional[int]:
    """
    1 if target_concept appears as a whole word in report (case-
    insensitive), 0 if not, OR if the report is a degenerate
    repetition loop (see is_degenerate_repetition), OR if
    candidate_labels is given and the report mentions multiple
    distinct labels from it (see mentions_multiple_labels) -- both
    are ways a report can trivially satisfy whole-word matching
    without being a genuine, specific report. None if there's no
    well-defined target concept for this direction (e.g. a generic
    SAE feature with no single associated word) -- those rely on
    human judging instead.

    candidate_labels is optional and backward compatible: omit it
    (as most existing call sites do) to skip the shotgun-listing
    check entirely -- only pass it when the caller actually knows the
    full set of labels a given run's model could be reciting from.
    """
    if not target_concept:
        return None
    if is_degenerate_repetition(report):
        return 0
    if candidate_labels and mentions_multiple_labels(report, candidate_labels):
        return 0
    pattern = r"\b" + re.escape(target_concept.strip()) + r"\b"
    return 1 if re.search(pattern, report, flags=re.IGNORECASE) else 0


def score_reports(
    reports: List[str], target_concept: Optional[str], candidate_labels: Optional[List[str]] = None,
) -> Dict:
    """
    Exact counts behind a reported rate. Returns
        k            reports scored 1
        n            reports that were scorable (target_concept given)
        n_total      reports generated
        n_degenerate reports flagged as repetition loops (scored 0, kept in n)
        n_multi_label reports flagged as shotgun listings (scored 0, kept in n)
        rate         k / n  (None if n == 0)
    so every percentage in a table can be traced to an integer k out of an integer n.
    Scoring itself is exactly exact_match_score -- nothing here changes what counts as a match.
    """
    scores = [exact_match_score(r, target_concept, candidate_labels=candidate_labels) for r in reports]
    scored = [s for s in scores if s is not None]
    k = int(sum(scored))
    n = len(scored)
    return {
        "k": k, "n": n, "n_total": len(reports),
        "n_degenerate": sum(1 for r in reports if is_degenerate_repetition(r)),
        "n_multi_label": (sum(1 for r in reports if mentions_multiple_labels(r, candidate_labels))
                          if candidate_labels else 0),
        "rate": (k / n) if n else None,
    }


def infer_target_concept(direction_key: str) -> Optional[str]:
    """
    J-space directions are named "jspace::{layer}::{token}" -- the
    token IS the target concept, so exact-match is well-defined for
    them. Other classes don't have one known target word by
    construction, so this returns None for them.
    """
    parts = direction_key.split("::")
    if parts[0] == "jspace" and len(parts) >= 3:
        return parts[2]
    return None


def export_for_blind_judging(records: List[Dict], out_path: str, seed: int = 0) -> str:
    """
    records: [{"direction_key":..., "trial": int, "report": str}, ...]
    Writes a shuffled, ID-only CSV for a human judge to fill in a
    `judge_score` column (1 = report describes the intervened
    concept, 0 = it doesn't) with no direction_key visible. Returns
    the path to a separate answer-key JSON (not shown to the judge)
    used to merge scores back to (direction, trial) pairs afterward.
    """
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)

    answer_key = {}
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["blind_id", "report", "judge_score"])
        for i, r in enumerate(shuffled):
            blind_id = f"item_{i:04d}"
            writer.writerow([blind_id, r["report"], ""])
            answer_key[blind_id] = {"direction_key": r["direction_key"], "trial": r["trial"]}

    answer_key_path = out_path.replace(".csv", "_answer_key.json")
    with open(answer_key_path, "w") as f:
        json.dump(answer_key, f, indent=2)
    return answer_key_path


def merge_judge_scores(judged_csv_path: str, answer_key_path: str) -> Dict[Tuple[str, int], int]:
    """
    Reads a completed blind-judging CSV + its answer key. Returns
    {(direction_key, trial): judge_score}, skipping rows the judge
    left blank.
    """
    answer_key = json.load(open(answer_key_path))
    scores = {}
    with open(judged_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["judge_score"] not in ("", None):
                entry = answer_key[row["blind_id"]]
                scores[(entry["direction_key"], entry["trial"])] = int(row["judge_score"])
    return scores
