#!/usr/bin/env python3
"""
03_delay_reconstruction.py — Aim 1, step 3: observed feedback delay from the
trial-number logical clock.

WHAT THIS SCRIPT DOES, AND WHAT IT REFUSES TO DO
------------------------------------------------
It measures `d_observed`: the number of trial-number slots consumed by OTHER
workers while one worker ran one trial, for every trial in the archive where
that is measurable.

It does NOT reconstruct, relabel, impute, or otherwise assert a nominal worker
count W. No output column of this script may be used as W. Three quantities are
kept strictly separate throughout:

  w_nominal      The `parallel_workers` field in the JSON records, i.e. the
                 `--parallel-workers` CLI argument echoed back by train.py.
                 A LABEL of what was requested. Reported verbatim, never
                 corrected, never inferred where absent.

  realized_concurrency
                 The number of workers actually executing simultaneously at a
                 given moment. NOT MEASURABLE from this archive. No column of
                 this script contains it. It requires per-trial wall-clock
                 timestamps, which exist only in the Optuna SQLite databases
                 (`$PROJECT_DIR/optuna_storage/*.db`, not committed) or in the
                 Eagle SLURM logs (not committed).

  d_observed     What this script measures. An OBSERVED DELAY PROXY, defined
                 below. It is a property of the trial-numbering sequence, not a
                 count of machines.

`d_observed` and `realized_concurrency` are related but are not the same
quantity, and this script never equates them. Where the report needs to compare
`d_observed` against a nominal label it does so as a CONSISTENCY CHECK and
reports the comparison, not a corrected value.

DEFINITION OF d_observed
------------------------
Optuna assigns `trial.number` at ask time, monotonically within a study, from
the shared storage. `train.py` runs `study.optimize(..., n_trials=N)` so a single
worker's trials are strictly sequential: it asks, runs, then asks again.

For one worker (identified by `slurm_task`) with observed trial numbers
n_1 < n_2 < ... < n_k in one study:

    d_observed(n_i) = n_{i+1} - n_i        for i = 1..k-1

This counts every trial number allocated between two consecutive asks by the
same worker, i.e. the asks made by all other live workers during the execution
of trial n_i. The last observed trial of a worker has no successor and receives
an empty `d_observed`.

Two properties make this usable on a damaged archive:

  * It counts NUMBERING SLOTS, not files. Pruned trials, failed trials and
    records destroyed by the v1 filename-overwrite defect still consume numbers,
    so they are counted even though no JSON exists for them. Numbering coverage
    in this archive runs as low as 25%, so this property is doing real work.
  * It is local to one worker. It does not require the archive to be complete,
    only that two of one worker's trial numbers survived.

WHY THIS QUANTITY IS THE INTERESTING ONE
----------------------------------------
Under a delayed-feedback bandit model, the quantity entering the regret bound is
the number of outstanding, unreturned observations at action time. `d_observed`
counts exactly that: the asks issued but not yet informed by this worker's
result. It is the observable reparameterisation, so the archive can supply the
theory's variable even though it cannot supply the nominal W.

REQUIRED INPUT
--------------
--results-root   Archive root containing per-study subdirectories.
                 Repository path: results_zip/results

OPTIONAL INPUT
--------------
--output-dir     Default ./aim1_output
--interior-quantile
                 Fraction trimmed from each end of a study's trial-number range
                 for the ramp-up/drain sensitivity analysis. Default 0.10.
--min-gaps       Minimum measurable gaps before a study's summary statistics are
                 reported rather than suppressed. Default 5.

OUTPUTS
-------
<output-dir>/delay_per_trial.csv        One row per trial record, with
                                        d_observed where measurable. Joinable to
                                        trials_deduplicated.csv on
                                        (study_dir, trial_number).
<output-dir>/delay_per_study.csv        Per-study distribution summary and every
                                        diagnostic below.
<output-dir>/sequential_execution_test.csv
                                        The exact-1 test, per study.
<output-dir>/delay_report.md            Human-readable report.
<output-dir>/delay_report.json          Machine-readable equivalent.

THE EXACT-1 TEST (sequential execution)
---------------------------------------
Under strictly sequential execution — concurrency 1, whether that is one array
task or many array tasks throttled to `%1` — the worker that just finished a
trial is the one that makes the next ask. Therefore:

    every within-worker gap must equal exactly 1.

This holds regardless of how many array tasks exist in total, because a throttle
of 1 serialises them: task 1 takes numbers 0,1; task 2 takes 2,3; and so on, so
within-task gaps remain 1.

Two documented exceptions can produce a gap > 1 under true sequential execution,
and the script reports the evidence needed to judge them rather than ruling
them out silently:

  (E1) The same worker's own intervening trials were pruned or failed, consuming
       numbers without writing files. `train.py` raises TrialPruned at line 233,
       before the json.dump at line 269. Both SLURM scripts set
       N_TRIALS_PER_WORKER / --n-trials to 2, so a worker can consume at most 2
       numbers; a within-worker gap of g implies at least g-1 intervening asks,
       which cannot come from that worker once g-1 exceeds its own remaining
       trial budget.
  (E2) The array task was requeued and resumed later, so its two surviving
       records come from different execution episodes.

The report gives, per study: the number of gaps equal to 1, the number greater
than 1, the gap distribution, and the largest gap. A study in which no gap
equals 1 and gaps substantially exceed the per-worker trial budget is
inconsistent with sequential execution. The script states that conclusion only
as a refutation of the sequential hypothesis; it does not substitute a number.

DIAGNOSTICS AND BIASES (reported per study, not hidden in this docstring)
------------------------------------------------------------------------
  numbering_coverage       distinct observed trial numbers / (max-min+1).
                           Low coverage means many trials left no record. Does
                           not bias d_observed (it counts slots) but indicates
                           how much of the study is invisible.
  pruning_inflation_risk   Derived from coverage and the configured pruner.
                           Pruned trials are short, so a worker running a
                           full-length trial sees MORE intervening asks than it
                           would if all trials ran full length. This inflates
                           d_observed upward. Highest where coverage is lowest.
  single_trial_task_frac   Fraction of workers contributing only one observed
                           trial. These contribute no gap at all. Workers that
                           died early are exactly the ones excluded, so the
                           surviving sample is biased toward long-lived workers:
                           SURVIVORSHIP BIAS, direction not determinable a priori.
  interior_median          d_observed median restricted to the interior of the
                           study's numbering range. Ramp-up and drain periods
                           have artificially small gaps because few workers are
                           live. A large difference between the full-range and
                           interior medians indicates strong ramp/drain effects.
  duration_heterogeneity   Ratio of the 90th to 10th percentile of per-trial
                           duration within the study. Workers running slow
                           trials accumulate more intervening asks, inflating
                           their d_observed independently of concurrency. See
                           04_node_heterogeneity_audit.py, which characterises
                           this directly.

NO INFERENTIAL STATISTICS
-------------------------
This script reports order statistics (min, quartiles, median, max, counts) of an
observed distribution. It computes no bootstrap confidence interval, no
Mann-Whitney U test, no multiple-comparison correction and no risk-adjusted
score. Those require independent, exchangeable observations. Gaps within one
study are serially dependent by construction — consecutive gaps share workers
and share the same numbering sequence — and there is exactly one independent
study per (sampler, dataset, label) cell. Neither independence assumption is
satisfied, so no such test is performed here.

REPRODUCIBILITY
---------------
Deterministic: no sampling, no randomness, no seeds. The report records the
script version, the input root, a content hash over the sorted (study_dir,
filename, size) inventory, and every parameter used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = "1.0.0"

# Pruner configured per sampler in train.py. Used only to annotate the pruning
# inflation diagnostic; never used to adjust a measured value.
SAMPLER_PRUNER = {
    "random": "NopPruner (no pruning)",
    "tpe": "MedianPruner(n_startup_trials=5, n_warmup_steps=5)",
    "hyperband": "HyperbandPruner(min_resource=3, reduction_factor=3)",
    "cmaes": "MedianPruner(n_startup_trials=5, n_warmup_steps=5)",
}
# Per-worker trial budget, from hpo_array.slurm (N_TRIALS_PER_WORKER=2) and
# hpo_v2.slurm (--n-trials 2).
WORKER_TRIAL_BUDGET = 2


def archive_hash(results_root: Path) -> str:
    """Deterministic content hash over the file inventory, for reproducibility."""
    digest = hashlib.sha256()
    for path in sorted(results_root.rglob("*.json")):
        rel = path.relative_to(results_root).as_posix()
        digest.update(f"{rel}:{path.stat().st_size}\n".encode())
    return digest.hexdigest()[:16]


def quantile(sorted_values, q):
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, int(q * len(sorted_values)))
    return sorted_values[idx]


def trial_duration_seconds(record):
    """Duration in seconds, or None. Never converts between the two fields."""
    if record.get("total_gpu_hours") is not None:
        return record["total_gpu_hours"] * 3600.0
    if record.get("elapsed_s") is not None:
        return record["elapsed_s"]
    return None


def load_study(directory: Path):
    """Read one study's trial records. Returns (records, label_values)."""
    records, labels = [], set()
    for path in sorted(directory.glob("trial_*.json")):
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if record.get("trial_number") is None:
            continue
        if record.get("parallel_workers") is not None:
            labels.add(record["parallel_workers"])
        records.append({
            "trial_number": record["trial_number"],
            "slurm_task": record.get("slurm_task"),
            "sampler": record.get("sampler", ""),
            "dataset": record.get("dataset", ""),
            "w_nominal": record.get("parallel_workers", ""),
            "duration_s": trial_duration_seconds(record),
            "source_file": path.name,
        })
    # Labels can also live only in summary files.
    for path in sorted(directory.glob("summary_*.json")):
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if record.get("parallel_workers") is not None:
            labels.add(record["parallel_workers"])
    return records, labels


def compute_gaps(records):
    """d_observed per trial, keyed by (slurm_task, trial_number).

    Workers with no slurm_task cannot be grouped and are excluded, with the
    exclusion reported rather than silently dropped.
    """
    by_task = defaultdict(list)
    no_task = 0
    for rec in records:
        if rec["slurm_task"] is None:
            no_task += 1
            continue
        by_task[rec["slurm_task"]].append(rec["trial_number"])

    gaps = {}
    for task, numbers in by_task.items():
        ordered = sorted(set(numbers))
        for i in range(len(ordered) - 1):
            gaps[(task, ordered[i])] = ordered[i + 1] - ordered[i]
    return gaps, by_task, no_task


def analyse_study(name, records, labels, interior_q, min_gaps):
    gaps, by_task, no_task = compute_gaps(records)
    numbers = sorted({r["trial_number"] for r in records})
    lo, hi = (numbers[0], numbers[-1]) if numbers else (None, None)
    span = (hi - lo + 1) if numbers else 0
    coverage = (len(numbers) / span) if span else 0.0

    gap_values = sorted(gaps.values())
    # Interior restriction for the ramp-up / drain sensitivity check.
    interior_values = []
    if numbers and span > 2:
        cut_lo = lo + interior_q * (hi - lo)
        cut_hi = hi - interior_q * (hi - lo)
        interior_values = sorted(
            g for (task, number), g in gaps.items() if cut_lo <= number <= cut_hi)

    durations = sorted(r["duration_s"] for r in records if r["duration_s"] is not None)
    dur_ratio = ""
    if len(durations) >= 10:
        p10, p90 = quantile(durations, 0.10), quantile(durations, 0.90)
        if p10 and p10 > 0:
            dur_ratio = round(p90 / p10, 2)

    single = sum(1 for v in by_task.values() if len(set(v)) == 1)
    samplers = sorted({r["sampler"] for r in records if r["sampler"]})
    pruner = "/".join(SAMPLER_PRUNER.get(s, "unknown") for s in samplers) or "unknown"

    ones = sum(1 for g in gap_values if g == 1)
    over = sum(1 for g in gap_values if g > 1)
    # A gap of g requires at least g-1 asks between the two. A worker's own
    # budget is WORKER_TRIAL_BUDGET trials total, so it can contribute at most
    # WORKER_TRIAL_BUDGET-1 intervening asks under exception (E1).
    unexplainable = sum(1 for g in gap_values if g - 1 > WORKER_TRIAL_BUDGET - 1)

    if not gap_values:
        verdict = "NOT_TESTABLE_NO_GAPS"
    elif over == 0:
        verdict = "CONSISTENT_WITH_SEQUENTIAL"
    elif unexplainable == 0:
        verdict = "INCONCLUSIVE_GAPS_WITHIN_OWN_TRIAL_BUDGET"
    else:
        verdict = "SEQUENTIAL_REFUTED"

    return {
        "study_dir": name,
        "w_nominal_label": ("/".join(str(v) for v in sorted(labels)) if labels else ""),
        "w_nominal_present": bool(labels),
        "realized_concurrency": "",          # never measurable here, always empty
        "n_trial_records": len(records),
        "n_workers_with_records": len(by_task),
        "n_records_without_slurm_task": no_task,
        "n_gaps_measurable": len(gap_values),
        "d_observed_min": gap_values[0] if gap_values else "",
        "d_observed_p25": quantile(gap_values, 0.25) if gap_values else "",
        "d_observed_median": (statistics.median(gap_values) if gap_values else ""),
        "d_observed_p75": quantile(gap_values, 0.75) if gap_values else "",
        "d_observed_max": gap_values[-1] if gap_values else "",
        "d_observed_interior_median": (statistics.median(interior_values)
                                       if interior_values else ""),
        "n_gaps_interior": len(interior_values),
        "reportable": len(gap_values) >= min_gaps,
        "trial_number_min": lo if lo is not None else "",
        "trial_number_max": hi if hi is not None else "",
        "numbering_coverage": round(coverage, 3),
        "pruner_configured": pruner,
        "pruning_inflation_risk": ("high" if coverage < 0.5 else
                                   "moderate" if coverage < 0.8 else "low"),
        "single_trial_task_frac": (round(single / len(by_task), 3) if by_task else ""),
        "duration_p90_over_p10": dur_ratio,
        "seq_gaps_equal_1": ones,
        "seq_gaps_greater_1": over,
        "seq_gaps_exceeding_own_budget": unexplainable,
        "sequential_test_verdict": verdict,
        "_gap_values": gap_values,
    }


PER_TRIAL_FIELDS = [
    "study_dir", "trial_number", "slurm_task", "sampler", "dataset",
    "w_nominal", "d_observed", "d_observed_measurable", "duration_s",
    "source_file",
]

PER_STUDY_FIELDS = [
    "study_dir", "w_nominal_label", "w_nominal_present", "realized_concurrency",
    "n_trial_records", "n_workers_with_records", "n_records_without_slurm_task",
    "n_gaps_measurable", "reportable", "d_observed_min", "d_observed_p25",
    "d_observed_median", "d_observed_p75", "d_observed_max",
    "d_observed_interior_median", "n_gaps_interior", "trial_number_min",
    "trial_number_max", "numbering_coverage", "pruner_configured",
    "pruning_inflation_risk", "single_trial_task_frac", "duration_p90_over_p10",
    "seq_gaps_equal_1", "seq_gaps_greater_1", "seq_gaps_exceeding_own_budget",
    "sequential_test_verdict",
]

SEQ_FIELDS = [
    "study_dir", "w_nominal_label", "n_gaps_measurable", "seq_gaps_equal_1",
    "seq_gaps_greater_1", "seq_gaps_exceeding_own_budget", "d_observed_max",
    "sequential_test_verdict", "gap_distribution",
]


def build_report(studies, params, meta):
    lines = []
    add = lines.append
    add("# Aim 1 — Observed Delay Reconstruction (`d_observed`)")
    add("")
    add(f"- Generated: {meta['generated']}")
    add(f"- Script version: {SCRIPT_VERSION}")
    add(f"- Archive root: `{meta['results_root']}`")
    add(f"- Archive content hash: `{meta['archive_hash']}`")
    add(f"- Parameters: interior_quantile={params['interior_quantile']}, "
        f"min_gaps={params['min_gaps']}")
    add("")
    add("## 0. What is and is not claimed here")
    add("")
    add("| Quantity | Status in this report |")
    add("|---|---|")
    add("| `w_nominal` | The `parallel_workers` label, reported verbatim. Never corrected. |")
    add("| `realized_concurrency` | **Not measurable from this archive.** Column is empty in every row. |")
    add("| `d_observed` | Measured here. Trial-number slots consumed by other workers during one trial. |")
    add("")
    add("`d_observed` is not a worker count and must not be substituted for one.")
    add("Where it is compared against a label below, that is a consistency check")
    add("reported as such, not a corrected value.")
    add("")
    add("## 1. Observed delay per study")
    add("")
    add("| study_dir | w_nominal | gaps | min | p25 | median | p75 | max | interior median |")
    add("|---|---|---|---|---|---|---|---|---|")
    for s in studies:
        if not s["reportable"]:
            add(f"| {s['study_dir']} | {s['w_nominal_label'] or '—'} | "
                f"{s['n_gaps_measurable']} | *suppressed: below min_gaps* | | | | | |")
            continue
        add(f"| {s['study_dir']} | {s['w_nominal_label'] or '—'} | "
            f"{s['n_gaps_measurable']} | {s['d_observed_min']} | {s['d_observed_p25']} | "
            f"{s['d_observed_median']} | {s['d_observed_p75']} | {s['d_observed_max']} | "
            f"{s['d_observed_interior_median'] or '—'} |")
    add("")
    add("## 2. Sequential-execution test (exact-1 test)")
    add("")
    add("Under strictly sequential execution the worker that just finished makes the")
    add("next ask, so **every within-worker gap must equal exactly 1**. This holds")
    add("even for a large array serialised by a `%1` throttle.")
    add("")
    add("A gap of g implies at least g-1 intervening asks. Both SLURM scripts give")
    add(f"each worker a budget of {WORKER_TRIAL_BUDGET} trials, so a worker can account")
    add(f"for at most {WORKER_TRIAL_BUDGET - 1} intervening ask(s) from its own pruned or")
    add("failed trials. Gaps larger than that cannot be explained by the worker itself.")
    add("")
    add("| study_dir | w_nominal | gaps | =1 | >1 | exceeding own budget | max gap | verdict |")
    add("|---|---|---|---|---|---|---|---|")
    for s in studies:
        add(f"| {s['study_dir']} | {s['w_nominal_label'] or '—'} | "
            f"{s['n_gaps_measurable']} | {s['seq_gaps_equal_1']} | "
            f"{s['seq_gaps_greater_1']} | {s['seq_gaps_exceeding_own_budget']} | "
            f"{s['d_observed_max'] or '—'} | {s['sequential_test_verdict']} |")
    add("")
    add("Verdicts: `SEQUENTIAL_REFUTED` — gaps exceed what the worker's own trial")
    add("budget could produce, so other workers were asking concurrently.")
    add("`CONSISTENT_WITH_SEQUENTIAL` — all gaps equal 1.")
    add("`INCONCLUSIVE_GAPS_WITHIN_OWN_TRIAL_BUDGET` — gaps exceed 1 but are small")
    add("enough to be explained by the worker's own pruned trials.")
    add("`NOT_TESTABLE_NO_GAPS` — no worker contributed two surviving trials.")
    add("")
    add("A refutation removes the sequential hypothesis. It does **not** establish")
    add("what the concurrency was instead.")
    add("")
    add("## 3. Diagnostics and biases")
    add("")
    add("| study_dir | numbering coverage | pruner | pruning inflation | single-trial workers | duration p90/p10 | full vs interior median |")
    add("|---|---|---|---|---|---|---|")
    for s in studies:
        shift = "—"
        if s["d_observed_median"] != "" and s["d_observed_interior_median"] != "":
            shift = f"{s['d_observed_median']} → {s['d_observed_interior_median']}"
        add(f"| {s['study_dir']} | {s['numbering_coverage']} | {s['pruner_configured']} | "
            f"{s['pruning_inflation_risk']} | {s['single_trial_task_frac']} | "
            f"{s['duration_p90_over_p10'] or '—'} | {shift} |")
    add("")
    add("**Pruning inflation (upward bias).** Pruned trials consume trial numbers but")
    add("finish early, so a worker running a full-length trial observes more")
    add("intervening asks than concurrency alone would produce. Severity tracks")
    add("numbering coverage: low coverage means many short, pruned trials are")
    add("invisible in the records but present in the numbering.")
    add("")
    add("**Incomplete coverage.** `numbering_coverage` below 1.0 means trial numbers")
    add("exist with no surviving record — pruned, failed, or destroyed by the v1")
    add("filename-overwrite defect. This does not bias `d_observed`, which counts")
    add("slots rather than files, but it bounds how much of each study is visible.")
    add("")
    add("**Ramp-up and drain (downward bias).** At the start of a study few workers")
    add("have launched and at the end most have exited, so gaps are artificially")
    add("small. The interior median trims the outer")
    add(f"{int(params['interior_quantile'] * 100)}% of the numbering range at each end.")
    add("A large full-vs-interior shift indicates the effect is material.")
    add("")
    add("**Survivorship bias (direction indeterminate).** Only workers with at least")
    add("two surviving trials contribute a gap. Workers that died after one trial")
    add("contribute nothing, and those are precisely the workers whose execution was")
    add("atypical. `single_trial_task_frac` reports how much of each study is")
    add("excluded on these grounds.")
    add("")
    add("**Duration heterogeneity (upward bias for slow trials).** A worker whose")
    add("trial ran long accumulates more intervening asks at fixed concurrency.")
    add("`duration_p90_over_p10` flags studies where this is large; see")
    add("`04_node_heterogeneity_audit.py` for a direct characterisation.")
    add("")
    add("## 4. No inferential statistics were computed")
    add("")
    add("Only order statistics of an observed distribution appear above. No")
    add("bootstrap confidence interval, Mann-Whitney U test, multiple-comparison")
    add("correction or risk-adjusted score is computed, because neither independence")
    add("condition holds: gaps within a study are serially dependent by construction")
    add("(consecutive gaps share workers and share one numbering sequence), and there")
    add("is exactly one independent study per (sampler, dataset, label) cell. These")
    add("tests remain blocked until independent replicates exist.")
    add("")
    add("## 5. What would settle this properly")
    add("")
    add("`realized_concurrency` needs per-trial wall-clock timestamps. Optuna's RDB")
    add("`TrialModel` stores `datetime_start` and `datetime_complete` for every")
    add("trial, including pruned ones. Those databases are declared at")
    add("`$PROJECT_DIR/optuna_storage/` in both SLURM scripts")
    add("(`hpo_array.slurm:70`, `hpo_v2.slurm:71`) and are not committed. Recovering")
    add("them would give exact concurrency and would supersede this entire script.")
    add("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure observed feedback delay from the trial-number logical clock.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("./aim1_output"))
    parser.add_argument("--interior-quantile", type=float, default=0.10)
    parser.add_argument("--min-gaps", type=int, default=5)
    args = parser.parse_args(argv)

    if not args.results_root.is_dir():
        print(f"ERROR: --results-root is not a directory: {args.results_root}",
              file=sys.stderr)
        return 2
    if not 0.0 <= args.interior_quantile < 0.5:
        print("ERROR: --interior-quantile must be in [0, 0.5)", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    studies, per_trial = [], []
    for directory in sorted(d for d in args.results_root.iterdir() if d.is_dir()):
        records, labels = load_study(directory)
        if not records:
            continue
        summary = analyse_study(directory.name, records, labels,
                                args.interior_quantile, args.min_gaps)
        studies.append(summary)

        gaps, _, _ = compute_gaps(records)
        for rec in records:
            key = (rec["slurm_task"], rec["trial_number"])
            measurable = key in gaps
            per_trial.append({
                "study_dir": directory.name,
                "trial_number": rec["trial_number"],
                "slurm_task": "" if rec["slurm_task"] is None else rec["slurm_task"],
                "sampler": rec["sampler"],
                "dataset": rec["dataset"],
                "w_nominal": rec["w_nominal"],
                "d_observed": gaps[key] if measurable else "",
                "d_observed_measurable": measurable,
                "duration_s": ("" if rec["duration_s"] is None
                               else round(rec["duration_s"], 3)),
                "source_file": rec["source_file"],
            })

    if not studies:
        print(f"ERROR: no study subdirectories with trial records under "
              f"{args.results_root}", file=sys.stderr)
        return 2

    per_trial.sort(key=lambda r: (r["study_dir"], r["trial_number"]))
    out = args.output_dir

    def write(path, fields, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fields})

    write(out / "delay_per_trial.csv", PER_TRIAL_FIELDS, per_trial)
    write(out / "delay_per_study.csv", PER_STUDY_FIELDS, studies)
    write(out / "sequential_execution_test.csv", SEQ_FIELDS,
          [{**s, "gap_distribution": " ".join(str(g) for g in s["_gap_values"][:40])}
           for s in studies])

    params = {"interior_quantile": args.interior_quantile, "min_gaps": args.min_gaps}
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results_root": str(args.results_root),
        "archive_hash": archive_hash(args.results_root),
        "script_version": SCRIPT_VERSION,
    }
    (out / "delay_report.md").write_text(build_report(studies, params, meta))
    (out / "delay_report.json").write_text(json.dumps(
        {"meta": meta, "params": params,
         "studies": [{k: v for k, v in s.items() if not k.startswith("_")}
                     for s in studies]}, indent=2) + "\n")

    refuted = [s["study_dir"] for s in studies
               if s["sequential_test_verdict"] == "SEQUENTIAL_REFUTED"]
    print(f"Studies analysed: {len(studies)}   trial rows: {len(per_trial)}")
    print(f"Trials with measurable d_observed: "
          f"{sum(1 for r in per_trial if r['d_observed_measurable'])}")
    print(f"Sequential execution REFUTED in {len(refuted)} study/studies: "
          f"{', '.join(refuted) if refuted else '(none)'}")
    print("realized_concurrency: not measurable from this archive; column left empty.")
    print(f"Wrote outputs to {out}/")
    print("No inferential statistics computed. See section 4 of delay_report.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
