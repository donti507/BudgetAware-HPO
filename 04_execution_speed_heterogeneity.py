#!/usr/bin/env python3
"""
04_execution_speed_heterogeneity.py — Aim 1, step 4: evidence for heterogeneous
per-trial execution speed, and its effect on GPU-hour comparisons.

WHAT THIS SCRIPT CLAIMS, AND WHAT IT DOES NOT
----------------------------------------------
It measures per-trial execution speed (seconds per epoch), holding the training
workload fixed, and tests whether the resulting distribution separates into
distinct classes.

It does NOT attribute any observed separation to a cause. It does not claim a
GPU model, a node partition, a driver version, a dataloader setting, or a
cluster change. The archive contains no field that identifies the executing
node: `train.py` writes `device` as the string `str(device)`, which is `"cuda"`
for every GPU trial, and records no hostname, no `SLURMD_NODENAME`, no GPU name,
and no library versions. Cause is therefore not determinable from what is
committed, and the report says so rather than guessing.

What the script does produce is: the measurement, a per-stratum test for
separation, the size of the effect, the per-study mixture, the consequence for
GPU-hour comparisons, and an explicit list of candidate causes each paired with
the specific evidence that would be needed to distinguish it.

REQUIRED INPUT
--------------
--results-root   Archive root containing per-study subdirectories.
                 Repository path: results_zip/results

OPTIONAL INPUT
--------------
--output-dir           Default ./aim1_output
--min-stratum-n        Minimum trials in a (dataset, model, batch_size) stratum
                       before separation is tested. Default 8.
--separation-ratio     A stratum is reported as separated only if the largest
                       multiplicative gap between consecutive sorted speeds is
                       at least this factor. Default 1.8.
--assumed-epochs       Epoch count assumed for the secondary measurement path
                       (see MEASUREMENT PATHS). Default 30, which is
                       `MAX_EPOCHS=30` in hpo_array.slurm and `--epochs 30` in
                       hpo_v2.slurm, matching train.py's `--epochs` default.

OUTPUTS
-------
<output-dir>/execution_speed_per_trial.csv    Per-trial speed, stratum, class.
<output-dir>/execution_speed_strata.csv       Per-stratum separation evidence.
<output-dir>/execution_speed_by_study.csv     Per-study class mixture.
<output-dir>/execution_speed_report.md        Human-readable report.
<output-dir>/execution_speed_report.json      Machine-readable equivalent.

MEASUREMENT PATHS
-----------------
Two disjoint paths, never mixed in one number, always labelled per row:

  MEASURED_FROM_CURVE  (v2 records; `accuracy_curve` present)
      The curve stores cumulative `total_gpu_hours` at each completed epoch.
      Speed is the median of consecutive differences, in seconds. Robust to a
      trial being pruned partway, because it uses only completed epochs, and it
      requires no assumption about how many epochs ran.

  ESTIMATED_FROM_ELAPSED  (v1 records; `elapsed_s` present, no curve)
      `elapsed_s` is total trial wall time with no epoch breakdown, so epoch
      count must be assumed. This path is used ONLY where the configured pruner
      cannot terminate a trial early — that is, sampler `random`, which
      `train.py` pairs with `NopPruner()`. For those trials every epoch ran, so
      speed = elapsed_s / assumed_epochs.
      Records using a pruning sampler (`tpe`, `hyperband`, `cmaes`) are EXCLUDED
      from this path, because the number of epochs actually completed is
      unknown and any division would fabricate it.

The two paths are reported separately and are never averaged together. Rows
carry a `measurement_path` column so any downstream use can restrict to one.

STRATIFICATION
--------------
Execution speed depends on the workload, so speeds are only compared within a
stratum of identical workload: (dataset, model, batch_size). Comparing a
resnet18 trial against an efficientnet_b1 trial would measure the model, not the
execution environment.

SEPARATION TEST
---------------
Within each stratum, speeds are normalised by the stratum's 5th percentile to
give a relative speed, then sorted. The largest multiplicative gap between
consecutive relative speeds is the candidate split point. A stratum is reported
as SEPARATED only when that gap is at least `--separation-ratio` and both
resulting classes are non-empty. Otherwise the stratum is reported as
NOT_SEPARATED and its trials are left unclassified.

The split is derived per stratum from the data. No global threshold is
hard-coded, so the classification does not depend on a number chosen by
inspecting these particular results.

Classes are named `FAST_CLASS` and `SLOW_CLASS`. These are labels for positions
in a measured distribution. They are not claims about hardware.

NO INFERENTIAL STATISTICS
-------------------------
Order statistics, counts and ratios only. No bootstrap confidence interval,
Mann-Whitney U test, multiple-comparison correction or risk-adjusted score is
computed. Trials within a study are not independent (one shared adaptive study
per directory), and there is one independent study per cell, so neither the
bootstrap's resampling assumption nor Mann-Whitney's exchangeability assumption
is satisfied. The separation test above is a deterministic gap statistic on an
observed distribution, not a hypothesis test, and yields no p-value.

REPRODUCIBILITY
---------------
Deterministic: no sampling, no randomness. The report records the script
version, the input root, a content hash over the file inventory, and every
parameter used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = "1.0.0"

# From train.py: only `random` is paired with NopPruner, so only its trials are
# guaranteed to have run the full epoch budget.
NON_PRUNING_SAMPLERS = {"random"}

PATH_CURVE = "MEASURED_FROM_CURVE"
PATH_ELAPSED = "ESTIMATED_FROM_ELAPSED"

FAST = "FAST_CLASS"
SLOW = "SLOW_CLASS"
UNCLASSIFIED = "UNCLASSIFIED"


def archive_hash(results_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(results_root.rglob("*.json")):
        rel = path.relative_to(results_root).as_posix()
        digest.update(f"{rel}:{path.stat().st_size}\n".encode())
    return digest.hexdigest()[:16]


def quantile(sorted_values, q):
    if not sorted_values:
        return None
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def speed_from_curve(record):
    """Median seconds per completed epoch. None if not derivable."""
    curve = record.get("accuracy_curve")
    if not isinstance(curve, dict) or len(curve) < 3:
        return None
    try:
        keys = sorted(curve, key=int)
    except (TypeError, ValueError):
        return None
    cumulative = []
    for key in keys:
        entry = curve[key]
        if not isinstance(entry, dict) or entry.get("gpu_hours") is None:
            return None
        cumulative.append(entry["gpu_hours"] * 3600.0)
    diffs = [cumulative[i + 1] - cumulative[i] for i in range(len(cumulative) - 1)]
    diffs = [d for d in diffs if d > 0]
    return statistics.median(diffs) if len(diffs) >= 2 else None


def collect(results_root: Path, study_dir: str, assumed_epochs: int):
    """Yield per-trial speed rows for one study. Never mixes measurement paths."""
    rows = []
    for path in sorted((results_root / study_dir).glob("trial_*.json")):
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        params = record.get("params") or {}
        # v1 trial records carry no `sampler` field; the directory name is the
        # only sampler evidence for them, and hpo_array.slurm names the results
        # directory after the sampler.
        sampler = record.get("sampler") or (study_dir if study_dir in
                                            {"random", "tpe", "hyperband"} else "")

        speed = speed_from_curve(record)
        if speed is not None:
            measurement_path = PATH_CURVE
            excluded = ""
        elif record.get("elapsed_s") is not None:
            if sampler in NON_PRUNING_SAMPLERS and assumed_epochs > 0:
                speed = record["elapsed_s"] / assumed_epochs
                measurement_path = PATH_ELAPSED
                excluded = ""
            else:
                speed, measurement_path = None, ""
                excluded = ("elapsed_s only, and the configured pruner can stop a "
                            "trial early, so completed epoch count is unknown")
        else:
            speed, measurement_path = None, ""
            excluded = "no accuracy_curve and no elapsed_s"

        rows.append({
            "study_dir": study_dir,
            "trial_number": record.get("trial_number", ""),
            "slurm_task": record.get("slurm_task", "") or "",
            "sampler": sampler,
            "dataset": record.get("dataset", "") or "",
            "model": params.get("model", "") or "",
            "batch_size": params.get("batch_size", "") or "",
            "w_nominal": record.get("parallel_workers", "") or "",
            "seconds_per_epoch": round(speed, 4) if speed is not None else "",
            "measurement_path": measurement_path,
            "exclusion_reason": excluded,
            "total_gpu_hours": record.get("total_gpu_hours", "") or "",
            "elapsed_s": record.get("elapsed_s", "") or "",
            "source_file": path.name,
        })
    return rows


def test_strata(rows, min_n, separation_ratio):
    """Per-stratum separation test. Returns (strata_summary, class_by_row_id)."""
    strata = defaultdict(list)
    for idx, row in enumerate(rows):
        if row["seconds_per_epoch"] == "":
            continue
        # Stratum must also fix the measurement path: the two paths are
        # different instruments and must not be compared to each other.
        key = (row["measurement_path"], row["dataset"], row["model"],
               str(row["batch_size"]))
        strata[key].append((idx, row["seconds_per_epoch"]))

    summaries, assignment = [], {}
    for key, members in sorted(strata.items()):
        values = sorted(v for _, v in members)
        baseline = quantile(values, 0.05) or values[0]
        relative = [v / baseline for v in values]

        best_gap, split_at = 1.0, None
        for i in range(len(relative) - 1):
            if relative[i] <= 0:
                continue
            ratio = relative[i + 1] / relative[i]
            if ratio > best_gap:
                best_gap, split_at = ratio, (values[i] + values[i + 1]) / 2.0

        separated = (len(values) >= min_n and split_at is not None
                     and best_gap >= separation_ratio)
        n_fast = n_slow = 0
        if separated:
            for idx, value in members:
                cls = FAST if value < split_at else SLOW
                assignment[idx] = cls
                n_fast += cls == FAST
                n_slow += cls == SLOW
        else:
            for idx, _ in members:
                assignment[idx] = UNCLASSIFIED

        fast_vals = [v for v in values if split_at and v < split_at]
        slow_vals = [v for v in values if split_at and v >= split_at]
        summaries.append({
            "measurement_path": key[0],
            "dataset": key[1],
            "model": key[2],
            "batch_size": key[3],
            "n": len(values),
            "min_s_per_epoch": round(values[0], 3),
            "median_s_per_epoch": round(statistics.median(values), 3),
            "max_s_per_epoch": round(values[-1], 3),
            "largest_consecutive_ratio": round(best_gap, 3),
            "split_point_s_per_epoch": (round(split_at, 3) if split_at else ""),
            "separation_verdict": ("SEPARATED" if separated else
                                   ("NOT_SEPARATED" if len(values) >= min_n
                                    else "UNDERPOWERED_BELOW_MIN_N")),
            "n_fast_class": n_fast,
            "n_slow_class": n_slow,
            "class_median_ratio": (round(statistics.median(slow_vals)
                                         / statistics.median(fast_vals), 2)
                                   if separated and fast_vals and slow_vals else ""),
        })
    return summaries, assignment


PER_TRIAL_FIELDS = [
    "study_dir", "trial_number", "slurm_task", "sampler", "dataset", "model",
    "batch_size", "w_nominal", "seconds_per_epoch", "measurement_path",
    "speed_class", "exclusion_reason", "total_gpu_hours", "elapsed_s",
    "source_file",
]

STRATA_FIELDS = [
    "measurement_path", "dataset", "model", "batch_size", "n",
    "min_s_per_epoch", "median_s_per_epoch", "max_s_per_epoch",
    "largest_consecutive_ratio", "split_point_s_per_epoch",
    "separation_verdict", "n_fast_class", "n_slow_class", "class_median_ratio",
]

STUDY_FIELDS = [
    "study_dir", "w_nominal_label", "n_trials_with_speed", "n_fast_class",
    "n_slow_class", "n_unclassified", "pct_slow_class",
    "median_gpu_hours_fast_class", "median_gpu_hours_slow_class",
    "gpu_hours_ratio",
]


def by_study(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["study_dir"]].append(row)
    out = []
    for study, members in sorted(grouped.items()):
        with_speed = [r for r in members if r["seconds_per_epoch"] != ""]
        fast = [r for r in with_speed if r["speed_class"] == FAST]
        slow = [r for r in with_speed if r["speed_class"] == SLOW]
        unc = [r for r in with_speed if r["speed_class"] == UNCLASSIFIED]
        classified = len(fast) + len(slow)

        def med_gpu(subset):
            vals = [r["total_gpu_hours"] for r in subset
                    if isinstance(r["total_gpu_hours"], (int, float))]
            return round(statistics.median(vals), 4) if vals else ""

        mf, ms = med_gpu(fast), med_gpu(slow)
        labels = {r["w_nominal"] for r in members if r["w_nominal"] != ""}
        out.append({
            "study_dir": study,
            "w_nominal_label": "/".join(str(v) for v in sorted(labels)),
            "n_trials_with_speed": len(with_speed),
            "n_fast_class": len(fast),
            "n_slow_class": len(slow),
            "n_unclassified": len(unc),
            "pct_slow_class": (round(100 * len(slow) / classified, 1)
                               if classified else ""),
            "median_gpu_hours_fast_class": mf,
            "median_gpu_hours_slow_class": ms,
            "gpu_hours_ratio": (round(ms / mf, 2)
                                if isinstance(mf, float) and isinstance(ms, float)
                                and mf > 0 else ""),
        })
    return out


def build_report(strata, studies, rows, params, meta):
    lines = []
    add = lines.append
    separated = [s for s in strata if s["separation_verdict"] == "SEPARATED"]

    add("# Aim 1 — Execution-Speed Heterogeneity Audit")
    add("")
    add(f"- Generated: {meta['generated']}")
    add(f"- Script version: {SCRIPT_VERSION}")
    add(f"- Archive root: `{meta['results_root']}`")
    add(f"- Archive content hash: `{meta['archive_hash']}`")
    add(f"- Parameters: min_stratum_n={params['min_stratum_n']}, "
        f"separation_ratio={params['separation_ratio']}, "
        f"assumed_epochs={params['assumed_epochs']}")
    add("")
    add("## 0. Scope of the claim")
    add("")
    add("This report establishes **that** per-trial execution speed varies, by how")
    add("much, and where. It does **not** establish **why**. The archive records no")
    add("node identity: `train.py` writes `device` as `str(device)`, which is")
    add("`\"cuda\"` for every GPU trial, and stores no hostname, no")
    add("`SLURMD_NODENAME`, no GPU name and no library versions. Any attribution to")
    add("a hardware or software cause would be unsupported by what is committed.")
    add("")
    add("`FAST_CLASS` and `SLOW_CLASS` name positions in a measured distribution.")
    add("They are not hardware claims.")
    add("")
    add("## 1. Measurement coverage")
    add("")
    counts = defaultdict(int)
    for row in rows:
        counts[row["measurement_path"] or "EXCLUDED"] += 1
    add("| measurement path | trials |")
    add("|---|---|")
    for key in (PATH_CURVE, PATH_ELAPSED, "EXCLUDED"):
        add(f"| {key} | {counts.get(key, 0)} |")
    add("")
    add(f"`{PATH_CURVE}` uses consecutive differences of the cumulative")
    add("`gpu_hours` series in `accuracy_curve`, so it needs no assumption about")
    add(f"epoch count. `{PATH_ELAPSED}` divides `elapsed_s` by")
    add(f"{params['assumed_epochs']} epochs and is applied only to the `random`")
    add("sampler, the one sampler `train.py` pairs with `NopPruner()`. Trials using")
    add("a pruning sampler with only `elapsed_s` are excluded, because the number of")
    add("epochs completed is unknown and dividing would fabricate it. The two paths")
    add("are separate strata and are never compared to each other.")
    add("")
    add("## 2. Separation test, per workload stratum")
    add("")
    add("| path | dataset | model | bs | n | min | median | max | largest ratio | split | verdict | fast/slow | class ratio |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in strata:
        add(f"| {s['measurement_path'].split('_')[0]} | {s['dataset'] or '—'} | "
            f"{s['model'] or '—'} | {s['batch_size'] or '—'} | {s['n']} | "
            f"{s['min_s_per_epoch']} | {s['median_s_per_epoch']} | "
            f"{s['max_s_per_epoch']} | {s['largest_consecutive_ratio']} | "
            f"{s['split_point_s_per_epoch'] or '—'} | {s['separation_verdict']} | "
            f"{s['n_fast_class']}/{s['n_slow_class']} | "
            f"{s['class_median_ratio'] or '—'} |")
    add("")
    if separated:
        ratios = [s["class_median_ratio"] for s in separated
                  if isinstance(s["class_median_ratio"], float)]
        add(f"**{len(separated)} of {len(strata)} strata separate** at the configured")
        add(f"ratio of {params['separation_ratio']}.")
        if ratios:
            add(f"Class median ratio across separated strata: min {min(ratios)}, "
                f"median {statistics.median(ratios)}, max {max(ratios)}.")
        add("")
        add("Each separated stratum holds dataset, model and batch size fixed, so the")
        add("difference is not workload. The SLURM scripts request")
        add("`--gres=gpu:1 --nodes=1 --ntasks=1`, so each array task held its own GPU;")
        add("contention between concurrent workers for one device is therefore not a")
        add("candidate explanation either.")
    else:
        add("No stratum separated at the configured ratio.")
    add("")
    add("## 3. Class mixture per study")
    add("")
    add("| study_dir | w_nominal | trials w/ speed | fast | slow | unclassified | % slow | median GPU-h fast | median GPU-h slow | ratio |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for s in studies:
        add(f"| {s['study_dir']} | {s['w_nominal_label'] or '—'} | "
            f"{s['n_trials_with_speed']} | {s['n_fast_class']} | {s['n_slow_class']} | "
            f"{s['n_unclassified']} | {s['pct_slow_class'] or '—'} | "
            f"{s['median_gpu_hours_fast_class'] or '—'} | "
            f"{s['median_gpu_hours_slow_class'] or '—'} | "
            f"{s['gpu_hours_ratio'] or '—'} |")
    add("")
    add("## 4. Consequence for GPU-hour comparisons")
    add("")
    add("GPU-hours are a product of workload and execution speed. Where the class")
    add("mixture differs between two studies, a difference in mean GPU-hours between")
    add("them is not attributable to the HPO method: it is confounded with whichever")
    add("mechanism produces the speed classes.")
    add("")
    add("This bears directly on the efficiency claims in the repository README and in")
    add("`analysis_output/analysis/output_v2/table1_summary.csv`, whose `Mean GPU")
    add("Hours` column aggregates trials across studies with different mixtures. Those")
    add("claims should be treated as unresolved until either the mixture is balanced")
    add("or the comparison is made within a single class.")
    add("")
    add("It also bears on `03_delay_reconstruction.py`: a worker whose trial ran in")
    add("the slow class accumulates more intervening asks at fixed concurrency, so")
    add("`d_observed` is biased upward for slow-class trials. The `speed_class`")
    add("column here joins to `delay_per_trial.csv` on (study_dir, trial_number) so")
    add("that bias can be examined directly.")
    add("")
    add("## 5. Candidate causes, and what would distinguish them")
    add("")
    add("None of these is asserted. Each is listed with the evidence that would")
    add("settle it, none of which is present in the committed archive.")
    add("")
    add("| Candidate cause | Evidence that would distinguish it |")
    add("|---|---|")
    add("| Heterogeneous GPU models within the partition | GPU name per trial, e.g. `torch.cuda.get_device_name(0)`, which `train.py` logs but does not persist |")
    add("| Different physical nodes with different CPU or I/O throughput | `SLURMD_NODENAME` per trial; echoed to the SLURM logs, never written to JSON |")
    add("| Dataloader throughput differences (`--num-workers`, filesystem state) | Recorded `num_workers` and data-path timing per trial |")
    add("| Driver, CUDA or PyTorch version change between campaigns | Library versions per trial; printed by both SLURM scripts, not persisted |")
    add("| Contention from other users' jobs on a shared node | Node-level utilisation traces, external to this project |")
    add("| Two separate execution campaigns months apart | Per-trial timestamps, present in the Optuna RDB (`datetime_start`, `datetime_complete`) but not committed |")
    add("")
    add("The single artifact that would resolve most of these at once is the Optuna")
    add("SQLite database, declared at `$PROJECT_DIR/optuna_storage/` in")
    add("`hpo_array.slurm:70` and `hpo_v2.slurm:71`.")
    add("")
    add("## 6. No inferential statistics were computed")
    add("")
    add("The separation test is a deterministic gap statistic on an observed")
    add("distribution and produces no p-value. No bootstrap confidence interval,")
    add("Mann-Whitney U test, multiple-comparison correction or risk-adjusted score")
    add("appears anywhere in this report. Trials within a study come from one shared")
    add("adaptive Optuna study and are not independent, and there is one independent")
    add("study per cell, so neither the bootstrap's resampling assumption nor")
    add("Mann-Whitney's exchangeability assumption is satisfied.")
    add("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit per-trial execution-speed heterogeneity without attributing cause.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("./aim1_output"))
    parser.add_argument("--min-stratum-n", type=int, default=8)
    parser.add_argument("--separation-ratio", type=float, default=1.8)
    parser.add_argument("--assumed-epochs", type=int, default=30)
    args = parser.parse_args(argv)

    if not args.results_root.is_dir():
        print(f"ERROR: --results-root is not a directory: {args.results_root}",
              file=sys.stderr)
        return 2
    if args.separation_ratio <= 1.0:
        print("ERROR: --separation-ratio must exceed 1.0", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for directory in sorted(d for d in args.results_root.iterdir() if d.is_dir()):
        rows.extend(collect(args.results_root, directory.name, args.assumed_epochs))
    if not rows:
        print(f"ERROR: no trial records under {args.results_root}", file=sys.stderr)
        return 2

    strata, assignment = test_strata(rows, args.min_stratum_n, args.separation_ratio)
    for idx, row in enumerate(rows):
        row["speed_class"] = assignment.get(idx, UNCLASSIFIED if
                                            row["seconds_per_epoch"] != "" else "")
    studies = by_study(rows)

    out = args.output_dir

    def write(path, fields, data):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in data:
                writer.writerow({k: item.get(k, "") for k in fields})

    write(out / "execution_speed_per_trial.csv", PER_TRIAL_FIELDS, rows)
    write(out / "execution_speed_strata.csv", STRATA_FIELDS, strata)
    write(out / "execution_speed_by_study.csv", STUDY_FIELDS, studies)

    params = {
        "min_stratum_n": args.min_stratum_n,
        "separation_ratio": args.separation_ratio,
        "assumed_epochs": args.assumed_epochs,
    }
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results_root": str(args.results_root),
        "archive_hash": archive_hash(args.results_root),
        "script_version": SCRIPT_VERSION,
    }
    (out / "execution_speed_report.md").write_text(
        build_report(strata, studies, rows, params, meta))
    (out / "execution_speed_report.json").write_text(json.dumps(
        {"meta": meta, "params": params, "strata": strata, "by_study": studies},
        indent=2) + "\n")

    n_speed = sum(1 for r in rows if r["seconds_per_epoch"] != "")
    n_sep = sum(1 for s in strata if s["separation_verdict"] == "SEPARATED")
    n_slow = sum(1 for r in rows if r["speed_class"] == SLOW)
    n_fast = sum(1 for r in rows if r["speed_class"] == FAST)
    print(f"Trial records: {len(rows)}   with a speed measurement: {n_speed}")
    print(f"Workload strata tested: {len(strata)}   separated: {n_sep}")
    print(f"Classified trials: FAST {n_fast}  SLOW {n_slow}")
    print("Cause NOT attributed; see section 5 of execution_speed_report.md.")
    print(f"Wrote outputs to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
