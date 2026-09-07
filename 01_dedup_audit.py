#!/usr/bin/env python3
"""
01_dedup_audit.py — Aim 1, step 1: trial archive deduplication audit.

PURPOSE
-------
Reconcile the raw BudgetAware-HPO trial archive into an explicit, auditable
analysis set, and report exactly what was dropped and why.

This script computes NO statistics. It does not compute bootstrap confidence
intervals, Mann-Whitney U tests, or Sharpe-style scores. See the
"STATISTICS ARE BLOCKED" section of the generated report for why.

REQUIRED INPUT
--------------
--results-root  Path to the archive root, i.e. the directory that CONTAINS the
                per-study subdirectories. In the repository as committed this is:
                    results_zip/results/
                Each immediate subdirectory of this path is treated as ONE
                shared Optuna study (see "STUDY IDENTITY" below).

                No other input is required. The script does not read the Optuna
                SQLite databases (they are not committed to the repository) and
                does not read SLURM logs (that is 02_provenance_reconstruction.py).

OPTIONAL INPUT
--------------
--output-dir        Where to write outputs. Default: ./aim1_output
--collision-policy  How to handle records that collide on the dedup key.
                    quarantine (default) | keep-all
                    There is deliberately NO "pick one" policy. See COLLISIONS.

OUTPUTS
-------
<output-dir>/trials_deduplicated.csv   The analysis set: one row per unique
                                       dedup key, collisions excluded (under the
                                       default policy).
<output-dir>/collisions.csv            Every record involved in a dedup-key
                                       collision, with a collision class.
<output-dir>/study_snapshots.csv       The summary_*.json records, kept separate.
                                       These are NOT trials and are NOT part of
                                       the analysis set.
<output-dir>/file_inventory.csv        Every file seen, its classification, and
                                       its disposition. Full audit trail.
<output-dir>/dedup_audit_report.md     Human-readable reconciliation report.
<output-dir>/dedup_audit_report.json   Same numbers, machine-readable.

STUDY IDENTITY
--------------
The JSON records do NOT contain a study name, storage URL, job ID, or timestamp.
Study identity is therefore recovered from the DIRECTORY, which is justified by
the committed orchestration scripts:

  hpo_v2.slurm:
      RESULTS_DIR="$PROJECT_DIR/results/v2_${SAMPLER}_${DATASET}_w${N_WORKERS}"
      STUDY_NAME="hpo_v2_${SAMPLER}_${DATASET}_w${N_WORKERS}"
  hpo_array.slurm:
      RESULTS_DIR="$PROJECT_DIR/results/${SAMPLER}"
      STUDY_NAME="hpo_cifar10_${SAMPLER}"

and by train.py:
      optuna.create_study(..., load_if_exists=True)

One results directory therefore corresponds to exactly one shared Optuna study,
and every SLURM array task writing into it joined that same study.

CONSEQUENCE, STATED PLAINLY: the number of INDEPENDENT STUDIES per
(sampler, dataset, declared W) cell is 1. Deduplication does not and cannot
change this. Trials within a study are not independent replicates: they were
produced by one adaptive sampler against one shared trial history.

DEDUPLICATION KEY
-----------------
    KEY = (study_dir, trial_number)

where study_dir is the immediate subdirectory name and trial_number is the
`trial_number` field written by train.py from Optuna's `trial.number`, which is
unique WITHIN a study by construction.

Rationale: trial.number is assigned by the shared Optuna storage, so it is the
only identifier in the data with study-scoped uniqueness semantics. The filename
(`trial_{number:05d}_task{task:04d}.json`) additionally encodes slurm_task, but
task ID is a writer identity, not a trial identity — including it in the key
would preserve exactly the duplicates this audit exists to surface.

summary_*.json records are excluded from the key entirely. They are per-worker
snapshots of study-level state (`study.best_value`), so N workers produce N
records describing ONE study. They are inventoried and reported, never analysed.

COLLISION DETECTABILITY (read before trusting any collision count)
------------------------------------------------------------------
The archive contains two filename schemes, and only one of them can record a
collision at all:

  V2_TASK_SUFFIX     trial_{number:05d}_task{task:04d}.json
                     Written by the committed train.py. Two workers assigned the
                     same trial_number write to DIFFERENT filenames, so the
                     collision survives on disk and this script can see it.
                     Directories: all v2_*.

  V1_NO_TASK_SUFFIX  trial_{number:05d}.json
                     No task ID in the filename. Two workers assigned the same
                     trial_number write to the SAME path, so the second write
                     OVERWRITES the first. Any collision in these directories was
                     destroyed at collection time and is unrecoverable from the
                     archive.
                     Directories: random/, tpe/, hyperband/, test/.

A collision count of zero in a V1 directory therefore means "not detectable",
NOT "none occurred". The report labels this per study and never reports a V1
zero as a clean result.

Note also that no committed version of train.py produces the V1 records: the
current and the initial committed train.py both write `total_gpu_hours` and the
task-suffixed filename, while the V1 records carry `elapsed_s` and no
`sampler` / `dataset` / `parallel_workers` fields. The V1 archive was produced by
an uncommitted script version. This is recorded, not worked around.

RUNTIME FIELDS ARE NOT UNIFIED
------------------------------
V2 records carry `total_gpu_hours`; V1 records carry `elapsed_s`. They are
emitted as two separate columns and are never merged or converted. They were
written by different, and in the V1 case uninspectable, code, so treating them
as one instrument would be an assumption presented as data.

COLLISIONS
----------
A collision is two or more records sharing one dedup key. In this archive the
colliding records are NOT byte-identical copies: they carry different
`best_val_acc` and different `total_gpu_hours`. They are classified as:

  SAME_CONFIG       identical `params`, differing outcome.
                    One configuration evaluated more than once.
  DIFFERENT_CONFIG  differing `params` under one trial_number.
                    The identity key is violated in the source data.

Neither class is a redundant copy that can be deleted. Choosing one record per
key would be a silent, unrecorded decision about which measurement counts, so
this script refuses to make it. Default policy is `quarantine`: colliding
records are removed from the analysis set and written in full to collisions.csv
for a human decision, which must then be recorded in the repository.

Use --collision-policy keep-all only to inspect the raw set. Records emitted
under keep-all are flagged `in_collision=True` and MUST NOT be fed to any
statistical routine.

NO VALUES ARE INVENTED
----------------------
Fields absent from a JSON record are written as empty. In particular
`parallel_workers`, `dataset`, and `sampler` are absent from the v1 archive
directories, and this script leaves them empty rather than inferring them.
Inference is the job of 02_provenance_reconstruction.py, which labels its
inferences as such.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Two filename schemes exist in the archive and are handled separately.
#   V2_TASK_SUFFIX      trial_{number:05d}_task{task:04d}.json   (v2_* directories)
#   V1_NO_TASK_SUFFIX   trial_{number:05d}.json                  (random/, tpe/,
#                                                                 hyperband/, test/)
# The distinction is load-bearing: see COLLISION DETECTABILITY below.
TRIAL_V2_RE = re.compile(r"^trial_(\d+)_task(\d+)\.json$")
TRIAL_V1_RE = re.compile(r"^trial_(\d+)\.json$")
SUMMARY_FILE_RE = re.compile(r"^summary_(.+?)_(\d+)\.json$")

SCHEME_V2 = "V2_TASK_SUFFIX"
SCHEME_V1 = "V1_NO_TASK_SUFFIX"

DETECTABLE = "DETECTABLE"
NOT_DETECTABLE = "NOT_DETECTABLE_OVERWRITE_ON_DISK"

CLASS_TRIAL = "TRIAL_RECORD"
CLASS_SNAPSHOT = "STUDY_SNAPSHOT"
CLASS_UNCLASSIFIED = "UNCLASSIFIED"
CLASS_UNREADABLE = "UNREADABLE"

TRIAL_FIELDS = [
    "study_dir", "trial_number", "slurm_task", "sampler", "dataset",
    "parallel_workers_declared", "best_val_acc", "total_gpu_hours", "elapsed_s",
    "device", "params_json", "params_hash", "n_epochs_recorded",
    "accuracy_thresholds_json", "filename_scheme", "collision_detectability",
    "source_file", "in_collision",
]


def params_fingerprint(params) -> str:
    if params is None:
        return ""
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def get(record: dict, key: str):
    """Return the field or empty string. Never substitutes a default value."""
    value = record.get(key)
    return "" if value is None else value


def scan_archive(results_root: Path):
    """Walk the archive. Returns (inventory, trials, snapshots, study_dirs)."""
    inventory, trials, snapshots = [], [], []
    study_dirs = sorted(
        d.name for d in results_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    for study_dir in study_dirs:
        for path in sorted((results_root / study_dir).glob("*.json")):
            name = path.name
            row = {
                "study_dir": study_dir,
                "file": name,
                "classification": CLASS_UNCLASSIFIED,
                "disposition": "",
                "note": "",
            }
            try:
                record = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                row["classification"] = CLASS_UNREADABLE
                row["disposition"] = "excluded"
                row["note"] = f"unreadable: {type(exc).__name__}"
                inventory.append(row)
                continue

            v2_match = TRIAL_V2_RE.match(name)
            v1_match = TRIAL_V1_RE.match(name)
            summary_match = SUMMARY_FILE_RE.match(name)
            scheme = SCHEME_V2 if v2_match else (SCHEME_V1 if v1_match else "")
            detectability = DETECTABLE if scheme == SCHEME_V2 else NOT_DETECTABLE

            if scheme and "trial_number" in record:
                row["classification"] = CLASS_TRIAL
                row["note"] = f"scheme={scheme}"
                curve = record.get("accuracy_curve") or {}
                trials.append({
                    "study_dir": study_dir,
                    "trial_number": record["trial_number"],
                    "slurm_task": get(record, "slurm_task"),
                    "sampler": get(record, "sampler"),
                    "dataset": get(record, "dataset"),
                    "parallel_workers_declared": get(record, "parallel_workers"),
                    "best_val_acc": get(record, "best_val_acc"),
                    "total_gpu_hours": get(record, "total_gpu_hours"),
                    "elapsed_s": get(record, "elapsed_s"),
                    "device": get(record, "device"),
                    "params_json": json.dumps(record.get("params"), sort_keys=True)
                                   if record.get("params") is not None else "",
                    "params_hash": params_fingerprint(record.get("params")),
                    "n_epochs_recorded": len(curve) if isinstance(curve, dict) else "",
                    "accuracy_thresholds_json": json.dumps(
                        record.get("accuracy_thresholds"), sort_keys=True)
                        if record.get("accuracy_thresholds") is not None else "",
                    "filename_scheme": scheme,
                    "collision_detectability": detectability,
                    "source_file": name,
                    "in_collision": False,
                })
            elif summary_match:
                row["classification"] = CLASS_SNAPSHOT
                row["disposition"] = "excluded_from_analysis_set"
                row["note"] = "per-worker snapshot of shared study state; not a trial"
                snapshots.append({
                    "study_dir": study_dir,
                    "slurm_task": get(record, "slurm_task"),
                    "sampler": get(record, "sampler"),
                    "dataset": get(record, "dataset"),
                    "parallel_workers_declared": get(record, "parallel_workers"),
                    "n_trials_completed": get(record, "n_trials_completed"),
                    "best_trial": get(record, "best_trial"),
                    "study_best_val_acc": get(record, "best_val_acc"),
                    "source_file": name,
                })
            else:
                row["disposition"] = "excluded"
                row["note"] = ("filename does not match trial_/summary_ pattern, "
                               "or trial_number field is absent")
            inventory.append(row)

    return inventory, trials, snapshots, study_dirs


def find_collisions(trials):
    """Group trials by dedup key and classify multi-record keys."""
    by_key = defaultdict(list)
    for trial in trials:
        by_key[(trial["study_dir"], trial["trial_number"])].append(trial)

    collisions = {}
    for key, group in by_key.items():
        if len(group) < 2:
            continue
        hashes = {t["params_hash"] for t in group}
        collisions[key] = "SAME_CONFIG" if len(hashes) == 1 else "DIFFERENT_CONFIG"
    return by_key, collisions


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def build_report(stats, per_study, collision_detail, policy, results_root):
    lines = []
    add = lines.append

    add("# Aim 1 — Deduplication Audit Report")
    add("")
    add(f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    add(f"- Archive root: `{results_root}`")
    add(f"- Collision policy: `{policy}`")
    add("- Dedup key: `(study_dir, trial_number)`")
    add("")
    add("## 1. Archive reconciliation")
    add("")
    add("| Quantity | Count |")
    add("|---|---|")
    add(f"| Raw JSON files scanned | {stats['files_scanned']} |")
    add(f"| Classified TRIAL_RECORD | {stats['trial_records']} |")
    add(f"| Classified STUDY_SNAPSHOT (excluded, not trials) | {stats['snapshots']} |")
    add(f"| Unclassified / unreadable (excluded) | {stats['unclassified']} |")
    add(f"| Unique dedup keys among trial records | {stats['unique_keys']} |")
    add(f"| Keys with a collision | {stats['colliding_keys']} |")
    add(f"| Trial records involved in a collision | {stats['records_in_collision']} |")
    add(f"| — of which SAME_CONFIG | {stats['same_config_records']} |")
    add(f"| — of which DIFFERENT_CONFIG | {stats['different_config_records']} |")
    add(f"| Records in final analysis set | {stats['analysis_set_size']} |")
    add(f"| Records removed from analysis set | {stats['records_removed']} |")
    add("")
    add("## 2. Independent-study count")
    add("")
    add(f"Study directories found: **{stats['n_studies']}**")
    add("")
    add("Each directory is one shared Optuna study (`load_if_exists=True` against a")
    add("single per-directory storage URL). The number of INDEPENDENT studies per")
    add("(sampler, dataset, declared W) cell is therefore **1**, before and after")
    add("deduplication. Trials within a study share one adaptive sampler and one")
    add("trial history and are not independent replicates.")
    add("")
    add("| study_dir | raw files | trial records | unique keys | colliding keys | collisions detectable? | snapshots | distinct snapshot best_val_acc | independent studies |")
    add("|---|---|---|---|---|---|---|---|---|")
    for study, row in sorted(per_study.items()):
        detectable = "yes" if row["collision_detectability"] == DETECTABLE else "NO"
        add(f"| {study} | {row['files']} | {row['trials']} | {row['unique_keys']} | "
            f"{row['colliding_keys']} | {detectable} | {row['snapshots']} | "
            f"{row['distinct_snapshot_acc']} | 1 |")
    add("")
    add("The `distinct snapshot best_val_acc` column is the duplication this audit")
    add("targets: N per-worker summary files reporting few or one distinct value are")
    add("N views of one study, not N measurements.")
    add("")
    add("Studies marked **NO** under `collisions detectable?` use the")
    add("`trial_{number:05d}.json` filename scheme, which has no task ID. Concurrent")
    add("workers assigned the same trial_number overwrote each other on disk at")
    add("collection time. A zero collision count for those studies means NOT")
    add("MEASURABLE, not clean. The archive cannot answer the question for them, and")
    add("no amount of post-hoc processing can recover it.")
    add("")
    add("## 3. Collisions")
    add("")
    if not collision_detail:
        add("No dedup-key collisions found.")
    else:
        add("Colliding records are NOT byte-identical duplicates. They carry different")
        add("outcomes under one trial identity, so none is a redundant copy that can be")
        add("deleted. All are quarantined in `collisions.csv` for a recorded human")
        add("decision. This script does not choose between them.")
        add("")
        add("| study_dir | trial_number | class | records | slurm_tasks | best_val_acc values |")
        add("|---|---|---|---|---|---|")
        for (study, number), info in sorted(collision_detail.items()):
            tasks = ", ".join(str(t) for t in info["tasks"])
            accs = ", ".join(str(a) for a in info["accs"])
            add(f"| {study} | {number} | {info['class']} | {info['n']} | {tasks} | {accs} |")
        add("")
        add("**SAME_CONFIG** — identical `params`, differing outcome. One configuration")
        add("evaluated more than once. These are the only repeat evaluations anywhere in")
        add("the archive, and they are incidental, not designed.")
        add("")
        add("**DIFFERENT_CONFIG** — differing `params` under one `trial_number`. The")
        add("study-scoped uniqueness of `trial.number` is violated in the source data.")
        add("This must be explained from the Eagle SLURM logs before these records are")
        add("used for anything.")
    add("")
    add("## 4. STATISTICS ARE BLOCKED")
    add("")
    add("This script does not compute bootstrap confidence intervals, Mann-Whitney U")
    add("tests, multiple-comparison corrections, or Sharpe-style scores, and no")
    add("downstream script should, until all of the following are resolved and each")
    add("resolution is recorded in `11_REPO_COMMIT_LEDGER.md`:")
    add("")
    add("1. **Independent replicates.** Independent studies per cell is 1. Bootstrap")
    add("   CIs over trials within one adaptive study measure within-study spread,")
    add("   not run-to-run variance, and the exchangeability assumption that both the")
    add("   bootstrap and Mann-Whitney U require does not hold.")
    add("2. **Seeding.** `train.py` fixes `seed=42` for the TPE, Hyperband and CMA-ES")
    add("   samplers for every worker. Only Random Search varies (`seed=42 + trial_id`).")
    add("   No seed replicates exist in this archive by construction.")
    add("3. **Provenance of W.** Run `02_provenance_reconstruction.py`. Worker count is")
    add("   absent from the v1 directories and contradicted in at least two v2")
    add("   directories. Comparisons across W cannot be made until W is established.")
    add("4. **Collision disposition.** Every quarantined collision needs a recorded")
    add("   decision.")
    add("5. **Undetectable collisions.** The V1 directories cannot report collisions")
    add("   at all, and the code that produced them is not committed. Their trial")
    add("   counts are upper bounds of unknown tightness.")
    add("6. **Sharpe specification.** Accuracy has no risk-free rate, so a mean/sigma")
    add("   ratio on raw accuracy is uninterpretable. Fix the numerator (e.g. regret")
    add("   against a per-dataset best-known configuration) before any code is written.")
    add("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Aim 1 deduplication audit for the BudgetAware-HPO trial archive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-root", required=True, type=Path,
                        help="Directory containing the per-study subdirectories "
                             "(repository path: results_zip/results)")
    parser.add_argument("--output-dir", type=Path, default=Path("./aim1_output"))
    parser.add_argument("--collision-policy", choices=["quarantine", "keep-all"],
                        default="quarantine",
                        help="quarantine (default): exclude colliding records from "
                             "the analysis set. keep-all: emit them flagged, for "
                             "inspection only.")
    args = parser.parse_args(argv)

    if not args.results_root.is_dir():
        print(f"ERROR: --results-root is not a directory: {args.results_root}",
              file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    inventory, trials, snapshots, study_dirs = scan_archive(args.results_root)
    if not study_dirs:
        print(f"ERROR: no study subdirectories under {args.results_root}",
              file=sys.stderr)
        return 2

    by_key, collisions = find_collisions(trials)

    collision_detail = {}
    for key, klass in collisions.items():
        group = by_key[key]
        collision_detail[key] = {
            "class": klass,
            "n": len(group),
            "tasks": [t["slurm_task"] for t in group],
            "accs": [t["best_val_acc"] for t in group],
        }

    collision_rows, analysis_rows = [], []
    for key, group in by_key.items():
        if key in collisions:
            for trial in group:
                row = dict(trial)
                row["in_collision"] = True
                row["collision_class"] = collisions[key]
                collision_rows.append(row)
                if args.collision_policy == "keep-all":
                    analysis_rows.append(row)
        else:
            analysis_rows.append(group[0])

    analysis_rows.sort(key=lambda r: (r["study_dir"], r["trial_number"],
                                      str(r["slurm_task"])))
    collision_rows.sort(key=lambda r: (r["study_dir"], r["trial_number"],
                                       str(r["slurm_task"])))

    # Per-study reconciliation.
    file_counts = Counter(row["study_dir"] for row in inventory)
    per_study = {}
    for study in study_dirs:
        study_trials = [t for t in trials if t["study_dir"] == study]
        keys = {(t["study_dir"], t["trial_number"]) for t in study_trials}
        study_snaps = [s for s in snapshots if s["study_dir"] == study]
        schemes = {t["filename_scheme"] for t in study_trials}
        detect = {t["collision_detectability"] for t in study_trials}
        per_study[study] = {
            "filename_scheme": "/".join(sorted(schemes)) if schemes else "",
            "collision_detectability": ("/".join(sorted(detect)) if detect
                                        else "NO_TRIAL_RECORDS"),
            "files": file_counts.get(study, 0),
            "trials": len(study_trials),
            "unique_keys": len(keys),
            "colliding_keys": sum(1 for k in keys if k in collisions),
            "snapshots": len(study_snaps),
            "distinct_snapshot_acc": len(
                {s["study_best_val_acc"] for s in study_snaps
                 if s["study_best_val_acc"] != ""}
            ),
        }

    records_in_collision = len(collision_rows)
    same_cfg = sum(d["n"] for d in collision_detail.values()
                   if d["class"] == "SAME_CONFIG")
    diff_cfg = sum(d["n"] for d in collision_detail.values()
                   if d["class"] == "DIFFERENT_CONFIG")

    stats = {
        "files_scanned": len(inventory),
        "trial_records": len(trials),
        "snapshots": len(snapshots),
        "unclassified": sum(1 for r in inventory
                            if r["classification"] in (CLASS_UNCLASSIFIED,
                                                       CLASS_UNREADABLE)),
        "unique_keys": len(by_key),
        "colliding_keys": len(collisions),
        "records_in_collision": records_in_collision,
        "same_config_records": same_cfg,
        "different_config_records": diff_cfg,
        "analysis_set_size": len(analysis_rows),
        "records_removed": len(trials) - len(analysis_rows),
        "n_studies": len(study_dirs),
        "independent_studies_per_study_dir": 1,
        "collision_policy": args.collision_policy,
    }

    out = args.output_dir
    write_csv(out / "trials_deduplicated.csv", TRIAL_FIELDS, analysis_rows)
    write_csv(out / "collisions.csv", TRIAL_FIELDS + ["collision_class"],
              collision_rows)
    write_csv(out / "study_snapshots.csv",
              ["study_dir", "slurm_task", "sampler", "dataset",
               "parallel_workers_declared", "n_trials_completed", "best_trial",
               "study_best_val_acc", "source_file"], snapshots)
    write_csv(out / "file_inventory.csv",
              ["study_dir", "file", "classification", "disposition", "note"],
              inventory)

    (out / "dedup_audit_report.md").write_text(
        build_report(stats, per_study, collision_detail,
                     args.collision_policy, args.results_root)
    )
    (out / "dedup_audit_report.json").write_text(json.dumps(
        {"stats": stats,
         "per_study": per_study,
         "collisions": [{"study_dir": k[0], "trial_number": k[1], **v}
                        for k, v in sorted(collision_detail.items())]},
        indent=2) + "\n")

    print(f"Studies (= independent studies): {stats['n_studies']}")
    print(f"Raw files: {stats['files_scanned']}  "
          f"trial records: {stats['trial_records']}  "
          f"snapshots excluded: {stats['snapshots']}")
    print(f"Unique dedup keys: {stats['unique_keys']}  "
          f"colliding keys: {stats['colliding_keys']}  "
          f"records in collision: {stats['records_in_collision']} "
          f"(SAME_CONFIG {same_cfg} / DIFFERENT_CONFIG {diff_cfg})")
    print(f"Analysis set: {stats['analysis_set_size']}  "
          f"removed: {stats['records_removed']}  "
          f"policy: {args.collision_policy}")
    print(f"Wrote outputs to {out}/")
    print("No statistics computed. See section 4 of dedup_audit_report.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
