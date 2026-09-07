#!/usr/bin/env python3
"""
02_provenance_reconstruction.py — Aim 1, step 2: establish worker count W per study.

PURPOSE
-------
Determine, for each shared Optuna study in the archive, what the parallel worker
count W actually was — and, where that cannot be determined, say so explicitly
rather than filling in a number.

W is the independent variable of the entire project. It is absent from part of
the archive and contradicted in another part. This script does not repair either
condition. It classifies every study by the strength of the evidence available
and produces a list of studies that block downstream analysis.

This script computes NO statistics, and deliberately implements no bootstrap
confidence intervals, no Mann-Whitney U tests, and no Sharpe-style scores.

REQUIRED INPUT
--------------
--results-root   Path to the archive root containing per-study subdirectories.
                 Repository path: results_zip/results/

OPTIONAL INPUT (each one strengthens the evidence class it feeds)
----------------------------------------------------------------
--slurm-script   Path to a committed orchestration script. Repeatable.
                 Repository paths: hpo_array.slurm, hpo_v2.slurm
                 Supplies the `#SBATCH --array=A-B%C` concurrency throttle C,
                 which is an UPPER BOUND on W, never a measurement of it.

--slurm-log-dir  Directory of SLURM stdout logs from the Eagle cluster.
                 NOT COMMITTED TO THE REPOSITORY. Expected filenames follow the
                 patterns declared in the scripts:
                     hpo_%x_%A_%a.out    (hpo_array.slurm)
                     hpo2_%x_%A_%a.out   (hpo_v2.slurm)
                 This is the ONLY input that can produce a CONFIRMED worker
                 count, because it is the only source containing per-task start
                 and end timestamps. Without it, nothing is confirmed.

--study-map      Optional JSON mapping {study_dir: slurm_script_path}, to
                 override the default mapping heuristic (see ASSUMPTIONS).

OUTPUTS
-------
<output-dir>/provenance_reconstruction.csv  One row per study, all evidence
                                            columns, plus a status.
<output-dir>/provenance_report.md           Human-readable report, split into
                                            confirmed / consistent / contradicted
                                            / inferred / unresolved sections.
<output-dir>/provenance_report.json         Machine-readable equivalent.
<output-dir>/unresolved_studies.txt          Study dirs that block analysis, one
                                            per line, for use as a gate in CI or
                                            in downstream scripts.
<output-dir>/log_parse_failures.csv          Any log file that could not be
                                            parsed, with the reason. Never
                                            silently skipped.

EVIDENCE CLASSES — the core of this script
------------------------------------------
Requirement: confirmed worker counts must be distinguishable from inferred or
unresolved ones. The status column takes exactly one of five values.

  CONFIRMED
      Concurrency was MEASURED from SLURM log timestamps: the maximum number of
      array tasks of one job whose [start, end] intervals overlapped. Requires
      --slurm-log-dir. `w_confirmed` is populated only in this case.

  DECLARED_CONSISTENT
      The `parallel_workers` field is present in the JSON records and equals the
      number of distinct `slurm_task` IDs observed writing into the study.
      This is a LABEL that nothing contradicts. It is NOT a measurement: the
      field is simply the `--parallel-workers` CLI argument echoed back by
      train.py, and equality with the task count does not establish that those
      tasks ran concurrently.

  DECLARED_PARTIAL
      The `parallel_workers` label is present and FEWER distinct slurm_task IDs
      were observed than the label claims. This is consistent with array tasks
      that failed or were cancelled before writing any output, so the label is
      neither corroborated nor refuted. Not usable as W.

  CONTRADICTED
      The `parallel_workers` label is present and MORE distinct slurm_task IDs
      were observed than the label permits. Since one directory is one shared
      study, more writers than declared workers means the label cannot be
      correct. The true W is unknown. `w_confirmed` stays empty.

  INFERRED_UPPER_BOUND
      No `parallel_workers` field in any record (the v1 directories), but a
      mapped orchestration script declares `#SBATCH --array=A-B%C`. C bounds
      concurrency from above. An upper bound is not a value: `w_confirmed`
      stays empty and the study still blocks W-comparative analysis.

  UNRESOLVED
      No label, no usable script evidence, no logs.

`w_confirmed` is populated ONLY for CONFIRMED. Every other row leaves it empty.
No status other than CONFIRMED may be used as W in any comparison across
worker counts.

WHAT THIS SCRIPT WILL NOT DO
----------------------------
- It will not copy `parallel_workers` into `w_confirmed`. That field is an
  argument label, and in this archive it is demonstrably wrong in at least two
  studies.
- It will not infer W from the number of distinct slurm_task IDs. That is a
  count of tasks that WROTE OUTPUT, which bounds neither concurrency nor the
  array size: tasks can fail before writing, and tasks can run sequentially.
- It will not infer W from a directory name suffix (`_w40`). The suffix is
  generated from the same CLI argument as the JSON label, so it is the same
  evidence twice, not corroboration.
- It will not treat a `%C` throttle as W. It is a cap.

ASSUMPTIONS, STATED SO THEY CAN BE CHECKED
------------------------------------------
1. STUDY IDENTITY. One results subdirectory = one shared Optuna study. Basis:
   `hpo_v2.slurm` sets RESULTS_DIR and STUDY_NAME from the same
   (SAMPLER, DATASET, N_WORKERS) triple; `hpo_array.slurm` sets both from
   SAMPLER; `train.py` calls create_study(..., load_if_exists=True).

2. SCRIPT MAPPING (heuristic, override with --study-map). Study dirs beginning
   `v2_` map to hpo_v2.slurm, whose RESULTS_DIR template is
   `results/v2_${SAMPLER}_${DATASET}_w${N_WORKERS}`. All others map to
   hpo_array.slurm, whose template is `results/${SAMPLER}`. The mapping basis is
   recorded per row in `script_mapping_basis` so it can be audited.

3. LOG PARSER IS UNTESTED AGAINST REAL DATA. No SLURM logs are committed to the
   repository, so the parser below was written against the `echo` statements in
   the committed scripts and has never been run on an actual Eagle log. If it
   fails on the real logs, it reports the failure in log_parse_failures.csv and
   downgrades the affected studies to their next-best evidence class. It never
   guesses a timestamp.

KNOWN ISSUE THIS SCRIPT WILL SURFACE, NOT FIX
---------------------------------------------
`hpo_v2.slurm` contains:
      #SBATCH --array=1-${N_WORKERS}%${N_WORKERS}
placed after the shell variable assignments. SLURM parses `#SBATCH` directives
before any shell expansion occurs, so this directive does not expand and does
not take effect. The array size actually used came from the `--array` argument
passed on the sbatch command line, which is not recorded anywhere in the
archive. This is why the JSON `parallel_workers` label and the observed task
count can disagree, and why the logs are the only route to a confirmed answer.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Status values, ordered by evidence strength.
CONFIRMED = "CONFIRMED"
DECLARED_CONSISTENT = "DECLARED_CONSISTENT"
DECLARED_PARTIAL = "DECLARED_PARTIAL"
CONTRADICTED = "CONTRADICTED"
INFERRED_UPPER_BOUND = "INFERRED_UPPER_BOUND"
UNRESOLVED = "UNRESOLVED"

BLOCKING = {DECLARED_PARTIAL, CONTRADICTED, INFERRED_UPPER_BOUND, UNRESOLVED}

ARRAY_DIRECTIVE_RE = re.compile(
    r"^\s*#SBATCH\s+--array=(?P<spec>[^\s#]+)", re.MULTILINE)
ARRAY_SPEC_RE = re.compile(
    r"^(?P<start>\d+)-(?P<end>\d+)(?:%(?P<throttle>\d+))?$")

# Log filename: hpo_%x_%A_%a.out / hpo2_%x_%A_%a.out  →  %A = array job id,
# %a = array task id. %x (job name) may itself contain underscores, so anchor on
# the final two numeric fields.
LOG_NAME_RE = re.compile(r"^(?P<prefix>hpo2?)_(?P<jobname>.+)_"
                         r"(?P<arrayjob>\d+)_(?P<task>\d+)\.out$")

# v2 header/footer (hpo_v2.slurm)
V2_TASK_RE = re.compile(r"^\s*Task\s*:\s*(\d+)\s*$", re.MULTILINE)
V2_WORKERS_RE = re.compile(r"^\s*Workers\s*:\s*(\d+)\s*$", re.MULTILINE)
V2_DATE_RE = re.compile(r"^\s*Date\s*:\s*(.+?)\s*$", re.MULTILINE)
V2_FINISH_RE = re.compile(r"^Task\s+(\d+)\s+finished\s*\|\s*exit=(-?\d+)\s*\|\s*(.+?)\s*$",
                          re.MULTILINE)
# v1 header/footer (hpo_array.slurm)
V1_TASK_RE = re.compile(r"^\s*Array Task\s*:\s*(\d+)\s*/\s*(\d+)\s*$", re.MULTILINE)
V1_FINISH_RE = re.compile(r"^\s*Task\s+(\d+)\s+finished with exit code\s+(-?\d+)\s*$",
                          re.MULTILINE)

DATE_FORMATS = [
    "%a %b %d %H:%M:%S %Y",      # `date` with the TZ token stripped
    "%a %d %b %H:%M:%S %Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
]


def parse_date(raw: str):
    """Parse a `date` string, or return None. Never guesses."""
    text = " ".join(raw.split())
    candidates = [text]
    tokens = text.split()
    # `date` emits e.g. "Mon Sep  7 14:44:02 CEST 2026"; drop the TZ token,
    # which strptime cannot portably read.
    if len(tokens) == 6:
        candidates.append(" ".join(tokens[:4] + tokens[5:]))
    for candidate in candidates:
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:
                continue
    return None


def parse_array_directives(script_path: Path):
    """Extract every --array directive. Returns (records, unexpanded_count)."""
    try:
        text = script_path.read_text()
    except OSError as exc:
        return [], 0, f"unreadable: {exc}"

    records, unexpanded = [], 0
    for match in ARRAY_DIRECTIVE_RE.finditer(text):
        spec = match.group("spec")
        if "$" in spec:
            # e.g. 1-${N_WORKERS}%${N_WORKERS} — SLURM parses #SBATCH before
            # shell expansion, so this directive never takes effect.
            unexpanded += 1
            records.append({"spec": spec, "expanded": False,
                            "array_size": None, "throttle": None})
            continue
        parsed = ARRAY_SPEC_RE.match(spec)
        if not parsed:
            records.append({"spec": spec, "expanded": True,
                            "array_size": None, "throttle": None})
            continue
        start = int(parsed.group("start"))
        end = int(parsed.group("end"))
        throttle = parsed.group("throttle")
        records.append({
            "spec": spec, "expanded": True,
            "array_size": end - start + 1,
            "throttle": int(throttle) if throttle else None,
        })
    return records, unexpanded, ""


def scan_archive_provenance(results_root: Path):
    """Collect per-study evidence from the JSON records only."""
    studies = {}
    for directory in sorted(d for d in results_root.iterdir() if d.is_dir()):
        declared, tasks, samplers, datasets = set(), set(), set(), set()
        files = 0
        for path in sorted(directory.glob("*.json")):
            files += 1
            try:
                record = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if record.get("parallel_workers") is not None:
                declared.add(record["parallel_workers"])
            if record.get("slurm_task") is not None:
                tasks.add(record["slurm_task"])
            if record.get("sampler"):
                samplers.add(record["sampler"])
            if record.get("dataset"):
                datasets.add(record["dataset"])
        studies[directory.name] = {
            "study_dir": directory.name,
            "files": files,
            "declared_w_values": sorted(declared),
            "distinct_slurm_tasks": len(tasks),
            "slurm_task_min": min(tasks) if tasks else "",
            "slurm_task_max": max(tasks) if tasks else "",
            "samplers_in_records": sorted(samplers),
            "datasets_in_records": sorted(datasets),
        }
    return studies


def parse_logs(log_dir: Path):
    """Parse SLURM stdout logs into per-(array job, task) intervals.

    Returns (by_job, failures). Never invents a timestamp: a task without both a
    parseable start and end is recorded as a failure, not as a zero-length run.
    """
    by_job = defaultdict(list)
    failures = []
    for path in sorted(log_dir.rglob("*.out")):
        name_match = LOG_NAME_RE.match(path.name)
        if not name_match:
            failures.append({"file": str(path),
                             "reason": "filename does not match hpo[2]_<name>_<A>_<a>.out"})
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            failures.append({"file": str(path), "reason": f"unreadable: {exc}"})
            continue

        array_job = name_match.group("arrayjob")
        task = int(name_match.group("task"))

        dates = V2_DATE_RE.findall(text)
        start = parse_date(dates[0]) if dates else None

        end = None
        finish_v2 = V2_FINISH_RE.search(text)
        if finish_v2:
            end = parse_date(finish_v2.group(3))
        elif V1_FINISH_RE.search(text) and len(dates) >= 2:
            end = parse_date(dates[-1])

        workers_match = V2_WORKERS_RE.search(text)
        declared_in_log = int(workers_match.group(1)) if workers_match else None

        if start is None or end is None:
            failures.append({
                "file": str(path),
                "reason": "could not parse both a start and an end timestamp "
                          f"(start={'ok' if start else 'MISSING'}, "
                          f"end={'ok' if end else 'MISSING'})",
            })
            continue

        by_job[array_job].append({
            "task": task, "start": start, "end": end,
            "declared_workers_in_log": declared_in_log,
            "file": str(path),
        })
    return by_job, failures


def max_concurrency(intervals) -> int:
    """Maximum number of simultaneously running tasks via an interval sweep."""
    events = []
    for item in intervals:
        events.append((item["start"], 1))
        events.append((item["end"], -1))
    # Ends before starts at an identical timestamp: a task finishing exactly as
    # another begins was not concurrent with it.
    events.sort(key=lambda e: (e[0], e[1]))
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def default_script_for(study_dir: str, scripts: dict):
    if study_dir.startswith("v2_"):
        for name, data in scripts.items():
            if "hpo_v2" in name:
                return name, "study_dir prefix 'v2_' -> hpo_v2.slurm RESULTS_DIR template"
    else:
        for name, data in scripts.items():
            if "hpo_array" in name:
                return name, "no 'v2_' prefix -> hpo_array.slurm RESULTS_DIR template"
    return "", "no mapped script supplied"


def classify(study, script_evidence, log_evidence):
    """Assign exactly one evidence class. Returns (status, w_confirmed, notes)."""
    notes = []
    declared = study["declared_w_values"]
    observed_tasks = study["distinct_slurm_tasks"]

    if log_evidence and log_evidence.get("max_concurrency"):
        notes.append(
            f"measured from {log_evidence['n_tasks']} log intervals across "
            f"array job(s) {log_evidence['array_jobs']}")
        return CONFIRMED, log_evidence["max_concurrency"], notes

    if len(declared) > 1:
        notes.append(f"records disagree among themselves: declared W values {declared}")
        return CONTRADICTED, "", notes

    if len(declared) == 1:
        label = declared[0]
        if label == observed_tasks:
            notes.append(
                f"label W={label} equals {observed_tasks} distinct slurm_task IDs; "
                "consistent, but concurrency is unmeasured")
            return DECLARED_CONSISTENT, "", notes
        if observed_tasks > label:
            notes.append(
                f"label W={label} but {observed_tasks} distinct slurm_task IDs wrote "
                "into this single shared study; more tasks joined the study than the "
                "label permits, so the label cannot be correct")
            return CONTRADICTED, "", notes
        notes.append(
            f"label W={label} but only {observed_tasks} distinct slurm_task IDs wrote "
            "output; consistent with tasks that failed before writing, so the label is "
            "neither corroborated nor refuted by the archive")
        return DECLARED_PARTIAL, "", notes

    throttle = script_evidence.get("throttle")
    if throttle:
        notes.append(
            f"no parallel_workers field in any record; mapped script declares "
            f"--array={script_evidence.get('spec')} giving concurrency cap {throttle} "
            "(upper bound, not a measurement)")
        return INFERRED_UPPER_BOUND, "", notes

    notes.append("no parallel_workers field, no usable --array throttle, no logs")
    return UNRESOLVED, "", notes


FIELDS = [
    "study_dir", "status", "w_confirmed", "w_declared_label",
    "w_observed_distinct_slurm_tasks", "w_upper_bound_from_script",
    "confirmed_from_logs", "blocks_w_comparative_analysis",
    "mapped_script", "script_mapping_basis", "script_array_spec",
    "script_directive_expanded", "samplers_in_records", "datasets_in_records",
    "slurm_task_range", "files", "evidence_notes",
]


def build_report(rows, script_summary, log_summary, failures, results_root, log_dir):
    lines = []
    add = lines.append
    by_status = defaultdict(list)
    for row in rows:
        by_status[row["status"]].append(row)

    add("# Aim 1 — Provenance Reconstruction Report (worker count W)")
    add("")
    add(f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    add(f"- Archive root: `{results_root}`")
    add(f"- SLURM log dir: {'`' + str(log_dir) + '`' if log_dir else '**not supplied**'}")
    add("")
    add("## 1. Summary by evidence class")
    add("")
    add("| Status | Studies | Usable as W in cross-W comparisons? |")
    add("|---|---|---|")
    for status, usable in [
        (CONFIRMED, "**yes**"),
        (DECLARED_CONSISTENT, "no — label only, concurrency unmeasured"),
        (DECLARED_PARTIAL, "no — fewer tasks wrote output than the label claims"),
        (CONTRADICTED, "no — more tasks joined than the label permits"),
        (INFERRED_UPPER_BOUND, "no — upper bound only"),
        (UNRESOLVED, "no — no evidence"),
    ]:
        add(f"| {status} | {len(by_status.get(status, []))} | {usable} |")
    add("")
    confirmed_n = len(by_status.get(CONFIRMED, []))
    if confirmed_n == 0:
        add("**No study has a confirmed worker count.** `w_confirmed` is empty for")
        add("every row. Concurrency can only be measured from SLURM log timestamps,")
        add("and no logs were supplied (or none parsed). Until logs from the Eagle")
        add("cluster are recovered and parsed, W is not established anywhere in this")
        add("archive, and no comparison across worker counts is supportable.")
        add("")
    add("## 2. Per-study detail")
    add("")
    add("| study_dir | status | w_confirmed | declared label | distinct slurm_tasks | script upper bound | blocks analysis |")
    add("|---|---|---|---|---|---|---|")
    for row in rows:
        add(f"| {row['study_dir']} | {row['status']} | "
            f"{row['w_confirmed'] or '—'} | {row['w_declared_label'] or '—'} | "
            f"{row['w_observed_distinct_slurm_tasks']} | "
            f"{row['w_upper_bound_from_script'] or '—'} | "
            f"{'YES' if row['blocks_w_comparative_analysis'] else 'no'} |")
    add("")
    add("`w_confirmed` is populated only for CONFIRMED rows. A dash means the value")
    add("is not known, not that it is zero or default.")
    add("")

    if by_status.get(CONTRADICTED) or by_status.get(DECLARED_PARTIAL):
        add("## 3. Studies whose declared label does not match the archive")
        add("")
        add("The `parallel_workers` label in these studies disagrees with the archive.")
        add("The label is the `--parallel-workers` CLI argument echoed by train.py; it")
        add("records what was declared, not what ran. Because `hpo_v2.slurm`'s")
        add("`#SBATCH --array=1-${N_WORKERS}%${N_WORKERS}` directive never expands, the")
        add("array size actually used came from the sbatch command line and is not")
        add("recorded anywhere in the archive.")
        add("")
        if by_status.get(CONTRADICTED):
            add("### CONTRADICTED — more writers than the label permits")
            add("")
            for row in by_status[CONTRADICTED]:
                add(f"- **{row['study_dir']}** — declared W={row['w_declared_label']}, "
                    f"but {row['w_observed_distinct_slurm_tasks']} distinct slurm_task "
                    f"IDs (range {row['slurm_task_range']}) wrote into this single "
                    "shared study. The label cannot be correct.")
            add("")
        if by_status.get(DECLARED_PARTIAL):
            add("### DECLARED_PARTIAL — fewer writers than the label claims")
            add("")
            add("Consistent with tasks that failed before writing output. Neither")
            add("corroborated nor refuted; still not usable as a measured W.")
            add("")
            for row in by_status[DECLARED_PARTIAL]:
                add(f"- **{row['study_dir']}** — declared W={row['w_declared_label']}, "
                    f"{row['w_observed_distinct_slurm_tasks']} distinct slurm_task IDs "
                    f"(range {row['slurm_task_range']}) wrote output.")
            add("")
        add("Resolution requires the Eagle SLURM logs for these array jobs. No")
        add("post-hoc processing of the archive can substitute.")
        add("")

    if by_status.get(INFERRED_UPPER_BOUND) or by_status.get(UNRESOLVED):
        add("## 4. Studies with no worker-count label at all")
        add("")
        add("These studies were produced before `--parallel-workers` was passed:")
        add("`hpo_array.slurm` never passes it, so `train.py` records nothing. The")
        add("trial records in these directories also lack `sampler` and `dataset`")
        add("fields and use the `trial_{number:05d}.json` filename scheme, which no")
        add("committed version of `train.py` produces. The code that generated them")
        add("is not in the repository.")
        add("")
        for row in by_status.get(INFERRED_UPPER_BOUND, []) + by_status.get(UNRESOLVED, []):
            add(f"- **{row['study_dir']}** — {row['evidence_notes']}")
        add("")

    add("## 5. Orchestration script evidence")
    add("")
    if not script_summary:
        add("No --slurm-script supplied. Upper bounds unavailable.")
    else:
        add("| script | --array spec | expanded by SLURM? | array size | concurrency cap |")
        add("|---|---|---|---|---|")
        for name, records in script_summary.items():
            if not records:
                add(f"| {name} | (none found) | — | — | — |")
            for rec in records:
                add(f"| {name} | `{rec['spec']}` | "
                    f"{'yes' if rec['expanded'] else '**NO**'} | "
                    f"{rec['array_size'] if rec['array_size'] is not None else '—'} | "
                    f"{rec['throttle'] if rec['throttle'] is not None else '—'} |")
        add("")
        add("A directive marked **NO** contains an unexpanded shell variable. SLURM")
        add("parses `#SBATCH` lines before shell substitution, so the directive has no")
        add("effect and the real array size is whatever was passed to sbatch.")
    add("")

    add("## 6. Log parsing")
    add("")
    if log_dir is None:
        add("No --slurm-log-dir supplied. Nothing could be confirmed.")
    else:
        add(f"- Log files parsed successfully: {log_summary.get('parsed', 0)}")
        add(f"- Log files that failed to parse: {len(failures)}")
        add(f"- Array jobs reconstructed: {log_summary.get('array_jobs', 0)}")
        if failures:
            add("")
            add("Failures are listed in `log_parse_failures.csv`. They are not skipped")
            add("silently: a study whose logs failed to parse keeps its next-best")
            add("evidence class and continues to block analysis.")
    add("")

    add("## 7. Gate for downstream work")
    add("")
    blocking = [r["study_dir"] for r in rows if r["blocks_w_comparative_analysis"]]
    add(f"**{len(blocking)} of {len(rows)} studies block cross-W analysis.** They are")
    add("listed in `unresolved_studies.txt`.")
    add("")
    add("No bootstrap confidence interval, Mann-Whitney U test, multiple-comparison")
    add("correction, or Sharpe-style score should be computed while this list is")
    add("non-empty, for two independent reasons:")
    add("")
    add("1. W, the independent variable, is not established for these studies.")
    add("2. Independent studies per cell is 1 and every sampler is seeded 42")
    add("   (`train.py`: TPE, Hyperband and CMA-ES all take `seed=42`; only Random")
    add("   Search varies). There are no independent replicates to bootstrap over,")
    add("   and the exchangeability that Mann-Whitney U requires does not hold.")
    add("")
    add("Resolving (1) needs the Eagle logs. Resolving (2) needs new runs — the")
    add("planned 5 seeds x 2 samplers experiment is the only source of independent")
    add("replicates in the project and should be run before any further baselines.")
    add("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct worker-count provenance per Optuna study.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--slurm-script", action="append", default=[], type=Path,
                        help="Committed orchestration script. Repeatable.")
    parser.add_argument("--slurm-log-dir", type=Path, default=None,
                        help="Directory of Eagle SLURM .out logs. The only source "
                             "that can produce a CONFIRMED worker count.")
    parser.add_argument("--study-map", type=Path, default=None,
                        help="Optional JSON {study_dir: script_path} override.")
    parser.add_argument("--output-dir", type=Path, default=Path("./aim1_output"))
    args = parser.parse_args(argv)

    if not args.results_root.is_dir():
        print(f"ERROR: --results-root is not a directory: {args.results_root}",
              file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    studies = scan_archive_provenance(args.results_root)
    if not studies:
        print(f"ERROR: no study subdirectories under {args.results_root}",
              file=sys.stderr)
        return 2

    # Script evidence.
    script_summary = {}
    for script_path in args.slurm_script:
        records, unexpanded, error = parse_array_directives(script_path)
        if error:
            print(f"WARNING: {script_path}: {error}", file=sys.stderr)
        script_summary[script_path.name] = records

    override = {}
    if args.study_map:
        override = json.loads(args.study_map.read_text())

    # Log evidence.
    log_by_study = {}
    failures = []
    log_summary = {}
    if args.slurm_log_dir:
        if not args.slurm_log_dir.is_dir():
            print(f"ERROR: --slurm-log-dir is not a directory: {args.slurm_log_dir}",
                  file=sys.stderr)
            return 2
        by_job, failures = parse_logs(args.slurm_log_dir)
        log_summary = {
            "parsed": sum(len(v) for v in by_job.values()),
            "array_jobs": len(by_job),
        }
        # Logs carry no study identifier, so they can only be attached to a study
        # via the worker count echoed in the v2 header. Anything not attributable
        # is reported rather than assigned.
        for job, intervals in by_job.items():
            declared = {i["declared_workers_in_log"] for i in intervals
                        if i["declared_workers_in_log"] is not None}
            if len(declared) != 1:
                failures.append({
                    "file": f"<array job {job}>",
                    "reason": "cannot attribute this array job to a study: the "
                              f"'Workers' header is absent or inconsistent {declared}",
                })
                continue
            label = declared.pop()
            targets = [name for name, data in studies.items()
                       if data["declared_w_values"] == [label]]
            if len(targets) != 1:
                failures.append({
                    "file": f"<array job {job}>",
                    "reason": f"Workers={label} matches {len(targets)} studies; "
                              "cannot attribute unambiguously",
                })
                continue
            entry = log_by_study.setdefault(
                targets[0], {"intervals": [], "array_jobs": []})
            entry["intervals"].extend(intervals)
            entry["array_jobs"].append(job)

        for name, entry in log_by_study.items():
            entry["max_concurrency"] = max_concurrency(entry["intervals"])
            entry["n_tasks"] = len(entry["intervals"])
            entry["array_jobs"] = ",".join(sorted(entry["array_jobs"]))

    rows = []
    for name, study in studies.items():
        if name in override:
            mapped, basis = Path(override[name]).name, "explicit --study-map entry"
        else:
            mapped, basis = default_script_for(name, script_summary)

        records = script_summary.get(mapped, [])
        usable = [r for r in records if r["expanded"] and r["throttle"]]
        script_evidence = usable[0] if usable else {}
        spec_display = "; ".join(r["spec"] for r in records) if records else ""
        expanded_display = ("; ".join("yes" if r["expanded"] else "NO"
                                      for r in records) if records else "")

        status, w_confirmed, notes = classify(
            study, script_evidence, log_by_study.get(name))

        rows.append({
            "study_dir": name,
            "status": status,
            "w_confirmed": w_confirmed,
            "w_declared_label": (study["declared_w_values"][0]
                                 if len(study["declared_w_values"]) == 1
                                 else ("/".join(map(str, study["declared_w_values"]))
                                       if study["declared_w_values"] else "")),
            "w_observed_distinct_slurm_tasks": study["distinct_slurm_tasks"],
            "w_upper_bound_from_script": script_evidence.get("throttle", ""),
            "confirmed_from_logs": bool(log_by_study.get(name)),
            "blocks_w_comparative_analysis": status in BLOCKING,
            "mapped_script": mapped,
            "script_mapping_basis": basis,
            "script_array_spec": spec_display,
            "script_directive_expanded": expanded_display,
            "samplers_in_records": "/".join(study["samplers_in_records"]),
            "datasets_in_records": "/".join(study["datasets_in_records"]),
            "slurm_task_range": (f"{study['slurm_task_min']}-{study['slurm_task_max']}"
                                 if study["slurm_task_min"] != "" else ""),
            "files": study["files"],
            "evidence_notes": "; ".join(notes),
        })

    rows.sort(key=lambda r: (r["status"], r["study_dir"]))
    out = args.output_dir

    with (out / "provenance_reconstruction.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})

    with (out / "log_parse_failures.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "reason"])
        writer.writeheader()
        for failure in failures:
            writer.writerow(failure)

    blocking = [r["study_dir"] for r in rows if r["blocks_w_comparative_analysis"]]
    (out / "unresolved_studies.txt").write_text(
        "".join(f"{name}\n" for name in sorted(blocking)))

    (out / "provenance_report.md").write_text(
        build_report(rows, script_summary, log_summary, failures,
                     args.results_root, args.slurm_log_dir))
    (out / "provenance_report.json").write_text(
        json.dumps({"rows": rows,
                    "log_summary": log_summary,
                    "log_parse_failures": failures}, indent=2, default=str) + "\n")

    counts = defaultdict(int)
    for row in rows:
        counts[row["status"]] += 1
    print("Provenance status:")
    for status in (CONFIRMED, DECLARED_CONSISTENT, DECLARED_PARTIAL,
                   CONTRADICTED, INFERRED_UPPER_BOUND, UNRESOLVED):
        print(f"  {status:22s} {counts[status]}")
    print(f"Studies with a CONFIRMED worker count: {counts[CONFIRMED]}")
    print(f"Studies blocking cross-W analysis: {len(blocking)} of {len(rows)}")
    if failures:
        print(f"Log parse failures: {len(failures)} (see log_parse_failures.csv)")
    print(f"Wrote outputs to {out}/")
    print("No statistics computed. See section 7 of provenance_report.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
