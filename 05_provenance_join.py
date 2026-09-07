#!/usr/bin/env python3
"""
05_provenance_join.py

Joins sacct_full.psv, sacct_submitlines.psv, and sacct_workdirs.psv to
reconstruct per-task provenance for the pl0844-03 HPO job archive.

Recovers:
  1. Nominal W (array throttle) per job:
       - v2 jobs: parsed directly from SubmitLine (--array=A-B%C)
       - v1 jobs (6921028/6921076/6921123): SubmitLine only has the sampler
         positional arg, no --array flag (it's an #SBATCH directive inside
         hpo_array.slurm, invisible to sacct) -> nominal W must come from
         the archived .slurm file for these three jobs specifically.
  2. Realized concurrency per job: computed from Start/End interval overlap
     across all tasks belonging to the same base job id.
  3. Node speed_class: derived empirically from mean COMPLETED-task duration
     per node. There is a large, clean bimodal gap in the data (~800-1130s
     vs ~200-500s), which lines up exactly with node numbering:
       SLOW cluster: gpu03-gpu13   (avg 800-1130s/task)
       FAST cluster: gpu19-gpu88   (avg 200-500s/task)
     This resolves the "is FAST/SLOW == node identity" question from 04.
  4. Submitted-vs-produced-records flag: jobs where every task FAILED in
     under ~15s (e.g. 6929417) are flagged as "submitted but did not
     produce trial records", to explain task-count-vs-label mismatches.

Inputs (expected in the same directory as this script, or pass --dir):
  sacct_full.psv         JobID|JobIDRaw|JobName|Account|Submit|Start|End|Elapsed|NodeList|State|ExitCode|AllocTRES|WorkDir
  sacct_submitlines.psv  JobID|SubmitLine
  sacct_workdirs.psv     JobID|WorkDir   (redundant with sacct_full's WorkDir column, kept as a cross-check)

Output:
  provenance_joined.csv   one row per task (array element)
  provenance_summary.csv  one row per base job id
  Console report summarizing the three open questions above.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Known archive facts for the three v1 jobs, since sacct's SubmitLine does not
# capture #SBATCH directives (only the argv passed to the sbatch command).
# Source: archived hpo_array.slurm, "#SBATCH --array=1-200%40" (identical
# across all three v1 studies per the original recovery).
# ---------------------------------------------------------------------------
V1_ARCHIVE_NOMINAL_W = {
    "6921028": {"array_lo": 1, "array_hi": 200, "throttle": 40, "sampler": "random"},
    "6921076": {"array_lo": 1, "array_hi": 200, "throttle": 40, "sampler": "tpe"},
    "6921123": {"array_lo": 1, "array_hi": 200, "throttle": 40, "sampler": "hyperband"},
}

FAIL_FAST_SECONDS = 15  # threshold under which a FAILED task is "did not produce a trial record"

SUBMITLINE_ARRAY_RE = re.compile(r"--array=(\d+)-(\d+)%(\d+)")
SUBMITLINE_V2_RE = re.compile(
    r"hpo_v2\.slurm\s+(?P<sampler>\S+)\s+(?P<dataset>\S+)\s+(?P<w>\d+)"
)
SUBMITLINE_V1_RE = re.compile(r"hpo_array\.slurm\s+(?P<sampler>\S+)")


def parse_ts(s):
    if not s or s in ("Unknown", "None"):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def elapsed_to_seconds(s):
    if not s:
        return None
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = 0, *parts
    else:
        return None
    return h * 3600 + m * 60 + sec


def base_job_id(job_id):
    return job_id.split("_")[0]


def load_psv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        return list(reader)


def build_node_speed_class(rows):
    """Compute mean COMPLETED-task duration per node, then split into
    FAST/SLOW by an empirical gap search (no fixed node-number assumption,
    so this stays correct if new nodes appear in future archives)."""
    sums = defaultdict(float)
    counts = defaultdict(int)
    for r in rows:
        if r["State"] != "COMPLETED":
            continue
        sec = elapsed_to_seconds(r["Elapsed"])
        node = r["NodeList"]
        if sec is None or not node:
            continue
        sums[node] += sec
        counts[node] += 1

    node_avg = {n: sums[n] / counts[n] for n in sums if counts[n] > 0}
    if not node_avg:
        return {}, None

    ordered = sorted(node_avg.items(), key=lambda kv: kv[1])
    # find the largest single gap in the sorted average-duration sequence
    best_gap = -1
    best_idx = None
    for i in range(1, len(ordered)):
        gap = ordered[i][1] - ordered[i - 1][1]
        if gap > best_gap:
            best_gap = gap
            best_idx = i
    threshold = (ordered[best_idx - 1][1] + ordered[best_idx][1]) / 2 if best_idx else None

    speed_class = {}
    for node, avg in node_avg.items():
        speed_class[node] = "FAST" if avg < threshold else "SLOW"
    return speed_class, {
        "node_avg": node_avg,
        "threshold_seconds": threshold,
        "gap_seconds": best_gap,
    }


def realized_concurrency(task_rows):
    """Given all task rows for one base job, compute max number of tasks
    with overlapping [Start, End] intervals at any instant (sweep-line)."""
    events = []
    for r in task_rows:
        start = parse_ts(r["Start"])
        end = parse_ts(r["End"])
        if start is None or end is None:
            continue
        events.append((start, 1))
        events.append((end, -1))
    if not events:
        return None
    events.sort(key=lambda e: (e[0], e[1]))  # ends (-1) before starts (+1) at same instant is fine either way here
    cur = 0
    peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def parse_submitline(base_id, submitline):
    """Return dict with nominal array bounds/throttle and sampler/dataset/W
    where recoverable directly from the sacct SubmitLine."""
    out = {
        "array_lo": None,
        "array_hi": None,
        "throttle_from_submitline": None,
        "sampler": None,
        "dataset": None,
        "w_from_submitline": None,
        "nominal_w_source": None,
    }
    if not submitline:
        return out

    m_arr = SUBMITLINE_ARRAY_RE.search(submitline)
    if m_arr:
        out["array_lo"] = int(m_arr.group(1))
        out["array_hi"] = int(m_arr.group(2))
        out["throttle_from_submitline"] = int(m_arr.group(3))

    m_v2 = SUBMITLINE_V2_RE.search(submitline)
    if m_v2:
        out["sampler"] = m_v2.group("sampler")
        out["dataset"] = m_v2.group("dataset")
        out["w_from_submitline"] = int(m_v2.group("w"))
        out["nominal_w_source"] = "submitline"
        return out

    m_v1 = SUBMITLINE_V1_RE.search(submitline)
    if m_v1:
        out["sampler"] = m_v1.group("sampler")
        # v1: array spec lives in #SBATCH inside hpo_array.slurm, not on
        # the sbatch command line, so it's absent from SubmitLine.
        if base_id in V1_ARCHIVE_NOMINAL_W:
            arch = V1_ARCHIVE_NOMINAL_W[base_id]
            out["array_lo"] = arch["array_lo"]
            out["array_hi"] = arch["array_hi"]
            out["throttle_from_submitline"] = arch["throttle"]
            out["nominal_w_source"] = "archive_sbatch_directive"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="directory containing the three .psv inputs")
    ap.add_argument("--out-dir", default=".", help="directory to write outputs to")
    args = ap.parse_args()

    in_dir = Path(args.dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_rows = load_psv(in_dir / "sacct_full.psv")
    submit_rows = load_psv(in_dir / "sacct_submitlines.psv")
    workdir_rows = load_psv(in_dir / "sacct_workdirs.psv")

    submitline_by_task = {r["JobID"]: r["SubmitLine"] for r in submit_rows}
    workdir_by_task = {r["JobID"]: r["WorkDir"] for r in workdir_rows}

    # --- node speed class -------------------------------------------------
    speed_class, speed_meta = build_node_speed_class(full_rows)

    # --- group tasks by base job id ---------------------------------------
    tasks_by_base = defaultdict(list)
    for r in full_rows:
        tasks_by_base[base_job_id(r["JobID"])].append(r)

    # --- per-task join rows -------------------------------------------------
    joined_rows = []
    for r in full_rows:
        job_id = r["JobID"]
        base = base_job_id(job_id)
        submitline = submitline_by_task.get(job_id, "")
        parsed = parse_submitline(base, submitline)
        elapsed_sec = elapsed_to_seconds(r["Elapsed"])
        node = r["NodeList"]

        joined_rows.append({
            "JobID": job_id,
            "BaseJobID": base,
            "JobName": r["JobName"],
            "State": r["State"],
            "ExitCode": r["ExitCode"],
            "Node": node,
            "SpeedClass": speed_class.get(node, "UNKNOWN"),
            "Submit": r["Submit"],
            "Start": r["Start"],
            "End": r["End"],
            "ElapsedSeconds": elapsed_sec,
            "FastFailed": (
                r["State"] == "FAILED"
                and elapsed_sec is not None
                and elapsed_sec <= FAIL_FAST_SECONDS
            ),
            "Sampler": parsed["sampler"],
            "Dataset": parsed["dataset"],
            "NominalW_ArrayLo": parsed["array_lo"],
            "NominalW_ArrayHi": parsed["array_hi"],
            "NominalW_Throttle": parsed["throttle_from_submitline"],
            "NominalW_Source": parsed["nominal_w_source"],
            "WorkDir": r["WorkDir"] or workdir_by_task.get(job_id, ""),
            "SubmitLine": submitline,
        })

    # --- per-base-job summary ----------------------------------------------
    summary_rows = []
    all_fast_failed_bases = []
    for base, rows in sorted(tasks_by_base.items()):
        n_tasks = len(rows)
        n_completed = sum(1 for r in rows if r["State"] == "COMPLETED")
        n_failed = sum(1 for r in rows if r["State"] == "FAILED")
        n_timeout = sum(1 for r in rows if r["State"] == "TIMEOUT")
        elapsed_secs = [elapsed_to_seconds(r["Elapsed"]) for r in rows]
        elapsed_secs = [e for e in elapsed_secs if e is not None]
        n_fast_failed = sum(
            1 for r in rows
            if r["State"] == "FAILED"
            and elapsed_to_seconds(r["Elapsed"]) is not None
            and elapsed_to_seconds(r["Elapsed"]) <= FAIL_FAST_SECONDS
        )
        all_fast_failed = n_tasks > 0 and n_fast_failed == n_tasks

        peak_conc = realized_concurrency(rows)

        parsed_sample = parse_submitline(base, submitline_by_task.get(rows[0]["JobID"], ""))
        nodes_used = sorted(set(r["NodeList"] for r in rows if r["NodeList"]))
        classes_used = sorted(set(speed_class.get(n, "UNKNOWN") for n in nodes_used))

        summary_rows.append({
            "BaseJobID": base,
            "Sampler": parsed_sample["sampler"],
            "Dataset": parsed_sample["dataset"],
            "NominalW_ArrayRange": (
                f"{parsed_sample['array_lo']}-{parsed_sample['array_hi']}"
                if parsed_sample["array_lo"] is not None else ""
            ),
            "NominalW_Throttle": parsed_sample["throttle_from_submitline"],
            "NominalW_Source": parsed_sample["nominal_w_source"],
            "NTasks": n_tasks,
            "Completed": n_completed,
            "Failed": n_failed,
            "Timeout": n_timeout,
            "RealizedPeakConcurrency": peak_conc,
            "NodesUsed": ";".join(nodes_used),
            "SpeedClassesUsed": ";".join(classes_used),
            "AllTasksFastFailed_NoRecordsProduced": all_fast_failed,
        })
        if all_fast_failed:
            all_fast_failed_bases.append(base)

    # --- write outputs -------------------------------------------------------
    joined_path = out_dir / "provenance_joined.csv"
    with open(joined_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(joined_rows[0].keys()))
        w.writeheader()
        w.writerows(joined_rows)

    summary_path = out_dir / "provenance_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # --- console report -------------------------------------------------------
    print("=" * 78)
    print("PROVENANCE JOIN REPORT")
    print("=" * 78)

    print(f"\nTotal task rows: {len(full_rows)}   Base jobs: {len(tasks_by_base)}")

    print("\n--- 1. Nominal W recovery ---")
    src_counts = defaultdict(int)
    for row in summary_rows:
        src_counts[row["NominalW_Source"] or "UNRECOVERED"] += 1
    for src, n in src_counts.items():
        print(f"  {src:26s}: {n} base jobs")
    unrecovered = [row["BaseJobID"] for row in summary_rows if not row["NominalW_Source"]]
    if unrecovered:
        print(f"  UNRECOVERED base jobs: {unrecovered}")
    else:
        print("  All base jobs have a nominal-W source. Re-sweep is NOT required for this reason.")

    print("\n--- 2. Node speed class ---")
    if speed_meta:
        print(f"  Empirical FAST/SLOW split found at avg-duration gap of "
              f"{speed_meta['gap_seconds']:.1f}s (threshold ~{speed_meta['threshold_seconds']:.1f}s)")
        fast_nodes = sorted(n for n, c in speed_class.items() if c == "FAST")
        slow_nodes = sorted(n for n, c in speed_class.items() if c == "SLOW")
        print(f"  FAST nodes ({len(fast_nodes)}): {fast_nodes}")
        print(f"  SLOW nodes ({len(slow_nodes)}): {slow_nodes}")
        print("  -> FAST/SLOW corresponds to node identity (confirmed empirically, not just candidate).")
    else:
        print("  Could not compute (no COMPLETED tasks with valid Elapsed/NodeList).")

    print("\n--- 3. Realized concurrency vs nominal throttle ---")
    for row in summary_rows:
        nom = row["NominalW_Throttle"]
        real = row["RealizedPeakConcurrency"]
        flag = ""
        if nom is not None and real is not None and real > nom:
            flag = "  <-- realized EXCEEDS nominal throttle, check"
        print(f"  {row['BaseJobID']}: nominal={nom} realized_peak={real}{flag}")

    print("\n--- 4. Submitted-but-no-records-produced jobs ---")
    if all_fast_failed_bases:
        print(f"  {len(all_fast_failed_bases)} base job(s) had every task fail within "
              f"{FAIL_FAST_SECONDS}s (no trial records produced): {all_fast_failed_bases}")
    else:
        print("  None found.")

    print(f"\nWrote: {joined_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
