# ============================================================
#  HPO Project – Eagle Cluster Command Cheatsheet
#  User: yad123 | Grant: pl0844-01
# ============================================================

# ── 0. After web SSH login at pcss.plcloud.pl ────────────────
whoami                          # confirm: yad123
hostname                        # confirm: eagle login node
echo $HOME                      # e.g. /home/yad123

# ── 1. Check available modules ───────────────────────────────
module avail python              # find Python modules
module avail CUDA                # find CUDA modules
module avail GCC                 # find GCC modules

# ── 2. Upload project files (from local machine) ─────────────
# Option A: scp (from your laptop terminal)
scp -r hpo_project/ yad123@eagle.man.poznan.pl:~/

# Option B: paste files directly in web SSH terminal
# (use the file manager in pcss.plcloud.pl interface)

# ── 3. One-time environment setup ────────────────────────────
cd ~/hpo_project
bash setup_env.sh               # installs Miniconda + hpo_env + directories
# (takes ~10-15 min first time)

# ── 4. Quick sanity test (interactive node) ──────────────────
srun --partition=gpu --gres=gpu:1 --time=00:10:00 \
     --account=pl0844-01 --pty bash
# Inside interactive node:
conda activate hpo_env
cd ~/hpo_project
python train.py --sampler random --n-trials 1 --epochs 2
exit

# ── 5. Launch HPO experiments ────────────────────────────────
cd ~/hpo_project

# Submit all three methods:
sbatch hpo_array.slurm random      # 200 tasks × 2 trials = 400 trials
sbatch hpo_array.slurm tpe         # 200 tasks × 2 trials = 400 trials
sbatch hpo_array.slurm hyperband   # 200 tasks × 2 trials = 400 trials

# ── 6. Monitor jobs ───────────────────────────────────────────
squeue -u yad123                          # all your jobs
squeue -u yad123 --format="%.10i %.9P %.20j %.8u %.8T %.10M %.6D %R"
sacct -u yad123 --format=JobID,JobName,State,Elapsed,NCPUS,AllocGRES

# Check GPU usage (on compute node):
srun --jobid=<JOBID> --pty nvidia-smi

# ── 7. Watch logs live ────────────────────────────────────────
tail -f logs/hpo_*_<ARRAY_JOB_ID>_1.out   # first task
ls logs/ | wc -l                           # how many log files created

# ── 8. Check Optuna study progress (SQLite) ───────────────────
conda activate hpo_env
python - <<'EOF'
import optuna
for sampler in ["random", "tpe", "hyperband"]:
    try:
        study = optuna.load_study(
            study_name=f"hpo_cifar10_{sampler}",
            storage=f"sqlite:///optuna_storage/{sampler}.db"
        )
        trials = study.trials
        done = [t for t in trials if t.state.name == "COMPLETE"]
        print(f"{sampler}: {len(done)}/{len(trials)} complete | "
              f"best={study.best_value:.4f}")
    except Exception as e:
        print(f"{sampler}: not started yet ({e})")
EOF

# ── 9. Cancel jobs if needed ─────────────────────────────────
scancel <JOBID>                # single job
scancel -u yad123              # ALL your jobs (careful!)
scancel --name=hpo_1           # by job name

# ── 10. Analyze results (after jobs finish) ───────────────────
conda activate hpo_env
python analysis/analyze_results.py \
    --results-dir ./results \
    --output-dir  ./analysis/output

# Outputs: analysis/output/
#   fig1_convergence.pdf/.png
#   fig2_distribution.pdf/.png
#   fig3_efficiency.pdf/.png
#   table1_summary.csv + .tex
#   table2_stats.csv

# ── 11. Check quota / storage ────────────────────────────────
quota -s                        # disk quota
du -sh ~/hpo_project/           # your project size
du -sh ~/hpo_project/data/      # CIFAR-10 data (~175 MB)

# ── 12. Resource check against grant ─────────────────────────
# CPU hours used:
sacct -u yad123 --format=CPUTimeRAW,Elapsed --starttime=2024-01-01 | tail -20

# GPU hours (rough estimate):
# Each GPU job runs max 2h, 200 tasks → max 400 GPU-hours
# BUT with HyperbandPruner most trials terminate early (~0.3h avg)
# Estimated: 200 trials × 1.0h avg = ~200 GPU-hours per sampler
# TOTAL for all 3 samplers: ~600 GPU-hours   ← exceeds 50h GPU grant!
#
# RECOMMENDATION: Use CPU partition first, or reduce array size:
#   --array=1-25%10   (25 tasks = ~50 GPU-hours per sampler)
