#!/bin/bash
# =============================================================================
# SLURM Array Job – HPO Research on Eagle Cluster (PCSS Poznan)
# Grant: pl0844-01 | User: yad123
# =============================================================================
# Usage:
#   sbatch hpo_array.slurm random      # Random Search   (200 tasks x 2 trials)
#   sbatch hpo_array.slurm tpe         # Bayesian (TPE)  (200 tasks x 2 trials)
#   sbatch hpo_array.slurm hyperband   # Hyperband       (200 tasks x 2 trials)
# =============================================================================

#SBATCH --job-name=hpo_%a
#SBATCH --account=pl0844-01
#SBATCH --partition=tesla               # Use 'gpu' partition for GPU nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                  # 1 GPU per task
#SBATCH --mem=16G
#SBATCH --time=02:00:00               # 2h wall time per task
#SBATCH --array=1-200%40              # 200 trials, max 40 running at once
#SBATCH --output=/mnt/storage_6/project_data/pl0844-01/hpo_project/logs/hpo_%x_%A_%a.out
#SBATCH --error=/mnt/storage_6/project_data/pl0844-01/hpo_project/logs/hpo_%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yad123@eagle.man.poznan.pl

# =============================================================================
# 0. Validate sampler argument
# =============================================================================
SAMPLER="${1:-random}"
if [[ "$SAMPLER" != "random" && "$SAMPLER" != "tpe" && "$SAMPLER" != "hyperband" ]]; then
    echo "[ERROR] Unknown sampler: '$SAMPLER'. Use: random | tpe | hyperband"
    exit 1
fi

echo "============================================================"
echo "  HPO Job Array Starting"
echo "  Sampler      : $SAMPLER"
echo "  Array Task   : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  Job ID       : $SLURM_JOB_ID"
echo "  Node         : $SLURMD_NODENAME"
echo "  Date         : $(date)"
echo "============================================================"

# =============================================================================
# 1. Environment setup
# =============================================================================
module purge

module load cuda/12.4.1_550.54.15          # Adjust to what Eagle has (check: module avail CUDA)


# Activate conda environment
source /mnt/storage_6/project_data/pl0844-01/miniconda3/etc/profile.d/conda.sh
conda activate hpo_env

# Verify GPU
echo "--- GPU Info ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "--- Python/PyTorch ---"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# =============================================================================
# 2. Directories
# =============================================================================
PROJECT_DIR="/mnt/storage_6/project_data/pl0844-01/hpo_project"
DATA_DIR="$PROJECT_DIR/data"
RESULTS_DIR="$PROJECT_DIR/results/${SAMPLER}"
LOGS_DIR="$PROJECT_DIR/logs"
DB_PATH="$PROJECT_DIR/optuna_storage/${SAMPLER}.db"

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$LOGS_DIR" "$PROJECT_DIR/optuna_storage"

# =============================================================================
# 3. Optuna shared storage (SQLite – fine up to ~500 workers; use PostgreSQL for more)
#    All array tasks share the same study via this DB → true parallel HPO
# =============================================================================
STORAGE_URL="sqlite:///${DB_PATH}"
STUDY_NAME="hpo_cifar10_${SAMPLER}"

# =============================================================================
# 4. Per-task trial count
#    200 tasks × 2 trials = 400 total trials per sampler
#    Adjust N_TRIALS_PER_WORKER to hit 200-500 total depending on budget
# =============================================================================
N_TRIALS_PER_WORKER=2
MAX_EPOCHS=30
TIMEOUT=6900   # 115 min (leaves 5 min margin before 2h wall time)

# =============================================================================
# 5. Run training
# =============================================================================
cd "$PROJECT_DIR"

python train.py \
    --sampler      "$SAMPLER" \
    --n-trials     $N_TRIALS_PER_WORKER \
    --storage      "$STORAGE_URL" \
    --study-name   "$STUDY_NAME" \
    --data-dir     "$DATA_DIR" \
    --output-dir   "$RESULTS_DIR" \
    --epochs       $MAX_EPOCHS \
    --timeout      $TIMEOUT \
    --num-workers  4 \
    --trial-id     "$SLURM_ARRAY_TASK_ID"

EXIT_CODE=$?
echo "============================================================"
echo "  Task $SLURM_ARRAY_TASK_ID finished with exit code $EXIT_CODE"
echo "  Date: $(date)"
echo "============================================================"
exit $EXIT_CODE
