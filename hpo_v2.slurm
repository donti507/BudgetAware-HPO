#!/bin/bash
# =============================================================================
# SLURM Array Job – HPO v2 with Parallelism Sweep
# Grant: pl0844-01 | Eagle Cluster PCSS Poznan
# =============================================================================
# Usage:
#   sbatch hpo_v2.slurm random   cifar10   40   # original experiment
#   sbatch hpo_v2.slurm tpe      cifar10   1    # TPE sequential (baseline)
#   sbatch hpo_v2.slurm tpe      cifar10   5    # TPE 5 workers
#   sbatch hpo_v2.slurm tpe      cifar10   10   # TPE 10 workers
#   sbatch hpo_v2.slurm tpe      cifar10   20   # TPE 20 workers
#   sbatch hpo_v2.slurm tpe      cifar100  40   # CIFAR-100
#   sbatch hpo_v2.slurm cmaes    cifar10   40   # CMA-ES new method
# =============================================================================

#SBATCH --job-name=hpo2_%a
#SBATCH --account=pl0844-01
#SBATCH --partition=tesla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/storage_6/project_data/pl0844-01/hpo_project/logs/hpo2_%x_%A_%a.out
#SBATCH --error=/mnt/storage_6/project_data/pl0844-01/hpo_project/logs/hpo2_%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yad123@eagle.man.poznan.pl

# =============================================================================
# Arguments
# =============================================================================
SAMPLER="${1:-random}"
DATASET="${2:-cifar10}"
N_WORKERS="${3:-40}"

# Set array size based on worker count
# Total tasks = N_WORKERS, each runs 2 trials
# So total trials = N_WORKERS * 2
#SBATCH --array=1-${N_WORKERS}%${N_WORKERS}

echo "============================================================"
echo "  HPO v2 Job"
echo "  Sampler  : $SAMPLER"
echo "  Dataset  : $DATASET"
echo "  Workers  : $N_WORKERS"
echo "  Task     : $SLURM_ARRAY_TASK_ID"
echo "  Node     : $SLURMD_NODENAME"
echo "  Date     : $(date)"
echo "============================================================"

# =============================================================================
# Environment
# =============================================================================
module purge
module load cuda/12.4.1_550.54.15

source /mnt/storage_6/project_data/pl0844-01/miniconda3/etc/profile.d/conda.sh
conda activate hpo_env

echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# =============================================================================
# Paths
# =============================================================================
PROJECT_DIR="/mnt/storage_6/project_data/pl0844-01/hpo_project"
DATA_DIR="$PROJECT_DIR/data"
RESULTS_DIR="$PROJECT_DIR/results/v2_${SAMPLER}_${DATASET}_w${N_WORKERS}"
DB_PATH="$PROJECT_DIR/optuna_storage/v2_${SAMPLER}_${DATASET}_w${N_WORKERS}.db"
STORAGE_URL="sqlite:///${DB_PATH}"
STUDY_NAME="hpo_v2_${SAMPLER}_${DATASET}_w${N_WORKERS}"

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$PROJECT_DIR/logs" "$PROJECT_DIR/optuna_storage"

# =============================================================================
# Run
# =============================================================================
cd "$PROJECT_DIR"

python train.py \
    --sampler          "$SAMPLER" \
    --dataset          "$DATASET" \
    --n-trials         2 \
    --storage          "$STORAGE_URL" \
    --study-name       "$STUDY_NAME" \
    --data-dir         "$DATA_DIR" \
    --output-dir       "$RESULTS_DIR" \
    --epochs           30 \
    --timeout          6900 \
    --num-workers      4 \
    --trial-id         "$SLURM_ARRAY_TASK_ID" \
    --parallel-workers "$N_WORKERS"

EXIT_CODE=$?
echo "Task $SLURM_ARRAY_TASK_ID finished | exit=$EXIT_CODE | $(date)"
exit $EXIT_CODE
