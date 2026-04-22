#!/bin/bash

# One-time environment setup for HPO project on Eagle (PCSS Poznan)
# Run once: bash setup_env.sh


set -e  # exit on first error

echo "======================================================"
echo "  Eagle HPC – HPO Environment Setup"
echo "  User: $USER | Host: $(hostname) | Date: $(date)"
echo "======================================================"

# Step 1: Check available Python modules
echo ""
echo "[1/6] Checking available Python/CUDA modules..."
module purge
module avail python 2>&1 | head -30
module avail CUDA   2>&1 | head -20
echo "(Review above and update CUDA module names in hpo_array.slurm if needed)"

# Step 2: Install Miniconda (if not present) 
echo ""
echo "[2/6] Checking for conda..."
if ! command -v conda &>/dev/null; then
    echo "conda not found – installing Miniconda3..."
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    wget -q "$MINICONDA_URL" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda init bash
    echo "Miniconda installed at $HOME/miniconda3"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    echo "conda $(conda --version) already available"
fi

# Step 3: Create conda environment 
echo ""
echo "[3/6] Creating conda environment 'hpo_env'..."
if conda env list | grep -q "^hpo_env"; then
    echo "Environment 'hpo_env' already exists – skipping creation"
else
    conda create -y -n hpo_env python=3.10
fi
conda activate hpo_env

#  Step 4: Install PyTorch + dependencies 
echo ""
echo "[4/6] Installing PyTorch (CUDA 12.1) and dependencies..."
pip install --upgrade pip

# PyTorch with CUDA 12.1 (matches Eagle CUDA 12.1.1 module)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# HPO & experiment tracking
pip install \
    optuna==3.6.1 \
    optuna-dashboard==0.15.0 \
    plotly \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    rich

# Optional: MLflow for experiment tracking
pip install mlflow

echo ""
echo "Installed packages:"
pip list | grep -E "torch|optuna|mlflow|numpy|pandas"

# Step 5: Verify GPU access 
echo ""
echo "[5/6] Verifying PyTorch + CUDA..."
python - <<'PYEOF'
import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version    : {torch.version.cuda}")
    print(f"GPU count       : {torch.cuda.device_count()}")
    print(f"GPU name        : {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available – GPU needed for training jobs")
    print("This is expected if running on a login node (no GPU).")
    print("GPU will be available on compute nodes via SLURM.")
import optuna
print(f"Optuna version  : {optuna.__version__}")
PYEOF

# Step 6: Project directory structure 
echo ""
echo "[6/6] Creating project directories..."
mkdir -p \
    "$HOME/hpo_project/data" \
    "$HOME/hpo_project/results/random" \
    "$HOME/hpo_project/results/tpe" \
    "$HOME/hpo_project/results/hyperband" \
    "$HOME/hpo_project/logs" \
    "$HOME/hpo_project/optuna_storage" \
    "$HOME/hpo_project/analysis"

echo ""
echo "======================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. cd ~/hpo_project"
echo "  2. sbatch hpo_array.slurm random     # launch Random Search"
echo "  3. sbatch hpo_array.slurm tpe        # launch Bayesian (TPE)"
echo "  4. sbatch hpo_array.slurm hyperband  # launch Hyperband"
echo "  5. squeue -u $USER                   # monitor jobs"
echo "  6. python analysis/analyze_results.py # after jobs finish"
echo "======================================================"
