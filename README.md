# BudgetAware-HPO: Parallelism-Aware Hyperparameter Optimization at HPC Scale

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange.svg)](https://pytorch.org)
[![Optuna](https://img.shields.io/badge/Optuna-3.6.1-green.svg)](https://optuna.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HPC](https://img.shields.io/badge/Cluster-Eagle%20HPC%20PCSS-purple.svg)](https://pcss.pl)

---

## Overview

This repository contains the full experimental code, analysis scripts, and results for the research paper:

> **"Parallelism-Aware Hyperparameter Optimization: Empirical Analysis of Sequential Method Degradation on HPC Infrastructure"**

**Author:** MD Ali Ashraf  
**Institution:** Wroclaw University of Science and Technology, Wroclaw, Poland  
**Cluster:** Eagle HPC, PCSS Poznan — Grant: `pl0844-01`  
**Hardware:** Tesla V100-SXM2-32GB GPUs  
**Target venue:** AutoML Conference 2026 (PMLR)

---

## Research Question

> *"Given a fixed GPU budget and a specific degree of parallelism, which hyperparameter optimization method should an HPC researcher choose?"*

Existing HPO benchmarks compare methods by trial count. We compare them by **GPU hours spent** under **realistic HPC parallel execution conditions** — a distinction that fundamentally changes which method wins.

---

## Key Findings

| Parallelism | Best Method | Reason |
|---|---|---|
| 1–10 workers | **TPE (Bayesian)** | Sequential feedback loop intact |
| 40 workers | **Random Search** | TPE's sequential assumption breaks down |
| Any budget, efficiency focus | **Hyperband** | 65% fewer GPU hours than Random Search |
| High parallelism, any budget | **Avoid CMA-ES** | Collapses under parallel execution |

**The core insight:** The ranking of HPO methods *inverts* as parallelism increases. What works best on a laptop does not work best on a supercomputer.

---

## Experimental Setup

### Methods Compared
- **Random Search** — baseline, no sequential assumptions
- **Bayesian Optimization (TPE)** — Tree-structured Parzen Estimator via Optuna
- **Hyperband** — multi-fidelity bandit-based pruning
- **CMA-ES** — Covariance Matrix Adaptation Evolution Strategy

### Datasets
- **CIFAR-10** — 60,000 images, 10 classes (primary benchmark)
- **CIFAR-100** — 60,000 images, 100 classes (generalization test)

### Models
- ResNet-18
- EfficientNet-B0

### Scale
| Experiment | Trials | Workers |
|---|---|---|
| CIFAR-10 main comparison | 480 / 328 / 160 / 32 | 40 |
| Parallelism sweep (TPE) | 41–328 | 1, 5, 10, 20, 40 |
| CIFAR-100 comparison | 136 / 33 / 25 | 40 |
| **Total** | **1,336** | — |

---

## Results Summary

| Method | Trials | Best Acc | Mean Acc | Std | Mean GPU Hours |
|---|---|---|---|---|---|
| Random Search | 480 | **0.9310** | 0.6126 | 0.2678 | 0.144 |
| Bayesian (TPE) | 328 | 0.9272 | **0.8200** | 0.1621 | 0.077 |
| Hyperband | 160 | 0.9185 | 0.8442 | 0.1430 | **0.050** |
| CMA-ES | 32 | 0.5057 | 0.4412 | 0.0204 | 0.047 |

- Random Search achieves the highest peak accuracy but is the **most inconsistent** (std = 0.2678)
- TPE achieves **47% better GPU efficiency** than Random Search
- Hyperband is the **most GPU-efficient** overall (65% less than Random Search)
- CMA-ES **fails completely** under parallel execution

---

## HPO Advisor

We trained a meta-learning recommendation system on all 1,336 trial results.  
Given your GPU budget and worker count, it recommends the optimal method.

**Cross-validation accuracy: 74.4%**

| Budget | 1–10 Workers | 40 Workers |
|---|---|---|
| Any | TPE | Random Search |

```python
python analysis/hpo_advisor.py \
    --results-dir ./results \
    --output-dir ./analysis/output_v2
```

---

## Repository Structure

```
BudgetAware-HPO/
│
├── train.py                      # Main training script (v1)
├── train_v2.py                   # Updated: race condition fix, CIFAR-100, CMA-ES
├── hpo_array.slurm               # Original SLURM array job script
├── hpo_v2.slurm                  # Updated SLURM script (v2)
├── setup_env.sh                  # One-time environment setup script
├── COMMANDS.sh                   # Full command cheatsheet
│
├── analysis/
│   ├── analyze_results.py        # Basic analysis script (v1)
│   ├── analyze_results_v2.py     # Full budget-aware analysis (v2)
│   └── hpo_advisor.py            # Meta-learning recommendation system
│
├── results/                      # All 1,336 trial JSON outputs
│   ├── random/                   # 480 Random Search trials
│   ├── tpe/                      # TPE trials (original run)
│   ├── hyperband/                # Hyperband trials (original run)
│   ├── v2_tpe_cifar10_w1/        # Parallelism sweep: 1 worker
│   ├── v2_tpe_cifar10_w5/        # Parallelism sweep: 5 workers
│   ├── v2_tpe_cifar10_w10/       # Parallelism sweep: 10 workers
│   ├── v2_tpe_cifar10_w20/       # Parallelism sweep: 20 workers
│   ├── v2_tpe_cifar10_w40/       # Parallelism sweep: 40 workers
│   ├── v2_random_cifar100_w40/   # CIFAR-100 Random Search
│   ├── v2_tpe_cifar100_w40/      # CIFAR-100 TPE
│   ├── v2_hyperband_cifar100_w40/# CIFAR-100 Hyperband
│   └── v2_cmaes_cifar10_w40/     # CMA-ES trials
│
└── analysis/output_v2/           # Generated figures and tables
    ├── fig1_method_comparison.pdf/png
    ├── fig2_parallelism_degradation.pdf/png
    ├── fig3_dataset_comparison.pdf/png
    ├── fig4_budget_thresholds.pdf/png
    ├── fig5_hpo_advisor_heatmap.pdf/png
    ├── table1_summary.csv/.tex
    └── table2_stats.csv
```

---

## Reproduction Guide

### Requirements
```
Python 3.10
PyTorch 2.6.0+cu124
Optuna 3.6.1
torchvision 0.21.0
scikit-learn, matplotlib, seaborn, pandas, numpy, tqdm
```

### Environment Setup
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /your/project/miniconda3
conda create -y -n hpo_env python=3.10
conda activate hpo_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install optuna==3.6.1 scikit-learn matplotlib seaborn pandas tqdm
```

### Sanity Test (1 trial, no GPU needed)
```bash
python train.py --sampler random --n-trials 1 --epochs 2 \
    --data-dir ./data --output-dir ./results/test
```

### Full Experiment (SLURM)
```bash
sbatch --array=1-40%40 hpo_v2.slurm random cifar10 40
sbatch --array=1-40%40 hpo_v2.slurm tpe cifar10 40
sbatch --array=1-40%40 hpo_v2.slurm hyperband cifar10 40
```

### Parallelism Sweep
```bash
sbatch --array=1-40%40 hpo_v2.slurm tpe cifar10 40
sbatch --array=1-20%20 hpo_v2.slurm tpe cifar10 20
sbatch --array=1-10%10 hpo_v2.slurm tpe cifar10 10
sbatch --array=1-5%5   hpo_v2.slurm tpe cifar10 5
sbatch --array=1-1%1   hpo_v2.slurm tpe cifar10 1
```

### Reproduce All Figures
```bash
python analysis/analyze_results_v2.py \
    --results-dir ./results \
    --output-dir ./analysis/output_v2

python analysis/hpo_advisor.py \
    --results-dir ./results \
    --output-dir ./analysis/output_v2
```

---

## Technical Notes

### SQLite Race Condition Fix
When 40 parallel workers start simultaneously, they race to create the same Optuna database and crash each other. Fixed in `train_v2.py` with:
- Random stagger delay on worker startup
- Retry mechanism with exponential backoff
- `constant_liar=True` in TPE sampler (built for parallel workers)

This bug itself demonstrates the core research problem: parallel HPC execution breaks sequential HPO assumptions.

### GPU Budget Warning
Running large array jobs can consume GPU hours much faster than expected. Always monitor your allocation before submitting large experiments.

---

## Citation

```bibtex
@inproceedings{ashraf2026budgetaware,
  title     = {Parallelism-Aware Hyperparameter Optimization: 
               Empirical Analysis of Sequential Method Degradation 
               on HPC Infrastructure},
  author    = {Ashraf, MD Ali},
  booktitle = {Proceedings of the AutoML Conference},
  year      = {2026},
  publisher = {PMLR}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Research conducted on Eagle HPC cluster, PCSS Poznan, Poland.*  
*Grant: pl0844-01 | April 2026*
