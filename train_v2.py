#!/usr/bin/env python3
"""
HPO Research: Parallelism-Aware Hyperparameter Optimization at HPC Scale
ResNet/EfficientNet on CIFAR-10/100 using Optuna
Grant: pl0844-01 | Eagle Cluster, PCSS Poznan
Author: yad123

VERSION 2 CHANGES:
- Fixed SQLite race condition for parallel workers
- Added CIFAR-100 support
- Added CMA-ES sampler
- Added GPU hours tracking (budget-aware metric)
- Added parallelism degree logging
- Added retry mechanism for storage conflicts
"""

import os
import sys
import time
import json
import random
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import optuna
from optuna.trial import Trial

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


#Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="HPO on CIFAR-10/100")
    parser.add_argument("--sampler", type=str, default="random",
                        choices=["random", "tpe", "hyperband", "cmaes"],
                        help="HPO sampler: random | tpe | hyperband | cmaes")
    parser.add_argument("--n-trials", type=int, default=2,
                        help="Number of Optuna trials per worker")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL")
    parser.add_argument("--study-name", type=str, default="hpo_cifar10",
                        help="Optuna study name")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"],
                        help="Dataset to use")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=6900)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--trial-id", type=int, default=None)
    parser.add_argument("--parallel-workers", type=int, default=40,
                        help="Total parallel workers (for logging)")
    return parser.parse_args()


# Data
def get_dataloaders(data_dir, batch_size, num_workers, dataset="cifar10"):
    num_classes = 10 if dataset == "cifar10" else 100

    if dataset == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std  = [0.2470, 0.2435, 0.2616]
    else:
        mean = [0.5071, 0.4867, 0.4408]
        std  = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    DatasetClass = (torchvision.datasets.CIFAR10
                    if dataset == "cifar10"
                    else torchvision.datasets.CIFAR100)

    full_train = DatasetClass(root=data_dir, train=True,
                              download=True, transform=train_transform)
    val_set    = DatasetClass(root=data_dir, train=False,
                              download=True, transform=val_transform)

    n_val = 5000
    train_set, _ = random_split(
        full_train, [len(full_train) - n_val, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_set, batch_size=256, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True)
    return train_loader, val_loader, num_classes


# ── Model factory 
def build_model(model_name, dropout, num_classes=10):
    if model_name == "resnet18":
        m = torchvision.models.resnet18(weights=None)
        m.conv1   = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        m.fc      = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))
    elif model_name == "resnet34":
        m = torchvision.models.resnet34(weights=None)
        m.conv1   = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        m.fc      = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))
    elif model_name == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(m.classifier[1].in_features, num_classes)
        )
    elif model_name == "efficientnet_b1":
        m = torchvision.models.efficientnet_b1(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(m.classifier[1].in_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return m


# Trial objective 
def make_objective(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    def objective(trial: Trial) -> float:
        # search space
        lr             = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        weight_decay   = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size     = trial.suggest_categorical("batch_size", [64, 128, 256])
        dropout        = trial.suggest_float("dropout", 0.0, 0.5)
        optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
        scheduler_name = trial.suggest_categorical("scheduler", ["cosine", "step", "plateau"])
        model_name     = trial.suggest_categorical("model",
                            ["resnet18", "resnet34", "efficientnet_b0", "efficientnet_b1"])
        momentum = trial.suggest_float("momentum", 0.8, 0.99) if optimizer_name == "SGD" else 0.9

        # GPU start time (for budget tracking)
        gpu_start = time.time()

        # Build model & data 
        train_loader, val_loader, num_classes = get_dataloaders(
            args.data_dir, batch_size, args.num_workers, args.dataset
        )
        model     = build_model(model_name, dropout, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        if optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                  momentum=momentum, weight_decay=weight_decay,
                                  nesterov=True)
        elif optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr,
                                    weight_decay=weight_decay)

        if scheduler_name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif scheduler_name == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

        # Training loop 
        best_val_acc    = 0.0
        accuracy_at_epoch = {}

        for epoch in range(args.epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(bx), by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    correct += (model(bx).argmax(1) == by).sum().item()
                    total   += by.size(0)
            val_acc      = correct / total
            best_val_acc = max(best_val_acc, val_acc)

            # Track accuracy at each epoch (for budget analysis)
            elapsed_gpu_hours = (time.time() - gpu_start) / 3600
            accuracy_at_epoch[epoch] = {
                "val_acc": val_acc,
                "gpu_hours": elapsed_gpu_hours
            }

            if scheduler_name == "plateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()

            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            logger.info(f"Trial {trial.number} | Epoch {epoch+1}/{args.epochs} "
                        f"| val_acc={val_acc:.4f} | best={best_val_acc:.4f} "
                        f"| gpu_h={elapsed_gpu_hours:.3f}")

        #Final GPU hours 
        total_gpu_hours = (time.time() - gpu_start) / 3600

        # Save trial metadata
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Find accuracy thresholds (budget-aware metrics)
        thresholds = {}
        for thresh in [0.80, 0.85, 0.90, 0.93]:
            for ep, data in accuracy_at_epoch.items():
                if data["val_acc"] >= thresh:
                    thresholds[f"gpu_h_to_{int(thresh*100)}pct"] = data["gpu_hours"]
                    break

        meta = {
            "trial_number":     trial.number,
            "params":           trial.params,
            "best_val_acc":     best_val_acc,
            "total_gpu_hours":  total_gpu_hours,
            "slurm_task":       args.trial_id,
            "parallel_workers": args.parallel_workers,
            "sampler":          args.sampler,
            "dataset":          args.dataset,
            "device":           str(device),
            "accuracy_thresholds": thresholds,
            "accuracy_curve":   {str(k): v for k, v in accuracy_at_epoch.items()},
        }
        fname = out / f"trial_{trial.number:05d}_task{args.trial_id or 0:04d}.json"
        with open(fname, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Trial {trial.number} done | "
                    f"best={best_val_acc:.4f} | gpu_h={total_gpu_hours:.3f}")
        return best_val_acc

    return objective


# ── Create study with retry (fixes SQLite race condition)
def create_study_with_retry(study_name, storage, sampler, pruner, max_retries=10):
    """
    When 40 workers start simultaneously they all try to create the same
    SQLite table at once. This retry mechanism staggers the attempts.
    """
    for attempt in range(max_retries):
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True,
            )
            return study
        except Exception as e:
            if attempt < max_retries - 1:
                wait = random.uniform(1, 5) * (attempt + 1)
                logger.warning(f"Storage conflict (attempt {attempt+1}), "
                               f"retrying in {wait:.1f}s: {e}")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Failed to create study after max retries")


# main
def main():
    args = parse_args()

    # Stagger worker starts to reduce SQLite conflicts
    if args.trial_id:
        stagger = random.uniform(0, min(args.trial_id * 0.1, 10))
        logger.info(f"Worker stagger delay: {stagger:.1f}s")
        time.sleep(stagger)

    logger.info(f"Starting HPO | sampler={args.sampler} | "
                f"dataset={args.dataset} | trials={args.n_trials} | "
                f"task={args.trial_id} | parallel_workers={args.parallel_workers}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Sampler & pruner
    if args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(
            seed=42 + (args.trial_id or 0)
        )
        pruner  = optuna.pruners.NopPruner()
        logger.info("Sampler: RandomSampler")

    elif args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=10,
            multivariate=True,
            seed=42,
            constant_liar=True,   # KEY: designed for parallel workers!
        )
        pruner  = optuna.pruners.MedianPruner(n_startup_trials=5,
                                              n_warmup_steps=5)
        logger.info("Sampler: TPE (constant_liar=True for parallel workers)")

    elif args.sampler == "hyperband":
        sampler = optuna.samplers.TPESampler(seed=42, constant_liar=True)
        pruner  = optuna.pruners.HyperbandPruner(
            min_resource=3, max_resource=args.epochs, reduction_factor=3
        )
        logger.info("Sampler: TPE + HyperbandPruner")

    elif args.sampler == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(
            seed=42,
            restart_strategy="ipop",
        )
        pruner  = optuna.pruners.MedianPruner(n_startup_trials=5,
                                              n_warmup_steps=5)
        logger.info("Sampler: CMA-ES")

    # Create study 
    study = create_study_with_retry(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
    )

    # Optimize
    study.optimize(
        make_objective(args),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=False,
        gc_after_trial=True,
    )

    # Summary
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning("No completed trials!")
        return

    logger.info("=" * 60)
    logger.info(f"Best trial: #{study.best_trial.number}")
    logger.info(f"Best val accuracy: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    summary = {
        "sampler":            args.sampler,
        "dataset":            args.dataset,
        "parallel_workers":   args.parallel_workers,
        "n_trials_completed": len(completed),
        "best_trial":         study.best_trial.number,
        "best_val_acc":       study.best_value,
        "best_params":        study.best_params,
        "slurm_task":         args.trial_id,
    }
    out_path = (Path(args.output_dir) /
                f"summary_{args.sampler}_{args.trial_id or 0:04d}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary → {out_path}")


if __name__ == "__main__":
    main()
