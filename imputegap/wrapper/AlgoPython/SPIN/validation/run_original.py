#!/usr/bin/env python
#SBATCH --job-name=spin_orig
#SBATCH --partition=GPU
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-14%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=antonia.spoerk@unifr.ch

import argparse
import csv
import os
import re
import socket
import subprocess
import sys
import time

DATASETS = ["air36", "la_point", "la_block", "bay_point", "bay_block"]
SEEDS = [1, 2, 3]
SPIN_REPO = os.path.expanduser("~/spin-original/spin")
RESULTS = os.path.expanduser("~/spin-original/validation/results")


def dataset_and_seed():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--seed", type=int)
    a = p.parse_args()
    if a.dataset is not None:
        return a.dataset, a.seed or 1
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    return DATASETS[task // len(SEEDS)], SEEDS[task % len(SEEDS)]


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip() or "-"
    except FileNotFoundError:
        return "-"


def main():
    dataset, seed = dataset_and_seed()
    os.makedirs(RESULTS, exist_ok=True)
    log_path = os.path.join(RESULTS, f"original_{dataset}_seed{seed}.log")
    print(f"Original SPIN | dataset={dataset} seed={seed} | {socket.gethostname()} | {gpu_name()}", flush=True)

    command = [sys.executable, "./experiments/run_imputation.py", "--config", "imputation/spin.yaml",
               "--model-name", "spin", "--dataset-name", dataset, "--seed", str(seed)]
    env = dict(os.environ, PYTHONPATH=SPIN_REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))

    start = time.time()
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, cwd=SPIN_REPO, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:          # show the output live and keep a copy
            print(line, end="", flush=True)
            log.write(line)
        process.wait()
    seconds = int(time.time() - start)

    with open(log_path) as log:
        found = re.findall(r"Test MAE: ([0-9.]+)", log.read())
    mae = found[-1] if found else "FAILED"

    out = os.path.join(RESULTS, "original.csv")
    new_file = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["dataset", "seed", "test_mae", "seconds", "node"])
        writer.writerow([dataset, seed, mae, seconds, socket.gethostname()])
    print(f"Done: {dataset} seed {seed} -> Test MAE {mae} ({seconds} s)")
    sys.exit(0 if found else 1)


if __name__ == "__main__":
    main()
