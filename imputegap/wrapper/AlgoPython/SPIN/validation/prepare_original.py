import os
import subprocess
import sys

REPO_URL = "https://github.com/Graph-Machine-Learning-Group/spin.git"
COMMIT = "7349ba31da7306e7e96c13668a3f1f0a4df90902"  
SPIN_REPO = os.path.expanduser("~/spin-original/spin")


def clone_repository():
    if not os.path.isdir(SPIN_REPO):
        os.makedirs(os.path.dirname(SPIN_REPO), exist_ok=True)
        subprocess.run(["git", "clone", REPO_URL, SPIN_REPO], check=True)
    subprocess.run(["git", "-C", SPIN_REPO, "checkout", "--quiet", COMMIT], check=True)
    print(f"Repository: {SPIN_REPO} at commit {COMMIT[:7]}")


def print_versions():
    import pytorch_lightning
    import torch
    import torch_geometric
    import torchmetrics
    import tsl
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | torch-geometric {torch_geometric.__version__} "
          f"| pytorch-lightning {pytorch_lightning.__version__} | torchmetrics {torchmetrics.__version__} "
          f"| tsl {tsl.__version__}")


def download_datasets():
    os.chdir(SPIN_REPO)  
    from tsl.datasets import AirQuality, MetrLA, PemsBay
    for name, load in [("AQI-36", lambda: AirQuality(impute_nans=True, small=True)),
                       ("METR-LA", MetrLA),
                       ("PEMS-BAY", PemsBay)]:
        dataset = load()
        print(f"{name}: {dataset.length} time steps x {dataset.n_nodes} sensors")


if __name__ == "__main__":
    clone_repository()
    print_versions()
    download_datasets()
    print("Preparation done.")
