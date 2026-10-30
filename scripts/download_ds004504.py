
import argparse
from pathlib import Path
import openneuro

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "ds004504"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=["sub-001"])
    args = parser.parse_args()

    include = ["participants.tsv", "participants.json", "dataset_description.json"]
    for sub in args.subjects:
        include += [f"{sub}/*", f"derivatives/{sub}/*"]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    openneuro.download(dataset="ds004504", target_dir=RAW_DIR, include=include)
    print(f"Downloaded to {RAW_DIR}")


if __name__ == "__main__":
    main()
