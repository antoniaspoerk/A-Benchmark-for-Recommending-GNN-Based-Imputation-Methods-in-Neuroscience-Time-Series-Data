
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.ds004504 import load_subject, save_subject

RAW_DIR = ROOT / "data" / "raw" / "ds004504"
OUT_DIR = ROOT / "data" / "processed" / "ds004504"
DATASETS_DIR = ROOT / "ImputeGAP" / "imputegap" / "datasets"  # ImputeGAP loads datasets from here


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=["sub-001"])
    parser.add_argument("--sfreq", type=float, default=128.0)
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--raw", action="store_true", help="use raw instead of derivatives")
    args = parser.parse_args()

    for sub in args.subjects:
        X, meta = load_subject(RAW_DIR, sub, sfreq=args.sfreq, duration=args.duration,
                               use_derivatives=not args.raw)
        meta["source_file"] = str(Path(meta["source_file"]).relative_to(ROOT))
        path = save_subject(X, meta, OUT_DIR, DATASETS_DIR)
        print(f"{sub}: X shape {X.shape} (T x N), {meta['sfreq']} Hz, "
              f"group={meta.get('group')} -> {path}")


if __name__ == "__main__":
    main()