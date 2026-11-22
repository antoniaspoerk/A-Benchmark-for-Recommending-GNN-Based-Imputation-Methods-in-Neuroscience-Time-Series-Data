import csv
import os
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Desktop/Master_Thesis/cluster_results/spin_comparison")
REPO_ASSETS = os.path.expanduser("~/Desktop/Master_Thesis/A-Benchmark-for-Recommending-GNN-Based-Imputation-Methods-"
                                 "in-Neuroscience-Time-Series-Data/imputegap/imputegap_assets/spin_validation")
PAPER = "original_spin_reproduces_paper"
TRAINING = "imputegap_spin_matches_original_training"
MODEL = "imputegap_spin_matches_original_model"
MASKING = "imputegap_masking_matches_paper"
A, B, C, M = (os.path.join(ROOT, n) for n in (PAPER, TRAINING, MODEL, MASKING))
SCALE = 83.49
PAPER_MAE, PAPER_STD = 11.77, 0.54

for old, new in (("level_A_original_spin", PAPER), ("level_B_imputegap_version", TRAINING),
                 ("level_C_same_model", MODEL)):
    if os.path.isdir(os.path.join(ROOT, old)) and not os.path.exists(os.path.join(ROOT, new)):
        os.rename(os.path.join(ROOT, old), os.path.join(ROOT, new))
for d in (A, B):
    os.makedirs(os.path.join(d, "raw"), exist_ok=True)
for d in (C, M):
    os.makedirs(d, exist_ok=True)


def move(src, dst):
    src, dst = os.path.join(ROOT, src), os.path.join(ROOT, dst)
    if not os.path.exists(src):
        return
    if os.path.getsize(src) == 0 or (os.path.exists(dst) and os.path.getsize(dst) >= os.path.getsize(src)):
        os.remove(src)
        return
    shutil.move(src, dst)


move("original_air36_seed1.log", f"{PAPER}/raw/original_air36_seed1.log")
move("results/original_air36_seed1.log", f"{PAPER}/raw/original_air36_seed1.log")
move("results/original.csv", f"{PAPER}/raw/original.csv")
if os.path.isdir(os.path.join(ROOT, "logs")):
    for f in os.listdir(os.path.join(ROOT, "logs")):
        target = {"spin_orig": PAPER, "spin_mine": TRAINING}.get(f[:9])
        if target:
            move(f"logs/{f}", f"{target}/raw/{f}")
move("results/mine_yours.csv", f"{TRAINING}/raw/mine_yours.csv")
for src in ("level_c_same_model.txt", "results/level_c_same_model.txt", f"{MODEL}/level_c_same_model.txt"):
    move(src, f"{MODEL}/output_difference_same_weights.txt")
move("results/masking_check.txt", f"{MASKING}/masking_check.txt")
move("results/masking_masks.png", f"{MASKING}/masking_masks.png")
for d in ("results", "logs"):
    p = os.path.join(ROOT, d)
    if os.path.isdir(p) and not [x for x in os.listdir(p) if x != ".DS_Store"]:
        shutil.rmtree(p)
for f in ("README.txt", "comparison_A_vs_B.txt", "comparison_A_vs_B.csv", "comparison_A_vs_B_validation.png",
          f"{PAPER}/summary.txt", f"{TRAINING}/summary.txt", f"{PAPER}/training_curve.png",
          f"{TRAINING}/training_curve.png"):
    if os.path.exists(os.path.join(ROOT, f)):
        os.remove(os.path.join(ROOT, f))


def last_row(path):
    if not os.path.exists(path):
        return {}
    rows = list(csv.DictReader(open(path)))
    return rows[-1] if rows else {}


def parse_original():
    path = os.path.join(A, "raw", "original_air36_seed1.log")
    if not os.path.exists(path):
        return None
    text = open(path, errors="replace").read().replace("\r", "\n")
    last = {}
    for m in re.finditer(r"^Epoch (\d+): 100%\|.*?val_mae=([\d.e+]+).*?train_mae=([\d.e+]+)", text, re.M):
        last[int(m.group(1)) + 1] = (float(m.group(2)), float(m.group(3)))
    test = re.findall(r"Test MAE: ([0-9.]+)", text)
    row = last_row(os.path.join(A, "raw", "original.csv"))
    ep = sorted(last)
    return dict(epochs=ep, val=[last[e][0] for e in ep], extra=[last[e][1] for e in ep],
                test=float(test[-1]) if test else None,
                seconds=row.get("seconds", ""), node=row.get("node", "diufrd204"), job="32381_0")


def parse_imputegap():
    out = [f for f in os.listdir(os.path.join(B, "raw")) if f.endswith(".out")]
    if not out:
        return None
    text = open(os.path.join(B, "raw", out[0])).read()
    ep = re.findall(r"epoch (\d+)/\d+\s+train_loss ([\d.]+)\s+val_mae ([\d.]+)", text)
    test = re.findall(r"Test MAE: ([0-9.]+)", text)
    row = last_row(os.path.join(B, "raw", "mine_yours.csv"))
    return dict(epochs=[int(e) for e, _, _ in ep], val=[float(v) * SCALE for _, _, v in ep],
                extra=[float(l) for _, l, _ in ep], test=float(test[-1]) if test else None,
                seconds=row.get("seconds", ""), node=row.get("node", ""), job="32391_0")


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def plot(path, curves):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ep, val, label, color in curves:
        ax.plot(ep, val, color=color, lw=1.3, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MAE")
    ax.set_ylim(top=min(30, max(max(c[1]) for c in curves)))
    ax.grid(alpha=.3)
    if any(c[2] for c in curves):
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def best(r):
    i = min(range(len(r["val"])), key=r["val"].__getitem__)
    return r["epochs"][i], r["val"][i]


orig, imp = parse_original(), parse_imputegap()
summary = []
for r, d, extra_name, color, model, png in (
        (orig, A, "train_mae", "#d9622b", "original SPIN", "validation_mae_original_spin.png"),
        (imp, B, "train_loss", "#2a6fdb", "ImputeGAP version", "validation_mae_imputegap_spin.png")):
    if not r or not r["epochs"]:
        continue
    write_csv(os.path.join(d, "training_curve.csv"), ["epoch", "val_mae", extra_name],
              [[e, f"{v:.4f}", f"{x:.4f}"] for e, v, x in zip(r["epochs"], r["val"], r["extra"])])
    be, bv = best(r)
    res = ["air36", 1, r["job"], r["node"], "" if r["test"] is None else f"{r['test']:.4f}",
           f"{bv:.4f}", be, len(r["epochs"]), r["seconds"]]
    write_csv(os.path.join(d, "result.csv"),
              ["dataset", "seed", "job", "node", "test_mae", "best_val_mae", "best_epoch", "epochs", "runtime_s"], [res])
    summary.append([model] + res[4:])
    plot(os.path.join(d, png), [(r["epochs"], r["val"], "", color)])

if summary:
    summary.append(["paper (Marisca et al. 2022)", f"{PAPER_MAE} ± {PAPER_STD}", "", "", "", ""])
    write_csv(os.path.join(B, "original_vs_imputegap.csv"),
              ["model", "test_mae", "best_val_mae", "best_epoch", "epochs", "runtime_s"], summary)
if orig and imp and orig["epochs"] and imp["epochs"]:
    plot(os.path.join(B, "validation_mae_original_vs_imputegap.png"),
         [(orig["epochs"], orig["val"], "original SPIN", "#d9622b"),
          (imp["epochs"], imp["val"], "ImputeGAP version", "#2a6fdb")])

if os.path.isdir(os.path.dirname(REPO_ASSETS)):
    for folder in (PAPER, TRAINING, MODEL, MASKING):
        src = os.path.join(ROOT, folder)
        files = [f for f in os.listdir(src) if f.endswith((".png", ".txt"))] if os.path.isdir(src) else []
        if files:
            os.makedirs(os.path.join(REPO_ASSETS, folder), exist_ok=True)
        for f in files:
            shutil.copy2(os.path.join(src, f), os.path.join(REPO_ASSETS, folder, f))
