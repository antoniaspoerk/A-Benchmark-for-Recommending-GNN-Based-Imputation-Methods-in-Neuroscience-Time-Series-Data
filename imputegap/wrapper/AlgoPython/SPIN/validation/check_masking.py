"""
Masking test on 1000 times 5 matrix:

    Test 1  training mask (observed)
            imputegap:   get_dataloader() in imputegap/.../SPIN/lib/data/dataset_imputegap.py
            original: tsl MissingValuesMixin.training_mask (used by run_imputation.py)
    Test 2  whitening (values hidden in every training batch)
            imputegap:   recovSPIN.train()
            original: tsl Imputer.on_train_batch_start() on the original SPINImputer
                     + the loss target of SPINImputer.training_step: original_mask - mask
    Test 3  evaluation masks   which values are missing and scored
            original: tsl sample_mask() with the paper's "point" and "block" settings
            imputegap:   SPIN's internal 25 % hold-out + ImputeGAP contamination patterns
    Test 4  final output       recovSPIN() fills only missing values, keeps observed values
"""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "masking_check.npz")
MASKS = os.path.join(HERE, "masking_masks.npz")
RESULTS = os.path.join(HERE, "results")
T, N, WINDOW, BATCH = 1000, 5, 24, 4
WHITEN_PROB = [0.2, 0.5, 0.8]
SEEDS = [0, 1, 2, 3, 4]
PATTERNS = ["aligned", "disjoint", "mcar", "scattered", "overlap"]
RATE = 0.2


def test_matrix():
    rng = np.random.default_rng(0)
    t = np.arange(T)[:, None]
    return (np.sin(2 * np.pi * t / 50 + np.arange(N)) + 0.1 * rng.standard_normal((T, N))).astype(np.float64)


def gap_stats(missing):
    """missing: (T, N) bool -> dict of missing-rate and gap statistics."""
    missing = np.asarray(missing, dtype=bool)
    lengths = []
    for col in missing.T:
        padded = np.concatenate([[0], col.astype(int), [0]])
        starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
        lengths.extend(ends - starts)
    lengths = np.array(lengths) if lengths else np.array([0])
    return dict(rate=missing.mean(), gaps=int((lengths > 0).sum()), mean_len=lengths.mean(),
                median_len=float(np.median(lengths)), max_len=int(lengths.max()),
                series_hit=int(missing.any(axis=0).sum()),
                co_missing=(missing.sum(axis=1) >= 2).mean())

def save():
    import torch
    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils
    from imputegap.wrapper.AlgoPython.SPIN import recovSPIN as mine
    from imputegap.wrapper.AlgoPython.SPIN.lib.data.dataset_imputegap import get_dataloader

    X = test_matrix()
    ts = TimeSeries()
    ts.import_matrix(X)
    out = {"X": X}

    for p in PATTERNS:
        incomp = utils.config_contamination(ts, p, dataset_rate=RATE, series_rate=RATE, verbose=False)
        out[f"pattern_{p}"] = np.isnan(incomp)
    incomp = np.where(out["pattern_aligned"], np.nan, X)

    # Test 1 
    loaders = get_dataloader(incomp, seq_len=WINDOW, batch_size=8, verbose=False)
    out["t1_observed"] = ~np.isnan(incomp)
    out["t1_eval_mine"] = loaders[0].dataset.eval_mask[..., 0].numpy().astype(bool)
    out["t1_train_mine"] = loaders[0].dataset.mask[..., 0].numpy().astype(bool)
    full = get_dataloader(X, seq_len=WINDOW, batch_size=8, verbose=False)  # on complete data
    out["internal_holdout"] = full[0].dataset.eval_mask[..., 0].numpy().astype(bool)

    # Test 2
    starts = [0, 200, 400, 600]
    M0 = torch.stack([torch.as_tensor(out["t1_train_mine"][s:s + WINDOW]) for s in starts])[..., None]
    y = torch.as_tensor(np.nan_to_num(incomp), dtype=torch.float32)
    Y = torch.stack([y[s:s + WINDOW] for s in starts])[..., None]
    batch = dict(x=Y * M0, y=Y, mask=M0, eval_mask=torch.zeros_like(M0), u=torch.zeros(BATCH, WINDOW, 4))

    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.seen = []

        def forward(self, x, u, mask, edge_index, edge_weight=None):
            self.seen.append(mask.detach().clone())
            return x * 0 + self.w, []

    captured = []
    for s in SEEDS:
        spy = Spy()
        torch.manual_seed(s)
        mine.train(spy, [batch], [batch], torch.zeros(2, 0, dtype=torch.long), None, torch.device("cpu"),
                   epochs=1, lr=1e-3, patience=1, batches_epoch=1, whiten_prob=WHITEN_PROB,
                   grad_clip_val=5.0, verbose=False, split_batch_in=1, precision=32)
        captured.append(spy.seen[0].numpy().astype(bool))  # first call = the training step
    out["t2_M0"] = M0.numpy().astype(bool)
    out["t2_captured_mine"] = np.stack(captured)

    # Test 4
    recov = mine.recovSPIN(incomp, epochs=2, batches_epoch=5, verbose=False)
    obs = ~np.isnan(incomp)
    out["t4_observed_unchanged"] = np.allclose(recov[obs], incomp[obs])
    out["t4_nans_left"] = int(np.isnan(recov).sum())
    out["t4_nans_before"] = int(np.isnan(incomp).sum())
    out["torch_version_mine"] = torch.__version__

    np.savez_compressed(FILE, **out)
    print(f"saved -> {FILE}")


SPIN_COMMIT = "7349ba31da7306e7e96c13668a3f1f0a4df90902"


def authors_repo():
    """Path to the authors' SPIN repository; cloned at the pinned commit if it is missing."""
    import subprocess
    repo = os.environ.get("SPIN_REPO", os.path.expanduser("~/spin-original/spin"))
    if not os.path.isdir(os.path.join(repo, "spin")):
        print(f"authors' repository not found -> cloning it to {repo}")
        os.makedirs(os.path.dirname(repo), exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "https://github.com/Graph-Machine-Learning-Group/spin.git",
                        repo], check=True)
        subprocess.run(["git", "-C", repo, "checkout", "--quiet", SPIN_COMMIT], check=True)
    return repo


def check():
    import torch
    sys.path.insert(0, authors_repo())
    from spin.imputers import SPINImputer
    from tsl.datasets.prototypes.mixin import MissingValuesMixin
    from tsl.ops.imputation import sample_mask

    z = dict(np.load(FILE))
    lines, ok = [], {}

    def say(text=""):
        print(text)
        lines.append(text)

    say("SPIN masking: ImputeGAP integration vs. authors' code")
    say(f"test matrix {T} x {N}; torch imputegap {z['torch_version_mine']} / authors {torch.__version__}\n")

    # Test 1 
    obj = types.SimpleNamespace(mask=z["t1_observed"], eval_mask=z["t1_eval_mine"].astype("uint8"))
    train_authors = np.asarray(MissingValuesMixin.training_mask.fget(obj)).astype(bool)
    ok[1] = np.array_equal(train_authors, z["t1_train_mine"])
    say("Test 1  training mask (observed AND NOT held-out)")
    say(f"        identical: {ok[1]}  ({int(z['t1_train_mine'].sum())} usable values, "
        f"{int((train_authors != z['t1_train_mine']).sum())} differences)\n")

    # Test 2 
    class AttrDict(dict):
        __getattr__ = dict.__getitem__

        def __setattr__(self, key, value):
            self[key] = value

    for cls in SPINImputer.__mro__:
        hook = cls.__dict__.get("on_train_batch_start")
        if hook is not None and cls.__module__.split(".")[0] in ("pytorch_lightning", "lightning"):
            setattr(cls, "on_train_batch_start", lambda self, *args, **kwargs: None)
            break

    imputer = SPINImputer.__new__(SPINImputer)  # authors' class; only the whitening method is used
    object.__setattr__(imputer, "whiten_prob", torch.Tensor(WHITEN_PROB))
    M0 = torch.as_tensor(z["t2_M0"]).to(torch.uint8)
    same_mask, same_target, fractions = [], [], []
    for i, s in enumerate(SEEDS):
        batch = AttrDict(input=AttrDict(x=torch.randn(M0.shape) * 0, mask=M0.clone()))
        torch.manual_seed(s)
        imputer.on_train_batch_start(batch, 0, 0)
        kept_authors = batch.input.mask.bool().numpy()
        target_authors = (batch.original_mask - batch.input.mask).bool().numpy()  # SPINImputer.training_step
        kept_mine = z["t2_captured_mine"][i]
        target_mine = z["t2_M0"] & ~kept_mine                                     # recovSPIN.train
        same_mask.append(np.array_equal(kept_authors, kept_mine))
        same_target.append(np.array_equal(target_authors, target_mine))
        observed = z["t2_M0"].reshape(BATCH, -1).sum(1)
        fractions.append((target_mine.reshape(BATCH, -1).sum(1) / observed).round(2).tolist())
    hides_only_observed = not (z["t2_captured_mine"] & ~z["t2_M0"][None]).any()
    ok[2] = all(same_mask) and all(same_target) and hides_only_observed
    say("Test 2  whitening during training (whiten_prob = [0.2, 0.5, 0.8], one p per sample)")
    say(f"        identical kept-mask for seeds {SEEDS}: {same_mask}")
    say(f"        identical loss target (injected missing):  {same_target}")
    say(f"        never un-hides missing values: {hides_only_observed}")
    for s, f in zip(SEEDS, fractions):
        say(f"        seed {s}: hidden fraction per sample = {f}")
    if not all(same_mask):
        say("        NOTE: masks differ value by value -> the two torch versions draw different random")
        say("        numbers; check that the hidden fractions still cluster around 0.2 / 0.5 / 0.8.")
    say()

    # Test 3
    point = sample_mask((T, N), p=0.0, p_noise=0.25, rng=np.random.default_rng(9101112), verbose=False).astype(bool)
    block = sample_mask((T, N), p=0.0015, p_noise=0.05, min_seq=12, max_seq=48,
                        rng=np.random.default_rng(9101112), verbose=False).astype(bool)
    masks = {"paper: point (25 % random)": point,
             "paper: block (faults 12-48 + 5 % points)": block,
             "imputegap: SPIN internal hold-out (25 %)": z["internal_holdout"]}
    masks.update({f"ImputeGAP pattern: {p} ({RATE:.0%})": z[f"pattern_{p}"] for p in PATTERNS})
    say("Test 3  evaluation / missing-value masks on a 1000 x 5 matrix (expected to differ)")
    say(f"        {'mask':44s} {'rate':>6s} {'gaps':>5s} {'mean len':>9s} {'median':>7s} {'max':>5s} "
        f"{'series':>7s} {'co-missing':>11s}")
    for name, m in masks.items():
        g = gap_stats(m)
        say(f"        {name:44s} {g['rate']:6.1%} {g['gaps']:5d} {g['mean_len']:9.1f} {g['median_len']:7.1f} "
            f"{g['max_len']:5d} {g['series_hit']:7d} {g['co_missing']:11.1%}")
    say("        (co-missing = share of time steps where >= 2 series are missing at the same time)\n")

    # Test 4 
    ok[4] = bool(z["t4_observed_unchanged"]) and int(z["t4_nans_left"]) == 0
    say("Test 4  recovSPIN() output")
    say(f"        observed values unchanged: {bool(z['t4_observed_unchanged'])}; "
        f"missing values filled: {int(z['t4_nans_before'])} -> {int(z['t4_nans_left'])} left\n")

    say("Summary")
    for k, name in [(1, "training mask"), (2, "whitening"), (4, "final output")]:
        say(f"        Test {k} {name:15s}: {'PASS' if ok[k] else 'CHECK'}")
    say("        Test 3 evaluation masks : documented above (differences by design)")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "masking_check.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    np.savez_compressed(MASKS, **{k.replace(" ", "_"): v for k, v in masks.items()})
    print(f"\nsaved -> {os.path.join(RESULTS, 'masking_check.txt')}")
    try:
        plot()
    except ImportError:
        print("(matplotlib not installed here -> run 'python check_masking.py plot' in env thesis for the figure)")
    os.remove(FILE)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masks = dict(np.load(MASKS))
    fig, axes = plt.subplots(len(masks), 2, figsize=(14, 1.6 * len(masks)),
                             gridspec_kw={"width_ratios": [4, 1]})
    for row, (name, m) in enumerate(masks.items()):
        axes[row, 0].imshow(m.T, aspect="auto", cmap="Greys", interpolation="nearest")
        axes[row, 0].set_title(name.replace("_", " "), fontsize=9, loc="left")
        axes[row, 0].set_yticks(range(N))
        axes[row, 0].set_ylabel("series", fontsize=8)
        lengths = []
        for col in np.asarray(m, bool).T:
            padded = np.concatenate([[0], col.astype(int), [0]])
            lengths.extend(np.flatnonzero(np.diff(padded) == -1) - np.flatnonzero(np.diff(padded) == 1))
        axes[row, 1].hist(lengths or [0], bins=30, color="tab:blue")
        axes[row, 1].set_title("gap lengths", fontsize=8)
    axes[-1, 0].set_xlabel("time step")
    fig.suptitle("Missing-value masks: paper (tsl) vs. SPIN in ImputeGAP (black = missing)", fontsize=11)
    fig.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "masking_masks.png")
    fig.savefig(path, dpi=150)
    print(f"saved -> {path}")
    os.remove(MASKS)


def run_all():
    """Run every step this environment can run: save (needs imputegap), check (needs tsl)."""
    import importlib.util
    has = lambda module: importlib.util.find_spec(module) is not None
    if has("imputegap"):
        save()
    if has("tsl"):
        check()
    elif has("imputegap"):
        print("\nNext: activate the authors' environment and run  python check_masking.py check")


if __name__ == "__main__":
    steps = {"save": save, "check": check, "plot": plot, "all": run_all}
    steps[sys.argv[1] if len(sys.argv) > 1 else "all"]()
