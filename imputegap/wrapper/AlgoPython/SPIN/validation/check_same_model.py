
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "same_model.npz")
HPARAMS = dict(input_size=1, hidden_size=32, n_nodes=12, u_size=4, output_size=1,
               temporal_self_attention=True, reweight="softmax", n_layers=4, eta=3, message_layers=1)
B, T = 2, 24


def run(model, z):
    model.eval()
    with torch.no_grad():
        x_hat, imps = model(torch.from_numpy(z["x"]), torch.from_numpy(z["u"]), torch.from_numpy(z["mask"]),
                            torch.from_numpy(z["edge_index"]))
    return [x_hat.numpy()] + [i.numpy() for i in imps]


def save():
    from imputegap.wrapper.AlgoPython.SPIN.lib.models.spin import SPINModel
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    N = HPARAMS["n_nodes"]
    adj = np.triu(rng.random((N, N)) < 0.35, 1)
    adj = adj | adj.T
    z = dict(x=rng.standard_normal((B, T, N, 1)).astype(np.float32),
             mask=(rng.random((B, T, N, 1)) > 0.3),
             u=rng.standard_normal((B, T, 4)).astype(np.float32),
             edge_index=np.stack(np.nonzero(adj)).astype(np.int64))
    model = SPINModel(**HPARAMS)
    outs = run(model, z)
    weights = {f"w__{k}": v.detach().numpy() for k, v in model.state_dict().items()}
    np.savez(FILE, **z, **weights, **{f"out_{i}": o for i, o in enumerate(outs)})
    print(f"ImputeGAP version: {sum(v.size for v in weights.values())} parameters in {len(weights)} tensors")
    print(f"saved -> {FILE}")


def check():
    sys.path.insert(0, os.environ.get("SPIN_REPO", os.path.expanduser("~/spin-original/spin")))
    from spin.models import SPINModel
    z = dict(np.load(FILE))
    os.remove(FILE)
    model = SPINModel(**HPARAMS)
    theirs = model.state_dict()
    yours = {k[3:]: v for k, v in z.items() if k.startswith("w__")}
    missing, extra = set(theirs) - set(yours), set(yours) - set(theirs)
    shape_diff = [k for k in set(theirs) & set(yours) if tuple(theirs[k].shape) != yours[k].shape]
    print(f"parameter names: authors {len(theirs)}, ImputeGAP version {len(yours)}; "
          f"only in authors: {sorted(missing)}; only in ImputeGAP version: {sorted(extra)}; shape mismatches: {shape_diff}")
    if missing or extra or shape_diff:
        print("RESULT: DIFFERENT architecture (parameters do not line up)")
        return
    model.load_state_dict({k: torch.from_numpy(v) for k, v in yours.items()})
    outs = run(model, z)
    worst = 0.0
    for i, o in enumerate(outs):
        d = float(np.max(np.abs(o - z[f"out_{i}"])))
        worst = max(worst, d)
        print(f"  output {i} ({'final x_hat' if i == 0 else f'layer {i} imputation'}): max |diff| = {d:.2e}")
    print("RESULT:", "SAME model (identical outputs)" if worst < 1e-5 else "outputs DIFFER")


if __name__ == "__main__":
    {"save": save, "check": check}[sys.argv[1] if len(sys.argv) > 1 else "save"]()
