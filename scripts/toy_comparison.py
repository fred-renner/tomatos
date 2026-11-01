#!/usr/bin/env python

# A/B on the toy: same data, trained once with all systematics in the loss and
# once with stat uncertainties only. Both are judged the same way: sharp
# histograms on held-out events with the full model (test_loss).

import argparse
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np

runs = {
    "with systematics": ("toy", "#2a78d6"),
    "stat only": ("toy_nosys", "#eb6834"),
}
objectives = {
    "cls_nn": "NN analysis",
    "cls_var": "mass-fit analysis",
}


def ema(x, alpha=0.1):
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def load(results, objective, suffix, steps):
    path = f"{results}/tomatos_{objective}_steps_{steps}_bins_5_{suffix}/"
    with h5py.File(path + "metrics.h5", "r") as f:
        m = {k: f[k][:] for k in ["test_loss", "train_loss", "cut_j1_pt", "cut_j2_pt"]}
    with open(path + "infer_metrics.json") as f:
        m["best"] = json.load(f)["epoch_best"]
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--out", default="assets/toy_comparison.png")
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, (objective, title) in zip(axes, objectives.items()):
        for label, (suffix, color) in runs.items():
            m = load(args.results, objective, suffix, args.steps)
            steps = np.arange(len(m["test_loss"]))
            # step 0 is the solver init, skip it
            cls = m["test_loss"][1:]
            row[0].plot(steps[1:], cls, color=color, lw=1, alpha=0.25)
            row[0].plot(steps[1:], ema(cls), color=color, lw=2, label=label)
            row[1].plot(steps, m["cut_j1_pt"] / 1e3, color=color, lw=2, label=label)
            best = m["best"]["epoch"]
            print(
                f"{objective:8s} {label:17s} best epoch {best:3d}: "
                f"test CLs {m['test_loss'][best]:.4f}, "
                f"j1_pt > {m['best']['cut_j1_pt'] / 1e3:.1f} GeV, "
                f"j2_pt > {m['best']['cut_j2_pt'] / 1e3:.1f} GeV"
            )
        row[0].set_yscale("log")
        row[0].set_ylabel("expected $CL_s$ (test, full model)")
        row[0].set_title(title, loc="left", fontweight="bold")
        row[1].set_ylabel("leading-jet $p_T$ cut [GeV]")
        for ax in row:
            ax.grid(alpha=0.25)
            ax.set_xlabel("training step")
        row[0].legend(frameon=False)
    fig.savefig(args.out, dpi=150)
    print(f"written {args.out}")
