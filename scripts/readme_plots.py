#!/usr/bin/env python3
"""README figures for the toy demos, from finished runs.

inputs.png      the problem: a small peak on a large, partly unknown background
mass.gif        mass fit: selected events, Z of both trainings, the pt cuts
nn.gif          nn: score distribution, Z of both trainings
comparison.png  claimed vs delivered Z, with and without systematics

    python scripts/readme_plots.py \
        --mass results/.../tomatos_cls_var_..._mass results/.../tomatos_cls_var_..._mass_nosys \
        --nn results/.../tomatos_cls_nn_..._nn results/.../tomatos_cls_nn_..._nn_nosys
"""
import argparse
import json
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import generate_toy_files as toy  # noqa: E402

TOMATO, BKG, INK, BLUE = "#e5533d", "#8fb0d6", "#2b2b2b", "#4a78b5"
RUNS = {"sys": ("trained with uncertainties", TOMATO), "nosys": ("trained without uncertainties", BLUE)}
TITLES = {"mass": "Mass-fit analysis", "nn": "Neural-network analysis"}
SIG, EST = "ggZH125_vvbb_NOSYS", "bkg_estimate_NOSYS"
CUT_LABELS = {"j1_pt": "$p_{T,1}$ [GeV]", "j2_pt": "$p_{T,2}$ [GeV]"}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    }
)


class Run:
    """Everything the figures need from one training, loaded once."""

    def __init__(self, path):
        with open(os.path.join(path, "train_config.json")) as f:
            self.config = json.load(f)
        with h5py.File(os.path.join(path, "metrics.h5"), "r") as m:
            self.m = {k: m[k][:] for k in m}
        # losses are -Z for the discovery test, step 0 holds the solver init
        self.z = -self.m["test_loss"]
        self.z_claimed = -self.m["test_loss_stat_only"]
        self.steps = len(self.z)
        self.frames = list(range(0, self.steps, self.config["movie_batch_modulo"]))

    def edges(self, step):
        # physical units for mass (see train_utils.log_bins), score for nn
        return np.asarray(self.m["bins"][step])

    def to_x(self, u):
        # kde grid lives in [0, 1] of the binning range
        if self.config["objective"] == "cls_var":
            lo, hi = self.config["fit_range"]
            return (lo + u * (hi - lo)) / 1e3
        return u

    def final(self, z, n=20):
        return float(np.median(z[-n:]))


def selection_panel(ax, run, step, mode):
    """Counted test events per bin (stacked) and the smooth kde the optimizer sees."""
    edges = run.edges(step)
    if mode == "mass":
        edges = edges / 1e3
    width = np.diff(edges)
    b = run.m["h_" + EST + "_test"][step]
    s = run.m["h_" + SIG + "_test"][step]
    ax.bar(edges[:-1], b / width, width, align="edge", color=BKG, label="background estimate")
    ax.bar(edges[:-1], s / width, width, bottom=b / width, align="edge", color=TOMATO, label="signal")
    # kde of the closest logged frame, scaled like train_utils.rescale_kde:
    # matched to the largest train bin, hence divide by that bin's width
    kde_step = step - step % run.config["movie_batch_modulo"]
    grid = run.to_x(np.asarray(run.config["kde_bins"][:-1]))
    for key, color in ((EST, INK), (SIG, TOMATO)):
        h_train = run.m["h_" + key][kde_step]
        kde = run.m["kde_" + key][kde_step]
        ax.plot(grid, kde / width[np.argmax(h_train)], color=color, lw=1.3, ls="--",
                label="smooth (KDE) " + ("background" if key == EST else "signal"))
    for e in edges[1:-1]:
        ax.axvline(e, color=INK, lw=0.6, alpha=0.35)
    if mode == "mass":
        ax.set(xlim=(edges[0], edges[-1]), xlabel="$m_{bb}$ [GeV]", ylabel="events / GeV")
    else:
        ax.set(xlim=(0, 1), xlabel="NN score", ylabel="events / unit score")
    ax.set_yscale("log")
    top = max((b + s) / width) * 8
    ax.set_ylim(top / 1e4, top)
    ax.set_title("Selected events (test set)")
    ax.legend(loc="upper right", ncol=2)


def z_panel(ax, runs, step):
    for name, run in runs.items():
        label, color = RUNS[name]
        x = np.arange(1, step + 1)
        ax.plot(x, run.z[1 : step + 1], color=color, lw=2, label=label)
        if step >= 1:
            ax.plot(step, run.z[step], "o", color=color, ms=4)
    ymax = 1.15 * max(np.nanmax(r.z[1:]) for r in runs.values())
    ax.set(xlim=(0, runs["sys"].steps), ylim=(0, ymax), xlabel="training step", ylabel="Z  [σ]")
    ax.set_title("Discovery significance (test set, all uncertainties)")
    ax.legend(loc="lower right")


def cut_panel(ax, runs, step, var, events):
    bins = np.linspace(30, 300, 55)
    for sample, color, label in (("bkg", BKG, "background"), ("sig", TOMATO, "signal")):
        x, w = events[sample][var] / 1e3, events[sample]["weight"]
        style = dict(color=color, label=label, density=True)
        style |= dict(alpha=0.8) if sample == "bkg" else dict(histtype="step", lw=2)
        ax.hist(x, bins, weights=w, **style)
    for name, run in runs.items():
        label, color = RUNS[name]
        cut = run.m["cut_" + var][step] / 1e3
        ax.axvline(cut, color=color, lw=2.2, label=f"cut, {label}")
        if name == "sys":
            ax.axvspan(bins[0], cut, color="white", alpha=0.6, lw=0)
    ax.set(yscale="log", xlim=(bins[0], bins[-1]), ylim=(1e-5, 0.1), xlabel=CUT_LABELS[var], ylabel="a.u.")
    ax.set_title(f"Cut on {CUT_LABELS[var].split(' ')[0]}")
    ax.legend(loc="upper right")


def load_events(ntuples):
    events = {}
    for sample, folder in (("sig", "ggZH125_vvbb"), ("bkg", "bkg")):
        with uproot.open(os.path.join(ntuples, folder, "NOSYS.root")) as f:
            events[sample] = f["FilteredTree"].arrays(library="np")
    return events


def movie(runs, mode, out, events, fps=12, hold=30, dpi=80):
    sys_run = runs["sys"]
    cuts = list(sys_run.config["opt_cuts"])
    shape, size = ((2, 2), (13, 8.6)) if cuts else ((1, 2), (13, 4.9))
    fig, axes = plt.subplots(*shape, figsize=size, layout="constrained")
    frames = sys_run.frames + [sys_run.steps - 1] * hold

    def draw(step):
        for ax in axes.flat:
            ax.clear()
        panels = list(axes.flat)
        selection_panel(panels[0], sys_run, step, mode)
        z_panel(panels[1], runs, step)
        for ax, var in zip(panels[2:], cuts):
            cut_panel(ax, runs, step, var, events)
        fig.suptitle(f"{TITLES[mode]}   ·   step {step:3d}", x=0.01, ha="left", fontsize=13, color=INK)

    anim = FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames))
    anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    # still of the last frame, handy to look at without playing the gif
    draw(sys_run.steps - 1)
    fig.savefig(out.replace(".gif", "_final.png"), dpi=dpi)
    plt.close(fig)
    print(f"written {out}")


def inputs(events, out, dpi=130):
    b, s = events["bkg"], events["sig"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained")

    ax, bins = axes[0], np.linspace(50, 250, 81)
    onset = toy.bkg_modelling_pars[1]
    m, pt, w = b["h_m"] / 1e3, b["j1_pt"] / 1e3, b["weight"]
    hard = pt > onset
    h = np.histogram(m, bins, weights=w)[0]
    h_hard = np.histogram(m[hard], bins, weights=w[hard])[0]
    unc = np.histogram(m, bins, weights=w * (b["bkg_mod_up"] - 1))[0]
    ax.stairs(h, bins, fill=True, color=BKG, alpha=0.55, label="background")
    ax.stairs(h_hard, bins, fill=True, color=BLUE, alpha=0.8, label=f"… with $p_{{T,1}}$ > {onset:.0f} GeV")
    ax.stairs(h_hard + unc, bins, baseline=h_hard - unc, fill=True, color="none", hatch="////",
              edgecolor=INK, lw=0, label="its uncertainty")
    ax.hist(s["h_m"] / 1e3, bins, weights=20 * s["weight"], histtype="step", color=TOMATO, lw=2, label="signal ×20")
    ax.set(xlim=(bins[0], bins[-1]), xlabel="$m_{bb}$ [GeV]", ylabel="events")
    ax.set_title("Mass spectrum")
    ax.legend(loc="upper right")

    ax, bins = axes[1], np.linspace(30, 300, 55)
    ax.hist(pt, bins, weights=w, density=True, color=BKG, alpha=0.8, label="background")
    ax.hist(s["j1_pt"] / 1e3, bins, weights=s["weight"], density=True, histtype="step", color=TOMATO, lw=2,
            label="signal")
    ax.axvspan(onset, bins[-1], color=INK, alpha=0.07, lw=0)
    ax.text(onset + 4, 0.05, "background poorly\nknown under the peak", fontsize=8.5, color=INK, va="top")
    ax.set(yscale="log", xlim=(bins[0], bins[-1]), ylim=(1e-5, 0.1), xlabel=CUT_LABELS["j1_pt"], ylabel="a.u.")
    ax.set_title("Leading-jet momentum")
    ax.legend(loc="lower left")
    fig.suptitle("The problem: a small peak on a large, partly unknown background", x=0.01, ha="left",
                 fontsize=13, color=INK)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"written {out}")


def comparison(all_runs, out, dpi=130):
    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    ymax = 0
    for g, (mode, runs) in enumerate(all_runs.items()):
        claimed = runs["nosys"].final(runs["nosys"].z_claimed)
        without = runs["nosys"].final(runs["nosys"].z)
        with_ = runs["sys"].final(runs["sys"].z)
        ymax = max(ymax, claimed, without, with_)
        for dx, z, color, alpha in [(-0.27, claimed, BLUE, 0.3), (0.0, without, BLUE, 1.0), (0.27, with_, TOMATO, 1.0)]:
            ax.bar(g + dx, z, 0.25, color=color, alpha=alpha)
            ax.text(g + dx, z, f"{z:.1f}", ha="center", va="bottom", fontsize=9.5,
                    fontweight="bold" if alpha == 1 else None)
    handles = [
        Patch(color=BLUE, alpha=0.3, label="without uncertainties: claimed"),
        Patch(color=BLUE, label="without uncertainties: delivered"),
        Patch(color=TOMATO, label="with uncertainties: delivered"),
    ]
    ax.set_xticks(range(len(all_runs)), [TITLES[m] for m in all_runs])
    ax.set(ylabel="discovery significance Z  [σ]", ylim=(0, 1.35 * ymax))
    ax.set_title("Ignoring uncertainties overpromises")
    ax.legend(handles=handles, loc="upper right")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"written {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mass", nargs=2, metavar=("SYS", "NOSYS"))
    parser.add_argument("--nn", nargs=2, metavar=("SYS", "NOSYS"))
    parser.add_argument("--ntuples", default="tests/toy_files")
    parser.add_argument("--out", default="assets")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    events = load_events(args.ntuples)
    inputs(events, os.path.join(args.out, "inputs.png"))
    all_runs = {}
    for mode in ("mass", "nn"):
        paths = getattr(args, mode)
        if not paths:
            continue
        all_runs[mode] = {"sys": Run(paths[0]), "nosys": Run(paths[1])}
        if not args.no_gif:
            movie(all_runs[mode], mode, os.path.join(args.out, f"{mode}.gif"), events)
    if all_runs:
        comparison(all_runs, os.path.join(args.out, "comparison.png"))
