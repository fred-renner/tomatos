#!/usr/bin/env python3
"""README figures for the toy demos, from finished runs.

inputs.png      the problem: a small peak on a large, partly unknown background
mass.gif        mass fit: selected events, CLs of both trainings, the pt1 cut
nn.gif          nn: score distribution, CLs of both trainings
comparison.png  claimed vs delivered CLs, with and without systematics

    python scripts/readme_plots.py \
        --mass results/tomatos_cls_var_..._mass results/tomatos_cls_var_..._mass_nosys \
        --nn results/tomatos_cls_nn_..._nn results/tomatos_cls_nn_..._nn_nosys
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
SIG, BKG_KEY = "sig_NOSYS", "bkg_NOSYS"
LABELS = {"m": "$m$ [GeV]", "pt1": "leading-jet $p_T$ [GeV]"}
EXCLUSION = 0.05

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
        with open(os.path.join(path, "preprocess_md.json")) as f:
            self.md = json.load(f)
        with h5py.File(os.path.join(path, "metrics.h5"), "r") as m:
            self.m = {k: m[k][:] for k in m}
        # step 0 holds the solver init
        self.cls = self.m["test_loss"]
        self.cls_claimed = self.m["test_loss_stat_only"]
        self.steps = len(self.cls)
        self.modulo = self.config["movie_batch_modulo"]
        self.frames = list(range(0, self.steps, self.modulo))
        self.is_var = self.config["objective"] == "cls_var"
        self.width_kde = {}

    def to_x(self, u):
        # kde grid and bins live in the min max scaled [0, 1]
        if self.is_var:
            i = self.config["cls_var_idx"]
            return (np.asarray(u) - self.md["scaler_min"][i]) / self.md["scaler_scale"][i]
        return np.asarray(u)

    def edges(self, step):
        # bins are logged in physical units for cls_var, see train_utils.log_bins
        return np.asarray(self.m["bins"][step])

    def densities(self, step):
        """Test counts and kde curves per unit x, at one step."""
        edges = self.edges(step)
        width = np.diff(edges)
        b = self.m["h_" + BKG_KEY + "_test"][step] / width
        s = self.m["h_" + SIG + "_test"][step] / width
        # kde of the closest logged frame, scaled like train_utils.rescale_kde:
        # matched to the largest train bin, so divide by that bin's width
        k = step - step % self.modulo
        kde = {}
        for key in (BKG_KEY, SIG):
            w = width[np.argmax(self.m["h_" + key][k])]
            kde[key] = self.m["kde_" + key][k] / w
        return edges, b, s, kde

    def ylim(self):
        # fixed over the whole movie, so the plot moves and not the axis
        top = 0
        for step in self.frames:
            _, b, s, kde = self.densities(step)
            top = max(top, np.max(b + s), np.max(kde[BKG_KEY]))
        top *= 3
        return top / 3e4, top

    def final(self, cls, n=20):
        return float(np.median(cls[-n:]))


def selection_panel(ax, run, step, ylim):
    """Counted test events per bin (stacked) and the smooth kde the optimizer sees."""
    edges, b, s, kde = run.densities(step)
    width = np.diff(edges)
    ax.bar(edges[:-1], b, width, align="edge", color=BKG, label="background")
    ax.bar(edges[:-1], s, width, bottom=b, align="edge", color=TOMATO, label="signal")
    grid = run.to_x(run.config["kde_bins"][:-1])
    ax.plot(grid, kde[BKG_KEY], color=INK, lw=1.3, ls="--", label="smooth (KDE) background")
    ax.plot(grid, kde[SIG], color=TOMATO, lw=1.3, ls="--", label="smooth (KDE) signal")
    for e in edges[1:-1]:
        ax.axvline(e, color=INK, lw=0.6, alpha=0.35)
    if run.is_var:
        ax.set(xlim=(edges[0], edges[-1]), xlabel=LABELS["m"], ylabel="events / GeV")
    else:
        ax.set(xlim=(0, 1), xlabel="NN score", ylabel="events / unit score")
    ax.set(yscale="log", ylim=ylim)
    ax.set_title("Selected events (test set)")
    ax.legend(loc="upper right", ncol=2)


def cls_panel(ax, runs, step, ylim):
    for name, run in runs.items():
        label, color = RUNS[name]
        x = np.arange(1, step + 1)
        ax.plot(x, run.cls[1 : step + 1], color=color, lw=2, label=label)
        if step >= 1:
            ax.plot(step, run.cls[step], "o", color=color, ms=4)
    ax.axhline(EXCLUSION, color=INK, lw=0.8, ls="--")
    ax.text(runs["sys"].steps, EXCLUSION, "95% exclusion ", ha="right", va="bottom", fontsize=8.5, color=INK)
    ax.set(xlim=(0, runs["sys"].steps), ylim=ylim, yscale="log", xlabel="training step",
           ylabel="expected $CL_s$ (lower = more sensitive)")
    ax.set_title("Sensitivity (test set, all uncertainties)")
    ax.legend(loc="upper right")


def cut_panel(ax, runs, step, var, events):
    bins = np.linspace(30, 300, 55)
    for sample, color, label in (("bkg", BKG, "background"), ("sig", TOMATO, "signal")):
        x, w = events[sample][var], events[sample]["weight"]
        style = dict(color=color, label=label, density=True)
        style |= dict(alpha=0.8) if sample == "bkg" else dict(histtype="step", lw=2)
        ax.hist(x, bins, weights=w, **style)
    onset = toy.bkg_mod_pars[1]
    ax.axvspan(onset, bins[-1], color=INK, alpha=0.07, lw=0)
    for name, run in runs.items():
        label, color = RUNS[name]
        ax.axvline(run.m["cut_" + var][step], color=color, lw=2.2, label=f"cut, {label}")
    ax.set(yscale="log", xlim=(bins[0], bins[-1]), ylim=(1e-5, 0.1), xlabel=LABELS[var], ylabel="a.u.")
    ax.set_title("Cut on the leading jet (keep events right of it)")
    ax.legend(loc="upper right")


def load_events(ntuples):
    events = {}
    for sample in ("sig", "bkg"):
        with uproot.open(os.path.join(ntuples, sample, "NOSYS.root")) as f:
            events[sample] = f["FilteredTree"].arrays(library="np")
    return events


def cls_ylim(runs):
    lo = min(np.min(r.cls[1:]) for r in runs.values())
    return lo / 2, 1


def movie(runs, mode, out, events, fps=12, hold=30, dpi=80):
    sys_run = runs["sys"]
    cuts = list(sys_run.config["opt_cuts"])
    n_panels = 2 + len(cuts)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.9), layout="constrained")
    frames = sys_run.frames + [sys_run.steps - 1] * hold
    ylim_sel, ylim_cls = sys_run.ylim(), cls_ylim(runs)

    def draw(step):
        for ax in axes:
            ax.clear()
        selection_panel(axes[0], sys_run, step, ylim_sel)
        cls_panel(axes[1], runs, step, ylim_cls)
        for ax, var in zip(axes[2:], cuts):
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

    ax, bins = axes[0], np.linspace(*toy.m_range, 71)
    onset = toy.bkg_mod_pars[1]
    hard = b["pt1"] > onset
    w = b["weight"]
    h = np.histogram(b["m"], bins, weights=w)[0]
    h_hard = np.histogram(b["m"][hard], bins, weights=w[hard])[0]
    unc = np.histogram(b["m"], bins, weights=w * (b["bkg_mod_up"] - 1))[0]
    ax.stairs(h, bins, fill=True, color=BKG, alpha=0.55, label="background")
    ax.stairs(h_hard, bins, fill=True, color=BLUE, alpha=0.8, label=f"… with $p_T$ > {onset:.0f} GeV")
    ax.stairs(h_hard + unc, bins, baseline=np.maximum(h_hard - unc, 0), fill=True, color="none", hatch="////",
              edgecolor=INK, lw=0, label="its uncertainty")
    ax.hist(s["m"], bins, weights=10 * s["weight"], histtype="step", color=TOMATO, lw=2, label="signal ×10")
    ax.set(xlim=(bins[0], bins[-1]), xlabel=LABELS["m"], ylabel="events")
    ax.set_title("Mass spectrum")
    ax.legend(loc="upper right")

    ax, bins = axes[1], np.linspace(30, 300, 55)
    ax.hist(b["pt1"], bins, weights=w, density=True, color=BKG, alpha=0.8, label="background")
    ax.hist(s["pt1"], bins, weights=s["weight"], density=True, histtype="step", color=TOMATO, lw=2, label="signal")
    ax.axvspan(onset, bins[-1], color=INK, alpha=0.07, lw=0)
    ax.text(onset + 4, 0.05, "background poorly\nknown under the peak", fontsize=8.5, color=INK, va="top")
    ax.set(yscale="log", xlim=(bins[0], bins[-1]), ylim=(1e-5, 0.1), xlabel=LABELS["pt1"], ylabel="a.u.")
    ax.set_title("Leading-jet momentum")
    ax.legend(loc="lower left")
    fig.suptitle("The problem: a small peak on a large, partly unknown background", x=0.01, ha="left",
                 fontsize=13, color=INK)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"written {out}")


def comparison(all_runs, out, dpi=130):
    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    values = []
    for g, (mode, runs) in enumerate(all_runs.items()):
        claimed = runs["nosys"].final(runs["nosys"].cls_claimed)
        without = runs["nosys"].final(runs["nosys"].cls)
        with_ = runs["sys"].final(runs["sys"].cls)
        values += [claimed, without, with_]
        for dx, v, color, alpha in [(-0.27, claimed, BLUE, 0.3), (0.0, without, BLUE, 1.0), (0.27, with_, TOMATO, 1.0)]:
            ax.bar(g + dx, v, 0.25, color=color, alpha=alpha)
            ax.text(g + dx, v * 1.15, f"{v:.2g}", ha="center", va="bottom", fontsize=9.5,
                    fontweight="bold" if alpha == 1 else None)
    ax.axhline(EXCLUSION, color=INK, lw=0.8, ls="--")
    handles = [
        Patch(color=BLUE, alpha=0.3, label="without uncertainties: claimed"),
        Patch(color=BLUE, label="without uncertainties: delivered"),
        Patch(color=TOMATO, label="with uncertainties: delivered"),
    ]
    ax.set_xticks(range(len(all_runs)), [TITLES[m] for m in all_runs])
    ax.set(yscale="log", ylim=(min(values) / 5, 1), ylabel="expected $CL_s$ (lower = more sensitive)")
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
