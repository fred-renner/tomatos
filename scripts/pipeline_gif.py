#!/usr/bin/env python3
"""Hero gifs: the analysis as a chain of boxes, optimized two ways.

top     Manual Optimization: one box after the other, each on S/sqrt(B) or a
        classifier loss (scripts/stepwise.py), the CLs is only seen at the end.
        The middle box first shows the fine distribution while the box before
        it is tuned, then the fine bars flow together into the binned
        histogram whose edges get optimized
bottom  Automated Optimization: all boxes at once on the expected CLs with
        all uncertainties, from a tomatos training run

    python scripts/pipeline_gif.py --run results/tomatos_cls_var_..._mass \
        --stepwise results/stepwise/mass.npz --out assets/pipeline_mass.gif
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "tests"))
import generate_toy_files as toy  # noqa: E402

# own palette: tomato signal on a blue background, what we don't know is a
# grey layer on top, leaf green for the automated part
SIG, BKG, UNC, AUTO = "#e5533d", "#6f9bc4", "#3d3d3d", "#4c9a5b"
# the smooth background the optimizer sees, darker than the fill
KDE = "#1f4e79"
INK, MUTED, PANEL, EDGE = "#2b2b2b", "#8a8f98", "#fbfaf8", "#d9d6d0"
TOP_TITLE = "Manual Optimization"
BOTTOM_TITLE = "Automated Optimization"
BKG_DENSITY = LinearSegmentedColormap.from_list("bkg", ["#e9f0f7", "#2f5f8a"])
M_RANGE, PT_RANGE = (60, 200), (30, 200)
# fine binning before the fit binning, aligned with the equal start bins
FINE = {"mass": np.linspace(*M_RANGE, 71), "nn": np.linspace(0, 1, 41)}
# manual row: share of frames per phase, the test phase gets the rest
PHASES = {"first": 0.35, "hold": 0.05, "morph": 0.15, "bins": 0.30}
# the signal outline is drawn scaled up, the stack keeps the true size
SIG_SCALE = 10
# densities per GeV of mass, per 0.01 of nn score: similar numbers for both
UNIT = {True: 1.0, False: 0.01}
# automated row: the training is converged well before its end, show this
# share of the steps, played in this share of the frames, then rest
AUTO_STEPS, AUTO_PLAY = 0.75, 2 / 3
YLABEL = {True: "events / GeV", False: "events / 0.01"}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.labelsize": 7.5,
        "hatch.color": UNC,
        "hatch.linewidth": 0.6,
    }
)


class Run:
    """What the gif needs from one training, loaded once."""

    def __init__(self, path):
        with open(os.path.join(path, "train_config.json")) as f:
            self.config = json.load(f)
        with open(os.path.join(path, "preprocess_md.json")) as f:
            self.md = json.load(f)
        with h5py.File(os.path.join(path, "metrics.h5"), "r") as m:
            self.m = {k: m[k][:] for k in m}
        # step 0 holds the solver init
        self.cls = self.m["test_loss"]
        self.steps = len(self.cls)
        self.modulo = self.config["movie_batch_modulo"]
        self.is_var = self.config["objective"] == "cls_var"

    def to_x(self, u):
        # kde grid and bins live in the min max scaled [0, 1]
        if self.is_var:
            i = self.config["cls_var_idx"]
            return (np.asarray(u) - self.md["scaler_min"][i]) / self.md["scaler_scale"][i]
        return np.asarray(u)

    def edges(self, step):
        # bins are logged in physical units for cls_var, see train_utils.log_bins
        return np.asarray(self.m["bins"][step])

    def kde(self, step, key):
        # kde per unit x of the closest logged frame, scaled like
        # train_utils.rescale_kde: matched to the largest train bin, so
        # divide by that bin's width
        width = np.diff(self.edges(step))
        k = step - step % self.modulo
        return self.m["kde_" + key][k] / width[np.argmax(self.m["h_" + key][k])]


def load_events(ntuples):
    events = {}
    for sample in ("sig", "bkg"):
        with uproot.open(os.path.join(ntuples, sample, "NOSYS.root")) as f:
            events[sample] = f["FilteredTree"].arrays(library="np")
    return events


def resample(n_src, n):
    # n frame indices spread over n_src source frames
    return np.round(np.linspace(0, n_src - 1, n)).astype(int)


def smooth(t):
    # ease in and out
    return t * t * (3 - 2 * t)


def fmt(cls):
    return f"{cls:.2g}"


def mirror(src):
    # cls with fixed, equally spaced bins does not change under
    # score -> 1 - score, the training picks either orientation. shown the
    # usual way round, signal at 1
    src.binned = [(1 - e[::-1], b[::-1], s[::-1], d[::-1]) for e, b, s, d in src.binned]
    src.edges = [x[0] for x in src.binned]
    src.nn_map = [1 - np.asarray(x) for x in src.nn_map]
    src.kde = [(1 - x, ks, kb) for x, ks, kb in src.kde]


class Source:
    """One row's per frame state."""

    def __init__(self, n):
        self.n = n


def from_run(run, n):
    src = Source(n)
    last = int(AUTO_STEPS * (run.steps - 1))
    n_play = int(AUTO_PLAY * n)
    steps = np.concatenate([resample(last + 1, n_play), np.full(n - n_play, last)])
    m = run.m
    ups = [k for k in m if k.startswith("h_bkg_BKG_") and k.endswith("_1UP_test")]
    src.binned = []
    for s in steps:
        e = np.asarray(run.edges(s), dtype=float)
        b, sig = m["h_bkg_NOSYS_test"][s], m["h_sig_NOSYS_test"][s]
        d = np.sqrt(sum((m[k][s] - b) ** 2 for k in ups))
        w = np.diff(e) / UNIT[run.is_var]
        src.binned.append((e, b / w, sig / w, d / w))
    src.edges = [x[0] for x in src.binned]
    src.cut = [m["cut_pt1"][s] for s in steps] if "cut_pt1" in m else None
    # nn map is logged on movie frames only
    frames = [s - s % run.modulo for s in steps]
    src.nn_map = [m["nn_map"][f] for f in frames] if "nn_map" in m else None
    # smooth (kde) histograms the optimizer sees, per unit x like the counts
    grid = run.to_x(run.config["kde_bins"][:-1])
    src.kde = [
        (grid, *[UNIT[run.is_var] * run.kde(s, k) for k in ("sig_NOSYS", "bkg_NOSYS")]) for s in steps
    ]
    src.mode = ["bins"] * n
    src.edge_alpha = np.ones(n)
    src.flow = np.ones(n)
    # a test fit can fail on a single step, carry the value before it on
    cls = np.asarray(run.cls, dtype=float).copy()
    for i in np.where(~np.isfinite(cls))[0]:
        cls[i] = cls[i - 1]
    src.cls = cls
    src.cls_steps = steps
    src.last = last
    src.progress = steps / last
    if not run.is_var:
        _, b, sig, _ = src.binned[-1]
        x = np.arange(len(b))
        if np.average(x, weights=sig) < np.average(x, weights=b):
            mirror(src)
    return src


def stepwise_phases(sw, is_var, n):
    """Per frame: the box being tuned, the stepwise frame shown, the frame
    of the first box shown and what the middle box draws."""
    stage = sw["frame_stage"]
    idx_first = np.where(stage == ("cut" if is_var else "nn"))[0]
    idx_bins = np.where(stage == "bins")[0]
    n1, n2, n3, n4 = (int(PHASES[p] * n) for p in ("first", "hold", "morph", "bins"))
    n5 = n - n1 - n2 - n3 - n4
    active = np.array([0] * (n1 + n2) + [1] * (n3 + n4) + [2] * n5)
    shown = np.array(
        [idx_bins[0]] * (n1 + n2 + n3)
        + list(idx_bins[resample(len(idx_bins), n4)])
        + [idx_bins[-1]] * n5
    )
    first = np.array(list(idx_first[resample(len(idx_first), n1)]) + [idx_first[-1]] * (n - n1))
    mode = ["fine"] * (n1 + n2) + ["morph"] * n3 + ["bins"] * (n4 + n5)
    # morph: edges fade in first, then the fine bars flow to the bin heights
    t = np.linspace(0, 1, n3 + 2)[1:-1]
    edge_alpha = np.concatenate([np.zeros(n1 + n2), np.clip(t / 0.3, 0, 1), np.ones(n4 + n5)])
    flow = np.concatenate([np.zeros(n1 + n2), smooth(np.clip((t - 0.3) / 0.7, 0, 1)), np.ones(n4 + n5)])
    return active, shown, first, mode, edge_alpha, flow


class Pipeline:
    def __init__(self, run, sw, events, n=120, hold=24):
        self.run, self.is_var = run, run.is_var
        self.events = events
        self.sw = sw
        self.n, self.hold = n, hold
        self.fine_edges = FINE["mass" if self.is_var else "nn"]
        self.top = self.from_stepwise(sw, n)
        self.bottom = from_run(run, n)
        self.md = run.md
        self.vars = run.config["vars"]
        self.first_name = "Event selection" if self.is_var else "Neural network"
        self.fine_name = "Mass" if self.is_var else "NN score"
        self.xlabel = "mass [GeV]" if self.is_var else "NN score"
        rng = np.random.default_rng(1)
        self.sig_points = rng.choice(len(events["sig"]["m"]), 220, replace=False)
        self.ylims()
        self.layout()

    # ---------------------------------------------------------------- counts
    def counts(self, edges, cut=None, frame=None):
        """Background, signal and the +-1 sigma background uncertainty per
        unit x, histogrammed from events: the toy ntuples above the cut for
        the mass, the test events at one nn training frame for the score.
        Fine and binned come from the same events, so the morph matches."""
        if self.is_var:
            ev = self.events
            sel = {k: ev[k]["pt1"] > cut for k in ("sig", "bkg")}
            x = {k: ev[k]["m"][sel[k]] for k in sel}
            w = {k: ev[k]["weight"][sel[k]] for k in sel}
            mods = [toy.bkg_unc(x["bkg"], ev["bkg"]["pt1"][sel["bkg"]])]
        else:
            sw = self.sw
            x = {k: sw[f"frame_score_{k}"][frame] for k in ("sig", "bkg")}
            w = {k: sw[f"event_weight_{k}"] for k in ("sig", "bkg")}
            mods = sw["event_mod_bkg"]
        b = np.histogram(x["bkg"], edges, weights=w["bkg"])[0]
        s = np.histogram(x["sig"], edges, weights=w["sig"])[0]
        d = np.sqrt(sum(np.histogram(x["bkg"], edges, weights=w["bkg"] * mod)[0] ** 2 for mod in mods))
        width = np.diff(edges) / UNIT[self.is_var]
        return b / width, s / width, d / width

    def from_stepwise(self, sw, n):
        src = Source(n)
        src.active, shown, first, src.mode, src.edge_alpha, src.flow = stepwise_phases(sw, self.is_var, n)
        src.edges = [np.asarray(sw["frame_edges"][i], dtype=float) for i in shown]
        src.cut = [sw["frame_cut"][i] for i in first] if "frame_cut" in sw else None
        src.nn_map = [sw["frame_nn_map"][i] for i in first] if "frame_nn_map" in sw else None
        # the fine and binned counts for every frame
        src.fine, src.binned = [], []
        for k in range(n):
            key = dict(cut=src.cut[k] if src.cut else None, frame=first[k])
            src.fine.append((self.fine_edges, *self.counts(self.fine_edges, **key)))
            src.binned.append((src.edges[k], *self.counts(src.edges[k], **key)))
        src.final_cls = float(sw["stage_cls"][-1])
        src.bce = [sw["frame_bce"][i] for i in first] if "frame_bce" in sw else None
        src.scan = (sw["cut_grid_pt1"], sw["cut_proxy_pt1"]) if "cut_grid_pt1" in sw else None
        src.kde = None
        return src

    def ylims(self):
        # per frame, so signal and band stay readable however hard the cut.
        # manual row: the fine view follows its distribution, zooms over to
        # the histogram while the bars flow together, which then stays fixed
        # headroom on the log axis for the labels, the bottom just below the
        # lowest band: on log the band height only depends on the decades
        # shown
        # the nn piles the background up at zero, that pile may run off the
        # top so the bins the signal lives in get the decades
        def shown(e, *hists):
            keep = 0.5 * (e[1:] + e[:-1]) >= (0 if self.is_var else 0.2)
            return [h[keep] for h in hists]

        def top_of(e, b, s, d):
            b, s, d = shown(e, b, s, d)
            return 8 * max(np.max(b + d), SIG_SCALE * np.max(s))

        def lo_of(e, b, s, d):
            low = np.min(np.maximum(b - d, 1e-9))
            return max(0.4 * low, top_of(e, b, s, d) / 1e4)

        src = self.top
        # the scan tries odd bins on the way, the final ones set the range
        final = src.binned[self.n - 1]
        top, lo = top_of(*final), lo_of(*final)
        src.ytop = [top_of(*src.fine[k]) ** (1 - f) * top**f for k, f in enumerate(src.flow)]
        src.ylo = [lo_of(*src.fine[k]) ** (1 - f) * lo**f for k, f in enumerate(src.flow)]
        self.bottom.ytop = [top_of(*x) for x in self.bottom.binned]
        self.bottom.ylo = [lo_of(*x) for x in self.bottom.binned]

    # ---------------------------------------------------------------- layout
    def layout(self):
        self.fig = plt.figure(figsize=(8.8, 6.6), dpi=90)
        self.fig.patch.set_facecolor("white")
        widths, gap, x0, h = (0.30, 0.30, 0.20), 0.035, 0.03, 0.32
        self.rows = {}
        self.scan_axes = {}
        for row, (y0, title, color) in enumerate(
            ((0.55, TOP_TITLE, INK), (0.08, BOTTOM_TITLE, AUTO))
        ):
            self.fig.text(x0, y0 + h + 0.03, title, fontsize=13, fontweight="bold", color=color)
            boxes, axes, titles, labels = [], [], [], []
            bx = x0
            for j, w in enumerate(widths):
                box = FancyBboxPatch(
                    (bx, y0), w, h, boxstyle="round,pad=0.004,rounding_size=0.018",
                    transform=self.fig.transFigure, facecolor=PANEL, edgecolor=EDGE, lw=1.5,
                    zorder=-1,
                )
                self.fig.patches.append(box)
                name = [self.first_name, "Histogram", "Hypothesis test"][j]
                titles.append(self.fig.text(bx + 0.012, y0 + h - 0.035, name, fontsize=10,
                                            fontweight="bold", color=INK))
                # the proxy the first manual box is tuned on, under the box
                labels.append(self.fig.text(bx + w / 2, y0 - 0.03, "", fontsize=8.5,
                                            color=MUTED, ha="center"))
                if j < 2:
                    # the first box of the mass pipeline gets a narrow side
                    # panel for the cut scan
                    side = 0.05 if (j == 0 and self.is_var) else 0
                    ax = self.fig.add_axes([bx + 0.06, y0 + 0.06, w - 0.075 - side, h - 0.12])
                    if side:
                        scan_ax = self.fig.add_axes([bx + w - 0.01 - side, y0 + 0.06, side - 0.008, h - 0.12])
                        scan_ax.set_facecolor("none")
                        self.scan_axes[row] = scan_ax
                else:
                    # a number, or the loss curve
                    ax = self.fig.add_axes([bx + 0.03, y0 + 0.06, w - 0.06, h - 0.13])
                ax.set_facecolor("none")
                boxes.append(box)
                axes.append(ax)
                if j < len(widths) - 1:
                    arrow = FancyArrowPatch(
                        (bx + w + 0.005, y0 + h / 2), (bx + w + gap - 0.005, y0 + h / 2),
                        transform=self.fig.transFigure, arrowstyle="-|>", mutation_scale=13,
                        color=MUTED, lw=1.5,
                    )
                    self.fig.patches.append(arrow)
                bx += w + gap
            self.rows[row] = (boxes, axes, titles, labels)

    # ---------------------------------------------------------------- panels
    def to_phys(self, var, u):
        i = self.vars.index(var)
        return (np.asarray(u) - self.md["scaler_min"][i]) / self.md["scaler_scale"][i]

    def plane(self, ax, cut=None, nn_map=None, edges=None):
        ev = self.events
        # background as a density, signal as points on top
        ax.hist2d(ev["bkg"]["m"], ev["bkg"]["pt1"], bins=[48, 48], range=[M_RANGE, PT_RANGE],
                  weights=ev["bkg"]["weight"], cmap=BKG_DENSITY, norm=LogNorm(), zorder=1)
        i = self.sig_points
        ax.scatter(ev["sig"]["m"][i], ev["sig"]["pt1"][i], s=7, color=SIG, lw=0.3,
                   edgecolor="white", alpha=0.9, zorder=2)
        ax.text(0.97, 0.95, "signal", transform=ax.transAxes, ha="right", va="top",
                color=SIG, fontsize=7.5, fontweight="bold", zorder=7)
        ax.text(0.97, 0.86, "background", transform=ax.transAxes, ha="right", va="top",
                color="#2f5f8a", fontsize=7.5, fontweight="bold", zorder=7)
        if cut is not None:
            ax.axhspan(PT_RANGE[0], cut, color="white", alpha=0.75, lw=0, zorder=4)
            ax.axhline(cut, color=INK, lw=2, zorder=5)
            ax.text(M_RANGE[0] + 3, cut + 3, f"keep $p_T$ > {cut:.0f}", fontsize=7.5,
                    color=INK, fontweight="bold", zorder=6)
        if nn_map is not None:
            n = int(np.sqrt(len(nn_map)))
            u = np.linspace(0, 1, n)
            gm, gp = np.meshgrid(self.to_phys("m", u), self.to_phys("pt1", u), indexing="ij")
            score = np.asarray(nn_map).reshape(n, n)
            ax.contourf(gm, gp, score, levels=edges, cmap="Greens", alpha=0.3, zorder=3)
            ax.contour(gm, gp, score, levels=edges[1:-1], colors=[AUTO], linewidths=1.0, zorder=3)
        ax.set(xlim=M_RANGE, ylim=PT_RANGE, xlabel="mass [GeV]", ylabel="jet $p_T$ [GeV]")

    def scan(self, ax, src, k):
        # S/sqrt(B) vs the cut, traced while the line sweeps, peak marked once
        # the scan is done
        grid, proxy = src.scan
        # the scan starts at the left edge, only the improvement counts
        ax.set(ylim=PT_RANGE, xlim=(proxy[0], proxy[0] + 1.15 * (np.max(proxy) - proxy[0])), xticks=[])
        ax.tick_params(labelleft=False)
        ax.set_xlabel("S/√B", fontsize=7)
        reach = max(src.cut[: k + 1])
        sel = grid <= reach
        ax.plot(proxy[sel], grid[sel], color=INK, lw=1.6)
        if reach >= grid[-1]:
            # scanned through, back to the maximum
            best = np.argmax(proxy)
            ax.plot(proxy[best], grid[best], "o", color=SIG, ms=5)
        ax.axhline(src.cut[k], color=INK, lw=1, alpha=0.5)

    @staticmethod
    def stack(ax, e, b, s, d, lo):
        # background, its uncertainty as a grey layer around it, and the
        # signal alone as a scaled up outline
        ax.stairs(b, e, fill=True, color=BKG, baseline=lo, lw=0)
        band = np.maximum(b - d, lo)
        ax.stairs(b + d, e, baseline=band, fill=True, facecolor=UNC, alpha=0.3, lw=0)
        ax.stairs(b + d, e, baseline=band, fill=False, hatch="////", edgecolor=UNC, lw=0)
        # its edges, so a narrow band still reads as one
        for edge in (b + d, band):
            ax.stairs(edge, e, baseline=None, color=UNC, lw=0.6, alpha=0.8)
        # down to the axis at both ends, so the outer bins are closed, not
        # clipped there so the outer edges keep the full line width
        ax.stairs(np.maximum(SIG_SCALE * s, lo), e, color=SIG, lw=1.6, baseline=lo, fill=False,
                  clip_on=False)

    def middle(self, ax, src, k):
        # log, the signal sits where the background has fallen off
        lo = src.ylo[k]
        e, *binned = src.binned[k]
        if src.mode[k] == "bins":
            self.stack(ax, e, *binned, lo)
        else:
            # every fine bar flows to the height of the bin it ends up in
            ef, *fine = src.fine[k]
            centres = 0.5 * (ef[1:] + ef[:-1])
            j = np.clip(np.searchsorted(e, centres) - 1, 0, len(e) - 2)
            f = src.flow[k]
            self.stack(ax, ef, *[(1 - f) * a + f * c[j] for a, c in zip(fine, binned)], lo)
        for x in e[1:-1]:
            ax.axvline(x, color=INK, lw=0.9, alpha=0.7 * src.edge_alpha[k])
        labels = [("background", KDE), (f"signal ×{SIG_SCALE}", SIG), ("background ±1σ", UNC)]
        if src.kde is not None:
            x, ks, kb = src.kde[k]
            top = src.ytop[k]
            ax.plot(x, np.clip(kb, lo, top), color=KDE, lw=1.1, ls="--")
            ax.plot(x, np.clip(SIG_SCALE * ks, lo, top), color=SIG, lw=1.1, ls="--")
        for i, (text, color) in enumerate(labels):
            ax.text(0.97, 0.96 - 0.085 * i, text, transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.5, color=color, fontweight="bold")
        xlim = M_RANGE if self.is_var else (0, 1)
        ax.set(xlim=xlim, yscale="log", ylim=(lo, src.ytop[k]), xlabel=self.xlabel,
               ylabel=YLABEL[self.is_var])

    def loss(self, ax, src, k):
        # the cls along the training against the one manual result, the
        # shape matters, no numbers
        cls = np.asarray(src.cls[1 : src.last + 1], dtype=float)
        x = np.linspace(0, 1, len(cls))
        manual = self.top.final_cls
        lo, hi = min(cls.min(), manual), max(cls.max(), manual)
        ax.set(yscale="log", xlim=(0, 1.05), ylim=(lo / 1.6, hi * 1.6), xticks=[], yticks=[])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("optimization →", fontsize=7)
        ax.set_ylabel("CLs", fontsize=7, labelpad=2)
        ax.axhline(manual, color=MUTED, lw=1.4, ls="--")
        ax.text(1.04, manual * 1.15, "manual", ha="right", va="bottom", fontsize=7, color=MUTED,
                fontweight="bold")
        now = x <= src.progress[k] + 1e-9
        ax.plot(x[now], cls[now], color=AUTO, lw=2)
        ax.plot(x[now][-1], cls[now][-1], "o", color=AUTO, ms=5)

    @staticmethod
    def test(ax, value, color):
        # a single number comes out, nothing to plot
        ax.axis("off")
        if value is None:
            ax.text(0.5, 0.5, "?", transform=ax.transAxes, ha="center", va="center", fontsize=40,
                    fontweight="bold", color=MUTED)
            return
        ax.text(0.5, 0.56, fmt(value), transform=ax.transAxes, ha="center", va="center",
                fontsize=22, fontweight="bold", color=color)
        ax.text(0.5, 0.38, "expected CLs", transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color=MUTED)

    # ---------------------------------------------------------------- frames
    def draw_row(self, row, src, k):
        boxes, axes, titles, labels = self.rows[row]
        for ax in axes:
            ax.cla()
        if row in self.scan_axes:
            self.scan_axes[row].cla()
            if row == 0 and src.scan is not None:
                self.scan(self.scan_axes[row], src, k)
            else:
                self.scan_axes[row].axis("off")
        cut = src.cut[k] if src.cut is not None else None
        nn_map = src.nn_map[k] if src.nn_map is not None else None
        self.plane(axes[0], cut=cut, nn_map=nn_map, edges=np.asarray(src.edges[k], dtype=float))
        self.middle(axes[1], src, k)
        if row == 0:
            active = src.active[k]
            # the middle box is the plain distribution until its bins get tuned
            titles[1].set_text(self.fine_name if src.mode[k] == "fine" else "Histogram")
            self.test(axes[2], src.final_cls if active == 2 else None, INK)
            # the proxy each manual box is tuned on
            if self.is_var:
                labels[0].set_text("cut on S/√B")
            else:
                labels[0].set_text(f"classifier, BCE loss {src.bce[k]:.3f}")
            labels[1].set_text("bins on Σ s²/b" if src.mode[k] != "fine" else "")
            for j in (0, 1):
                labels[j].set_color(INK if active == j else MUTED)
            for j, box in enumerate(boxes):
                box.set_edgecolor(INK if j == active else EDGE)
                box.set_linewidth(2.4 if j == active else 1.5)
        else:
            self.loss(axes[2], src, k)
            for box in boxes:
                box.set_edgecolor(AUTO)
                box.set_linewidth(2.4)

    def draw(self, i):
        k = min(i, self.n - 1)
        self.draw_row(0, self.top, k)
        self.draw_row(1, self.bottom, k)

    def save(self, out, fps=10):
        anim = FuncAnimation(self.fig, self.draw, frames=self.n + self.hold)
        anim.save(out, writer=PillowWriter(fps=fps))
        self.draw(self.n + self.hold - 1)
        self.fig.savefig(out.replace(".gif", "_final.png"), dpi=110)
        print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--stepwise", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ntuples", default="tests/toy_files")
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()

    run = Run(args.run)
    sw = dict(np.load(args.stepwise))
    Pipeline(run, sw, load_events(args.ntuples), n=args.frames).save(args.out)
