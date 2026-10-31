#!/usr/bin/env python3
"""Render the training evolution of a finished run as a GIF.

Left: signal and background estimate in the fit region per training step
(binned KDE as stairs, KDE as line). Right: CLs vs. training step.

    python scripts/make_gif.py --config configs/demo_cls_var.yaml --out assets/cls_var.gif
"""
import argparse
import io
import json

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import tomatos.config

SIG_COLOR = "#d62728"
BKG_COLOR = "#1f77b4"


def unscale(x, md, idx):
    return (np.asarray(x) - md["scaler_min"][idx]) / md["scaler_scale"][idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--signal-scale", type=float, default=None)
    gif_args = parser.parse_args()

    args = argparse.Namespace(config=gif_args.config, debug=False, train=False)
    config = tomatos.config.Setup(args)
    with open(config.preprocess_md_file_path) as f:
        md = json.load(f)

    sig = f"{config.signal_sample}_{config.nominal}"
    bkg = f"bkg_estimate_{config.nominal}"
    is_var = config.objective == "cls_var"

    with h5py.File(config.metrics_file_path, "r") as m:
        h_sig, h_bkg = m["h_" + sig][:], m["h_" + bkg][:]
        kde_sig, kde_bkg = m["kde_" + sig][:], m["kde_" + bkg][:]
        bins = m["bins"][:]
        train_cls, test_cls = m["train_loss"][:], m["test_loss"][:]
        cuts = {k: m["cut_" + k][:] for k in config.opt_cuts}

    n_steps = len(train_cls)
    x_kde = np.asarray(config.kde_bins[:-1])
    if is_var:
        # cls_var hists live in the min-max scaled space, show GeV
        x_kde = unscale(x_kde, md, config.cls_var_idx) / 1e3
        bins = bins / 1e3
        x_label = f"{config.cls_var} [GeV]"
        # bin widths vary by orders of magnitude, show densities
        widths = np.diff(bins, axis=1)
        h_sig, h_bkg = h_sig / widths, h_bkg / widths
        y_label = "Events / GeV"
    else:
        x_label = "Neural network score"
        y_label = "Events"

    # normalize the kde line to the area of the drawn histogram
    def match_area(kde, h):
        area_h = np.sum(h * np.diff(bins, axis=1), axis=1)
        area_kde = np.trapz(kde, x_kde, axis=1)
        return kde * (area_h / np.where(area_kde > 0, area_kde, np.inf))[:, None]

    kde_sig, kde_bkg = match_area(kde_sig, h_sig), match_area(kde_bkg, h_bkg)

    # signal is tiny compared to background, scale it up for visibility
    sf = gif_args.signal_scale
    if sf is None:
        sf = 10 ** np.round(np.log10(h_bkg[-1].sum() / h_sig[-1].sum()))
    ymax = 1.15 * max(h_bkg.max(), (sf * h_sig).max())
    ymin = 1e-4 * ymax
    x_range = (bins[:, 0].min(), bins[:, -1].max())
    # sharp test CLs on a small batch is noisy, show a running mean on top
    test_cls = np.where(np.isfinite(test_cls), test_cls, np.nan)
    train_cls = np.where(np.isfinite(train_cls), train_cls, np.nan)
    window = max(1, n_steps // 25)
    test_smooth = np.array(
        [np.nanmean(test_cls[max(0, j - window + 1) : j + 1]) for j in range(n_steps)]
    )
    both = np.r_[train_cls[1:], test_smooth]
    cls_lim = (0.5 * np.nanmin(both), 2 * np.nanpercentile(both, 99))

    frames = []
    steps = np.unique(np.linspace(0, n_steps - 1, gif_args.frames).astype(int))
    for i in steps:
        fig, (ax, ax_cls) = plt.subplots(
            1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.5, 1]}
        )
        ax.stairs(h_bkg[i], bins[i], color=BKG_COLOR, lw=2, label="Background estimate")
        ax.stairs(sf * h_sig[i], bins[i], color=SIG_COLOR, lw=2, label="Signal" if sf == 1 else f"Signal × {sf:g}")
        ax.plot(x_kde, kde_bkg[i], color=BKG_COLOR, lw=1, alpha=0.5)
        ax.plot(x_kde, sf * kde_sig[i], color=SIG_COLOR, lw=1, alpha=0.5)
        for edge in bins[i][1:-1]:
            ax.axvline(edge, color="grey", lw=0.6, ls=":")
        if is_var:
            # bins start out spanning the full mass range
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlim(*x_range)
        ax.set_ylim(ymin if is_var else 0, ymax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        cut_txt = "\n".join(
            f"{k} > {v[i] / 1e3:.1f} GeV" for k, v in cuts.items()
        )
        ax.text(
            0.98, 0.97, f"optimized cuts\n{cut_txt}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, family="monospace",
        )

        ax_cls.plot(train_cls[: i + 1], color="black", lw=1.2, label="train (differentiable)")
        ax_cls.plot(test_cls[: i + 1], color="tab:orange", lw=0.6, alpha=0.35)
        ax_cls.plot(test_smooth[: i + 1], color="tab:orange", lw=1.5, label="test (sharp hists, running mean)")
        ax_cls.scatter([i], [test_smooth[i]], color="tab:orange", zorder=3, s=15)
        ax_cls.set_xlim(0, n_steps)
        ax_cls.set_yscale("log")
        ax_cls.set_ylim(*cls_lim)
        ax_cls.set_xlabel("Training step")
        ax_cls.set_ylabel(r"Expected $CL_s$  (lower is better)")
        ax_cls.legend(loc="upper right", frameon=False, fontsize=9)

        fig.suptitle(f"{config.objective}   step {i:4d}", family="monospace")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

    # hold the last frame
    durations = [80] * (len(frames) - 1) + [2500]
    frames[0].save(
        gif_args.out, save_all=True, append_images=frames[1:],
        duration=durations, loop=0, optimize=True,
    )
    print(f"GIF here: {gif_args.out}")


if __name__ == "__main__":
    main()
