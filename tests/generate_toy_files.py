#!/usr/bin/env python

# Toy ntuples, ntuple_path/SAMPLE/SYSTEMATIC.root: a resonance on a falling
# background, designed around one effect. The signal jets are hard, the
# background falls steeply, so S/sqrt(B) says: cut hard. But the
# background uncertainty grows with jet pT and mass, exactly where the hard
# cut goes (bkg_unc), uncorrelated between bins. Knowing about it, cut and
# bins together settle for a looser cut.

import os

import numpy as np
import uproot

here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(here, "toy_files")

m_range = (60.0, 200.0)  # GeV, events outside are dropped
signal_mass = 125.0
yields = {"sig": 70.0, "bkg": 10_000.0}
n_events = {"sig": 20_000, "bkg": 100_000}
# signal pt1 peak and width [GeV]
sig_pt_pars = (100.0, 30.0)
# background uncertainty, relative: at pt1 = 30 GeV and the
# lowest mass, the pt1 scale [GeV] it grows with, and how much larger it
# is at the highest mass
bkg_unc_pars = (0.006, 25.0, 2.0)


def bkg_unc(m, pt1):
    # relative 1 sigma background uncertainty, capped so
    # the down variation stays positive
    rel_30, scale, high_mass = bkg_unc_pars
    in_mass = (m - m_range[0]) / (m_range[1] - m_range[0])
    return np.minimum(rel_30 * np.exp((pt1 - 30) / scale) * (1 + (high_mass - 1) * in_mass), 0.9)


def sample(rng, n, kind):
    # oversample, then keep events inside the mass range
    k = 3 * n
    if kind == "sig":
        ev = {
            "m": rng.normal(signal_mass, 10, k),
            "pt1": rng.normal(*sig_pt_pars, k),
        }
    else:
        ev = {
            "m": m_range[0] + rng.exponential(45, k),
            "pt1": 30 + rng.exponential(30, k),
        }
    keep = (ev["m"] > m_range[0]) & (ev["m"] < m_range[1]) & (ev["pt1"] > 30)
    return {key: v[keep][:n] for key, v in ev.items()}


def to_ntuple(ev, total, is_bkg):
    n = len(ev["m"])
    # only defined for bkg, neutral for signal
    rel = bkg_unc(ev["m"], ev["pt1"]) if is_bkg else np.zeros(n)
    return {
        **ev,
        "weight": np.full(n, total / n),
        "bkg_unc_up": 1 + rel,
        "bkg_unc_down": 1 - rel,
    }


def write(ntup, sample_name, systematic="NOSYS"):
    path = os.path.join(out_dir, sample_name)
    os.makedirs(path, exist_ok=True)
    with uproot.recreate(os.path.join(path, systematic + ".root")) as f:
        f.mktree("FilteredTree", {k: v.dtype for k, v in ntup.items()})
        f["FilteredTree"].extend(ntup)
    print(f"{sample_name}/{systematic}.root: {len(ntup['m'])} events")


def generate():
    rng = np.random.default_rng(0)
    for kind in ["sig", "bkg"]:
        ev = sample(rng, n_events[kind], kind)
        write(to_ntuple(ev, yields[kind], kind == "bkg"), kind)


if __name__ == "__main__":
    generate()
