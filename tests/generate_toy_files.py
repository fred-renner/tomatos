#!/usr/bin/env python

# Toy ntuples, ntuple_path/SAMPLE/SYSTEMATIC.root: a resonance on a falling
# background, designed around one effect. Signal has harder jets than
# background, so a tight cut on the leading jet pt1 looks like the way to
# go. But for hard jets the background is poorly known right under the
# peak (bkg_mod), a peaking uncertainty that looks exactly like signal.
# Without uncertainties in the objective the optimization cuts hard, with
# them it learns to back off.

import os

import numpy as np
import uproot

here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(here, "toy_files")

m_range = (60.0, 200.0)  # GeV, events outside are dropped
signal_mass = 125.0
yields = {"sig": 100.0, "bkg": 10_000.0}
n_events = {"sig": 20_000, "bkg": 100_000}
# peaking bkg uncertainty: amplitude, pt1 onset and width, m width [GeV]
bkg_mod_pars = (0.6, 140.0, 10.0, 15.0)


def bkg_mod(m, pt1):
    # relative 1 sigma variation of the bkg weight, large under the peak
    # and for hard jets
    amplitude, onset, pt_width, m_width = bkg_mod_pars
    hard = 1 / (1 + np.exp(-(pt1 - onset) / pt_width))
    return amplitude * hard * np.exp(-0.5 * ((m - signal_mass) / m_width) ** 2)


def sample(rng, n, kind):
    # oversample, then keep events inside the mass range
    k = 2 * n
    if kind == "sig":
        ev = {
            "m": rng.normal(signal_mass, 10, k),
            "pt1": 30 + rng.gamma(3.0, 30, k),
            "dphi": np.pi - np.minimum(np.abs(rng.normal(0, 0.8, k)), np.pi),
        }
    else:
        ev = {
            "m": m_range[0] + rng.exponential(45, k),
            "pt1": 30 + rng.exponential(35, k),
            "dphi": rng.uniform(0, np.pi, k),
        }
    keep = (ev["m"] > m_range[0]) & (ev["m"] < m_range[1])
    return {key: v[keep][:n] for key, v in ev.items()}


def to_ntuple(ev, weight, is_bkg):
    n = len(ev["m"])
    # only defined for bkg, neutral for signal
    mod = bkg_mod(ev["m"], ev["pt1"]) if is_bkg else np.zeros(n)
    return {
        **ev,
        "weight": np.full(n, weight),
        "bkg_mod_up": 1 + mod,
        "bkg_mod_down": 1 - mod,
    }


def write(ntup, sample_name, systematic="NOSYS"):
    path = os.path.join(out_dir, sample_name)
    os.makedirs(path, exist_ok=True)
    with uproot.recreate(os.path.join(path, systematic + ".root")) as f:
        f.mktree("FilteredTree", {k: v.dtype for k, v in ntup.items()})
        f["FilteredTree"].extend(ntup)
    print(f"{sample_name}/{systematic}.root: {len(ntup['m'])} events")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for kind in ["sig", "bkg"]:
        ev = sample(rng, n_events[kind], kind)
        write(to_ntuple(ev, yields[kind] / n_events[kind], kind == "bkg"), kind)
