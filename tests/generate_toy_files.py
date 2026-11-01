#!/usr/bin/env python

# Toy ntuples in the same layout as tests/files, so the full pipeline can be
# run on a problem where we know the answer. A resonance on a falling
# background, ZH(->vvbb)-flavoured, with two catches built in that only an
# optimization which sees the uncertainties can learn to avoid:
#
# * the background 2b fraction rises with j1_pt, the ABCD estimate assumes it
#   does not --> non-closure wherever the selection pushes to hard jets
# * for hard jets the background has a poorly known peaking component right
#   under the signal (bkg_mod), an aggressive pt cut buys separation at the
#   price of an uncertainty that looks exactly like signal

import os

import numpy as np
import uproot

here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(here, "toy_files")

GeV = 1e3  # ntuples are in MeV, like the ATLAS ones
signal_mass = 125.0
m_min = 50.0  # lowest simulated m_bb
yields = {"sig": 100.0, "bkg": 30_000.0}
sig_2b_fraction = 0.7
jes_shift = 0.05  # relative shift of j1_pt, j2_pt in signal
# peaking bkg uncertainty: amplitude, j1_pt onset and width, m_bb width [GeV]
bkg_modelling_pars = (0.3, 90.0, 10.0, 12.0)


def bkg_modelling(m_bb, pt_1):
    # relative 1 sigma variation of the bkg weight, large under the peak and
    # for hard jets
    amplitude, onset, pt_width, m_width = bkg_modelling_pars
    hard = 1 / (1 + np.exp(-(pt_1 - onset) / pt_width))
    return amplitude * hard * np.exp(-0.5 * ((m_bb - signal_mass) / m_width) ** 2)


def bkg_2b_fraction(pt_1):
    return 0.08 + 0.3 / (1 + np.exp(-(pt_1 - 120) / 20))


def signal(rng, n):
    return {
        "m_bb": rng.normal(signal_mass, 12, n),
        "pt_1": 40 + rng.gamma(3.0, 30, n),
        "pt_2": 30 + rng.gamma(2.5, 20, n),
        "dphi": np.pi - np.minimum(np.abs(rng.normal(0, 0.7, n)), np.pi),
        "is_2b": rng.uniform(size=n) < sig_2b_fraction,
    }


def background(rng, n):
    pt_1 = 40 + rng.exponential(35, n)
    return {
        "m_bb": m_min + rng.exponential(50, n),
        "pt_1": pt_1,
        "pt_2": 30 + rng.exponential(20, n),
        "dphi": rng.uniform(0, np.pi, n),
        "is_2b": rng.uniform(size=n) < bkg_2b_fraction(pt_1),
    }


def to_ntuple(ev, weight, pt_sf=1.0, is_bkg=False):
    n = len(ev["m_bb"])
    # only defined for bkg, neutral for signal
    mod = bkg_modelling(ev["m_bb"], ev["pt_1"]) if is_bkg else np.zeros(n)
    return {
        "j1_pt": ev["pt_1"] * pt_sf * GeV,
        "j2_pt": ev["pt_2"] * pt_sf * GeV,
        "h_dphi": ev["dphi"],
        "h_m": ev["m_bb"] * GeV,
        "weight": np.full(n, weight),
        "bool_btag_1": (~ev["is_2b"]).astype(np.int32),
        "bool_btag_2": ev["is_2b"].astype(np.int32),
        # no signal sf uncertainty in the toy, keep the branches for the
        # pipeline
        "my_sf_unc_up": np.ones(n),
        "my_sf_unc_down": np.ones(n),
        "bkg_mod_up": 1 + mod,
        "bkg_mod_down": 1 - mod,
    }


def write(ntup, sample, systematic):
    path = os.path.join(out_dir, sample)
    os.makedirs(path, exist_ok=True)
    with uproot.recreate(os.path.join(path, systematic + ".root")) as f:
        f.mktree("FilteredTree", {k: v.dtype for k, v in ntup.items()})
        f["FilteredTree"].extend(ntup)
    print(f"{sample}/{systematic}.root: {len(ntup['h_m'])} events")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # bkg is the largest sample, so it is never upsampled (see preprocess),
    # more events than the yield for enough statistics in the test split
    n_bkg = 100_000
    n_sig = 20_000
    sig, bkg = signal(rng, n_sig), background(rng, n_bkg)

    write(to_ntuple(bkg, yields["bkg"] / n_bkg, is_bkg=True), "bkg", "NOSYS")
    w_sig = yields["sig"] / n_sig
    write(to_ntuple(sig, w_sig), "ggZH125_vvbb", "NOSYS")
    write(to_ntuple(sig, w_sig, pt_sf=1 + jes_shift), "ggZH125_vvbb", "JET_PT_1UP")
    write(to_ntuple(sig, w_sig, pt_sf=1 - jes_shift), "ggZH125_vvbb", "JET_PT_1DOWN")
