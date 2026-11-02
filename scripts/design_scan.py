#!/usr/bin/env python
"""Expected CLs vs a cut, with and without systematics, before any training.

Sharp histograms through the real pipeline, initial binning. Used to design
the toy: the optimum without systematics should cut much harder than the
one with them.

    python scripts/design_scan.py configs/toy_mass.yaml --var pt1
"""
import argparse
import os
import tempfile

import jax.numpy as jnp
import numpy as np
import yaml

import tomatos.config
import tomatos.pipeline
import tomatos.preprocess
import tomatos.training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--var", default="pt1")
    parser.add_argument("--cuts", default="30,160,27", help="start,stop,n")
    parser.add_argument("--split", default="train")
    parser.add_argument("--set", nargs="*", default=[], help="key=value config overrides")
    args = parser.parse_args()

    with open(args.config) as f:
        yml = yaml.safe_load(f)
    for kv in args.set:
        key, value = kv.split("=", 1)
        yml[key] = yaml.safe_load(value)
    tmp = tempfile.mkdtemp()
    yml["results_path"] = tmp + "/"
    yml["plot_inputs"] = False
    path = os.path.join(tmp, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(yml, f)

    config = tomatos.config.Setup(argparse.Namespace(config=path, debug=False, train=False))
    tomatos.preprocess.run(config)
    _, _, pars, batch, _ = tomatos.training.train_init(config)
    data, scale = next(batch[args.split])

    full = tomatos.pipeline.jit_loss_fn(config, validate_only=True, stat_only=False)
    stat = tomatos.pipeline.jit_loss_fn(config, validate_only=True, stat_only=True)
    idx = config.vars.index(args.var)
    start, stop, n = args.cuts.split(",")
    print(f"{args.var:>8s}  {'CLs full':>10s}  {'CLs stat only':>13s}  {'sig':>7s}  {'bkg':>8s}")
    rows = []
    for cut in np.linspace(float(start), float(stop), int(n)):
        p = {**pars, f"cut_{args.var}": jnp.array(cut * config.scaler_scale[idx] + config.scaler_min[idx])}
        cls_full, hists = full(p, data, scale)
        cls_stat, _ = stat(p, data, scale)
        s = float(jnp.sum(hists[f"{config.signal_sample}_{config.nominal}"]))
        b = float(jnp.sum(hists[f"{config.bkg_sample}_{config.nominal}"]))
        rows.append((cut, float(cls_full), float(cls_stat)))
        print(f"{cut:8.1f}  {float(cls_full):10.4f}  {float(cls_stat):13.2e}  {s:7.1f}  {b:8.1f}")
    rows = np.array(rows)
    print(f"\noptimum full: {rows[np.argmin(rows[:, 1]), 0]:.1f}   "
          f"stat only: {rows[np.argmin(rows[:, 2]), 0]:.1f}")
