#!/usr/bin/env python

# Quick hyperparameter loop: run configs with overrides one after another
# (memory) and print one summary line per run.
#
#   python scripts/sweep.py configs/toy_mass.yaml configs/toy_mass_nosys.yaml \
#       --set lr=0.002,0.005 penalty_intensity=0.5 --steps 300

import argparse
import itertools
import os
import subprocess
import sys

import h5py
import numpy as np
import yaml

here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.dirname(here)


def parse_sets(sets):
    grid = {}
    for s in sets:
        key, values = s.split("=", 1)
        grid[key] = [yaml.safe_load(v) for v in values.split(",")]
    return grid


def summarize(path, test):
    with h5py.File(os.path.join(path, "metrics.h5"), "r") as m:
        # step 0 is the solver init
        test_loss = m["test_loss"][1:]
        train_loss = m["train_loss"][1:]
        bw = m["bw"][-1]
        cuts = {k: m[k][-1] / 1e3 for k in m if k.startswith("cut_")}
    # loss is -Z for discovery
    sign = -1 if test == "discovery" else 1
    test_metric, train_metric = sign * test_loss, sign * train_loss
    tail = test_metric[-30:]
    # step to step noise of the test metric, relative to its level
    noise = np.median(np.abs(np.diff(test_metric[-100:]))) / max(abs(np.median(tail)), 1e-12)
    out = (
        f"test {np.median(tail):7.3f} (best {test_metric.max() if sign < 0 else test_metric.min():7.3f}) "
        f"train {train_metric[-1]:7.3f}  noise {noise:5.3f}  bw {bw:.4f}"
    )
    out += "".join(f"  {k[4:]} {v:5.1f}" for k, v in cuts.items())
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--set", nargs="*", default=[], help="key=v1,v2 grids")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out", default="results/sweep/")
    args = parser.parse_args()

    grid = parse_sets(args.set)
    combos = list(itertools.product(*grid.values())) or [()]
    lines = []
    for config_path in args.configs:
        with open(config_path) as f:
            base = yaml.safe_load(f)
        for combo in combos:
            yml = dict(base)
            overrides = dict(zip(grid.keys(), combo))
            yml.update(overrides)
            if args.steps:
                yml["num_steps"] = args.steps
            tag = "_".join(f"{k}{v}" for k, v in overrides.items()).replace(" ", "")
            name = os.path.basename(config_path).removesuffix(".yaml") + (f"_{tag}" if tag else "")
            yml["results_path"] = os.path.join(args.out, name) + "/"
            yml["plot_inputs"] = False
            os.makedirs(yml["results_path"], exist_ok=True)
            run_config = os.path.join(yml["results_path"], "config.yaml")
            with open(run_config, "w") as f:
                yaml.safe_dump(yml, f)
            log = os.path.join(yml["results_path"], "out.txt")
            with open(log, "w") as f:
                ret = subprocess.run(
                    [os.path.join(os.path.dirname(sys.executable), "tomatos"),
                     "--config", run_config, "--prep", "--train", "--plot"],
                    cwd=repo, stdout=f, stderr=subprocess.STDOUT,
                ).returncode
            if ret != 0:
                line = f"{name:45s} FAILED, see {log}"
            else:
                run_dir = [d for d in os.listdir(yml["results_path"]) if d.startswith("tomatos_")][0]
                line = f"{name:45s} " + summarize(os.path.join(yml["results_path"], run_dir), yml.get("test", "cls"))
            print(line, flush=True)
            lines.append(line)
    print("\n".join(["", "summary:"] + lines))
