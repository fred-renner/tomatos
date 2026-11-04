#!/usr/bin/env python
"""The manual, stepwise way: optimize one box after the other, each on its
own simple proxy that knows nothing about uncertainties, and only look at
the expected CLs at the very end.

    cut   maximize S/sqrt(B) of the selected events
    nn    train a classifier (binary cross entropy)
    bins  maximize sum s^2/b over the bins (the significance with
          statistics only), from the right: one edge after the other, a
          few sweeps

Everything runs through the real pipeline (sharp cuts and histograms), the
final CLs is the same full model (all uncertainties, test set) the
automated training reports. Frames for the pipeline gif and the CLs after
every stage go to results/stepwise/<suffix>.npz

    python scripts/stepwise.py configs/toy_mass.yaml
    python scripts/stepwise.py configs/toy_nn.yaml
"""
import argparse
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

import tomatos.config
import tomatos.histograms
import tomatos.pipeline
import tomatos.preprocess
import tomatos.train_utils
import tomatos.training

HIST_KEYS = ["sig_NOSYS", "bkg_NOSYS"]


def bkg_unc(h):
    # +-1 sigma background modelling uncertainty per bin, the independent
    # components added in quadrature
    ups = [v for k, v in h.items() if k.startswith("bkg_BKG_") and k.endswith("_1UP")]
    return np.sqrt(sum((u - h["bkg_NOSYS"]) ** 2 for u in ups))


def setup(config_path):
    with open(config_path) as f:
        yml = yaml.safe_load(f)
    tmp = tempfile.mkdtemp()
    yml["results_path"] = tmp + "/"
    yml["plot_inputs"] = False
    path = os.path.join(tmp, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(yml, f)
    config = tomatos.config.Setup(argparse.Namespace(config=path, debug=False, train=False))
    tomatos.preprocess.run(config)
    _, _, pars, batch, _ = tomatos.training.train_init(config)
    return config, pars, batch


class Stepwise:
    def __init__(self, config, pars, batch):
        self.config, self.pars = config, dict(pars)
        self.train = next(batch["train"])
        self.test = next(batch["test"])
        self.frames = []
        self.stages = []
        self.compile()

    def compile(self):
        # the config is closed over, recompile when it changes
        self.hists_fn = tomatos.pipeline.jit_make_hists(self.config, validate_only=True)
        self.cls_fn = tomatos.pipeline.jit_loss_fn(self.config, validate_only=True, stat_only=False)
        # what the analysis would claim without the uncertainties
        self.cls_stat_fn = tomatos.pipeline.jit_loss_fn(self.config, validate_only=True, stat_only=True)

    def hists(self, pars, split="train"):
        data, scale = self.train if split == "train" else self.test
        return {k: np.asarray(v) for k, v in self.hists_fn(pars, data, scale).items()}

    def record(self, stage, pars, **extra):
        # what the gif shows per frame: test histograms at these parameters
        h = self.hists(pars, "test")
        if "edges" not in extra:
            inner = pars["bins"] if self.config.include_bins else self.config.bins[1:-1]
            extra["edges"] = self.physical_edges(inner)
        self.frames.append({"stage": stage, **{k: h[k] for k in HIST_KEYS}, "bkg_unc": bkg_unc(h), **extra})

    def finish_stage(self, name):
        cls = float(self.cls_fn(self.pars, *self.test)[0])
        claimed = float(self.cls_stat_fn(self.pars, *self.test)[0])
        self.stages.append((name, cls, claimed))
        print(f"after {name:5s}: expected CLs (test) {cls:.4f}, claimed without uncertainties {claimed:.2e}")

    def physical_cut(self, var, u):
        i = self.config.vars.index(var)
        return (u - self.config.scaler_min[i]) / self.config.scaler_scale[i]

    def scaled_cut(self, var, x):
        i = self.config.vars.index(var)
        return x * self.config.scaler_scale[i] + self.config.scaler_min[i]

    def physical_edges(self, inner):
        edges = np.array([0, *np.asarray(inner), 1])
        if self.config.objective == "cls_var":
            return self.physical_cut(self.config.cls_var, edges)
        return edges

    def opt_cut(self, var, grid):
        # S/sqrt(B) counting, the line sweeps the scan
        proxy = []
        for x in grid:
            pars = {**self.pars, f"cut_{var}": jnp.array(self.scaled_cut(var, x))}
            h = self.hists(pars)
            s, b = h["sig_NOSYS"].sum(), h["bkg_NOSYS"].sum()
            proxy.append(s / np.sqrt(b))
            self.record("cut", pars, cut=x)
        best = grid[int(np.argmax(proxy))]
        # back to the maximum and pick it
        for x in np.linspace(grid[-1], best, 10)[1:]:
            self.record("cut", {**self.pars, f"cut_{var}": jnp.array(self.scaled_cut(var, x))}, cut=x)
        self.pars[f"cut_{var}"] = jnp.array(self.scaled_cut(var, best))
        print(f"cut {var} > {best:.1f} (S/sqrt(B) {max(proxy):.2f})")
        return np.array(proxy), best

    def opt_bins(self, n_try=100, every=3, passes=3):
        # bins on sum s^2/b, every bin needs at least min_bkg background to
        # be a sane histogram
        if not self.config.include_bins:
            self.config.include_bins = True
            self.pars["bins"] = jnp.array(self.config.bins[1:-1])
            self.compile()

        def counts(inner):
            h = self.hists({**self.pars, "bins": jnp.array(inner)})
            return h["sig_NOSYS"], h["bkg_NOSYS"]

        def record(inner):
            self.record("bins", {**self.pars, "bins": jnp.array(inner)}, edges=self.physical_edges(inner))

        # start from equally spaced bins, shown for a moment
        inner = np.linspace(0, 1, len(self.pars["bins"]) + 2)[1:-1]
        for _ in range(4):
            record(inner)
        # from the right, each edge where the sum s^2/b over all bins peaks,
        # the edges below it equally spaced until their turn
        n = len(inner)
        upper = 1.0
        for k in reversed(range(n)):
            best, best_x = -np.inf, None
            grid = np.linspace(upper, 0, n_try + 1)[1:-1]
            for j, x in enumerate(grid):
                # provisional edges below x, only the bin [x, upper] counts
                trial = np.concatenate([np.linspace(0, x, k + 2)[1:-1], [x], inner[k + 1 :]])
                s, b = counts(trial)
                if np.all(b >= self.config.min_bkg):
                    value = np.sum(s**2 / b)
                    if value > best:
                        best, best_x = value, x
                if j % every == 0:
                    record(trial)
            inner = np.concatenate([np.linspace(0, best_x, k + 2)[1:-1], [best_x], inner[k + 1 :]])
            for _ in range(3):
                record(inner)
            upper = best_x
        # one sweep is greedy, sweep again with the other edges in place,
        # each edge between its neighbours
        for _ in range(passes - 1):
            for k in reversed(range(n)):
                lower = inner[k - 1] if k > 0 else 0.0
                upper = inner[k + 1] if k < n - 1 else 1.0
                best, best_x = -np.inf, inner[k]
                for j, x in enumerate(np.linspace(upper, lower, n_try // 2 + 1)[1:-1]):
                    trial = np.concatenate([inner[:k], [x], inner[k + 1 :]])
                    s, b = counts(trial)
                    if np.all(b >= self.config.min_bkg) and np.sum(s**2 / b) > best:
                        best, best_x = np.sum(s**2 / b), x
                    if j % (2 * every) == 0:
                        record(trial)
                inner = np.concatenate([inner[:k], [best_x], inner[k + 1 :]])
                record(inner)
        self.pars["bins"] = jnp.array(inner)
        s, b = counts(inner)
        print(f"bins {np.round(self.physical_edges(inner), 3)} (bkg {np.round(b, 0)})")

    def opt_nn(self, steps=300, lr=1e-3, every=10):
        # classifier on the nominal samples, classes weighted to equal total
        config = self.config
        data, scale = self.train
        i_sig = config.sample_sys.index(f"{config.signal_sample}_{config.nominal}")
        i_bkg = config.sample_sys.index(f"{config.bkg_sample}_{config.nominal}")
        w = data[:, :, config.weight_idx] * scale[:, None]
        w_sig, w_bkg = w[i_sig] / w[i_sig].sum(), w[i_bkg] / w[i_bkg].sum()

        def bce(nn_pars):
            out = tomatos.histograms.get_nn_output(
                {"nn": nn_pars}, data, config.nn_arch, config.nn_inputs_idx_end
            )
            out = jnp.clip(out, 1e-6, 1 - 1e-6)
            return -(jnp.sum(w_sig * jnp.log(out[i_sig])) + jnp.sum(w_bkg * jnp.log(1 - out[i_bkg]))) / 2

        opt = optax.adam(lr)
        nn_pars = self.pars["nn"]
        opt_state = opt.init(nn_pars)

        @jax.jit
        def step(nn_pars, opt_state):
            loss, grads = jax.value_and_grad(bce)(nn_pars)
            updates, opt_state = opt.update(grads, opt_state)
            return optax.apply_updates(nn_pars, updates), opt_state, loss

        losses = []
        for i in range(steps + 1):
            if i % every == 0:
                self.pars["nn"] = nn_pars
                metrics = {}
                tomatos.train_utils.log_nn_map(config, metrics, self.pars)
                loss = float(bce(nn_pars)) if not losses else losses[-1]
                self.record("nn", self.pars, nn_map=np.asarray(metrics["nn_map"]), bce=loss, **self.scores())
            if i < steps:
                nn_pars, opt_state, loss = step(nn_pars, opt_state)
                losses.append(float(loss))
        self.pars["nn"] = nn_pars
        print(f"classifier bce {losses[0]:.3f} -> {losses[-1]:.3f}")
        return np.array(losses)

    def scores(self):
        # nn score of every test event, the gif histograms them any way it
        # likes. weights and uncertainties per event are fixed, see events()
        config = self.config
        data, _ = self.test
        out = np.asarray(tomatos.histograms.get_nn_output(
            self.pars, data, config.nn_arch, config.nn_inputs_idx_end
        ))
        return {
            f"score_{key}": out[config.sample_sys.index(f"{sample}_{config.nominal}")]
            for key, sample in (("sig", config.signal_sample), ("bkg", config.bkg_sample))
        }

    def events(self):
        # per test event weight (with the split scale) and the relative bkg
        # modelling variation per component
        config = self.config
        data, scale = self.test
        res = {}
        for key, sample in (("sig", config.signal_sample), ("bkg", config.bkg_sample)):
            i = config.sample_sys.index(f"{sample}_{config.nominal}")
            res[f"event_weight_{key}"] = np.asarray(data[i, :, config.weight_idx]) * float(scale[i])
        res["event_mod_bkg"] = np.stack([
            np.asarray(data[i, :, config.vars.index(d["up"])]) - 1 for d in config.weight_systematics.values()
        ])
        return res

    def save(self, out, **extra):
        stacked = {}
        keys = sorted({k for f in self.frames for k in f})
        for k in keys:
            if k == "stage":
                stacked["frame_stage"] = np.array([f["stage"] for f in self.frames])
                continue
            # fill frames that lack a key with the last known value
            last, col = None, []
            for f in self.frames:
                last = f.get(k, last)
                col.append(last)
            first = next(c for c in col if c is not None)
            stacked["frame_" + k] = np.array([first if c is None else c for c in col], dtype=float)
        np.savez(
            out,
            stage_names=np.array([s[0] for s in self.stages]),
            stage_cls=np.array([s[1] for s in self.stages]),
            stage_cls_claimed=np.array([s[2] for s in self.stages]),
            **stacked,
            **extra,
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--out", default="results/stepwise/")
    args = parser.parse_args()

    config, pars, batch = setup(args.config)
    sw = Stepwise(config, pars, batch)
    sw.finish_stage("start")
    extra = {}
    if config.objective == "cls_nn":
        extra |= sw.events()
        extra["bce"] = sw.opt_nn()
        sw.finish_stage("nn")
    for var in config.opt_cuts:
        grid = np.arange(30.0, 150.1, 2.0)
        proxy, best = sw.opt_cut(var, grid)
        extra |= {f"cut_grid_{var}": grid, f"cut_proxy_{var}": proxy}
        sw.finish_stage("cut")
    sw.opt_bins()
    sw.finish_stage("bins")
    os.makedirs(args.out, exist_ok=True)
    sw.save(os.path.join(args.out, f"{config.suffix}.npz"), **extra)
