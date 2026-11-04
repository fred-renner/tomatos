# TOMATOS: auTOMATed Optimization of Sensitivity

**End-to-end differentiable optimization of a particle-physics analysis, from event selection to the final statistical test, written in JAX.**

![cut + mass fit](assets/pipeline_mass.gif)

![neural network](assets/pipeline_nn.gif)

## The problem

A search is a chain of steps: select events, maybe train a classifier, bin a distribution, then fit a statistical model with all its uncertainties to get the expected sensitivity ($CL_s$). Usually every step is tuned on its own stand-in objective:

- cuts on $S/\sqrt{B}$
- a classifier on cross-entropy
- bins on $\sum s^2/b$

The real figure of merit, including the uncertainties, is only seen at the very end.

None of these stand-ins knows about the uncertainties, so the analysis ends up optimized for the wrong target.

## What the GIFs show

Both GIFs use the same toy:

- **Signal:** a resonance at 125 GeV with fairly hard jets.
- **Background:** steeply falling, and taken from a data-driven estimate extrapolated from soft jets. That estimate runs out of statistics for hard jets and high masses. Its uncertainty grows exactly where $S/\sqrt{B}$ wants to cut, and it is uncorrelated from bin to bin, so the mass sidebands cannot constrain it.

**Manual Optimization.** One box after the other, each on its proxy:

1. The cut is scanned on $S/\sqrt{B}$, which cuts hard, into the region where the background is poorly known.
2. The bin edges are swept on $\sum s^2/b$.
3. Only then is the hypothesis test run.

**Automated Optimization.** tomatos tunes the cut and the bin edges together, directly on the expected $CL_s$ with all uncertainties:

- It loosens the cut, keeping more background but background that is well known.
- It puts narrow bins on the peak.
- It ends below the manual result.

With a neural network the same happens. The classifier trained on cross-entropy separates signal well but drags in poorly known background. The network trained on the expected $CL_s$ learns to avoid it.

The mass pipeline stays fully interpretable. The result is an ordinary cut in GeV and bin edges in GeV, a plain cut-and-fit analysis you can read, cross-check and reuse. The optimizer only picks the values.

## How it works

The whole chain is made differentiable:

| step | normally | in tomatos |
| --- | --- | --- |
| cuts | hard thresholds | sigmoid cuts with learnable thresholds |
| classifier | trained on cross-entropy | MLP trained on the final sensitivity |
| histograms | hard binning | binned kernel-density estimate (KDE) with learnable bin edges |
| statistics | separate fit afterwards | differentiable [pyhf](https://github.com/scikit-hep/pyhf) model with all systematics, expected $CL_s$ through the fit |

Gradients flow from the expected $CL_s$ back through the likelihood fit, the histograms and the cuts into the network weights. Everything is trained together with Adam.

The KDE bandwidth starts wide, so that gradients exist, and is annealed during training. Validation and test always use sharp histograms, so the reported $CL_s$ is the one a normal analysis would get.

## Built on neos and relaxed

The core idea is not new. It comes from [neos](https://github.com/gradhep/neos) and [relaxed](https://github.com/gradhep/relaxed) (gradhep): relaxed cuts and KDE histograms, and the expected $CL_s$ differentiated through a pyhf fit. tomatos uses both directly.

What tomatos adds is the analysis around it:

- **ROOT ntuples as input:** `ntuple_path/SAMPLE/SYSTEMATIC.root`, with event weights and scale factors.
- **Systematics:**
  - tree variations, one file per systematic
  - per-event weight variations from branches (`weight_systematics`), optionally uncorrelated bin by bin (`bin_by_bin: true`), e.g. for the statistics of a data-driven estimate
  - normalization uncertainties (`norm_systematics`)
- **Cuts, a neural network and bin edges optimized in one chain**, for a variable fit (`cls_var`) or a fit of the NN output (`cls_nn`).
- **Batching from disk**, train/valid/test splits and k-folds.
- **A bandwidth controller** that anneals the KDE while keeping at least `min_bkg` background events in every bin.

## Quickstart

```bash
make install   # Python 3.9 venv in .venv
make demo      # toy ntuples, train both toy configs, render the README gifs
make test      # smoke test: a few training steps per config
```

On a laptop CPU the two trainings take a few minutes each. Rendering the GIFs takes a few more.

Run a single config:

```bash
tomatos --config configs/toy_mass.yaml --prep --train --plot
```

Outputs land in `results/`: metrics, models, plots and an mp4 of the histogram evolution. `scripts/stepwise.py` runs the manual, stepwise optimization on the same config for comparison.

The dependency stack is pinned to `jax==0.3.14`, because the likelihood fit relies on `jaxopt`, which is retired and incompatible with newer JAX. The install therefore needs Python 3.9. `make install` runs:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install --editable .
pip install jaxlib==0.3.14 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## Configuration

A config names the samples, the variables, what to optimize and the uncertainties. From `configs/toy_mass.yaml`:

```yaml
objective: "cls_var"     # fit a variable; "cls_nn" fits the nn output
cls_var: "m"
n_bins: 5
include_bins: true       # bin edges are optimized too
opt_cuts:
  pt1:
    keep: "above"
    init: 110
weight_systematics:
  BKG_EST:
    sample: "bkg"
    up: "bkg_est_up"     # per-event weight branches
    down: "bkg_est_down"
    bin_by_bin: true     # one nuisance parameter per bin
min_bkg: 10              # background events every bin keeps
```

## Layout

| module | role |
| --- | --- |
| `config.py` | reads the YAML config, sets up the sample/systematic structure and output paths |
| `preprocess.py` | ROOT ntuples → upsampled, shuffled, min-max scaled train/valid/test arrays (HDF5) |
| `batcher.py` | streams batches from disk |
| `select.py` | differentiable cuts and region selection via event weights |
| `histograms.py` | differentiable histograms (binned KDE), NN forward pass |
| `workspace.py` | systematics (correlated or bin by bin) and the pyhf model |
| `pipeline.py` | the loss: cuts → histograms → likelihood → $CL_s$ |
| `training.py`, `solver.py`, `constraints.py` | optimization loop, Adam with a one-cycle schedule, parameter limits, bandwidth controller |
| `nn.py` | the MLP (equinox) |
| `plotting.py` | diagnostics and the training movie |

| script | role |
| --- | --- |
| `tests/generate_toy_files.py` | the toy ntuples |
| `scripts/stepwise.py` | the manual baseline: $S/\sqrt{B}$ cut, BCE classifier, $\sum s^2/b$ bins |
| `scripts/pipeline_gif.py` | the README gifs |
| `scripts/design_scan.py`, `scripts/sweep.py` | expected $CL_s$ vs a cut, hyperparameter loops |

Built with [JAX](https://github.com/jax-ml/jax), [equinox](https://github.com/patrick-kidger/equinox), [optax](https://github.com/google-deepmind/optax), [pyhf](https://github.com/scikit-hep/pyhf), [relaxed](https://github.com/gradhep/relaxed) and [neos](https://github.com/gradhep/neos).

## License

Apache 2.0
