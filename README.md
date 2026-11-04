# TOMATOS: auTOMATed Optimization of Sensitivity

End-to-end differentiable optimization of a particle-physics analysis in JAX.

A search for new physics is a chain of steps: event selection, a classifier, histogram binning and a statistical fit. Usually each step is optimized separately, on its own proxy objective such as S/√B or cross-entropy, and without the systematic uncertainties of the final fit. This takes many iterations and gives a suboptimal sensitivity.

tomatos makes the chain differentiable and optimizes all steps together on the expected sensitivity of the fit, systematic uncertainties included.

## Neural network

The data is a toy: a small signal on a large background, with a background uncertainty that grows where there are few events.

- **Top:** the classifier is trained on cross-entropy, then the bins are chosen. The sensitivity is only known after the fit.
- **Bottom:** tomatos trains the network on the fit result. It avoids the region where the background is poorly known.

![neural network](assets/pipeline_nn.gif)

## Cut and mass

The same problem without a network: a cut on jet momentum and the bin edges of the mass distribution. All parameters stay physical quantities, so the result is a standard, interpretable analysis.

![cut and mass](assets/pipeline_mass.gif)

## Quickstart

```bash
make install   # Python 3.9 venv in .venv
make demo      # toy data, both trainings, the gifs above
make test      # smoke test
```

```bash
tomatos --config configs/toy_nn.yaml --prep --train --plot
```

Outputs go to `results/`. The stack is pinned to `jax==0.3.14` because the fit uses `jaxopt`, which needs Python 3.9.

Built on JAX, equinox, optax, pyhf and neos.

## License

Apache 2.0
