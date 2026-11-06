# TOMATOS: auTOMATed Optimization of Sensitivity

Particle physics analyses at the LHC are typically optimized step by step, using proxy objectives aiming to improve the final sensitivity. Systematic uncertainties often play a large role in that sensitivity, but only enter fully in the last statistical fit. tomatos is a framework that turns the whole chain into a single differentiable procedure and optimizes it end-to-end on the final, uncertainty-aware sensitivity.


## Toy example
Find a small signal in a large background using 2 kinematic features and a background uncertainty with two different analysis strategies. Each GIF shows the manual iterative procedure on top and the automated version below.

### Neural Network 
Manual optimization uses the features to train a signal/background classifier and optimizes the histogram bins to increase the overall signal-to-background ratio. The final sensitivity that includes uncertainties is then obtained in the hypothesis test. The classifier favours the highest signal-to-background ratio towards high jet pT, the top of the 2D plane, where the background uncertainty increases. The automated optimization instead learns to avoid regions of higher uncertainty and reaches a better sensitivity. 

![neural network](assets/pipeline_nn.gif)

### Event Selection
The same data but now using a selection on jet pT and a fit in the mass variable. The automated chain optimizes the selection and the histogram bins together on the final sensitivity. It selects more inclusively, where the background is better known, and reaches a better sensitivity than the manual one.

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

Built on [JAX](https://github.com/jax-ml/jax), [equinox](https://github.com/patrick-kidger/equinox), [optax](https://github.com/google-deepmind/optax), [pyhf](https://github.com/scikit-hep/pyhf) and [neos](https://github.com/gradhep/neos).

