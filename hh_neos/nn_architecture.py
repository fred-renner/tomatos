import jax.example_libraries.stax as stax
from jax.random import PRNGKey
import equinox as eqx
import jax


class NeuralNetwork(eqx.Module):
    layers: list

    def __init__(self, n_features):
        rng_state = 0
        key = PRNGKey(rng_state)
        key1, key2, key3 = jax.random.split(key, 3)
        # These contain trainable parameters.
        self.layers = [
            eqx.nn.Linear(n_features, 100, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(100, 100, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(100, 1, key=key3),
            jax.nn.sigmoid,
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def make_nn(in_size):
    nn = NeuralNetwork(in_size)
    params, static = eqx.partition(nn, eqx.is_inexact_array)

    def init_fn():
        return params

    def apply_fn(_params, x):
        model = eqx.combine(_params, static)
        return model(x)

    return init_fn, apply_fn


def init(config):
    init_random_params, nn = make_nn(in_size=config.n_features)
    init_pars = dict(nn_pars=init_random_params())

    return init_pars, nn
