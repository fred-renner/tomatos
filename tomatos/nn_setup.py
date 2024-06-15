import equinox as eqx
import jax


class NeuralNetwork(eqx.Module):
    layers: list

    def __init__(self, n_features):
        rng_state = 0
        key = jax.random.PRNGKey(rng_state)
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


# https://docs.kidger.site/equinox/examples/init_apply/
def make_nn(config):
    model = NeuralNetwork(config.n_features)

    if config.preload_model:
        model = eqx.tree_deserialise_leaves(config.preload_model_path, model)

    params, static = eqx.partition(model, eqx.is_inexact_array)

    def init_fn():
        return params

    def apply_fn(_params, x):
        model = eqx.combine(_params, static)
        return model(x)

    return init_fn, apply_fn, static


def init(config):
    init_random_params, nn, nn_setup = make_nn(config)
    init_pars = dict(nn_pars=init_random_params())

    return init_pars, nn, nn_setup
