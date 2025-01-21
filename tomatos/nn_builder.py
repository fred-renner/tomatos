import equinox as eqx
import jax

# the choice for equinox is ease of use with jax, by a core jax developer
# https://www.reddit.com/r/MachineLearning/comments/u34oh2/d_what_jax_nn_library_to_use/
# https://docs.kidger.site/equinox/


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


def init(config):
    model = NeuralNetwork(config.nn_inputs_idx_end)

    if config.preload_model:
        model = eqx.tree_deserialise_leaves(config.preload_model_path, model)

    # split model into parameters to optimize and the nn architecture
    nn_pars, nn_arch = eqx.partition(model, eqx.is_array)

    return nn_pars, nn_arch
