import equinox as eqx
import jax


# the choice for equinox is ease of use with jax
# https://www.reddit.com/r/MachineLearning/comments/u34oh2/d_what_jax_nn_library_to_use/
# https://docs.kidger.site/equinox/
class NeuralNetwork(eqx.Module):
    layers: list

    def __init__(self, n_features):
        key = jax.random.PRNGKey(seed=0)
        key1, key2, key3 = jax.random.split(key, 3)
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
