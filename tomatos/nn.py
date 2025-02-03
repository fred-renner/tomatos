import equinox as eqx
import jax


# the choice for equinox is ease of use with jax
# https://www.reddit.com/r/MachineLearning/comments/u34oh2/d_what_jax_nn_library_to_use/
# https://docs.kidger.site/equinox/
class NeuralNetwork(eqx.Module):
    layers: list

    def __init__(self, n_features):
        rng_state = 0
        key = jax.random.PRNGKey(rng_state)
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
        self.layers = [
            eqx.nn.Linear(n_features, 100, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(100, 100, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(100, 100, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(100, 100, key=key4),
            jax.nn.relu,
            eqx.nn.Linear(100, 1, key=key5),
            jax.nn.sigmoid,
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
