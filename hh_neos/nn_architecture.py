import jax.example_libraries.stax as stax
from jax.random import PRNGKey


def init(config):
    rng_state = 0  # random state

    # feel free to modify :)
    init_random_params, nn = stax.serial(
        stax.Dense(100),
        stax.Relu,
        stax.Dense(100),
        stax.Relu,
        stax.Dense(100),
        stax.Relu,
        stax.Dense(1),
        stax.Sigmoid,
    )

    # we have one less because of weights
    num_features = len(config.vars)
    _, init = init_random_params(PRNGKey(rng_state), (-1, num_features))
    init_pars = dict(nn_pars=init)
    return init_pars, nn


# from typing import Sequence

# import flax.linen as nn

# class MLP(nn.Module):
#   features: Sequence[int]

#   @nn.compact
#   def __call__(self, x):
#     for feat in self.features[:-1]:
#       x = nn.relu(nn.Dense(feat)(x))
#     x = nn.Dense(self.features[-1])(x)
#     return x

# model = MLP([12, 8, 4])
# batch = jnp.ones((32, 10))
# variables = model.init(jax.random.PRNGKey(0), batch)
# output = model.apply(variables, batch)
