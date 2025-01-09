from typing import Generator

import jax.numpy as jnp
import numpy.random as npr
from sklearn.model_selection import train_test_split

Array = jnp.ndarray
rng_state = 0


def split_data(data, ratio):
    """Split arrays or matrices into random train and test subsets."""
    split = train_test_split(*data, train_size=ratio, random_state=rng_state)
    train, test = split[::2], split[1::2]

    return train, test


def make_iterator(train, batch_size):
    def batches(training_data: Array, batch_size: int) -> Generator:
        num_train = training_data[0].shape[0]
        num_complete_batches, leftover = divmod(num_train, batch_size)
        num_batches = num_complete_batches + bool(leftover)

        # batching mechanism, ripped and adjusted from the JAX docs
        def data_stream():
            rng = npr.RandomState(rng_state)
            while True:
                perm = rng.permutation(num_train)
                for i in range(num_batches):
                    batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                    batch_data = [points[batch_idx] for points in train]

                    # Calculate the fraction of total for the current yield
                    fraction_of_total = len(batch_idx) / num_train

                    yield batch_data, i, num_batches, fraction_of_total

        return data_stream()

    return batches(train, batch_size)

# buffered shuffling of sequentially loaded indices might also be interesting