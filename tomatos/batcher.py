import jax.numpy as jnp
import itertools
import h5py
import numpy as np


def get_iterator(config, split):
    with h5py.File(config.preprocess_files[split], "r") as f:
        ds = f["stack"]
        # this makes all non-repetitive combinations of 2 batches
        # [1,2,3] --> [[1,2], [1,3], [2,3]]
        # avoid tiny rest batch, more data variability
        batch_combi = list(itertools.combinations(list(ds.iter_chunks()), 2))
        while True:
            for batch in batch_combi:
                yield jnp.concatenate([ds[b] for b in batch], axis=1)
            np.random.shuffle(batch_combi)
