import jax.numpy as jnp
import itertools
import h5py
import numpy as np


def get_generator(config, split):
    with h5py.File(config.preprocess_files[split], "r") as f:
        ds = f["stack"]
        # this makes all non-repetitive combinations of batches, for 2 pairs
        # [1,2,3] --> [[1,2], [1,3], [2,3]]
        # load the combinations together to avoid tiny rest batch, more data
        # variability
        # this returns slices of the memory layout on disk
        batch_combi = list(itertools.combinations(list(ds.iter_chunks()), 2))
        while True:
            for batch in batch_combi:
                # shape is (n_sample_sys, n_events, n_vars)
                batch = jnp.concatenate([ds[b] for b in batch], axis=1)
                batch_sf = config.split_events[split] / batch.shape[1]
                # upscale to original hist content
                batch = batch.at[:, :, config.weight_idx].multiply(batch_sf)
                yield batch
            np.random.shuffle(batch_combi)
