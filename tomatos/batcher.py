import jax.numpy as jnp
import itertools
import h5py
import numpy as np


def get_generator(config, split):
    with h5py.File(config.preprocess_files[split], "r") as f:
        ds = f["stack"]
        # this makes all non-repetitive combinations of batches, for r=2
        # [1,2,3] --> [[1,2], [1,3], [2,3]]
        # load the combinations together to avoid tiny rest batch, more data
        # variability
        # r slice combinations of the memory layout on disk
        batch_combi = list(itertools.combinations(list(ds.iter_chunks()), r=2))
        while True:
            for batch in batch_combi:
                # shape is (n_sample_sys, n_events, n_vars)
                batch = jnp.concatenate([ds[b] for b in batch], axis=1)
                # scale to total events
                batch_sf = config.splitting[split]["events"] / batch.shape[1]
                # this is an array of sample_sys size
                config.splitting[split]["scale_factor"] = (
                    config.splitting[split]["preprocess_scale_factor"] * batch_sf
                )

                yield batch
            np.random.shuffle(batch_combi)
