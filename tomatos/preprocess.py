import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import uproot


def init_2d_data(config):
    with h5py.File(config.preprocess_files["data"], "w") as f:
        for sample_sys in config.sample_sys:
            f.create_dataset(
                name=sample_sys,
                shape=(0, len(config.vars)),
                maxshape=(
                    None,  # allows resize, makes appending easy, instead of idx tracking
                    len(config.vars),
                ),
                compression="gzip" if config.compress_input_files else None,
                dtype="f4",
                chunks=True,  # autochunking, since resizing
            )


def fill(
    config,
    sample_sys,
    data,
    scaler,
):
    # transform dict data into 2d array of dimension (events, vars)
    data_2d = np.array([data[var] for var in config.vars]).T
    # iteratively update min max for scaling of train vars
    scaler.partial_fit(data_2d[:, : config.nn_inputs_idx_end])
    with h5py.File(config.preprocess_files["data"], "r+") as f:
        ds = f[sample_sys]
        old_shape = ds.shape
        new_shape = (old_shape[0] + data_2d.shape[0], old_shape[1])
        # resize array on disk
        ds.resize(new_shape)
        # append on disk
        ds[old_shape[0] :] = data_2d

    return


def fill_2d_data(config, scaler):
    for sample_sys, path in config.sample_files_dict.items():
        with uproot.open(path) as file:
            tree = file[config.tree_name]
            for data in tree.iterate(step_size=config.chunk_size, library="np"):
                #######
                # pre/analysis-selection could be here
                # ...transfer generate test files code
                #######
                fill(config, sample_sys, data, scaler)


def get_max_events(config):
    max_events = 0
    with h5py.File(config.preprocess_files["data"], "r") as f:
        for sample_sys in config.sample_sys:
            n = f[sample_sys].shape[0]
            if n > max_events:
                max_events = n
    return max_events


def init_splits(config, idx_bounds):
    for split in ["train", "valid", "test"]:
        n_events = idx_bounds[split][1] - idx_bounds[split][0]
        logging.info(f"{split} events: {n_events}")
        config.splitting[split]["events"] = n_events
        with h5py.File(config.preprocess_files[split], "w") as f_split:
            f_split.create_dataset(
                name="stack",
                shape=(len(config.sample_sys), n_events, len(config.vars)),
                compression="gzip" if config.compress_input_files else None,
                dtype="f4",
                chunks=(
                    len(config.sample_sys),
                    np.minimum(config.chunk_size, n_events / 2),  # / 2 see batcher
                    len(config.vars),
                ),
            )


def fill_splits(config, max_events, idx_bounds, scaler):

    # upsampling is straightforward and intuitive to avoid a class imbalance
    # makes particular sense for jax as it likes predefined array sizes,
    # also unifies workflow a lot

    # these are just file handles
    with (
        h5py.File(config.preprocess_files["data"], "r") as f_data,
        h5py.File(config.preprocess_files["train"], "r+") as f_train,
        h5py.File(config.preprocess_files["valid"], "r+") as f_valid,
        h5py.File(config.preprocess_files["test"], "r+") as f_test,
    ):
        np.random.seed(1)
        file = {
            "data": f_data,
            "train": f_train,
            "valid": f_valid,
            "test": f_test,
        }

        # upsampling, shuffling, scaling nn inputs, scaling event weights
        # having them all together here avoids multiple IO
        for i, sample_sys in enumerate(config.sample_sys):
            # quick and cheap for now
            # ds=(max_events=1e7, config.vars=20) ~ 1.6 GB for float32
            # in particular avoids heavy random index read/write
            ds = file["data"][sample_sys][:]
            upsample_sf = ds.shape[0] / max_events
            # shuffle possibly ordered event correlations (in place)
            np.random.shuffle(ds)
            # this replicates from the beginning, i prefer this to random
            # resampling as duplication only happens if necessary
            ds = np.resize(ds, (max_events, len(config.vars)))
            # shuffle again to remove periodicity
            np.random.shuffle(ds)
            # min max scale nn input vars for this sample_sys
            ds[:, : config.nn_inputs_idx_end] = scaler.transform(
                ds[:, : config.nn_inputs_idx_end]
            )
            # write
            for split in ["train", "valid", "test"]:
                out = ds[idx_bounds[split][0] : idx_bounds[split][1]]
                split_sf = 1 / config.splitting[split]["ratio"]
                # account for resampling, splitting, and k-folding, such that
                # except for stats the hist yields in each split are the same
                config.splitting[split]["preprocess_scale_factor"][i] *= (
                    upsample_sf * split_sf * config.k_fold_sf
                )
                file[split]["stack"][i, :] = out


def run(config):
    # create data.h5 with 2d data structure (n_events, config.vars)
    init_2d_data(config)
    # scaler for nn inputs
    scaler = MinMaxScaler()
    # fill data.h5 from trees
    fill_2d_data(config, scaler)
    # scale to sample with most events
    max_events = get_max_events(config)
    # train valid test indices, can go in order since they are shuffled
    train_end = round(max_events * config.splitting["train"]["ratio"])
    valid_end = train_end + round(max_events * config.splitting["valid"]["ratio"])
    idx_bounds = {
        "train": [0, train_end],
        "valid": [train_end, valid_end],
        "test": [valid_end, max_events],
    }
    # create train.h5, valid.h5, test.h5 with according event_sizes
    init_splits(config, idx_bounds)
    fill_splits(config, max_events, idx_bounds, scaler)
    # collect scaling, avoid writing objects to config -->jax
    config.scaler_scale = scaler.scale_
    config.scaler_min = scaler.min_
