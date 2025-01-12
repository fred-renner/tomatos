import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import logging
import uproot
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def init_2d_data(config):
    with h5py.File(config.preprocess_path + "data.h5", "w") as f:
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
                chunks=True,  # autochunking because of maxshape None, however makes good performance choices
            )


def fill_2d_data(config, scaler):
    for sample_sys, path in config.sample_files_dict.items():
        with uproot.open(path) as file:
            tree = file[config.tree_name]
            for data in tree.iterate(step_size=config.batch_size, library="np"):
                #######
                # preselection could be here
                #######
                stack_inputs(config, sample_sys, data, scaler)


def stack_inputs(
    config,
    sample_sys,
    data,
    scaler,
):
    # transform dict data into 2d array of dimension (events, vars)
    data_2d = np.array([data[var] for var in config.vars]).T
    # iteratively update min max for scaling of train vars
    scaler.partial_fit(data_2d[:, : config.nn_inputs_idx])
    with h5py.File(config.preprocess_path + "data.h5", "r+") as f:
        ds = f[sample_sys]
        old_shape = ds.shape
        new_shape = (old_shape[0] + data_2d.shape[0], old_shape[1])
        # resize array on disk
        ds.resize(new_shape)
        # append on disk
        ds[old_shape[0] :] = data_2d

    return


def get_max_events(config):
    max_events = 0
    with h5py.File(config.preprocess_path + "data.h5", "r") as f:
        for sample_sys in config.sample_sys:
            n = f[sample_sys].shape[0]
            if n > max_events:
                max_events = n
    return max_events


def init_splits(config, idx_bounds):
    for split in ["train", "valid", "test"]:
        out_file = config.preprocess_path + split + ".h5"
        n_events = idx_bounds[split][1] - idx_bounds[split][0]
        logging.info(f"{split} events: {n_events}")
        with h5py.File(out_file, "w") as f_split:
            for sample_sys in config.sample_sys:
                f_split.create_dataset(
                    name=sample_sys,
                    shape=(n_events, len(config.vars)),
                    compression="gzip" if config.compress_input_files else None,
                    dtype="f4",
                    chunks=True,  # autochunks makes good performance choices
                )


def fill_splits(config, max_events, idx_bounds, scaler):

    # upsampling is straightforward and intuitive to avoid a class imbalance
    # makes particular sense for jax as it likes predefined array sizes,
    # also simplifies batching.

    # these are just file handles
    with (
        h5py.File(config.preprocess_path + "data.h5", "r") as f_data,
        h5py.File(config.preprocess_path + "train.h5", "r+") as f_train,
        h5py.File(config.preprocess_path + "valid.h5", "r+") as f_valid,
        h5py.File(config.preprocess_path + "test.h5", "r+") as f_test,
    ):
        np.random.seed(1)
        file = {
            "data": f_data,
            "train": f_train,
            "valid": f_valid,
            "test": f_test,
        }

        # upsampling, shuffling, scaling nn inputs, scaling event weights
        # having it compact here avoids multiple IO
        for sample_sys in config.sample_sys:
            # quick and cheap for now, avoid IO heavy random idx
            # reading, should be fine since this is after preselection and
            # requires less than e.g.
            # ds=(max_events=1e7, config.vars=20) ~ 1.6 GB for float32
            ds = file["data"][sample_sys][:]
            upsample_sf = ds.shape[0] / max_events
            # suffle events in place
            np.random.shuffle(ds)
            # this replicates from the beginning, i prefer this to random
            # resampling as duplication only happens if necessary
            ds = np.resize(ds, (max_events, len(config.vars)))
            # shuffle again to remove periodicity
            np.random.shuffle(ds)
            # min max scale nn input vars
            ds[:, : config.nn_inputs_idx] = scaler.transform(
                ds[:, : config.nn_inputs_idx]
            )
            # write
            for split in ["train", "valid", "test"]:
                out = ds[idx_bounds[split][0] : idx_bounds[split][1]]  # (this a view)
                split_sf = 1 / config.split_ratio[split]
                # account for resampling, splitting, and k-folding, such that
                # except for stats the hist yields in each split are the same
                out[:, config.weight_idx] *= upsample_sf * split_sf * config.k_fold_sf
                file[split][sample_sys][:] = out


def prepare_data(config):
    # create data.h5 with 2d data structure (n_events, config.vars)
    init_2d_data(config)
    # scaler for nn inputs
    scaler = MinMaxScaler()
    # fill data.h5 from trees
    fill_2d_data(config, scaler)
    # scale to sample with most events
    max_events = get_max_events(config)
    # train valid test indices, can go in order since they are shuffled
    train_end = round(max_events * config.split_ratio["train"])
    valid_end = train_end + round(max_events * config.split_ratio["valid"])
    idx_bounds = {
        "train": [0, train_end],
        "valid": [train_end, valid_end],
        "test": [valid_end, max_events],
    }
    # create train.h5, valid.h5, test.h5 with according event_sizes
    init_splits(config, idx_bounds)
    fill_splits(config, max_events, idx_bounds, scaler)
    config.scaler = scaler
