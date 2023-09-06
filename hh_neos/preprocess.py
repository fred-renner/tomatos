import h5py
import jax.numpy as jnp
import numpy as np
from sklearn.preprocessing import MinMaxScaler

w_CR = 0.007691879267891989
err_w_CR = 0.0007230248761909538


def stack_inputs(filepath, config):
    """make matrix of input variables

    Parameters
    ----------
    filepath : str
        path to file
    config : hh_neos.configuration.Setup
        config object

    Returns
    -------
    np.arrray
        matrix of input variables per event
    """
    if "run2" in filepath:
        config.region = "SR_xbb_1"
    else:
        config.region = "SR_xbb_2"

    with h5py.File(filepath, "r") as f:
        # init array
        arr = np.zeros(
            (f[config.vars[0] + "." + config.region].shape[0], len(config.vars))
        )
        for i, var in enumerate(config.vars):
            # put in matrix
            arr[:, i] = f[var + "." + config.region][:]

        return arr


def min_max_norm(data):
    # need to concatenate to find overall min max
    data_ = np.concatenate([data[key] for key in data.keys()])

    shapes = [data[key].shape[0] for key in data.keys()]
    shapes_cumulative = np.cumsum(shapes)

    scaler = MinMaxScaler()
    # find min max in columns
    scaler.fit(data_)
    data_min = scaler.data_min_
    data_max = scaler.data_max_
    # apply
    scaled_data = scaler.transform(data_)
    # split to original samples
    scaled_data_splitted = np.split(scaled_data, shapes_cumulative[:-1])

    for i, key in enumerate(data.keys()):
        data[key] = scaled_data_splitted[i]

    return data, data_min, data_max


def append_weights(
    arr,
    filepath,
    config,
    replicate_weight=1,
):
    if "run2" in filepath:
        config.region = "SR_xbb_1"
    else:
        config.region = "SR_xbb_2"
    with h5py.File(filepath, "r") as f:
        weights = f["weights." + config.region][:] * replicate_weight
        weights = weights.reshape((weights.shape[0], 1))
        arr = np.append(arr, weights, axis=1)
    return arr


def prepare_data(config):
    data = {
        "sig": stack_inputs(config.files["k2v0"], config),
        # "ttbar": stack_inputs(config.files["ttbar"],config),
        "multijet": stack_inputs(config.files["run2"], config),
    }
    data_min = 0
    data_max = 1
    if config.include_bins:
        data, data_min, data_max = min_max_norm(data)

    print([data[key].shape[0] for key in data.keys()])

    # replicate to have same sample size as signal
    replicate_weight = 1 / (data["sig"].shape[0] / data["multijet"].shape[0])

    # I know its bad to put the weights in the data, but there is so much
    # shuffling etc happening, that it is the easiest for now
    data = {
        "sig": append_weights(data["sig"], config.files["k2v0"], config=config),
        # "ttbar": append_weights(config.files["ttbar"]),
        "multijet": append_weights(
            data["multijet"],
            config.files["run2"],
            config=config,
            replicate_weight=replicate_weight * w_CR,
        ),
    }
    # replicate to have same size sample input
    data["multijet"] = np.asarray(np.resize(data["multijet"][:], data["sig"].shape))

    data = (jnp.asarray(data["sig"]), jnp.asarray(data["multijet"]))
    config.data_min = data_min
    config.data_max = data_max
    return data
