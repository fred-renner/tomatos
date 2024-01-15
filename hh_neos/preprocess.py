import h5py
import jax.numpy as jnp
import numpy as np
from sklearn.preprocessing import MinMaxScaler

w_CR = 0.007691879267891989
err_w_CR = 0.0007230248761909538


def stack_inputs(
    filepath,
    config,
    sys="NOSYS",
    use_NOSYS_weights=True,
    custom_weights=1.0,
):
    """make array of input variables and attach desired weights to bookeep the
    weights, so it gets the shape:

    (n_events, 2, n_vars,)

    so we get sth like e.g. with 3 features
        val = np.array(
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ]
        )

    with wights
        w = np.array(
            [
                [90, 90, 90],
                [91, 91, 91],
                [92, 92, 92],
            ]
        )

    put together like
        data = np.array(
            [
                [
                    [[0, 1, 2], [90, 90, 90]],
                    [[3, 4, 5], [91, 91, 91]],
                    [[6, 7, 8], [92, 92, 92]],
                ]
            ]
        )


    Parameters
    ----------
    filepath : str
        path to file
    config : hh_neos.configuration.Setup
        config object
    filepath : sys
        systematic
    use_NOSYS_weights : bool
        attach NOSYS weights, but load given sys vars

    Returns
    -------
    out : ndarray
        array holding per event pairs of feature values and weights

    """

    if "run2" in filepath:
        config.region = "SR_xbb_1"
        is_mc = False
    else:
        config.region = "SR_xbb_2"
        is_mc = True

    with h5py.File(filepath, "r") as f:
        n_events = f[config.vars[0] + "_NOSYS" + "." + config.region].shape[0]
        arr = np.zeros((n_events, 2, config.n_features))

        if use_NOSYS_weights:
            w_sys = "NOSYS"
        else:
            w_sys = sys

        for i, var in enumerate(config.vars):
            # fill
            var_name = var + "_" + sys + "." + config.region
            w_name = "weights_" + w_sys + "." + config.region
            arr[:, 0, i] = f[var_name][:]
            if is_mc:
                arr[:, 1, i] = f[w_name][:]
            else:
                arr[:, 1, i] = np.full(n_events, custom_weights)

        return arr


def min_max_norm(data):
    # need to concatenate to find overall min max
    data_ = np.concatenate([data[key][:, 0, :] for key in data.keys()])

    shapes = [data[key].shape[0] for key in data.keys()]
    shapes_cumulative = np.cumsum(shapes)

    scaler = MinMaxScaler()
    # find min max in columns
    scaler.fit(data_)
    # apply
    scaled_data = scaler.transform(data_)
    # split to original samples
    scaled_data_splitted = np.split(scaled_data, shapes_cumulative[:-1])

    for i, key in enumerate(data.keys()):
        data[key][:, 0, :] = scaled_data_splitted[i]

    return data, scaler


def get_n_events(filepath, var):
    with h5py.File(filepath, "r") as f:
        return f[var].shape[0]


def prepare_data(config):
    data = {}

    # as we are upscaling the background to match the number of events in the
    # signal, need to account for in weights
    replicate_weight = 1 / (
        get_n_events(filepath=config.files["k2v0"], var="m_hh_NOSYS.SR_xbb_2")
        / get_n_events(filepath=config.files["run2"], var="m_hh_NOSYS.SR_xbb_1")
    )
    data["bkg"] = stack_inputs(
        config.files["run2"],
        config,
        custom_weights=replicate_weight * w_CR,
    )

    for sys in config.systematics:
        if "NOSYS" in sys:
            use_NOSYS_weights = True
        else:
            use_NOSYS_weights = False
        data[sys] = stack_inputs(
            config.files["k2v0"],
            config,
            use_NOSYS_weights=use_NOSYS_weights,
            sys=sys,
        )

    # if config.include_bins:
    data, scaler = min_max_norm(data)
    # print([data[key].shape[0] for key in data.keys()])

    # replicate to have same size sample input
    data["bkg"] = np.asarray(np.resize(data["bkg"][:], data["NOSYS"].shape))
    # print([data[key].shape for key in data.keys()])
    config.data_types = []
    jnp_data = []
    for k in data.keys():
        config.data_types += [k]
        jnp_data += [np.asarray(data[k])]

    config.scaler_scale = scaler.scale_
    config.scaler_min = scaler.min_

    # samples, events, features
    return jnp_data
