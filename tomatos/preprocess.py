import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def stack_inputs(
    filepath,
    config,
    region,
    n_events=0,
    sys="NOSYS",
    rescale_weights=True,
):
    """make array of input variables and attach desired weights to keep the
    weights during shuffling, so it gets the shape:

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
    config : tomatos.configuration.Setup
        config object
    filepath : sys
        systematic

    Returns
    -------
    out : ndarray
        array holding per event pairs of feature values and weights

    """

    if "run2" in filepath:
        is_mc = False
    else:
        is_mc = True

    with h5py.File(filepath, "r") as f:
        arr = np.zeros((n_events, 2, config.n_features))

        for i, var in enumerate(config.vars):
            # fill
            var_name = var + "_" + sys + "." + region
            w_name = "weights_" + sys + "." + region

            # auto up and down scale
            # values
            n_var_events = f[var_name].shape[0]
            arr[:, 0, i] = np.resize(f[var_name][:], (n_events))

            # weights
            if is_mc:
                arr[:, 1, i] = np.resize(f[w_name][:], (n_events))
            else:
                arr[:, 1, i] = np.ones(n_events)

            # account for scaling in weights
            if rescale_weights:
                arr[:, 1, i] *= n_var_events / n_events

        return arr


def min_max_norm(data, estimate_regions_data):
    scaler = MinMaxScaler()

    # find the min max by going over all samples
    for key in data.keys():
        scaler.partial_fit(data[key][:, 0, :])
    for key in estimate_regions_data.keys():
        scaler.partial_fit(estimate_regions_data[key][:, 0, :])

    # apply scaling
    for key in data.keys():
        data[key][:, 0, :] = scaler.transform(data[key][:, 0, :])
    # scale the estimate regions
    for key in estimate_regions_data.keys():
        estimate_regions_data[key][:, 0, :] = scaler.transform(
            estimate_regions_data[key][:, 0, :]
        )

    return data, estimate_regions_data, scaler


def get_n_events(filepath, var):
    with h5py.File(filepath, "r") as f:
        return f[var].shape[0]


def prepare_data(config):
    data = {}
    estimate_regions_data = {}

    max_events = 0
    for var in config.vars:
        for sys in config.systematics:
            var_sys = var + "_" + sys + ".SR_xbb_2"
            n = get_n_events(filepath=config.files["k2v0"], var=var_sys)
            if n > max_events:
                max_events = n
                max_var_sys = var_sys

    data["bkg"] = stack_inputs(
        config.files["run2"],
        config,
        region="SR_xbb_1",
        n_events=max_events,
        rescale_weights=True,
    )

    for sys in config.systematics:
        data[sys] = stack_inputs(
            config.files["k2v0"],
            config,
            region="SR_xbb_2",
            n_events=max_events,
            sys=sys,
            rescale_weights=True,
        )

    data["ps"] = stack_inputs(
        config.files["ps"],
        config,
        region="SR_xbb_2",
        n_events=max_events,
        sys="NOSYS",
        rescale_weights=True,
    )

    estimate_regions = [
        "CR_xbb_1",
        "CR_xbb_2",
        "VR_xbb_1",
        "VR_xbb_2",
    ]

    for reg in estimate_regions:
        estimate_regions_data[reg] = stack_inputs(
            config.files["run2"],
            config,
            region=reg,
            n_events=get_n_events(
                filepath=config.files["run2"], var=f"m_hh_NOSYS.{reg}"
            ),
            sys="NOSYS",
            rescale_weights=False,
        )

    data, estimate_regions_data, scaler = min_max_norm(data, estimate_regions_data)

    config.data_types = []
    jnp_data = []
    for k in data.keys():
        config.data_types += [k]
        jnp_data += [np.asarray(data[k])]

    config.scaler_scale = scaler.scale_
    config.scaler_min = scaler.min_
    config.bkg_estimate = estimate_regions_data

    return jnp_data
