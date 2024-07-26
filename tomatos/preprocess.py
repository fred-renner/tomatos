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
    event_range=[0.0, 0.8],
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
        # scale max events
        ranged_n_events = int(np.abs(event_range[1] - event_range[0]) * n_events)
        arr = np.zeros((ranged_n_events, 2, config.n_features))

        for i, var in enumerate(config.vars):
            # fill
            var_name = var + "_" + sys + "." + region
            w_name = "weights_" + sys + "." + region

            # auto up and down scale
            # values
            n_var_events = f[var_name].shape[0]
            idx_0 = int(np.floor(event_range[0] * n_var_events))
            idx_1 = int(np.floor(event_range[1] * n_var_events))
            indices = np.arange(idx_0, idx_1)

            arr[:, 0, i] = np.resize(f[var_name][indices], (ranged_n_events))

            # weights
            if is_mc:
                arr[:, 1, i] = np.resize(f[w_name][indices], (ranged_n_events))
            else:
                arr[:, 1, i] = np.ones(ranged_n_events)

            selected_sf = n_events / ranged_n_events
            rescale_sf = len(indices) / ranged_n_events

            # account for scaling in weights
            if rescale_weights:
                arr[:, 1, i] *= selected_sf * rescale_sf

        return arr


def min_max_norm(train, valid, test, estimate_regions_data):
    scaler = MinMaxScaler()

    # find the min max by going over all samples
    # train, valid, test
    for data in [train, valid, test]:
        for key in data.keys():
            scaler.partial_fit(data[key][:, 0, :])

    for key in estimate_regions_data.keys():
        scaler.partial_fit(estimate_regions_data[key][:, 0, :])

    # apply scaling to train, valid, test
    for data in [train, valid, test]:
        for key in data.keys():
            data[key][:, 0, :] = scaler.transform(data[key][:, 0, :])
    # scale the estimate regions
    for key in estimate_regions_data.keys():
        estimate_regions_data[key][:, 0, :] = scaler.transform(
            estimate_regions_data[key][:, 0, :]
        )
    return train, valid, test, estimate_regions_data, scaler


def get_n_events(filepath, var):
    with h5py.File(filepath, "r") as f:
        return f[var].shape[0]


def stack_data(config, max_events, event_range):
    data = {}
    data["bkg"] = stack_inputs(
        config.files["run2"],
        config,
        region="SR_xbb_1",
        n_events=max_events,
        rescale_weights=True,
        event_range=event_range,
    )

    for sys in config.systematics:
        data[sys] = stack_inputs(
            config.files["k2v0"],
            config,
            region="SR_xbb_2",
            n_events=max_events,
            sys=sys,
            rescale_weights=True,
            event_range=event_range,
        )

    data["ps"] = stack_inputs(
        config.files["ps"],
        config,
        region="SR_xbb_2",
        n_events=max_events,
        sys="NOSYS",
        rescale_weights=True,
        event_range=event_range,
    )

    return data


def prepare_data(config):
    max_events = 0
    for var in config.vars:
        for sys in config.systematics:
            var_sys = var + "_" + sys + ".SR_xbb_2"
            n = get_n_events(filepath=config.files["k2v0"], var=var_sys)
            if n > max_events:
                max_events = n
                max_var_sys = var_sys

    train = stack_data(config, max_events, event_range=[0.0, 0.8])
    valid = stack_data(config, max_events, event_range=[0.8, 0.9])
    test = stack_data(config, max_events, event_range=[0.9, 1.0])

    estimate_regions = [
        "CR_xbb_1",
        "CR_xbb_2",
        "VR_xbb_1",
        "VR_xbb_2",
    ]

    estimate_regions_data = {}
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
            event_range=[0.0, 1.0],
        )

    train, valid, test, estimate_regions_data, scaler = min_max_norm(
        train, valid, test, estimate_regions_data
    )

    config.data_types = []
    train_ = []
    valid_ = []
    test_ = []
    for k in train.keys():
        config.data_types += [k]
        train_ += [np.asarray(train[k])]
        valid_ += [np.asarray(valid[k])]
        test_ += [np.asarray(test[k])]

    config.scaler_scale = scaler.scale_
    config.scaler_min = scaler.min_
    config.bkg_estimate = estimate_regions_data

    return train_, valid_, test_
