import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pyhf
import relaxed

import tomatos.histograms

Array = jnp.ndarray


def get_hist(config, nn, best_params, data):
    if config.include_bins:
        bins = best_params["bins"]
        logging.info(best_params["bins"])
    else:
        bins = config.bins
    if config.do_m_hh:
        # use whole data set to get correct norm
        yields = tomatos.histograms.hists_from_mhh(
            data={k: v for k, v in zip(config.data_types, data)},
            bandwidth=1e-8,
            bins=bins,
        )

    else:
        # to get correct yields would also need to pass whole data
        yields = tomatos.histograms.hists_from_nn(
            nn_pars=best_params["nn_pars"],
            data={k: v + 1e-8 for k, v in zip(config.data_types, data)},
            nn=nn,
            config=config,
            vbf_cut=best_params["vbf_cut"],
            eta_cut=best_params["eta_cut"],
            bandwidth=1e-8,
            bins=best_params["bins"] if config.include_bins else config.bins,
        )
    logging.info(
        (
            "Asimov Significance: ",
            relaxed.metrics.asimov_sig(s=yields["NOSYS"], b=yields["bkg"]),
        )
    )

    return bins, yields


def print_cls(config, yields):
    model = tomatos.workspace.model_from_hists(config, yields)

    CLs_obs, CLs_exp = pyhf.infer.hypotest(
        1.0,  # null hypothesis
        model.expected_data([0, 0.0]),
        model,
        test_stat="q",
        return_expected_set=True,
    )
    logging.info(f"      Observed CLs: {CLs_obs:.6f}")
    for expected_value, n_sigma in zip(CLs_exp, np.arange(-2, 3)):
        logging.info(f"Expected CLs({n_sigma:2d} σ): {expected_value:.6f}")


def to_python_lists(obj):
    """converts (also nested) nd.array or jax.array into a list living in dicts

    Parameters
    ----------
    obj : dict
        input dict

    Returns
    -------
    dict
        output dict
    """
    if isinstance(obj, (np.ndarray, jnp.DeviceArray)):
        # Convert arrays to Python lists
        return obj.tolist()
    elif isinstance(obj, dict):
        # Recursively process each dictionary value
        return {k: to_python_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively process each list element
        return [to_python_lists(x) for x in obj]
    else:
        # Return other objects as is
        return obj


# this is for conservative NN training tests


def binary_cross_entropy(preds, labels):
    epsilon = 1e-15  # To avoid log(0)
    preds = jnp.clip(preds, epsilon, 1 - epsilon)
    return -jnp.mean(labels * jnp.log(preds) + (1 - labels) * jnp.log(1 - preds))


def bce(data, nn, pars):
    # only need sig and bkg
    values = {k: data[k][:, 0, :] for k in ["NOSYS", "bkg"]}

    # apply the neural network to each data sample, and keep track of the
    # sample names in a dict
    nn_apply = partial(nn, pars)
    preds = {k: jax.vmap(nn_apply)(values[k]).ravel() for k in values}

    sig = preds["NOSYS"]
    bkg = preds["bkg"]
    labels = jnp.concatenate([jnp.ones_like(sig), jnp.zeros_like(bkg)])
    preds = jnp.concatenate((sig, bkg))

    return binary_cross_entropy(preds, labels)


# def bin_correction(bins):
#     bins = np.concatenate([[0], bins, [1]])
#     # calculate neighbor distance
#     diff = bins[1:] - bins[:-1]
#     # sort check
#     increasing = diff > 0
#     # check if they are some distance apart, to not break in next update step
#     # with config.lr = 1e-2 the bins are pulled around ~ 0.01
#     neighbor_distance = np.abs(diff) > 0.05
#     # since neighbor calc add the first one
#     increasing = np.append(True, increasing)
#     neighbor_distance = np.append(True, neighbor_distance)
#     # pop the one left to the one if the last is too close to 1
#     if neighbor_distance[-1] == False:
#         neighbor_distance[-2] = False
#     combined_condition = (bins < 1) & (bins > 0) & increasing & neighbor_distance
#     corrected_bins = bins[combined_condition]

#     # Ensure at least one bin remains after filtering.
#     return corrected_bins if corrected_bins.size > 0 else np.array([0.5])


# def bin_correction_(bins):
#     # make sure bins don't overlap and are unique, need to avoid loops and
#     # whatnot since this is a jitted function --> jnp.where

#     # # take care of out of bound
#     # bins = jnp.where(bins > 0, bins, 0.01)
#     # bins = jnp.where(bins < 1, bins, 0.99)
#     # find duplicates
#     is_not_duplicate = bins[1:] != bins[:-1]
#     # comparison does not include last value for condition
#     is_not_duplicate = jnp.concatenate((is_not_duplicate, jnp.array([True])))
#     # pad duplicates
#     unique_increment = jnp.arange(bins.size) * 0.001
#     # now return former values or pad if duplicate
#     bins = jnp.where(is_not_duplicate, bins, bins + unique_increment)
#     # monotonically increase

#     # calculate neighbor distance
#     diff = bins[1:] - bins[:-1]
#     # sort check
#     increasing = diff > 0
#     # check if they are some distance apart, to not break in next update step
#     # with config.lr = 1e-2 the bins are pulled around ~ 0.01
#     neighbor_distance = np.abs(diff) > 0.05
#     bins = jnp.sort(bins)

#     return bins


def delete_aux_data(config):
    # I know its bad to write data on config, but for now..., delete the data
    # from config for metadata writeout
    aux_data = []
    for k in config.__dict__.keys():
        if "bkg_estimate" in k:
            aux_data += [k]
    for k in aux_data:
        delattr(config, k)
    return config
