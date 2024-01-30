import jax.numpy as jnp
import pyhf
import relaxed
import sys
import numpy as np
import hh_neos.histograms
from functools import partial
import jax

Array = jnp.ndarray


class Logger(object):
    def __init__(self, config):
        self.terminal = sys.stdout
        self.log = open(config.results_path + "log.txt", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def get_hist(config, nn, best_params, data):
    if config.include_bins:
        bins = jnp.array([0, *best_params["bins"], 1])
        print(best_params["bins"])
    else:
        bins = config.bins
    if config.do_m_hh:
        # use whole data set to get correct norm
        yields = hh_neos.histograms.hists_from_mhh(
            data={k: v for k, v in zip(config.data_types, data)},
            bandwidth=1e-8,
            bins=bins,
        )

    else:
        # to get correct yields would also need to pass whole data
        yields = hh_neos.histograms.hists_from_nn(
            pars=best_params["nn_pars"],
            data={k: v + 1e-8 for k, v in zip(config.data_types, data)},
            nn=nn,
            bandwidth=1e-8,
            bins=jnp.array([0, *best_params["bins"], 1])
            if config.include_bins
            else config.bins,
        )
    print(
        bins[1:-1],
    )
    print(yields)
    print(
        "Asimov Significance: ",
        relaxed.metrics.asimov_sig(s=yields["NOSYS"], b=yields["bkg"]),
    )

    return bins, yields


def print_cls(config, yields):
    model = hh_neos.workspace.model_from_hists(config, yields)

    CLs_obs, CLs_exp = pyhf.infer.hypotest(
        1.0,  # null hypothesis
        model.expected_data([0, 0.0]),
        model,
        test_stat="q",
        return_expected_set=True,
    )
    print(f"      Observed CLs: {CLs_obs:.6f}")
    for expected_value, n_sigma in zip(CLs_exp, np.arange(-2, 3)):
        print(f"Expected CLs({n_sigma:2d} σ): {expected_value:.6f}")


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


def sigmoid_cross_entropy_with_logits(preds, labels):
    return jnp.mean(
        jnp.maximum(preds, 0) - preds * labels + jnp.log1p(jnp.exp(-jnp.abs(preds)))
    )


def bce(data, nn, pars):
    values = {k: data[k][:, 0, :] for k in data}

    # apply the neural network to each data sample, and keep track of the
    # sample names in a dict
    nn_apply = partial(nn, pars)
    preds = {k: jax.vmap(nn_apply)(values[k]).ravel() for k in values}
    print(preds.keys())
    bkg = preds["bkg"]
    sig = preds["NOSYS"]
    # I have no clue why learning only works with opposite labels?
    labels = jnp.concatenate([jnp.zeros_like(bkg),jnp.ones_like(sig)])
    preds =  jnp.concatenate((sig, bkg))

    return sigmoid_cross_entropy_with_logits(preds, labels).mean()




# def bce(data, nn, pars):
#     preds = {k: nn(pars, data[k]).ravel() for k in data}
#     bkg = jnp.concatenate([preds[k] for k in preds if "sig" not in k])
#     sig = preds["sig"]
#     labels = jnp.concatenate([jnp.ones_like(sig), jnp.zeros_like(bkg)])
#     return sigmoid_cross_entropy_with_logits(
#         jnp.concatenate(list(preds.values())).ravel(), labels
#     ).mean()
