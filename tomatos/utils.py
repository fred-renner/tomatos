import gc
import json
import logging
import os
import pprint
import sys
from collections import namedtuple
from functools import partial

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import psutil
import pyhf
import h5py

import tomatos.histograms
import tomatos.training
import tomatos.workspace


def inverse_min_max_scale(config, arr, var_idx):
    unscaled_arr = (arr - config.scaler_min[var_idx]) / config.scaler_scale[var_idx]
    return unscaled_arr


def flatten_dict(nested_dict):
    # e.g.
    # hists[sel][sample][sys] -->
    # hists[sel_sample_sys]
    flat_dict = {}

    def recurse(d, parent_key=""):
        for key, value in d.items():
            new_key = f"{parent_key}_{key}" if parent_key else key
            if isinstance(value, dict):
                recurse(value, new_key)
            elif isinstance(value, jnp.ndarray):
                flat_dict[new_key] = value

    recurse(nested_dict)
    return flat_dict


def filter_hists(config, hists):
    hists = flatten_dict(hists)

    filtered_hists = {}
    for h_key, h in hists.items():
        if any([f in h_key for f in config.plot_hists_filter]):
            filtered_hists[h_key] = h

    return filtered_hists


def clear_caches(config):
    # clear jax compilation cache for larger than 16 GB, note that this does
    # not easily translate to rss
    # https://github.com/google/jax/issues/10828
    vms_gb = psutil.Process().memory_info().vms / (2**30)
    if (vms_gb - config.initial_vms_gb) > config.memory_limit_gb:
        for module_name, module in sys.modules.items():
            if module_name.startswith("jax"):
                for obj_name in dir(module):
                    obj = getattr(module, obj_name)
                    if hasattr(obj, "cache_clear"):
                        obj.cache_clear()
    gc.collect()


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
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
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
