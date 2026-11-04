import pprint

import jax.numpy as jnp
import pyhf


def zero_protect(hists, thresh=0.001):
    # opt and fit do not like zeros/tiny numbers
    # go recursively through all hists and replace values with thresh if below
    if isinstance(hists, dict):
        return {key: zero_protect(value, thresh) for key, value in hists.items()}

    if isinstance(hists, jnp.ndarray):
        return jnp.where(hists < thresh, thresh, hists)


def stat_modifiers(hists, config, sample):
    # this an ad-hoc hack for stat uncertainty, and nothing more than
    # that until we have the diffable modifier. Blows up the number of fit
    # parameters unnecessarily for stat unc, which instead of having one
    # parameter per bin, histosy makes a parameter for each bin per
    # modifier
    nom = hists[sample][config.nominal]
    modifiers = []
    for i in range(len(nom)):
        modifiers += [
            {
                "name": f"STAT_{sample}_{i+1}",
                "type": "histosys",
                "data": {
                    "hi_data": nom.at[i].set(hists[sample]["STAT_1UP"][i]),
                    "lo_data": nom.at[i].set(hists[sample]["STAT_1DOWN"][i]),
                },
            }
        ]
    return modifiers


def sys_modifiers(hists, config, sample):
    modifiers = []
    # autocollect 1UP 1DOWN shape uncertainties
    for sys in hists[sample]:
        if sys.endswith("_1UP") and not sys.startswith("STAT"):
            name = sys.removesuffix("_1UP")
            hi, lo = hists[sample][name + "_1UP"], hists[sample][name + "_1DOWN"]
            if config.weight_systematics.get(name, {}).get("bin_by_bin"):
                # uncorrelated between bins, e.g. the statistics of a data
                # driven estimate: one parameter per bin, as for stat
                nom = hists[sample][config.nominal]
                modifiers += [
                    {
                        "name": f"{name}_{sample}_{i+1}",
                        "type": "histosys",
                        "data": {"hi_data": nom.at[i].set(hi[i]), "lo_data": nom.at[i].set(lo[i])},
                    }
                    for i in range(len(nom))
                ]
                continue
            modifiers += [
                {
                    "name": name,
                    "type": "histosys",
                    "data": {"hi_data": hi, "lo_data": lo},
                }
            ]
    # normalization uncertainties, e.g. cross sections
    if sample in config.norm_systematics:
        unc = config.norm_systematics[sample]
        modifiers += [
            {
                "name": f"NORM_{sample}",
                "type": "normsys",
                "data": {"hi": 1 + unc, "lo": 1 - unc},
            }
        ]
    return modifiers


def pyhf_model(hists, config, stat_only=False):
    # here you build the json workspace structure with hists
    # stat_only: what an analysis ignoring the systematics would see, the
    # A/B check of what that costs
    samples = []
    for sample in config.samples:
        modifiers = []
        if sample == config.signal_sample:
            # signal strength modifier (parameter of interest)
            modifiers += [{"name": "mu", "type": "normfactor", "data": None}]
        if not stat_only:
            modifiers += sys_modifiers(hists, config, sample)
        modifiers += stat_modifiers(hists, config, sample)
        samples += [
            {
                "name": sample,
                "data": hists[sample][config.nominal],
                "modifiers": modifiers,
            }
        ]

    # this is the workspace json
    spec = {"channels": [{"name": "fit_region", "samples": samples}]}

    # # this is very handy for debugging when you turn off jit in the main.py
    # if config.debug:
    #     pprint.pprint(spec)

    return pyhf.Model(spec, validate=False)
