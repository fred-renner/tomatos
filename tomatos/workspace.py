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


def per_bin(name, nom, hi, lo):
    # one histosys per bin, uncorrelated between bins. an ad-hoc stand-in
    # until there is a differentiable staterror/shapesys modifier, costs one
    # fit parameter per bin and uncertainty
    return [
        {
            "name": f"{name}_{i+1}",
            "type": "histosys",
            "data": {"hi_data": nom.at[i].set(hi[i]), "lo_data": nom.at[i].set(lo[i])},
        }
        for i in range(len(nom))
    ]


def stat_modifiers(hists, config, sample):
    h = hists[sample]
    return per_bin(f"STAT_{sample}", h[config.nominal], h["STAT_1UP"], h["STAT_1DOWN"])


def sys_modifiers(hists, config, sample):
    modifiers = []
    # autocollect 1UP 1DOWN shape uncertainties
    for sys in hists[sample]:
        if sys.endswith("_1UP") and not sys.startswith("STAT"):
            name = sys.removesuffix("_1UP")
            hi, lo = hists[sample][name + "_1UP"], hists[sample][name + "_1DOWN"]
            if name == "BKG_UNC":
                # example: the toy's background uncertainty is uncorrelated
                # between bins, one parameter per bin like stat
                modifiers += per_bin(f"{name}_{sample}", hists[sample][config.nominal], hi, lo)
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
