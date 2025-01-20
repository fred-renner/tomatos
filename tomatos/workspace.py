import jax.numpy as jnp
import numpy as np
import pyhf

Array = jnp.ndarray
from functools import partial


def get_generator_weight_envelope(hists):
    # adapt to the way you've set this up
    nominal = hists["blah"]
    gens = [
        "GEN_MUR05_MUF05_PDF260000",
        "GEN_MUR05_MUF10_PDF260000",
        "GEN_MUR10_MUF05_PDF260000",
        "GEN_MUR10_MUF10_PDF260000",
        "GEN_MUR10_MUF20_PDF260000",
        "GEN_MUR20_MUF10_PDF260000",
        "GEN_MUR20_MUF20_PDF260000",
    ]

    gen_hists = jnp.array([hists[gen] for gen in gens])
    diffs = jnp.abs(gen_hists - nominal)
    max_diffs = jnp.max(diffs, axis=0)
    envelope_up = jnp.array(nominal + max_diffs)
    envelope_down = jnp.array(nominal - max_diffs)
    return envelope_up, envelope_down


def get_abcd_weight(A, B):
    w_CR = jnp.sum(A) / jnp.sum(B)
    errA = jnp.sqrt(A)
    errB = jnp.sqrt(B)
    stat_err_w_CR = w_CR * jnp.sqrt(jnp.square(errA / A) + jnp.square(errB / B))
    return w_CR, stat_err_w_CR


def symmetric_up_down_sf(nom, sys):
    relative = jnp.abs((nom - sys) / nom)
    up = 1 + relative
    down = 1 - relative
    # limit to some extent
    up = jnp.where(up > 100, 100, up)
    down = jnp.where(down < 0, 0, down)

    return up, down


def zero_protect(hists, thresh=0.001):
    # opt and fit do not like zeros/tiny numbers
    # go recursively through all hists and replace values with thresh if below
    if isinstance(hists, dict):
        return {key: zero_protect(value, thresh) for key, value in hists.items()}

    if isinstance(hists, jnp.ndarray):
        return jnp.where(hists < thresh, thresh, hists)


def hist_transforms(hists):

    # to also protect for e.g. divisions in the following
    hists = zero_protect(hists)

    # aim to scale btag_1 to btag_2 in SR from ratio in CR
    w_CR, stat_err_w_CR = get_abcd_weight(
        A=hists["CR_btag_2"]["bkg"]["NOSYS"],
        B=hists["CR_btag_1"]["bkg"]["NOSYS"],
    )
    # scale single tagged for bkg estimate
    hists["bkg_estimate"] = hists["SR_btag_1"]["bkg"]["NOSYS"] * w_CR

    # current hack until proper diffable norm/stat modifier
    hists["bkg_estimate_stat_up"] = hists["SR_btag_1"]["bkg"]["NOSYS_stat_up"] * w_CR
    hists["bkg_estimate_stat_down"] = (
        hists["SR_btag_1"]["bkg"]["NOSYS_stat_down"] * w_CR
    )

    # normilzation uncertainty background estimate
    w_CR_stat_up, w_CR_stat_down = symmetric_up_down_sf(w_CR, w_CR + stat_err_w_CR)
    hists["SR_btag_2"]["bkg"]["NORM_up"] = (
        hists["SR_btag_1"]["bkg"]["NOSYS"] * w_CR_stat_up
    )
    hists["SR_btag_2"]["bkg"]["NORM_down"] = (
        hists["SR_btag_1"]["bkg"]["NOSYS"] * w_CR_stat_down
    )

    # this is bad, as opt sculpts unc for kinematics in VR
    # which is not an unbiased unc proxy for the SR, maybe only apply to valid,
    # and testing? too little events?
    # may be also illustrative to have here?
    # fit afterwards in train loop without opt?
    # hists["bkg_estimate_VR"] = hists["VR_btag_1"]["bkg_NOSYS"] * w_CR
    # bkg_shapesys_up, bkg_shapesys_down = symmetric_up_down_sf(
    #     nom=hists["bkg_estimate_in_VR"],
    #     sys=hists["VR_btag_2"]["bkg_NOSYS"],
    # )
    # hists["bkg_shape_sys_up"] = hists["bkg_estimate_SR"] * bkg_shapesys_up
    # hists["bkg_shape_sys_down"] = hists["bkg_estimate_SR"] * bkg_shapesys_down

    # e.g. if generatorweights are available
    # hists["gen_up"], hists["gen_down"] = get_generator_weight_envelope(hists)

    # make sure again after transforms!
    hists = zero_protect(hists)

    return hists


def model_from_hists(
    hists,
    config,
):
    # attentive user action needed here

    hists = hist_transforms(hists)

    # standard uncerainty modifiers per sample
    # enforce 1UP, 1DOWN for autosetup here
    fit_region = "SR_btag_2"

    modifiers = {k: [] for k in config.samples}
    for sample in hists[fit_region]:
        for sys in hists[fit_region][sample]:
            if "1UP" in sys:
                sys_ = sys.replace("1UP", "")
                modifiers[sample] += (
                    {
                        "name": sys,
                        "type": "histosys",
                        "data": {
                            "hi_data": hists[fit_region][sample][sys_ + "1UP"],
                            "lo_data": hists[fit_region][sample][sys_ + "1DOWN"],
                        },
                    },
                )

    modifiers["bkg"] += [
        {
            "name": "bkg_estimate_norm",
            "type": "histosys",
            "data": {
                "hi_data": hists["SR_btag_2"]["bkg"]["NORM_up"],
                "lo_data": hists["SR_btag_2"]["bkg"]["NORM_down"],
            },
        }
    ]

    # this an ad-hoc hack for stat uncertainty, and nothing more than
    # that until we have the diffable modifier. Blows up the number of fit
    # parameters unnecessarily for stat unc, which instead of having one
    # parameter per bin, histosys makes a parameter for each bin and each
    # modifier
    for i in jnp.arange(len(config.bins) - 1):
        for sample in config.samples:
            nom = hists["SR_btag_2"][sample][config.nominal + "_stat_up"]
            nom_up = jnp.copy(nom)
            stat_up_i = nom_up.at[i].set(hists["bkg_estimate_stat_up"][i])
            nom_down = jnp.copy(nom)
            stat_down_i = nom_down.at[i].set(hists["bkg_estimate_stat_up"][i])
            modifiers[sample] += (
                {
                    "name": f"stat_err_{sample}_{i}",
                    "type": "histosys",
                    "data": {
                        "hi_data": jnp.copy(stat_up_i),
                        "lo_data": jnp.copy(stat_down_i),
                    },
                },
            )

        nom_up = jnp.copy(hists["bkg_estimate"])
        stat_up_i = nom_up.at[i].set(hists["bkg_estimate_stat_up"][i])
        nom_down = jnp.copy(hists["bkg_estimate"])
        stat_down_i = nom_down.at[i].set(hists["bkg_estimate_stat_up"][i])
        modifiers["bkg"] += (
            {
                "name": f"stat_err_bkg_{i}",
                "type": "histosys",
                "data": {
                    "hi_data": jnp.copy(stat_up_i),
                    "lo_data": jnp.copy(stat_down_i),
                },
            },
        )

    spec = {
        "channels": [
            {
                "name": "singlechannel",  # we only have one "channel" (data region)
                "samples": [
                    {
                        "name": "ggZH125_vvbb",
                        "data": hists["ggZH125_vvbb"],  # signal
                        "modifiers": [
                            {
                                "name": "mu",
                                "type": "normfactor",
                                "data": None,
                            },  # our signal strength modifier (parameter of interest)
                            *signal_modifiers,
                        ],
                    },
                    {
                        "name": "background",
                        "data": hists["bkg"],  # background
                        "modifiers": [
                            *bkg_modifiers,
                        ],
                    },
                ],
            },
        ],
    }

    return pyhf.Model(spec, validate=False), hists
