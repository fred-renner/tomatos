import jax.numpy as jnp
import numpy as np
import pyhf
import pprint
import tomatos.utils


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

    w_CR = A / B
    errA = jnp.sqrt(A)
    errB = jnp.sqrt(B)
    stat_err_w_CR = jnp.sqrt(jnp.square(errA / A) + jnp.square(errB / B))
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
        A=jnp.sum(hists["CR_btag_2"]["bkg"]["NOSYS"]),
        B=jnp.sum(hists["CR_btag_1"]["bkg"]["NOSYS"]),
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
    hists["bkg_estimate_NORM_up"] = hists["SR_btag_1"]["bkg"]["NOSYS"] * w_CR_stat_up
    hists["bkg_estimate_NORM_down"] = (
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


def get_modifiers(hists, config, fit_region):
    modifiers = {k: [] for k in config.samples}
    # autocollect 1UP 1DOWN
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

    # custom
    modifiers["bkg_estimate"] = [
        {
            "name": "bkg_estimate_norm",
            "type": "histosys",
            "data": {
                "hi_data": hists["bkg_estimate_NORM_up"],
                "lo_data": hists["bkg_estimate_NORM_down"],
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
            nom = hists["SR_btag_2"][sample][config.nominal]
            nom_up = jnp.copy(nom)
            stat_up_i = nom_up.at[i].set(
                hists["SR_btag_2"][sample][config.nominal + "_stat_up"][i]
            )
            nom_down = jnp.copy(nom)
            stat_down_i = nom_down.at[i].set(
                hists["SR_btag_2"][sample][config.nominal + "_stat_down"][i]
            )
            modifiers[sample] += (
                {
                    "name": f"stat_err_{sample}_{i+1}",
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
        stat_down_i = nom_down.at[i].set(hists["bkg_estimate_stat_down"][i])
        modifiers["bkg_estimate"] += (
            {
                "name": f"stat_err_bkg_{i+1}",
                "type": "histosys",
                "data": {
                    "hi_data": jnp.copy(stat_up_i),
                    "lo_data": jnp.copy(stat_down_i),
                },
            },
        )

    return modifiers


def get_auto_sample_spec(hists, config, fit_region, modifiers):
    # sample setup for all systematics
    no_auto_setup_samples = [config.signal_sample, "bkg"]
    auto_setup_samples = list(set(config.samples) - set(no_auto_setup_samples))
    # this list is empty here since these are the only two samples, but  auto
    # sets up the modifiers for all up down uncertainties
    auto_samples = [
        {
            "name": sample,
            "data": hists[fit_region][sample][config.nominal],
            "modifiers": modifiers[sample],
        }
        for sample in auto_setup_samples
    ]

    return auto_samples


def get_pyhf_model(
    hists,
    config,
):
    # attentive user action needed here

    # configurable?
    fit_region = "SR_btag_2"

    # standard uncerainty modifiers per sample
    # enforce 1UP, 1DOWN for autosetup here
    modifiers = get_modifiers(hists, config, fit_region)

    spec = {
        "channels": [
            {
                "name": fit_region,
                "samples": [
                    {
                        "name": config.signal_sample,
                        "data": hists[fit_region][config.signal_sample]["NOSYS"],
                        "modifiers": [
                            {
                                "name": "mu",
                                "type": "normfactor",
                                "data": None,
                            },  # our signal strength modifier (parameter of interest)
                            *modifiers[config.signal_sample],
                        ],
                    },
                    {
                        "name": "bkg_estimate",
                        "data": hists["bkg_estimate"],  # background
                        "modifiers": modifiers["bkg_estimate"],
                    },
                    *get_auto_sample_spec(hists, config, fit_region, modifiers),
                ],
            }
        ],
    }

    if config.debug:
        pprint.pprint(tomatos.utils.to_python_lists(spec))

    return pyhf.Model(spec, validate=False), hists
