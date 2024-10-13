import jax.numpy as jnp
import numpy as np
import pyhf

Array = jnp.ndarray
from functools import partial


def get_generator_weight_envelope(hists):
    nominal = hists["NOSYS"]
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


def get_bkg_weight(hists, config):

    if config.binned_w_CR:
        # binned transferfactor
        CR_4b_Data = hists["bkg_CR_xbb_2"]
        CR_2b_Data = hists["bkg_CR_xbb_1"]
    else:
        # simple transferfactor
        CR_4b_Data = jnp.sum(hists["bkg_CR_xbb_2"])
        CR_2b_Data = jnp.sum(hists["bkg_CR_xbb_1"])

    errCR1 = jnp.sqrt(CR_4b_Data)
    errCR2 = jnp.sqrt(CR_2b_Data)

    # it really does not like literally empty bins
    CR_4b_Data += 1e-15
    CR_2b_Data += 1e-15
    w_CR = CR_4b_Data / CR_2b_Data
    err_w_CR = w_CR * jnp.sqrt(
        jnp.square(errCR1 / CR_4b_Data) + jnp.square(errCR2 / CR_2b_Data)
    )

    return w_CR, err_w_CR


def get_symmetric_up_down(nom, sys):
    relative = jnp.abs((nom - sys) / nom)
    up = 1 + relative
    down = 1 - relative

    up = jnp.where(up > 100, 100, up)
    down = jnp.where(down < 0, 0, down)

    return up, down


def threshold_uncertainty(h, threshold, a, find="below"):
    # even though a divergent behavior at low h is desirable, power law, exp
    # etc. were too aggressive for a stable training
    if find == "below":
        penalty = jnp.where(h < threshold, -a * (h - threshold) / h, 0)
    elif find == "above":
        penalty = jnp.where(h > threshold, a * (h - threshold) / h, 0)
    up = 1 + penalty
    down = 1 - penalty

    down = jnp.where(down < 0, 0, down)

    return up, down


def model_from_hists(
    do_m_hh,
    hists: dict[str, Array],
    config: object,
    do_systematics: bool,
    do_stat_error: bool,
    validate_only: bool,
) -> pyhf.Model:
    """How to make your HistFactory model from your histograms."""

    # bkg
    w_CR, err_w_CR = get_bkg_weight(hists, config)
    rel_err_w_CR = err_w_CR / w_CR
    hists["bkg"] *= w_CR
    if config.do_stat_error:
        hists["bkg_stat_up"] *= w_CR
        hists["bkg_stat_down"] *= w_CR

    hists["bkg_estimate_in_VR"] = hists["bkg_VR_xbb_1"] * w_CR

    # need to protect several times
    hists = {k: jnp.where(v < 0.01, 0.01, v) for k, v in hists.items()}
    hists["gen_up"], hists["gen_down"] = get_generator_weight_envelope(hists)

    bkg_shapesys_up, bkg_shapesys_down = get_symmetric_up_down(
        hists["bkg_estimate_in_VR"],
        hists["bkg_VR_xbb_2"],
    )
    hists["bkg_shape_sys_up"] = hists["bkg"] * bkg_shapesys_up
    hists["bkg_shape_sys_down"] = hists["bkg"] * bkg_shapesys_down

    # minimum counts via penalization
    bkg_protect_up, bkg_protect_down = threshold_uncertainty(
        hists["bkg"],
        threshold=1,
        a=config.aux,
        find="below",
    )
    hists["bkg_protect_up"] = hists["bkg"] * bkg_protect_up
    hists["bkg_protect_down"] = hists["bkg"] * bkg_protect_down

    bkg_vr_protect_up, bkg_vr_protect_down = threshold_uncertainty(
        hists["bkg_VR_xbb_2"],
        threshold=1,
        a=config.aux,
        find="below",
    )
    hists["bkg_vr_protect_up"] = hists["bkg"] * bkg_vr_protect_up
    hists["bkg_vr_protect_down"] = hists["bkg"] * bkg_vr_protect_down

    bkg_shape_sys_protect_up, bkg_shape_sys_protect_down = threshold_uncertainty(
        bkg_shapesys_up,
        threshold=2,
        a=config.aux,
        find="above",
    )

    hists["bkg_shape_sys_protect_up"] = hists["bkg"] * bkg_shape_sys_protect_up
    hists["bkg_shape_sys_protect_down"] = hists["bkg"] * bkg_shape_sys_protect_down

    # signal
    ps_up, ps_down = get_symmetric_up_down(hists["NOSYS"], hists["ps"])
    hists["ps_up"] = hists["NOSYS"] * ps_up
    hists["ps_down"] = hists["NOSYS"] * ps_down

    # minimum bin value otherwise optimization can fail
    # important that this happens after the last nominal hist creations
    hists = {k: jnp.where(v < 0.01, 0.01, v) for k, v in hists.items()}

    signal_modifiers = []
    bkg_modifiers = []
    if do_systematics:
        for sys in config.systematics_raw:
            signal_modifiers += (
                {
                    "name": sys,
                    "type": "histosys",
                    "data": {
                        "hi_data": hists[f"{sys}__1up"],
                        "lo_data": hists[f"{sys}__1down"],
                    },
                },
            )
        if "GEN_MUR05_MUF05_PDF260000" in config.systematics:
            signal_modifiers += (
                {
                    "name": "scale_variations",
                    "type": "histosys",
                    "data": {
                        "hi_data": hists["gen_up"],
                        "lo_data": hists["gen_down"],
                    },
                },
            )
        signal_modifiers += (
            {
                "name": "branching_ratio_bb",
                "type": "histosys",
                "data": {
                    "hi_data": hists["NOSYS"] * (1 + 0.034230167215544956),
                    "lo_data": hists["NOSYS"] * (1 - 0.03479541236132045),
                },
            },
            {
                "name": "ps",
                "type": "histosys",
                "data": {
                    "hi_data": hists["ps_up"],
                    "lo_data": hists["ps_down"],
                },
            },
        )

        bkg_modifiers += [
            {
                "name": "bkg_estimate_norm",
                "type": "histosys",
                "data": {
                    "hi_data": hists["bkg"] * (1 + rel_err_w_CR),
                    "lo_data": hists["bkg"] * (1 - rel_err_w_CR),
                },
            }
        ]
        # if config.binned_w_CR:
        #     for i in range(len(config.bins) - 1):
        #         # this .at.set makes a copy without altering the original!
        #         hists[f"bkg_shape_up_bin_{i}"] = (
        #             hists["bkg"].at[i].set(hists["bkg_protect_up"][i])
        #         )
        #         hists[f"bkg_shape_down_bin_{i}"] = (
        #             hists["bkg"].at[i].set(hists["bkg_protect_down"][i])
        #         )
        #         bkg_modifiers += (
        #             {
        #                 "name": f"bkg_estimate_shape_bin_{i}",
        #                 "type": "histosys",
        #                 "data": {
        #                     "hi_data": hists[f"bkg_shape_up_bin_{i}"],
        #                     "lo_data": hists[f"bkg_shape_down_bin_{i}"],
        #                 },
        #             },
        #         )
        # else:
        #     bkg_modifiers += (
        #         {
        #             "name": "bkg_estimate_shape",
        #             "type": "histosys",
        #             "data": {
        #                 "hi_data": hists["bkg_shape_sys_up"],
        #                 "lo_data": hists["bkg_shape_sys_down"],
        #             },
        #         },
        #     )

        if not validate_only:
            bkg_modifiers += (
                {
                    "name": "bkg_protect",
                    "type": "histosys",
                    "data": {
                        "hi_data": hists["bkg_protect_up"],
                        "lo_data": hists["bkg_protect_down"],
                    },
                },
                # {
                #     "name": "bkg_vr_protect",
                #     "type": "histosys",
                #     "data": {
                #         "hi_data": hists["bkg_vr_protect_up"],
                #         "lo_data": hists["bkg_vr_protect_down"],
                #     },
                # },
                # {
                #     "name": "bkg_shape_sys_protect",
                #     "type": "histosys",
                #     "data": {
                #         "hi_data": hists["bkg_shape_sys_protect_up"],
                #         "lo_data": hists["bkg_shape_sys_protect_down"],
                #     },
                # },
            )

    if config.do_stat_error:
        for i in range(len(config.bins) - 1):
            # this .at.set makes a copy without altering the original!
            hists[f"NOSYS_stat_up_bin_{i}"] = (
                hists["NOSYS"].at[i].set(hists["NOSYS_stat_up"][i])
            )
            hists[f"NOSYS_stat_down_bin_{i}"] = (
                hists["NOSYS"].at[i].set(hists["NOSYS_stat_down"][i])
            )
            signal_modifiers += (
                {
                    "name": "stat_err_signal",
                    "type": "histosys",
                    "data": {
                        "hi_data": hists[f"NOSYS_stat_up_bin_{i}"],
                        "lo_data": hists[f"NOSYS_stat_down_bin_{i}"],
                    },
                },
            )

            hists[f"bkg_stat_up_bin_{i}"] = (
                hists["bkg"].at[i].set(hists["bkg_stat_up"][i])
            )
            hists[f"bkg_stat_down_bin_{i}"] = (
                hists["bkg"].at[i].set(hists["bkg_stat_down"][i])
            )
            bkg_modifiers += (
                {
                    "name": f"stat_err_bkg_{i}",
                    "type": "histosys",
                    "data": {
                        "hi_data": hists[f"bkg_stat_up_bin_{i}"],
                        "lo_data": hists[f"bkg_stat_down_bin_{i}"],
                    },
                },
            )
    spec = {
        "channels": [
            {
                "name": "singlechannel",  # we only have one "channel" (data region)
                "samples": [
                    {
                        "name": "signal",
                        "data": hists["NOSYS"],  # signal
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
