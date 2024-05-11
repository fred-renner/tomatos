import jax
import jax.numpy as jnp
import numpy as np
import pyhf

Array = jnp.ndarray


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
    diffs = jnp.zeros((len(gens), nominal.shape[0]))
    for i, gen in enumerate(gens):
        diffs.at[i].set(jnp.abs(hists[gen] - nominal))
    max_diffs = jnp.max(diffs, axis=0)
    envelope_up = jnp.array(nominal + max_diffs)
    envelope_down = jnp.array(nominal - max_diffs)
    return envelope_up, envelope_down


def get_bkg_weight(hists):
    CR_4b_Data = jnp.sum(hists["bkg_CR_xbb_2"])
    CR_2b_Data = jnp.sum(hists["bkg_CR_xbb_1"])
    errCR1 = jnp.sqrt(CR_4b_Data)
    errCR2 = jnp.sqrt(CR_2b_Data)
    w_CR = CR_4b_Data / CR_2b_Data
    err_w_CR = w_CR * jnp.sqrt(jnp.square(errCR1 / CR_4b_Data) + jnp.square(errCR2 / CR_2b_Data))

    return w_CR, err_w_CR


def model_from_hists(
    do_m_hh,
    hists: dict[str, Array],
    config: object,
    do_systematics: bool,
    do_stat_error: bool,
) -> pyhf.Model:
    """How to make your HistFactory model from your histograms."""
    w_CR, err_w_CR = get_bkg_weight(hists)
    rel_err_w_CR = err_w_CR / w_CR
    hists["bkg"] *= w_CR
    # it really does not like literally empty bins 
    hists["bkg_VR_xbb_1"]+=1e-9 
    hists["bkg_VR_xbb_2"]+=1e-9
    bkg_estimate_validation = hists["bkg_VR_xbb_1"] * w_CR
    relative_bkg_validation = jnp.abs(
        (bkg_estimate_validation - hists["bkg_VR_xbb_2"]) / bkg_estimate_validation
    )
    relative_bkg_validation = jnp.where(
        jnp.isnan(relative_bkg_validation), 0, relative_bkg_validation
    )
    bkg_shapesys_up = 1 + relative_bkg_validation
    bkg_shapesys_down = 1 - relative_bkg_validation
    bkg_shapesys_down = jnp.where(bkg_shapesys_down < 0, 0, bkg_shapesys_down)


    print(hists["bkg_CR_xbb_2"])
    print(hists["bkg_CR_xbb_1"])
    print(hists["bkg_VR_xbb_2"])
    print(hists["bkg_VR_xbb_1"])
    print(w_CR)
    print(hists["bkg"])

    print(bkg_estimate_validation)
    print(hists["bkg_VR_xbb_2"])
    print(bkg_estimate_validation * bkg_shapesys_up)
    print(bkg_estimate_validation * bkg_shapesys_down)
    hists["bkg_shape_sys_up"]=bkg_estimate_validation * bkg_shapesys_up
    hists["bkg_shape_sys_down"]=bkg_estimate_validation * bkg_shapesys_down
    
    
    if do_m_hh:
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
                            ],
                        },
                        {
                            "name": "background",
                            "data": hists["bkg"],  # background
                            "modifiers": [],
                        },
                    ],
                },
            ],
        }

    else:
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
                gen_up, gen_down = get_generator_weight_envelope(hists)
                signal_modifiers += (
                    {
                        "name": "scale_variations",
                        "type": "histosys",
                        "data": {
                            "hi_data": gen_up,
                            "lo_data": gen_down,
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
            )

            bkg_modifiers += (
                {
                    "name": "bkg_estimate_norm",
                    "type": "histosys",
                    "data": {
                        "hi_data": hists["bkg"] * (1 + rel_err_w_CR),
                        "lo_data": hists["bkg"] * (1 - rel_err_w_CR),
                    },
                },
                {
                    "name": "bkg_estimate_shape",
                    "type": "histosys",
                    "data": {
                        "hi_data": hists["bkg_shape_sys_up"],
                        "lo_data": hists["bkg_shape_sys_down"],
                    },
                },
            )
        if do_stat_error:
            signal_modifiers += (
                {
                    "name": "stat_err_signal",
                    "type": "histosys",
                    "data": {
                        "hi_data": hists["NOSYS_stat_up"],
                        "lo_data": hists["NOSYS_stat_down"],
                    },
                },
            )
            bkg_modifiers += (
                {
                    "name": "stat_err_bkg",
                    "type": "histosys",
                    "data": {
                        "hi_data": hists["bkg_stat_up"],
                        "lo_data": hists["bkg_stat_down"],
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

    return pyhf.Model(spec, validate=False)
