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


# assume we give a dict of histograms with keys "sig", "bkg_nominal", "bkg_up",
# "bkg_down".
def model_from_hists(
    do_m_hh,
    hists: dict[str, Array],
    config: object,
    do_systematics: bool,
    do_stat_error: bool,
) -> pyhf.Model:
    """How to make your HistFactory model from your histograms."""

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
                        # {
                        #     "name": "ttbar",
                        #     "data": hists["ttbar"],  # background
                        #     "modifiers": [],
                        # },
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

            bkg_modifiers += (
                {
                    "name": "norm_err_bkg",
                    "type": "histosys",
                    "data": {
                        "hi_data": hists["bkg"] * (1 + 0.05658641863973597),
                        "lo_data": hists["bkg"] * (1 - 0.05658641863973597),
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
                        # {
                        #     "name": "ttbar",
                        #     "data": hists["ttbar"],  # background
                        #     "modifiers": [],
                        # },
                    ],
                },
            ],
        }

    return pyhf.Model(spec, validate=False)
