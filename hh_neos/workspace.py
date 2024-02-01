import jax
import jax.numpy as jnp
import pyhf
import numpy as np

Array = jnp.ndarray


def make_stat_err(hist):
    stat_err_signal = jnp.sqrt(hist)
    hi = hist + stat_err_signal
    low = hist - stat_err_signal
    # make 0 if negative
    low = jnp.where(low > 0, low, 0)
    return {"hi": hi, "low": low}


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
def model_from_hists(do_m_hh, hists: dict[str, Array], config: object) -> pyhf.Model:
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
        stat_err_signal = make_stat_err(hists["NOSYS"])
        stat_err_bkg = make_stat_err(hists["bkg"])
        signal_modifiers += (
            {
                "name": "stat_err_signal",
                "type": "histosys",
                "data": {
                    "hi_data": stat_err_signal["hi"],
                    "lo_data": stat_err_signal["low"],
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
                                {
                                    "name": "stat_err_bkg",
                                    "type": "histosys",
                                    "data": {
                                        "hi_data": stat_err_bkg["hi"],
                                        "lo_data": stat_err_bkg["low"],
                                    },
                                },
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
