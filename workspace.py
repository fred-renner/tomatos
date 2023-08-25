import pyhf
import jax.numpy as jnp

Array = jnp.ndarray


# assume we give a dict of histograms with keys "sig", "bkg_nominal", "bkg_up",
# "bkg_down".
def model_from_hists(hists: dict[str, Array]) -> pyhf.Model:
    """How to make your HistFactory model from your histograms."""
    stat_err = jnp.sqrt(hists["bkg_nominal"])

    spec = {
        "channels": [
            {
                "name": "singlechannel",  # we only have one "channel" (data region)
                "samples": [
                    {
                        "name": "signal",
                        "data": hists["sig"],  # signal
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
                        "data": jnp.array(hists["bkg_nominal"]),  # background
                        "modifiers": [
                            {
                                "name": "multijet_shape_unc",
                                "type": "histosys",
                                "data": {
                                    "hi_data": jnp.array(
                                        hists["bkg_nominal"] * 1.1
                                    ),  # up sample
                                    "lo_data": jnp.array(
                                        hists["bkg_nominal"] * 0.9
                                    ),  # down sample
                                },
                            },
                            # {
                            #     "name": "multijet_stat_unc",
                            #     "type": "histosys",
                            #     "data": {
                            #         "hi_data": jnp.array(
                            #             hists["bkg_nominal"] + stat_err
                            #         ),  # up sample
                            #         "lo_data": jnp.array(
                            #             hists["bkg_nominal"] - stat_err
                            #         ),  # down sample
                            #     },
                            # },
                        ],
                    },
                    # {
                    #     "name": "ttbar",
                    #     "data": hists["ttbar"],  # background
                    #     "modifiers": [
                    #         {
                    #             "name": "ttbar_unc",
                    #             "type": "histosys",
                    #             "data": {
                    #                 "hi_data": jnp.array(
                    #                     hists["ttbar"] * 1.05
                    #                 ),  # up sample
                    #                 "lo_data": jnp.array(
                    #                     hists["ttbar"] * 0.95
                    #                 ),  # down sample
                    #             },
                    #         },
                    #     ],
                    # },
                ],
            },
        ],
    }

    return pyhf.Model(spec, validate=False)
