import jax
import jax.numpy as jnp
import pyhf

Array = jnp.ndarray


# assume we give a dict of histograms with keys "sig", "bkg_nominal", "bkg_up",
# "bkg_down".
def model_from_hists(do_m_hh, hists: dict[str, Array]) -> pyhf.Model:
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
        # stat_err = jnp.sqrt(hists["bkg_nominal"])
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
                            "modifiers": [
                                {
                                    "name": f"xbb_pt_bin_{bin}",
                                    "type": "histosys",
                                    "data": {
                                        "hi_data": hists[
                                            f"xbb_pt_bin_{bin}__1up"
                                        ],  # up sample
                                        "lo_data": hists[
                                            f"xbb_pt_bin_{bin}__1down"
                                        ],  # down sample
                                    },
                                }
                                for bin in [0, 1, 2, 3]
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
