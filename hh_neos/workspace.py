import jax
import jax.numpy as jnp
import pyhf

Array = jnp.ndarray


def make_stat_err(hist):
    stat_err_signal = jnp.sqrt(hist)
    hi = hist + stat_err_signal
    low = hist - stat_err_signal
    # make 0 if negative
    low = jnp.where(low > 0, low, 0)
    return {"hi": hi, "low": low}


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
        stat_err_signal = make_stat_err(hists["NOSYS"])
        stat_err_bkg = make_stat_err(hists["bkg"])

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
                                # {
                                #     "name": "signal_stat",
                                #     "type": "histosys",
                                #     "data": {
                                #         "hi_data": stat_err_signal["hi"],
                                #         "lo_data": stat_err_signal["low"],
                                #     },
                                # },
                            ],
                        },
                        {
                            "name": "background",
                            "data": hists["bkg"],  # background
                            "modifiers": [
                                {
                                    "name": f"xbb_pt_bin_0",
                                    "type": "histosys",
                                    "data": {
                                        "hi_data": hists[
                                            f"xbb_pt_bin_0__1up"
                                        ],  # up sample
                                        "lo_data": hists[
                                            f"xbb_pt_bin_0__1down"
                                        ],  # down sample
                                    },
                                },
                                {
                                    "name": f"xbb_pt_bin_1",
                                    "type": "histosys",
                                    "data": {
                                        "hi_data": hists[
                                            f"xbb_pt_bin_1__1up"
                                        ],  # up sample
                                        "lo_data": hists[
                                            f"xbb_pt_bin_1__1down"
                                        ],  # down sample
                                    },
                                },
                                {
                                    "name": f"xbb_pt_bin_2",
                                    "type": "histosys",
                                    "data": {
                                        "hi_data": hists[
                                            f"xbb_pt_bin_2__1up"
                                        ],  # up sample
                                        "lo_data": hists[
                                            f"xbb_pt_bin_2__1down"
                                        ],  # down sample
                                    },
                                },
                                {
                                    "name": f"xbb_pt_bin_3",
                                    "type": "histosys",
                                    "data": {
                                        "hi_data": hists[
                                            f"xbb_pt_bin_3__1up"
                                        ],  # up sample
                                        "lo_data": hists[
                                            f"xbb_pt_bin_3__1down"
                                        ],  # down sample
                                    },
                                },
                                
                                # {
                                #     "name": "signal_stat",
                                #     "type": "histosys",
                                #     "data": {
                                #         "hi_data": stat_err_bkg["hi"],
                                #         "lo_data": stat_err_bkg["low"],
                                #     },
                                # },
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
