import jax.numpy as jnp
import numpy as np


class Setup:
    def __init__(self):
        self.do_m_hh = False
        self.include_bins = True
        self.vars = [
            "m_hh_NOSYS",
            "pt_hh_NOSYS",
            "eta_hh_NOSYS",
            "m_h1_NOSYS",
            "pt_h1_NOSYS",
            "eta_h1_NOSYS",
            "m_h2_NOSYS",
            "pt_h2_NOSYS",
            "eta_h2_NOSYS",
        ]

        if self.do_m_hh:
            self.vars = ["m_hh_NOSYS"]
        if self.include_bins:
            self.num_bins = 7
            self.bins = jnp.linspace(
                0, 1, self.num_bins + 1
            )  # keep in [0,1] if using sigmoid activation

        self.bandwidth = 0.2  # bandwidth ~ bin width is good choice

        self.region = "SR_xbb_2"
        self.plot_path = "/lustre/fs22/group/atlas/freder/hh/run/neos/"
        if self.do_m_hh and not self.include_bins:
            self.bins = np.array(
                [
                    -np.inf,
                    500e3,
                    900e3,
                    1080e3,
                    1220e3,
                    1360e3,
                    1540e3,
                    1800e3,
                    2500e3,
                    np.inf,
                ]
            )  # rel 21 analysis

        self.lr = 1e-2
        self.num_steps = 2
        # can choose from "CLs", "discovery", "poi_uncert" [approx. uncert. on mu], "bce" [classifier]
        self.objective = "cls"
        self.test_metric = "discovery"

        # the same keys you used in the model building step [model_from_hists]
        self.data_types = ["sig", "bkg_nominal"]  # , "ttbar"]
