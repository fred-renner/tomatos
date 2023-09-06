import jax.numpy as jnp
import numpy as np
import os


class Setup:
    def __init__(self):
        self.files = {
            "SM": "/lustre/fs22/group/atlas/freder/hh/run/dump/bdt_vars/dump-mc20_SM.h5",
            "k2v0": "/lustre/fs22/group/atlas/freder/hh/run/dump/bdt_vars/dump-mc20_k2v0.h5",
            "ttbar": "/lustre/fs22/group/atlas/freder/hh/run/dump/bdt_vars/dump-mc20_ttbar.h5",
            "run2": "/lustre/fs22/group/atlas/freder/hh/run/dump/bdt_vars/dump-run2.h5",
        }

        self.do_m_hh = True
        self.include_bins = False
        self.debug = False
        self.vars = [
            "pt_h1_NOSYS",
            "eta_h1_NOSYS",
            # "phi_h1_NOSYS",
            # "m_h1_NOSYS",
            "pt_h2_NOSYS",
            "eta_h2_NOSYS",
            # "phi_h2_NOSYS",
            # "m_h2_NOSYS",
            "pt_hh_NOSYS",
            "eta_hh_NOSYS",
            # "phi_hh_NOSYS",
            "m_hh_NOSYS",
            "pt_j1_NOSYS",
            "eta_j1_NOSYS",
            # "phi_j1_NOSYS",
            "E_j1_NOSYS",
            "pt_j2_NOSYS",
            "eta_j2_NOSYS",
            # "phi_j2_NOSYS",
            "E_j2_NOSYS",
        ]

        if self.do_m_hh:
            self.vars = ["m_hh_NOSYS"]
        if self.include_bins:
            self.num_bins = 7
            self.bins = jnp.linspace(
                0, 1, self.num_bins + 1
            )  # keep in [0,1] if using sigmoid activation

        # bandwidth ~ bin width is a good choice
        self.bandwidth = 0.2

        self.region = "SR_xbb_2"
        self.results_path = "/lustre/fs22/group/atlas/freder/hh/run/"

        if self.do_m_hh:
            results_folder = "neos_m_hh/"
        else:
            results_folder = "neos_nn/"
        if self.debug:
            results_folder = "neos_debug/"
        self.results_path += results_folder
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)

        self.results_file_path = self.results_path + "/saved_results.pkl"

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
        if self.debug:
            self.num_steps = 2
        else:
            self.num_steps = 300

        # can choose from "CLs", "discovery", "poi_uncert" [approx. uncert. on mu], "bce" [classifier]
        self.objective = "cls"

        # the same keys you used in the model building step [model_from_hists]
        self.data_types = ["sig", "bkg_nominal"]  # , "ttbar"]
