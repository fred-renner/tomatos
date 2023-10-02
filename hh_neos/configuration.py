import jax.numpy as jnp
import numpy as np
import os


class Setup:
    def __init__(self):
        # fmt: off
        self.files = {
            "SM": "/lustre/fs22/group/atlas/freder/hh/run/dump/m_hh_all_sys/dump-mc20_SM.h5",
            "k2v0": "/lustre/fs22/group/atlas/freder/hh/run/dump/m_hh_all_sys/dump-mc20_k2v0.h5",
            "ttbar": "/lustre/fs22/group/atlas/freder/hh/run/dump/m_hh_all_sys/dump-mc20_ttbar.h5",
            "run2": "/lustre/fs22/group/atlas/freder/hh/run/dump/m_hh_all_sys/dump-run2.h5",
        }
        # fmt: on

        self.do_m_hh = False
        self.include_bins = True
        self.debug = True

        self.vars = [
            "pt_h1",
            "eta_h1",
            # "phi_h1",
            # "m_h1",
            "pt_h2",
            "eta_h2",
            # "phi_h2",
            # "m_h2",
            "pt_hh",
            "eta_hh",
            # "phi_hh",
            "m_hh",
            "pt_j1",
            "eta_j1",
            # "phi_j1",
            "E_j1",
            "pt_j2",
            "eta_j2",
            # "phi_j2",
            "E_j2",
        ]

        self.systematics = [
            "NOSYS",
            "xbb_pt_bin_0__1up",
            "xbb_pt_bin_0__1down",
            "xbb_pt_bin_1__1up",
            "xbb_pt_bin_1__1down",
            "xbb_pt_bin_2__1up",
            "xbb_pt_bin_2__1down",
            "xbb_pt_bin_3__1up",
            "xbb_pt_bin_3__1down",
            # "GEN_MUR05_MUF05_PDF260000",
            # "GEN_MUR05_MUF10_PDF260000",
            # "GEN_MUR10_MUF05_PDF260000",
            # "GEN_MUR10_MUF10_PDF260000",
            # "GEN_MUR10_MUF20_PDF260000",
            # "GEN_MUR20_MUF10_PDF260000",
            # "GEN_MUR20_MUF20_PDF260000",
        ]

        if self.do_m_hh:
            self.vars = ["m_hh"]

        self.n_features = len(self.vars)

        if self.include_bins:
            self.num_bins = 8
            # keep in [0,1] if using sigmoid activation
            self.bins = jnp.linspace(0, 1, self.num_bins)

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
