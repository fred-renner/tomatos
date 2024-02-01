import os

import numpy as np


class Setup:
    def __init__(self, args):
        # fmt: off
        self.files = {
            # "SM": "/lustre/fs22/group/atlas/freder/hh/run/dump/important_sys/dump-mc20_SM.h5",
            "k2v0": "/lustre/fs22/group/atlas/freder/hh/run/dump/important_sys/dump-mc20_k2v0.h5",
            # "ttbar": "/lustre/fs22/group/atlas/freder/hh/run/dump/important_sys/dump-mc20_ttbar.h5",
            "run2": "/lustre/fs22/group/atlas/freder/hh/run/dump/important_sys/dump-run2.h5",
        }
        # fmt: on

        self.do_m_hh = False
        self.include_bins = True
        self.debug = False

        # # rel 21
        # self.vars = [
        #     "pt_h1",
        #     "eta_h1",
        #     # "phi_h1",
        #     # "m_h1",
        #     "pt_h2",
        #     "eta_h2",
        #     # "phi_h2",
        #     # "m_h2",
        #     "pt_hh",
        #     "eta_hh",
        #     # "phi_hh",
        #     "m_hh",
        #     "pt_j1",
        #     "eta_j1",
        #     # "phi_j1",
        #     "E_j1",
        #     "pt_j2",
        #     "eta_j2",
        #     # "phi_j2",
        #     "E_j2",
        # ]
        self.vars = [
            "m_hh",
            "pt_h1",
            "eta_h1",
            "phi_h1",
            "m_h1",
            "pt_h2",
            "eta_h2",
            "phi_h2",
            "m_h2",
            "pt_hh",
            "eta_hh",
            "phi_hh",
            "pt_j1",
            "eta_j1",
            "phi_j1",
            "E_j1",
            "pt_j2",
            "eta_j2",
            "phi_j2",
            "E_j2",
        ]

        self.systematics = [
            "NOSYS",
            # "xbb_pt_bin_0__1up",
            # "xbb_pt_bin_0__1down",
            # "xbb_pt_bin_1__1up",
            # "xbb_pt_bin_1__1down",
            # "xbb_pt_bin_2__1up",
            # "xbb_pt_bin_2__1down",
            # "xbb_pt_bin_3__1up",
            # "xbb_pt_bin_3__1down",
            # "JET_MassRes_Top__1up",
            # "JET_MassRes_Hbb__1up",
            # "JET_MassRes_WZ__1up",
            # "JET_Rtrk_Modelling_pT__1up",
            # "JET_Comb_Modelling_mass__1up",
            # "JET_MassRes_Top__1down",
            # "JET_MassRes_Hbb__1down",
            # "JET_MassRes_WZ__1down",
            # "JET_Rtrk_Modelling_pT__1down",
            # "JET_Comb_Modelling_mass__1down",
            # "JET_Flavor_Composition__1up",
            # "JET_Flavor_Composition__1down",
            # "GEN_MUR05_MUF05_PDF260000",
            # "GEN_MUR05_MUF10_PDF260000",
            # "GEN_MUR10_MUF05_PDF260000",
            # "GEN_MUR10_MUF10_PDF260000",
            # "GEN_MUR10_MUF20_PDF260000",
            # "GEN_MUR20_MUF10_PDF260000",
            # "GEN_MUR20_MUF20_PDF260000",
        ]

        self.systematics_raw = []

        for sys in self.systematics:
            if "1up" in sys:
                self.systematics_raw += [sys.split("__")[0]]

        if self.do_m_hh:
            self.vars = ["m_hh"]

        self.n_features = len(self.vars)

        self.bins = np.linspace(0, 1, args.bins + 1)

        # some bad bin settings for testing
        # self.bins = np.array([0, 0.061, 0.142, 0.889, 1, 1])
        # self.bins = np.array([0, 0.0594, 0.133, 0.975, 0.969, 1])
        # bad_edges = np.linspace(0.9900, 0.9999, args.bins - 1)
        # bad_edges = np.insert(bad_edges, 0, 0)
        # edges = np.append(bad_edges, 0.9)
        # edges = np.array([0.2, 0.3, 0.35, 0.95])

        # self.bins = np.array(edges)

        # bandwidth ~ bin width is a good choice
        self.bandwidth = 0.2

        self.region = "SR_xbb_2"
        self.results_path = "/lustre/fs22/group/atlas/freder/hh/run/"

        if self.do_m_hh:
            results_folder = "neos_m_hh/"
        else:
            results_folder = f"neos_nn_{args.bins}_bins/"
        if self.debug:
            results_folder = "neos_debug/"
        self.results_path += results_folder
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)

        self.metadata_file_path = self.results_path + "metadata.json"

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

        self.batch_size = int(1e5)  # int is necessary
        self.lr = 1e-2
        # one step is one batch, not epoch
        if self.debug:
            self.num_steps = 10
        else:
            self.num_steps = 400

        # share of data used for training vs testing
        self.train_data_ratio = 0.8

        # can choose from "CLs", "discovery", "poi_uncert" [approx. uncert. on mu], "bce" [classifier]
        self.objective = "cls"
