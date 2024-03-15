import os

import numpy as np


class Setup:
    def __init__(self, args):
        # fmt: off
        self.files = {
            "k2v0": "/lustre/fs22/group/atlas/freder/hh/run/dump/important_sys/dump-mc20_k2v0_70.h5",
            "run2": "/lustre/fs22/group/atlas/freder/hh/run/dump/important_sys/dump-run2_70.h5",
        }
        # fmt: on

        self.do_m_hh = False
        self.include_bins = False
        self.debug = True

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
            "xbb_pt_bin_0__1up",
            "xbb_pt_bin_0__1down",
            "xbb_pt_bin_1__1up",
            "xbb_pt_bin_1__1down",
            "xbb_pt_bin_2__1up",
            "xbb_pt_bin_2__1down",
            "xbb_pt_bin_3__1up",
            "xbb_pt_bin_3__1down",
            "JET_MassRes_Top__1up",
            "JET_MassRes_Hbb__1up",
            "JET_MassRes_WZ__1up",
            "JET_Rtrk_Modelling_pT__1up",
            "JET_Comb_Modelling_mass__1up",
            "JET_MassRes_Top__1down",
            "JET_MassRes_Hbb__1down",
            "JET_MassRes_WZ__1down",
            "JET_Rtrk_Modelling_pT__1down",
            "JET_Comb_Modelling_mass__1down",
            "JET_Flavor_Composition__1up",
            "JET_Flavor_Composition__1down",
            "GEN_MUR05_MUF05_PDF260000",
            "GEN_MUR05_MUF10_PDF260000",
            "GEN_MUR10_MUF05_PDF260000",
            "GEN_MUR10_MUF10_PDF260000",
            "GEN_MUR10_MUF20_PDF260000",
            "GEN_MUR20_MUF10_PDF260000",
            "GEN_MUR20_MUF20_PDF260000",
        ]

        self.systematics_raw = []
        self.do_stat_error = False
        self.do_systematics = True
        for sys in self.systematics:
            if "1up" in sys:
                self.systematics_raw += [sys.split("__")[0]]

        if self.do_m_hh:
            self.vars = ["m_hh"]

        self.n_features = len(self.vars)

        self.bins = np.linspace(0, 1, args.bins + 1)

        self.bandwidth = 0.2

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

        # with all systs + stats 0.001 seems too small
        self.lr = 0.01
        # one step is one batch, not epoch
        if self.debug:
            self.num_steps = 3
        else:
            self.num_steps = 400

        # share of data used for training vs testing
        self.train_data_ratio = 0.9

        # can choose from "cls", "discovery", "bce"
        self.objective = "cls"

        self.results_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/"

        if self.do_m_hh:
            results_folder = "tomatos_m_hh/"
        else:
            results_folder = f"tomatos_{self.objective}_{args.bins}_sys/"
        if self.debug:
            results_folder = "tomatos_debug/"
        self.results_path += results_folder
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)

        self.metadata_file_path = self.results_path + "metadata.json"
