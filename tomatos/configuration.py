import os

import numpy as np


class Setup:
    def __init__(self, args):
        # fmt: off
        self.files = {
            "k2v0": "/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_no_vbf_cut/dump-l1cvv0cv1.h5",
            "run2": "/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_no_vbf_cut/dump-run2.h5",
            "ps": "/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_no_vbf_cut/dump-ps.h5",
        }
        # fmt: on

        self.do_m_hh = False
        self.include_bins = False
        self.debug = args.debug
        self.vars = [
            "pt_j1",
            "eta_j1",
            "phi_j1",
            "E_j1",
            "pt_j2",
            "eta_j2",
            "phi_j2",
            "E_j2",
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
            "m_hh",
            "lead_xbb_score",
            "m_jj",
            "eta_jj",
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
            # "JET_EtaIntercalibration_NonClosure_PreRec__1up",
            # "JET_EtaIntercalibration_Modelling__1up",
            # "JET_EtaIntercalibration_NonClosure_PreRec__1down",
            # "JET_EtaIntercalibration_Modelling__1down",
            "GEN_MUR05_MUF05_PDF260000",
            "GEN_MUR05_MUF10_PDF260000",
            "GEN_MUR10_MUF05_PDF260000",
            "GEN_MUR10_MUF10_PDF260000",
            "GEN_MUR10_MUF20_PDF260000",
            "GEN_MUR20_MUF10_PDF260000",
            "GEN_MUR20_MUF20_PDF260000",
        ]

        self.systematics_raw = []
        self.do_stat_error = True
        self.do_systematics = True
        for sys in self.systematics:
            if "1up" in sys:
                self.systematics_raw += [sys.split("__")[0]]

        if self.do_m_hh:
            self.vars = ["m_hh"]

        self.n_features = len(self.vars)

        self.bins = np.linspace(0, 1, args.bins + 1)

        # 0.2 seems optimal, smaller results in nan's at some point
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

        # with all systs 0.001 seems too small
        self.lr = 0.01
        # one step is one batch, not epoch
        if self.debug:
            self.num_steps = 10
        else:
            self.num_steps = 2500

        # share of data used for training vs testing
        self.train_valid_ratio = 0.9
        self.valid_test_ratio = 0.8

        # can choose from "cls", "discovery", "bce"
        self.objective = "cls"

        # if initialize parameters of a trained model
        self.preload_model = False
        self.preload_model_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/tomatos_cls_5_500/neos_model.eqx"

        # paths
        self.results_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/"
        if self.do_m_hh:
            results_folder = "tomatos_m_hh/"
        else:
            results_folder = (
                f"tomatos_{self.objective}_{args.bins}_{self.num_steps}_slope_50/"
            )
        if self.debug:
            results_folder = "tomatos_debug/"
        self.results_path += results_folder
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)
        self.metadata_file_path = self.results_path + "metadata.json"
