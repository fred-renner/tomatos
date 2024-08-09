import os

import numpy as np


class Setup:
    def __init__(self, args):
        # fmt: off
        self.files = {
            "k2v0": "/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_no_vbf_cut_vr_split/dump-l1cvv0cv1.h5",
            "run2": "/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_no_vbf_cut_vr_split/dump-run2.h5",
            "ps": "/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_no_vbf_cut_vr_split/dump-ps.h5",
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

        # norm.cdf in histogramming includes 1.0
        self.bins = np.linspace(0, 1, args.bins + 1)

        self.bandwidth = args.bw

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

        # Actual batching needs to implemented properly, e.g. account for in
        # weights...currently not needed
        self.batch_size = int(1e6)  # int is necessary

        # with all systs 0.001 seems too small
        self.lr = args.lr
        # one step is one batch, not epoch
        if self.debug:
            self.num_steps = 5
        else:
            self.num_steps = args.steps

        # slope parameter used by the sigmoid for cut optimization
        self.slope = args.slope
        # can choose from "cls", "discovery", "bce"
        self.objective = "cls"
        # promote minimum count for shape systematic estimate
        self.unc_estimate_min_count = args.unc_estimate_min_count
        # simple factor or binned transferfactor
        self.binned_w_CR = False

        # if initialize parameters of a trained model
        self.preload_model = False
        if self.preload_model:
            self.preload_model_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/tomatos_cls_5_500_slope_6400_lr_0p001_bw_0p2_slope_study_bkg_penalize_on/neos_model.eqx"

        # paths
        self.results_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/"
        if self.do_m_hh:
            results_folder = "tomatos_m_hh/"
        elif self.objective == "cls":
            results_folder = f"tomatos_{self.objective}_{args.bins}_{self.num_steps}_slope_{self.slope}_lr_{self.lr}_bw_{self.bandwidth}_no_bkg_shape/"
        elif self.objective == "bce":
            results_folder = (
                f"tomatos_{self.objective}_{args.bins}_{self.num_steps}_lr_{self.lr}/"
            )
        results_folder = results_folder.replace(".", "p")
        if self.debug:
            results_folder = "tomatos_debug/"
        self.results_path += results_folder
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)

        self.metadata_file_path = self.results_path + "metadata.json"
