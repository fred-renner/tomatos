import os

import numpy as np


class Setup:
    def __init__(self, args):

        self.files = {
            "k2v0": f"/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_4_fold_k_{args.k_fold}/dump-l1cvv0cv1.h5",
            "run2": f"/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_4_fold_k_{args.k_fold}/dump-run2.h5",
            "ps": f"/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_4_fold_k_{args.k_fold}/dump-ps.h5",
        }

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

        # Actual batching needs a reimplementation
        self.batch_size = int(1e6)  # int is necessary

        # with all systs 0.001 seems too small
        self.lr = args.lr

        # one step is one batch, not epoch
        self.num_steps = args.steps

        if args.debug:
            self.num_steps = 10 if args.steps == 200 else args.steps

        # bw per epoch
        # also tried bw in optimization, basically linear decrease, but
        # unbounded, so manual
        self.decay_quantile = 0.8 # also used for slope 
        # self.bw = np.linspace(
        #     0.2,
        #     0.01,
        #     int(self.num_steps * self.decay_quantile),
        # )
        # # Pad the array to the desired size
        # self.bw = np.pad(self.bw, (0, self.num_steps - self.bw.size), mode="edge")
        # # fixed bw
        # # self.bw = np.full(self.bw.shape, 0.15)

        # can choose from "cls", "discovery", "bce"
        self.objective = args.loss
        # cuts scaled to parameter range [0,1]
        self.cuts_init = 0.001
        # scale cut parameter to speed up convergence on cuts
        self.cuts_factor = 1
        # nr of k-folds used for scaling the weights
        self.n_k_folds = 4
        # simple transfer factor or binned transferfactor
        self.binned_w_CR = False

        # promote minimum count for shape systematic estimate
        # 0 turns it off
        self.unc_estimate_min_count = args.unc_estimate_min_count

        # if initialize parameters of a trained model
        self.preload_model = False
        if self.preload_model:
            self.preload_model_path = f"/lustre/fs22/group/atlas/freder/hh/run/tomatos/tomatos_cls_5_2000_lr_0p0001_bw_0p15_slope_100k_{args.k_fold}/neos_model.eqx"

        # paths
        self.results_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/"
        if self.do_m_hh:
            results_folder = "tomatos_m_hh/"
        elif self.objective == "cls":
            # k_fold at end!
            if args.suffix != "":
                results_folder = f"tomatos_{self.objective}_{args.bins}_{self.num_steps}_lr_{self.lr}_{args.suffix}_k_{args.k_fold}/"
            else:
                results_folder = f"tomatos_{self.objective}_{args.bins}_{self.num_steps}_lr_{self.lr}_k_{args.k_fold}/"

            # results_folder = "tomatos_cls_5_500_slope_16000_lr_0p001_bw_0p16_k_1/"
        elif self.objective == "bce":
            results_folder = f"tomatos_{self.objective}_{args.bins}_{self.num_steps}_lr_{self.lr}_k_{args.k_fold}/"
        results_folder = results_folder.replace(".", "p")
        if self.debug:
            results_folder = "tomatos_debug/"
        self.results_path += results_folder
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)

        self.metadata_file_path = self.results_path + "metadata.json"

        self.best_epoch_results_path = self.results_path + "best_epoch_results.json"
