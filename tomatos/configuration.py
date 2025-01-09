import os
import numpy as np
import pathlib


class Setup:
    def __init__(self, args):

        self.run_bkg_init = False

        self.do_m_hh = False
        self.include_bins = False

        if self.do_m_hh:
            self.files = {
                "signal": f"/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_trigger_sf/dump-l1cvv0cv1.h5",
                # "signal": f"/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_trigger_sf/dump-l1cvv1cv1.h5",
                # "signal": f"/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_trigger_sf/dump-l1cvv1p5cv1.h5",
                "k2v0": f"/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_trigger_sf/dump-l1cvv0cv1.h5",
                "run2": f"/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_trigger_sf/dump-run2.h5",
                "ps": f"/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_trigger_sf/dump-ps.h5",
            }
        else:
            self.files = {
                "signal": f"/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_4_fold_trigger_sf_k_{args.k_fold}/dump-l1cvv0cv1.h5",
                # "signal": f"/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_4_fold_trigger_sf_k_{args.k_fold}/dump-l1cvv1cv1.h5",
                # "signal": f"/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_4_fold_trigger_sf_k_{args.k_fold}/dump-l1cvv1p5cv1.h5",
                "k2v0": f"/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_4_fold_trigger_sf_k_{args.k_fold}/dump-l1cvv0cv1.h5",
                "run2": f"/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_4_fold_trigger_sf_k_{args.k_fold}/dump-run2.h5",
                "ps": f"/lustre/fs22/group/atlas/freder/hh/run/dump/tomatos_vars_4_fold_trigger_sf_k_{args.k_fold}/dump-ps.h5",
            }

        self.input_path = "/lustre/fs22/group/atlas/freder/hh/tomatos_inputs/"
        self.tree_name = "AnalysisMiniTree"

        # collect input files
        self.input_paths = [
            str(file)
            for file in pathlib.Path(self.input_path).rglob("*.root")
            if file.is_file()
        ]
        # put nominal samples first, not strictly necessary but useful for a couple of reasons
        self.input_paths.sort(key=lambda x: ("NOSYS" not in x))

        # expected structure: self.sample_path/SAMPLE/SYSTEMATIC.root
        self.sample_names = []
        for p in self.input_paths:
            sample_name = p.removesuffix(".root")
            sample_name = sample_name.split("/")
            self.sample_names += [sample_name[-2] + "_" + sample_name[-1]]

        self.plot_inputs = True
        self.debug = args.debug
        self.vars = (
            "j1_pt",
            "j1_eta",
            "j1_phi",
            "j1_m",
            "j2_pt",
            "j2_eta",
            "j2_phi",
            "j2_m",
            "h_pt",
            "h_eta",
            "h_phi",
            "h_m",
            "weights",
            "weights_sf_unc__1up",
            "weights_sf_unc__1down",
        )

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
        self.do_systematics = False
        for sys in self.systematics:
            if "1up" in sys:
                self.systematics_raw += [sys.split("__")[0]]

        self.n_features = len(self.vars)

        self.bins = np.linspace(0, 1, args.bins + 1)

        # k2v 1p5
        # total: 113251
        # k2v0
        # total: 84776
        self.batch_size = 1e6

        if self.do_m_hh:
            self.bw_init = 0.25
            self.bw_min = 1e-100
            self.batch_size = 10000
        elif args.loss == "bce":
            self.bw_init = 1e-100
            self.bw_min = 1e-100
        else:
            self.bw_init = 0.2
            self.bw_min = 0.001

        # self.slope = args.aux
        self.slope = 20_000

        # with all systs 0.001 seems too small
        self.lr = args.lr

        # one step is one batch, not epoch
        self.num_steps = args.steps

        # fixed bw
        # self.bw = np.full(self.bw.shape, 0.15)

        if args.debug:
            self.num_steps = 10 if args.steps == 200 else args.steps
            # self.bw = np.full(self.num_steps, 0.15)

        # can choose from "cls", "discovery", "bce"
        self.objective = args.loss
        # cuts scaled to parameter range [0,1]
        self.cuts_init = 0.01
        # scale cut parameter to increase update steps
        self.cuts_factor = 1
        self.aux = float(args.aux)

        # nr of k-folds used for scaling the weights
        # if not 1 -> somewhere
        self.n_k_folds = 2
        # simple transfer factor or binned transferfactor
        self.binned_w_CR = False

        # if initialize parameters of a trained model
        self.preload_model = False
        if self.preload_model:
            self.preload_model_path = f"/lustre/fs22/group/atlas/freder/hh/run/tomatos/tomatos_cls_5_2000_lr_0p0001_bw_0p15_slope_100k_{args.k_fold}/neos_model.eqx"

        # paths
        self.results_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/"
        if self.do_m_hh:
            results_folder = "tomatos_m_hh/"
        # k_fold at end!
        if args.suffix != "":
            results_folder = f"tomatos_{self.objective}_{args.bins}_{self.num_steps}_{args.suffix}_k_{args.k_fold}/"
        else:
            results_folder = f"tomatos_{self.objective}_{args.bins}_{self.num_steps}_k_{args.k_fold}/"

        # results_folder = "tomatos_cls_5_500_slope_16000_lr_0p001_bw_0p16_k_1/"
        results_folder = results_folder.replace(".", "p")
        if self.debug:
            results_folder = "tomatos_debug/"
        self.model = results_folder.split("/")[0]
        self.results_path += results_folder
        self.model_path = self.results_path + "models/"

        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)
            os.makedirs(self.model_path)

        self.metadata_file_path = self.results_path + "metadata.json"
        self.metrics_file_path = self.results_path + "metrics.json"

        self.best_epoch_results_path = self.results_path + "best_epoch_results.json"
