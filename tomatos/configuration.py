import os
import numpy as np
import pathlib

import pyhf

pyhf.set_backend("jax", default=True, precision="32b")


class Setup:
    def __init__(self, args):

        self.run_bkg_init = False

        self.include_bins = False

        self.input_path = "/lustre/fs22/group/atlas/freder/hh/tomatos_inputs/"
        self.tree_name = "FilteredTree"
        # collect input files
        self.input_paths = [
            str(file)
            for file in pathlib.Path(self.input_path).rglob("*.root")
            if file.is_file()
        ]
        # put non-systematic samples first
        self.nominal = "NOSYS"
        self.input_paths.sort(key=lambda x: (self.nominal not in x))

        # expected structure: sample_path/SAMPLE/SYSTEMATIC.root
        # sample_sys will be the list of SAMPLE_SYSTEMATIC
        self.samples = []
        self.sample_sys = []

        for p in self.input_paths:
            sample_name = p.removesuffix(".root")
            sample_name = sample_name.split("/")
            sample = sample_name[-2]
            systematic = sample_name[-1]
            if self.nominal == systematic:
                self.samples += [sample]
            self.sample_sys += [sample + "_" + systematic]

        self.sample_files_dict = {
            k: v for k, v in zip(self.sample_sys, self.input_paths)
        }
        # total events that are batched in training from all samplesys combined
        self.batch_size = 1e6
        # memory layout on disk per sample_sys, not in yaml, see batcher for
        # factor 2
        self.chunk_size = int(self.batch_size / len(self.sample_sys) / 2)
        # slows down I/O, but saves disk memory
        self.compress_input_files = False
        # ratio need to add up to one
        self.splitting = {
            "train": {"ratio": 0.8},
            "valid": {"ratio": 0.1},
            "test": {"ratio": 0.1},
        }
        for k in self.splitting:
            self.splitting[k]["events"] = 0
            self.splitting[k]["preprocess_scale_factor"] = np.ones(len(self.sample_sys))
            self.splitting[k]["scale_factor"] = np.ones(len(self.sample_sys))

        self.plot_inputs = True
        self.debug = args.debug

        # jax likes predefined arrays.
        # self.vars defines the main data array ofdimension (n_events, self.vars).
        # Need to keep event correspondence for weights, preprocessing,
        # batching, hists,...
        # keeping them all together simplifies the program workflow a lot even
        # though it has to be setup with care. order matters, see below.
        # it also means that currently, except for histogram transformations,
        # vars created after the prepare step would be a bit tricky to input
        # scale and the nn input setup here would need to be changed. However I
        # can't think of an optimization calculation that couldn't be done
        # before opt.
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
            "weight",
            "bool_btag_1",
            "bool_btag_2",
            "weight_my_sf_unc_up",
            "weight_my_sf_unc_down",
        )
        # the last nn variable, in that order defines the nn inputs
        # sliced array access is a huge speed up when slicing
        # array[:, :nn_inputs_idx_end] much faster than array[:, np.arange(nn_inputs_idx_end)]
        # + 1  to also include the given one
        self.nn_inputs_idx_end = self.vars.index("h_m") + 1
        # nominal event weight
        self.weight_idx = self.vars.index("weight")

        # bce, cls_nn, cls_var (bins, cuts) in some variable
        self.objective = "cls_nn"
        # you can speed up cls_var, if you only setup var and the cut_vars in self.vars
        self.cls_var_idx = self.vars.index("h_m")

        # cuts on vars to be optimized, keep variables either "above", "below"
        # or below, start somewhere where the cut actually does something to
        # find a gradient
        self.opt_cuts = {
            "j1_pt": {
                "keep": "above",
                "idx": self.vars.index("j1_pt"),
                "init": 20_000,
            },  # (MeV)
            "j2_pt": {
                "keep": "above",
                "idx": self.vars.index("j2_pt"),
                "init": 20_000,
            },
        }

        # this should be a different yaml config
        # if args.loss == "bce":
        #     self.bw_init = 1e-100
        #     self.bw_min = 1e-100
        self.bw_init = 0.25
        self.bw_min = 1e-100

        self.slope = 20_000

        self.systematics_raw = []
        self.do_stat_error = False
        self.do_systematics = False

        self.bins = np.linspace(0, 1, args.bins + 1)

        # with all systs 0.001 seems too small
        self.lr = args.lr

        # one step is one batch, not epoch
        self.num_steps = args.steps

        # fixed bw
        # self.bw = np.full(self.bw.shape, 0.15)

        if args.debug:
            self.num_steps = 10 if args.steps == 200 else args.steps
            # self.bw = np.full(self.num_steps, 0.15)

        # cuts scaled to parameter range [0,1]
        self.cuts_init = 0.01
        # scale cut parameter to increase update steps
        self.cuts_factor = 1
        self.aux = float(args.aux)

        # nr of k-folds used for scaling the weights
        self.n_k_folds = 1
        self.k_fold_sf = (
            self.n_k_folds / (self.n_k_folds - 1) if self.n_k_folds > 1 else 1
        )

        # simple transfer factor or binned transferfactor
        self.binned_w_CR = False

        # if initialize parameters of a trained model
        self.preload_model = False
        if self.preload_model:
            self.preload_model_path = f"/lustre/fs22/group/atlas/freder/hh/run/tomatos/tomatos_cls_5_2000_lr_0p0001_bw_0p15_slope_100k_{args.k_fold}/neos_model.eqx"

        # paths
        self.results_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/"

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

        self.preprocess_path = self.results_path + "preprocessed/"
        self.preprocess_files = {
            "data": self.preprocess_path + "data.h5",
            "train": self.preprocess_path + "train.h5",
            "valid": self.preprocess_path + "valid.h5",
            "test": self.preprocess_path + "test.h5",
        }
        if not os.path.isdir(self.preprocess_path):
            os.makedirs(self.preprocess_path)
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)
            os.makedirs(self.model_path)

        self.metadata_file_path = self.results_path + "metadata.json"
        self.metrics_file_path = self.results_path + "metrics.json"

        self.best_epoch_results_path = self.results_path + "best_epoch_results.json"
