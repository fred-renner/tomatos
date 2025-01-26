import os
import numpy as np
import pathlib
import pyhf
import jax

from alive_progress import config_handler

config_handler.set_global(enrich_print=False)

pyhf.set_backend("jax", default=True, precision="32b")

jax.config.update("jax_enable_x64", True)
# avoid some printing 
jax.config.update("jax_platforms", "cpu")
jax.numpy.set_printoptions(precision=5, suppress=True, floatmode="fixed")

# some debugging options
# jax.numpy.set_printoptions(suppress=True)
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_check_tracer_leaks", True)
# useful to find the cause of nan's
# jax.config.update("jax_debug_nans", True)


class Setup:
    def __init__(self, args):
        self.run_bkg_init = False

        self.include_bins = True

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
        # this defines the first dimension of the main data array
        self.sample_sys = []
        self.sample_sys_dict = {}
        for p in self.input_paths:
            sample, systematic = p.removesuffix(".root").split("/")[-2:]
            if self.nominal == systematic:
                self.samples += [sample]
            sample_sys = sample + "_" + systematic
            self.sample_sys += [sample_sys]
            self.sample_sys_dict[sample_sys] = (sample, systematic)
        # make them immutable
        self.samples = tuple(self.samples)
        self.sample_sys = tuple(self.sample_sys)

        # all of these lists also the lower define a index-mapping you can
        # acces with e.g. self.samples.index("bkg")
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
        self.train_ratio = 0.8
        self.valid_ratio = 0.1
        self.test_ratio = 0.1

        self.plot_inputs = True
        self.debug = args.debug

        # jax likes predefined arrays.
        # self.vars defines the main data array of dimension (samples, n_events, self.vars).
        # Need to keep event correspondence for weights, preprocessing,
        # batching, hists,...
        # keeping them all together simplifies the program workflow a lot even
        # though it has to be setup with care. order matters, see below.
        self.vars = [
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
            "my_sf_unc_up",
            "my_sf_unc_down",
        ]
        # the last nn variable, in that order defines the nn inputs and also
        # the last variable that will be min max scaled
        # sliced array access is a huge speed up when slicing
        # array[:, :nn_inputs_idx_end] much faster than array[:, np.arange(nn_inputs_idx_end)]
        # + 1 to also include the given one when slicing
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
                "init": 20_000,  # (MeV)
            },
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
        self.bw_init = 0.2
        self.bw_min = 0.01
        self.slope = 20_000
        self.bins = np.linspace(0, 1, args.bins + 1)

        self.signal_sample = "ggZH125_vvbb"
        self.fit_region = "SR_btag_2"
        self.kde_sampling = 1000

        # hists that contain these strings will be plotted
        self.plot_hists_filter = [self.fit_region, "bkg_estimate"]

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
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

        self.config_file_path = self.results_path + "config.json"
        self.preprocess_md_file_path = self.results_path + "preprocess_md.json"
        self.metrics_file_path = self.results_path + "metrics.h5"
        self.infer_metrics_file_path = self.results_path + "infer_metrics.json"

        self.best_epoch_results_path = self.results_path + "best_epoch_results.json"
