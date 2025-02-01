import os
import numpy as np
import pathlib
import pyhf
import jax
from alive_progress import config_handler
import yaml

config_handler.set_global(enrich_print=False)

pyhf.set_backend("jax", default=True, precision="32b")

jax.config.update("jax_enable_x64", True)
# avoid some warnings on cpu
jax.config.update("jax_platforms", "cpu")
# more readable logging
jax.numpy.set_printoptions(precision=5, suppress=True, floatmode="fixed")

# some debugging options, these can be very useful!
# jax.numpy.set_printoptions(suppress=True)
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_check_tracer_leaks", True)
# useful to find the cause of nan's
# jax.config.update("jax_debug_nans", True)


# analysis_path = pathlib.Path(__file__).parent / "../tests/demo.yaml"
class Setup:
    def __init__(self, args):

        with open(args.config, "r") as f:
            yml = yaml.safe_load(f)

        # # just copy some config
        # simple_values = [
        #     "ntuple_path",
        #     "results_path",
        #     "batch_size",
        #     "nominal",
        #     "tree_name",
        #     "signal_sample",
        #     "compress_input_files",
        #     "train_ratio",
        #     "valid_ratio",
        #     "test_ratio",
        #     "plot_inputs",
        #     "objective",
        #     "cls_var",
        #     "num_steps",
        #     "bw_init",
        #     "bw_min",
        #     "slope",
        #     "signal_sample",
        #     "fit_region",
        #     "suffix",
        # ]
        for k, v in yml.items():
            setattr(self, k, v)

        # collect input files
        self.ntuple_paths = [
            str(file)
            for file in pathlib.Path(yml["ntuple_path"]).rglob("*.root")
            if file.is_file()
        ]
        # put nominal samples first
        self.ntuple_paths.sort(key=lambda x: (yml["nominal"] not in x))
        # expected structure: yml["ntuple_path"]/SAMPLE/SYSTEMATIC.root
        # sample_sys will be the list of SAMPLE_SYSTEMATIC
        self.samples = []
        # this defines the first dimension of the main data array
        self.sample_sys = []
        self.sample_sys_dict = {}
        for p in self.ntuple_paths:
            sample, systematic = p.removesuffix(".root").split("/")[-2:]
            if yml["nominal"] == systematic:
                self.samples += [sample]
            sample_sys = sample + "_" + systematic
            self.sample_sys += [sample_sys]
            self.sample_sys_dict[sample_sys] = (sample, systematic)

        # list indices can be accessed with e.g. self.samples.index("bkg")
        self.sample_files_dict = {
            k: v for k, v in zip(self.sample_sys, self.ntuple_paths)
        }
        # memory layout on disk per sample_sys, not in yaml
        self.n_chunk_combine = 2  # see batcher for this
        self.chunk_size = int(
            yml["batch_size"] / len(self.sample_sys) / self.n_chunk_combine
        )
        self.debug = args.debug

        # jax likes predefined arrays.
        # self.vars defines the main data array of dimension (n_samples, n_events, self.vars).
        # Need to keep event correspondence for weights, preprocessing,
        # batching,...
        # keeping them all together simplifies the program workflow a lot even
        # though it comes with something to care. order matters, see below.
        self.vars = [
            *yml["nn_input_vars"],
            yml["event_weight_var"],
            *yml["aux_vars"],
        ]
        # the last nn variable, in that order defines the nn inputs and also
        # the last variable that will be min max scaled
        # sliced array access is a huge speed up when slicing
        # array[:, :nn_inputs_idx_end] much faster than array[:, np.arange(nn_inputs_idx_end)]
        # + 1 to also include the given one when slicing
        self.nn_inputs_idx_end = len(yml["nn_input_vars"])
        self.weight_idx = self.vars.index(yml["event_weight_var"])
        self.cls_var_idx = self.vars.index(yml["cls_var"])

        # add the var index
        for cut_var in yml["opt_cuts"]:
            yml["opt_cuts"][cut_var]["idx"] = self.vars.index(cut_var)
        self.opt_cuts = yml["opt_cuts"]

        self.bins = np.linspace(0, 1, yml["bins"] + 1)

        # datapoints in kde plot
        self.kde_sampling = 1000  # its smooth, but can be expensive
        self.kde_bins = np.linspace(0, 1, self.kde_sampling)

        if args.debug:
            self.num_steps = 5 if yml["num_steps"] == 200 else yml["num_steps"]

        # nr of k-folds used for scaling the weights
        self.k_fold_sf = (
            yml["n_k_folds"] / (yml["n_k_folds"] - 1) if yml["n_k_folds"] > 1 else 1
        )

        # hists that contain these strings will be plotted
        self.plot_hists_filter = [self.signal_sample, "bkg_estimate"]
        # skip frames by modulo for movie
        self.movie_batch_modulo = 1
        self.fig_size = (9, 5)

        # Create output directories
        self._setup_paths(args, yml)

    def _setup_paths(self, args, yml):
        if self.suffix != "":
            results_folder = f"tomatos_{self.objective}_bins_{yml['bins']}_steps_{self.num_steps}_{self.suffix}/"
        else:
            results_folder = (
                f"tomatos_{self.objective}_bins_{self.bins}_steps_{self.num_steps}/"
            )
        results_folder = results_folder.replace(".", "p")
        if args.debug:
            results_folder = "tomatos_debug/"
        self.model = results_folder.split("/")[0]
        self.results_path += results_folder
        self.model_path = self.results_path + "models/"
        self.preprocess_path = self.results_path + "preprocessed/"
        self.plot_path = self.results_path + "plots/"
        self.gif_path = self.plot_path + "gif_images/"

        # make directories
        for path in [
            self.results_path,
            self.model_path,
            self.preprocess_path,
            self.plot_path,
            self.gif_path,
        ]:
            os.makedirs(path, exist_ok=True)

        self.preprocess_files = {
            "data": self.preprocess_path + "data.h5",
            "train": self.preprocess_path + "train.h5",
            "valid": self.preprocess_path + "valid.h5",
            "test": self.preprocess_path + "test.h5",
        }
        self.config_file_path = self.results_path + "config.json"
        self.preprocess_md_file_path = self.results_path + "preprocess_md.json"
        self.metrics_file_path = self.results_path + "metrics.h5"
        self.infer_metrics_file_path = self.results_path + "infer_metrics.json"
