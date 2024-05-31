#!/usr/bin/env python3
import argparse
import logging
import equinox as eqx
import jax
import tomatos.batching
import tomatos.configuration
import tomatos.nn_setup
import tomatos.optimization
import tomatos.plotting
import tomatos.preprocess
import tomatos.utils
import json

jax.numpy.set_printoptions(precision=2, suppress=True, floatmode="fixed")
JAX_CHECK_TRACER_LEAKS = True
JAX_DEBUG_NANS=True 

jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument("--bins", type=int, required=True)
parser.add_argument("--debug", action="store_true", default=False)

args = parser.parse_args()


def run():
    config = tomatos.configuration.Setup(args)

    logging.basicConfig(
        filename=config.results_path + "log.txt",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger("pyhf").setLevel(logging.WARNING)
    logging.getLogger("relaxed").setLevel(logging.WARNING)
    logging.info(json.dumps(tomatos.utils.to_python_lists(config.__dict__), indent=4))

    data = tomatos.preprocess.prepare_data(config)
    logging.info(f"datasets: {len(data)}")
    init_pars, nn, nn_setup = tomatos.nn_setup.init(config)

    train, valid_test = tomatos.batching.split_data(
        data, ratio=config.train_valid_ratio
    )
    valid, test = tomatos.batching.split_data(valid_test, ratio=config.valid_test_ratio)
    # account for splitting with weights, so histograms have the same height,
    # even with less data
    train, valid, test = tomatos.batching.adjust_weights(config, train, valid, test)

    logging.info(f"train size: {train[0].shape[0]}")
    logging.info(f"valid size: {valid[0].shape[0]}")
    logging.info(f"test size: {test[0].shape[0]}")

    batch_iterator = tomatos.batching.make_iterator(train, batch_size=config.batch_size)

    best_params, metrics = tomatos.optimization.run(
        config=config,
        valid=valid,
        test=test,
        batch_iterator=batch_iterator,
        init_pars=init_pars,
        nn=nn,
    )

    bins, yields = tomatos.utils.get_hist(config, nn, best_params, data)
    config = tomatos.utils.delete_aux_data(config)

    results = {
        "config": tomatos.utils.to_python_lists(config.__dict__),
        "metrics": tomatos.utils.to_python_lists(metrics),
        "bins": tomatos.utils.to_python_lists(bins),
        "yields": tomatos.utils.to_python_lists(yields),
    }

    # save model to file
    model = eqx.combine(best_params["nn_pars"], nn_setup)
    eqx.tree_serialise_leaves(config.results_path + "neos_model.eqx", model)

    # save metadata
    with open(config.metadata_file_path, "w") as file:
        json.dump(results, file)
        logging.info(config.metadata_file_path)

    plot()


def plot():
    config = tomatos.configuration.Setup(args)
    results = {}
    with open(config.metadata_file_path, "r") as file:
        results = json.load(file)

    tomatos.plotting.plot_metrics(
        results["metrics"],
        results["config"],
    )
    tomatos.plotting.hist(
        results["config"],
        results["bins"],
        results["yields"],
    )


if __name__ == "__main__":
    # run()
    plot()
