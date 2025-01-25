#!/usr/bin/env python3
import argparse
import logging
import equinox as eqx
import jax
import tomatos.config
import tomatos.training
import tomatos.plotting
import tomatos.preprocess
import tomatos.utils
import tomatos.giffer
import json
import numpy as np
import tomatos.config


parser = argparse.ArgumentParser()
parser.add_argument("--bins", type=int, default=5)
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--k-fold", type=int, default=0)
parser.add_argument("--loss", type=str, default="cls")
parser.add_argument("--aux", type=float, default=1)
parser.add_argument("--aux-list", type=lambda s: [float(item) for item in s.split("_")])

args = parser.parse_args()


def run():
    config = tomatos.config.Setup(args)
    tomatos.utils.setup_logger(config)

    # if config.plot_inputs:
    #     logging.info("Plotting Inputs...")
    #     tomatos.plotting.plot_inputs(config)

    # logging.info("Preprocessing...")
    # # need to write scaler to json then load into config
    # tomatos.preprocess.run(config)

    logging.info("Training...")
    tomatos.training.run(config)

    # plot()
    # tomatos.giffer.run(config.model)


def plot():
    config = tomatos.configuration.Setup(args)
    results = {}
    with open(config.metadata_file_path, "r") as file:
        results = json.load(file)

    with open(config.metrics_file_path, "r") as file:
        metrics = json.load(file)
    tomatos.plotting.main(
        results["config"],
        results["bins"],
        results["yields"],
        metrics,
    )
