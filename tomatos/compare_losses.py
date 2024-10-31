#! python3

import matplotlib.pyplot as plt
import numpy as np
import json

models = [
    "tomatos_cls_5_2000_study_2_lr_0p0005_bw_min_0p005_slope_1000_k_0",
    "tomatos_cls_5_2000_study_2_lr_0p0005_bw_min_0p005_slope_5000_k_0",
    "tomatos_cls_5_2000_study_2_lr_0p0005_bw_min_0p005_slope_10000_k_0",
    "tomatos_cls_5_2000_study_2_lr_0p0005_bw_min_0p005_slope_20000_k_0",
    # "tomatos_cls_5_2000_study_2_lr_0p0005_bw_min_0p005_slope_50000_k_0",
    # "tomatos_cls_5_2000_study_2_lr_0p0005_bw_min_0p005_slope_100000_k_0",
]

plt.rcParams.update({"font.size": 18})

plt.figure(figsize=(10, 6))

for m in models:
    model_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/" + m + "/"
    with open(model_path + "metrics.json", "r") as file:
        metrics = json.load(file)

    epoch_grid = range(1, len(metrics["cls_train"]) + 1)
    no_fold_m = m.split("_k_")[0]
    aux_str = ", slope=" + no_fold_m.split("_")[-1]
    plt.plot(epoch_grid, metrics["cls_train"], label=r"$CL_s$ train" + aux_str)
    plt.plot(epoch_grid, metrics["cls_valid"], label=r"$CL_s$ valid" + aux_str)


plt.legend(prop={"size": 10}, ncols=2)  # prop={"size": 6})plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.ylim([0.005, np.max(metrics["cls_train"]) * 1.3])
plt.tight_layout()
plot_path = "/lustre/fs22/group/atlas/freder/hh/run/plots/compare_loss_slope.pdf"
print(plot_path)
plt.savefig(plot_path)
plt.close()


models = [
    "tomatos_cls_5_2000_study_1_lr_0p0005_bw_min_0p001_slope_5000_k_0",
    "tomatos_cls_5_2000_study_1_lr_0p0005_bw_min_0p005_slope_5000_k_0",
    "tomatos_cls_5_2000_study_1_lr_0p0005_bw_min_0p01_slope_5000_k_0",
]

plt.figure(figsize=(10, 6))
for m in models:

    model_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/" + m + "/"
    with open(model_path + "metrics.json", "r") as file:
        metrics = json.load(file)
    n_bins = len(metrics["signal_approximation_diff"][0])

    total_sig_diff = (
        np.max(np.abs(np.array(metrics["signal_approximation_diff"]) - 1), axis=1) * 100
    )
    total_bkg_diff = (
        np.max(np.abs(np.array(metrics["bkg_approximation_diff"]) - 1), axis=1) * 100
    )
    no_fold_m = m.split("_slope_5000_k_0")[0]

    aux_str = "bw=" + no_fold_m.split("_")[-1] + ", "

    # plt.plot(
    #     total_sig_diff,
    #     label=aux_str + r"$\kappa_\mathrm{2V}=0$ signal",
    #     alpha=0.7,
    # )
    plt.plot(total_bkg_diff / n_bins, label=aux_str + "Background Estimate", alpha=0.75)

plt.xlabel("Epoch")
plt.ylabel("Largest Bin Deviation (%)")
plt.ylim([0, 10])
plt.legend()
plt.tight_layout()
plot_path = "/lustre/fs22/group/atlas/freder/hh/run/plots/compare_summed_diff.pdf"
print(plot_path)
plt.savefig(plot_path)
plt.close()

plt.figure(figsize=(10, 6))

for m in models:
    model_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/" + m + "/"
    with open(model_path + "metrics.json", "r") as file:
        metrics = json.load(file)

    epoch_grid = range(1, len(metrics["cls_train"]) + 1)
    no_fold_m = m.split("_slope_5000_k_0")[0]
    aux_str = ", bw=" + no_fold_m.split("_")[-1]
    plt.plot(epoch_grid, metrics["cls_train"], label=r"$CL_s$ train" + aux_str)
    plt.plot(epoch_grid, metrics["cls_valid"], label=r"$CL_s$ valid" + aux_str)


plt.legend(prop={"size": 10}, ncols=2)  # prop={"size": 6})plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.ylim([0.005, np.max(metrics["cls_train"]) * 1.3])
plt.tight_layout()
plot_path = "/lustre/fs22/group/atlas/freder/hh/run/plots/compare_loss_bw.pdf"
print(plot_path)
plt.savefig(plot_path)
plt.close()
