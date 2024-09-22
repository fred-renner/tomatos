#! python3

import matplotlib.pyplot as plt
import numpy as np
import json

models = [
    "tomatos_cls_5_3000_study_11_lr_0p0001_slope_0500_k_1",
    "tomatos_cls_5_3000_study_11_lr_0p0001_slope_0750_k_1",
    "tomatos_cls_5_3000_study_11_lr_0p0001_slope_1000_k_1",
    "tomatos_cls_5_3000_study_11_lr_0p0001_slope_1250_k_1",
    "tomatos_cls_5_3000_study_11_lr_0p0001_slope_1500_k_1",
]

plt.rcParams.update({"font.size": 18})

plt.figure(figsize=(10, 6))

for m in models:
    model_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/" + m + "/"
    with open(model_path + "metadata.json", "r") as file:
        md = json.load(file)

    epoch_grid = range(1, md["config"]["num_steps"] + 1)
    no_fold_m = m.split("_k_")[0]
    aux_str = ", slope=" + no_fold_m.split("_")[-1]
    plt.plot(epoch_grid, md["metrics"]["cls_train"], label=r"$CL_s$ train" + aux_str)
    plt.plot(epoch_grid, md["metrics"]["cls_valid"], label=r"$CL_s$ valid" + aux_str)


plt.legend(prop={"size": 10}, ncols=2)  # prop={"size": 6})plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.ylim([0.005, np.max(md["metrics"]["cls_train"]) * 1.3])
plt.tight_layout()
plot_path = "/lustre/fs22/group/atlas/freder/hh/run/plots/compare_loss_slope.pdf"
print(plot_path)
plt.savefig(plot_path)
plt.close()


models = [
    "tomatos_cls_5_3000_study_11_lr_0p0001_bw_0p0050_k_1",
    "tomatos_cls_5_3000_study_11_lr_0p0001_bw_0p0075_k_1",
    "tomatos_cls_5_3000_study_11_lr_0p0001_bw_0p0250_k_1",
    "tomatos_cls_5_3000_study_11_lr_0p0001_bw_0p0500_k_1",
]

plt.figure(figsize=(10, 6))
for m in models:
    
    model_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/" + m + "/"
    with open(model_path + "metadata.json", "r") as file:
        md = json.load(file)
    n_bins = len(md["metrics"]["signal_approximation_diff"][0])
    total_sig_diff = np.sum(
        np.abs(np.array(md["metrics"]["signal_approximation_diff"]) - 1), axis=1
    )
    total_bkg_diff = np.sum(
        np.abs(np.array(md["metrics"]["bkg_approximation_diff"]) - 1), axis=1
    )
    no_fold_m = m.split("_k_")[0]

    aux_str = ", bw=" + no_fold_m.split("_")[-1]

    plt.plot(total_sig_diff, label=r"$\kappa_\mathrm{2V}=0$ signal" + aux_str)
    plt.plot(total_bkg_diff, label="Background Estimate" + aux_str)

plt.xlabel("Epoch")
plt.ylabel("Summed relative Binned KDE/Nominal")
plt.ylim([0, 1])
plt.legend()
plt.tight_layout()
plot_path = "/lustre/fs22/group/atlas/freder/hh/run/plots/compare_summed_diff.pdf"
print(plot_path)
plt.savefig(plot_path)
plt.close()

plt.figure(figsize=(10, 6))

for m in models:
    model_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/" + m + "/"
    with open(model_path + "metadata.json", "r") as file:
        md = json.load(file)

    epoch_grid = range(1, md["config"]["num_steps"] + 1)
    no_fold_m = m.split("_k_")[0]
    aux_str = ", bw=" + no_fold_m.split("_")[-1]
    plt.plot(epoch_grid, md["metrics"]["cls_train"], label=r"$CL_s$ train" + aux_str)
    plt.plot(epoch_grid, md["metrics"]["cls_valid"], label=r"$CL_s$ valid" + aux_str)


plt.legend(prop={"size": 10}, ncols=2)  # prop={"size": 6})plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.ylim([0.005, np.max(md["metrics"]["cls_train"]) * 1.3])
plt.tight_layout()
plot_path = "/lustre/fs22/group/atlas/freder/hh/run/plots/compare_loss_bw.pdf"
print(plot_path)
plt.savefig(plot_path)
plt.close()
