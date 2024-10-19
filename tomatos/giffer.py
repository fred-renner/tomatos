#! python3
import imageio
import matplotlib.pyplot as plt
import json
import numpy as np
import glob
import os


# def create_gif_from_folder(folder_path, output_filename, duration=0.5):
#     images = []
#     for file_name in sorted(os.listdir(folder_path)):
#         if file_name.endswith(".png") or file_name.endswith(".jpg"):
#             file_path = os.path.join(folder_path, file_name)
#             images.append(imageio.imread(file_path))
#     imageio.mimsave(output_filename, images, duration=duration)


def run(model=None):
    plt.rcParams.update({"font.size": 14})

    if model:
        models = [model]
    else:
        models = [
            "tomatos_cls_5_1000_study_9_simple_lr_decay_0p01_0p0005_0p0001_k_0",
        ]
    ymax = 0
    for m in models:
        model_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/" + m + "/"
        with open(model_path + "metadata.json", "r") as file:
            meta_data = json.load(file)

        with open(model_path + "metrics.json", "r") as file:
            metrics = json.load(file)
        image_path = model_path + "/images"
        if not os.path.isdir(image_path):
            os.makedirs(image_path)
        # loop over epochs

        # meta_data["config"]["data_types"] += ["bkg_shape_sys_up", "bkg_shape_sys_down"]
        # meta_data["config"]["data_types"] += ["ps_up", "ps_down"]
        # meta_data["config"]["data_types"] += ["gen_up", "gen_down"]
        # meta_data["config"]["data_types"].remove("ps")
        # # meta_data["config"]["data_types"] = ["NOSYS","bkg"]

        for i in range(len(metrics["NOSYS"])):
            if i % 1 != 0:
                continue
            # if i != 9999:
            #     continue
            print(i)
            # loop over hists
            # plt.figure(figsize=(10, 8))
            plt.figure(figsize=(5, 5))
            for hist_name, hist in metrics.items():
                hist = np.array(hist)
                if hist.ndim != 2:
                    continue

                bkg_regions = [
                    "bkg_CR_xbb_1",
                    "bkg_CR_xbb_2",
                    "bkg_VR_xbb_1",
                    "bkg_VR_xbb_2",
                    "bkg_VR_xbb_1_NW",
                    "bkg_VR_xbb_2_NW",
                    "bkg_stat_up",
                    "bkg_stat_down",
                    "bkg_stat_up",
                    "bkg_stat_down",
                    "NOSYS_stat_up",
                    "NOSYS_stat_down",
                    "kde",
                    "test",
                    "diff",
                    "bkg_estimate_in_VR",
                    "bins",
                    "VR",
                ]
                if any([reg in hist_name for reg in bkg_regions]):
                    continue

                if "GEN" in hist_name:
                    continue
                # if "protect" in hist_name:
                #     continue

                # if "xbb" in hist_name:
                #     continue
                label = hist_name.replace("_", " ")

                if "bkg" == label:
                    label = "Background Estimate"
                if "NOSYS" == label:
                    label = r"$\kappa_\mathrm{2V}=0$ signal"
                if "ps up" == label:
                    label = "Parton Shower up"
                if "ps down" == label:
                    label = "Parton Shower down"
                if "gen up" == label:
                    label = "Scale Varations up"
                if "gen down" == label:
                    label = "Scale Varations down"

                edges = (
                    metrics["bins"][i]
                    if meta_data["config"]["include_bins"]
                    else meta_data["config"]["bins"]
                )
                plt.stairs(
                    edges=edges,
                    values=metrics[hist_name][i],
                    # values=metrics[hist_name + "_test"][i], # does not match
                    # with kde!
                    label=label,
                    fill=None,
                    linewidth=1,
                    # align="edge",
                )

                ax = plt.gca()
                ylim = ax.get_ylim()
                if ylim[1] > ymax:
                    ymax = ylim[1] * 1.3

                ax.set_ylim([0, ymax])

            if meta_data["config"]["do_m_hh"]:
                plt.xlabel("m$_{hh}$ (MeV)")
            else:
                plt.xlabel("NN score")

            if meta_data["config"]["objective"] == "cls":
                plt.plot(
                    np.linspace(0, 1, len(metrics["kde_signal"][0]))[1:-1],
                    metrics["kde_signal"][i][1:-1],
                    label="kde signal",
                    color="tab:orange",
                )
                plt.plot(
                    np.linspace(0, 1, len(metrics["kde_bkg"][0]))[1:-1],
                    metrics["kde_bkg"][i][1:-1],
                    label="kde bkg",
                    color="tab:blue",
                )

            plt.stairs(
                edges=edges,
                values=metrics["bkg"][i],
                fill=None,
                linewidth=2,
                color="tab:blue",
            )
            plt.stairs(
                edges=edges,
                values=metrics["NOSYS"][i],
                fill=None,
                linewidth=2,
                color="tab:orange",
            )

            plt.title(f"Epoch {i}")
            plt.ylabel("Events")
            plt.legend(
                prop={"size": 5}, ncols=3, loc="upper center"
            )  # prop={"size": 6})
            plt.tight_layout()
            plt.savefig(image_path + "/" + f"{i:005d}" + ".png", dpi=192)
            plt.close()

        # gif
        # create_gif_from_folder(image_path, model_path + m + ".mp4", duration=0.0001)
        # print(model_path + m + ".gif")

        # movie

        writer = imageio.get_writer(model_path + m + ".mp4", fps=20)
        for file in sorted(glob.glob(os.path.join(image_path, f"*.png"))):
            im = imageio.imread(file)
            writer.append_data(im)
        print(model_path + m + ".mp4")


if __name__ == "__main__":
    run()
