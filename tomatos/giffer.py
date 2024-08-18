#! python3

# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter


# def update_plot(i, meta_data, ax, ymax):
#     ax.clear()
#     print(i)
#     for hist_name in meta_data["config"]["data_types"]:
#         if "JET" in hist_name or "GEN" in hist_name:
#             continue
#         ax.stairs(
#             edges=meta_data["config"]["bins"],
#             values=meta_data["metrics"][hist_name][i],
#             label=hist_name,
#             fill=None,
#             linewidth=1,
#         )

#     ax.set_ylim([0, ymax])
#     if meta_data["config"]["do_m_hh"]:
#         ax.set_xlabel("m$_{hh}$ (MeV)")
#     else:
#         ax.set_xlabel("NN score")

#     ax.text(
#         0.1,
#         0.95,
#         f"Epoch {i}",
#         horizontalalignment="center",
#         verticalalignment="center",
#         transform=ax.transAxes,
#     )
#     ax.set_ylabel("Events")
#     ax.legend(prop={"size": 5}, loc="upper right")
#     ax.figure.tight_layout()
#     plt.savefig(model_path + f"epoch_{i}.png")


# if __name__ == "__main__":
#     models = [
#         # "tomatos_cls_5_50",
#         # "tomatos_debug",
#         # "tomatos_cls_5_1000_fixed_cuts_m10",
#         "tomatos_cls_5_2500_slope_50",
#         # "tomatos_cls_5_1000_slope_500",
#         # "tomatos_cls_5_1000_slope_1000",
#     ]
#     ymax = 0
#     for m in models:
#         model_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/" + m + "/"
#         with open(model_path + "metadata.json", "r") as file:
#             meta_data = json.load(file)

#         meta_data["config"]["data_types"] += ["bkg_shape_sys_up", "bkg_shape_sys_down"]
#         meta_data["config"]["data_types"] += ["ps_up", "ps_down"]

#         for i in range(len(meta_data["metrics"]["NOSYS"])):
#             for hist_name in meta_data["config"]["data_types"]:
#                 if "JET" in hist_name or "GEN" in hist_name:
#                     continue
#                 values = meta_data["metrics"][hist_name][i]
#                 if np.max(values) > ymax:
#                     ymax = np.max(values)
#         ymax = 15
#         fig, ax = plt.subplots(figsize=(6, 4))
#         ani = FuncAnimation(
#             fig,
#             update_plot,
#             frames=range(len(meta_data["metrics"]["NOSYS"])),
#             fargs=(meta_data, ax, ymax),
#         )

#         gif_path = model_path + m + ".gif"
#         ani.save(gif_path, writer=PillowWriter(fps=60))

#         plt.close()

#         print(gif_path)

#! python3
import os
import imageio
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()


def create_gif_from_folder(folder_path, output_filename, duration=0.5):
    images = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            file_path = os.path.join(folder_path, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(output_filename, images, duration=duration)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 14})

    # models = ["tomatos_debug"]
    models = [args.model]
    ymax = 0
    for m in models:
        model_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/" + m + "/"
        with open(model_path + "metadata.json", "r") as file:
            meta_data = json.load(file)
        image_path = model_path + "/images"
        if not os.path.isdir(image_path):
            os.makedirs(image_path)
        # loop over epochs

        meta_data["config"]["data_types"] += ["bkg_shape_sys_up", "bkg_shape_sys_down"]
        meta_data["config"]["data_types"] += ["ps_up", "ps_down"]
        meta_data["config"]["data_types"].remove("ps")
        for i in range(len(meta_data["metrics"]["NOSYS"])):
            # if i == 5:
            #     break
            # if i != 1986:
            #     continue
            print(i)
            # loop over hists
            plt.figure(figsize=(10, 8))
            for hist_name in meta_data["config"]["data_types"]:

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
                ]
                if any([reg in hist_name for reg in bkg_regions]):
                    continue

                label = hist_name.replace("_", " ")

                if "bkg" == label:
                    label = "Background Estimate"
                if "NOSYS" == label:
                    label = r"$\kappa_\mathrm{2V}=0$ signal"
                if "ps up" == label:
                    label = "Parton Shower up"
                if "ps down" == label:
                    label = "Parton Shower down"

                plt.stairs(
                    edges=meta_data["config"]["bins"],
                    values=meta_data["metrics"][hist_name][i],
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

            if "kde_signal" in meta_data["metrics"]:
                plt.plot(
                    np.linspace(0, 1, len(meta_data["metrics"]["kde_signal"][0])),
                    meta_data["metrics"]["kde_signal"][i],
                    label="kde signal",
                    color="tab:orange",
                )
                plt.plot(
                    np.linspace(0, 1, len(meta_data["metrics"]["kde_bkg"][0])),
                    meta_data["metrics"]["kde_bkg"][i],
                    label="kde bkg",
                    color="tab:blue",
                )

            plt.stairs(
                edges=meta_data["config"]["bins"],
                values=meta_data["metrics"]["bkg"][i],
                fill=None,
                linewidth=2,
                color="tab:blue",
            )
            plt.stairs(
                edges=meta_data["config"]["bins"],
                values=meta_data["metrics"]["NOSYS"][i],
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
            plt.savefig(image_path + "/" + f"{i:004d}" + ".png", dpi=200)
            plt.close()

        create_gif_from_folder(image_path, model_path + m + ".gif", duration=0.0001)
        print(model_path + m + ".gif")
