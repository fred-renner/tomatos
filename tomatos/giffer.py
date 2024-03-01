#! python3
import os
import imageio
import matplotlib.pyplot as plt
import json
import os
import numpy as np


def create_gif_from_folder(folder_path, output_filename, duration=0.5):
    images = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            file_path = os.path.join(folder_path, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(output_filename, images, duration=duration)


if __name__ == "__main__":
    models = [
        # "m_hh",
        "atos_bce_3",
        "atos_bce_4",
        "atos_bce_5",
        # "atos_cls_3_blank",
        # "atos_cls_3_sys",
        # "atos_cls_3_stat_sys",
        # "atos_cls_4_blank",
        # "atos_cls_4_sys",
        # "atos_cls_4_stat_sys",
        # "atos_cls_5_blank",
        # "atos_cls_5_sys",
        # "atos_cls_5_stat_sys",
    ]
    ymax = 0
    for m in models:
        model_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/" + m + "/"
        with open(model_path + "metadata.json", "r") as file:
            meta_data = json.load(file)
        image_path = model_path + "/images"
        if not os.path.isdir(image_path):
            os.makedirs(image_path)
        for i in range(len(meta_data["metrics"]["hist_sig"])):
            plt.figure()
            plt.stairs(
                edges=meta_data["config"]["bins"],
                values=meta_data["metrics"]["hist_sig"][i],
                label="Sig",
                alpha=0.8,
                fill=None,
                linewidth=2,
                # align="edge",
            )
            plt.stairs(
                edges=meta_data["config"]["bins"],
                values=meta_data["metrics"]["hist_bkg"][i],
                label="Bkg",
                alpha=0.8,
                fill=None,
                linewidth=2,
                # align="edge",
            )
            if meta_data["config"]["do_m_hh"]:
                plt.xlabel("m$_{hh}$ (MeV)")
            else:
                plt.xlabel("NN score")

            plt.ylabel("Events")
            plt.legend()  # prop={"size": 6})
            plt.tight_layout()
            ax = plt.gca()
            ylim = ax.get_ylim()
            if ylim[1] > ymax[1]:
                ymax = ylim
            ax.set_ylim(ymax)
            plt.savefig(image_path + "/" + f"{i:004d}" + ".png")
            print(i)
            plt.close()

        create_gif_from_folder(image_path, model_path + m + ".gif", duration=0.005)
