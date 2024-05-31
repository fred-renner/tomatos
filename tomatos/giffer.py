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
        "tomatos_bce_5",
        "tomatos_cls_5",
    ]
    ymax = 0
    for m in models:
        model_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/" + m + "/"
        with open(model_path + "metadata.json", "r") as file:
            meta_data = json.load(file)
        image_path = model_path + "/images"
        if not os.path.isdir(image_path):
            os.makedirs(image_path)
        # loop over epochs
        for i in range(len(meta_data["metrics"]["NOSYS"])):
            # if i==1:
            #     break
            print(i)
            # loop over hists
            plt.figure()

            for hist_name in meta_data["config"]["data_types"]:
                # if "JET" in hist_name or "GEN" in hist_name:
                #     break
                plt.stairs(
                    edges=meta_data["config"]["bins"],
                    values=meta_data["metrics"][hist_name][i],
                    label=hist_name,
                    fill=None,
                    linewidth=1,
                    # align="edge",
                )

                ax = plt.gca()
                ylim = ax.get_ylim()
                if ylim[1] > ymax:
                    ymax = ylim[1]
                ax.set_ylim([0, ymax])

            if meta_data["config"]["do_m_hh"]:
                plt.xlabel("m$_{hh}$ (MeV)")
            else:
                plt.xlabel("NN score")

            plt.text(0.5, 0.8, f"Epoch {i}",
                horizontalalignment='center',
                verticalalignment='center',
                transform = ax.transAxes)
            plt.ylabel("Events")
            plt.legend(prop={"size": 5},loc="upper right")  # prop={"size": 6})
            plt.tight_layout()
            plt.savefig(image_path + "/" + f"{i:004d}" + ".png", dpi=100)
            plt.close()

        create_gif_from_folder(image_path, model_path + m + ".gif", duration=0.001)
        print(model_path + m + ".gif")
