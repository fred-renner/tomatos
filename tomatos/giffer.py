#! python3

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def update_plot(i, meta_data, ax, ymax):
    ax.clear()
    print(i)
    for hist_name in meta_data["config"]["data_types"]:
        if "JET" in hist_name or "GEN" in hist_name:
            continue
        ax.stairs(
            edges=meta_data["config"]["bins"],
            values=meta_data["metrics"][hist_name][i],
            label=hist_name,
            fill=None,
            linewidth=1,
        )

    ax.set_ylim([0, ymax])
    if meta_data["config"]["do_m_hh"]:
        ax.set_xlabel("m$_{hh}$ (MeV)")
    else:
        ax.set_xlabel("NN score")

    ax.text(
        0.1,
        0.95,
        f"Epoch {i}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.set_ylabel("Events")
    ax.legend(prop={"size": 5}, loc="upper right")
    ax.figure.tight_layout()


if __name__ == "__main__":
    models = [
        # "tomatos_cls_5_50",
        # "tomatos_debug",
        # "tomatos_cls_5_1000_fixed_cuts_m10",
        "tomatos_cls_5_2500_slope_50",
        # "tomatos_cls_5_1000_slope_500",
        # "tomatos_cls_5_1000_slope_1000",
    ]
    ymax = 0
    for m in models:
        model_path = "/lustre/fs22/group/atlas/freder/hh/run/tomatos/" + m + "/"
        with open(model_path + "metadata.json", "r") as file:
            meta_data = json.load(file)

        meta_data["config"]["data_types"] += ["bkg_shape_sys_up", "bkg_shape_sys_down"]
        meta_data["config"]["data_types"] += ["ps_up", "ps_down"]

        for i in range(len(meta_data["metrics"]["NOSYS"])):
            for hist_name in meta_data["config"]["data_types"]:
                if "JET" in hist_name or "GEN" in hist_name:
                    continue
                values = meta_data["metrics"][hist_name][i]
                if np.max(values) > ymax:
                    ymax = np.max(values)
        ymax = 15
        fig, ax = plt.subplots(figsize=(6, 4))
        ani = FuncAnimation(
            fig,
            update_plot,
            frames=range(len(meta_data["metrics"]["NOSYS"])),
            fargs=(meta_data, ax, ymax),
        )

        gif_path = model_path + m + ".gif"
        ani.save(gif_path, writer=PillowWriter(fps=60))

        plt.close()

        print(gif_path)
