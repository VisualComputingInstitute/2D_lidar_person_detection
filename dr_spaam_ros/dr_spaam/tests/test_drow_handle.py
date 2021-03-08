import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from dr_spaam.datahandle import DROWHandle
from dr_spaam.utils import utils as u


_X_LIM = (-7, 7)
_Y_LIM = (-7, 7)
_INTERACTIVE = False
_SAVE_DIR = "/home/jia/tmp_imgs/test_drow_handle"


def _plot_annotation(ann, ax, color, radius):
    if len(ann) > 0:
        ann = np.array(ann)
        det_x, det_y = u.rphi_to_xy(ann[:, 0], ann[:, 1])
        for x, y in zip(det_x, det_y):
            c = plt.Circle((x, y), radius=radius, color=color, fill=False)
            ax.add_artist(c)


def _plot_sequence():
    drow_handle = DROWHandle(
        split="train",
        cfg={"num_scans": 1, "scan_stride": 1, "data_dir": "./data/DROWv2-data"},
    )

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    _break = False

    if _INTERACTIVE:

        def p(event):
            nonlocal _break
            _break = True

        fig.canvas.mpl_connect("key_press_event", p)
    else:
        if os.path.exists(_SAVE_DIR):
            shutil.rmtree(_SAVE_DIR)
        os.makedirs(_SAVE_DIR)

    for i, data_dict in enumerate(drow_handle):
        if _break:
            break

        scan_x, scan_y = u.rphi_to_xy(data_dict["scans"][-1], data_dict["scan_phi"])

        plt.cla()
        ax.set_aspect("equal")
        ax.set_xlim(_X_LIM[0], _X_LIM[1])
        ax.set_ylim(_Y_LIM[0], _Y_LIM[1])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Frame {data_dict['idx']}. Press any key to exit.")
        # ax.axis("off")

        ax.scatter(scan_x, scan_y, s=1, c="black")

        _plot_annotation(data_dict["dets_wc"], ax, "red", 0.6)
        _plot_annotation(data_dict["dets_wa"], ax, "green", 0.4)
        _plot_annotation(data_dict["dets_wp"], ax, "blue", 0.35)

        if _INTERACTIVE:
            plt.pause(0.1)
        else:
            plt.savefig(os.path.join(_SAVE_DIR, f"frame_{data_dict['idx']:04}.png"))

    if _INTERACTIVE:
        plt.show()


if __name__ == "__main__":
    _plot_sequence()
