import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from dr_spaam.utils import utils as u
import dr_spaam.utils.jrdb_transforms as jt
from dr_spaam.detector import Detector
from dr_spaam.datahandle.jrdb_handle import JRDBHandle

_X_LIM = (-15, 15)
_Y_LIM = (-15, 15)
_INTERACTIVE = False
_SAVE_DIR = "/home/jia/tmp_imgs/test_detector"


def _plot_annotation(ann, ax, color, radius):
    if len(ann) > 0:
        ann = np.array(ann)
        det_x, det_y = u.rphi_to_xy(ann[:, 0], ann[:, 1])
        for x, y in zip(det_x, det_y):
            c = plt.Circle(
                (x, y), radius=radius, color=color, fill=False, linestyle="--"
            )
            ax.add_artist(c)


def test_detector():
    data_handle = JRDBHandle(
        split="train",
        cfg={"data_dir": "./data/JRDB", "num_scans": 10, "scan_stride": 1},
    )

    # ckpt_file = "/home/jia/ckpts/ckpt_jrdb_ann_drow3_e40.pth"
    # d = Detector(
    #     ckpt_file, model="DROW3", gpu=True, stride=1, panoramic_scan=True
    # )

    ckpt_file = "/home/jia/ckpts/ckpt_jrdb_ann_dr_spaam_e20.pth"
    d = Detector(ckpt_file, model="DR-SPAAM", gpu=True, stride=1, panoramic_scan=True)

    d.set_laser_fov(360)

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

    for i, data_dict in enumerate(data_handle):
        if _break:
            break

        # plot scans
        scan_r = data_dict["laser_data"][-1, ::-1]  # to DROW frame
        scan_x, scan_y = u.rphi_to_xy(scan_r, data_dict["laser_grid"])

        plt.cla()
        ax.set_aspect("equal")
        ax.set_xlim(_X_LIM[0], _X_LIM[1])
        ax.set_ylim(_Y_LIM[0], _Y_LIM[1])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Frame {data_dict['idx']}. Press any key to exit.")
        # ax.axis("off")

        ax.scatter(scan_x, scan_y, s=1, c="black")

        # plot annotation
        ann_xyz = [
            (ann["box"]["cx"], ann["box"]["cy"], ann["box"]["cz"])
            for ann in data_dict["pc_anns"]
        ]
        if len(ann_xyz) > 0:
            ann_xyz = np.array(ann_xyz, dtype=np.float32).T
            ann_xyz = jt.transform_pts_base_to_laser(ann_xyz)
            ann_xyz[1] = -ann_xyz[1]  # to DROW frame
            for xyz in ann_xyz.T:
                c = plt.Circle(
                    (xyz[0], xyz[1]),
                    radius=0.4,
                    color="red",
                    fill=False,
                    linestyle="--",
                )
                ax.add_artist(c)

        # plot detection
        dets_xy, dets_cls, _ = d(scan_r)
        dets_cls_norm = np.clip(dets_cls, 0, 0.3) / 0.3
        for xy, cls_norm in zip(dets_xy, dets_cls_norm):
            color = (1.0 - cls_norm, 1.0, 1.0 - cls_norm)
            c = plt.Circle(
                (xy[0], xy[1]), radius=0.4, color=color, fill=False, linestyle="-"
            )
            ax.add_artist(c)

        if _INTERACTIVE:
            plt.pause(0.1)
        else:
            plt.savefig(os.path.join(_SAVE_DIR, f"frame_{data_dict['idx']:04}.png"))

    if _INTERACTIVE:
        plt.show()


if __name__ == "__main__":
    test_detector()
