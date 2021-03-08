import os
import shutil
import yaml
import numpy as np
import matplotlib.pyplot as plt

import dr_spaam.utils.utils as u
from dr_spaam.dataset import get_dataloader

_X_LIM = (-7, 7)
_Y_LIM = (-7, 7)
# _X_LIM = (-15, 15)
# _Y_LIM = (-15, 15)
_MAX_COUNT = 3
_INTERACTIVE = False
_SAVE_DIR = "/home/jia/tmp_imgs/test_dataloader"


def _plot_sample_light(fig, ax, ib, count, data_dict):
    plt.cla()
    ax.set_xlim(_X_LIM[0], _X_LIM[1])
    ax.set_ylim(_Y_LIM[0], _Y_LIM[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    # ax.set_title(f"Frame {data_dict['idx'][ib]}. Press any key to exit.")

    # scan and cls label
    scan_r = data_dict["scans"][ib][-1]
    scan_phi = data_dict["scan_phi"][ib]
    scan_x, scan_y = u.rphi_to_xy(scan_r, scan_phi)
    ax.scatter(scan_x, scan_y, s=0.5, c="blue")

    # annotation
    ann = data_dict["dets_wp"][ib]
    ann_valid_mask = data_dict["anns_valid_mask"][ib]
    if len(ann) > 0:
        ann = np.array(ann)
        det_x, det_y = u.rphi_to_xy(ann[:, 0], ann[:, 1])
        for x, y, valid in zip(det_x, det_y, ann_valid_mask):
            if valid:
                # c = plt.Circle((x, y), radius=0.1, color="red", fill=True)
                c = plt.Circle((x, y), radius=0.4, color="red", fill=False)
                ax.add_artist(c)


def _plot_sample(fig, ax, ib, count, data_dict):
    plt.cla()
    ax.set_xlim(_X_LIM[0], _X_LIM[1])
    ax.set_ylim(_Y_LIM[0], _Y_LIM[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_title(f"Frame {data_dict['idx'][ib]}. Press any key to exit.")

    # scan and cls label
    scan_r = data_dict["scans"][ib][-1]
    scan_phi = data_dict["scan_phi"][ib]
    scan_x, scan_y = u.rphi_to_xy(scan_r, scan_phi)

    target_cls = data_dict["target_cls"][ib]
    ax.scatter(scan_x[target_cls == -2], scan_y[target_cls == -2], s=1, c="yellow")
    ax.scatter(scan_x[target_cls == -1], scan_y[target_cls == -1], s=1, c="orange")
    ax.scatter(scan_x[target_cls == 0], scan_y[target_cls == 0], s=1, c="black")
    ax.scatter(scan_x[target_cls > 0], scan_y[target_cls > 0], s=1, c="green")

    # annotation
    ann = data_dict["dets_wp"][ib]
    ann_valid_mask = data_dict["anns_valid_mask"][ib]
    if len(ann) > 0:
        ann = np.array(ann)
        det_x, det_y = u.rphi_to_xy(ann[:, 0], ann[:, 1])
        for x, y, valid in zip(det_x, det_y, ann_valid_mask):
            c = "blue" if valid else "orange"
            c = plt.Circle((x, y), radius=0.4, color=c, fill=False)
            ax.add_artist(c)

    # reg label
    target_reg = data_dict["target_reg"][ib]
    dets_r, dets_phi = u.canonical_to_global(
        scan_r, scan_phi, target_reg[:, 0], target_reg[:, 1]
    )
    dets_r = dets_r[target_cls > 0]
    dets_phi = dets_phi[target_cls > 0]
    dets_x, dets_y = u.rphi_to_xy(dets_r, dets_phi)
    ax.scatter(dets_x, dets_y, s=10, c="red")


def _test_dataloader():
    with open("./base_dr_spaam_jrdb_cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    cfg["dataset"]["pseudo_label"] = False
    cfg["dataset"]["pl_correction_level"] = 0

    test_loader = get_dataloader(
        split="val",
        batch_size=5,
        num_workers=1,
        shuffle=False,
        dataset_cfg=cfg["dataset"],
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

    for count, data_dict in enumerate(test_loader):
        if count >= _MAX_COUNT:
            break

        for ib in range(len(data_dict["input"])):
            _plot_sample(fig, ax, ib, count, data_dict)

            if _INTERACTIVE:
                plt.pause(0.1)
            else:
                plt.savefig(
                    os.path.join(
                        _SAVE_DIR, f"b{count:03}s{ib:02}f{data_dict['idx'][ib]:04}.pdf"
                    )
                )

    if _INTERACTIVE:
        plt.show()


if __name__ == "__main__":
    _test_dataloader()
