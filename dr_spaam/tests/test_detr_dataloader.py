import os
import shutil
import yaml
import matplotlib.pyplot as plt

import dr_spaam.utils.utils as u
from dr_spaam.dataset.get_dataloader import get_dataloader

_X_LIM = (-7, 7)
_Y_LIM = (-7, 7)
_INTERACTIVE = False
_SAVE_DIR = "/home/jia/tmp_imgs/test_detr_dataloader"


def _test_detr_dataloader():
    with open("./tests/test.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["DataHandle"]["tracking"] = True
    cfg["dataset"]["DataHandle"]["num_scans"] = 1

    test_loader = get_dataloader(
        split="train",
        batch_size=8,
        num_workers=1,
        shuffle=True,
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
        for ib in range(len(data_dict["input"])):
            fr_idx = data_dict["frame_dict_curr"][ib]["idx"]

            plt.cla()
            ax.set_xlim(_X_LIM[0], _X_LIM[1])
            ax.set_ylim(_Y_LIM[0], _Y_LIM[1])
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_aspect("equal")
            ax.set_title(f"Frame {fr_idx}. Press any key to exit.")

            # scan and cls label
            scan_r = data_dict["frame_dict_curr"][ib]["laser_data"][-1]
            scan_phi = data_dict["frame_dict_curr"][ib]["laser_grid"]
            scan_x, scan_y = u.rphi_to_xy(scan_r, scan_phi)

            target_cls = data_dict["target_cls"][ib]
            ax.scatter(scan_x[target_cls < 0], scan_y[target_cls < 0], s=1, c="orange")
            ax.scatter(scan_x[target_cls == 0], scan_y[target_cls == 0], s=1, c="black")
            ax.scatter(scan_x[target_cls > 0], scan_y[target_cls > 0], s=1, c="green")

            # annotation for tracking
            anns_tracking = data_dict["frame_dict_curr"][ib]["dets_rphi_prev"]
            anns_tracking_mask = data_dict["anns_tracking_mask"][ib]
            anns_tracking = anns_tracking[:, anns_tracking_mask]
            if len(anns_tracking) > 0:
                det_x, det_y = u.rphi_to_xy(anns_tracking[0], anns_tracking[1])
                for x, y in zip(det_x, det_y):
                    c = plt.Circle(
                        (x, y), radius=0.5, color="gray", fill=False, linestyle="--"
                    )
                    ax.add_artist(c)

            # annotation
            anns = data_dict["frame_dict_curr"][ib]["dets_rphi"]
            anns_valid_mask = data_dict["anns_valid_mask"][ib]
            if len(anns) > 0:
                det_x, det_y = u.rphi_to_xy(anns[0], anns[1])
                for x, y, valid in zip(det_x, det_y, anns_valid_mask):
                    c = "blue" if valid else "orange"
                    c = plt.Circle((x, y), radius=0.4, color=c, fill=False)
                    ax.add_artist(c)

            # reg label for previous frame
            target_reg_prev = data_dict["target_reg_prev"][ib]
            target_tracking_flag = data_dict["target_tracking_flag"][ib]
            dets_r_prev, dets_phi_prev = u.canonical_to_global(
                scan_r, scan_phi, target_reg_prev[:, 0], target_reg_prev[:, 1]
            )
            dets_r_prev = dets_r_prev[target_tracking_flag]
            dets_phi_prev = dets_phi_prev[target_tracking_flag]
            dets_x_prev, dets_y_prev = u.rphi_to_xy(dets_r_prev, dets_phi_prev)
            ax.scatter(dets_x_prev, dets_y_prev, s=25, c="gray")

            # reg label for current frame
            target_reg = data_dict["target_reg"][ib]
            dets_r, dets_phi = u.canonical_to_global(
                scan_r, scan_phi, target_reg[:, 0], target_reg[:, 1]
            )
            dets_r = dets_r[target_cls > 0]
            dets_phi = dets_phi[target_cls > 0]
            dets_x, dets_y = u.rphi_to_xy(dets_r, dets_phi)
            ax.scatter(dets_x, dets_y, s=10, c="red")

            if _INTERACTIVE:
                plt.pause(0.1)
            else:
                plt.savefig(
                    os.path.join(_SAVE_DIR, f"b{count:03}s{ib:02}f{fr_idx:04}.png",)
                )

    if _INTERACTIVE:
        plt.show()


if __name__ == "__main__":
    _test_detr_dataloader()
