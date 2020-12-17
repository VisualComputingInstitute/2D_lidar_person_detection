import numpy as np
import matplotlib.pyplot as plt

import dr_spaam.utils.utils as u

_X_LIM = (-7, 7)
_Y_LIM = (-7, 7)


def plot_one_frame(
    batch_dict,
    frame_idx,
    pred_cls=None,
    pred_reg=None,
    dets_cls=None,
    dets_xy=None,
    xlim=_X_LIM,
    ylim=_Y_LIM,
):
    """Plot one frame from a batch, specified by frame_idx.

    Returns:
        fig: figure handle
        ax: axis handle
    """
    fig, ax = _create_figure("", xlim, ylim)

    # scan and cls label
    scan_r = batch_dict["scans"][frame_idx][-1]
    scan_phi = batch_dict["scan_phi"][frame_idx]
    target_cls = batch_dict["target_cls"][frame_idx]
    _plot_scan(ax, scan_r, scan_phi, target_cls, s=1)

    # annotation
    ann = batch_dict["dets_wp"][frame_idx]
    ann_valid_mask = batch_dict["anns_valid_mask"][frame_idx]
    if len(ann) > 0:
        ann = np.array(ann)
        det_x, det_y = u.rphi_to_xy(ann[:, 0], ann[:, 1])
        for x, y, valid in zip(det_x, det_y, ann_valid_mask):
            c = "blue" if valid else "orange"
            c = plt.Circle((x, y), radius=0.4, color=c, fill=False)
            ax.add_artist(c)

    # regression target
    target_reg = batch_dict["target_reg"][frame_idx]
    _plot_target(ax, target_reg, target_cls > 0, scan_r, scan_phi, s=10, c="blue")

    # regression result
    if dets_xy is not None and dets_cls is not None:
        _plot_detection(ax, dets_cls, dets_xy, s=40, color_dim=1)

    if pred_cls is not None and pred_reg is not None:
        _plot_prediction(ax, pred_cls, pred_reg, scan_r, scan_phi, s=2, color_dim=1)

    return fig, ax


def plot_one_batch(batch_dict, xlim=_X_LIM, ylim=_Y_LIM):
    fig_ax_list = []
    for ib in range(len(batch_dict["input"])):
        fig_ax_list.append(plot_one_frame(batch_dict, ib))

    return fig_ax_list


def plot_one_batch_detr(batch_dict, xlim=_X_LIM, ylim=_Y_LIM):
    fig_ax_list = []

    for ib in range(len(batch_dict["input"])):
        fr_idx = batch_dict["frame_dict_curr"][ib]["idx"]
        fig, ax = _create_figure(fr_idx, xlim, ylim)

        # scan and cls label
        scan_r = batch_dict["frame_dict_curr"][ib]["laser_data"][-1]
        scan_phi = batch_dict["frame_dict_curr"][ib]["laser_grid"]
        target_cls = batch_dict["target_cls"][ib]
        _plot_scan(ax, scan_r, scan_phi, target_cls, s=1)

        # annotation for current frame
        anns = batch_dict["frame_dict_curr"][ib]["dets_rphi"]
        anns_valid_mask = batch_dict["anns_valid_mask"][ib]
        anns_valid = anns[:, anns_valid_mask]
        anns_invalid = anns[:, np.logical_not(anns_valid_mask)]
        _plot_annotation_detr(ax, anns_valid, radius=0.4, color="blue")
        _plot_annotation_detr(ax, anns_invalid, radius=0.4, color="orange")

        # annotation for previous frame
        anns_prev = batch_dict["frame_dict_curr"][ib]["dets_rphi_prev"]
        anns_tracking_mask = batch_dict["anns_tracking_mask"][ib]
        anns_prev = anns_prev[:, anns_tracking_mask]
        _plot_annotation_detr(ax, anns_prev, radius=0.4, color="gray", linestyle="--")

        # regression target for previous frame
        target_reg_prev = batch_dict["target_reg_prev"][ib]
        target_tracking_flag = batch_dict["target_tracking_flag"][ib]
        _plot_target(
            ax, target_reg_prev, target_tracking_flag, scan_r, scan_phi, s=25, c="gray"
        )

        # regression target for current frame
        target_reg = batch_dict["target_reg"][ib]
        _plot_target(ax, target_reg, target_cls > 0, scan_r, scan_phi, s=10, c="red")

        # regression result for previous frame
        pred_cls = batch_dict["pred_cls"][ib]
        pred_reg_prev = batch_dict["pred_reg_prev"][ib]
        _plot_prediction(
            ax, pred_cls, pred_reg_prev, scan_r, scan_phi, s=2, color_dim=2
        )

        # regression result for current frame
        pred_reg = batch_dict["pred_reg"][ib]
        _plot_prediction(ax, pred_cls, pred_reg, scan_r, scan_phi, s=2, color_dim=1)

        fig_ax_list.append((fig, ax))

    return fig_ax_list


def _cls_to_color(cls, color_dim):
    color = 1.0 - cls.reshape(-1, 1).repeat(3, axis=1)
    color[:, color_dim] = 1
    return color


def _create_figure(title, xlim, ylim):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_title(f"{title}")

    return fig, ax


def _plot_scan(ax, scan_r, scan_phi, target_cls, s):
    scan_x, scan_y = u.rphi_to_xy(scan_r, scan_phi)
    ax.scatter(scan_x[target_cls < 0], scan_y[target_cls < 0], s=s, c="orange")
    ax.scatter(scan_x[target_cls == 0], scan_y[target_cls == 0], s=s, c="black")
    ax.scatter(scan_x[target_cls > 0], scan_y[target_cls > 0], s=s, c="green")


def _plot_prediction(ax, pred_cls, pred_reg, scan_r, scan_phi, s, color_dim):
    pred_r, pred_phi = u.canonical_to_global(
        scan_r, scan_phi, pred_reg[:, 0], pred_reg[:, 1]
    )
    pred_x, pred_y = u.rphi_to_xy(pred_r, pred_phi)
    pred_color = _cls_to_color(pred_cls, color_dim=color_dim)
    ax.scatter(pred_x, pred_y, s=s, c=pred_color)


def _plot_detection(ax, dets_cls, dets_xy, s, color_dim):
    dets_color = _cls_to_color(dets_cls, color_dim=color_dim)
    ax.scatter(dets_xy[:, 0], dets_xy[:, 1], marker="x", s=s, c=dets_color)


def _plot_target(ax, target_reg, target_flag, scan_r, scan_phi, s, c):
    dets_r, dets_phi = u.canonical_to_global(
        scan_r, scan_phi, target_reg[:, 0], target_reg[:, 1]
    )
    dets_r = dets_r[target_flag]
    dets_phi = dets_phi[target_flag]
    dets_x, dets_y = u.rphi_to_xy(dets_r, dets_phi)
    ax.scatter(dets_x, dets_y, s=s, c=c)


def _plot_annotation_detr(ax, anns, radius, color, linestyle="-"):
    if len(anns) == 0:
        return

    det_x, det_y = u.rphi_to_xy(anns[0], anns[1])
    for x, y in zip(det_x, det_y):
        c = plt.Circle(
            (x, y), radius=radius, color=color, fill=False, linestyle=linestyle
        )
        ax.add_artist(c)
