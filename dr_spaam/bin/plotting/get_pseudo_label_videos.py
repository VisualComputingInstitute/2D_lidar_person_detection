import cv2
import os
from tqdm import tqdm
import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch

from dr_spaam.dataset import get_dataloader
import dr_spaam.utils.jrdb_transforms as jt
import dr_spaam.utils.utils as u

from dr_spaam.pipeline.logger import Logger
from dr_spaam.model.get_model import get_model

_X_LIM = (-15, 15)
# _Y_LIM = (-10, 4)
_Y_LIM = (-7, 7)

_PLOTTING_INTERVAL = 20
_MAX_COUNT = 1e9
# _MAX_COUNT = 1e1

# _COLOR_CLOSE_HSV = (1.0, 0.59, 0.75)
_COLOR_CLOSE_HSV = (0.0, 1.0, 1.0)
_COLOR_FAR_HSV = (0.0, 0.0, 1.0)
_COLOR_DIST_RANGE = (0.0, 20.0)

# _SPLIT = "train"
_SPLIT = "val"

_SAVE_DIR = f"/globalwork/jia/tmp/pseudo_label_videos/{_SPLIT}"
os.makedirs(_SAVE_DIR, exist_ok=True)


def _get_bounding_box_plotting_vertices(x0, y0, x1, y1):
    return np.array([(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)])


def _distance_to_bgr_color(dist):
    dist_normalized = (
        np.clip(dist, _COLOR_DIST_RANGE[0], _COLOR_DIST_RANGE[1]) / _COLOR_DIST_RANGE[1]
    ).reshape(-1, 1)

    c_hsv = (
        np.array(_COLOR_CLOSE_HSV).reshape(1, -1) * (1.0 - dist_normalized)
        + np.array(_COLOR_FAR_HSV).reshape(1, -1) * dist_normalized
    ).astype(np.float32)
    c_hsv = c_hsv[None, ...]
    c_bgr = cv2.cvtColor(c_hsv, cv2.COLOR_HSV2RGB)

    return c_bgr[0]


def _plot_frame_im(batch_dict, ib):
    frame_id = f"{batch_dict['frame_id'][ib]:06d}"
    sequence = batch_dict["sequence"][ib]

    im = batch_dict["im_data"][ib]["stitched_image0"]
    crop_min_x = 0
    im = im[:, crop_min_x:]
    height = im.shape[0]
    width = im.shape[1]
    dpi = height / 1.0

    fig = plt.figure()
    fig.set_size_inches(1.0 * width / height, 1, forward=False)
    ax_im = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    fig.add_axes(ax_im)

    # image
    ax_im.axis("off")
    ax_im.imshow(im)
    plt.xlim(0, width)
    plt.ylim(height, 0)

    # laser points on image
    scan_r = batch_dict["scans"][ib][-1]
    scan_phi = batch_dict["scan_phi"][ib]
    scan_x, scan_y = u.rphi_to_xy(scan_r, scan_phi)
    scan_z = batch_dict["laser_z"][ib]
    scan_xyz_laser = np.stack((scan_x, -scan_y, scan_z), axis=0)  # in JRDB laser frame
    p_xy, ib_mask = jt.transform_pts_laser_to_stitched_im(scan_xyz_laser)

    # detection bounding box
    for box_dict in batch_dict["im_dets"][ib]:
        x0, y0, w, h = box_dict["box"]
        x1 = x0 + w
        y1 = y0 + h
        verts = _get_bounding_box_plotting_vertices(x0, y0, x1, y1)
        ax_im.plot(verts[:, 0] - crop_min_x, verts[:, 1], c=(0.0, 0.0, 1.0), alpha=0.3)
        # c = max(float(box_dict["score"]) - 0.5, 0) * 2.0
        # ax_im.plot(verts[:, 0] - crop_min_x, verts[:, 1], c=(1.0 - c, 1.0 - c, 1.0))
        # ax_im.plot(verts[:, 0] - crop_min_x, verts[:, 1],
        # c=(0.0, 0.0, 1.0), alpha=1.0)

        # x1_large = x1 + 0.05 * w
        # x0_large = x0 - 0.05 * w
        # y1_large = y1 + 0.05 * w
        # y0_large = y0 - 0.05 * w
        # in_box_mask = np.logical_and(
        #     np.logical_and(p_xy[0] > x0_large, p_xy[0] < x1_large),
        #     np.logical_and(p_xy[1] > y0_large, p_xy[1] < y1_large)
        # )
        # neg_mask[in_box_mask] = False

    for box in batch_dict["pseudo_label_boxes"][ib]:
        x0, y0, x1, y1 = box
        verts = _get_bounding_box_plotting_vertices(x0, y0, x1, y1)
        ax_im.plot(verts[:, 0] - crop_min_x, verts[:, 1], c="green")

    # overlay laser points on image
    c_bgr = _distance_to_bgr_color(scan_r)
    ax_im.scatter(
        p_xy[0, ib_mask] - crop_min_x, p_xy[1, ib_mask], s=1, color=c_bgr[ib_mask]
    )
    # neg_mask = np.ones(p_xy.shape[1], dtype=np.bool)
    # ax_im.scatter(p_xy[0, neg_mask] - crop_min_x, p_xy[1, neg_mask], s=1,
    # color=c_bgr[neg_mask])

    # save fig
    fig_file = os.path.join(_SAVE_DIR, f"figs/{sequence}/im_{frame_id}.png")
    os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    fig.savefig(fig_file, dpi=dpi)
    plt.close(fig)


def _plot_frame_pts(batch_dict, ib, pred_cls, pred_reg, pred_cls_p, pred_reg_p):
    frame_id = f"{batch_dict['frame_id'][ib]:06d}"
    sequence = batch_dict["sequence"][ib]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.set_xlim(_X_LIM[0], _X_LIM[1])
    ax.set_ylim(_Y_LIM[0], _Y_LIM[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    # ax.set_title(f"Frame {data_dict['idx'][ib]}. Press any key to exit.")

    # scan and cls label
    scan_r = batch_dict["scans"][ib][-1]
    scan_phi = batch_dict["scan_phi"][ib]
    scan_x, scan_y = u.rphi_to_xy(scan_r, scan_phi)
    ax.scatter(scan_x, scan_y, s=0.5, c="blue")

    # annotation
    ann = batch_dict["dets_wp"][ib]
    ann_valid_mask = batch_dict["anns_valid_mask"][ib]
    if len(ann) > 0:
        ann = np.array(ann)
        det_x, det_y = u.rphi_to_xy(ann[:, 0], ann[:, 1])
        for x, y, valid in zip(det_x, det_y, ann_valid_mask):
            if valid:
                # c = plt.Circle((x, y), radius=0.1, color="red", fill=True)
                c = plt.Circle(
                    (x, y), radius=0.4, color="red", fill=False, linestyle="--"
                )
                ax.add_artist(c)

    # plot detections
    if pred_cls is not None and pred_reg is not None:
        dets_xy, dets_cls, _ = u.nms_predicted_center(
            scan_r, scan_phi, pred_cls[ib].reshape(-1), pred_reg[ib]
        )
        dets_xy = dets_xy[dets_cls >= 0.9438938]  # at EER
        if len(dets_xy) > 0:
            for x, y in dets_xy:
                c = plt.Circle((x, y), radius=0.4, color=(0, 0.56, 0.56), fill=False)
                ax.add_artist(c)
        fig_file = os.path.join(
            _SAVE_DIR, f"figs/{sequence}/scan_det_{frame_id}.png"
        )

        # plot in addition detections from a pre-trained
        if pred_cls_p is not None and pred_reg_p is not None:
            dets_xy, dets_cls, _ = u.nms_predicted_center(
                scan_r, scan_phi, pred_cls_p[ib].reshape(-1), pred_reg_p[ib]
            )
            dets_xy = dets_xy[dets_cls > 0.29919282]  # at EER
            if len(dets_xy) > 0:
                for x, y in dets_xy:
                    c = plt.Circle((x, y), radius=0.4, color="green", fill=False)
                    ax.add_artist(c)
    # plot pre-trained detections only
    elif pred_cls_p is not None and pred_reg_p is not None:
        dets_xy, dets_cls, _ = u.nms_predicted_center(
            scan_r, scan_phi, pred_cls_p[ib].reshape(-1), pred_reg_p[ib]
        )
        dets_xy = dets_xy[dets_cls > 0.29919282]  # at EER
        if len(dets_xy) > 0:
            for x, y in dets_xy:
                c = plt.Circle((x, y), radius=0.4, color="green", fill=False)
                ax.add_artist(c)
        fig_file = os.path.join(
            _SAVE_DIR, f"figs/{sequence}/scan_pretrain_{frame_id}.png"
        )
    # plot pseudo-labels only
    else:
        pl_xy = batch_dict["pseudo_label_loc_xy"][ib]
        if len(pl_xy) > 0:
            for x, y in pl_xy:
                c = plt.Circle((x, y), radius=0.4, color="green", fill=False)
                ax.add_artist(c)
        fig_file = os.path.join(_SAVE_DIR, f"figs/{sequence}/scan_pl_{frame_id}.png")

    # save fig
    os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    fig.savefig(fig_file, dpi=200)
    plt.close(fig)


def plot_pseudo_label_for_all_frames():
    with open("./cfgs/base_drow_jrdb_cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["pseudo_label"] = True
    cfg["dataset"]["pl_correction_level"] = 0

    test_loader = get_dataloader(
        split=_SPLIT,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        dataset_cfg=cfg["dataset"],
    )

    model = get_model(cfg["model"])
    model.cuda()
    model.eval()

    logger = Logger(cfg["pipeline"]["Logger"])
    logger.load_ckpt("./ckpts/ckpt_jrdb_pl_drow3_phce_e40.pth", model)

    model_pretrain = get_model(cfg["model"])
    model_pretrain.cuda()
    model_pretrain.eval()
    logger.load_ckpt("./ckpts/ckpt_drow_drow3_e40.pth", model_pretrain)

    # generate pseudo labels for all sample
    for count, batch_dict in enumerate(tqdm(test_loader)):
        if count >= _MAX_COUNT:
            break

        with torch.no_grad():
            net_input = torch.from_numpy(batch_dict["input"]).cuda().float()
            pred_cls, pred_reg = model(net_input)
            pred_cls = torch.sigmoid(pred_cls).data.cpu().numpy()
            pred_reg = pred_reg.data.cpu().numpy()

            pred_cls_p, pred_reg_p = model_pretrain(net_input)
            pred_cls_p = torch.sigmoid(pred_cls_p).data.cpu().numpy()
            pred_reg_p = pred_reg_p.data.cpu().numpy()

        if count % _PLOTTING_INTERVAL == 0:
            for ib in range(len(batch_dict["input"])):
                _plot_frame_im(batch_dict, ib)
                # _plot_frame_pts(batch_dict, ib, None, None, None, None)
                # _plot_frame_pts(batch_dict, ib, pred_cls, pred_reg, None, None)
                # # _plot_frame_pts(batch_dict, ib, pred_cls, pred_reg, pred_cls_p, pred_reg_p)

                _plot_frame_pts(batch_dict, ib, pred_cls, pred_reg, None, None)
                _plot_frame_pts(batch_dict, ib, None, None, pred_cls_p, pred_reg_p)


def plot_color_bar():
    dist = np.linspace(0, 20.0, int(1e4))
    c_bgr = _distance_to_bgr_color(dist)
    c_bgr = np.repeat(c_bgr[None, ...], 1000, axis=0)

    fig = plt.figure()
    fig.set_size_inches(10, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(c_bgr)

    fig_file = os.path.join(_SAVE_DIR, "color_bar.pdf")
    fig.savefig(fig_file)
    plt.close(fig)


if __name__ == "__main__":
    plot_color_bar()
    plot_pseudo_label_for_all_frames()
