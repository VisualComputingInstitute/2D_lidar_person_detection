import cv2
import os
import shutil
import time
from tqdm import tqdm
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from dr_spaam.dataset import get_dataloader
import dr_spaam.utils.jrdb_transforms as jt
import dr_spaam.utils.precision_recall as pru
import dr_spaam.utils.utils as u

_X_LIM = (-7, 7)
_Y_LIM = (-7, 7)

_PLOTTING_INTERVAL = 20
_MAX_COUNT = 1e9
_MAX_COUNT = 3

_DIST_BINS = np.arange(30) + 0.5

_SPLIT = "train"
# _SPLIT = "val"

_SAVE_DIR = f"/globalwork/jia/tmp/plot_clustering/{_SPLIT}"


def _distance_to_bgr_color(dist):
    # _COLOR_CLOSE_HSV = (1.0, 0.59, 0.75)
    _COLOR_CLOSE_HSV = (0.0, 1.0, 1.0)
    _COLOR_FAR_HSV = (0.0, 0.0, 1.0)
    _COLOR_DIST_RANGE = (0.0, 20.0)

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


def _plot_frame(batch_dict, ib):
    frame_id = f"{batch_dict['frame_id'][ib]:06d}"
    sequence = batch_dict["sequence"][ib]

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(3, 1, figure=fig)

    ax_im = fig.add_subplot(gs[0, 0])
    ax_bev = fig.add_subplot(gs[1:, 0])

    ax_bev.set_xlim(_X_LIM[0], _X_LIM[1])
    ax_bev.set_ylim(_Y_LIM[0], _Y_LIM[1])
    ax_bev.set_xlabel("x [m]")
    ax_bev.set_ylabel("y [m]")
    ax_bev.set_aspect("equal")
    ax_bev.set_title(f"Frame {batch_dict['idx'][ib]}")

    # scan and cls label
    scan_r = batch_dict["scans"][ib][-1]
    scan_phi = batch_dict["scan_phi"][ib]
    scan_x, scan_y = u.rphi_to_xy(scan_r, scan_phi)

    target_cls = batch_dict["target_cls"][ib]
    ax_bev.scatter(scan_x[target_cls == -2], scan_y[target_cls == -2], s=1, c="yellow")
    ax_bev.scatter(scan_x[target_cls == -1], scan_y[target_cls == -1], s=1, c="orange")
    ax_bev.scatter(scan_x[target_cls == 0], scan_y[target_cls == 0], s=1, c="black")
    ax_bev.scatter(scan_x[target_cls > 0], scan_y[target_cls > 0], s=1, c="green")

    # annotation
    ann = batch_dict["dets_wp"][ib]
    ann_valid_mask = batch_dict["anns_valid_mask"][ib]
    if len(ann) > 0:
        ann = np.array(ann)
        det_x, det_y = u.rphi_to_xy(ann[:, 0], ann[:, 1])
        for x, y, valid in zip(det_x, det_y, ann_valid_mask):
            c = "blue" if valid else "orange"
            c = plt.Circle((x, y), radius=0.4, color=c, fill=False)
            ax_bev.add_artist(c)

    # reg label
    target_reg = batch_dict["target_reg"][ib]
    dets_r, dets_phi = u.canonical_to_global(
        scan_r, scan_phi, target_reg[:, 0], target_reg[:, 1]
    )
    dets_r = dets_r[target_cls > 0]
    dets_phi = dets_phi[target_cls > 0]
    dets_x, dets_y = u.rphi_to_xy(dets_r, dets_phi)
    ax_bev.scatter(dets_x, dets_y, s=10, c="red")

    # image
    ax_im.axis("off")
    ax_im.imshow(batch_dict["im_data"][ib]["stitched_image0"])

    # detection bounding box
    for box_dict in batch_dict["im_dets"][ib]:
        x0, y0, w, h = box_dict["box"]
        x1 = x0 + w
        y1 = y0 + h
        verts = _get_bounding_box_plotting_vertices(x0, y0, x1, y1)
        c = max(float(box_dict["score"]) - 0.5, 0) * 2.0
        ax_im.plot(verts[:, 0], verts[:, 1], c=(1.0 - c, 1.0 - c, 1.0))

    for box in batch_dict["pseudo_label_boxes"][ib]:
        x0, y0, x1, y1 = box
        verts = _get_bounding_box_plotting_vertices(x0, y0, x1, y1)
        ax_im.plot(verts[:, 0], verts[:, 1], c="green")

    # laser points on image
    scan_z = batch_dict["laser_z"][ib]
    scan_xyz_laser = np.stack((scan_x, -scan_y, scan_z), axis=0)  # in JRDB laser frame

    p_xy, ib_mask = jt.transform_pts_laser_to_stitched_im(scan_xyz_laser)
    c = np.clip(scan_r, 0.0, 20.0) / 20.0
    c = c.reshape(-1, 1).repeat(3, axis=1)
    c[:, 0] = 1.0
    ax_im.scatter(p_xy[0, ib_mask], p_xy[1, ib_mask], s=1, c=c[ib_mask])

    # save fig
    fig_file = os.path.join(_SAVE_DIR, f"figs/{sequence}/{frame_id}.png")
    os.makedirs(os.path.dirname(fig_file), exist_ok=True)

    fig.savefig(fig_file)
    plt.close(fig)


def _plot_pseudo_labels(batch_dict, ib):
    # pseudo labels
    pl_xy = batch_dict["pseudo_label_loc_xy"][ib]
    pl_boxes = batch_dict["pseudo_label_boxes"][ib]

    if len(pl_xy) == 0:
        return

    # groundtruth
    anns_rphi = np.array(batch_dict["dets_wp"][ib], dtype=np.float32)[
        batch_dict["anns_valid_mask"][ib]
    ]

    # match pseudo labels with groundtruth
    if len(anns_rphi) > 0:
        gts_x, gts_y = u.rphi_to_xy(anns_rphi[:, 0], anns_rphi[:, 1])

        x_diff = pl_xy[:, 0].reshape(-1, 1) - gts_x.reshape(1, -1)
        y_diff = pl_xy[:, 1].reshape(-1, 1) - gts_y.reshape(1, -1)
        d_diff = np.sqrt(x_diff * x_diff + y_diff * y_diff)
        match_found = d_diff < 0.3  # (pl, gt)
        match_found = match_found.max(axis=1)
    else:
        match_found = np.zeros(len(pl_xy), dtype=np.bool)

    # overlay image with laser
    im = batch_dict["im_data"][ib]["stitched_image0"]
    scan_r = batch_dict["scans"][ib][-1]
    scan_phi = batch_dict["scan_phi"][ib]
    scan_x, scan_y = u.rphi_to_xy(scan_r, scan_phi)
    scan_z = batch_dict["laser_z"][ib]
    scan_xyz_laser = np.stack((scan_x, -scan_y, scan_z), axis=0)  # in JRDB laser frame
    p_xy, ib_mask = jt.transform_pts_laser_to_stitched_im(scan_xyz_laser)
    p_xy = p_xy[:, ib_mask]
    c_bgr = _distance_to_bgr_color(scan_r[ib_mask])

    # plot
    frame_id = f"{batch_dict['frame_id'][ib]:06d}"
    sequence = batch_dict["sequence"][ib]

    for count, (xy, box, is_pos) in enumerate(zip(pl_xy, pl_boxes, match_found)):
        # image
        x0, y0, x1, y1 = box
        x0 = int(x0)
        x1 = int(x1)
        y0 = int(y0)
        y1 = int(y1)
        im_box = im[y0 : y1 + 1, x0 : x1 + 1]
        height = y1 - y0
        width = x1 - x0

        fig_w_inch = 0.314961 * 2.0
        fig_h_inch = 0.708661 * 2.0

        fig_im = plt.figure()
        fig_im.set_size_inches(fig_w_inch, fig_h_inch, forward=False)
        ax_im = plt.Axes(fig_im, [0.0, 0.0, 1.0, 1.0])
        ax_im.imshow(im_box)
        ax_im.set_axis_off()
        ax_im.axis(([0, width, height, 0]))
        ax_im.set_aspect((fig_h_inch / fig_w_inch) / (height / width))
        fig_im.add_axes(ax_im)

        in_box_mask = np.logical_and(
            np.logical_and(p_xy[0] >= x0, p_xy[0] <= x1),
            np.logical_and(p_xy[1] >= y0, p_xy[1] <= y1),
        )
        plt.scatter(
            p_xy[0, in_box_mask] - x0,
            p_xy[1, in_box_mask] - y0,
            s=3,
            c=c_bgr[in_box_mask],
        )

        pos_neg_dir = "true" if is_pos else "false"
        fig_file = os.path.join(
            _SAVE_DIR, f"samples/{sequence}/{pos_neg_dir}/{frame_id}_{count}_im.pdf"
        )
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        plt.savefig(fig_file, dpi=height / fig_h_inch)
        plt.close(fig_im)

        # lidar
        plot_range = 0.5
        close_mask = np.hypot(scan_x - xy[0], scan_y - xy[1]) < plot_range

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot()
        ax.set_aspect("equal")
        ax.axis("off")
        # ax.set_xlim(-plot_range, plot_range)
        # ax.set_ylim(-plot_range, plot_range)
        # ax.set_xlabel("x [m]")
        # ax.set_ylabel("y [m]")
        # ax.set_aspect("equal")
        # ax.set_title(f"Frame {batch_dict['idx'][ib]}")

        # plot points in local frame (so it looks aligned with image)
        ang = np.mean(scan_phi[close_mask]) - 0.5 * np.pi
        ca, sa = np.cos(ang), np.sin(ang)
        xy_plotting = np.array([[ca, sa], [-sa, ca]]) @ np.stack(
            (scan_x[close_mask] - xy[0], scan_y[close_mask] - xy[1]), axis=0
        )

        ax.scatter(
            -xy_plotting[0], xy_plotting[1], s=80, color=(191 / 255, 83 / 255, 79 / 255)
        )
        ax.scatter(
            0, 0, s=500, color=(18 / 255, 105 / 255, 176 / 255), marker="+", linewidth=5
        )

        fig_file = os.path.join(
            _SAVE_DIR, f"samples/{sequence}/{pos_neg_dir}/{frame_id}_{count}_pt.pdf"
        )
        fig.savefig(fig_file)
        plt.close(fig)

        # # --------
        # # This block plots 3d figure of points and camera frustum, very slow

        # # plot a camera frustum, use pixel coordinates, assuming the camera
        # # is located at the box center
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection="3d")

        # # put cam at box center
        # cam_x = 0.5 * im_box.shape[1]
        # cam_y = 0.5 * im_box.shape[0]
        # cam_z = 0.0

        # # image
        # im_z = 300  # image plane
        # im_xs, im_ys = np.meshgrid(
        #     np.arange(im_box.shape[1]), np.arange(im_box.shape[0])
        # )
        # im_zs = im_z * np.ones_like(im_xs)
        # im_xs -= int(cam_x)  # put camera center at (0, 0, 0)
        # im_ys -= int(cam_y)
        # im_zs -= int(cam_z)
        # ax.plot_surface(
        #     im_xs,
        #     im_ys,
        #     im_zs,
        #     rstride=1,
        #     cstride=1,
        #     facecolors=im_box / 255.0,
        #     shade=False,
        # )

        # # frustum
        # ax.plot(
        #     (0, im_xs.min()),
        #     (0, im_ys.min()),
        #     (0, im_zs.min()),
        #     color="black",
        #     linewidth=1.0,
        # )
        # ax.plot(
        #     (0, im_xs.max()),
        #     (0, im_ys.min()),
        #     (0, im_zs.min()),
        #     color="black",
        #     linewidth=1.0,
        # )
        # ax.plot(
        #     (0, im_xs.min()),
        #     (0, im_ys.max()),
        #     (0, im_zs.min()),
        #     color="black",
        #     linewidth=1.0,
        # )
        # ax.plot(
        #     (0, im_xs.max()),
        #     (0, im_ys.max()),
        #     (0, im_zs.min()),
        #     color="black",
        #     linewidth=1.0,
        # )

        # # lidar points
        # l_x = p_xy[0, in_box_mask] - x0 - cam_x
        # l_y = p_xy[1, in_box_mask] - y0 - cam_y
        # l_z = np.ones_like(l_x) * im_z - cam_z
        # l_r = scan_r[ib_mask][in_box_mask]
        # # for ease of viz compress z component
        # l_r = l_r / 2
        # cutoff = 7
        # l_r_compressed = l_r.copy()
        # l_r_compressed[l_r > cutoff] = (l_r[l_r > cutoff] - cutoff) / 4 + cutoff

        # # pop up to 3d, not exactly correct, but for viz is sufficient
        # l_x = l_x * l_r
        # l_y = 60 * np.ones_like(l_x)
        # l_z = l_z * l_r_compressed

        # ax.scatter3D(l_x, l_y, l_z, color="red", s=10)

        # # # plot k-means results, from utils.py generate_pseudo_labels
        # # # do a k-means clustering on r space, with k=2, to seperate close and far points
        # # laser_r_inside = scan_r[ib_mask][in_box_mask]
        # # c0, c1 = np.min(laser_r_inside), np.max(laser_r_inside)
        # # prev_c0, prev_c1 = 1e6, 1e6
        # # converge_thresh = 0.1
        # # iter_count, max_iter = 0, 1000
        # # while iter_count < max_iter and (
        # #     np.abs(c0 - prev_c0) > converge_thresh
        # #     or np.abs(c1 - prev_c1) > converge_thresh
        # # ):
        # #     c0_mask = np.abs(laser_r_inside - c0) < np.abs(laser_r_inside - c1)
        # #     prev_c0 = c0
        # #     prev_c1 = c1
        # #     c0 = np.mean(laser_r_inside[c0_mask])
        # #     c1 = np.mean(laser_r_inside[np.logical_not(c0_mask)])

        # # assert iter_count < max_iter

        # # ax.scatter3D(l_x[c0_mask], l_y[c0_mask], l_z[c0_mask], color="green", s=10)
        # # c1_mask = np.logical_not(c0_mask)
        # # ax.scatter3D(l_x[c1_mask], l_y[c1_mask], l_z[c1_mask], color="blue", s=10)

        # # Hack to get equal aspect ratio https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to  # noqa
        # # Create cubic bounding box to simulate equal aspect ratio
        # max_range = np.array(
        #     [l_x.max() - l_x.min(), l_y.max() - l_y.min(), l_z.max() - 0]
        # ).max()
        # Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (
        #     l_x.max() + l_x.min()
        # )
        # Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (
        #     im_ys.max() + im_ys.min()
        # )
        # Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (
        #     l_z.max() + 0
        # )
        # # Comment or uncomment following both lines to test the fake bounding box:
        # for xbb, ybb, zbb in zip(Xb, Yb, Zb):
        #     ax.plot([xbb], [ybb], [zbb], "w")

        # ax.grid(False)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # ax.view_init(elev=-30, azim=-90)

        # # https://stackoverflow.com/questions/11448972/changing-the-background-color-of-the-axes-planes-of-a-matplotlib-3d-plot  # noqa
        # ax.xaxis.pane.fill = False
        # ax.yaxis.pane.fill = False
        # ax.zaxis.pane.fill = False

        # # Now set color to white (or whatever is "invisible")
        # ax.xaxis.pane.set_edgecolor("w")
        # ax.yaxis.pane.set_edgecolor("w")
        # ax.zaxis.pane.set_edgecolor("w")

        # fig_file = os.path.join(
        #     _SAVE_DIR, f"samples/{sequence}/{pos_neg_dir}/{frame_id}_{count}_viz.pdf"
        # )
        # plt.savefig(fig_file)
        # plt.close(fig)

        # return
        # # --------


def _get_bounding_box_plotting_vertices(x0, y0, x1, y1):
    return np.array([(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)])


def _write_file_make_dir(f_name, f_str):
    os.makedirs(os.path.dirname(f_name), exist_ok=True)
    with open(f_name, "w") as f:
        f.write(f_str)


def generate_pseudo_labels():
    with open("./cfgs/base_dr_spaam_jrdb_cfg.yaml", "r") as f:
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

    if os.path.exists(_SAVE_DIR):
        shutil.rmtree(_SAVE_DIR)
        time.sleep(1.0)
    os.makedirs(_SAVE_DIR)

    # generate pseudo labels for all sample
    for count, batch_dict in enumerate(tqdm(test_loader)):
        if count >= _MAX_COUNT:
            break

        for ib in range(len(batch_dict["input"])):
            if count % _PLOTTING_INTERVAL == 0:
                # # visualize the whole frame
                # _plot_frame(batch_dict, ib)

                # visualize each pseudo labels
                _plot_pseudo_labels(batch_dict, ib)


if __name__ == "__main__":
    generate_pseudo_labels()
