import cv2
import os
import shutil
import time
from tqdm import tqdm
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from dr_spaam.dataset.get_dataloader import get_dataloader
import dr_spaam.utils.jrdb_transforms as jt
import dr_spaam.utils.precision_recall as pru
import dr_spaam.utils.utils as u

_X_LIM = (-7, 7)
_Y_LIM = (-7, 7)

_PLOTTING_INTERVAL = 20
_MAX_COUNT = 1e9

_DIST_BINS = np.arange(30) + 0.5

_SPLIT = "train"
# _SPLIT = "val"

_SAVE_DIR = f"/globalwork/jia/tmp/new_plot_analyze_pseudo_labels/{_SPLIT}"
# _SAVE_DIR = f"/home/jia/tmp_imgs/analyze_pseudo_labels/{_SPLIT}"


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

    far_v = 1.0
    far_s = 0
    close_v = 0.75
    close_s = 0.59
    dist_normalized = np.clip(scan_r[ib_mask], 0.0, 20.0) / 20.0

    c_hsv = np.empty((1, p_xy.shape[1], 3), dtype=np.float32)
    c_hsv[0, :, 0] = 0.0
    # c_hsv[0, :, 1] = 1.0 - np.clip(scan_r[ib_mask], 0.0, 20.0) / 20.0
    c_hsv[0, :, 1] = close_s * (1.0 - dist_normalized) + far_s * dist_normalized
    c_hsv[0, :, 2] = close_v * (1.0 - dist_normalized) + far_v * dist_normalized
    c_bgr = cv2.cvtColor(c_hsv, cv2.COLOR_HSV2RGB)[0]

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


def _get_bounding_box_plotting_vertices(x0, y0, x1, y1):
    return np.array([(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)])


def _write_file_make_dir(f_name, f_str):
    os.makedirs(os.path.dirname(f_name), exist_ok=True)
    with open(f_name, "w") as f:
        f.write(f_str)


def generate_pseudo_labels():
    with open("./base_dr_spaam_jrdb_cfg.yaml", "r") as f:
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

    sequences_tp_fp_tn_fn = {}  # for computing true positive and negative rate

    # generate pseudo labels for all sample
    for count, batch_dict in enumerate(tqdm(test_loader)):
        if count >= _MAX_COUNT:
            break

        for ib in range(len(batch_dict["input"])):
            frame_id = f"{batch_dict['frame_id'][ib]:06d}"
            sequence = batch_dict["sequence"][ib]

            if count % _PLOTTING_INTERVAL == 0:
                # visualize the whole frame
                _plot_frame(batch_dict, ib)

                # visualize each pseudo labels
                _plot_pseudo_labels(batch_dict, ib)

            # save pseudo labels as detection results for evaluation
            pl_xy = batch_dict["pseudo_label_loc_xy"][ib]
            pl_str = (
                pru.drow_detection_to_kitti_string(pl_xy, None, None)
                if len(pl_xy) > 0
                else ""
            )

            pl_file = os.path.join(_SAVE_DIR, f"detections/{sequence}/{frame_id}.txt")
            _write_file_make_dir(pl_file, pl_str)

            # save groundtruth
            anns_rphi = batch_dict["dets_wp"][ib]
            if len(anns_rphi) > 0:
                anns_rphi = np.array(anns_rphi, dtype=np.float32)
                gts_xy = np.stack(
                    u.rphi_to_xy(anns_rphi[:, 0], anns_rphi[:, 1]), axis=1
                )
                gts_occluded = np.logical_not(batch_dict["anns_valid_mask"][ib]).astype(
                    np.int
                )
                gts_str = pru.drow_detection_to_kitti_string(gts_xy, None, gts_occluded)
            else:
                gts_str = ""

            gts_file = os.path.join(_SAVE_DIR, f"groundtruth/{sequence}/{frame_id}.txt")
            _write_file_make_dir(gts_file, gts_str)

            # compute true positive and negative rate
            target_cls = batch_dict["target_cls"][ib]
            target_cls_gt = batch_dict["target_cls_real"][ib]

            tn = np.sum(
                np.logical_or(
                    np.logical_and(target_cls == 0, target_cls_gt == 0),
                    np.logical_and(target_cls == 0, target_cls_gt == -1),
                )
            )
            fn = np.sum(target_cls == 0) - tn
            tp = np.sum(
                np.logical_or(
                    np.logical_and(target_cls == 1, target_cls_gt == 1),
                    np.logical_and(target_cls == 1, target_cls_gt == -1),
                )
            )
            fp = np.sum(target_cls == 1) - tp

            if sequence in sequences_tp_fp_tn_fn.keys():
                tp0, fp0, tn0, fn0 = sequences_tp_fp_tn_fn[sequence]
                sequences_tp_fp_tn_fn[sequence] = (
                    tp + tp0,
                    fp + fp0,
                    tn + tn0,
                    fn + fn0,
                )
            else:
                sequences_tp_fp_tn_fn[sequence] = (tp, fp, tn, fn)

    # write sequence statistics to file
    for sequence, (tp, fp, tn, fn) in sequences_tp_fp_tn_fn.items():
        st_file = os.path.join(_SAVE_DIR, f"evaluation/{sequence}/tp_fp_tn_fn.txt")
        _write_file_make_dir(st_file, f"{tp},{fp},{tn},{fn}")


def evaluate_pseudo_labels():
    sequences, sequences_results_03, sequences_results_05 = pru.evaluate_drow_one_hot(
        _SAVE_DIR, dist_bins=_DIST_BINS
    )

    tp_acc = 0
    fp_acc = 0
    tn_acc = 0
    fn_acc = 0
    for sequence, re03, re05 in zip(
        sequences, sequences_results_03, sequences_results_05
    ):
        result_str = (
            f"precision_0.3 {re03[0]}\n"
            f"recall_0.3 {re03[1]}\n"
            f"precision_0.5 {re05[0]}\n"
            f"recall_0.5 {re05[1]}\n"
        )

        if sequence != "all":
            with open(
                os.path.join(_SAVE_DIR, f"evaluation/{sequence}/tp_fp_tn_fn.txt"), "r"
            ) as f:
                tp, fp, tn, fn = f.readlines()[0].split(",")
                tp = int(tp)
                fp = int(fp)
                tn = int(tn)
                fn = int(fn)

            result_str += (
                f"true_positive {tp}\n"
                f"total_positive {tp + fp}\n"
                f"true_negative {tn}\n"
                f"total_negative {tn + fn}\n"
                f"true_positive_rate {float(tp) / float(tp + fp)}\n"
                f"true_negative_rate {float(tn) / float(tn + fn)}\n"
            )

            tp_acc += tp
            fp_acc += fp
            tn_acc += tn
            fn_acc += fn

        result_file = os.path.join(_SAVE_DIR, f"evaluation/{sequence}/result.txt")
        _write_file_make_dir(result_file, result_str)

        gt_hist = re05[2]
        tp_hist = re05[3]
        fp_hist = re05[4]

        gt_dist_file = os.path.join(
            _SAVE_DIR, f"evaluation/{sequence}/det_hist_jrdb_train.csv"
        )
        gt_dist_str = "distance, count\n"
        for b, v in zip(_DIST_BINS, gt_hist):
            gt_dist_str += f"{int(b + 0.5):>8}, {int(v):>5}\n"
        _write_file_make_dir(gt_dist_file, gt_dist_str)

        pl_dist_file = os.path.join(
            _SAVE_DIR, f"evaluation/{sequence}/det_hist_jrdb_train_pseudo.csv"
        )
        pl_dist_str = "distance, count_true, count_false\n"
        for b, v_tp, v_fp in zip(_DIST_BINS, tp_hist, fp_hist):
            pl_dist_str += f"{int(b + 0.5):>8}, {int(v_tp):>10}, {int(v_fp):>11}\n"
        _write_file_make_dir(pl_dist_file, pl_dist_str)

    # statistics for the whole split
    result_str = (
        f"true_positive {tp_acc}\n"
        f"total_positive {tp_acc + fp_acc}\n"
        f"true_negative {tn_acc}\n"
        f"total_negative {tn_acc + fn_acc}\n"
        f"true_positive_rate {float(tp_acc) / float(tp_acc + fp_acc)}\n"
        f"true_negative_rate {float(tn_acc) / float(tn_acc + fn_acc)}\n"
    )

    with open(os.path.join(_SAVE_DIR, "evaluation/all/result.txt"), "a") as f:
        f.write(result_str)


def display_evaluation_result():
    sequences = os.listdir(os.path.join(_SAVE_DIR, "evaluation"))
    for seq in sequences:
        f_name = os.path.join(_SAVE_DIR, "evaluation", seq, "result.txt")
        if os.path.isfile(f_name):
            with open(f_name, "r") as f:
                print(seq)
                print("".join(f.readlines()))


if __name__ == "__main__":
    generate_pseudo_labels()
    evaluate_pseudo_labels()
    display_evaluation_result()
