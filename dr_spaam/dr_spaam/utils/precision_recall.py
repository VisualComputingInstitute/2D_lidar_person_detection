from __future__ import absolute_import, print_function

import glob
import os
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import auc

# For plotting using lab cluster server
# https://github.com/matplotlib/matplotlib/issues/3466/
plt.switch_backend("agg")


def drow_detection_to_kitti_string(dets_xy, dets_cls, occluded):
    """Obtain a KITTI format string for DROW detection

    Args:
        dets_xy (np.array[N, 2])
        dets_cls (np.array[N])
        occluded (np.array[N]): Only revelent for annotation. Set to 1 if the
            annotation is not visible (less than 5 points in proximity)

    Returns:
        s (str)
    """
    if dets_cls is None:
        dets_cls = np.ones(len(dets_xy), dtype=np.float32)

    if occluded is None:
        occluded = np.zeros(len(dets_xy), dtype=np.int)

    s = ""
    for cls, xy, occ in zip(dets_cls, dets_xy, occluded):
        s += "Pedestrian 0 {} 0 0 0 0 0 0 0 0 0 {} {} 0 0 {}\n".format(
            occ, xy[0], xy[1], cls)
    s = s.strip("\n")

    return s


def kitti_string_to_drow_detection(s):
    dets_xy = []
    dets_cls = []
    occluded = []

    if s:
        lines = s.split("\n")
        for line in lines:
            vals = line.split(" ")
            dets_cls.append(float(vals[-1]))
            dets_xy.append((float(vals[-5]), float(vals[-4])))
            occluded.append(int(vals[2]))

    dets_cls = np.array(dets_cls, dtype=np.float32)
    dets_xy = np.array(dets_xy, dtype=np.float32)
    occluded = np.array(occluded, dtype=np.int)

    return dets_xy, dets_cls, occluded


def evaluate_drow(result_dir, verbose=True, remove_raw_files=False):
    """Expect results to be saved in a directory with following layout

    result_dir
    |-- detections
    |   |-- <sequence_name_0>
    |   |   |-- 000000.txt
    |   |   |-- 000001.txt
    |   |   |-- ...
    |   |-- <sequence_name_1>
    |   |   |-- 000000.txt
    |   |   |-- 000001.txt
    |   |   |-- ...
    |   |-- ...
    |-- groundtruth
    |   |-- <sequence_name_0>
    |   |   |-- 000000.txt
    |   |   |-- 000001.txt
    |   |   |-- ...
    |   |-- <sequence_name_1>
    |   |   |-- 000000.txt
    |   |   |-- 000001.txt
    |   |   |-- ...
    |   |-- ...

    Each result file should follow KITTI format, c.f. drow_detection_to_kitti_string

    If `remove_raw_files` set to true, files in the `detections` and the `groundtruth`
    directory will be removed after evaluation. This could take a while.

    Returns:
        sequences (list[str]): sequence names
        sequences_results_03 (list[dict]): sequence results, with 0.3 association
            radius, see get_precision_recall
        sequences_results_05 (list[dict])
    """
    det_dir = os.path.join(result_dir, "detections")

    sequences = os.listdir(det_dir)
    sequences_results_03 = []
    sequences_results_05 = []
    sequences_results_08 = []

    sequences_dets_xy = []
    sequences_dets_cls = []
    sequences_dets_inds = []
    sequences_gts_xy = []
    sequences_gts_inds = []

    counter = 0
    # evaluate each sequence
    for sequence in sequences:
        if verbose:
            print("Evaluating {}".format(sequence))

        dets_xy_accumulate = []
        dets_cls_accumulate = []
        dets_inds_accumulate = []
        gts_xy_accumulate = []
        gts_inds_accumulate = []

        # accumulate detections in all frames
        for det_file in glob.glob(os.path.join(det_dir, sequence, "*.txt")):
            counter += 1

            with open(det_file, "r") as f:
                dets_xy, dets_cls, _ = kitti_string_to_drow_detection(f.read())

            gt_file = det_file.replace("detections", "groundtruth")
            with open(gt_file, "r") as f:
                gts_xy, _, gts_occluded = kitti_string_to_drow_detection(
                    f.read())

            if remove_raw_files:
                os.remove(det_file)
                os.remove(gt_file)

            # evaluate only on visible groundtruth
            gts_xy = gts_xy[gts_occluded == 0]

            if len(dets_xy) > 0:
                dets_xy_accumulate.append(dets_xy)
                dets_cls_accumulate.append(dets_cls)
                dets_inds_accumulate += [counter] * len(dets_xy)

            if len(gts_xy) > 0:
                gts_xy_accumulate.append(gts_xy)
                gts_inds_accumulate += [counter] * len(gts_xy)

        dets_xy = np.concatenate(dets_xy_accumulate, axis=0)
        dets_cls = np.concatenate(dets_cls_accumulate)
        dets_inds = np.array(dets_inds_accumulate)
        gts_xy = np.concatenate(gts_xy_accumulate, axis=0)
        gts_inds = np.array(gts_inds_accumulate)

        # evaluate sequence
        sequences_results_03.append(
            get_precision_recall(
                dets_xy, dets_cls, dets_inds, gts_xy, gts_inds, association_radius=0.3,
            )
        )

        sequences_results_05.append(
            get_precision_recall(
                dets_xy, dets_cls, dets_inds, gts_xy, gts_inds, association_radius=0.5,
            )
        )

        sequences_results_08.append(
            get_precision_recall(
                dets_xy, dets_cls, dets_inds, gts_xy, gts_inds, association_radius=0.8,
            )
        )

        # store sequence detections and groundtruth for dataset evaluation
        sequences_dets_xy.append(dets_xy)
        sequences_dets_cls.append(dets_cls)
        sequences_dets_inds.append(dets_inds)
        sequences_gts_xy.append(gts_xy)
        sequences_gts_inds.append(gts_inds)

    # evaluate all dataset
    if len(sequences) > 1:
        if verbose:
            print("Evaluating whole dataset")
        sequences.append("all")

        dets_xy = np.concatenate(sequences_dets_xy, axis=0)
        dets_cls = np.concatenate(sequences_dets_cls)
        dets_inds = np.concatenate(sequences_dets_inds)
        gts_xy = np.concatenate(sequences_gts_xy, axis=0)
        gts_inds = np.concatenate(sequences_gts_inds)

        sequences_results_03.append(
            get_precision_recall(
                dets_xy, dets_cls, dets_inds, gts_xy, gts_inds, association_radius=0.3,
            )
        )

        sequences_results_05.append(
            get_precision_recall(
                dets_xy, dets_cls, dets_inds, gts_xy, gts_inds, association_radius=0.5,
            )
        )

        sequences_results_08.append(
            get_precision_recall(
                dets_xy, dets_cls, dets_inds, gts_xy, gts_inds, association_radius=0.8,
            )
        )

    return sequences, sequences_results_03, sequences_results_05, sequences_results_08


def evaluate_drow_one_hot(result_dir, dist_bins=None, verbose=True):
    """Similar to evaluate_drow(), except detections are regarded as binary without
    confidence value. A detection is positive if there is a groundtruth within
    the association radius, otherwise negative. One groundtruth can be matched to
    multiple detections. Used to evaluate pseudo labels.

    Returns:
        sequences (list[str]): sequence names
        sequences_results_03 (list[dict]): sequence results, with 0.3 association
            radius, see get_precision_recall
        sequences_results_05 (list[dict])
    """
    det_dir = os.path.join(result_dir, "detections")

    sequences = os.listdir(det_dir)
    sequences_results_03 = []
    sequences_results_05 = []
    sequence_results_08 = []

    sequences_dets_xy = []
    sequences_dets_inds = []
    sequences_gts_xy = []
    sequences_gts_inds = []

    counter = 0
    # evaluate each sequence
    for sequence in sequences:
        if verbose:
            print("Evaluating {}".format(sequence))

        dets_xy_accumulate = []
        dets_inds_accumulate = []
        gts_xy_accumulate = []
        gts_inds_accumulate = []

        # accumulate detections in all frames
        for det_file in glob.glob(os.path.join(det_dir, sequence, "*.txt")):
            counter += 1

            with open(det_file, "r") as f:
                dets_xy, _, _ = kitti_string_to_drow_detection(f.read())

            gt_file = det_file.replace("detections", "groundtruth")
            with open(gt_file, "r") as f:
                gts_xy, _, gts_occluded = kitti_string_to_drow_detection(
                    f.read())

            # evaluate only on visiable groundtruth
            gts_xy = gts_xy[gts_occluded == 0]

            if len(dets_xy) > 0:
                dets_xy_accumulate.append(dets_xy)
                dets_inds_accumulate += [counter] * len(dets_xy)

            if len(gts_xy) > 0:
                gts_xy_accumulate.append(gts_xy)
                gts_inds_accumulate += [counter] * len(gts_xy)

        dets_xy = np.concatenate(dets_xy_accumulate, axis=0)
        dets_inds = np.array(dets_inds_accumulate)
        gts_xy = np.concatenate(gts_xy_accumulate, axis=0)
        gts_inds = np.array(gts_inds_accumulate)

        # evaluate sequence
        sequences_results_03.append(
            get_precision_recall_one_hot(
                dets_xy, dets_inds, gts_xy, gts_inds, 0.3, dist_bins=dist_bins
            )
        )

        sequences_results_05.append(
            get_precision_recall_one_hot(
                dets_xy, dets_inds, gts_xy, gts_inds, 0.5, dist_bins=dist_bins
            )
        )

        sequences_results_08.append(
            get_precision_recall_one_hot(
                dets_xy, dets_inds, gts_xy, gts_inds, 0.8, dist_bins=dist_bins
            )
        )

        if verbose:
            print(
                "precision_0.3 {} ".format(sequences_results_03[-1][0]),
                "recall_0.3 {}\n".format(sequences_results_03[-1][1]),
                "precision_0.5 {} ".format(sequences_results_05[-1][0]),
                "recall_0.5 {}\n".format(sequences_results_05[-1][1]),
                "precision_0.8 {} ".format(sequences_results_08[-1][0]),
                "recall_0.8 {}\n".format(sequences_results_08[-1][1])
            )

        # store sequence detections and groundtruth for whole dataset evaluation
        sequences_dets_xy.append(dets_xy)
        sequences_dets_inds.append(dets_inds)
        sequences_gts_xy.append(gts_xy)
        sequences_gts_inds.append(gts_inds)

    # evaluate all dataset
    if len(sequences) > 1:
        if verbose:
            print("Evaluating whole dataset")
        sequences.append("all")

        dets_xy = np.concatenate(sequences_dets_xy, axis=0)
        dets_inds = np.concatenate(sequences_dets_inds)
        gts_xy = np.concatenate(sequences_gts_xy, axis=0)
        gts_inds = np.concatenate(sequences_gts_inds)

        sequences_results_03.append(
            get_precision_recall_one_hot(
                dets_xy, dets_inds, gts_xy, gts_inds, 0.3, dist_bins=dist_bins
            )
        )

        sequences_results_05.append(
            get_precision_recall_one_hot(
                dets_xy, dets_inds, gts_xy, gts_inds, 0.5, dist_bins=dist_bins
            )
        )

        sequences_results_08.append(
            get_precision_recall_one_hot(
                dets_xy, dets_inds, gts_xy, gts_inds, 0.8, dist_bins=dist_bins
            )
        )

        if verbose:
            print(
                "precision_0.3 {} ".format(sequences_results_03[-1][0]),
                "recall_0.3 {}\n".format(sequences_results_03[-1][1]),
                "precision_0.5 {} ".format(sequences_results_05[-1][0]),
                "recall_0.5 {}\n".format(sequences_results_05[-1][1]),
                "precision_0.8 {} ".format(sequences_results_08[-1][0]),
                "recall_0.8 {}\n".format(sequences_results_08[-1][1])
            )

    return sequences, sequences_results_03, sequences_results_05, sequences_results_08


def get_precision_recall(
    dets_xy, dets_cls, dets_inds, anns_xy, anns_inds, association_radius
):
    a_rad = association_radius * np.ones(len(anns_inds), dtype=np.float32)
    recalls, precisions, thresholds = _prec_rec_2d(
        dets_cls, dets_xy, dets_inds, anns_xy, anns_inds, a_rad
    )
    ap, peak_f1, eer = _eval_prec_rec(recalls, precisions)
    return {
        "precisions": precisions,
        "recalls": recalls,
        "thresholds": thresholds,
        "ap": ap,
        "peak_f1": peak_f1,
        "eer": eer,
    }


def plot_pr_curve(precisions, recalls, plot_title=None, output_file=None):
    fig, ax = _plot_prec_rec_wps_only(
        wps=(recalls, precisions), title=plot_title)

    if output_file is not None:
        plt.savefig(output_file, bbox_inches="tight")

    return fig, ax


def get_precision_recall_one_hot(
    dets_xy, dets_inds, gts_xy, gts_inds, assoc_radius, dist_bins=None
):
    tp = 0  # true positive
    fp = 0
    gt_matched = 0

    # get distance distribution
    if dist_bins is not None:
        gt_hist = np.zeros(len(dist_bins), dtype=np.int)
        tp_hist = np.zeros(len(dist_bins), dtype=np.int)
        fp_hist = np.zeros(len(dist_bins), dtype=np.int)

    # accmulate results for all frames
    max_idx = max(dets_inds.max(), gts_inds.max())
    min_idx = min(dets_inds.min(), gts_inds.min())
    for i in range(min_idx, max_idx + 1):
        dets_xy_i = dets_xy[dets_inds == i]
        gts_xy_i = gts_xy[gts_inds == i]

        if len(dets_xy_i) == 0:
            if len(gts_xy_i) > 0 and dist_bins is not None:
                gt_hist = _increment_dist_hist_count(
                    dist_bins, gts_xy_i, gt_hist)
            continue

        if len(gts_xy_i) == 0:
            fp += len(dets_xy_i)
            if dist_bins is not None:
                fp_hist = _increment_dist_hist_count(
                    dist_bins, dets_xy_i, fp_hist)
        else:
            x_diff = dets_xy_i[:,
                               0].reshape(-1, 1) - gts_xy_i[:, 0].reshape(1, -1)
            y_diff = dets_xy_i[:,
                               1].reshape(-1, 1) - gts_xy_i[:, 1].reshape(1, -1)
            match_found = (
                np.sqrt(x_diff * x_diff + y_diff * y_diff) < assoc_radius
            )  # (dets, gts)

            tp_mask = match_found.max(axis=1)
            tp += tp_mask.sum()
            fp += len(tp_mask) - tp_mask.sum()
            gt_matched += match_found.max(axis=0).sum()

            # distance distribution
            if dist_bins is not None:
                gt_hist = _increment_dist_hist_count(
                    dist_bins, gts_xy_i, gt_hist)
                tp_hist = _increment_dist_hist_count(
                    dist_bins, dets_xy_i[tp_mask], tp_hist
                )
                fp_hist = _increment_dist_hist_count(
                    dist_bins, dets_xy_i[np.logical_not(tp_mask)], fp_hist
                )

    precision = float(tp) / float(tp + fp)
    recall = float(gt_matched) / float(len(gts_inds))

    if dist_bins is None:
        return precision, recall
    else:
        return precision, recall, gt_hist, tp_hist, fp_hist


def _increment_dist_hist_count(dist_bins, pts_xy, hist_count):
    dist = np.hypot(pts_xy[:, 0], pts_xy[:, 1])
    bins_inds = np.abs(dist.reshape(-1, 1) -
                       dist_bins.reshape(1, -1)).argmin(axis=1)
    np.add.at(hist_count, bins_inds, 1)

    return hist_count


# Most of the code below comes from
# https://github.com/VisualComputingInstitute/DROW/blob/master/v2/utils/__init__.py


def _prec_rec_2d(det_scores, det_coords, det_frames, gt_coords, gt_frames, gt_radii):
    """ Computes full precision-recall curves at all possible thresholds.

    Arguments:
    - `det_scores` (D,) array containing the scores of the D detections.
    - `det_coords` (D,2) array containing the (x,y) coordinates of the D detections.
    - `det_frames` (D,) array containing the frame number of each of the D detections.
    - `gt_coords` (L,2) array containing the (x,y) coordinates of the L labels (ground-truth detections).
    - `gt_frames` (L,) array containing the frame number of each of the L labels.
    - `gt_radii` (L,) array containing the radius at which each of the L labels should consider detection associations.
                      This will typically just be an np.full_like(gt_frames, 0.5) or similar,
                      but could vary when mixing classes, for example.

    Returns: (recs, precs, threshs)
    - `threshs`: (D,) array of sorted thresholds (scores), from higher to lower.
    - `recs`: (D,) array of recall scores corresponding to the thresholds.
    - `precs`: (D,) array of precision scores corresponding to the thresholds.
    """
    # This means that all reported detection frames which are not in ground-truth frames
    # will be counted as false-positives.
    # TODO: do some sanity-checks in the "linearization" functions before calling `prec_rec_2d`.
    frames = np.unique(np.r_[det_frames, gt_frames])

    det_accepted_idxs = defaultdict(list)
    tps = np.zeros(len(frames), dtype=np.uint32)
    fps = np.zeros(len(frames), dtype=np.uint32)
    fns = np.array([np.sum(gt_frames == f) for f in frames], dtype=np.uint32)

    precs = np.full_like(det_scores, np.nan)
    recs = np.full_like(det_scores, np.nan)
    threshs = np.full_like(det_scores, np.nan)

    # mergesort for determinism.
    indices = np.argsort(det_scores, kind="mergesort")
    for i, idx in enumerate(reversed(indices)):
        frame = det_frames[idx]
        iframe = np.where(frames == frame)[0][0]  # Can only be a single one.

        # Accept this detection
        dets_idxs = det_accepted_idxs[frame]
        dets_idxs.append(idx)
        threshs[i] = det_scores[idx]

        dets = det_coords[dets_idxs]

        gts_mask = gt_frames == frame
        gts = gt_coords[gts_mask]
        radii = gt_radii[gts_mask]

        if len(gts) == 0:  # No GT, but there is a detection.
            fps[iframe] += 1
        else:  # There is GT and detection in this frame.
            not_in_radius = radii[:, None] < cdist(
                gts, dets
            )  # -> ngts x ndets, True (=1) if too far, False (=0) if may match.
            igt, idet = linear_sum_assignment(not_in_radius)

            tps[iframe] = np.sum(
                np.logical_not(not_in_radius[igt, idet])
            )  # Could match within radius
            fps[iframe] = (
                len(dets) - tps[iframe]
            )  # NB: dets is only the so-far accepted.
            fns[iframe] = len(gts) - tps[iframe]

        tp, fp, fn = np.sum(tps), np.sum(fps), np.sum(fns)
        precs[i] = tp / (fp + tp) if fp + tp > 0 else np.nan
        recs[i] = tp / (fn + tp) if fn + tp > 0 else np.nan

    return recs, precs, threshs


def _eval_prec_rec(rec, prec):
    # make sure the x-input to auc is sorted
    assert np.sum(np.diff(rec) >= 0) == len(rec) - 1
    # compute error matrices
    return auc(rec, prec), _peakf1(rec, prec), _eer(rec, prec)


def _peakf1(recs, precs):
    return np.max(2 * precs * recs / np.clip(precs + recs, 1e-16, 2 + 1e-16))


def _eer(recs, precs):
    # Find the first nonzero or else (0,0) will be the EER :)
    def first_nonzero_idx(arr):
        return np.where(arr != 0)[0][0]

    p1 = first_nonzero_idx(precs)
    r1 = first_nonzero_idx(recs)
    idx = np.argmin(np.abs(precs[p1:] - recs[r1:]))
    return (
        precs[p1 + idx] + recs[r1 + idx]
    ) / 2  # They are often the exact same, but if not, use average.


def _plot_prec_rec(wds, wcs, was, wps, figsize=(15, 10), title=None):
    fig, ax = plt.subplots(figsize=figsize)

    # make sure the x-input to auc is sorted
    assert np.sum(np.diff(wds[0]) >= 0) == len(wds[0]) - 1
    assert np.sum(np.diff(wcs[0]) >= 0) == len(wcs[0]) - 1
    assert np.sum(np.diff(was[0]) >= 0) == len(was[0]) - 1
    assert np.sum(np.diff(wps[0]) >= 0) == len(wps[0]) - 1

    ax.plot(
        *wds[:2],
        label="agn (AUC: {:.1%}, F1: {:.1%}, EER: {:.1%})".format(
            auc(*wds[:2]), _peakf1(*wds[:2]), _eer(*wds[:2])
        ),
        c="#E24A33"
    )
    ax.plot(
        *wcs[:2],
        label="wcs (AUC: {:.1%}, F1: {:.1%}, EER: {:.1%})".format(
            auc(*wcs[:2]), _peakf1(*wcs[:2]), _eer(*wcs[:2])
        ),
        c="#348ABD"
    )
    ax.plot(
        *was[:2],
        label="was (AUC: {:.1%}, F1: {:.1%}, EER: {:.1%})".format(
            auc(*was[:2]), _peakf1(*was[:2]), _eer(*was[:2])
        ),
        c="#988ED5"
    )
    ax.plot(
        *wps[:2],
        label="wps (AUC: {:.1%}, F1: {:.1%}, EER: {:.1%})".format(
            auc(*wps[:2]), _peakf1(*wps[:2]), _eer(*wps[:2])
        ),
        c="#8EBA42"
    )

    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.91)

    _prettify_pr_curve(ax)
    _lbplt_fatlegend(ax, loc="upper right")

    return fig, ax


def _plot_prec_rec_wps_only(wps, figsize=(15, 10), title=None):
    fig, ax = plt.subplots(figsize=figsize)

    # make sure the x-input to auc is sorted
    assert np.sum(np.diff(wps[0]) >= 0) == len(wps[0]) - 1

    ax.plot(
        *wps[:2],
        label="wps (AUC: {:.1%}, F1: {:.1%}, EER: {:.1%})".format(
            auc(*wps[:2]), _peakf1(*wps[:2]), _eer(*wps[:2])
        ),
        c="#8EBA42"
    )

    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.91)

    _prettify_pr_curve(ax)
    _lbplt_fatlegend(ax, loc="upper right")
    return fig, ax


def _prettify_pr_curve(ax):
    ax.plot([0, 1], [0, 1], ls="--", c=".6")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Recall [%]")
    ax.set_ylabel("Precision [%]")
    ax.axes.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, pos: "{:.0f}".format(x * 100))
    )
    ax.axes.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, pos: "{:.0f}".format(x * 100))
    )
    return ax


def _lbplt_fatlegend(ax=None, *args, **kwargs):
    # Copy paste from lbtoolbox.plotting.fatlegend
    if ax is not None:
        leg = ax.legend(*args, **kwargs)
    else:
        leg = plt.legend(*args, **kwargs)

    for lh in leg.legendHandles:
        lh.set_linewidth(lh.get_linewidth() * 2.0)
        lh.set_alpha(1)
    return leg
