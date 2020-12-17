import math
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial.distance import cdist
import torch
import cv2

# from nms import nms

# In numpy >= 1.17, np.clip is slow, use core.umath.clip instead
# https://github.com/numpy/numpy/issues/14281
if "clip" in dir(np.core.umath):
    _clip = np.core.umath.clip
    # print("use np.core.umath.clip")
else:
    _clip = np.clip
    # print("use np.clip")


def get_drow_laser_phi():
    """DROW use a SICK S300 laser, which has 450 points and 0.5 deg sampling resolution,
    mounted at 37 cm height.
    """
    angle_inc = np.radians(0.5)
    num_pts = 450
    laser_fov = (num_pts - 1) * angle_inc  # 450 points
    return np.linspace(-laser_fov * 0.5, laser_fov * 0.5, num_pts)


def get_jrdb_laser_phi():
    """DROW use a SICK S300 laser, which has 450 points and 0.5 deg sampling resolution,
    mounted at 37 cm height.
    """
    # raise DeprecationWarning
    angle_inc = np.radians(0.5)
    num_pts = 450
    laser_fov = (num_pts - 1) * angle_inc  # 450 points
    return np.linspace(-laser_fov * 0.5, laser_fov * 0.5, num_pts)


def xy_to_rphi(x, y):
    """DROW xy frame convention: x-forward (center of the scan), y-right, z-downward,
    phi is the angle w.r.t. x axis
    """
    return np.hypot(x, y), np.arctan2(y, x)


def rphi_to_xy(r, phi):
    return r * np.cos(phi), r * np.sin(phi)


def rphi_to_xy_torch(r, phi):
    return r * torch.cos(phi), r * torch.sin(phi)


def global_to_canonical(scan_r, scan_phi, dets_r, dets_phi):
    """DROW canonical frame convention: origin at the scan points, y forward
    along scan ray, x rightward"""
    dx = np.sin(dets_phi - scan_phi) * dets_r
    dy = np.cos(dets_phi - scan_phi) * dets_r - scan_r
    return dx, dy


def canonical_to_global(scan_r, scan_phi, dx, dy):
    tmp_y = scan_r + dy
    tmp_phi = np.arctan2(
        dx, tmp_y
    )  # dx first is correct due to problem geometry dx -> y axis and vice versa.
    dets_phi = tmp_phi + scan_phi
    dets_r = tmp_y / np.cos(tmp_phi)
    return dets_r, dets_phi


def canonical_to_global_torch(scan_r, scan_phi, dx, dy):
    tmp_y = scan_r + dy
    tmp_phi = torch.atan2(
        dx, tmp_y
    )  # dx first is correct due to problem geometry dx -> y axis and vice versa.
    dets_phi = tmp_phi + scan_phi
    dets_r = tmp_y / torch.cos(tmp_phi)
    return dets_r, dets_phi


def canonical_to_global_xy(scan_r, scan_phi, dx, dy):
    p_r, p_phi = canonical_to_global(scan_r, scan_phi, dx, dy)
    return rphi_to_xy(p_r, p_phi)


def data_augmentation(sample_dict):
    scans, target_reg = sample_dict["scans"], sample_dict["target_reg"]

    # # Random scaling
    # s = np.random.uniform(low=0.95, high=1.05)
    # scans = s * scans
    # target_reg = s * target_reg

    # Random left-right flip. Of whole batch for convenience, but should be the
    # same as individuals.
    if np.random.rand() < 0.5:
        scans = scans[:, ::-1]
        target_reg[:, 0] = -target_reg[:, 0]

    sample_dict.update({"target_reg": target_reg, "scans": scans})

    return sample_dict


# def get_regression_target(
#     scan,
#     scan_phi,
#     wcs,
#     was,
#     wps,
#     radius_wc=0.6,
#     radius_wa=0.4,
#     radius_wp=0.35,
#     label_wc=1,
#     label_wa=2,
#     label_wp=3,
#     pedestrian_only=False,
# ):
#     num_pts = len(scan)
#     target_cls = np.zeros(num_pts, dtype=np.int64)
#     target_reg = np.zeros((num_pts, 2), dtype=np.float32)

#     if pedestrian_only:
#         all_dets = list(wps)
#         all_radius = [radius_wp] * len(wps)
#         labels = [0] + [1] * len(wps)
#     else:
#         all_dets = list(wcs) + list(was) + list(wps)
#         all_radius = (
#             [radius_wc] * len(wcs) + [radius_wa] * len(was) + [radius_wp] * len(wps)
#         )
#         labels = (
#             [0] + [label_wc] * len(wcs) + [label_wa] * len(was) + [label_wp] * len(wps)
#         )

#     dets = closest_detection(scan, scan_phi, all_dets, all_radius)

#     for i, (r, phi) in enumerate(zip(scan, scan_phi)):
#         if 0 < dets[i]:
#             target_cls[i] = labels[dets[i]]
#             target_reg[i, :] = global_to_canonical(r, phi, *all_dets[dets[i] - 1])

#     return target_cls, target_reg


# def closest_detection(scan, scan_phi, dets, radii):
#     """
#     Given a single `scan` (450 floats), a list of r,phi detections `dets` (Nx2),
#     and a list of N `radii` for those detections, return a mapping from each
#     point in `scan` to the closest detection for which the point falls inside its radius.
#     The returned detection-index is a 1-based index, with 0 meaning no detection
#     is close enough to that point.
#     """
#     if len(dets) == 0:
#         return np.zeros_like(scan, dtype=int)

#     assert len(dets) == len(radii), "Need to give a radius for each detection!"

#     # Distance (in x,y space) of each laser-point with each detection.
#     scan_xy = np.array(rphi_to_xy(scan, scan_phi)).T  # (N, 2)
#     dists = cdist(scan_xy, np.array([rphi_to_xy(r, phi) for r, phi in dets]))

#     # Subtract the radius from the distances, such that they are < 0 if inside, > 0 if outside.
#     dists -= radii

#     # Prepend zeros so that argmin is 0 for everything "outside".
#     dists = np.hstack([np.zeros((len(scan), 1)), dists])

#     # And find out who's closest, including the threshold!
#     return np.argmin(dists, axis=1)


def scans_to_cutout(
    scans,
    scan_phi,
    stride=1,
    centered=True,
    fixed=False,
    window_width=1.66,
    window_depth=1.0,
    num_cutout_pts=48,
    padding_val=29.99,
    area_mode=False,
):
    # assert fixed
    # assert (
    #     np.sum((scan_phi[1:] - scan_phi[:-1]) <= 0) == 0
    # ), "scan_phi must be in ascending order"

    # assert (
    #     np.sum(scan_phi + scan_phi[::-1]) < 1e-5
    # ), "scan_phi must be symetric and centered around 0"

    num_scans, num_pts = scans.shape

    # size (width) of the window
    dists = (
        scans[:, ::stride]
        if fixed
        else np.tile(scans[-1, ::stride], num_scans).reshape(num_scans, -1)
    )
    half_alpha = np.arctan(0.5 * window_width / np.maximum(dists, 1e-2))

    # cutout indices
    delta_alpha = 2.0 * half_alpha / (num_cutout_pts - 1)
    ang_ct = (
        scan_phi[::stride]
        - half_alpha
        + np.arange(num_cutout_pts).reshape(num_cutout_pts, 1, 1) * delta_alpha
    )
    ang_ct = (ang_ct + np.pi) % (2.0 * np.pi) - np.pi  # warp angle
    inds_ct = (ang_ct - scan_phi[0]) / (scan_phi[1] - scan_phi[0])
    outbound_mask = np.logical_or(inds_ct < 0, inds_ct > num_pts - 1)

    # cutout (linear interp)
    inds_ct_low = _clip(np.floor(inds_ct), 0, num_pts - 1).astype(np.int)
    inds_ct_high = _clip(inds_ct_low + 1, 0, num_pts - 1).astype(np.int)
    inds_ct_ratio = _clip(inds_ct - inds_ct_low, 0.0, 1.0)
    inds_offset = (
        np.arange(num_scans).reshape(1, num_scans, 1) * num_pts
    )  # because np.take flattens array
    ct_low = np.take(scans, inds_ct_low + inds_offset)
    ct_high = np.take(scans, inds_ct_high + inds_offset)
    ct = ct_low + inds_ct_ratio * (ct_high - ct_low)

    # use area sampling for down-sampling (close points)
    if area_mode:
        num_pts_in_window = inds_ct[-1] - inds_ct[0]
        area_mask = num_pts_in_window > num_cutout_pts
        if np.sum(area_mask) > 0:
            # sample the window with more points than the actual number of points
            s_area = int(math.ceil(np.max(num_pts_in_window) / num_cutout_pts))
            num_ct_pts_area = s_area * num_cutout_pts
            delta_alpha_area = 2.0 * half_alpha / (num_ct_pts_area - 1)
            ang_ct_area = (
                scan_phi[::stride]
                - half_alpha
                + np.arange(num_ct_pts_area).reshape(num_ct_pts_area, 1, 1)
                * delta_alpha_area
            )
            ang_ct_area = (ang_ct_area + np.pi) % (2.0 * np.pi) - np.pi  # warp angle
            inds_ct_area = (ang_ct_area - scan_phi[0]) / (scan_phi[1] - scan_phi[0])
            inds_ct_area = np.rint(_clip(inds_ct_area, 0, num_pts - 1)).astype(np.int32)
            ct_area = np.take(scans, inds_ct_area + inds_offset)
            ct_area = ct_area.reshape(
                num_cutout_pts, s_area, num_scans, dists.shape[1]
            ).mean(axis=1)
            ct[:, area_mask] = ct_area[:, area_mask]

    # normalize cutout
    ct[outbound_mask] = padding_val
    ct = _clip(ct, dists - window_depth, dists + window_depth)
    if centered:
        ct = ct - dists
        ct = ct / window_depth

    return np.ascontiguousarray(
        ct.transpose((2, 1, 0)), dtype=np.float32
    )  # (scans, times, cutouts)


def scans_to_cutout_torch(
    scans,
    scan_phi,
    stride=1,
    centered=True,
    fixed=False,
    window_width=1.66,
    window_depth=1.0,
    num_cutout_pts=48,
    padding_val=29.99,
    area_mode=False,
):
    num_scans, num_pts = scans.shape

    # size (width) of the window
    dists = scans[:, ::stride] if fixed else scans[-1, ::stride].repeat(num_scans, 1)
    half_alpha = torch.atan(0.5 * window_width / torch.clamp(dists, min=1e-2))

    # cutout indices
    delta_alpha = 2.0 * half_alpha / (num_cutout_pts - 1)
    ang_step = (
        torch.arange(num_cutout_pts, device=scans.device).view(num_cutout_pts, 1, 1)
        * delta_alpha
    )
    ang_ct = scan_phi[::stride] - half_alpha + ang_step
    ang_ct = (ang_ct + np.pi) % (2.0 * np.pi) - np.pi  # warp angle
    inds_ct = (ang_ct - scan_phi[0]) / (scan_phi[1] - scan_phi[0])
    outbound_mask = torch.logical_xor(inds_ct < 0, inds_ct > num_pts - 1)

    # cutout (linear interp)
    inds_ct_low = inds_ct.floor().long().clamp(min=0, max=num_pts - 1)
    inds_ct_high = inds_ct.ceil().long().clamp(min=0, max=num_pts - 1)
    inds_ct_ratio = (inds_ct - inds_ct_low).clamp(min=0.0, max=1.0)
    ct_low = torch.gather(scans.expand_as(inds_ct_low), dim=2, index=inds_ct_low)
    ct_high = torch.gather(scans.expand_as(inds_ct_high), dim=2, index=inds_ct_high)
    ct = ct_low + inds_ct_ratio * (ct_high - ct_low)

    # use area sampling for down-sampling (close points)
    if area_mode:
        num_pts_in_window = inds_ct[-1] - inds_ct[0]
        area_mask = num_pts_in_window > num_cutout_pts
        if torch.sum(area_mask) > 0:
            # sample the window with more points than the actual number of points
            s_area = (num_pts_in_window.max() / num_cutout_pts).ceil().long().item()
            num_ct_pts_area = s_area * num_cutout_pts
            delta_alpha_area = 2.0 * half_alpha / (num_ct_pts_area - 1)
            ang_step_area = (
                torch.arange(num_ct_pts_area, device=scans.device).view(
                    num_ct_pts_area, 1, 1
                )
                * delta_alpha_area
            )
            ang_ct_area = scan_phi[::stride] - half_alpha + ang_step_area
            ang_ct_area = (ang_ct_area + np.pi) % (2.0 * np.pi) - np.pi  # warp angle
            inds_ct_area = (
                torch.round((ang_ct_area - scan_phi[0]) / (scan_phi[1] - scan_phi[0]))
                .long()
                .clamp(min=0, max=num_pts - 1)
            )
            ct_area = torch.gather(
                scans.expand_as(inds_ct_area), dim=2, index=inds_ct_area
            )
            ct_area = ct_area.view(
                num_cutout_pts, s_area, num_scans, dists.shape[1]
            ).mean(dim=1)
            ct[:, area_mask] = ct_area[:, area_mask]

    # normalize cutout
    ct[outbound_mask] = padding_val
    # torch.clamp does not support tensor min/max
    ct = torch.where(ct < (dists - window_depth), dists - window_depth, ct)
    ct = torch.where(ct > (dists + window_depth), dists + window_depth, ct)
    if centered:
        ct = ct - dists
        ct = ct / window_depth

    # # compare impl with numpy version
    # ct_numpy = scans_to_cutout(
    #     scans.data.cpu().numpy(), scan_phi.data.cpu().numpy(),
    #     stride=stride, centered=centered, fixed=fixed, window_width=window_width,
    #     window_depth=window_depth, num_cutout_pts=num_cutout_pts,
    #     padding_val=padding_val, area_mode=area_mode)
    # print("max(abs(ct_numpy - ct_torch)) = %f" % (np.max(np.abs(
    #     ct_numpy - ct.permute((2, 1, 0)).float().data.cpu().numpy()))))

    return ct.permute((2, 1, 0)).float().contiguous()  # (scans, times, cutouts)


def scans_to_cutout_original(
    scans,
    angle_incre,
    fixed=True,
    centered=True,
    pt_inds=None,
    window_width=1.66,
    window_depth=1.0,
    num_cutout_pts=48,
    padding_val=29.99,
):
    # assert False, "Deprecated"

    num_scans, num_pts = scans.shape
    if pt_inds is None:
        pt_inds = range(num_pts)

    scans_padded = np.pad(
        scans, ((0, 0), (0, 1)), mode="constant", constant_values=padding_val
    )  # pad boarder
    scans_cutout = np.empty((num_pts, num_scans, num_cutout_pts), dtype=np.float32)

    for scan_idx in range(num_scans):
        for pt_idx in pt_inds:
            # Compute the size (width) of the window
            pt_r = scans[scan_idx, pt_idx] if fixed else scans[-1, pt_idx]

            half_alpha = float(np.arctan(0.5 * window_width / max(pt_r, 0.01)))

            # Compute the start and end indices of cutout
            start_idx = int(round(pt_idx - half_alpha / angle_incre))
            end_idx = int(round(pt_idx + half_alpha / angle_incre))
            cutout_pts_inds = np.arange(start_idx, end_idx + 1)
            cutout_pts_inds = _clip(cutout_pts_inds, -1, num_pts)
            # cutout_pts_inds = np.core.umath.clip(cutout_pts_inds, -1, num_pts)
            # cutout_pts_inds = cutout_pts_inds.clip(-1, num_pts)

            # cutout points
            cutout_pts = scans_padded[scan_idx, cutout_pts_inds]

            # resampling/interpolation
            interp = (
                cv2.INTER_AREA
                if num_cutout_pts < len(cutout_pts_inds)
                else cv2.INTER_LINEAR
            )
            cutout_sampled = cv2.resize(
                cutout_pts, (1, num_cutout_pts), interpolation=interp
            ).squeeze()

            # center cutout and clip depth to avoid strong depth discontinuity
            cutout_sampled = _clip(
                cutout_sampled, pt_r - window_depth, pt_r + window_depth
            )
            # cutout_sampled = np.core.umath.clip(
            #         cutout_sampled,
            #         pt_r - window_depth,
            #         pt_r + window_depth)
            # cutout_sampled = cutout_sampled.clip(pt_r - window_depth,
            #                                      pt_r + window_depth)

            if centered:
                cutout_sampled -= pt_r  # center
                cutout_sampled = cutout_sampled / window_depth  # normalize
            scans_cutout[pt_idx, scan_idx, :] = cutout_sampled

    return scans_cutout


def scans_to_polar_grid(
    scans,
    min_range=0.0,
    max_range=30.0,
    range_bin_size=1.0,
    tsdf_clip=1.0,
    normalize=True,
):
    num_scans, num_pts = scans.shape
    num_range = int((max_range - min_range) / range_bin_size) + 1
    mag_range, mid_range = max_range - min_range, 0.5 * (max_range - min_range)

    polar_grid = np.empty((num_scans, num_range, num_pts), dtype=np.float32)

    scans = np.clip(scans, min_range, max_range)
    scans_grid_inds = ((scans - min_range) / range_bin_size).astype(np.int32)

    for i_scan in range(num_scans):
        for i_pt in range(num_pts):
            range_grid_ind = scans_grid_inds[i_scan, i_pt]
            scan_val = scans[i_scan, i_pt]

            if tsdf_clip > 0.0:
                min_dist, max_dist = 0 - range_grid_ind, num_range - range_grid_ind
                tsdf = (
                    np.arange(min_dist, max_dist, step=1).astype(np.float32)
                    * range_bin_size
                )
                tsdf = np.clip(tsdf, -tsdf_clip, tsdf_clip)
            else:
                tsdf = np.zeros(num_range, dtype=np.float32)

            if normalize:
                scan_val = (scan_val - mid_range) / mag_range * 2.0
                tsdf = tsdf / mag_range * 2.0

            tsdf[range_grid_ind] = scan_val
            polar_grid[i_scan, :, i_pt] = tsdf

    return polar_grid


def group_predicted_center(
    scan_grid,
    phi_grid,
    pred_cls,
    pred_reg,
    min_thresh=1e-5,
    class_weights=None,
    bin_size=0.1,
    blur_sigma=0.5,
    x_min=-15.0,
    x_max=15.0,
    y_min=-5.0,
    y_max=15.0,
    vote_collect_radius=0.3,
    cls_agnostic_vote=False,
):
    """
    Convert a list of votes to a list of detections based on Non-Max suppression.

    ` `vote_combiner` the combination function for the votes per detection.
    - `bin_size` the bin size (in meters) used for the grid where votes are cast.
    - `blur_win` the window size (in bins) used to blur the voting grid.
    - `blur_sigma` the sigma used to compute the Gaussian in the blur window.
    - `x_min` the left limit for the voting grid, in meters.
    - `x_max` the right limit for the voting grid, in meters.
    - `y_min` the bottom limit for the voting grid in meters.
    - `y_max` the top limit for the voting grid in meters.
    - `vote_collect_radius` the radius use during the collection of votes assigned
      to each detection.

    Returns a list of tuples (x,y,probs) where `probs` has the same layout as
    `probas`.
    """
    pred_r, pred_phi = canonical_to_global(
        scan_grid, phi_grid, pred_reg[:, 0], pred_reg[:, 1]
    )
    pred_xs, pred_ys = rphi_to_xy(pred_r, pred_phi)

    instance_mask = np.zeros(len(scan_grid), dtype=np.int32)
    scan_array_inds = np.arange(len(scan_grid))

    single_cls = pred_cls.shape[1] == 1

    if class_weights is not None and not single_cls:
        pred_cls = np.copy(pred_cls)
        pred_cls[:, 1:] *= class_weights

    # voting grid
    x_range = int((x_max - x_min) / bin_size)
    y_range = int((y_max - y_min) / bin_size)
    grid = np.zeros((x_range, y_range, pred_cls.shape[1]), np.float32)

    # update x/y max to correspond to the end of the last bin.
    x_max = x_min + x_range * bin_size
    y_max = y_min + y_range * bin_size

    # filter out all the weak votes
    pred_cls_agn = pred_cls[:, 0] if single_cls else np.sum(pred_cls[:, 1:], axis=-1)
    voters_inds = np.where(pred_cls_agn > min_thresh)[0]

    if len(voters_inds) == 0:
        return [], [], instance_mask

    pred_xs, pred_ys = pred_xs[voters_inds], pred_ys[voters_inds]
    pred_cls = pred_cls[voters_inds]
    scan_array_inds = scan_array_inds[voters_inds]
    pred_x_inds = np.int64((pred_xs - x_min) / bin_size)
    pred_y_inds = np.int64((pred_ys - y_min) / bin_size)

    # discard out of bound votes
    mask = (
        (0 <= pred_x_inds)
        & (pred_x_inds < x_range)
        & (0 <= pred_y_inds)
        & (pred_y_inds < y_range)
    )
    pred_x_inds, pred_xs = pred_x_inds[mask], pred_xs[mask]
    pred_y_inds, pred_ys = pred_y_inds[mask], pred_ys[mask]
    pred_cls = pred_cls[mask]
    scan_array_inds = scan_array_inds[mask]

    # vote into the grid, including the agnostic vote as sum of class-votes!
    # @TODO Do we need the class grids?
    if single_cls:
        np.add.at(grid, (pred_x_inds, pred_y_inds), pred_cls)
    else:
        np.add.at(
            grid,
            (pred_x_inds, pred_y_inds),
            np.concatenate(
                [np.sum(pred_cls[:, 1:], axis=1, keepdims=True), pred_cls[:, 1:]],
                axis=1,
            ),
        )

    # NMS, only in the "common" voting grid
    grid_all_cls = grid[:, :, 0]
    if blur_sigma > 0:
        blur_win = int(2 * ((blur_sigma * 5) // 2) + 1)
        grid_all_cls = cv2.GaussianBlur(grid_all_cls, (blur_win, blur_win), blur_sigma)
    grid_nms_val = maximum_filter(grid_all_cls, size=3)
    grid_nms_inds = (grid_all_cls == grid_nms_val) & (grid_all_cls > 0)
    nms_xs, nms_ys = np.where(grid_nms_inds)

    if len(nms_xs) == 0:
        return [], [], instance_mask

    # Back from grid-bins to real-world locations.
    nms_xs = nms_xs * bin_size + x_min + bin_size / 2
    nms_ys = nms_ys * bin_size + y_min + bin_size / 2

    # For each vote, get which maximum/detection it contributed to.
    # Shape of `distance_to_center` (ndets, voters) and outer is (voters)
    distance_to_center = np.hypot(pred_xs - nms_xs[:, None], pred_ys - nms_ys[:, None])
    detection_ids = np.argmin(distance_to_center, axis=0)

    # Generate the final detections by average over their voters.
    dets_xs, dets_ys, dets_cls = [], [], []
    for ipeak in range(len(nms_xs)):
        voter_inds = np.where(detection_ids == ipeak)[0]
        voter_inds = voter_inds[
            distance_to_center[ipeak, voter_inds] < vote_collect_radius
        ]

        support_xs, support_ys = pred_xs[voter_inds], pred_ys[voter_inds]
        support_cls = pred_cls[voter_inds]

        # mark instance, 0 is the background
        instance_mask[scan_array_inds[voter_inds]] = ipeak + 1

        if cls_agnostic_vote and not single_cls:
            weights = np.sum(support_cls[:, 1:], axis=1)
            norm = 1.0 / np.sum(weights)
            dets_xs.append(norm * np.sum(weights * support_xs))
            dets_ys.append(norm * np.sum(weights * support_ys))
            dets_cls.append(norm * np.sum(weights[:, None] * support_cls, axis=0))
        else:
            dets_xs.append(np.mean(support_xs))
            dets_ys.append(np.mean(support_ys))
            dets_cls.append(np.mean(support_cls, axis=0))

    return np.array([dets_xs, dets_ys]).T, np.array(dets_cls), instance_mask


def nms_predicted_center(
    scan_grid, phi_grid, pred_cls, pred_reg, pred_reg_prev=None, min_dist=0.5
):
    assert len(pred_cls.shape) == 1

    pred_r, pred_phi = canonical_to_global(
        scan_grid, phi_grid, pred_reg[:, 0], pred_reg[:, 1]
    )
    pred_xs, pred_ys = rphi_to_xy(pred_r, pred_phi)

    # sort prediction with descending confidence
    sort_inds = np.argsort(pred_cls)[::-1]
    pred_xs, pred_ys = pred_xs[sort_inds], pred_ys[sort_inds]
    pred_cls = pred_cls[sort_inds]

    # compute pair-wise distance
    num_pts = len(scan_grid)
    xdiff = pred_xs.reshape(num_pts, 1) - pred_xs.reshape(1, num_pts)
    ydiff = pred_ys.reshape(num_pts, 1) - pred_ys.reshape(1, num_pts)
    p_dist = np.sqrt(np.square(xdiff) + np.square(ydiff))

    # nms
    keep = np.ones(num_pts, dtype=np.bool_)
    instance_mask = np.zeros(num_pts, dtype=np.int32)
    instance_id = 1
    for i in range(num_pts):
        if not keep[i]:
            continue

        dup_inds = p_dist[i] < min_dist
        keep[dup_inds] = False
        keep[i] = True
        instance_mask[sort_inds[dup_inds]] = instance_id
        instance_id += 1

    det_xys = np.stack((pred_xs, pred_ys), axis=1)[keep]
    det_cls = pred_cls[keep]

    if pred_reg_prev is None:
        return det_xys, det_cls, instance_mask

    # process previous frame using nms result from the current frame
    pred_xs_prev, pred_ys_prev = canonical_to_global_xy(
        scan_grid, phi_grid, pred_reg_prev[:, 0], pred_reg_prev[:, 1]
    )
    pred_xs_prev, pred_ys_prev = pred_xs_prev[sort_inds], pred_ys_prev[sort_inds]
    det_xys_prev = np.stack((pred_xs_prev, pred_ys_prev), axis=1)[keep]

    return det_xys, det_xys_prev, det_cls, instance_mask


def nms_predicted_center_torch(scan_grid, phi_grid, pred_cls, pred_reg, min_dist=0.5):
    assert pred_cls.shape[1] == 1

    # scan_grid = torch.from_numpy(scan_grid).float().cuda(non_blocking=True)
    # phi_grid = torch.from_numpy(phi_grid).float().cuda(non_blocking=True)

    with torch.no_grad():
        pred_r, pred_phi = canonical_to_global_torch(
            scan_grid, phi_grid, pred_reg[:, 0], pred_reg[:, 1]
        )
        pred_xs, pred_ys = rphi_to_xy_torch(pred_r, pred_phi)
        pred_xys = torch.stack((pred_xs, pred_ys), dim=1).contiguous()

        top_k = 10000  # keep all detections
        keep, num_to_keep, parent_object_index = nms(
            pred_xys, pred_cls, min_dist, top_k
        )

    dets_xy = pred_xys[keep[:num_to_keep]]
    dets_cls = pred_cls[keep[:num_to_keep]]
    instance_mask = parent_object_index.long()

    return dets_xy, dets_cls, instance_mask


def _phi_to_rotation_matrix(phi):
    """Return a rotation matrix R from given phi such that x_world = R * x_sensor

    Args:
        phi (float): Phi angle defined in the odometry of the DROW dataset.

    Returns:
        R (np.array[2, 2]): Rotation matrix
    """
    cp, sp = np.cos(phi), np.sin(phi)
    return np.array([[cp, -sp], [sp, cp]], dtype=np.float32)


def get_velocity_from_odometry(scan1_xy, odom0, odom1):
    """Compute aparent velocity of stationary points caused by scanner movement.

    The computation is done using:
        v = - R_world2scan1 * (T_xy1 - T_xy0) / dt - [(phi1 - phi0) / dt] x scan1_xy
    where the first and second term are the aparent motion caused by linear and
    rotational motion of the scanner respectively. The return is v * dt.

    Args:
        scan1_xy (array[N, (x, y)]): Stationary points expressed in scanner
            frame at current step (time1).
        odom0 (array[(x, y, phi)]): Sensor odometry at previous step (time0).
        odom1 (array[(x, y, phi)]): Sensor odometry at current step (time1).

    Returns:
        v_dt (array[N, (x, y)]): Velocity for each point expressed in the
            current frame, scaled by dt.
    """

    odom_del = odom1 - odom0
    rot1 = _phi_to_rotation_matrix(odom1[2])  # scan1 to world
    odom_xy_del = odom_del[:2]  # in world frame
    odom_xy_del = np.matmul(rot1.T, odom_xy_del.reshape(2, 1))  # in scan1 frame
    # express cross-product in matrix form
    phi_cross = odom_del[2] * np.array([[0, -1], [1, 0]], dtype=np.float32)
    v_dt = -odom_xy_del.reshape(1, 2) - np.matmul(scan1_xy, phi_cross.T)

    return v_dt


def get_displacement_from_odometry(scan1_xy, odom0, odom1):
    """Compute aparent displacement of stationary points caused by scanner movement.

    First the current scan expressed in the previous scanner frame is computed,
        scan0_xy = R_frame0_to_world.T * (R_frame1_to_world * scan1_xy + T1 - T0)
    Then the displacement is computed as
        displacement = scan1_xy - scan0_xy

    Args:
        scan1_xy (array[N, (x, y)]): Stationary points expressed in current
            scanner frame.
        odom0 (array[(x, y, phi)]): Sensor odometry at previous step (time0).
        odom1 (array[(x, y, phi)]): Sensor odometry at current step (time1).

    Returns:
        disp (array[N, (x, y)]): Aparent displacement.
    """

    rot0 = _phi_to_rotation_matrix(odom0[2])  # scan0 to world
    rot1 = _phi_to_rotation_matrix(odom1[2])
    rot = np.eye(2) - np.matmul(rot0.T, rot1)
    tran = (odom1[:2] - odom0[:2]).reshape(2, 1)
    disp = np.matmul(scan1_xy, rot.T) - np.matmul(rot0.T, tran).reshape(1, 2)

    return disp


def nms_2d(dets, scores, iou_thresh):
    """NMS on 2D bounding boxes, from
    https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py

    Args:
        dets (np.ndarray[N, 4]): x1, y1, x2, y2
        scores (np.ndarray[N])
        thresh (float): [description]

    Returns:
        keep (np.ndarray[M]): Indices of remaining boxes
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def generate_pseudo_labels(
    laser_r,
    laser_phi,
    laser_pixel_xy,
    boxes,
    confs,
    min_conf=0.75,
    max_ar=0.45,
    max_overlap=0.4,
    mean_shift_radius=0.5,
):
    """Generate pseudo labels for laser points from image bounding boxes

    Args:
        laser_r (np.ndarray[N])
        laser_phi (np.ndarray[N])
        laser_pixel_xy (np.ndarray[2, N]): Pixel coordinates of the laser points
            projected on image
        boxes (np.ndarray[M, 4]): Bounding boxes (x1, y1, x2, y2)
        confs (np.ndarray[M]): Bounding box confidences
        min_conf (float, optional): For generating pseudo labels, only use bounding
            boxes with higher confidences. Defaults to 0.75.
        max_ar (float, optional): Only use bounding boxes with lower aspect ratios.
            Defaults to 0.45.
        max_overlap (float, optional): Only use bounding boxes with smaller overlap
            ratio. Defaults to 0.4.
        mean_shift_radius (float, optional): Defaults to 0.5.

    Returns:
        pl_loc_xy (np.ndarray[P, 2]): xy location of P pseudo labels
        pl_boxes (np.ndarray[P, 4]): corresponding bounding boxes
        neg_mask (np.ndarray[N]): True if the laser point is not near a bounding
            box and can thus be used as negative for training classifier
    """
    # select bounding boxes for positive samples
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    w = x2 - x1 + 1
    h = y2 - y1 + 1

    xx1 = np.maximum(x1.reshape(-1, 1), x1.reshape(1, -1))
    yy1 = np.maximum(y1.reshape(-1, 1), y1.reshape(1, -1))
    xx2 = np.minimum(x2.reshape(-1, 1), x2.reshape(1, -1))
    yy2 = np.minimum(y2.reshape(-1, 1), y2.reshape(1, -1))

    inter_w = np.maximum(0.0, xx2 - xx1 + 1)
    inter_h = np.maximum(0.0, yy2 - yy1 + 1)

    boxes_ar = w / h
    boxes_area = w * h

    overlap_area = inter_w * inter_h
    np.fill_diagonal(overlap_area, 0.0)
    boxes_overlap = np.max(overlap_area, axis=0) / boxes_area

    selection_mask = np.logical_and(boxes_ar < max_ar, boxes_overlap < max_overlap)
    selection_mask = np.logical_and(selection_mask, confs > min_conf)
    boxes_selected = boxes[selection_mask, :]

    # generate positive samples from each selected boxes
    pl_loc_xy = []
    pl_boxes = []
    for box in boxes_selected:
        # take points fall within the bottom half of the bounding box
        inside_mask = np.logical_and(
            laser_pixel_xy[0] > box[0], laser_pixel_xy[0] < box[2]
        )
        inside_mask = np.logical_and(inside_mask, laser_pixel_xy[1] < box[3])
        inside_mask = np.logical_and(
            inside_mask, laser_pixel_xy[1] > 0.5 * (box[1] + box[3])
        )

        # remove points that are far (JRDB pad range measurements with no value as 26)
        inside_mask = np.logical_and(inside_mask, laser_r < 20)

        # not enough points within the bounding box
        if inside_mask.sum() < 2:
            continue

        laser_r_inside = laser_r[inside_mask]
        laser_phi_inside = laser_phi[inside_mask]

        # do a k-means clustering on r space, with k=2, to seperate close and far points
        c0, c1 = np.min(laser_r_inside), np.max(laser_r_inside)
        prev_c0, prev_c1 = 1e6, 1e6
        converge_thresh = 0.1
        iter_count, max_iter = 0, 1000
        while iter_count < max_iter and (
            np.abs(c0 - prev_c0) > converge_thresh
            or np.abs(c1 - prev_c1) > converge_thresh
        ):
            c0_mask = np.abs(laser_r_inside - c0) < np.abs(laser_r_inside - c1)
            prev_c0 = c0
            prev_c1 = c1
            c0 = np.mean(laser_r_inside[c0_mask])
            c1 = np.mean(laser_r_inside[np.logical_not(c0_mask)])

        assert iter_count < max_iter

        # take the point that is closest to the center of the close cluster as
        # as the initial seed to mean shift
        laser_xy_inside = np.stack(rphi_to_xy(laser_r_inside, laser_phi_inside), axis=0)
        c_xy = np.mean(laser_xy_inside[:, c0_mask], axis=1)
        c_dist = np.linalg.norm(laser_xy_inside - c_xy.reshape(2, 1), axis=0)
        c_xy = laser_xy_inside[:, c_dist.argmin()]

        # mean-shift to identify the person cluster
        prev_c_xy = np.array([1e6, 1e6])
        converge_thresh = 0.1
        iter_count, max_iter = 0, 1000
        while (
            iter_count < max_iter and np.linalg.norm(prev_c_xy - c_xy) > converge_thresh
        ):
            c_mask = (
                np.linalg.norm(laser_xy_inside - c_xy.reshape(2, 1), axis=0)
                < mean_shift_radius
            )
            prev_c_xy = c_xy
            c_xy = np.mean(laser_xy_inside[:, c_mask], axis=1)

        assert iter_count < max_iter

        # take the center of mean-shift as the person location
        pl_loc_xy.append(c_xy)
        pl_boxes.append(box)

    # generate negative samples
    # enlarge bounding boxes, then take points that don't fall in any box
    x1 = x1 - 0.1 * w
    x2 = x2 + 0.1 * w

    in_box_mask = np.logical_and(
        laser_pixel_xy[0].reshape(-1, 1) > x1.reshape(1, -1),
        laser_pixel_xy[0].reshape(-1, 1) < x2.reshape(1, -1),
    )  # (pts, boxes), True if point i is in box j

    neg_mask = in_box_mask.max(axis=1) == 0

    return (
        np.array(pl_loc_xy, dtype=np.float32),
        np.array(pl_boxes, dtype=np.float32),
        neg_mask,
    )


def ravel_hash_vec(arr):
    assert arr.ndim == 2
    arr = arr - arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def get_unique_rows(arr):
    keys = ravel_hash_vec(arr)
    _, inds = np.unique(keys, return_index=True)
    return inds
