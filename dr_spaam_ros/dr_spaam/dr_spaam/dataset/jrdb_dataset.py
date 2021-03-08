import numpy as np
from torch.utils.data import Dataset

from dr_spaam.datahandle.jrdb_handle import JRDBHandle
import dr_spaam.utils.utils as u
import dr_spaam.utils.jrdb_transforms as jt


__all__ = ["JRDBDataset"]

# Customized train-val split
_JRDB_TRAIN_SEQUENCES = [
    "packard-poster-session-2019-03-20_2",
    "clark-center-intersection-2019-02-28_0",
    "huang-lane-2019-02-12_0",
    "memorial-court-2019-03-16_0",
    "cubberly-auditorium-2019-04-22_0",
    "tressider-2019-04-26_2",
    "jordan-hall-2019-04-22_0",
    "clark-center-2019-02-28_1",
    "gates-basement-elevators-2019-01-17_1",
    "stlc-111-2019-04-19_0",
    "forbes-cafe-2019-01-22_0",
    "tressider-2019-03-16_0",
    "svl-meeting-gates-2-2019-04-08_0",
    "huang-basement-2019-01-25_0",
    "nvidia-aud-2019-04-18_0",
    "hewlett-packard-intersection-2019-01-24_0",
    "bytes-cafe-2019-02-07_0",
]

_JRDB_TEST_SEQUENCES = [
    "packard-poster-session-2019-03-20_1",
    "gates-to-clark-2019-02-28_1",
    "packard-poster-session-2019-03-20_0",
    "tressider-2019-03-16_1",
    "clark-center-2019-02-28_0",
    "svl-meeting-gates-2-2019-04-08_1",
    "meyer-green-2019-03-16_0",
    "gates-159-group-meeting-2019-04-03_0",
    "huang-2-2019-01-25_0",
    "gates-ai-lab-2019-02-08_0",
]

# for online learning experiment
_JRDB_ONLINE_SEQUENCES = [
    "packard-poster-session-2019-03-20_2",
    "stlc-111-2019-04-19_0",
]


class JRDBDataset(Dataset):
    def __init__(self, split, cfg):
        if split == "train":
            self.__handle = JRDBHandle(
                "train", cfg["DataHandle"], sequences=_JRDB_TRAIN_SEQUENCES
            )
        elif split == "val" or split == "test":
            self.__handle = JRDBHandle(
                "train", cfg["DataHandle"], sequences=_JRDB_TEST_SEQUENCES
            )
        elif split == "train_val":
            self.__handle = JRDBHandle("train", cfg["DataHandle"])
        elif split == "train_online":
            self.__handle = JRDBHandle(
                "train", cfg["DataHandle"], sequences=_JRDB_ONLINE_SEQUENCES
            )
        else:
            raise RuntimeError(f"Invalid split: {split}")

        self.__split = split

        self._augment_data = cfg["augment_data"]
        self._person_only = cfg["person_only"]
        self._cutout_kwargs = cfg["cutout_kwargs"]
        self._pseudo_label = cfg["pseudo_label"]
        self._pl_correction_level = cfg["pl_correction_level"]
        self._mixup_alpha = cfg["mixup_alpha"] if "mixup_alpha" in cfg else 0.0

    def __len__(self):
        return len(self.__handle)

    def __getitem__(self, idx):
        if self._mixup_alpha > 0:
            return self._get_sample_with_mixup(idx)
        else:
            return self._get_sample(idx)

    @property
    def split(self):
        return self.__split  # used by trainer.py

    @property
    def sequence_beginning_inds(self):
        return self.__handle.sequence_beginning_inds

    def _get_sample(self, idx):
        data_dict = self.__handle[idx]

        # DROW defines laser frame as x-forward, y-right, z-downward
        # JRDB defines laser frame as x-forward, y-left, z-upward
        # Use DROW frame for DR-SPAAM or DROW3

        # equivalent of flipping y axis (inversing laser phi angle)
        data_dict["laser_data"] = data_dict["laser_data"][:, ::-1]
        scan_rphi = np.stack(
            (data_dict["laser_data"][-1], data_dict["laser_grid"]), axis=0
        )

        # get annotation in laser frame
        ann_xyz = [
            (ann["box"]["cx"], ann["box"]["cy"], ann["box"]["cz"])
            for ann in data_dict["pc_anns"]
        ]
        if len(ann_xyz) > 0:
            ann_xyz = np.array(ann_xyz, dtype=np.float32).T
            ann_xyz = jt.transform_pts_base_to_laser(ann_xyz)
            ann_xyz[1] = -ann_xyz[1]  # to DROW frame
            dets_rphi = np.stack(u.xy_to_rphi(ann_xyz[0], ann_xyz[1]), axis=0)
        else:
            dets_rphi = []

        # regression target
        target_cls, target_reg, anns_valid_mask = _get_regression_target(
            scan_rphi,
            dets_rphi,
            person_radius_small=0.4,
            person_radius_large=0.8,
            min_close_points=5,
        )

        data_dict["target_cls"] = target_cls
        data_dict["target_reg"] = target_reg
        data_dict["anns_valid_mask"] = anns_valid_mask

        # regression target from pseudo labels
        if self._pseudo_label:
            # get pixels of laser points projected on image
            scan_x, scan_y = u.rphi_to_xy(scan_rphi[0], scan_rphi[1])
            scan_y = -scan_y  # convert DROW frame to JRDB laser frame
            scan_xyz_laser = np.stack((scan_x, scan_y, data_dict["laser_z"]), axis=0)
            scan_pixel_xy, _ = jt.transform_pts_laser_to_stitched_im(scan_xyz_laser)

            # get detection boxes
            boxes = []
            boxes_confs = []
            for box_dict in data_dict["im_dets"]:
                x0, y0, w, h = box_dict["box"]
                boxes.append((x0, y0, x0 + w, y0 + h))
                boxes_confs.append(box_dict["score"])
            ########
            # NOTE for ablation, using 2D annotation to generate pseudo labels
            # for box_dict in data_dict["im_anns"]:
            #     x0, y0, w, h = box_dict["box"]
            #     boxes.append((x0, y0, x0 + w, y0 + h))
            #     boxes_confs.append(1.0)
            ########
            boxes = np.array(boxes, dtype=np.float32)
            boxes_confs = np.array(boxes_confs, dtype=np.float32)

            # pseudo label
            pl_xy, pl_boxes, pl_neg_mask = u.generate_pseudo_labels(
                scan_rphi[0], scan_rphi[1], scan_pixel_xy, boxes, boxes_confs
            )
            (
                target_cls_pseudo,
                target_reg_pseudo,
            ) = _get_regression_target_from_pseudo_labels(
                scan_rphi,
                pl_xy,
                pl_neg_mask,
                person_radius_small=0.4,
                person_radius_large=0.8,
                min_close_points=5,
                pl_correction_level=self._pl_correction_level,
                target_cls_annotated=data_dict["target_cls"],
                target_reg_annotated=data_dict["target_reg"],
            )

            data_dict["pseudo_label_loc_xy"] = pl_xy
            data_dict["pseudo_label_boxes"] = pl_boxes

            # still keep the original target for debugging purpose
            data_dict["target_cls_real"] = data_dict["target_cls"]
            data_dict["target_reg_real"] = data_dict["target_reg"]
            data_dict["target_cls"] = target_cls_pseudo
            data_dict["target_reg"] = target_reg_pseudo

        # to be consistent with DROWDataset in order to use the same evaluation function
        dets_wp = []
        for i in range(dets_rphi.shape[1]):
            dets_wp.append((dets_rphi[0, i], dets_rphi[1, i]))
        data_dict["dets_wp"] = dets_wp
        data_dict["scans"] = data_dict["laser_data"]
        data_dict["scan_phi"] = data_dict["laser_grid"]

        if self._augment_data:
            data_dict = u.data_augmentation(data_dict)

        data_dict["input"] = u.scans_to_cutout(
            data_dict["laser_data"],
            data_dict["laser_grid"],
            stride=1,
            **self._cutout_kwargs,
        )

        return data_dict

    def _get_sample_with_mixup(self, idx):
        # randomly find another sample
        mixup_idx = idx
        while mixup_idx == idx:
            mixup_idx = int(np.random.randint(0, len(self.__handle), 1)[0])

        data_dict_0 = self._get_sample(idx)
        data_dict_1 = self._get_sample(mixup_idx)

        input_mixup, target_cls_mixup = _mixup_samples(
            data_dict_0["input"],
            data_dict_0["target_cls"],
            data_dict_1["input"],
            data_dict_1["target_cls"],
            self._mixup_alpha,
        )

        data_dict_0["input_mixup"] = input_mixup
        data_dict_0["target_cls_mixup"] = target_cls_mixup

        return data_dict_0

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, _ in batch[0].items():
            if k in [
                "target_cls",
                "target_reg",
                "input",
                "target_cls_mixup",
                "input_mixup",
            ]:
                rtn_dict[k] = np.array([sample[k] for sample in batch])
            else:
                rtn_dict[k] = [sample[k] for sample in batch]

        return rtn_dict


def _get_regression_target(
    scan_rphi, dets_rphi, person_radius_small, person_radius_large, min_close_points
):
    """Generate classification and regression label.

    Args:
        scan_rphi (np.ndarray[2, N]): Scan points in polar coordinate
        dets_rphi (np.ndarray[2, M]): Annotated person centers in polar coordinate
        person_radius_small (float): Points less than this distance away
            from an annotation is assigned to that annotation and marked as fg.
        person_radius_large (float): Points with no annotation smaller
            than this distance is marked as bg.
        min_close_points (int): Annotations with supportive points fewer than this
            value is marked as invalid. Supportive points are those within the small
            radius.

    Returns:
        target_cls (np.ndarray[N]): Classification label, 1=fg, 0=bg, -1=ignore
        target_reg (np.ndarray[N, 2]): Regression label
        anns_valid_mask (np.ndarray[M])
    """
    N = scan_rphi.shape[1]

    # no annotation in this frame
    if len(dets_rphi) == 0:
        return np.zeros(N, dtype=np.int64), np.zeros((N, 2), dtype=np.float32), []

    scan_xy = np.stack(u.rphi_to_xy(scan_rphi[0], scan_rphi[1]), axis=0)
    dets_xy = np.stack(u.rphi_to_xy(dets_rphi[0], dets_rphi[1]), axis=0)

    dist_scan_dets = np.hypot(
        scan_xy[0].reshape(1, -1) - dets_xy[0].reshape(-1, 1),
        scan_xy[1].reshape(1, -1) - dets_xy[1].reshape(-1, 1),
    )  # (M, N) pairwise distance between scan and detections

    # mark out annotations that has too few scan points
    anns_valid_mask = (
        np.sum(dist_scan_dets < person_radius_small, axis=1) > min_close_points
    )  # (M, )

    # for each point, find the distance to its closest annotation
    argmin_dist_scan_dets = np.argmin(dist_scan_dets, axis=0)  # (N, )
    min_dist_scan_dets = dist_scan_dets[argmin_dist_scan_dets, np.arange(N)]

    # points within small radius, whose corresponding annotation is valid, is marked
    # as foreground
    target_cls = -1 * np.ones(N, dtype=np.int64)
    fg_mask = np.logical_and(
        anns_valid_mask[argmin_dist_scan_dets], min_dist_scan_dets < person_radius_small
    )
    target_cls[fg_mask] = 1
    target_cls[min_dist_scan_dets > person_radius_large] = 0

    # regression target
    dets_matched_rphi = dets_rphi[:, argmin_dist_scan_dets]
    target_reg = np.stack(
        u.global_to_canonical(
            scan_rphi[0], scan_rphi[1], dets_matched_rphi[0], dets_matched_rphi[1]
        ),
        axis=1,
    )

    return target_cls, target_reg, anns_valid_mask


def _get_regression_target_from_pseudo_labels(
    scan_rphi,
    pseudo_label_xy,
    pseudo_label_neg_mask,
    person_radius_small,
    person_radius_large,
    min_close_points,
    pl_correction_level,
    target_cls_annotated,
    target_reg_annotated,
):
    """c.f. _get_regression_target

    Args:
        scan_rphi (np.ndarray[2, N])
        pseudo_label_xy (np.ndarray[M, 2])
        pseudo_label_neg_mask (np.ndarray[N])
    """
    N = scan_rphi.shape[1]

    # NOTE: regression is done for both close (cls = 1) and far (cls = -1) neighbors
    # Set cls = -2 to ignore both classification and regression of the point

    # no pseudo label in this frame
    if len(pseudo_label_xy) == 0:
        target_cls = -2 * np.ones(N, dtype=np.int64)
        target_cls[pseudo_label_neg_mask] = 0
        target_reg = np.zeros((N, 2), dtype=np.float32)

        return target_cls, target_reg

    pseudo_label_rphi = np.stack(
        u.xy_to_rphi(pseudo_label_xy[:, 0], pseudo_label_xy[:, 1]), axis=0
    )

    target_cls, target_reg, _ = _get_regression_target(
        scan_rphi,
        pseudo_label_rphi,
        person_radius_small,
        person_radius_large,
        min_close_points,
    )

    # apply negatives from pseudo labels
    target_cls[target_cls == 0] = -2
    target_cls[pseudo_label_neg_mask] = 0

    # correct pseudo labels using real annotation, for ablation study
    if pl_correction_level in (1, 3, 4):
        mismatch_mask = np.logical_or(
            np.logical_and(target_cls == 1, target_cls_annotated != 1),
            np.logical_and(target_cls == -1, target_cls_annotated != -1),
        )
        target_cls[mismatch_mask] = -2

    if pl_correction_level in (2, 3, 4):
        mismatch_mask = np.logical_and(target_cls == 0, target_cls_annotated != 0)
        target_cls[mismatch_mask] = -2

    if pl_correction_level == 4:
        mismatch_mask = np.logical_or(target_cls == 1, target_cls == -1)
        target_reg[mismatch_mask] = target_reg_annotated[mismatch_mask]

    return target_cls, target_reg


def _mixup_samples(x0, t0, x1, t1, alpha):
    lam = np.random.beta(alpha, alpha, 1)
    x01 = x0 * lam + x1 * (1.0 - lam)
    t01 = t0 * lam + t1 * (1.0 - lam)

    # only mixup points that are valid for classification in both scans
    invalid_mask = np.logical_or(t0 < 0, t1 < 0)
    t01[invalid_mask] = -2

    return x01, t01
