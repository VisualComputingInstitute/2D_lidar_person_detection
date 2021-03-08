import numpy as np
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset

from dr_spaam.datahandle.drow_handle import DROWHandle
import dr_spaam.utils.utils as u


class DROWDataset(Dataset):
    def __init__(self, split, cfg):
        self.__handle = DROWHandle(split, cfg["DataHandle"])
        self.__split = split

        self._augment_data = cfg["augment_data"]
        self._person_only = cfg["person_only"]
        self._cutout_kwargs = cfg["cutout_kwargs"]

    @property
    def split(self):
        return self.__split  # used by trainer.py

    def __len__(self):
        return len(self.__handle)

    def __getitem__(self, idx):
        data_dict = self.__handle[idx]

        # regression target
        target_cls, target_reg = _get_regression_target(
            data_dict["scans"][-1],
            data_dict["scan_phi"],
            data_dict["dets_wc"],
            data_dict["dets_wa"],
            data_dict["dets_wp"],
            person_only=self._person_only,
        )

        data_dict["target_cls"] = target_cls
        data_dict["target_reg"] = target_reg

        if self._augment_data:
            data_dict = u.data_augmentation(data_dict)

        data_dict["input"] = u.scans_to_cutout(
            data_dict["scans"], data_dict["scan_phi"], stride=1, **self._cutout_kwargs
        )

        # to be consistent with JRDB dataset
        data_dict["frame_id"] = data_dict["idx"]
        data_dict["sequence"] = "all"

        # this is used by JRDB dataset to mask out annotations, to be consistent
        data_dict["anns_valid_mask"] = np.ones(len(data_dict["dets_wp"]), dtype=np.bool)

        return data_dict

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, _ in batch[0].items():
            if k in ["target_cls", "target_reg", "input"]:
                rtn_dict[k] = np.array([sample[k] for sample in batch])
            else:
                rtn_dict[k] = [sample[k] for sample in batch]

        return rtn_dict


def _get_regression_target(
    scan,
    scan_phi,
    wcs,
    was,
    wps,
    radius_wc=0.6,
    radius_wa=0.4,
    radius_wp=0.35,
    label_wc=1,
    label_wa=2,
    label_wp=3,
    person_only=False,
):
    num_pts = len(scan)
    target_cls = np.zeros(num_pts, dtype=np.int64)
    target_reg = np.zeros((num_pts, 2), dtype=np.float32)

    if person_only:
        all_dets = list(wps)
        all_radius = [radius_wp] * len(wps)
        labels = [0] + [1] * len(wps)
    else:
        all_dets = list(wcs) + list(was) + list(wps)
        all_radius = (
            [radius_wc] * len(wcs) + [radius_wa] * len(was) + [radius_wp] * len(wps)
        )
        labels = (
            [0] + [label_wc] * len(wcs) + [label_wa] * len(was) + [label_wp] * len(wps)
        )

    dets = _closest_detection(scan, scan_phi, all_dets, all_radius)

    for i, (r, phi) in enumerate(zip(scan, scan_phi)):
        if 0 < dets[i]:
            target_cls[i] = labels[dets[i]]
            target_reg[i, :] = u.global_to_canonical(r, phi, *all_dets[dets[i] - 1])

    return target_cls, target_reg


def _closest_detection(scan, scan_phi, dets, radii):
    """
    Given a single `scan` (450 floats), a list of r,phi detections `dets` (Nx2),
    and a list of N `radii` for those detections, return a mapping from each
    point in `scan` to the closest detection for which the point falls inside its
    radius. The returned detection-index is a 1-based index, with 0 meaning no
    detection is close enough to that point.
    """
    if len(dets) == 0:
        return np.zeros_like(scan, dtype=int)

    assert len(dets) == len(radii), "Need to give a radius for each detection!"

    # Distance (in x,y space) of each laser-point with each detection.
    scan_xy = np.array(u.rphi_to_xy(scan, scan_phi)).T  # (N, 2)
    dists = cdist(scan_xy, np.array([u.rphi_to_xy(r, phi) for r, phi in dets]))

    # Subtract the radius from the distances, such that they are < 0 if inside,
    # > 0 if outside.
    dists -= radii

    # Prepend zeros so that argmin is 0 for everything "outside".
    dists = np.hstack([np.zeros((len(scan), 1)), dists])

    # And find out who's closest, including the threshold!
    return np.argmin(dists, axis=1)
