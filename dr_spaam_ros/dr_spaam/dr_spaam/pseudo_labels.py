from dr_spaam.dataset.jrdb_dataset import _get_regression_target_from_pseudo_labels
from dr_spaam.utils import utils as u


def get_regression_target_using_bounding_boxes(
    scan_r, scan_phi, scan_uv, boxes, boxes_conf
):
    pl_xy, pl_boxes, pl_neg_mask = u.generate_pseudo_labels(
        scan_r, scan_phi, scan_uv, boxes, boxes_conf
    )

    (target_cls_pseudo, target_reg_pseudo,) = _get_regression_target_from_pseudo_labels(
        scan_r,
        pl_xy,
        pl_neg_mask,
        person_radius_small=0.4,
        person_radius_large=0.8,
        min_close_points=5,
        pl_correction_level=-1,
        target_cls_annotated=None,
        target_reg_annotated=None,
    )

    return target_cls_pseudo, target_reg_pseudo
