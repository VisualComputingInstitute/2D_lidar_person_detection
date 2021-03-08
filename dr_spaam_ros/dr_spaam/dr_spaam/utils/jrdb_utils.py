import numpy as np


def box_to_kitti_string(dets_xy, dets_cls, occluded):
    """Obtain a KITTI format string for a detected box

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
        s += f"Pedestrian 0 {occ} 0 0 0 0 0 0 0 0 0 {xy[0]} {xy[1]} 0 0 {cls}\n"
    s = s.strip("\n")

    return s


def kitti_string_to_box(s):
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
