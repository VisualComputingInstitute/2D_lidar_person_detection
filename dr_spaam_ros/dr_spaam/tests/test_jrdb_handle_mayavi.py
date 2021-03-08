import random
import time

import numpy as np
from mayavi import mlab

from dr_spaam.datahandle.jrdb_handle import JRDBHandle
import dr_spaam.utils.utils as u
import dr_spaam.utils.utils_box3d as ub3d
import dr_spaam.utils.jrdb_transforms as jt

_COLOR_INSTANCE = True


def _test_loading_speed():
    data_handle = JRDBHandle(
        split="train",
        cfg={"data_dir": "./data/JRDB", "num_scans": 10, "scan_stride": 1},
    )

    total_frame = 100
    inds = random.sample(range(len(data_handle)), total_frame)

    t0 = time.time()
    for idx in inds:
        _ = data_handle[idx]
    t1 = time.time()

    print(f"Loaded {total_frame} frames in {t1 - t0} seconds.")


def _plot_sequence():
    jrdb_handle = JRDBHandle(
        split="train",
        cfg={"data_dir": "./data/JRDB", "num_scans": 10, "scan_stride": 1},
    )

    color_pool = np.random.uniform(size=(100, 3))

    for i, data_dict in enumerate(jrdb_handle):
        # lidar
        pc_xyz_upper = jt.transform_pts_upper_velodyne_to_base(
            data_dict["pc_data"]["upper_velodyne"]
        )
        pc_xyz_lower = jt.transform_pts_lower_velodyne_to_base(
            data_dict["pc_data"]["lower_velodyne"]
        )

        # laser
        laser_r = data_dict["laser_data"][-1]
        laser_phi = data_dict["laser_grid"]
        laser_z = data_dict["laser_z"]
        laser_x, laser_y = u.rphi_to_xy(laser_r, laser_phi)
        pc_xyz_laser = jt.transform_pts_laser_to_base(
            np.stack((laser_x, laser_y, laser_z), axis=0)
        )

        if _COLOR_INSTANCE:
            # labels
            boxes, label_ids = [], []
            for ann in data_dict["pc_anns"]:
                # jrdb_handle.box_is_on_ground(ann)
                box, b_id = ub3d.box_from_jrdb(ann)
                boxes.append(box)
                label_ids.append(b_id)
            boxes = np.array(boxes)  # (B, 7)
            pc = np.concatenate([pc_xyz_laser, pc_xyz_upper, pc_xyz_lower], axis=1)
            in_box_mask, closest_box_inds = ub3d.associate_points_and_boxes(
                pc, boxes, resize_factor=1.0
            )

            # plot bg points
            bg_pc = pc[:, np.logical_not(in_box_mask)]
            mlab.points3d(
                bg_pc[0],
                bg_pc[1],
                bg_pc[2],
                scale_factor=0.05,
                color=(1.0, 0.0, 0.0),
            )

            # plot box and fg points
            fg_pc = pc[:, in_box_mask]
            fg_box_inds = closest_box_inds[in_box_mask]
            corners_xyz, connect_inds = ub3d.boxes_to_corners(
                boxes, rtn_connect_inds=True
            )
            for box_idx, (p_id, corner_xyz) in enumerate(zip(label_ids, corners_xyz)):
                color = tuple(color_pool[p_id % 100])
                # box
                for inds in connect_inds:
                    mlab.plot3d(
                        corner_xyz[0, inds],
                        corner_xyz[1, inds],
                        corner_xyz[2, inds],
                        tube_radius=None,
                        line_width=5,
                        color=color,
                    )

                # point
                in_box_pc = fg_pc[:, fg_box_inds == box_idx]
                mlab.points3d(
                    in_box_pc[0],
                    in_box_pc[1],
                    in_box_pc[2],
                    scale_factor=0.05,
                    color=color,
                )

        else:
            # plot points
            mlab.points3d(
                pc_xyz_lower[0],
                pc_xyz_lower[1],
                pc_xyz_lower[2],
                scale_factor=0.05,
                color=(0.0, 1.0, 0.0),
            )
            mlab.points3d(
                pc_xyz_upper[0],
                pc_xyz_upper[1],
                pc_xyz_upper[2],
                scale_factor=0.05,
                color=(0.0, 0.0, 1.0),
            )
            mlab.points3d(
                pc_xyz_laser[0],
                pc_xyz_laser[1],
                pc_xyz_laser[2],
                scale_factor=0.05,
                color=(1.0, 0.0, 0.0),
            )

            # plot box
            boxes = []
            for ann in data_dict["pc_anns"]:
                # jrdb_handle.box_is_on_ground(ann)
                box = ub3d.box_from_jrdb(ann, fast_mode=False)
                corners_xyz, connect_inds = box.to_corners(resize_factor=1.0, rtn_connect_inds=True)
                for inds in connect_inds:
                    mlab.plot3d(
                        corners_xyz[0, inds],
                        corners_xyz[1, inds],
                        corners_xyz[2, inds],
                        tube_radius=None,
                        line_width=5,
                        color=tuple(color_pool[box.get_id() % 100]),
                    )

        mlab.show()


if __name__ == "__main__":
    # _test_loading_speed()
    _plot_sequence()
