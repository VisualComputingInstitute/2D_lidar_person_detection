import time
import numpy as np

from dr_spaam.detector import Detector
from dr_spaam.datahandle.drow_handle import DROWHandle
from dr_spaam.datahandle.jrdb_handle import JRDBHandle


_FRAME_NUM = 100
_STRIDE = 1


def test_inference_speed_on_drow():
    data_handle = DROWHandle(
        split="test",
        cfg={"num_scans": 1, "scan_stride": 1, "data_dir": "./data/DROWv2-data"},
    )

    ckpt_file = "./ckpts/ckpt_jrdb_ann_drow3_e40.pth"
    detector_drow3 = Detector(
        ckpt_file, model="DROW3", gpu=True, stride=_STRIDE, panoramic_scan=True
    )
    detector_drow3.set_laser_fov(225)

    ckpt_file = "./ckpts/ckpt_jrdb_ann_dr_spaam_e20.pth"
    detector_dr_spaam = Detector(
        ckpt_file, model="DR-SPAAM", gpu=True, stride=_STRIDE, panoramic_scan=True
    )
    detector_dr_spaam.set_laser_fov(225)

    # sample random frames, discard beginning frames, where PyTorch is searching
    # for optimal algorithm
    frame_inds = np.random.randint(0, len(data_handle), size=(_FRAME_NUM + 20,))

    for n, detector in zip(["DROW3", "DR-SPAAM"], [detector_drow3, detector_dr_spaam]):
        t_list = []
        for frame_idx in frame_inds:
            data_dict = data_handle[frame_idx]
            scan_r = data_dict["scans"][-1]

            t0 = time.time()
            dets_xy, dets_cls, _ = detector(scan_r)
            t_list.append(time.time() - t0)

        t_ave = np.array(t_list[20:]).mean()
        print(f"{n} on DROW: {1.0 / t_ave:.1f} FPS " f"({t_ave:.6f} seconds per frame)")


def test_inference_speed_on_jrdb():
    data_handle = JRDBHandle(
        split="train",
        cfg={"data_dir": "./data/JRDB", "num_scans": 1, "scan_stride": 1},
    )

    ckpt_file = "./ckpts/ckpt_jrdb_ann_drow3_e40.pth"
    detector_drow3 = Detector(
        ckpt_file, model="DROW3", gpu=True, stride=_STRIDE, panoramic_scan=True
    )
    detector_drow3.set_laser_fov(360)

    ckpt_file = "./ckpts/ckpt_jrdb_ann_dr_spaam_e20.pth"
    detector_dr_spaam = Detector(
        ckpt_file, model="DR-SPAAM", gpu=True, stride=_STRIDE, panoramic_scan=True
    )
    detector_dr_spaam.set_laser_fov(360)

    frame_inds = np.random.randint(0, len(data_handle), size=(_FRAME_NUM,))

    # sample random frames, discard beginning frames, where PyTorch is searching
    # for optimal algorithm
    frame_inds = np.random.randint(0, len(data_handle), size=(_FRAME_NUM + 20,))

    for n, detector in zip(["DROW3", "DR-SPAAM"], [detector_drow3, detector_dr_spaam]):
        t_list = []
        for frame_idx in frame_inds:
            data_dict = data_handle[frame_idx]
            scan_r = data_dict["laser_data"][-1, ::-1]  # to DROW frame

            t0 = time.time()
            dets_xy, dets_cls, _ = detector(scan_r)
            t_list.append(time.time() - t0)

        t_ave = np.array(t_list[20:]).mean()
        print(f"{n} on JRDB: {1.0 / t_ave:.1f} FPS " f"({t_ave:.6f} seconds per frame)")


if __name__ == "__main__":
    test_inference_speed_on_drow()
    test_inference_speed_on_jrdb()
