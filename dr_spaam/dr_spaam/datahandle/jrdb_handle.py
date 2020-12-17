import copy
import cv2
import json
import numpy as np
import os
from ._pypcd import point_cloud_from_path

# NOTE: Don't use open3d to load point cloud since it spams the console. Setting
# verbosity level does not solve the problem
# https://github.com/intel-isl/Open3D/issues/1921
# https://github.com/intel-isl/Open3D/issues/884

# Force the dataloader to load only one sample, in which case the network should
# fit perfectly.
_DEBUG_ONE_SAMPLE = False

# Pointcloud and image is only needed for visualization. Turn off for fast dataloading
_LOAD_PC_IM = True

__all__ = ["JRDBHandle"]


class JRDBHandle:
    def __init__(self, split, cfg, sequences=None, exclude_sequences=None):
        if _DEBUG_ONE_SAMPLE:
            split = "train"
            sequences = None
            exclude_sequences = None

        self.__num_scans = cfg["num_scans"]
        self.__scan_stride = cfg["scan_stride"]

        data_dir = os.path.abspath(os.path.expanduser(cfg["data_dir"]))
        data_dir = (
            os.path.join(data_dir, "train_dataset")
            if split == "train" or split == "val"
            else os.path.join(data_dir, "test_dataset")
        )

        self.data_dir = data_dir
        self.timestamp_dir = os.path.join(data_dir, "timestamps")
        self.pc_label_dir = os.path.join(data_dir, "labels", "labels_3d")
        self.im_label_dir = os.path.join(data_dir, "labels", "labels_2d_stitched")

        if sequences is not None:
            sequence_names = sequences
        else:
            sequence_names = os.listdir(self.timestamp_dir)
            # NOTE it is important to sort the return of os.listdir, since its order
            # changes for different file system.
            sequence_names.sort()

        if exclude_sequences is not None:
            sequence_names = [s for s in sequence_names if s not in exclude_sequences]

        self.sequence_names = sequence_names

        self.sequence_handle = []
        self._sequence_beginning_inds = [0]
        self.__flat_inds_sequence = []
        self.__flat_inds_frame = []
        for seq_idx, seq_name in enumerate(self.sequence_names):
            self.sequence_handle.append(_SequenceHandle(self.data_dir, seq_name))

            # build a flat index for all sequences and frames
            sequence_length = len(self.sequence_handle[-1])
            self.__flat_inds_sequence += sequence_length * [seq_idx]
            self.__flat_inds_frame += range(sequence_length)

            self._sequence_beginning_inds.append(
                self._sequence_beginning_inds[-1] + sequence_length
            )

    def __len__(self):
        if _DEBUG_ONE_SAMPLE:
            return 80
        else:
            return len(self.__flat_inds_frame)

    def __getitem__(self, idx):
        if _DEBUG_ONE_SAMPLE:
            idx = 500

        idx_sq = self.__flat_inds_sequence[idx]
        idx_fr = self.__flat_inds_frame[idx]

        frame_dict, pc_anns, im_anns, im_dets = self.sequence_handle[idx_sq][idx_fr]

        pc_data = {}
        im_data = {}
        if _LOAD_PC_IM:
            for pc_dict in frame_dict["pc_frame"]["pointclouds"]:
                pc_data[pc_dict["name"]] = self._load_pointcloud(pc_dict["url"])

            for im_dict in frame_dict["im_frame"]["cameras"]:
                im_data[im_dict["name"]] = self._load_image(im_dict["url"])

        laser_data = self._load_consecutive_lasers(frame_dict["laser_frame"]["url"])

        frame_dict.update(
            {
                "frame_id": int(frame_dict["frame_id"]),
                "sequence": self.sequence_handle[idx_sq].sequence,
                "first_frame": idx_fr == 0,
                "idx": idx,
                "pc_data": pc_data,
                "im_data": im_data,
                "laser_data": laser_data,
                "pc_anns": pc_anns,
                "im_anns": im_anns,
                "im_dets": im_dets,
                "laser_grid": np.linspace(
                    -np.pi, np.pi, laser_data.shape[1], dtype=np.float32
                ),
                "laser_z": -0.5 * np.ones(laser_data.shape[1], dtype=np.float32),
            }
        )

        return frame_dict

    @staticmethod
    def box_is_on_ground(jrdb_ann_dict):
        bottom_h = float(jrdb_ann_dict["box"]["cz"]) - 0.5 * float(
            jrdb_ann_dict["box"]["h"]
        )

        return bottom_h < -0.69  # value found by examining dataset

    @property
    def sequence_beginning_inds(self):
        return copy.deepcopy(self._sequence_beginning_inds)

    def _load_pointcloud(self, url):
        """Load a point cloud given file url.

        Returns:
            pc (np.ndarray[3, N]):
        """
        # pcd_load =
        # o3d.io.read_point_cloud(os.path.join(self.data_dir, url), format='pcd')
        # return np.asarray(pcd_load.points, dtype=np.float32)
        pc = point_cloud_from_path(os.path.join(self.data_dir, url)).pc_data
        # NOTE: redundent copy, ok for now
        pc = np.array([pc["x"], pc["y"], pc["z"]], dtype=np.float32)
        return pc

    def _load_image(self, url):
        """Load an image given file url.

        Returns:
            im (np.ndarray[H, W, 3]): (H, W) = (480, 3760) for stitched image,
                (480, 752) for individual image
        """
        im = cv2.imread(os.path.join(self.data_dir, url), cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        return im

    def _load_consecutive_lasers(self, url):
        """Load current and previous consecutive laser scans.

        Args:
            url (str): file url of the current scan

        Returns:
            pc (np.ndarray[self.num_scan, N]): Forward in time with increasing
                row index, i.e. the latest scan is pc[-1]
        """
        fpath = os.path.dirname(url)
        current_frame_idx = int(os.path.basename(url).split(".")[0])
        frames_list = []
        for del_idx in reversed(range(self.__num_scans)):
            frame_idx = max(0, current_frame_idx - del_idx * self.__scan_stride)
            url = os.path.join(fpath, str(frame_idx).zfill(6) + ".txt")
            frames_list.append(self._load_laser(url))

        return np.stack(frames_list, axis=0)

    def _load_laser(self, url):
        """Load a laser given file url.

        Returns:
            pc (np.ndarray[N, ]):
        """
        return np.loadtxt(os.path.join(self.data_dir, url), dtype=np.float32)


class _SequenceHandle:
    def __init__(self, data_dir, sequence, use_unlabeled_frames=False):
        self.sequence = sequence
        self._use_unlabeled_frames = use_unlabeled_frames

        # load frames of the sequence
        timestamp_dir = os.path.join(data_dir, "timestamps")
        fname = os.path.join(timestamp_dir, self.sequence, "frames_pc_im_laser.json")
        with open(fname, "r") as f:
            """
            list[dict]. Each dict has following keys:
                pc_frame: dict with keys frame_id, pointclouds, laser, timestamp
                im_frame: same as above
                laser_frame: dict with keys url, name, timestamp
                frame_id: same as pc_frame["frame_id"]
                timestamp: same as pc_frame["timestamp"]
            """
            self.frames = json.load(f)["data"]

        # load 3D annotation
        pc_label_dir = os.path.join(data_dir, "labels", "labels_3d")
        fname = os.path.join(pc_label_dir, f"{self.sequence}.json")
        with open(fname, "r") as f:
            """
            dict, key is the pc file name, value is the labels (list[dict])
            Each label is a dict with keys:
                attributes
                box
                file_id
                observation_angle
                label_id
            """
            self.pc_labels = json.load(f)["labels"]

        # load 2D annotation
        im_label_dir = os.path.join(data_dir, "labels", "labels_2d_stitched")
        fname = os.path.join(im_label_dir, f"{self.sequence}.json")
        with open(fname, "r") as f:
            """
            dict, key is the im file name, value is the labels (list[dict])
            Each label is a dict with keys:
                attributes
                truncated (bool)
                interpolated (bool)
                occlusion (str)Fully_visible
                area (float)
                no_eval (bool)
                box (list) (x0, y0, w, h)
                file_id (str)
                label_id (str) e.g. "pedestrian:46"
            """
            self.im_labels = json.load(f)["labels"]

        # load 2D detection
        im_det_dir = os.path.join(data_dir, "detections", "detections_2d_stitched")
        fname = os.path.join(im_det_dir, f"{self.sequence}.json")
        with open(fname, "r") as f:
            """
            dict, key is the im file name, value is the provided detections (list[dict])
            Each detection is a dict with keys:
                box (list) (x0, y0, w, h)
                file_id (str)
                label_id (str) e.g. "person:-1"
                score (float)
            """
            self.im_dets = json.load(f)["detections"]

        # find out which frames has 3D annotation
        self.frames_labeled = []
        for frame in self.frames:
            pc_file = os.path.basename(frame["pc_frame"]["pointclouds"][0]["url"])
            if pc_file in self.pc_labels:
                self.frames_labeled.append(frame)

        # choose if labeled or all frames are used
        self.data_frames = (
            self.frames if self._use_unlabeled_frames else self.frames_labeled
        )

    def __len__(self):
        return len(self.data_frames)

    def __getitem__(self, idx):
        # NOTE It's important to use a copy as the return dict, otherwise the
        # original dict in the data handle will be corrupted
        frame = copy.deepcopy(self.data_frames[idx])

        if self._use_unlabeled_frames:
            return frame, [], [], []

        pc_file = os.path.basename(frame["pc_frame"]["pointclouds"][0]["url"])
        pc_anns = copy.deepcopy(self.pc_labels[pc_file])

        im_file = os.path.basename(frame["im_frame"]["cameras"][0]["url"])
        im_anns = copy.deepcopy(self.im_labels[im_file])
        im_dets = copy.deepcopy(self.im_dets[im_file])

        return frame, pc_anns, im_anns, im_dets
