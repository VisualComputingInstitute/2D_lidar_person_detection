import copy
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


__all__ = ["JRDBHandleDet3D"]


class JRDBHandleDet3D:
    def __init__(self, split, cfg, sequences=None, exclude_sequences=None):
        if _DEBUG_ONE_SAMPLE:
            split = "train"
            sequences = None
            exclude_sequences = None

        data_dir = os.path.abspath(os.path.expanduser(cfg["data_dir"]))
        data_dir = (
            os.path.join(data_dir, "train_dataset")
            if split == "train" or split == "val"
            else os.path.join(data_dir, "test_dataset")
        )

        self.data_dir = data_dir

        sequence_names = (
            os.listdir(os.path.join(data_dir, "timestamps"))
            if sequences is None
            else sequences
        )

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

        frame_dict = self.sequence_handle[idx_sq][idx_fr]
        urls = frame_dict["url"]

        frame_dict.update(
            {
                "frame_id": int(frame_dict["frame_id"]),
                "sequence": self.sequence_handle[idx_sq].sequence,
                "first_frame": idx_fr == 0,
                "dataset_idx": idx,
                "pc_upper": self.load_pointcloud(urls["pc_upper"]),
                "pc_lower": self.load_pointcloud(urls["pc_lower"]),
            }
        )

        if urls["label"] is not None:
            frame_dict["label_str"] = self.load_label(urls["label"])

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

    def load_pointcloud(self, url):
        """Load a point cloud given file url.

        Returns:
            pc (np.ndarray[3, N]):
        """
        # pcd_load =
        # o3d.io.read_point_cloud(os.path.join(self.data_dir, url), format='pcd')
        # return np.asarray(pcd_load.points, dtype=np.float32)
        pc = point_cloud_from_path(url).pc_data
        # NOTE: redundent copy, ok for now
        pc = np.array([pc["x"], pc["y"], pc["z"]], dtype=np.float32)
        return pc

    def load_label(self, url):
        with open(url, "r") as f:
            s = f.read()
        return s


class _SequenceHandle:
    def __init__(self, data_dir, sequence):
        self.sequence = sequence

        # pc frames
        pc_dir = os.path.join(data_dir, "pointclouds", "upper_velodyne", sequence)
        frames = [f.split(".")[0] for f in os.listdir(pc_dir)]

        # labels
        label_dir = os.path.join(data_dir, "labels_kitti", sequence)
        if os.path.exists(label_dir):
            labeled_frames = [f.split(".")[0] for f in os.listdir(label_dir)]
            frames = list(set(frames) & set(labeled_frames))

        self._upper_pc_dir = pc_dir
        self._lower_pc_dir = os.path.join(
            data_dir, "pointclouds", "lower_velodyne", sequence
        )
        self._label_dir = label_dir
        self._frames = frames
        self._frames.sort()
        self._load_labels = os.path.exists(label_dir)

    def __len__(self):
        return self._frames.__len__()

    def __getitem__(self, idx):
        frame = self._frames[idx]
        url_upper_pc = os.path.join(self._upper_pc_dir, frame + ".pcd")
        url_lower_pc = os.path.join(self._lower_pc_dir, frame + ".pcd")
        url_label = (
            os.path.join(self._label_dir, frame + ".txt") if self._load_labels else None
        )

        return {
            "frame_id": frame,
            "url": {
                "pc_upper": url_upper_pc,
                "pc_lower": url_lower_pc,
                "label": url_label,
            },
        }
