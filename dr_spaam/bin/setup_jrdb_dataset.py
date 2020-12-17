import json
import numpy as np
import os
import rosbag
import shutil

# Set root dir to JRDB
_jrdb_dir = "./data/JRDB"

# Variables defining output location.
# Do not change unless for you know what you are doing.
_output_laser_dir_name = "lasers"
_output_frames_laser_im_fname = "frames_img_laser.json"
_output_frames_laser_pc_fname = "frames_pc_laser.json"
_output_laser_timestamp_fname = "timestamps.txt"


def _laser_idx_to_fname(idx):
    return str(idx).zfill(6) + ".txt"


def extract_laser_from_rosbag(split):
    """Extract and save combined laser scan from rosbag. Existing files will be overwritten.
    """
    data_dir = os.path.join(_jrdb_dir, split + "_dataset")

    timestamp_dir = os.path.join(data_dir, "timestamps")
    bag_dir = os.path.join(data_dir, "rosbags")
    sequence_names = os.listdir(timestamp_dir)

    laser_dir = os.path.join(data_dir, _output_laser_dir_name)
    if os.path.exists(laser_dir):
        shutil.rmtree(laser_dir)
    os.mkdir(laser_dir)

    for idx, seq_name in enumerate(sequence_names):
        seq_laser_dir = os.path.join(laser_dir, seq_name)
        os.mkdir(seq_laser_dir)

        bag_file = os.path.join(bag_dir, seq_name + ".bag")
        print(
            "({}/{}) Extract laser from {} to {}".format(
                idx + 1, len(sequence_names), bag_file, seq_laser_dir
            )
        )
        bag = rosbag.Bag(bag_file)

        # extract all laser msgs
        timestamp_list = []
        for count, (topic, msg, t) in enumerate(
            bag.read_messages(topics=["segway/scan_multi"])
        ):
            scan = np.array(msg.ranges)
            fname = _laser_idx_to_fname(count)
            np.savetxt(os.path.join(seq_laser_dir, fname), scan, newline=" ")

            timestamp_list.append(t.to_sec())

        np.savetxt(
            os.path.join(seq_laser_dir, _output_laser_timestamp_fname),
            np.array(timestamp_list),
            newline=" ",
        )

        bag.close()


def _get_frame_image_dict(cameras_dict, sequence_name):
    """Get a dictionary containing images of the frame, for looking up image files.

    Args:
        cameras_dict (list[dict]): frames_img.json["data"][frame]["cameras"]
        sequence_name (str)

    Return:
        im_dict (dict): Keys "image_stitched" or "image_X", where X is camera id.
    """
    im_dict = {}
    for cam_dict in cameras_dict:
        url = cam_dict["url"]
        assert sequence_name in url
        fname = os.path.split(url)[1]
        cam_name = cam_dict["name"]
        if "stitched" in cam_name:
            im_dict["image_stitched"] = {
                "url": os.path.join("./images/image_stitched", sequence_name, fname),
                "timestamp": cam_dict["timestamp"],
                "name": "image_stitched",
            }
        else:
            cam_name = "image_" + cam_name[-1]
            im_dict[cam_name] = {
                "url": os.path.join("images", cam_name, sequence_name, fname),
                "timestamp": cam_dict["timestamp"],
                "name": cam_name,
            }

    return im_dict


def _get_frame_pointcloud_dict(pointclouds_dict, sequence_name):
    """Get a dictionary containing pointclouds of the frame, for looking up
    corresponding files.

    Args:
        pointclouds_dict (list[dict]): frames_pc.json["data"][frame]["pointclouds"]
        sequence_name (str)

    Return:
        pc_dict (dict): Keys "lower_velodyne" or "upper_velodyne
    """
    pc_dict = {}
    for pc in pointclouds_dict:
        url = pc["url"]
        assert sequence_name in url
        fname = os.path.split(url)[1]
        pc_name = pc["name"]
        pc_dict[pc_name] = {
            "url": os.path.join("pointclouds", pc_name, sequence_name, fname),
            "timestamp": pc["timestamp"],
            "name": pc_name,
        }

    return pc_dict


def _add_laser_to_frames_dict(frames_dict, sequence_name, laser_dir):
    """Add laser information to each frame in frames_dict

    Args:
        frames_dict (list[dict]): frames_img.json["data"] (or similar)
        sequence_name (str)
        laser_dir (str): Directory to the laser data of the same sequence

    Returns:
        frames_dict (list[dict]): Same frames dict with an additional field for laser
    """
    # timestamps of all frame
    frames_timestamp = [fd["timestamp"] for fd in frames_dict]
    frames_timestamp = np.array(frames_timestamp, dtype=np.float64)

    # timestamps of laser
    laser_timestamp = np.loadtxt(
        os.path.join(laser_dir, _output_laser_timestamp_fname), dtype=np.float64
    )

    # match timestamp
    del_t = np.abs(frames_timestamp.reshape(-1, 1) - laser_timestamp.reshape(1, -1))
    argmin_del_t = np.argmin(del_t, axis=1)
    # min_del_t = del_t[np.arange(del_t.shape[0]), argmin_del_t]
    # argmin_del_t[min_del_t > max_del_t] = -1  # no matching data found

    # add matching laser
    for f_idx, (f_dict, closest_laser_idx) in enumerate(zip(frames_dict, argmin_del_t)):
        min_del_t = del_t[f_idx, closest_laser_idx]
        if min_del_t > 0.1 or min_del_t < 0:
            print(
                "Bad matching between frame and laser timestamp, "
                "sequence {}, frame index {}, frame timestamp {}, laser timestamp {}, "
                "timestamp difference {}".format(
                    sequence_name,
                    f_idx,
                    frames_timestamp[f_idx],
                    laser_timestamp[closest_laser_idx],
                    min_del_t,
                )
            )

        f_dict["laser"] = {
            "url": os.path.join(
                _output_laser_dir_name,
                sequence_name,
                _laser_idx_to_fname(closest_laser_idx),
            ),
            "name": "laser_combined",
            "timestamp": laser_timestamp[closest_laser_idx],
        }

    return frames_dict


def _match_laser_with_image_one_sequence(split, sequence_name):
    """Write in timestamp dir a json file that contains url to matching laser and
    stitched image file. Existing files will be overwritten.

    Args:
        split (str): "train" or "test"
        sequence_name (str):
    """
    data_dir = os.path.join(_jrdb_dir, split + "_dataset")

    timestamp_dir = os.path.join(data_dir, "timestamps", sequence_name)
    laser_dir = os.path.join(data_dir, _output_laser_dir_name, sequence_name)

    im_frames_file = os.path.join(timestamp_dir, "frames_img.json")
    with open(im_frames_file, "r") as f:
        im_frames = json.load(f)

    im_laser_frames = {"data": []}
    for im_frame in im_frames["data"]:
        im_laser_frame = {
            "images": _get_frame_image_dict(im_frame["cameras"], sequence_name),
            "frame_id": im_frame["frame_id"],
            "timestamp": im_frame["timestamp"],
        }
        im_laser_frames["data"].append(im_laser_frame)

    # add matching laser scan for each frame
    im_laser_frames["data"] = _add_laser_to_frames_dict(
        im_laser_frames["data"], sequence_name, laser_dir
    )

    # check url is correct
    for frame_dict in im_laser_frames["data"]:
        for _, v in frame_dict["images"].items():
            url = v["url"]
            assert os.path.isfile(os.path.join(data_dir, url))
        laser_url = frame_dict["laser"]["url"]
        assert os.path.isfile(os.path.join(data_dir, laser_url))

    # write to file
    frame_fname = os.path.join(timestamp_dir, _output_frames_laser_im_fname)
    with open(frame_fname, "w") as fp:
        json.dump(im_laser_frames, fp)


def _match_laser_with_pointcloud_one_sequence(split, sequence_name):
    """Write in timestamp dir a json file that contains url to matching laser and
    pointcloud file. Existing files will be overwritten.

    Args:
        split (str): "train" or "test"
        sequence_name (str):
    """
    data_dir = os.path.join(_jrdb_dir, split + "_dataset")

    timestamp_dir = os.path.join(data_dir, "timestamps", sequence_name)
    laser_dir = os.path.join(data_dir, "lasers", sequence_name)

    pc_frames_file = os.path.join(timestamp_dir, "frames_pc.json")
    with open(pc_frames_file, "r") as f:
        pc_frames = json.load(f)

    pc_laser_frames = {"data": []}
    for pc_frame in pc_frames["data"]:
        pc_laser_frame = {
            "pointclouds": _get_frame_pointcloud_dict(
                pc_frame["pointclouds"], sequence_name
            ),
            "frame_id": pc_frame["frame_id"],
            "timestamp": pc_frame["timestamp"],
        }
        pc_laser_frames["data"].append(pc_laser_frame)

    # add matching laser scan for each frame
    pc_laser_frames["data"] = _add_laser_to_frames_dict(
        pc_laser_frames["data"], sequence_name, laser_dir
    )

    # check url is correct
    for frame_dict in pc_laser_frames["data"]:
        for _, v in frame_dict["pointclouds"].items():
            url = v["url"]
            assert os.path.isfile(os.path.join(data_dir, url))
        laser_url = frame_dict["laser"]["url"]
        assert os.path.isfile(os.path.join(data_dir, laser_url))

    # write to file
    frame_fname = os.path.join(timestamp_dir, _output_frames_laser_pc_fname)
    with open(frame_fname, "w") as fp:
        json.dump(pc_laser_frames, fp)


def match_laser_with_image_and_pointcloud(split):
    sequence_names = os.listdir(
        os.path.join(_jrdb_dir, split + "_dataset", "timestamps")
    )
    for idx, seq_name in enumerate(sequence_names):
        print(
            "({}/{}) Match laser data for sequence {}".format(
                idx + 1, len(sequence_names), seq_name
            )
        )
        _match_laser_with_image_one_sequence(split, seq_name)
        _match_laser_with_pointcloud_one_sequence(split, seq_name)


def _match_pc_im_laser_one_sequence(split, sequence_name):
    """Write in timestamp dir a json file that contains url to matching pointcloud,
    laser, and image. Existing files will be overwritten. Pointcloud is used as
    the main sensors which other sensors are synchronized to.

    Args:
        split (str): "train" or "test"
        sequence_name (str):
    """
    data_dir = os.path.join(_jrdb_dir, split + "_dataset")

    timestamp_dir = os.path.join(data_dir, "timestamps", sequence_name)
    laser_dir = os.path.join(data_dir, "lasers", sequence_name)

    # pc frames
    pc_frames_file = os.path.join(timestamp_dir, "frames_pc.json")
    with open(pc_frames_file, "r") as f:
        pc_frames = json.load(f)["data"]

    # im frames
    im_frames_file = os.path.join(timestamp_dir, "frames_img.json")
    with open(im_frames_file, "r") as f:
        im_frames = json.load(f)["data"]

    # synchronize pc and im frame
    pc_timestamp = np.array([float(f["timestamp"]) for f in pc_frames])
    im_timestamp = np.array([float(f["timestamp"]) for f in im_frames])

    pc_im_ft_diff = np.abs(pc_timestamp.reshape(-1, 1) - im_timestamp.reshape(1, -1))
    pc_im_matching_inds = pc_im_ft_diff.argmin(axis=1)

    # synchronize pc and laser
    laser_timestamp = np.loadtxt(
        os.path.join(laser_dir, _output_laser_timestamp_fname), dtype=np.float64
    )
    pc_laser_ft_diff = np.abs(
        pc_timestamp.reshape(-1, 1) - laser_timestamp.reshape(1, -1)
    )
    pc_laser_matching_inds = pc_laser_ft_diff.argmin(axis=1)

    # create a merged frame dict
    output_frames = []
    for i in range(len(pc_frames)):
        frame = {
            "pc_frame": pc_frames[i],
            "im_frame": im_frames[pc_im_matching_inds[i]],
            "laser_frame": {
                "url": os.path.join(
                    _output_laser_dir_name,
                    sequence_name,
                    _laser_idx_to_fname(pc_laser_matching_inds[i]),
                ),
                "name": "laser_combined",
                "timestamp": laser_timestamp[pc_laser_matching_inds[i]],
            },
            "timestamp": pc_frames[i]["timestamp"],
            "frame_id": pc_frames[i]["frame_id"],
        }

        # correct file url for pc and im
        for pc_dict in frame["pc_frame"]["pointclouds"]:
            f_name = os.path.basename(pc_dict["url"])
            pc_dict["url"] = os.path.join(
                "pointclouds", pc_dict["name"], sequence_name, f_name
            )

        for im_dict in frame["im_frame"]["cameras"]:
            f_name = os.path.basename(im_dict["url"])
            cam_name = (
                "image_stitched"
                if im_dict["name"] == "stitched_image0"
                else im_dict["name"][:-1] + "_" + im_dict["name"][-1]
            )
            im_dict["url"] = os.path.join("images", cam_name, sequence_name, f_name)

        output_frames.append(frame)

    # write to file
    output_dict = {"data": output_frames}
    f_name = os.path.join(timestamp_dir, "frames_pc_im_laser.json")
    with open(f_name, "w") as fp:
        json.dump(output_dict, fp)


def match_pc_im_laser(split):
    sequence_names = os.listdir(
        os.path.join(_jrdb_dir, split + "_dataset", "timestamps")
    )
    for idx, seq_name in enumerate(sequence_names):
        print(
            "({}/{}) Match sensor data for sequence {}".format(
                idx + 1, len(sequence_names), seq_name
            )
        )
        _match_pc_im_laser_one_sequence(split, seq_name)


if __name__ == "__main__":
    # extract_laser_from_rosbag("train")
    # match_laser_with_image_and_pointcloud("train")
    match_pc_im_laser("train")
    # extract_laser_from_rosbag("test")
    # match_laser_with_image_and_pointcloud("test")
    match_pc_im_laser("test")
