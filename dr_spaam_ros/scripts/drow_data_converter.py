#!/usr/bin/env python
import argparse
from math import sin, cos
import numpy as np

import rospy
import rosbag

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage


def load_scans(fname):
    data = np.genfromtxt(fname, delimiter=",")
    seqs, times, scans = data[:, 0].astype(np.uint32), data[:, 1].astype(np.float32), data[:, 2:].astype(np.float32)
    return seqs, times, scans


def load_odoms(fname):
    data = np.genfromtxt(fname, delimiter=",")
    seqs, times = data[:, 0].astype(np.uint32), data[:, 1].astype(np.float32)
    odos = data[:, 2:].astype(np.float32)   # x, y, phi
    return seqs, times, odos


def sequence_to_bag(seq_name, bag_name):
    scan_msg = LaserScan()
    scan_msg.header.frame_id = 'sick_laser_front'
    scan_msg.angle_min = np.radians(-225.0 / 2)
    scan_msg.angle_max = np.radians(225.0 / 2)
    scan_msg.range_min = 0.005
    scan_msg.range_max = 100.0
    scan_msg.scan_time = 0.066667
    scan_msg.time_increment = 0.000062
    scan_msg.angle_increment = (scan_msg.angle_max - scan_msg.angle_min) / 450

    tran = TransformStamped()
    tran.header.frame_id = 'base_footprint'
    tran.child_frame_id = 'sick_laser_front'

    with rosbag.Bag(bag_name, 'w') as bag:
        # write scans
        seqs, times, scans = load_scans(seq_name)
        for seq, time, scan in zip(seqs, times, scans):
            time = rospy.Time(time)
            scan_msg.header.seq = seq
            scan_msg.header.stamp = time
            scan_msg.ranges = scan
            bag.write('/sick_laser_front/scan', scan_msg, t=time)
        
        # write odometry
        seqs, times, odoms = load_odoms(seq_name[:-3] + 'odom2')
        for seq, time, odom in zip(seqs, times, odoms):
            time = rospy.Time(time)
            tran.header.seq = seq
            tran.header.stamp = time
            tran.transform.translation.x = odom[0]
            tran.transform.translation.y = odom[1]
            tran.transform.translation.z = 0.0
            tran.transform.rotation.x = 0.0
            tran.transform.rotation.y = 0.0
            tran.transform.rotation.z = sin(odom[2] * 0.5)
            tran.transform.rotation.w = cos(odom[2] * 0.5)
            tf_msg = TFMessage([tran])
            bag.write('/tf', tf_msg, t=time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--seq", type=str, required=True, help="path to sequence")
    parser.add_argument("--output", type=str, required=False, default="./out.bag")
    args = parser.parse_args()

    sequence_to_bag(args.seq, args.output)
