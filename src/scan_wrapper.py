#!/usr/bin/env python3
"""
Class to load scans into ros and send them in at a fixed rate.
"""
import argparse
import numpy as np

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from neural_network_semantic_mapping.msg import *

def parse_calibration(filename):
    """
    read calibration file with given filename

    Returns
    -------
    dict: Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            key, content = line.strip().split(':')
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

    return calib


# Note: this depends on the structure of our project
# Need to make a messag eto hold pose and point cloud
def load_scan(counter, folder, poses):
    """
    Load a scan from a file and return a point cloud message.

    Inputs:
        - filename: Name of scan to load.
    Outputs:
        - pose, as SE(3) transform
        - pointcloud, as list of x,y,z points
    """
    filename = folder + "/" + "{:06}".format(counter) + ".bin"
    point_cloud = np.fromfile(filename, dtype=np.float32).reshape(-1,4)
    pose = poses[counter]

    pc_msg = PointCloud()
    pose_msg = Point()
    pose_msg.label = -1
    pose_msg.data = pose
    pc_msg.loc = pose_msg

    points = []
    for i in range(point_cloud.shape[0]):
        new_pt = Point()
        new_pt.label = -1
        new_pt.data = point_cloud[i]
        points.append(new_pt)
    pc_msg.points = points
    return pc_msg


def load_poses_from_file(filename, calibration=None):
    scan_poses = []
    poses_raw = np.genfromtxt(filename, delimiter=' ')
    # Tr = calibration["Tr"]
    # Tr_inv = inv(Tr)

    for idx in range(0, poses_raw.shape[0]):
        pose = np.eye(4)
        pose[0, 0:4] = poses_raw[idx, 0:4]
        pose[1, 0:4] = poses_raw[idx, 4:8]
        pose[2, 0:4] = poses_raw[idx, 8:12]
        pose[3, 3] = 1.0

        # TODO: Apply calibration
        # scan_poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        scan_poses.append(pose)

    return scan_poses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--poses", required=True, help="path to poses file")
    parser.add_argument('-v', "--velo", required=True, help="path to velodyne folder")

    # Add calib later
    # parser.add_argument('-c', "--calib", required=True, help="path to calib file")

    args = parser.parse_args()

    lidar_publisher = rospy.Publisher("point_cloud", PointCloud, queue_size=10)

    rospy.init_node("scan_wrapper")
    rate = rospy.Rate(10)

    scan_poses = load_poses_from_file(args.poses)
    # scan_poses = load_poses_from_file(args.poses, args.calib)

    counter = 0

    try:
        while not rospy.is_shutdown():
            pc_with_pose = load_scan(counter, args.velo, scan_poses)
            print("New pose: ", pc_with_pose.loc)
            lidar_publisher.publish(pc_with_pose)
            counter += 1

            rate.sleep()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
