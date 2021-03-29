#!/usr/bin/env python3
"""
Class to load scans into ros and send them in at a fixed rate.
"""
import argparse
import numpy as np

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose

# Note: this depends on the structure of our project
def load_scan(filename):
    """
    Load a scan from a file and return a point cloud message.

    Inputs:
        - filename: Name of scan to load.
    Outputs:
        - pose, as SE(3) transform
        - pointcloud, as list of x,y,z points
    """
    # Pass filename
    # point_cloud = np.fromfile(filename, dtype=np.float32).reshape(-1,4)
    point_cloud = "PC"
    pose = "Pose"
    return point_cloud, pose


def load_poses_from_file(filename):
    scan_poses = []
    poses_raw = np.genfromtxt(filename, delimiter=' ')
    for idx in range(0, poses_raw.shape[0]):
        pose = np.eye(4)
        pose[0, 0:4] = poses_raw[idx, 0:4]
        pose[1, 0:4] = poses_raw[idx, 4:8]
        pose[2, 0:4] = poses_raw[idx, 8:12]
        pose[3, 3] = 1.0

        # TODO: Apply calibration
        scan_poses.append(pose)

    return scan_poses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--poses", required=True, help="path to poses file")

    args = parser.parse_args()

    lidar_publisher = rospy.Publisher("point_cloud", String, queue_size=10)
    pose_publisher = rospy.Publisher("pose", String, queue_size=10)

    rospy.init_node("scan_wrapper")
    rate = rospy.Rate(10)

    load_poses_from_file(args.poses)

    counter = 0

    try:
        while not rospy.is_shutdown():
            pc, pose = load_scan(counter)
            lidar_publisher.publish(pc)
            pose_publisher.publish(pose)

            rate.sleep()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
