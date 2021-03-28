#!/usr/bin/env python3
"""
Class to load scans into ros and send them in at a fixed rate.
"""
import rospy
from std_msgs.msg import String
import numpy as np


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
    point_cloud = np.fromfile(filename, dtype=np.float32).reshape(-1,4)
    return None, None


def main():
    lidar_publisher = rospy.Publisher("point_cloud", String, queue_size=10)
    pose_publisher = rospy.Publisher("pose", String, queue_size=10)

    rospy.init_node("scan_wrapper")
    rate = rospy.Rate(10)

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
