#!/usr/bin/env python3
"""
Class to perform semantic mapping.
"""
# Takes pose from wrapper and PC labeled from NeuralNet
# Publishes map to visualization
import rospy
from std_msgs.msg import String

def pose_callback(data):
    # Take the point cloud input
    pose = data

def pc_callback(data):
    # Take the point cloud input
    pose = data

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('mapping', anonymous=False)

    rospy.Subscriber('scan_wrapper', String, pose_callback)

    rospy.Subscriber('NeuralNet', String, pc_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()