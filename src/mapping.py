#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from neural_network_semantic_mapping.msg import *
import numpy as np

class MappingSubscriber:
    def __init__(self, publisher):
        self.publisher = publisher
        self.map = PointCloud()
    def callback(self, sem_pc):
        pc_data, pc_labels = self.make_np(sem_pc)
        pose = self.make_pose(sem_pc)
        print(pose)
        self.publisher.publish(sem_pc)
    def make_np(self, pc_msg):
        pc_points = pc_msg.points
        pc_data = [point.data for point in pc_points]
        pc_labels = [point.label for point in pc_points]
        pc_data = np.asarray(pc_data)
        pc_labels = np.asarray(pc_labels)
        return pc_data, pc_labels
    def make_pose(self, sem_pc):
        pose = sem_pc.loc.data
        pose = np.asarray(pose)
        pose = pose.reshape((4, 4))
        return pose


def semantic_mapping(pose, pc):
    return pc


# Mapping takes a point cloud and pose, and creates a map
def main_loop():
    # Publisher for semantic point clouds
    map_publisher = rospy.Publisher("map", PointCloud, queue_size=10)
    # Listener for point clouds
    rospy.init_node('mapping')

    MS = MappingSubscriber(map_publisher)
    rospy.Subscriber('semantic_pc', PointCloud, MS.callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    main_loop()