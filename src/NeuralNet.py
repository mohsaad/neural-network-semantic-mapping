import rospy
from std_msgs.msg import String
from neural_network_semantic_mapping.msg import *
import numpy as np


class PublishSubscribe:
    def __init__(self, publisher):
        self.publisher = publisher
    def callback(self, pc_msg):
        pc_data = self.make_np(pc_msg)
        sem_pc = PointCloud()
        sem_pc.loc = pc_msg.loc
        sem_pc.points = semantic_labeling(pc_data)
        self.publisher.publish(sem_pc)
    def make_np(self, pc_msg):
        pc_points = pc_msg.points
        pc_data = [point.data for point in pc_points]
        pc_data = np.asarray(pc_data)
        return pc_data


# This would be where the neural network would do stuff
def semantic_labeling(pc):
    points = []
    for i in range(pc.shape[0]):
        new_pt = Point()
        new_pt.label = 2
        new_pt.data = pc[i]
        points.append(new_pt)
    return points


# Neural Net takes PC, then publishes semantic PC
def main_loop():
    # Publisher for semantic point clouds
    semantic_publisher = rospy.Publisher("semantic_pc", PointCloud, queue_size=10)
    # Listener for point clouds
    rospy.init_node('NeuralNet')

    PS = PublishSubscribe(semantic_publisher)
    rospy.Subscriber('point_cloud', PointCloud, PS.callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    main_loop()