import rospy
from std_msgs.msg import String


class MappingSubscriber:
    def __init__(self, publisher):
        self.publisher = publisher
    def pc_callback(self, data):
        print("pc")
    def pose_callback(self, data):
        print("pose")


def semantic_mapping(pc):
    return pc


# Mapping takes a point cloud and pose, and creates a map
def main_loop():
    # Publisher for semantic point clouds
    map_publisher = rospy.Publisher("map", String, queue_size=10)
    # Listener for point clouds
    rospy.init_node('mapping')

    MS = MappingSubscriber(map_publisher)
    rospy.Subscriber('semantic_pc', String, MS.pc_callback)
    rospy.Subscriber('pose', String, MS.pose_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    main_loop()