import rospy
from std_msgs.msg import String

# Neural Net takes PC, then publishes semantic PC
def main_loop():
    # Publisher for semantic point clouds
    semantic_publisher = rospy.Publisher("semantic_pc", String, queue_size=10)
    # Listener for point clouds
    rospy.init_node('NeuralNet')
    rospy.Subscriber('point_cloud', String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

def semantic_mapping(pc):
    return pc

def callback(data):
    # Take the point cloud input
    pc = data
    sem_pc = semantic_mapping(pc)
    semantic_publisher.publish(sem_pc)

if __name__ == '__main__':
    main_loop()