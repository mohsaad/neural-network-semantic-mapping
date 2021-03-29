import rospy
from std_msgs.msg import String


class PublishSubscribe:
    def __init__(self, publisher):
        self.publisher = publisher
    def callback(self, data):
        pc = data.data
        sem_pc = semantic_labeling(pc)
        print(sem_pc)
        self.publisher.publish(sem_pc)


def semantic_labeling(pc):
    return pc


# Neural Net takes PC, then publishes semantic PC
def main_loop():
    # Publisher for semantic point clouds
    semantic_publisher = rospy.Publisher("semantic_pc", String, queue_size=10)
    # Listener for point clouds
    rospy.init_node('NeuralNet')

    PS = PublishSubscribe(semantic_publisher)
    rospy.Subscriber('point_cloud', String, PS.callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    main_loop()