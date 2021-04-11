#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from neural_network_semantic_mapping.msg import *
import numpy as np

# Occupancy Grid Mapping Class
class Mapping:
    def __init__(self):
        # map dimensions
        self.range_x = [-200, 200]
        self.range_y = [-200, 200]
        self.range_z = [-2, 2]

        # senesor parameters
        self.z_max = 30     # max range in meters
        self.n_beams = 133  # number of beams, we set it to 133 because not all measurements in the dataset contains 180 beams 

        # grid map parameters
        self.grid_size = 1                  # adjust this for task 4.B
        self.nn = 16                            # number of nearest neighbor search

        # map structure
        self.map = {}  # map
        self.pose = {}  # pose data
        self.scan = []  # laser scan data
        self.m_i = {}  # cell i
        self.target_map = []

        # continuous kernel parameter
        self.l = 0.5      # kernel parameter
        self.sigma = 0.1  # kernel parameter

        # semantic
        self.num_classes = 20

        self.prior = 0            # prior for setting up mean and variance
        self.prior_alpha = 1e-10  # a small, uninformative prior for setting up alpha

    def construct_map(self):
        # class constructor
        # construct map points, i.e., grid centroids
        
        x = np.arange(self.range_x[0], self.range_x[1]+self.grid_size, self.grid_size)
        y = np.arange(self.range_y[0], self.range_y[1]+self.grid_size, self.grid_size)
        z = np.arange(self.range_z[0], self.range_z[1]+self.grid_size, self.grid_size)
        X, Y, Z = np.meshgrid(x, y, z)
        self.X = X
        self.Y = Y
        self.Z = Z
        print(X.shape, Y.shape)
        t = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)))

        self.poses = []


        self.map['mean'] = self.prior * np.ones([X.shape[0], X.shape[1],X.shape[2], self.num_classes+1])        # size should be (number of data) x (number of classes + 1)
        self.map['variance'] = self.prior * np.ones([X.shape[0], X.shape[1],X.shape[2], 1])                     # size should be (number of data) x (1)
        self.map['alpha'] = self.prior_alpha * np.ones([X.shape[0], X.shape[1],X.shape[2], self.num_classes+1]) # size should be (number of data) x (number of classes + 1)
        self.max_x = X.shape[0]
        self.max_y = X.shape[1]
        self.max_z = X.shape[2]

    def point_in_percep_field(self, dist):
      """
      Given a point in the sensor frame, return whether it's in the perceptual field.
      """
      inside = False
      if dist > 0 and dist < self.z_max:
        inside = True

      return inside

    # Check if the indices are within the map's grid
    def map_to_data_structure(self, m_x, m_y, m_z):
      if m_x + 200 > self.max_x or m_y + 200 > self.max_y or m_z+2 >self.max_z:
        return -1, -1 , -1

      return m_x + 200, m_y + 200, m_z+2

    def _kernel(self, distance, sigma_0=0.1):
        if distance >= self.l:
            return 0
        pt1 = 1.0 / 3 * (2 + np.cos(2 * np.pi * distance / self.l))
        pt2 = 1 - distance  / self.l
        pt3 = (1 / (2 * np.pi)) * np.sin(2*np.pi * distance / self.l)

        return sigma_0 * (pt1 * pt2 + pt3)


    def build_ogm_iterative(self, pose, scan, labels):
        """
        Iteratively build a map using poses and scans.
        """
        pose_xyz = pose[0:3, 3]
        print(pose_xyz)
        pose_theta = np.arctan2(pose_xyz[1], pose_xyz[0])
        for idx in range(0, scan.shape[0]):
          global_x = pose_xyz[0] + scan[idx][0]
          global_y = pose_xyz[1] + scan[idx][1]
          global_z = pose_xyz[2] + scan[idx][2]

          m_x = int(np.floor(global_x))
          m_y = int(np.floor(global_y))
          m_z = int(np.floor(global_z))

          distance = np.sqrt(np.sum(np.power(scan[idx], 2)))
          theta = np.arctan2(scan[idx][1], scan[idx][0])
          theta2 = np.arctan2(scan[idx][2], scan[idx][0])

          if not self.point_in_percep_field(distance):
            continue

          # convert world map to data structure
          ds_x, ds_y, ds_z = self.map_to_data_structure(m_x, m_y , m_z)
          if ds_x < 0 or ds_y < 0 or ds_z <0:
            continue

          if distance < self.l:
            self.map['alpha'][ds_x, ds_y,ds_z, labels[idx]] += self._kernel(distance)

          for r in np.arange(0, distance, self.l):
              p_x = pose_xyz[0] + r * np.cos(theta + pose_theta)
              p_y = pose_xyz[1] + r * np.sin(theta + pose_theta) 
              p_z = pose_xyz[2] + r * np.sin(theta2)          

              d2 = np.sqrt((m_x - p_x) ** 2 + (m_y - p_y) ** 2+(m_z - p_z)**2)
              self.map['alpha'][ds_x, ds_y, ds_z, labels[idx]] += self._kernel(d2)

          # global_z = pose_xyz[2] + scan[idx][2]
    def optimize_map(self):
      """
      Optimize the map and compute mean and variance.
      """
      for idx in range(self.map['alpha'].shape[0]):
        for jdx in range(self.map['alpha'].shape[1]):
          for kdx in range(self.map['alpha'].shape[2]):
            alpha_sum = np.sum(self.map['alpha'][idx, jdx,kdx, :])
            self.map['mean'][idx, jdx,kdx, :] = self.map['alpha'][idx, jdx,kdx, :] / alpha_sum

            alpha_i = self.map['alpha'][idx, jdx,kdx, :]
            max_idx = np.argmax(alpha_i)
            max_alpha = alpha_i[max_idx] / alpha_sum
            var_num_i = (max_alpha)*(1 - max_alpha)
            var_den_i = alpha_sum + 1
            self.map['variance'][idx, jdx,kdx] = var_num_i / var_den_i

    def build_map_message(self):
        self.map.optimize_map()
        msg = MarkerArray()


class MappingSubscriber:
    def __init__(self, publisher):
        self.publisher = publisher
        self.map = Mapping()
        self.map.construct_map()

    def callback(self, sem_pc):
        pc_data, pc_labels = self.make_np(sem_pc)
        pose = self.make_pose(sem_pc)
        print(pose)

        self.map.build_ogm_iterative(pose, pc_data, pc_labels)

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
