#!/usr/bin/env python3
import rospy
from std_msgs.msg import Header, ColorRGBA, String
from geometry_msgs.msg import Point as Point3d
from visualization_msgs.msg import Marker, MarkerArray
from neural_network_semantic_mapping.msg import *
import numpy as np

import ctypes as c
import itertools
import multiprocessing as mp
import time

COLOR_MAP = np.array(['#f59664', '#f5e664', '#963c1e',
                      '#b41e50', '#ff0000', '#1e1eff',
                      '#c828ff', '#5a1e96', '#ff00ff',
                      '#ff96ff', '#4b004b', '#4b00af', 
                      '#00c8ff', '#3278ff', '#00af00', 
                      '#003c87', '#50f096', '#96f0ff', 
                      '#0000ff', '#ffffff'])

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def _init(a, l):
    global update_array, lock
    update_array = a
    lock = l
# Occupancy Grid Mapping Class
class Mapping:
    def __init__(self):
        # map dimensions
        self.x_r = 400
        self.y_r = 400
        self.z_r = 2

        self.range_x = [-self.x_r, self.x_r]
        self.range_y = [-self.y_r, self.y_r]
        self.range_z = [-self.z_r, self.z_r]

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

        # MarkerArray for viz
        self.map_msg = Marker()
        self.map_msg.type = 6
        self.map_msg.action = 0
        self.map_msg.header = Header()
        self.map_msg.header.frame_id = "map"
        self.map_msg.scale.x = 1
        self.map_msg.scale.y = 1
        self.map_msg.scale.z = 1


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
      if m_x + self.x_r >= self.max_x or m_y + self.y_r >= self.max_y or m_z + self.z_r >= self.max_z:
        return -1, -1 , -1

      return m_x + self.x_r, m_y + self.y_r, m_z + self.z_r


    def data_structure_to_map(self, idx, jdx, kdx):
        """
        Return the map cell associated with the current index in the map.
        """
        if idx < 0 or idx > 2*self.x_r or jdx < 0 or jdx > 2*self.y_r or kdx < 0 or kdx > 2*self.z_r:
            return -1, -1, -1

        return idx - self.x_r, jdx - self.y_r, kdx - self.z_r

    def _kernel(self, distance, sigma_0=0.1):
        if distance >= self.l:
            return 0

        pt1 = 1.0 / 3 * (2 + np.cos(2 * np.pi * distance / self.l))
        pt2 = 1 - distance  / self.l
        pt3 = (1 / (2 * np.pi)) * np.sin(2*np.pi * distance / self.l)

        return sigma_0 * (pt1 * pt2 + pt3)

    def process_point(self, pose_xyz, point, label):
        """
        Multiprocessing function for processing points.
        """
        global_x = pose_xyz[0] + point[0]
        global_y = pose_xyz[1] + point[1]
        global_z = pose_xyz[2] + point[2]

        m_x = int(np.floor(global_x))
        m_y = int(np.floor(global_y))
        m_z = int(np.floor(global_z))

        distance = np.sqrt(np.sum(np.power(point, 2)))
        pose_theta = np.arctan2(pose_xyz[1], pose_xyz[0])
        theta = np.arctan2(point[1], point[0])
        theta2 = np.arctan2(point[2], point[0])

        if not self.point_in_percep_field(distance):
            return None

        # convert world map to data structure
        ds_x, ds_y, ds_z = self.map_to_data_structure(m_x, m_y , m_z)
        if ds_x < 0 or ds_y < 0 or ds_z <0:
            return None

        update = 0.0

        if distance < self.l:
            update += self._kernel(distance)


        for r in np.arange(0, distance, self.l):
            p_x = pose_xyz[0] + r * np.cos(theta + pose_theta)
            p_y = pose_xyz[1] + r * np.sin(theta + pose_theta) 
            p_z = pose_xyz[2] + r * np.sin(theta2)          

            d2 = np.sqrt((m_x - p_x) ** 2 + (m_y - p_y) ** 2+(m_z - p_z)**2)
            update += self._kernel(d2)

        return (ds_x, ds_y, ds_z, label, update)

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



    def build_ogm_iterative_parallel(self, pose, scan, labels):
        """
        Iteratively build a map, processing points in parallel
        """
        pose_xyz = pose[0:3, 3]
        print(pose_xyz)

        assert(len(scan) == len(labels))
        pool = mp.Pool(12)

        result = pool.starmap(self.process_point, zip(itertools.repeat(pose_xyz), scan, labels))
        pool.close()
        pool.join()

        for idx in range(0, len(result)):
            if result[idx] == None:
                continue
            self.map['alpha'][result[idx][0], result[idx][1], result[idx][2], result[idx][3]] += result[idx][4]

    def optimize_map(self):
        """
        Optimize the map and compute mean and variance.
        """
        self.map_msg.points = []
        self.map_msg.colors = []

        for idx in range(self.map['alpha'].shape[0]):
            for jdx in range(self.map['alpha'].shape[1]):
                for kdx in range(self.map['alpha'].shape[2]):
                    alpha_sum = np.sum(self.map['alpha'][idx, jdx,kdx, :])
                    self.map['mean'][idx, jdx,kdx, :] = self.map['alpha'][idx, jdx,kdx, :] * 1.0 / alpha_sum

                    alpha_i = self.map['alpha'][idx, jdx,kdx, :]
                    max_idx = np.argmax(alpha_i)
                    max_alpha = alpha_i[max_idx] / alpha_sum
                    var_num_i = (max_alpha)*(1 - max_alpha)
                    var_den_i = alpha_sum + 1
                    self.map['variance'][idx, jdx,kdx] = var_num_i / var_den_i


        classes = self.map['mean'].reshape(-1, self.num_classes + 1)
        grid = np.vstack([self.X.flatten(), self.Y.flatten(), self.Z.flatten()]).T
        semantic_class = np.argmax(classes, axis=1)
        semantic_class[semantic_class == self.num_classes] = 19

        indices = np.where(semantic_class != 0)
        grid = grid[indices]
        semantic_class = semantic_class[indices]
        assert(len(grid) > 0)

        for idx in range(0, len(grid)):
            p = Point3d()
            p.x = grid[idx, 0]
            p.y = grid[idx, 1]
            p.z = grid[idx, 2]

            c = ColorRGBA()
            rgb = hex_to_rgb(COLOR_MAP[semantic_class[idx]])
            c.r = rgb[0]
            c.g = rgb[1]
            c.b = rgb[2]
            c.a = 1

            self.map_msg.points.append(p)
            self.map_msg.colors.append(c)

        return self.map_msg

class MappingSubscriber:
    def __init__(self, publisher):
        self.publisher = publisher
        self.map = Mapping()
        self.map.construct_map()

    def callback(self, sem_pc):
        pc_data, pc_labels = self.make_np(sem_pc)
        pose = self.make_pose(sem_pc)
        t1 = time.time()
        self.map.build_ogm_iterative(pose, pc_data, pc_labels)

        map_msg = self.map.optimize_map()
        t2 = time.time()
        print(t2 - t1)
        self.publisher.publish(map_msg)

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
    map_publisher = rospy.Publisher("map", Marker, queue_size=10)
    # Listener for point clouds
    rospy.init_node('mapping')

    MS = MappingSubscriber(map_publisher)
    rospy.Subscriber('semantic_pc', PointCloud, MS.callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    main_loop()
