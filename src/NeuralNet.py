#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from neural_network_semantic_mapping.msg import *

import numpy as np
# PyTorch
import torch
import torch.nn as nn
# torchsparse is our high-performance 3D sparse convolution library.
import torchsparse
import torchsparse.nn as spnn
from torchsparse import SparseTensor
from torchsparse.utils import sparse_quantize, sparse_collate_fn

# import SPVNAS model from model zoo
from model_zoo import spvnas_specialized
from model_zoo import spvnas_supernet
from model_zoo import minkunet
from model_zoo import spvcnn

def process_point_cloud(input_point_cloud, voxel_size=0.05, ignore_label=19):
    input_point_cloud[:, 3] = input_point_cloud[:, 3]
    # get rounded coordinates
    pc_ = np.round(input_point_cloud[:, :3] / voxel_size)
    pc_ -= pc_.min(0, keepdims=1)
    feat_ = input_point_cloud
    # filter out unlabeled points
    out_pc = input_point_cloud[:, :3]
    pc_ = pc_[:]
    feat_ = feat_[:]

    # sparse quantization: filter out duplicate points after downsampling
    inds, inverse_map = sparse_quantize(pc_,
                                        feats=feat_,
                                        labels=None,
                                        return_index=True,
                                        return_invs=True)
    # construct members as sparse tensor so that they can be collated
    pc = pc_[inds]
    feat = feat_[inds]
    lidar = SparseTensor(
        feat, pc
    )
    inverse_map = SparseTensor(
        inverse_map, out_pc
    )
    out_pc = SparseTensor(
        out_pc, out_pc
    )
    # construct the feed_dict
    feed_dict = {
        'pc': out_pc,
        'lidar': lidar,
        'inverse_map': inverse_map
    }
    return feed_dict


# train_label_name_mapping = {
#       0: 'car', 1: 'bicycle', 2: 'motorcycle', 3: 'truck', 4:
#       'other-vehicle', 5: 'person', 6: 'bicyclist', 7: 'motorcyclist',
#       8: 'road', 9: 'parking', 10: 'sidewalk', 11: 'other-ground',
#       12: 'building', 13: 'fence', 14: 'vegetation', 15: 'trunk',
#       16: 'terrain', 17: 'pole', 18: 'traffic-sign'
#   }
class PointCloudSegmentation:
  def __init__(self):
    self.model = minkunet('SemanticKITTI_val_MinkUNet@29GMACs').to(device)
    self.model.eval()

  # Pass index into the point clouds/labels
  def segment_pc(self, point_cloud):
    # use sparse_collate_fn to create batch
    feed_dict = sparse_collate_fn([process_point_cloud(point_cloud)])

    # run inference
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    inputs = feed_dict['lidar'].to(device)
    outputs = self.model(inputs)
    predictions = outputs.argmax(1).cpu().numpy()

    # map predictions from downsampled sparse voxels to original points
    predictions = predictions[feed_dict['inverse_map'].F.int().cpu().numpy()]

    # Map back from sparse
    pc = feed_dict['pc'].F.cpu().numpy()
    inverse_map = feed_dict['inverse_map'].F.cpu().numpy()
    return pc, predictions

class PublishSubscribe:
    def __init__(self, publisher):
        self.publisher = publisher

        self.net = PointCloudSegmentation()

    def callback(self, pc_msg):
        """
        Processes each point cloud message.

        """
        pc_data = self.make_np(pc_msg)
        sem_pc = PointCloud()
        sem_pc.loc = pc_msg.loc
        sem_pc.points = self.semantic_labeling(pc_data)
        self.publisher.publish(sem_pc)
    def make_np(self, pc_msg):

        pc_points = pc_msg.points
        pc_data = [point.data for point in pc_points]
        pc_data = np.asarray(pc_data)
        return pc_data

    # This would be where the neural network would do stuff
    def semantic_labeling(self, pc):
        points = []
        point_cloud, predictions = self.net.segment_pc(oc)

        for i in range(point_cloud.shape[0]):
            new_pt = Point()
            new_pt.label = predictions[idx]
            new_pt.data = point_cloud[idx]
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
