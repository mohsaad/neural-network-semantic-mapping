# Neural Network-based Semantic Mapping

![image](https://user-images.githubusercontent.com/6224951/114659252-411d6a00-9cc1-11eb-8051-5e5b1ea8dcc7.png)

### Joey Wilson, Jianyang Tang, Justin Chao, Mohammad Saad

This repo contains the work done for our final project for EECS568: Mobile Robotics. We developed a semantic mapping algorithm
that iteratively takes in poses and point cloud scans, passes them through a neural network, and builds a 3-dimensional voxel grid
semantic represenation of the world. We build this entire pipeline in ROS so it can be run on any computer, and visualize using rviz.

The pipeline uses the MinkuNet implemented by Tang et. al in their paper [Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution](https://arxiv.org/abs/2007.16100) for the semantic segmentation on point clouds, and the mapping algorithm defined in [Bayesian Spatial Kernel Smoothing for Scalable Dense Semantic Mapping](https://ieeexplore.ieee.org/abstract/document/8954837) by Gan et al. We utilize the kernel function defined to output a semantic class
score per voxel and display the mean of each cell.

## Requirements

```
Python 3
numpy
PyTorch 1.8
libsparsehash-dev (sudo apt install libsparsehash-dev)
torchsparse (pip3 install --upgrade git+https://github.com/mit-han-lab/torchsparse.git)
spvnas (included in repo)
```

Tested with ROS Melodic and Ubuntu 18.04. If you're using ROS < Noetic, you'll need to compile ROS with python3 support. [This](https://gist.github.com/drmaj/20b365ddd3c4d69e37c79b01ca17587a) is a good tutorial on how to do so.

## Dataset

![image](https://user-images.githubusercontent.com/6224951/114766606-9c3c7480-9d34-11eb-816c-555ee526bfd3.png)

We evaluated our dataset using the SemanticKitti dataset, using sequence 04 as our baseline. The dataset can be found [here](http://semantic-kitti.org/).

## Usage

Install dependencies:

```
sudo apt install libsparsehash-dev
pip3 install --upgrade git+https://github.com/mit-han-lab/torchsparse.git
```

Create a folder called catkin_ws:

```
mkdir -p ~/catkin_ws/src
```

Clone this repo into there:

```
git clone https://github.com/mohsaad/neural-network-semantic-mapping.git
```

Run `git submodule update` to pull the spvnas submodule (needed for running Minkunet)

```
git submodule update
```

Build:

```
cd ~/catkin_ws
catkin_make
```

To run, run the following scripts and rviz:

```
python3 src/scan_wrapper.py -p <path to poses.txt file> -v <path to folder of scans
python3 src/NeuralNet.py
python3 src/mapping.py
```

To visualize the output, open up rviz:

```
rosrun rviz rviz
```

and subscribe to the /map topic. You will also likely need to run static_transform_publisher:

```
rosrun tf2 static_transform_publisher 0 0 0 0 0 0 0.1
```

## Demo

https://youtu.be/DWks8yIeaGQ

## Acknowledgements

We would like to thank Prof. Maani Ghaffari Jadidi, GSIs Xi Lin and Cynthia Lin for their help and instruction in this class.
