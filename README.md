# Neural Network-based Semantic Mapping

![image](https://user-images.githubusercontent.com/6224951/114659252-411d6a00-9cc1-11eb-8051-5e5b1ea8dcc7.png)

### Joey Wilson, Jianyang Tang, Justin Chao, Mohammad Saad

This repo contains the work done for our final project for EECS568: Mobile Robotics. We developed a semantic mapping algorithm
that iteratively takes in poses and point cloud scnas, passes them through a neural network, and builds a 3-dimensional voxel grid
semantic represenation of the world. We build this entire pipeline in ROS so it can be run on any computer, and visualize using rviz.


## Requirements

```
Python 3
numpy
PyTorch 1.8
torchsparse
spvnas (included in repo)
```

Tested with ROS Melodic and Ubuntu 18.04.

## Usage

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
