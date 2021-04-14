# Neural Network-based Semantic Mapping

### Joey Wilson, Jianyang Tang, Justin Chao, Mohammad Saad

This repo contains the work done for our final project for EECS568: Mobile Robotics. We developed a semantic mapping algorithm
that iteratively takes in poses and point cloud scnas, passes them through a neural network, and builds a 3-dimensional voxel grid
semantic represenation of the world.


## Requirements

```
Python 3
numpy
PyTorch 1.9
torchsparse
```

## Usage

Create a folder called catkin_ws:

```
mkdir -p ~/catkin_ws/src
```

Clone this repo into there:

```
git clone
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

## Demo

