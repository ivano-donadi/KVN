# Introduction

This repository includes the experimental 3D Object Localization module developed inside the Saipem Augmented Reality project. This module is basically composed by two customized version of the following 3D pose estimation frameworks:


[D2CO: Direct Directional Chamfer Optimization registration method](https://albertopretto.altervista.org/papers/miap_arxiv2016.pdf)

[PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation](https://arxiv.org/pdf/1812.11788.pdf)


In the following you can find an installation guide to install the prerequisites and to build the software.

## Installation

This guide explains how to install and build the software on a Ubuntu 18.04 x86_64 clean distribution, with no Nvidia drivers already installed. Commands that start with the '$' symbols should be exectuted inside a Bash terminal.

### Upgrade all packages

~~~bash
$ sudo apt update
$ sudo apt upgrade
~~~

### Install the Nvidia drivers

~~~bash
$ sudo add-apt-repository ppa:graphics-drivers
$ sudo apt install nvidia-driver-450
~~~

### Install the CUDA Toolkit

~~~bash
$ cd ~/Downloads
$ wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.39_linux.run
$ sudo sh cuda_10.1.105_418.39_linux.run
~~~

Be careful to install **only** the CUDA toolkit and nothing else, i.e. during the CUDA installation procedure select only the CUDA Toolkit checkbox:

~~~
│ CUDA Installer                                                               │
│ - [ ] Driver                                                                 │
│      [ ] 418.39                                                              │
│ + [X] CUDA Toolkit 10.1                                                      │
│   [ ] CUDA Samples 10.1                                                      │
│   [ ] CUDA Demo Suite 10.1                                                   │
│   [ ] CUDA Documentation 10.1                                                │
~~~

Add the following two lines at the end of your ~/.bashrc configuration file:

~~~
export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
~~~

Reboot and check the Nvidia drivers and CUDA installation with the command:

~~~bash
$ nvidia-smi
~~~

### [Optional] Install the cuDNN Libraries

There isn't a direct link: you need to register to the nvidia site to do that:

[developer.nvidia.com/rdp/cudnn-download](https://developer.nvidia.com/rdp/cudnn-download)

Look for the cuDNN Runtime Library and the cuDNN Developer Library, both for Ubuntu18.04 x86_64 (Deb) and CUDA 10.1. Install the downloaded packages as a normal .dep packages, e.g.:

~~~bash
$ cd ~/Downloads
$ sudo dpkg -i libcudnn8_8.0.4.30-1+cuda11.0_amd64.deb
$ sudo dpkg -i libcudnn8-dev_8.0.4.30-1+cuda11.0_amd64.deb
~~~

### Install some general prerequisites

~~~bash
$ sudo apt-get install build-essential git cmake libgoogle-glog-dev libdime-dev
$ sudo apt-get install libatlas-base-dev libeigen3-dev libsuitesparse-dev libopencv-dev
$ sudo apt-get install libpcl-dev libyaml-cpp-dev libgtest-dev libfreeimage-dev libglew-dev
$ sudo apt-get install libglfw3-dev libglm-dev meshlab python3-pip python3-numpy python3-tk
~~~

There is a known issue with Ubuntu 18.04 and libgtest, please follow this furter instruction for completing the installation of the gtest-dev library:

~~~bash
$ cd /usr/src/gtest
$ sudo cmake CMakeLists.txt
$ sudo make
$ sudo cp *.a /usr/lib
~~~

### Install Ceres Solver

~~~bash
$ cd ~/Downloads
$ wget http://ceres-solver.org/ceres-solver-1.14.0.tar.gz
$ tar zxf ceres-solver-1.14.0.tar.gz
$ cd ceres-solver-1.14.0/
$ mkdir build
$ cd build/
$ cmake ..
$ make -j3
$ sudo make install
~~~

### Install OpenMesh

~~~bash
$ cd ~/Downloads
$ wget https://www.graphics.rwth-aachen.de/media/openmesh_static/Releases/8.1/OpenMesh-8.1.tar.gz
$ tar zxf OpenMesh-8.1.tar.gz 
$ cd OpenMesh-8.1/
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make -j3
$ sudo make install
~~~

### Clone the software (if not done yet)

It is important to add the '--recurse-submodules' when cloning the repository.

~~~bash
$ git clone --recurse-submodules git@saipem.gitlab.host:Simulator/augmentedreality/object_localization.git
$ cd object_localization 
$ git checkout --track origin/initial_commit 
$ git submodule update
~~~

### Build the C++ stuff

~~~bash
$ mkdir build
$ cd build/
$ cmake ..
$ make -j3
~~~

### Install the Python3 prerequisistes

~~~bash

$ pip3 install --no-dependencies numpy==1.16.4 pillow==8.0.1 six==1.15.0
$ pip3 install torch==1.4.0 torchvision==0.5.0 --no-dependencies --no-cache-dir
$ pip3 install --no-dependencies torchvision==0.2.1
~~~

Check if Cython is already installed with the command:

~~~bash
$ pip3 show Cython
~~~

If it is already installed, and the version is not == 0.28.2, take note of the current version X.Y.Z and remove it with:

~~~bash
$ pip3 uninstall Cython
~~~

At the end of this installation procedure, you may restore the already installed Cython X.Y.Z version with:

~~~bash
pip3 install --no-cache-dir Cython==X.Y.Z
~~~

Then install Cython version 0.28.2

~~~bash
$ pip3 install --no-cache-dir Cython==0.28.2
~~~

Install the remaining requirements with

~~~bash
$ pip3 install --no-cache-dir -r pvnet/requirements.txt
~~~

### Build the Cython stuff:

~~~bash
$ cd ~/object_localization/pvnet/lib/csrc
$ cd dcn_v2
$ python3 setup.py build_ext --inplace --force
$ cd ../ransac_voting
$ python3 setup.py build_ext --inplace --force
$ cd ../nn
$ python3 setup.py build_ext --inplace --force
$ cd ../fps
$ python3 setup.py build_ext --inplace --force
$ cd ../uncertainty_pnp
$ python3 setup.py build_ext --inplace --force
~~~

## PVNet Object Localizer

PVNet is a data-driven based pose estimation method that requires labeled dataset to learn a model of the objects. A dataset is basically composed by images of the objects of interest and the related labels. Labels for pose estimation are represented by the 3D positions (translations and rotations) of the objects with respect to the camera frame.

### PVNet datasets

A typical PVNet  dataset folder should containt 3 empty folders (mask/, pose/, rgb/) with object black/white masks, rgb images and ground truth positions of the objects, respectively. The folder should also contain the 3D cad model of the object (with name model.ply), a file that just reports the diameter in meters of a sphere inscribing the object (diameter.txt), and file that reports the calibration matrix K in space separataed format, with column-major order:

~~~
k11 k12 k13
k21 k22 k33
k31 k32 k33
~~~

Here an example of the folder tree:

~~~
├── output/pvnet/dataset/folder
│   ├── model.ply
│   ├── camera.txt
│   ├── diameter.txt
│   ├── rgb/
│   │   ├── 0.jpg
│   │   ├── ...
│   │   ├── 1234.jpg
│   │   ├── ...
│   ├── mask/
│   │   ├── 0.png
│   │   ├── ...
│   │   ├── 1234.png
│   │   ├── ...
│   ├── pose/
│   │   ├── pose0.npy
│   │   ├── ...
│   │   ├── pose1234.npy
│   │   ├── ...
│   │   └──
~~~

It is possible to synthetically generate datasets from a CAD models using the generate_pvnet_dataset application (see notes in the following of this guide).

### PVNet network training

To train a model, move to the object_localization/pvnet directory. First you have to prepare the dataset you want to use for training (contained in the &lt;dataset_folder&gt; direcotry) with the command:

~~~bash
$ python3 run.py --type custom <dataset_folder>
~~~

Then you can start the trainig phase with the command:

~~~bash
$ python3 train_net.py --cfg_file configs/custom.yaml model_dir <model_folder> train.batch_size \
          <batch_size> train.dataset_dir <train dataset_dir> test.dataset_dir <test dataset_dir>
~~~

Where:

~~~
<model_folder> is the directory where the models will be stored
<batch_size> is the number of images used to train the network at each iteration
<train dataset_dir> and <test dataset_dir> are the train and test dataset folders, respectively,
                                           prepared with the command   
                                           $ python3 run.py --type custom_test ...
~~~

In most cases &lt;train dataset_dir&gt; and &lt;test dataset_dir&gt; are the same folder

### Network testing (with visualization)

To test the trained network over a set of stored  images, use command:

~~~bash
$ python3 run.py --type visualize --cfg_file configs/custom.yaml model_dir <model_folder> \
          test.dataset_dir <dataset_folder> [test.epoch <epoch_number>]
~~~

Where:

~~~
<model_folder> is the directory where the trained models have been stored
<test dataset_dir> is the images folder
<epoch_number> is the optional model epoch number (E.g., if you want to use the model file 123.pth, 
               use 123 as  <epoch_number>) if not provided, the test will be performed using the model 
               with the highest epoch number.
~~~

### Synthetic dataset generation for PVNet

The generate_pvnet_dataset application is capable to synthetically generate training datasets strating
from 3D CAD models (.ply or .stl file), to be used to train the PVNet network.
This application renders a set of view of the objects, randomply sampling surfece colors, lights position,
and optionally the image background, sampled from a given background set.

To generate a basic dataset, move to the object_localization/bin folder and run:


~~~bash

$ ./generate_pvnet_dataset -m <cad_model> -c <camera_model> -d <output_images_folder> 
                           -r <num_icosphere_iteration> 
                           --min_d <object_min_depth> --max_d <object_max_depth> 
                           --d_step <object_depth_sample_step>
                           --min_h <object_min_height> --max_h <object_max_height> 
                           --h_step <object_height_sample_step>
                           [-u <unit> -b <background_images_folder> --display]
~~~

where

~~~
<cad_model> is a STL, PLY, OBJ, ...  3D CAD model file
<camera_model> is A YAML file that stores all the camera parameters 
               (see the cv_ext::PinholeCameraModel object)
<output_images_folder> is the output dataset directory
<num_icosphere_iteration>  Rotation subdivision level [0,1,2,3, ..]: the greater this number, 
                           the greater the number of rotations
<object_min_depth> Minimum distance in meters from the object (can be negative)
<object_max_depth> Maximum distance in meters from the object (can be negative)
<object_depth_sample_step> Maximum distance in meters from the object (can be negative)
<object_min_height> Minimum height in meters from the object (can be negative)
<object_max_height> Maximum height in meters from the object (can be negative)
<object_height_sample_step> Maximum height in meters from the object (can be negative)
<unit> Optional unit of measure of the CAD model: [m|cm|mm], default: cm
<background_images_folder> is the optional background images directory
The --display option can optionally display dataset images
~~~

At the end of the process, you will find a dataset in <output_images_folder> ready to be processed with
PVNet.

For an online guide to the available options, run the command:

~~~bash
./generate_pvnet_dataset -h
~~~

#### Examples

~~~bash
$ ./generate_pvnet_dataset -m 01.ply -c unity_cam.yml -d pvnet_datasets/01 -r 2 -b abstract_images 
                           --min_d 6 --max_d 16 --d_step 1.5 --min_h -2 --max_h 2 --h_step 0.5 --display
~~~

### TODOs  

./generate_d2co_templates  
./obj_model   
$ python3 run.py --type visualize

./ generate_d2co_templates -m /home/albe/Datasets/saipem/simplified_cad_models/01.ply -c /home/albe/Datasets/saipem/pvnet_datasets/unity_cam.yml -t /home/albe/Datasets/saipem/d2co_templates/01.bin -n 256 -r 2 --min_d 3.5 --max_d 16 --d_step 0.2 --display  
./generate_d2co_templates -m /home/albe/Datasets/saipem/simplified_cad_models/02.ply -c /home/albe/Datasets/saipem/pvnet_datasets/unity_cam.yml -t /home/albe/Datasets/saipem/d2co_templates/02.bin -n 256 -r 2 --min_d 3.5 --max_d 16 --d_step 0.2 --display
./generate_d2co_templates -m /home/albe/Datasets/saipem/simplified_cad_models/03.ply -c /home/albe/Datasets/saipem/pvnet_datasets/unity_cam.yml -t /home/albe/Datasets/saipem/d2co_templates/03.bin -n 256 -r 2 --min_d 3.5 --max_d 16 --d_step 0.2 --off_y 0.125 --display

./tm_localization -m /home/albe/Datasets/saipem/simplified_cad_models/01.ply -c /home/albe/Datasets/saipem/pvnet_datasets/unity_cam.yml -f /home/albe/Datasets/saipem/unity_hd/eni_1_cropped_flipped -t /home/albe/Datasets/saipem/d2co_templates/01.bin
./tm_localization -m /home/albe/Datasets/saipem/simplified_cad_models/02.ply -c /home/albe/Datasets/saipem/pvnet_datasets/unity_cam.yml -f /home/albe/Datasets/saipem/unity_hd/eni_2_cropped_flipped -t /home/albe/Datasets/saipem/d2co_templates/02.bin
./tm_localization -m /home/albe/Datasets/saipem/simplified_cad_models/03.ply -c /home/albe/Datasets/saipem/pvnet_datasets/unity_cam.yml -f /home/albe/Datasets/saipem/unity_hd/eni_3_cropped_flipped -t /home/albe/Datasets/saipem/d2co_templates/03.bin --off_y 0.125

CUDA 10.1
