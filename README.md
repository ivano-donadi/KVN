# KVNet: Keypoints Voting Network with Differentiable RANSAC for Stereo Pose Estimation


## Prerequisites (tested on Ubuntu 20.04)

 - Python 3.8
 - CUDA drivers 11.6 or higher
 - cuDNN libraries 8.7.0 or higher
 - ceres sover (tested on version 1.14.0)

### required libraries

~~~bash
$ sudo apt-get install build-essential git cmake libgoogle-glog-dev libdime-dev
$ sudo apt-get install libatlas-base-dev libeigen3-dev libsuitesparse-dev libopencv-dev
$ sudo apt-get install libpcl-dev libyaml-cpp-dev libgtest-dev libfreeimage-dev libglew-dev
$ sudo apt-get install libglfw3-dev libglm-dev meshlab python3-pip python3-numpy python3-tk
~~~

## Installation

Open a terminal window inside the cloned repository.

### Build the C++ stuff

~~~bash
$ mkdir build
$ cd build/
$ cmake ..
$ make -j3
~~~

### Install the Python3 prerequisistes

~~~bash
$ pip3 install --no-cache-dir -r pvnet/requirements.txt
~~~

### Build the Cython stuff:

~~~bash
$ cd pvnet/lib/csrc
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

## TOD dataset

To replicate the experiments presented in the paper you will first need to download the origianl TOD dataset, available at  [here](https://sites.google.com/view/keypose/home). The zip file for each object should be unzipped inside the `data/` directory and renamed to add a `_orig` suffix. For example, the original dataset for object 'ball_0' should be in `data/ball_0_orig`. To convert TOD's annotation into KVNet format, you can use the script `convert_all_textures` which is located inside `pvent/tod_utils`, which requires the path to the data folder and the name of the object. Assuming to have a terminal window inside `pvnet/tod_utils`, and assuming that we want to convert the dataset for the object `ball_0`, the command to use is the following:

```bash
$ sh convert_all_textures.sh ../../data ball_0
```
This process might take a while, since it has to generate all ground truth object egmentation masks for the right camera images. Metadata such as keypoints 3D position and object models is taken from the corresponding folder inside `data/metafiles`. The generated annotations are stored inside `data/sy_datasets` divided by object and texture. For example, the annotations for the object ball_0 when the training textures are 1-9 and the test texture is texture 0 are stored inside `data/sy_datasets/ball_0_texture_0`.

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

