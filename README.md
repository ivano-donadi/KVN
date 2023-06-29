# KVNet: Keypoints Voting Network with Differentiable RANSAC for Stereo Pose Estimation


## Prerequisites (tested on Ubuntu 20.04)

 - Python 3.8
 - CUDA drivers (tested on version 11.6)
 - cuDNN libraries (tested on vesion 8.7.0)
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
This process might take a while, since it has to generate all ground truth object egmentation masks for the right camera images. Metadata such as keypoints 3D position and object models is taken from the corresponding folder inside `data/metafiles`. The generated annotations are stored inside `data/sy_datasets` divided by object and texture. For example, the annotations for the object ball_0 when the training textures are 1-9 and the test texture is texture 0 are stored inside `data/sy_datasets/ball_0_stereo_0`.

## KVNet training 

To train a model, ensure to have completed the TOD dataset annotation steps detailed above, then move to the `pvnet` directory. From here you can start training the model with the `pvnet_train_parallel.py` script:

```bash
$ python3 pvnet_train_parallel.py -h

    usage: pvnet_train_parallel.py [-h] -d DATASET_DIR -m MODEL_DIR [-b BATCH_SIZE] [-n NUM_EPOCH] [-e EVAL_EP] [-s SAVE_EP] [--bkg_imgs_dir BKG_IMGS_DIR] [--disable_resume] [--cfg_file CFG_FILE]

    KVNet training tool

    -h, --help            show this help message and exit
    -d DATASET_DIR, --dataset_dir DATASET_DIR
                            Input directory containing the training dataset
    -m MODEL_DIR, --model_dir MODEL_DIR
                            Output directory where the trained models will be stored
    -b BATCH_SIZE, --batch_size BATCH_SIZE
                            Number of training examples in one forward/backward pass (default = 2)
    -n NUM_EPOCH, --num_epoch NUM_EPOCH
                            Number of epochs to train (default = 240)
    -e EVAL_EP, --eval_ep EVAL_EP
                            Number of epochs after which to evaluate (and eventually save) the model (default = 5)
    -s SAVE_EP, --save_ep SAVE_EP
                            Number of epochs after which to save the model (default = 5)
    --bkg_imgs_dir BKG_IMGS_DIR
                            Optional background images directory, to be used to augment the dataset
    --disable_resume      If specified, disable train resume and start a new train
    --cfg_file CFG_FILE   Low level configuration file, DO NOT CHANGE THIS PARAMETER IF YOU ARE NOT SURE (default = configs/custom_dsac.yaml)

    You need at least to provide an input training dataset and to specify the output directory where the trained models will be stored.The best model checkpoint will be stored inside the best_model subdirectory
```

For example, following the procedure at the previous section, it is possible to train the model on textures 1-9 of object ball_0 by using:

```bash
python3 pvnet_train_parallel.py -d ../data/sy_datasets/ball_0_stereo_0 -m ../results -n 150 -e 10 -s 10 --cfg_file configs/custom_dsac.yaml
```

This command will train the model using DSAC as the training loss and will save a checkpoint every 10 epochs inside the results folder, named 9.pth, 19.pth and so on. The checkpoint with the best validation parameters is saved inside `results/best_model`. To perform the same training but with PVNet's l1 vote loss you simply need to choose `configs/custom_vanilla.yaml` as the configuration file. Additionally, it is possible to perform random background augmentation by expliciting the `--bkg_imgs` option with the path to the backgrounds dataset. We provide the set of backgrounds that were used in our experiments at [this link](). #### TODO: add link 
In case of correct execution, the output of this script will be the network's training process and the evaluation results on the validation set at the specified epochs interval.

## KVNet evaluation

Assuming to have completed the training procedure at the previous section, it is now possible to evaluate the trained model on tecture 0 of object ball_0 with the `pvnet_eval_parallel.py` script:

```bash
$ python3 pvnet_eval_parallel.py -h
    usage: pvnet_eval_parallel.py [-h] -d DATASET_DIR -m MODEL [--cfg_file CFG_FILE]

    KVNet evaluation tool

    -h, --help            show this help message and exit
    -d DATASET_DIR, --dataset_dir DATASET_DIR
                            Input directory containing the test dataset
    -m MODEL, --model MODEL
                            KVNet trained model
    --cfg_file CFG_FILE   Low level configuration file, DO NOT CHANGE THIS PARAMETER IF YOU ARE NOT SURE (default = configs/custom_dsac.yaml)

    You need at least to provide an (annotated) input test dataset and a KVNet trained model
```

In the case of the current example it should be used in the following way (assuming that the best checkpoint is 89.pth):

```bash
$ python3 pvnet_eval_parallel.py -d ../data/sy_datasets/ball_0_stereo_0 -m ../results/best_model/89.pth --cfg_file configs/custom_dsac.yaml
```

If you are evaluating a model trained with the classical PVNet loss, then you just need to specify `configs/custom_vanilla.yaml` as the configuration file.
The output of this script will be the evaluation results both using monocular images and stereo images.

## KVNet prediction visualization

```bash
$ python3 pvnet_test_localization_parallel.py -h

    usage: pvnet_test_localization_parallel.py [-h] -l MODEL -f META_FILE -i IMAGE [--cfg_file CFG_FILE] ...

    Locate an object from an input image

    -h, --help            show this help message and exit
    -l MODEL, --model MODEL
                            KVNet trained model
    -f META_FILE, --meta_file META_FILE
                            PVNet inference meta file (e.g., inference_meta.yaml)
    -i IMAGE, --image IMAGE
                            Input image or folder (e.g., image.jpg or images/)
    --cfg_file CFG_FILE   Low level configuration file, DO NOT CHANGE THIS PARAMETER IF YOU ARE NOT SURE (default = configs/custom_dsac.yaml)
```