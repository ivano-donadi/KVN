# D2CO: Direct Directional Chamfer Optimization #

This software implements the D2CO (Direct Directional Chamfer Optimization) algorithm, a fast edge-based registration technique for accurate 3D object pose estimation.

## References ##

Paper Describing the Approach:

M. Imperoli and A. Pretto D2CO: Fast and Robust Registration of 3D Textureless Objects Using the Directional Chamfer Distance In Proceedings of the 10th International Conference on Computer Vision Systems (ICVS 2015), July 6-9 , 2015 Copenhagen, Denmark, pages: 316-328 ([PDF](http://www.dis.uniroma1.it/~pretto/papers/miap_icvs2015.pdf))


```
#!bibtex

@inproceedings{miap_icvs2015,
  author={Imperoli, M. and Pretto, A.},
  title={{D\textsuperscript{2}CO}: Fast and Robust Registration of {3D}
         Textureless Objects Using the {Directional 
         Chamfer Distance}},
  booktitle={Proc. of 10th International Conference on 
             Computer Vision Systems (ICVS 2015)},
  year={2015},
  pages={316--328}
}
```

## Requirements ##

The code is tested on Ubuntu 16.04. D2CO requires different tools and libraries. To install them on Ubuntu, use the terminal command:


```
#!bash

sudo apt-get install build-essential cmake libeigen3-dev libdime-dev libdime1 libglew-dev libglew1.10 libglm-dev libglfw3-dev
```
- Follow this [guide](http://ceres-solver.org/installation.html) to install Ceres Solver.
- Click [here](http://www.openmesh.org/media/Releases/3.3/OpenMesh-3.3.tar.gz) to download OpenMesh.

## Building ##

To build D2CO on Ubuntu, type in a terminal the following command sequence.

```
#!bash

cd d2co
mkdir externals
cd externals
git clone https://bitbucket.org/alberto_pretto/cv_ext
cd cv_ext
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 2
cd ../../..
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 2

```
Test the library with the **test_localization** app (binary in /bin, source code in apps/test_localization.cpp, inline help with ./test_localization -h): given the 3D CAD model of the object and the image of the scene, **test_localization** can perform different registration methods including D2CO and ICP-based algorithms:

```
#!bash
cd bin
./test_localization -m  3D_models/AX-01b_bearing_box.stl -c test_images/test_camera_calib.yml -i test_images/2.png

```

## Contact information ##

- Alberto Pretto [pretto@diag.uniroma1.it](mailto:pretto@dis.uniroma1.it)
- Marco Imperoli [imperoli@diag.uniroma1.it](mailto:imperoli@dis.uniroma1.it)