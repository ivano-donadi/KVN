/*
 * cv_ext - openCV EXTensions
 *
 *  Copyright (c) 2020, Alberto Pretto <alberto.pretto@flexsight.eu>
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "cv_ext/cv_ext.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

using namespace boost::python;
namespace py = boost::python;

std::vector<cv::Point3f> read_3d_ndarray(const boost::python::numpy::ndarray& input){
  int input_size = input.shape(0);
  double* input_ptr = reinterpret_cast<double*>(input.get_data());
  std::vector<cv::Point3f> output(input_size);
  for(int i = 0; i < input_size; i++){
    cv::Point3f pt;
    pt.x = (float) input_ptr[(3*i)+0];
    pt.y = (float) input_ptr[(3*i)+1];
    pt.z = (float) input_ptr[(3*i)+2];
    output[i] = pt;
  }
  return output;
}

std::vector<cv::Point2f> read_2d_ndarray(const boost::python::numpy::ndarray& input){
  int input_size = input.shape(0);
  float* input_ptr = reinterpret_cast<float*>(input.get_data());
  std::vector<cv::Point2f> output(input_size);
  for(int i = 0; i < input_size; i++){
    cv::Point2f pt;
    pt.x = input_ptr[(2*i)+0];
    pt.y = input_ptr[(2*i)+1];
    output[i] = pt;
  }
  return output;
}

std::vector<Eigen::Matrix2f> read_variance(const boost::python::numpy::ndarray& input){
  int input_size = input.shape(0);
  float* input_ptr = reinterpret_cast<float*>(input.get_data());
  std::vector<Eigen::Matrix2f> output(input_size);
  for(int i = 0; i < input_size; i++){
    Eigen::Matrix2f var;
    var(0,0) = input_ptr[(4*i)+0];
    var(0,1) = input_ptr[(4*i)+1];
    var(1,0) = input_ptr[(4*i)+2];
    var(1,1) = input_ptr[(4*i)+3];
    //std::cout<<"var "<<var<<std::endl;
    output[i] = var;
  }
  return output;
}


cv::Mat read_K_matrix(const boost::python::numpy::ndarray& input) {
  int input_size = input.shape(0) * input.shape(1);
  if(input_size != 9)
    throw std::runtime_error("K matrix should have 9 elements");
  double* input_ptr = reinterpret_cast<double*>(input.get_data());
  cv::Mat K(3,3,CV_64FC1);
  for (int i = 0; i < 9; i++){
    K.at<double>((i / 3), (i%3)) = input_ptr[i];
  }
  return K;
}

Eigen::Matrix3d read_R_matrix(const boost::python::numpy::ndarray& input) {
  int input_size = input.shape(0) * input.shape(1);
  if(input_size != 9)
    throw std::runtime_error("R matrix should have 9 elements");
  double* input_ptr = reinterpret_cast<double*>(input.get_data());
  Eigen::Matrix3d R;
  for (int i = 0; i < 9; i++){
    R((i / 3), (i%3)) = input_ptr[i];
  }
  return R;
}

Eigen::Vector3d read_t_vec(const boost::python::numpy::ndarray& input){
  int input_size = input.shape(0);
  if(input_size != 3)
    throw std::runtime_error("t vector should have 3 elements");
  double* input_ptr = reinterpret_cast<double*>(input.get_data());
  Eigen::Vector3d t_vec;
  t_vec(0,0) = input_ptr[0];
  t_vec(1,0) = input_ptr[1];
  t_vec(2,0) = input_ptr[2];
  return t_vec;
}


boost::python::numpy::ndarray array_to_ndarray(const double* array, int size){
  py::tuple shape = py::make_tuple(size);
  py::tuple stride = py::make_tuple(sizeof(double));
  boost::python::numpy::dtype dt = boost::python::numpy::dtype::get_builtin<double>();
  boost::python::numpy::ndarray output = boost::python::numpy::from_data(&array[0], dt, shape, stride, py::object());
  return output;
}

boost::python::numpy::ndarray r_mat_to_ndarray(const Eigen::Matrix3d& r_mat){
  boost::python::numpy::dtype dt = boost::python::numpy::dtype::get_builtin<double>();
  py::tuple stride = py::make_tuple(sizeof(double)*3,sizeof(double));
  py::tuple shape = py::make_tuple(3,3);
  double data[] = {r_mat(0,0), r_mat(0,1), r_mat(0,2),r_mat(1,0), r_mat(1,1), r_mat(1,2), r_mat(2,0), r_mat(2,1), r_mat(2,2) };
  boost::python::numpy::ndarray output = boost::python::numpy::from_data(data, dt, shape, stride, py::object());
  return output.copy();
}

boost::python::numpy::ndarray t_vec_to_ndarray(const Eigen::Vector3d& t_vec){
  boost::python::numpy::dtype dt = boost::python::numpy::dtype::get_builtin<double>();
  py::tuple stride = py::make_tuple(sizeof(double));
  py::tuple shape = py::make_tuple(3);
  double data[] = {t_vec(0,0), t_vec(1,0), t_vec(2,0)};
  boost::python::numpy::ndarray output = boost::python::numpy::from_data(&data[0], dt, shape, stride, py::object());
  return output.copy();
}

class IterativePnPStereoVariance
{
public:

  void setCamModel ( const cv_ext::PinholeCameraModel& cam_model );
  void py_setCamModel (const boost::python::numpy::ndarray& K, int width, int height, float baseline);
  void py_setInitialTransformation(const boost::python::numpy::ndarray& R, const boost::python::numpy::ndarray& t);
  void setNumIterations( int n ){ num_iterations_ = n; };

  void compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts, const std::vector<cv::Point2f> &proj_pts_r,
                double r_quat[4], double t_vec[3], const std::vector<Eigen::Matrix2f>& var, const std::vector<Eigen::Matrix2f>& var_r );
  void compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts, const std::vector<cv::Point2f> &proj_pts_r,
                Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec, const std::vector<Eigen::Matrix2f>& var, const std::vector<Eigen::Matrix2f>& var_r);
  void compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts, const std::vector<cv::Point2f> &proj_pts_r,
                cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec, const std::vector<Eigen::Matrix2f>& var, const std::vector<Eigen::Matrix2f>& var_r );
  py::tuple py_compute(const boost::python::numpy::ndarray& obj_pts_py, const boost::python::numpy::ndarray& proj_pts_py, const boost::python::numpy::ndarray& proj_pts_py_r, const boost::python::numpy::ndarray& var, const boost::python::numpy::ndarray& var_r);
  
  IterativePnPStereoVariance() {
    transf_ << 1.0, 0, 0, 0, 0, 0, 1.0, 0;
  }
private:
  
  void compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts, const std::vector<cv::Point2f> &proj_pts_r, const std::vector<Eigen::Matrix2f>& var, const std::vector<Eigen::Matrix2f>& var_r );
  
  Eigen::Matrix< double, 8, 1> transf_;

  cv_ext::PinholeCameraModel cam_model_;

  int num_iterations_ = 100;
  double fixed_value_;
  double baseline_;
  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  
};

BOOST_PYTHON_MODULE(iterative_pnp_stereo_variance){
  boost::python::numpy::initialize();
  class_<IterativePnPStereoVariance>("PNPSolverVariance", init<>())
    .def("compute", &IterativePnPStereoVariance::py_compute)
    .def("setNumIterations", &IterativePnPStereoVariance::setNumIterations)
    .def("setCamModel", &IterativePnPStereoVariance::py_setCamModel)
    .def("setInitialTransformation", &IterativePnPStereoVariance::py_setInitialTransformation)
    ;
}
