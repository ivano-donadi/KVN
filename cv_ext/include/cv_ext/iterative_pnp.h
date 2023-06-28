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

class IterativePnP
{
public:

  void setCamModel ( const cv_ext::PinholeCameraModel& cam_model );
  void setNumIterations( int n ){ num_iterations_ = n; };
  void constrainsTranslationComponent(int index, double value );
  void removeConstraints();

  void compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts, 
                double r_quat[4], double t_vec[3] );
  void compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts, 
                Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec );
  void compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts, 
                cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec );
  
private:
  
  void compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts );
  
  Eigen::Matrix< double, 8, 1> transf_;

  cv_ext::PinholeCameraModel cam_model_;

  int num_iterations_ = 100;
  int fixed_index_ = -1;
  double fixed_value_;
  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  
};
