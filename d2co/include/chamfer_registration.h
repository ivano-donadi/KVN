/*
 * d2co - Direct Directional Chamfer Optimization
 *
 *  Copyright (c) 2020, Alberto Pretto <alberto.pretto@flexsight.eu>
 *                      Marco Imperoli <marco.imperoli@flexsight.eu>
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

#include <memory>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

#include "cv_ext/cv_ext.h"
#include "object_registration.h"
#include "distance_transforms.h"

class ChamferRegistration : public ObjectRegistration
{
public:
  
  virtual ~ChamferRegistration(){};  
  void  setInput( const cv::Mat &dist_map );

  using ObjectRegistration::refinePosition;
  double refinePosition( const ObjectTemplate &templ,
                         cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec ) override
  {
    // TODO
    std::runtime_error("Implement me!!");
    return 0;
  };
protected:

  void updateOptimizer() override;
  double optimize() override;
  double avgDistance() override;
  
private:
  
  cv::Mat dist_map_;
  
  /* Pimpl idiom */
  class Optimizer;
  std::shared_ptr< Optimizer > optimizer_ptr_;
  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


class ICPChamferRegistration : public ObjectRegistration
{
public:
  
  ~ICPChamferRegistration() override {} ;
  
  void setInput( const cv::Mat &closest_edgels_map );
  void setNumIcpIterations( int n ){ num_icp_iterations_ = n; };

  using ObjectRegistration::refinePosition;
  double refinePosition( const ObjectTemplate &templ,
                         cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec ) override
  {
    // TODO
    throw std::runtime_error("Implement me!!");
    return 0;
  };
protected:

  void updateOptimizer() override;
  double optimize() override;
  double avgDistance() override;
  
private:
  
  cv::Mat closest_edgels_map_;
  std::vector<cv::Point3f> model_pts_;
  int num_icp_iterations_ = 50;
  
  /* Pimpl idiom */
  class Optimizer;
  std::shared_ptr< Optimizer > optimizer_ptr_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class DirectionalChamferRegistration : public ObjectRegistration
{
public:
  
  DirectionalChamferRegistration();
  ~DirectionalChamferRegistration() override {};

  void setNumDirections( int n );
  int numDirections(){ return num_directions_; };
  
  void setInput( const cv_ext::ImageTensorPtr &dist_map_tensor_ptr );

  using ObjectRegistration::refinePosition;
  double refinePosition( const ObjectTemplate &templ,
                         cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec );

  double refinePosition( const ObjectTemplate &templ,
                         Eigen::Quaterniond r_quat, Eigen::Vector3d t_vec );

  // TODO integrate&remove this function
  double getDistance( const std::vector< cv::Point2f >& proj_pts,
                      const std::vector< float >& normal_directions ) const;

protected:

  void updateOptimizer() override;
  double optimize() override;
  double avgDistance() override;
  
private:

  // TODO: remove this function?
//   inline float templateDist( const std::vector<cv::Point> &proj_pts, const std::vector<int> &dirs );
  
  cv_ext::ImageTensorPtr dist_map_tensor_ptr_;

//   DirectionalIntegralImageVectorPtr int_dist_map_tensor_ptr_;
  
  /* Pimpl idiom */
  class Optimizer;
  std::shared_ptr< Optimizer > optimizer_ptr_;
  
  int num_directions_ = 60;
  float eta_direction_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


class HybridDirectionalChamferRegistration : public ObjectRegistration
{
public:

  HybridDirectionalChamferRegistration();
  virtual ~HybridDirectionalChamferRegistration() override {};

  void setNumDirections( int n );
  int numDirections(){ return num_directions_; };

  void setInput( const cv_ext::ImageTensorPtr &dist_map_tensor_ptr,
                 const cv_ext::ImageTensorPtr &edgels_map_tensor_ptr );

  void setMaxIcpIterations( int n ){ max_icp_iterations_ = n; };

  using ObjectRegistration::refinePosition;
  double refinePosition( const ObjectTemplate &templ,
                         cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec ) override
  {
    // TODO
    throw std::runtime_error("Implement me!!");
    return 0;
  };
protected:

  void updateOptimizer() override;
  double optimize() override;
  double avgDistance() override;

private:

  cv_ext::ImageTensorPtr dist_map_tensor_ptr_;
  cv_ext::ImageTensorPtr edgels_map_tensor_ptr_;
//   DirectionalIntegralImageVectorPtr int_dist_map_tensor_ptr_;
  std::vector<cv::Point3f> model_pts_, model_dpts_;

  /* Pimpl idiom */
  class Optimizer;
  std::shared_ptr< Optimizer > d2co_optimizer_ptr_, icp_optimizer_ptr_;

  int max_icp_iterations_ = 5;
  int num_directions_ = 60;
  float eta_direction_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class BidirectionalChamferRegistration  : public ObjectRegistration
{
public:
  
  BidirectionalChamferRegistration ();
  ~BidirectionalChamferRegistration () override {};

  void setNumDirections( int n );
  int numDirections(){ return num_directions_; };
  void setInput( const cv_ext::ImageTensorPtr &x_dist_map_tensor_ptr,
                 const cv_ext::ImageTensorPtr &y_dist_map_tensor_ptr );

  using ObjectRegistration::refinePosition;
  double refinePosition( const ObjectTemplate &templ,
                         cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec ) override
  {
    // TODO
    throw std::runtime_error("Implement me!!");
    return 0;
  };

protected:

  void updateOptimizer() override;
  double optimize() override;
  double avgDistance() override;
  
private:
  
  cv_ext::ImageTensorPtr x_dist_map_tensor_ptr_, y_dist_map_tensor_ptr_;
//   DirectionalIntegralImageVectorPtr int_dist_map_tensor_ptr_;
  
  /* Pimpl idiom */
  class Optimizer;
  std::shared_ptr< Optimizer > optimizer_ptr_;
  
  int num_directions_;
  float eta_direction_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

//typedef std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d> > PoseVec;
//typedef std::vector<cv_ext::ImageTensorPtr, Eigen::aligned_allocator<cv_ext::ImageTensorPtr> > ImageTensorPtrVec;
//struct MultiViewsInput
//{
//  RasterObjectModel3DPtr model_ptr;
//  PoseVec views;
//  ImageTensorPtrVec dist_map_tensor_ptr_vec;
//
//public:
//  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//};

//typedef std::vector<MultiViewsInput, Eigen::aligned_allocator<MultiViewsInput> > MultiViewsInputVec;

class MultiViewsDirectionalChamferRegistration  : public MultiViewObjectRegistration
{
public:
  MultiViewsDirectionalChamferRegistration ();
  virtual ~MultiViewsDirectionalChamferRegistration () override {};

  void setNumDirections( int n );
  int numDirections(){ return num_directions_; };
  void setInput( const std::vector < cv_ext::ImageTensorPtr > &dist_map_tensor_ptrs );

  using MultiViewObjectRegistration::refinePosition;
  double refinePosition( const ObjectTemplate &templ,
                         cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec ) override;
protected:

  void updateOptimizer() override;
  double optimize() override;
  double avgDistance() override;

private:

  /* Pimpl idiom */
  class Optimizer;
  std::shared_ptr< Optimizer > optimizer_ptr_;

  std::vector < cv_ext::ImageTensorPtr > dist_map_tensor_ptrs_;

  int num_directions_ = 60;
  float eta_direction_;
  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class ICPDirectionalChamferRegistration : public ObjectRegistration
{
public:
  
  ICPDirectionalChamferRegistration();
  ~ICPDirectionalChamferRegistration() override  {};

  void setNumDirections( int n );
  int numDirections(){ return num_directions_; };
  void setInput( const cv_ext::ImageTensorPtr &edgels_map_tensor_ptr );
  void setNumIcpIterations( int n ){ num_icp_iterations_ = n; };

  using ObjectRegistration::refinePosition;
  double refinePosition( const ObjectTemplate &templ,
                         cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec ) override
  {
    std::cout<<"Implement me!!"<<std::endl;
    // TODO
    throw std::runtime_error("Implement me!!");
    return 0;
  };
protected:

  void updateOptimizer() override;
  double optimize() override;
  double avgDistance() override;
  
private:
  
  cv_ext::ImageTensorPtr edgels_map_tensor_ptr_;
  std::vector<cv::Point3f> model_pts_;
  int num_icp_iterations_;
  
  /* Pimpl idiom */
  class Optimizer;
  std::shared_ptr< Optimizer > optimizer_ptr_;
  
  int num_directions_ = 60;
  float eta_direction_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
