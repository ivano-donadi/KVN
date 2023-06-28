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

#include "raster_object_model3D.h"
#include "object_templates.h"

class ObjectRegistrationBase
{
public:
  ObjectRegistrationBase();
  virtual ~ObjectRegistrationBase(){};

  void ensureCheiralityConstraint( bool enabled ) { ensure_cheirality_constraint_ = enabled; };
  bool cheiralityConstraintEnesured() const { return ensure_cheirality_constraint_; };

  void enableVerboseMode( bool enable ) { verbose_mode_ = enable; };
  void setNumOptimIterations( int n ){ num_optim_iterations_ = n; };
  int numOptimIterations() const { return num_optim_iterations_; };


  virtual double getAvgDistance( const double r_quat[4], const double t_vec[3] ) = 0;
  virtual double getAvgDistance( const Eigen::Quaterniond &r_quat, const Eigen::Vector3d &t_vec ) = 0;
  virtual double getAvgDistance( const cv::Mat_<double> &r_vec, const cv::Mat_<double> &t_vec ) = 0;
// TODO
//  virtual double getAvgDistance( const ObjectTemplate &templ, const cv::Mat_<double> &r_vec,
//                                 const cv::Mat_<double> &t_vec ) = 0;
//  double getAvgDistance( int idx );

  virtual double refinePosition( double r_quat[4], double t_vec[3] ) = 0;
  virtual double refinePosition( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec ) = 0;
  virtual double refinePosition( cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec ) = 0;
  virtual double refinePosition( const ObjectTemplate &templ, cv::Mat_<double> &r_vec,
                                 cv::Mat_<double> &t_vec ) = 0;

//  double refinePosition( int idx, double r_quat[4], double t_vec[3] );
//  double refinePosition( int idx, Eigen::Quaterniond& r_quat, Eigen::Vector3d& t_vec );
//  double refinePosition( int idx, cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec );
  
protected:

  virtual void updateOptimizer() = 0;
  virtual double optimize() = 0;
  virtual double avgDistance() = 0;
  
  void setPos( const double r_quat[4], const double t_vec[3] );
  void setPos( const Eigen::Quaterniond &r_quat, const Eigen::Vector3d &t_vec );
  void setPos( const cv::Mat_<double> &r_vec, const cv::Mat_<double> &t_vec );

  void getPos( double r_quat[4], double t_vec[3] ) const;
  void getPos( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec ) const;
  void getPos( cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec ) const;
  
  Eigen::Matrix< double, 8, 1> transf_;

  int num_optim_iterations_ = 100;
  bool ensure_cheirality_constraint_ = false;

  double verbose_mode_ = false;

private:

  ObjectRegistrationBase( const ObjectRegistrationBase &other );
  ObjectRegistrationBase& operator=( const ObjectRegistrationBase &other );
  
  bool update_optimizer_ = false;
  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  
};

class ObjectRegistration : public ObjectRegistrationBase
{
 public:

  virtual ~ObjectRegistration(){};

  void setObjectModel (const RasterObjectModelPtr& model_ptr );

  double getAvgDistance( const double r_quat[4], const double t_vec[3] ) override;
  double getAvgDistance( const Eigen::Quaterniond &r_quat, const Eigen::Vector3d &t_vec ) override;
  double getAvgDistance( const cv::Mat_<double> &r_vec, const cv::Mat_<double> &t_vec );
//  double getAvgDistance( int idx );

  double refinePosition( double r_quat[4], double t_vec[3] ) override;
  double refinePosition( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec ) override;
  double refinePosition( cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec ) override;

 protected:
  cv_ext::PinholeCameraModel cam_model_;
  RasterObjectModelPtr model_ptr_;

 private:
  bool update_optimizer_ = false;

};

class MultiViewObjectRegistration : public ObjectRegistrationBase
{
 public:

  virtual ~MultiViewObjectRegistration(){};

  void setObjectModels ( const std::vector< RasterObjectModel3DPtr > &model_ptrs,
                         const cv_ext::vector_Quaterniond &view_r_quats,
                         const cv_ext::vector_Vector3d &view_t_vec );

  double getAvgDistance( const double r_quat[4], const double t_vec[3] ) override;
  double getAvgDistance( const Eigen::Quaterniond &r_quat, const Eigen::Vector3d &t_vec ) override;
  double getAvgDistance( const cv::Mat_<double> &r_vec, const cv::Mat_<double> &t_vec );
//  double getAvgDistance( int idx );

  double refinePosition( double r_quat[4], double t_vec[3] ) override;
  double refinePosition( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec ) override;
  double refinePosition( cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec ) override;

 protected:
  std::vector< cv_ext::PinholeCameraModel > cam_models_;
  std::vector< RasterObjectModel3DPtr > model_ptrs_;
  cv_ext::vector_Quaterniond view_r_quats_;
  cv_ext::vector_Vector3d view_t_vec_;

 private:

  bool update_optimizer_ = false;
};