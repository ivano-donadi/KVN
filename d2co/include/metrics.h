#pragma once

#include "cv_ext/cv_ext.h"

class EvaluationMetricBase
{
 public:

  virtual void setGroundTruth( Eigen::Quaterniond &gt_r_quat, Eigen::Vector3d &gt_t_vec ) = 0;
  virtual bool performTest( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec ) = 0;
};

class MaxRtErrorsMetric : public EvaluationMetricBase
{
 public:

  MaxRtErrorsMetric( double max_rot_err_deg = 5., double max_t_err = .05 );

  void setGroundTruth( Eigen::Quaterniond &gt_r_quat, Eigen::Vector3d &gt_t_vec ) override;
  bool performTest( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec ) override;

 private:

  double max_rot_err_deg_, max_t_err_;

  Eigen::Quaterniond gt_r_quat_;
  Eigen::Vector3d gt_t_vec_;
};

class Projection2DMetric : public EvaluationMetricBase
{
 public:

  Projection2DMetric( const cv_ext::PinholeCameraModel &cam_model, const std::vector<cv::Point3f> &model_vtx,
                      bool symmetric_object = false, double max_proj_err_pix = 5.  );

  void setGroundTruth( Eigen::Quaterniond &gt_r_quat, Eigen::Vector3d &gt_t_vec ) override;
  bool performTest( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec  ) override;

 private:

  cv_ext::PinholeSceneProjector cam_proj_;
  std::vector<cv::Point3f> model_vtx_;
  bool symmetric_object_;
  double max_proj_err_pix_;

  std::vector<cv::Point2f> gt_proj_vtx_;
};

class Pose6DMetric : public EvaluationMetricBase
{
 public:

  Pose6DMetric(const cv_ext::PinholeCameraModel &cam_model, const std::vector<cv::Point3f> &model_vtx,
               double obj_diameter, bool symmetric_object = false, double max_err_percentage = .1 );

  void setGroundTruth( Eigen::Quaterniond &gt_r_quat, Eigen::Vector3d &gt_t_vec ) override;
  bool performTest( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec ) override;

 private:

  cv_ext::PinholeSceneProjector cam_proj_;
  std::vector<cv::Point3f> model_vtx_;
  double obj_diameter_;
  bool symmetric_object_;
  double max_err_percentage_;

  std::vector<cv::Point3f> gt_vtx_;
};
