#pragma once

#include "cv_ext/pinhole_camera_model.h"
#include "Eigen/Core"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
namespace cv_ext
{

struct CalibReprojectionError
{
  CalibReprojectionError( const PinholeCameraModel &cam_model, const Eigen::Vector3d &pattern_pt,
                          const Eigen::Vector2d &observed_pt ) :
      cam_model(cam_model),
      pattern_pt(pattern_pt),
      observed_pt( observed_pt) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const pattern,
                  T* residuals) const
  {
    T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
      ref_cam_pt[3], cam_pt[3];

    // pattern[0,1,2] is the angle-axis rotation of the pattern respect to the reference camera.
    ceres::AngleAxisRotatePoint(pattern, ptn_pt, ref_cam_pt);

    // pattern[3,4,5] is the translation of the pattern respect to the reference camera.
    ref_cam_pt[0] += pattern[3];
    ref_cam_pt[1] += pattern[4];
    ref_cam_pt[2] += pattern[5];

    // camera[0,1,2] is the angle-axis rotation of the reference camera respect to the current camera.
    ceres::AngleAxisRotatePoint(camera, ref_cam_pt, cam_pt);

    // camera[3,4,5] is the translation of the reference camera respect to the current camera.
    cam_pt[0] += camera[3];
    cam_pt[1] += camera[4];
    cam_pt[2] += camera[5];

    // Projection.
    T proj_pt[2];
    cam_model.project(cam_pt, proj_pt);

    // The error is the difference between the predicted and observed position.
    residuals[0] = proj_pt[0] - T(observed_pt(0));
    residuals[1] = proj_pt[1] - T(observed_pt(1));

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create( const PinholeCameraModel &cam_model,
                                      const Eigen::Vector3d &pattern_pt,
                                      const Eigen::Vector2d &observed_pt )
  {
    return (new ceres::AutoDiffCostFunction<CalibReprojectionError, 2, 6, 6>(
        new CalibReprojectionError( cam_model, pattern_pt, observed_pt )));
  }

  const PinholeCameraModel &cam_model;
  Eigen::Vector3d pattern_pt;
  Eigen::Vector2d observed_pt;
};

struct CalibReferenceCalibReprojectionError
{
  CalibReferenceCalibReprojectionError( const PinholeCameraModel &cam_model, const Eigen::Vector3d &pattern_pt,
                                        const Eigen::Vector2d &observed_pt ) :
      cam_model(cam_model),
      pattern_pt(pattern_pt),
      observed_pt( observed_pt) {}

  template <typename T>
  bool operator()(const T* const pattern,
                  T* residuals) const
  {
    T ptn_pt[3] = {T(pattern_pt(0)), T(pattern_pt(1)), T(pattern_pt(2))},
        ref_cam_pt[3];

    // pattern[0,1,2] is the angle-axis rotation of the pattern respect to the reference camera.
    ceres::AngleAxisRotatePoint(pattern, ptn_pt, ref_cam_pt);

    // pattern[3,4,5] is the translation of the pattern respect to the reference camera.
    ref_cam_pt[0] += pattern[3];
    ref_cam_pt[1] += pattern[4];
    ref_cam_pt[2] += pattern[5];

    // Projection.
    T proj_pt[2];
    cam_model.project(ref_cam_pt, proj_pt);

    // The error is the difference between the predicted and observed position.
    residuals[0] = proj_pt[0] - T(observed_pt(0));
    residuals[1] = proj_pt[1] - T(observed_pt(1));

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create( const PinholeCameraModel &cam_model,
                                      const Eigen::Vector3d &pattern_pt,
                                      const Eigen::Vector2d &observed_pt )
  {
    return (new ceres::AutoDiffCostFunction<CalibReferenceCalibReprojectionError, 2, 6>(
        new CalibReferenceCalibReprojectionError( cam_model, pattern_pt, observed_pt )));
  }

  const PinholeCameraModel &cam_model;
  Eigen::Vector3d pattern_pt;
  Eigen::Vector2d observed_pt;
};

}