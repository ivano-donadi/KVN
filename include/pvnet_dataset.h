#pragma once

#include <string>
#include <Eigen/Dense>
#include <boost/filesystem.hpp>

#include "cv_ext/cv_ext.h"

class PVNetDataset
{
 public:

  void init( const std::string &base_dir, bool stereo );
  void setModel( const std::string &model_filename, int obj_id, double diameter, double scale = 1.0 );
  void setCamera( const cv_ext::PinholeCameraModel &cam_model );
  bool addView( const Eigen::Matrix3d &r_mat, const Eigen::Vector3d &t_vec,
                const cv::Mat &img,  const cv::Mat &mask, const std::string suffix = "");
  bool addViewL(const Eigen::Matrix3d &r_mat, const Eigen::Vector3d &t_vec,
                const cv::Mat &img,  const cv::Mat &mask);
  bool addViewR(const Eigen::Matrix3d &r_mat, const Eigen::Vector3d &t_vec,
                const cv::Mat &img,  const cv::Mat &mask);

  void prepare_for_second_stereo_image();

  int numViews(){ return  num_views_; };
  cv::Mat getViewImage( int idx );

 private:

  bool initialized_ = false;
  bool stereo_ = false;
  int num_views_ = 0;
  int obj_id_;
  boost::filesystem::path base_path_, imgs_path_, masks_path_,  poses_path_, bb_path_;
};

void loadPVNetPose( const std::string &filename, Eigen::Matrix3d &r_mat, Eigen::Vector3d &t_vec);