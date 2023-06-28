#pragma once

#include <string>
#include "cv_ext/cv_ext.h"
#include "raster_object_model3D.h"
#include "chamfer_matching.h"

using namespace std;
using namespace cv;
using namespace cv_ext;


struct ObjIstance
{
  ObjIstance( int id, double score, Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec ) : 
    id(id), score(score), r_quat(r_quat), t_vec(t_vec){};
  
  int id;
  double score;
  Eigen::Quaterniond r_quat;
  Eigen::Vector3d t_vec;
  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class AutomaticaLocalization
{
public:
  AutomaticaLocalization( const PinholeCameraModel cam_models[2], 
                          const Mat& stereo_r_mat, const Mat& stereo_t_vec,
                          cv::Rect roi[2] );
  
  void addObj( std::string model_filename, std::string template_filename[2], double threshold = 0.6 );
  void initialize();
  void localize( cv::Mat src_img[2], std::vector< ObjIstance > &found_obj );
//   void enableDebug( bool enabled ){ debug_enabled_ = };
private:

  void refinePosition( int i_obj, Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec, ImageTensorPtr dst_map_tensor_ptrs[2] );
 double evaluateScore ( cv::Mat img, vector<cv::Point2f> &raster_pts, const vector<float> &normal_directions );

  cv_ext::StereoRectification stereo_rect_;
  DistanceTransform dc_;
  PinholeCameraModel rect_cam_models_[2];
  cv::Rect roi_[2];
  Point2f stereo_disp_;
  std::vector < std::vector < RasterObjectModel3DPtr > > obj_model_ptrs_;
  std::vector < std::vector < TemplateSetPtr > > ts_ptrs_;
  
  std::vector < std::shared_ptr< DirectionalChamferMatching > > dcm_;
  std::vector < std::shared_ptr< MultiViewsDirectionalChamferMatching > > mdcm_;
  
  std::vector < double > thresholds_;
  
  const int num_directions = 60, increment = 4;
  const double tensor_lambda = 6.0;
  const bool smooth_tensor = false;  
  const int num_matches = 3;
  bool debug_enabled_ = true;
  int stereo_bb_offset_ = 0;
  Mat img_pair_, display_, rect_img_[2], h_display_[2];
  Mat score_mask_;
};
