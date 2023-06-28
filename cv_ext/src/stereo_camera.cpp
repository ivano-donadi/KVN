#include "cv_ext/stereo_camera.h"
#include "cv_ext/macros.h"

#include <stdexcept>

using namespace cv;
using namespace std;
using namespace cv_ext;

void StereoRectification::setCameraParameters ( const std::vector < PinholeCameraModel > &cam_models,
                                                const Mat& r_mat, const Mat& t_vec )
{
  cv_ext_assert( cam_models.size() == 2 );
  cv_ext_assert( cam_models[0].imgSize() == cam_models[1].imgSize() );
  cv_ext_assert( r_mat.rows == 3 && r_mat.cols == 3 && t_vec.rows == 3 && t_vec.cols == 1 );

  for ( int k = 0; k < 2; k++ )
  {
    camera_matrices_[k] = cam_models[k].cameraMatrix();
    dist_coeffs_[k] = cam_models[k].distorsionCoeff();
  }
  if ( cam_models[0].imgSize() != cam_models[1].imgSize() )
    return;

  image_size_ = cam_models[0].imgSize();

  r_mat_ = r_mat;
  t_vec_ = t_vec;
}

void StereoRectification::setImageScaleFacor ( double scale_factor )
{
  scale_factor_ = scale_factor;
}

void StereoRectification::setScalingParameter ( double val )
{
  scaling_param_ = val;
  if( scaling_param_ < 0 || scaling_param_ > 1.0 )
    scaling_param_ = 0;
}

void StereoRectification::update()
{
  Mat r_mat1, r_mat2, inv_proj_mat1, inv_proj_mat2;
  float scale = 1.0/scale_factor_; 
  out_image_size_ = Size( round(scale*image_size_.width), 
                          round(scale*image_size_.height));
  
  Mat r_vec, g_mat, inv_g_mat, inv_r_vec, inv_r_mat, inv_t_vec;
  rotMat2AngleAxis<double>(r_mat_, r_vec);
  exp2TransfMat<double>(r_vec, t_vec_, g_mat);
  cv::invert(g_mat, inv_g_mat);
  transfMat2Exp<double>(inv_g_mat,inv_r_vec, inv_t_vec);
  angleAxis2RotMat<double>(inv_r_vec, inv_r_mat);
  
  cv::stereoRectify(camera_matrices_[1], dist_coeffs_[1], camera_matrices_[0], dist_coeffs_[0],
                    image_size_, inv_r_mat, inv_t_vec, r_mat1, r_mat2, inv_proj_mat1, inv_proj_mat2, inv_disp2depth_mat_,
                    zero_disp_?cv::CALIB_ZERO_DISPARITY:0, scaling_param_,
                    out_image_size_ );

  cv::stereoRectify(camera_matrices_[0], dist_coeffs_[0], camera_matrices_[1], dist_coeffs_[1],
                    image_size_, r_mat_, t_vec_, r_mat1, r_mat2, proj_mat_[0], proj_mat_[1], disp2depth_mat_,
                    zero_disp_?cv::CALIB_ZERO_DISPARITY:0, scaling_param_,
                    out_image_size_, &rect_roi_[0], &rect_roi_[1]);
  
  //Precompute maps for cv::remap()
  initUndistortRectifyMap( camera_matrices_[0], dist_coeffs_[0], r_mat1, proj_mat_[0], out_image_size_, CV_16SC2, 
                           rect_map_[0][0], rect_map_[0][1]);
  initUndistortRectifyMap( camera_matrices_[1], dist_coeffs_[1], r_mat2, proj_mat_[1], out_image_size_, CV_16SC2, 
                           rect_map_[1][0], rect_map_[1][1]);
}

std::vector <cv::Rect> StereoRectification::getRegionsOfInterest ()
{
  std::vector <cv::Rect> roi(2);
  for ( int k = 0; k < 2; k++ )
    roi[k] = rect_roi_[k];
  return roi;
}

void StereoRectification::rectifyImagePair ( const std::vector< cv::Mat > &imgs, std::vector< cv::Mat > &rect_imgs )
{
  cv_ext_assert( imgs.size() == 2 );

  if( rect_imgs.size() < 2 )
    rect_imgs.resize(2);

  if( rect_map_[0][0].empty() || rect_map_[0][1].empty() || 
      rect_map_[1][0].empty() || rect_map_[1][1].empty() )
  {
    rect_imgs[0] = rect_imgs[1] = Mat();
    return;
  }
  
  for( int k = 0; k < 2; k++ )
    cv::remap(imgs[k], rect_imgs[k], rect_map_[k][0], rect_map_[k][1], INTER_LINEAR);
}

std::vector< PinholeCameraModel > StereoRectification::getCamModels()
{
  std::vector< PinholeCameraModel > cam_models(2);
  if ( !proj_mat_[0].empty() && !proj_mat_[1].empty() )
  {
    for ( int k = 0; k < 2; k++ )
      cam_models[k] = PinholeCameraModel ( proj_mat_[k].colRange(0,3), 
                                           out_image_size_.width, out_image_size_.height );
  }
  else
  {
    for ( int k = 0; k < 2; k++ )
      cam_models[k] = PinholeCameraModel();
  }
  return cam_models;
}

Point2f StereoRectification::getCamDisplacement()
{
  Point2f cam_shift = Point2f(0,0);
  if ( !proj_mat_[0].empty() && !proj_mat_[1].empty() )
  {
    if( proj_mat_[1].at<double>(0,3) )
      cam_shift.x = proj_mat_[1].at<double>(0,3)/proj_mat_[1].at<double>(0,0);
    if( proj_mat_[1].at<double>(1,3) )
      cam_shift.y = proj_mat_[1].at<double>(1,3)/proj_mat_[1].at<double>(0,0);      
  }
  return cam_shift;
}
