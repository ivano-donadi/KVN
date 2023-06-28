#include "cv_ext/pinhole_camera_model.h"
#include "cv_ext/serialization.h"

#include <fstream>

using namespace std;
using namespace cv;
using namespace cv_ext;

PinholeCameraModel::PinholeCameraModel( const Mat &camera_matrix, int img_width, int img_height,
                                        const Mat &dist_coeff ) :
  orig_img_width_(img_width), 
  orig_img_height_(img_height)
{
  cv_ext_assert(img_width > 0 && img_height > 0 );
  cv_ext_assert( camera_matrix.channels() == 1 );
  cv_ext_assert( camera_matrix.rows == 3 && camera_matrix.cols == 3 );
  cv_ext_assert( camera_matrix.rows == 3 && camera_matrix.cols == 3 );
  cv_ext_assert( dist_coeff.empty() || ( dist_coeff.channels() == 1 &&
                 ( ( dist_coeff.rows > 3 && dist_coeff.cols == 1 ) ||
                   ( dist_coeff.rows == 1 && dist_coeff.cols > 3 ) ) ) );

  Mat_<double>tmp_cam_mat(camera_matrix);
  
  orig_fx_ = tmp_cam_mat(0,0); 
  orig_fy_ = tmp_cam_mat(1,1);
  orig_cx_ = tmp_cam_mat(0,2); 
  orig_cy_ = tmp_cam_mat(1,2);

  dist_px_ = dist_py_ = dist_k0_ = dist_k1_ = dist_k2_ = dist_k3_ = dist_k4_ = dist_k5_ = 0;

  if( !dist_coeff.empty() )
  {
    Mat_<double>tmp_dist_coeff(dist_coeff);
    if( tmp_dist_coeff.rows < tmp_dist_coeff.cols )
      cv::transpose(tmp_dist_coeff, tmp_dist_coeff);

    dist_k0_ = tmp_dist_coeff(0,0);
    dist_k1_ = tmp_dist_coeff(1,0); 
    dist_px_ = tmp_dist_coeff(2,0);
    dist_py_ = tmp_dist_coeff(3,0);
    
    if( tmp_dist_coeff.rows > 4 )
      dist_k2_ = tmp_dist_coeff(4,0);
    if( tmp_dist_coeff.rows > 5 )
      dist_k3_ = tmp_dist_coeff(5,0);
    if( tmp_dist_coeff.rows > 6 )
      dist_k4_ = tmp_dist_coeff(6,0);
    if( tmp_dist_coeff.rows > 7 )
      dist_k5_ = tmp_dist_coeff(7,0);      
  }
  
  reset();
}

PinholeCameraModel::PinholeCameraModel( double fx, double fy, double cx, double cy, int img_width, int img_height,
                                        double dist_k0, double dist_k1, double dist_px, double dist_py,
                                        double dist_k2 , double dist_k3, double dist_k4, double dist_k5 ) :
  orig_img_width_(img_width),
  orig_img_height_(img_height),
  orig_fx_(fx),
  orig_fy_(fy),
  orig_cx_(cx),
  orig_cy_(cy),
  dist_px_(dist_px),
  dist_py_(dist_py),
  dist_k0_(dist_k0),
  dist_k1_(dist_k1),
  dist_k2_(dist_k2),
  dist_k3_(dist_k3),
  dist_k4_(dist_k4),
  dist_k5_(dist_k5)
{
  cv_ext_assert(img_width > 0 && img_height > 0 );

  reset();
}

void PinholeCameraModel::read(const YAML::Node &input)
{
  orig_img_width_ = input["img_width"].as<int>();
  orig_img_height_ = input["img_height"].as<int>();

  orig_fx_ = input["fx"].as<double>();
  orig_fy_ = input["fy"].as<double>();
  orig_cx_ = input["cx"].as<double>();
  orig_cy_ = input["cy"].as<double>();

  dist_px_ = dist_py_ = dist_k0_ = dist_k1_ = dist_k2_ = dist_k3_ = dist_k4_ = dist_k5_ = 0;

  dist_px_ = input["dist_px"].as<double>();
  dist_py_ = input["dist_py"].as<double>();
  dist_k0_ = input["dist_k0"].as<double>();
  dist_k1_ = input["dist_k1"].as<double>();
  dist_k2_ = input["dist_k2"].as<double>();
  dist_k3_ = input["dist_k3"].as<double>();
  dist_k4_ = input["dist_k4"].as<double>();
  dist_k5_ = input["dist_k5"].as<double>();

  // TODO Check here!
  reset();
}

bool PinholeCameraModel::readFromFile ( const string &filename )
{
  string yml_filename = generateYAMLFilename(filename);

  try
  {
    YAML::Node in_node = YAML::LoadFile(yml_filename);
    read(in_node);
  }
  catch(std::exception &e)
  {
    std::cerr << "PinholeCameraModel::readFromFile() failed to load file" << std::endl;
    return false;
  }

  return true;
}

void PinholeCameraModel::write( YAML::Node &out ) const
{
  out["img_width"] = orig_img_width_;
  out["img_height"] = orig_img_height_;

  out["fx"] = orig_fx_;
  out["fy"] = orig_fy_;
  out["cx"] = orig_cx_;
  out["cy"] = orig_cy_;

  out["dist_px"] = dist_px_;
  out["dist_py"] = dist_py_;
  out["dist_k0"] = dist_k0_;
  out["dist_k1"] = dist_k1_;
  out["dist_k2"] = dist_k2_;
  out["dist_k3"] = dist_k3_;
  out["dist_k4"] = dist_k4_;
  out["dist_k5"] = dist_k5_;
}

bool PinholeCameraModel::writeToFile ( const string &filename ) const
{
  string yml_filename = generateYAMLFilename(filename);

  try
  {
    YAML::Node out_node;
    write(out_node);
    std::ofstream out(yml_filename);
    out << out_node;
    out.close();
  }
  catch(std::exception &e)
  {
    std::cerr << "PinholeCameraModel::writeToFile() failed to write file" << std::endl;
    return false;
  }

  return true;
}

void PinholeCameraModel::setSizeScaleFactor ( double scale_factor ) 
{
  cv_ext_assert( scale_factor > 0);

  size_scale_factor_ = scale_factor;
  updateCurrentParamters();
}

void PinholeCameraModel::setRegionOfInterest ( Rect& roi )
{
  cv_ext_assert( roi.x >= 0 && roi.y >= 0 && roi.x < orig_img_width_ && roi.y < orig_img_height_ );
  cv_ext_assert( roi.width <= orig_img_width_ - roi.x && roi.height <= orig_img_height_ - roi.y );

  orig_roi_ = roi;
  updateCurrentParamters();
}

void PinholeCameraModel::enableRegionOfInterest ( bool enable )
{
  roi_enabled_ = enable;
  updateCurrentParamters();
}

void PinholeCameraModel::updateCurrentParamters() 
{  
  fx_ = orig_fx_/size_scale_factor_; 
  fy_ = orig_fy_/size_scale_factor_;
  cx_ = orig_cx_/size_scale_factor_; 
  cy_ = orig_cy_/size_scale_factor_;
  
  inv_fx_ = 1.0/fx_;
  inv_fy_ = 1.0/fy_;
  
  roi_.x = orig_roi_.x/size_scale_factor_;
  roi_.y = orig_roi_.y/size_scale_factor_;
  roi_.width = orig_roi_.width/size_scale_factor_;
  roi_.height = orig_roi_.height/size_scale_factor_;
  
  if( roi_enabled_ )
  {
    cx_ -= roi_.x;
    cy_ -= roi_.y;
    img_width_ = roi_.width;
    img_height_ = roi_.height;
  }
  else
  {
    img_width_ = orig_img_width_/size_scale_factor_;
    img_height_ = orig_img_height_/size_scale_factor_;    
  }
}

void PinholeCameraModel::reset()
{
  size_scale_factor_ = 1.0;
  orig_roi_ = Rect(0,0,orig_img_width_,orig_img_height_);
  roi_enabled_ = false;
  term_epsilon_ = 1e-7;
  has_dist_coeff_ = dist_k0_ || dist_k1_ || dist_k2_ || dist_k3_ || dist_k4_ || dist_k5_ || dist_px_ || dist_py_;

  updateCurrentParamters();
}
