#include "pvnet_dataset.h"

#include "cv_ext/cv_ext.h"
#include "cnpy.h"
#include "mesh_converter.h"

#include <iostream>
#include <fstream>

using namespace boost;

cv::Rect_<double> getYoloBB( const cv::Mat &mask )
{
  auto bb = cv::boundingRect	( mask );
  cv::Rect_<double> yolo_bb;

  yolo_bb.x = (static_cast<double>(bb.x) + bb.width/2)/mask.cols;
  yolo_bb.y = (static_cast<double>(bb.y) + bb.height/2)/mask.rows;
  yolo_bb.width = static_cast<double>(bb.width)/mask.cols;
  yolo_bb.height = static_cast<double>(bb.height)/mask.rows;

  return  yolo_bb;
}
void PVNetDataset::init( const std::string &base_dir, bool stereo )
{
  initialized_ = false;
  num_views_ = 0;
  stereo_ = stereo;

  base_path_ = base_dir;
  if ( !filesystem::exists(base_path_) )
    filesystem::create_directory(base_path_);

  imgs_path_ = base_path_/"rgb";
  masks_path_ = base_path_/"mask";
  poses_path_ = base_path_/"pose";
  bb_path_ = base_path_/"bbox";

  if (filesystem::exists(imgs_path_))
    filesystem::remove_all(imgs_path_);

  if (filesystem::exists(masks_path_))
    filesystem::remove_all(masks_path_);

  if (filesystem::exists(poses_path_))
    filesystem::remove_all(poses_path_);

  if (filesystem::exists(bb_path_))
    filesystem::remove_all(bb_path_);

  filesystem::create_directory(imgs_path_);
  filesystem::create_directory(masks_path_);
  filesystem::create_directory(poses_path_);
  filesystem::create_directory(bb_path_);

  initialized_ = true;
}

void PVNetDataset::setCamera(const cv_ext::PinholeCameraModel &cam_model)
{
  if( !initialized_ )
    return;

  filesystem::path cam_path(base_path_);
  cam_path /= "camera.txt";

  cv::Mat_<double> cam_mat = cam_model.cameraMatrix();
  std::fstream fs;
  fs.open( cam_path.string(), std::fstream::out );
  for( int r = 0; r < 3; r++ )
  {
    for( int c = 0; c < 3; c++ )
    {
      fs << cam_mat(r,c)<<" ";
    }
    fs << std::endl;
  }
  fs.close();
}

void PVNetDataset::setModel(const std::string &model_filename, int obj_id, double diameter, double scale )
{
  if( !initialized_ )
    return;

  filesystem::path diam_path(base_path_);
  diam_path /= "diameter.txt";

  obj_id_ = obj_id;

  std::fstream fs;
  fs.open( diam_path.string(), std::fstream::out );
  fs << diameter;
  fs.close();

  filesystem::path in_model_path(model_filename);
  if ( filesystem::exists(in_model_path) )
  {
    filesystem::path out_model_path(base_path_);
    out_model_path /= "model.ply";

    MeshConverter mc;
    mc.enforceASCIIOputput();
    mc.setScale(scale);
    mc.convert(in_model_path.string(), out_model_path.string());
  }
}

bool PVNetDataset::addViewL( const Eigen::Matrix3d &r_mat, const Eigen::Vector3d &t_vec, const cv::Mat &img, const cv::Mat &mask){
  return addView(r_mat, t_vec, img, mask, "_L");
}

bool PVNetDataset::addViewR( const Eigen::Matrix3d &r_mat, const Eigen::Vector3d &t_vec, const cv::Mat &img, const cv::Mat &mask){
  return addView(r_mat, t_vec, img, mask, "_R");
}

void PVNetDataset::prepare_for_second_stereo_image(){
  num_views_--;
}

bool PVNetDataset::addView( const Eigen::Matrix3d &r_mat, const Eigen::Vector3d &t_vec,
                            const cv::Mat &img, const cv::Mat &mask, const std::string suffix )
{
  cv_ext_assert( !img.empty() && !mask.empty() );

  if( !initialized_ )
    return false;

  if(!cv::countNonZero(mask))
    return false;

  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> tf_mat;
  tf_mat.block<3,3>(0,0) = r_mat;
  tf_mat.block<3,1>(0,3) = t_vec;

  filesystem::path pose_path(poses_path_);
  pose_path /= "pose" + std::to_string(num_views_) + suffix + ".npy";
  cnpy::npy_save( pose_path.string(), tf_mat.data(), {3,4} );

  filesystem::path img_path(imgs_path_);
  img_path /= std::to_string(num_views_) + suffix + ".jpg";
  cv::imwrite( img_path.string(), img );

  filesystem::path mask_path(masks_path_);
  mask_path /= std::to_string(num_views_) + suffix + ".png";
  cv::imwrite( mask_path.string(), mask );

  filesystem::path bb_path(bb_path_);
  bb_path /= std::to_string(num_views_) + suffix +".txt";
  auto yolo_bb = getYoloBB( mask );
  std::ofstream bb_file( bb_path.string() );
  bb_file<< obj_id_<<" "<<yolo_bb.x<<" "<<yolo_bb.y<<" "<<yolo_bb.width<<" "<<yolo_bb.height<<std::endl;

  num_views_++;

  return true;
}

cv::Mat PVNetDataset::getViewImage(int idx)
{
  if( idx < num_views_ )
  {
    filesystem::path img_path(imgs_path_);
    img_path /= std::to_string(idx) + ".jpg";
    return cv::imread( img_path.string() );
  }
  else
  {
    return cv::Mat();
  }
}

void loadPVNetPose( const std::string &filename, Eigen::Matrix3d &r_mat, Eigen::Vector3d &t_vec)
{
  cnpy::NpyArray npy_array = cnpy::npy_load(filename);
  Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor> > tf_mat(npy_array.data<double>());
  r_mat = tf_mat.block<3,3>(0,0);
  t_vec = tf_mat.block<3,1>(0,3);
}