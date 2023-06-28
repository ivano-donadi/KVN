#include "io_utils.h"

#include <boost/filesystem.hpp>
#include <boost/iterator/iterator_concepts.hpp>
#include <boost/tokenizer.hpp>
#include <boost/range/iterator_range.hpp>

using namespace boost;
using namespace boost::filesystem;
using namespace std;

void writeCommonCameraParameters(cv::FileStorage& fs, const cv::Size &image_size, const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs) {
  fs << "width" << image_size.width;
  fs << "height" << image_size.height;
  fs << "K" << camera_matrix;
  fs << "D" << dist_coeffs;
}

bool readFileNamesFromFolder ( const string& input_folder_name, vector< string >& names )
{
  names.clear();
  if ( !input_folder_name.empty() )
  {
    path p ( input_folder_name );
    for ( auto& entry : make_iterator_range ( directory_iterator ( p ), {} ) )
      names.push_back ( entry.path().string() );
    std::sort ( names.begin(), names.end() );
    return true;
  }
  else
  {
    return false;
  }
}

void loadCommonCameraParameters(cv::FileStorage& fs, cv::Size &image_size, cv::Mat &camera_matrix, cv::Mat &dist_coeffs){
  fs["width"]>>image_size.width;
  fs["height"]>>image_size.height;

  fs["K"]>>camera_matrix;
  fs["D"]>>dist_coeffs;
}

bool loadCameraParams( const std::string &file_name, cv::Size &image_size,
                       cv::Mat &camera_matrix, cv::Mat &dist_coeffs )
{
  cv::FileStorage fs(file_name, cv::FileStorage::READ);

  if( !fs.isOpened() )
    return false;

  loadCommonCameraParameters(fs, image_size, camera_matrix, dist_coeffs);

  fs.release();

  return true;
}

bool loadStereoCameraParams( const std::string &file_name, cv::Size &image_size, cv::Mat &camera_matrix, cv::Mat &dist_coeffs, double& baseline)
{
  cv::FileStorage fs(file_name, cv::FileStorage::READ);

  if( !fs.isOpened() )
    return false;

  loadCommonCameraParameters(fs, image_size, camera_matrix, dist_coeffs);
  fs["b"]>>baseline;

  fs.release();

  return true;
}

void saveCameraParams( const std::string &file_name, const cv::Size &image_size,
                       const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs )
{
  cv::FileStorage fs(file_name, cv::FileStorage::WRITE);

  writeCommonCameraParameters(fs, image_size, camera_matrix, dist_coeffs);

  fs.release();
}

void saveStereoCameraParams( const std::string &file_name, const cv::Size &image_size,
                       const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, const double baseline )
{
  cv::FileStorage fs(file_name, cv::FileStorage::WRITE);

  writeCommonCameraParameters(fs, image_size, camera_matrix, dist_coeffs);

  fs << "b" << baseline;

  fs.release();
}