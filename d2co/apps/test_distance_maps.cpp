#include <iostream>
#include <string>
#include <map>
#include <boost/program_options.hpp>

#include "edge_detectors.h"
#include "distance_transforms.h"

using namespace std;
using namespace cv;
namespace po = boost::program_options;

template<typename _T> bool compareMat( Mat &m1, Mat &m2)
{
  if( m1.rows != m2.rows || m1.cols != m2.cols || m1.type() != m2.type() )
    return false;

  for( int r = 0; r < m1.rows; r++ )
  {
    _T *m1_p = m1.ptr<_T>(r),*m2_p = m2.ptr<_T>(r);
    for( int r = 0; r < m1.rows; r++, m1_p++, m2_p++ )
    {
      if( *m1_p != *m2_p )
        return false;
    }
  }
  return true;
}
int main(int argc, char **argv)
{
  string app_name( argv[0] ), model_filename, camera_filename, image_filename;

  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "model_filename,m", po::value<string > ( &model_filename )->required(),
    "DXF, STL or PLY model file" )
  ( "camera_filename,c", po::value<string > ( &camera_filename )->required(),
    "A YAML file that stores all the camera parameters (see the PinholeCameraModel object)" )
  ( "image_filename,i", po::value<string > ( &image_filename)->required(),
    "Input image filename" );

  po::variables_map vm;

  try
  {
    po::store ( po::parse_command_line ( argc, argv, desc ), vm );

    if ( vm.count ( "help" ) )
    {
      cout << "USAGE: "<<app_name<<" OPTIONS"
                << endl << endl<<desc;
      return 0;
    }

    po::notify ( vm );
  }

  catch ( boost::program_options::required_option& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    return -1;
  }

  catch ( boost::program_options::error& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    return -1;
  }

  cout << "Loading model from file : "<<model_filename<< endl;
  cout << "Loading camera parameters from file : "<<camera_filename<< endl;
  cout << "Loading input images from file : "<<image_filename<< endl;

  Mat r_vec = (Mat_<double>(3,1) << 0,0,0), t_vec = (Mat_<double>(3,1) << 0,0,0.5);

  cv_ext::PinholeCameraModel cam_model;
  cam_model.readFromFile(camera_filename);

  Mat src_img = imread(image_filename, cv::IMREAD_GRAYSCALE), edge_map, dir_edge_map;

  LSDEdgeDetector lsd_edge_detector;
  lsd_edge_detector.enableWhiteBackground(true);
  lsd_edge_detector.setNumEdgeDirections(60);
  lsd_edge_detector.setImage(src_img);
  lsd_edge_detector.getEdgeMap(edge_map);
  cv_ext::showImage(edge_map);

  lsd_edge_detector.enableWhiteBackground(false);

  for( int i = 0; i < lsd_edge_detector.numEdgeDirections(); i++ )
  {
    Mat dbg_img;
    lsd_edge_detector.getDirectionalEdgeMap(i, dir_edge_map);
    cvtColor(edge_map, dbg_img, cv::COLOR_GRAY2BGR );
    dbg_img.setTo(cv::Scalar(0,0,255),dir_edge_map );
    cv_ext::showImage(dir_edge_map);
    cv_ext::showImage(dbg_img);
  }

  CannyEdgeDetector canny_edge_detector;
  canny_edge_detector.enableWhiteBackground(true);
  canny_edge_detector.setNumEdgeDirections(60);
  canny_edge_detector.setImage(src_img);
  canny_edge_detector.getEdgeMap(edge_map);
  cv_ext::showImage(edge_map);

  canny_edge_detector.enableWhiteBackground(false);

  for( int i = 0; i < canny_edge_detector.numEdgeDirections(); i++ )
  {
    Mat dbg_img;
    canny_edge_detector.getDirectionalEdgeMap(i, dir_edge_map);
    cvtColor(edge_map, dbg_img, cv::COLOR_GRAY2BGR );
    dbg_img.setTo(cv::Scalar(0,0,255),dir_edge_map );
    cv_ext::showImage(dir_edge_map);
    cv_ext::showImage(dbg_img);
  }

//   cv_ext::BasicTimer timer;
//   Mat dist_map_old, dist_map, closest_edgels_map_old, closest_edgels_map;
//   ImageTensorPtr dist_map_tensor_ptr_old, x_dist_map_tensor_ptr_old,
//                  y_dist_map_tensor_ptr_old, edgels_map_tensor_ptr_old,
//                  dist_map_tensor_ptr, x_dist_map_tensor_ptr,
//                  y_dist_map_tensor_ptr, edgels_map_tensor_ptr;
//
//   DistanceTransform dc;
//   dc.enableParallelism(true);
//
// //   computeDistanceMap(src_img, dist_map_old);
//   dc.computeDistanceMap(src_img, dist_map);
//
//   if(!compareMat<float>(dist_map, dist_map_old))
//     cout<<"Test 1 Failed!"<<endl;
//
// //   computeDistanceMap(src_img, dist_map_old, closest_edgels_map_old);
//   dc.computeDistanceMap(src_img, dist_map, closest_edgels_map);
//
//   if(!compareMat<float>(dist_map, dist_map_old))
//     cout<<"Test 2 Failed!"<<endl;
//
//   if(!compareMat<Point2f>(closest_edgels_map, closest_edgels_map_old))
//     cout<<"Test 3 Failed!"<<endl;
//
//   timer.reset();
//   computeDistanceMapTensor( src_img, dist_map_tensor_ptr_old );
//   cout<<"Old computeDistanceMapTensor() "<<timer.elapsedTimeMs()<<endl;
//   timer.reset();
//   dc.computeDistanceMapTensor( src_img, dist_map_tensor_ptr );
//   cout<<"New computeDistanceMapTensor() "<<timer.elapsedTimeMs()<<endl;
//
//   for( int i = 0; i < 60; i++ )
//   {
//     if(!compareMat<float>(dist_map_tensor_ptr->at(i), dist_map_tensor_ptr_old->at(i)))
//       cout<<"Test 4 Failed!"<<endl;
//   }
//
//   timer.reset();
//   computeDistanceMapTensor( src_img, dist_map_tensor_ptr_old, edgels_map_tensor_ptr_old );
//   cout<<"Old computeDistanceMapTensor() "<<timer.elapsedTimeMs()<<endl;
//   timer.reset();
//   dc.computeDistanceMapTensor( src_img, dist_map_tensor_ptr, edgels_map_tensor_ptr );
//   cout<<"New computeDistanceMapTensor() "<<timer.elapsedTimeMs()<<endl;
//   for( int i = 0; i < 60; i++ )
//   {
//     if(!compareMat<float>(dist_map_tensor_ptr->at(i), dist_map_tensor_ptr_old->at(i)))
//       cout<<"Test 5 Failed!"<<endl;
//     if(!compareMat<Point2f>(edgels_map_tensor_ptr->at(i), edgels_map_tensor_ptr_old->at(i)))
//       cout<<"Test 6 Failed!"<<endl;
//   }
//
//   timer.reset();
//   computeDistanceMapTensor( src_img, dist_map_tensor_ptr_old, x_dist_map_tensor_ptr_old,
//                             y_dist_map_tensor_ptr_old, edgels_map_tensor_ptr_old );
//   cout<<"Old computeDistanceMapTensor() "<<timer.elapsedTimeMs()<<endl;
//   timer.reset();
//   dc.computeDistanceMapTensor( src_img, dist_map_tensor_ptr, x_dist_map_tensor_ptr,
//                                y_dist_map_tensor_ptr, edgels_map_tensor_ptr );
//   cout<<"New computeDistanceMapTensor() "<<timer.elapsedTimeMs()<<endl;
//   for( int i = 0; i < 60; i++ )
//   {
//     if(!compareMat<float>(dist_map_tensor_ptr->at(i), dist_map_tensor_ptr_old->at(i)))
//       cout<<"Test 7 Failed!"<<endl;
//     if(!compareMat<float>(x_dist_map_tensor_ptr->at(i), x_dist_map_tensor_ptr_old->at(i)))
//       cout<<"Test 8 Failed!"<<endl;
//     if(!compareMat<float>(y_dist_map_tensor_ptr->at(i), y_dist_map_tensor_ptr_old->at(i)))
//       cout<<"Test 9 Failed!"<<endl;
//     if(!compareMat<Point2f>(edgels_map_tensor_ptr->at(i), edgels_map_tensor_ptr_old->at(i)))
//       cout<<"Test 10 Failed!"<<endl;
//   }
//
//   cout<<"Test OK!"<<endl;

  return 0;
}
