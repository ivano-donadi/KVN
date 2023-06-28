#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <boost/program_options.hpp>

#include <opencv2/opencv.hpp>
#include <cv_ext/cv_ext.h>

#include "apps_utils.h"

using namespace std;
using namespace cv;
namespace po = boost::program_options;

void writeTextBlock( Mat &img, const string &text, const Point &pos, Scalar color = Scalar(0,255,0),
                     double font_size = 1.5 )
{
  putText(img, text, pos, cv::FONT_HERSHEY_PLAIN, font_size, Scalar(0,0,0), 2, 8);
  putText(img, text, pos, cv::FONT_HERSHEY_PLAIN, font_size, color, 1, 8);
}

int main(int argc, char **argv)
{

  string app_name( argv[0] ), imgs_folder_name[2], camera_filename[2],  output_filename;
  bool show_rectified = false, precomputed_intrinsics = true;
  float scale_factor = 1.0f;
  
  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "f0", po::value<string > ( &imgs_folder_name[0] )->required(),
    "Camera 0 input images folder path" )
  ( "f1", po::value<string > ( &imgs_folder_name[1] )->required(),
    "Camera 1 input images folder path" )
  ( "c0", po::value<string > ( &camera_filename[0] ),
    "Camera 0 model filename" )
  ( "c1", po::value<string > ( &camera_filename[1] ),
    "Camera 1 model filename" )
  ( "output_filename,o", po::value<string > ( &output_filename )->required(),
    "Output stereo extrinsics parameters basic filename" )
  ( "scale_factor,s", po::value<float> ( &scale_factor),
     "Optional scale factor used to scale the output images [Default: 1.0]"  )
  ( "show_rectified,r", "Show rectified pairs" );

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

    if( !vm.count ( "c0" ) || !vm.count ( "c1" ) )
      precomputed_intrinsics = false;
    
    if ( vm.count ( "show_rectified" ) )
      show_rectified = true;

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

  for( int k = 0; k < 2; k++ )
    cout << "Loading camera "<<k<<" images from folder : "<<imgs_folder_name[k]<< endl;
  if( precomputed_intrinsics )
    for( int k = 0; k < 2; k++ )
      cout << "Loading camera "<<k<<" model from file : "<<camera_filename[k]<< endl;
  else
  {
    cout << "Intrinsics will automatically computed"<< endl;
    cout << "WARNING : this is a bad idea, you should calibrate each camera using CameraCalibration"
         <<" and then calibrate the stereo pair using StereoCameraCalibration"<< endl;
  }
  cout << "Output images scale factor : "<<scale_factor<< endl;
  
  vector<string> filelist[2];
  if( !readFileNamesFromFolder ( imgs_folder_name[0], filelist[0] ) ||
      !readFileNamesFromFolder ( imgs_folder_name[1], filelist[1] ) )
  {
    cerr<<"Wrong or empty folders"<<endl;
    exit(EXIT_FAILURE);      
  }

  if( filelist[0].size() != filelist[1].size() )
  {
    cerr<<"Images folder should contain the same number of images, in the same order"<<endl;
    exit(EXIT_FAILURE);      
  }

  cv_ext::StereoCameraCalibration calib;

  Size board_size(8,6);
  cv::Mat pattern_mask  = cv_ext::getStandardPatternMask (board_size);
  pattern_mask.at<uchar>(0,0) = 1;
  pattern_mask.at<uchar>(0,6) = 1;
  calib.setBoardData(board_size, 0.04, pattern_mask);
  calib.setPyramidNumLevels(2);
  if( precomputed_intrinsics )
  {
    std::vector < cv_ext::PinholeCameraModel > cam_models(2);
    for( int k = 0; k < 2; k++ )
      cam_models[k].readFromFile(camera_filename[k]);
    calib.setCamModels(cam_models);
  }
  
  cout << "Loading images and extracting corners ..."<<endl;
  
  for( int i = 0; i < int(filelist[0].size()); i++ )
  {
    vector< string > fn;
    for( int k = 0; k < 2; k++ )
      fn.push_back( filelist[k][i] );
    
    calib.addImagePairFiles(fn);
  }
  
  cout << "...done"<<endl;
  
  if( !calib.numCheckerboards() )
  {
    cerr<<"Unable to load images or to extract corners, exiting"<<endl;
    exit(EXIT_FAILURE);
  }
  
  bool update_img = true, time_to_exit = false;
  int pair_idx = 0;
  cv_ext::BasicTimer timer;
  Point text_org_guide_01 (20, 20), text_org_guide_11(20, 40), 
        text_org_info_01 (20, 60), text_org_info_11 (20, 80);
  
  string quick_guide_01("N : next pair;   P : prev pair;   [SPACE]: active/deactivate pair;"),
         quick_guide_11("C : perform calibration;   S : save parameters;   [ESC]: exit.");

  double epi_error = numeric_limits<double>::infinity();
  
  float scale = 1.0f/scale_factor;
  Size img_size = calib.imagesSize();
  // Pre-alloc all the display images (just to avoid to reallocate at each iteration )
  Mat corners_img_pair(Size(2*round(scale*img_size.width), round(scale*img_size.height)), cv::DataType<Vec3b>::type ),
      corner_dist_pair(Size(2*round(scale*img_size.width), round(scale*img_size.height)), cv::DataType<float>::type ), 
      corner_dist_rgb_pair(Size(2*round(scale*img_size.width), round(scale*img_size.height)), cv::DataType<Vec3b>::type );
  vector < Mat > corners_img(2), corner_dist(2), corner_dist_rgb(2);
  int start_col = 0, w = round(scale*img_size.width);
  for( int k = 0; k < 2; k++, start_col += w )
  {
    corners_img[k] = corners_img_pair.colRange(start_col, start_col + w);
    corner_dist[k] = corner_dist_pair.colRange(start_col, start_col + w);
    corner_dist_rgb[k] = corner_dist_rgb_pair.colRange(start_col, start_col + w);
  }
  
  Mat r_mat, t_vec;
  while ( !time_to_exit )
  {
    if( update_img )
    {
      calib.getCornersImagePair(pair_idx, corners_img, corners_img[0].size());
      stringstream info_01, info_11;
      info_01<<"Pair "<<pair_idx<<"/"<<calib.numCheckerboards()<<"; epipolar error : "
             <<calib.getEpipolarError(pair_idx);
      
      if( !calib.isCheckerboardActive(pair_idx) )
      {
        info_01<<" - INACTIVE";
        corners_img_pair *= 0.1;
      }
      
      info_11<<"# used for calibartion : "<<calib.numActiveCheckerboards()
             <<"; global epipolar error : "<<epi_error;

      writeTextBlock(corners_img_pair, quick_guide_01, text_org_guide_01);
      writeTextBlock(corners_img_pair, quick_guide_11, text_org_guide_11);
      writeTextBlock(corners_img_pair, info_01.str(), text_org_info_01, Scalar(0,0,255));
      writeTextBlock(corners_img_pair, info_11.str(), text_org_info_11, Scalar(0,0,255));
      imshow("Current corners", corners_img_pair);

      calib.getCornersDistribution( 30, corner_dist, corner_dist[0].size());
      for( int k = 0; k < 2; k++ )
      {
        double min_val, max_val;
        cv::minMaxLoc(corner_dist[k], &min_val, &max_val );
        cv_ext::mapMat2RGB<float>(corner_dist[k], corner_dist_rgb[k], max_val );
      }
      imshow("Global corners distribution", corner_dist_rgb_pair);
      update_img = false;
    }
    
    switch( cv_ext::waitKeyboard() )
    {
      case 'n':
      case 'N':
        if( ++pair_idx >= calib.numCheckerboards() )
          pair_idx = 0;
        update_img = true;
        break;
      
      case 'p':
      case 'P':
        if( --pair_idx < 0 ) 
          pair_idx = calib.numCheckerboards() - 1;
        update_img = true;
        break;
      case ' ':
        calib.setCheckerboardActive( pair_idx, !calib.isCheckerboardActive(pair_idx) );
        update_img = true;
        break;
      case 'c':
      case 'C':
        cout << endl<< "Calibrating ..." << endl;
        timer.reset();
        cout << "Average epipolar error :" << (epi_error = calib.calibrate()) << endl;
        cout << "Time elapsed: " << timer.elapsedTimeMs() << endl;
        calib.getExtrinsicsParameters(r_mat, t_vec);
        cout << "Estimated extrinsics : "<<endl
             <<r_mat<<endl<<t_vec<<endl;
        
        update_img = true;
       break;

      case 's':
      case 'S':
         if( r_mat.empty() || t_vec.empty() )
           cout<<"Error saving extrinsics parameter: you should first run the calibration"<<endl;
         else
         {
           cv_ext::write3DTransf(output_filename, r_mat, t_vec);
           cout<<"Saving extrinsics parameter to "<<output_filename<<endl;
         }
         break;
      case 27:
        time_to_exit = true;
        break;
    }
  }
  
  if ( show_rectified && epi_error != numeric_limits<double>::infinity() )
  {    
    cv::destroyWindow("Current corners");
    cv::destroyWindow("Global corners distribution");
   
    Mat rect_img_pair(Size(2*round(scale*img_size.width), round(scale*img_size.height)), calib.imagesType() );
    vector < Mat > rect_img(2);
    
    int start_col = 0, w = round(scale*img_size.width);
    for( int k = 0; k < 2; k++, start_col += w )
      rect_img[k] = rect_img_pair.colRange(start_col, start_col + w);
    
    quick_guide_01 = string("N : next pair;   P : prev pair;   [ESC]: exit.");
    pair_idx = 0;
    update_img = true, time_to_exit = false;
    
    while ( !time_to_exit )
    {
      if(update_img)
      {
        Mat und_img;
        calib.getRectifiedImagePair(pair_idx,rect_img,rect_img[0].size());
        for( int j = 0; j < rect_img_pair.rows; j += 16 )
          line(rect_img_pair, Point(0, j), Point(rect_img_pair.cols, j), Scalar(0, 255, 0), 1, 8);
        writeTextBlock(rect_img_pair, quick_guide_01, text_org_guide_01);
        imshow("Rectified image", rect_img_pair );
      }
      
      switch( cv_ext::waitKeyboard() )
      {
       case 'n':
       case 'N':
         if( ++pair_idx >= calib.numCheckerboards() )
           pair_idx = 0;
         update_img = true;
         break;
      
        case 'p':
        case 'P':
          if( --pair_idx < 0 ) 
            pair_idx = calib.numCheckerboards() - 1;
          update_img = true;
          break;
        case 27:
          time_to_exit = true;
          break;
      }
    }
  }
  
  return 0;
}
