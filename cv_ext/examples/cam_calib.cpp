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

  string app_name( argv[0] ), imgs_folder_name, calibration_filename;
  bool show_undistorted = false;
  float scale_factor = 1.0f;

  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "images_folder,f", po::value<string > ( &imgs_folder_name )->required(),
    "Input images folder path" )
  ( "calibration_filename,c", po::value<string > ( &calibration_filename )->required(),
    "Output calibration basic filename" )
  ( "scale_factor,s", po::value<float> ( &scale_factor),
    "Optional scale factor used to scale the output images [Default: 1.0]" )
  ( "show_undistorted,u", "Show undistorted images" );

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

    if ( vm.count ( "show_undistorted" ) )
      show_undistorted = true;

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

  cout << "Loading images from folder : "<<imgs_folder_name<< endl;
  cout << "Saving camera parameters to file : "<<calibration_filename<< endl;
  cout << "Output images scale factor : "<<scale_factor<< endl;
  
  std::vector<std::string> filelist;
  if( !readFileNamesFromFolder ( imgs_folder_name, filelist ) )
  {
    std::cerr<<"Wrong or empty folder"<<endl;
    exit(EXIT_FAILURE);      
  }

  cv_ext::CameraCalibration calib;

  Size board_size(8,6);
  cv::Mat pattern_mask  = cv_ext::getStandardPatternMask (board_size);
  pattern_mask.at<uchar>(0,0) = 1;
  pattern_mask.at<uchar>(0,6) = 1;
  calib.setBoardData(board_size, 0.04, pattern_mask);
  calib.setPyramidNumLevels(2);
  
  cout << "Loading images and extracting corners ..."<<endl;
  for( auto &f : filelist )
    calib.addImageFile (f);
  cout << "...done"<<endl;
  
  if( !calib.numCheckerboards() )
  {
    std::cerr<<"Unable to load images or to extract corners, exiting"<<endl;
    exit(EXIT_FAILURE);
  }
  
  bool update_img = true, time_to_exit = false;
  int img_idx = 0;
  cv_ext::BasicTimer timer;
  Point text_org_guide_01 (20, 20), text_org_guide_11(20, 40), 
        text_org_info_01 (20, 60), text_org_info_11 (20, 80);
  
  string quick_guide_01("N : next image;   P : prev image;   [SPACE]: active/deactivate image;"),
         quick_guide_11("C : perform calibration;   S : save parameters;   [ESC]: exit.");

  double rep_error = std::numeric_limits<double>::infinity();
  cv_ext::PinholeCameraModel cam_model;
  
  Size img_size = calib.imagesSize();
  float scale = 1.0f/scale_factor;
  // Pre-alloc all the display images (just to avoid to reallocate at each iteration )
  Mat corners_img(Size(round(scale*img_size.width), round(scale*img_size.height)), cv::DataType<Vec3b>::type ),
      corner_dist(Size(round(scale*img_size.width), round(scale*img_size.height)), cv::DataType<float>::type ), 
      corner_dist_rgb(Size(round(scale*img_size.width), round(scale*img_size.height)), cv::DataType<Vec3b>::type );
      
  while ( !time_to_exit )
  {
    if( update_img )
    {
      calib.getCornersImage(img_idx,corners_img, corners_img.size());
      
      stringstream info_01, info_11;
      info_01<<"Image "<<img_idx<<"/"<<calib.numCheckerboards()<<"; reprojection error : "
             <<calib.getReprojectionError(img_idx);
      
      if( !calib.isCheckerboardActive(img_idx) )
      {
        info_01<<" - INACTIVE";
        corners_img *= 0.1;
      }
      
      info_11<<"# used for calibartion : "<<calib.numActiveCheckerboards()
             <<"; global reprojection error : "<<rep_error;
      writeTextBlock(corners_img, quick_guide_01, text_org_guide_01);
      writeTextBlock(corners_img, quick_guide_11, text_org_guide_11);
      writeTextBlock(corners_img, info_01.str(), text_org_info_01, Scalar(0,0,255));
      writeTextBlock(corners_img, info_11.str(), text_org_info_11, Scalar(0,0,255));
      imshow("Current corners", corners_img);
      
      calib.getCornersDistribution(30, corner_dist, corner_dist.size());
      double min_val, max_val;
      cv::minMaxLoc(corner_dist, &min_val, &max_val );
      cv_ext::mapMat2RGB<float>(corner_dist, corner_dist_rgb, max_val );
      imshow("Global corners distribution", corner_dist_rgb);
      
      update_img = false;
    }
    
    switch( cv_ext::waitKeyboard() )
    {
      case 'n':
      case 'N':
        if( ++img_idx >= calib.numCheckerboards() )
          img_idx = 0;
        update_img = true;
        break;
      
      case 'p':
      case 'P':
        if( --img_idx < 0 ) 
          img_idx = calib.numCheckerboards() - 1;
        update_img = true;
        break;
      case ' ':
        calib.setCheckerboardActive( img_idx, !calib.isCheckerboardActive(img_idx) );
        update_img = true;
        break;
      case 'c':
      case 'C':
        cout << endl<< "Calibrating ..." << endl;
        timer.reset();
        cout << "RMS reprojection error :" << (rep_error = calib.calibrate()) << endl;
        cout << "Time elapsed: " << timer.elapsedTimeMs() << endl;
        update_img = true;
       break;

      case 's':
      case 'S':
        cam_model = calib.getCamModel();
        cam_model.writeToFile(calibration_filename);
        cout << "Write calibration to file"<<endl;
        break;
      case 27:
        time_to_exit = true;
        break;
    }
  }
  
  if ( show_undistorted && rep_error != std::numeric_limits<double>::infinity() )
  {
    cv::destroyWindow("Current corners");
    cv::destroyWindow("Global corners distribution");
    
    quick_guide_01 = string("N : next image;   P : prev image;   [ESC]: exit.");
    img_idx = 0;
    update_img = true, time_to_exit = false;
    Mat und_img(Size(round(scale*img_size.width), round(scale*img_size.height)), calib.imagesType() );
    
    while ( !time_to_exit )
    {
      if(update_img)
      {
        
        calib.getUndistortedImage(img_idx,und_img,und_img.size());
        writeTextBlock(und_img, quick_guide_01, text_org_guide_01);
        imshow("Undistorted image", und_img);
      }
      
      switch( cv_ext::waitKeyboard() )
      {
       case 'n':
       case 'N':
         if( ++img_idx >= calib.numCheckerboards() )
           img_idx = 0;
         update_img = true;
         break;
      
        case 'p':
        case 'P':
          if( --img_idx < 0 ) 
            img_idx = calib.numCheckerboards() - 1;
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
