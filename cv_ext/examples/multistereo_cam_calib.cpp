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
  int num_cameras;
  string app_name( argv[0] ), imgs_folder_basename, camera_model_basename,  output_basename;
  float scale_factor = 1.0f;

  po::options_description desc ( "OPTIONS" );
  desc.add_options()
      ( "help,h", "Print this help messages" )
      ( "num_cameras,n", po::value<int> ( &num_cameras )->required(),
        "Num cameras og the stereo rig" )
      ( "imgs_folders,f", po::value<string > ( &imgs_folder_basename )->required(),
        "Input images folders path basename. Folder names are composed as: <imgs_folders>X/, X=0,..,num_cameras - 1" )
      ( "camera_models,c", po::value<string > ( &camera_model_basename ),
        "Camera models basename. Model filenames are composed as: <camera_models>X.yml, X=0,..,num_cameras - 1" )
      ( "output_basename,o", po::value<string > ( &output_basename )->required(),
        "Output extrinsics parameters basic filename" )
      ( "scale_factor,s", po::value<float> ( &scale_factor),
        "Optional scale factor used to scale the output images [Default: 1.0]"  );

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

  cout << "Calibrating a multistereo rig composed by "<<num_cameras<<" cameras"<< endl;

  vector < string > imgs_folder_name(num_cameras);
  vector < string > camera_filename(num_cameras);

  for( int k = 0; k < num_cameras; k++ )
  {
    stringstream f_sstr, c_sstr;
    f_sstr<<imgs_folder_basename;
    f_sstr<<k;
    imgs_folder_name[k] = f_sstr.str();
    c_sstr<<camera_model_basename;
    c_sstr<<k;
    c_sstr<<".yml";
    camera_filename[k] = c_sstr.str();
  }

  for( int k = 0; k < num_cameras; k++ )
  {
    cout << "Loading camera "<<k<<" images from folder : "<<imgs_folder_name[k]<< endl;
  }

  for( int k = 0; k < num_cameras; k++ )
    cout << "Loading camera "<<k<<" model from file : "<<camera_filename[k]<< endl;

  cout << "Output images scale factor : "<<scale_factor<< endl;

  vector < vector<string> > filelist(num_cameras);
  for( int k = 0; k < num_cameras; k++ )
  {
    if( !readFileNamesFromFolder ( imgs_folder_name[k], filelist[k] ) )
    {
      cerr<<"Wrong or empty folders"<<endl;
      exit(EXIT_FAILURE);
    }

    if( k && filelist[k].size() != filelist[k-1].size() )
    {
      cerr<<"Images folder should contain the same number of images, in the same order"<<endl;
      exit(EXIT_FAILURE);
    }
  }

  cv_ext::MultiStereoCameraCalibration calib(num_cameras);

  Size board_size(8,6);
  cv::Mat pattern_mask  = cv_ext::getStandardPatternMask (board_size);
  pattern_mask.at<uchar>(0,0) = 1;
  pattern_mask.at<uchar>(0,6) = 1;
  calib.setBoardData(board_size, 0.04, pattern_mask);
  calib.setPyramidNumLevels(2);

  std::vector < cv_ext::PinholeCameraModel > cam_models(num_cameras);
  for( int k = 0; k < num_cameras; k++ )
    cam_models[k].readFromFile(camera_filename[k]);
  calib.setCamModels(cam_models);
  
  cout << "Loading images and extracting corners ..."<<endl;

  for( int i = 0; i < static_cast<int>( filelist[0].size() ); i++ )
  {
    vector< string > fn;
    fn.reserve(num_cameras);
    for( int k = 0; k < num_cameras; k++ )
      fn.push_back( filelist[k][i] );

    cout<<"Adding tuple "<<i<<" of "<<filelist[0].size()<<std::endl;
    calib.addImageTupleFiles(fn);
  }

  cout << "...done"<<endl;

  if( !calib.numCheckerboards() )
  {
    cerr<<"Unable to load images or to extract corners, exiting"<<endl;
    exit(EXIT_FAILURE);
  }

  bool update_img = true, time_to_exit = false;
  int tuple_idx = 0;
  cv_ext::BasicTimer timer;
  Point text_org_guide_01 (20, 20), text_org_guide_11(20, 40),
      text_org_info_01 (20, 60), text_org_info_11 (20, 80);

  string quick_guide_01("N : next pair;   P : prev pair;   [SPACE]: active/deactivate pair;"),
         quick_guide_11("C : perform calibration;   S : save parameters;   [ESC]: exit.");

  double epi_rmse = numeric_limits<double>::infinity();

  float scale = 1.0f/scale_factor;
  Size img_size = calib.imagesSize();

  cv::Size grid_size;
  if( num_cameras < 4)
  {
    grid_size.width = num_cameras;
    grid_size.height = 1;
  }
  else
  {
    grid_size.width = ceil(sqrt(num_cameras));
    grid_size.height = ceil(num_cameras/grid_size.width);
  }

  cv::Size cell_size(round(scale*img_size.width), round(scale*img_size.height));
  vector < Mat > corners_img, corner_dist, corner_dist_rgb;
  Mat corners_img_grid = cv_ext::createImageGrid(grid_size, cell_size, CV_8UC3, corners_img );
  Mat corner_dist_rgb_grid = cv_ext::createImageGrid(grid_size, cell_size, CV_8UC3, corner_dist_rgb );

  vector< Mat > r_mats, t_vecs;
  while ( !time_to_exit )
  {
    if( update_img )
    {
      calib.getCornersImageTuple(tuple_idx, corners_img, cell_size);
      stringstream info_01, info_11;
      info_01<<"Tuple "<<tuple_idx<<"/"<<calib.numCheckerboards()<<"; epipolar RMSE : "
             <<calib.getEpipolarError(tuple_idx);

      if( !calib.isCheckerboardActive(tuple_idx) )
      {
        info_01<<" - INACTIVE";
        corners_img_grid *= 0.1;
      }

      info_11<<"# used for calibartion : "<<calib.numActiveCheckerboards()
             <<"; global epipolar RMSE : "<<epi_rmse;

      writeTextBlock(corners_img_grid, quick_guide_01, text_org_guide_01);
      writeTextBlock(corners_img_grid, quick_guide_11, text_org_guide_11);
      writeTextBlock(corners_img_grid, info_01.str(), text_org_info_01, Scalar(0,0,255));
      writeTextBlock(corners_img_grid, info_11.str(), text_org_info_11, Scalar(0,0,255));
      imshow("Current corners", corners_img_grid);

      calib.getCornersDistribution( 30, corner_dist, cell_size);
      for( int k = 0; k < num_cameras; k++ )
      {
        double min_val, max_val;
        cv::minMaxLoc(corner_dist[k], &min_val, &max_val );
        cv_ext::mapMat2RGB<float>(corner_dist[k], corner_dist_rgb[k], max_val );
      }
      imshow("Global corners distribution", corner_dist_rgb_grid);
      update_img = false;
    }

    switch( cv_ext::waitKeyboard() )
    {
      case 'n':
      case 'N':
        if( ++tuple_idx >= calib.numCheckerboards() )
          tuple_idx = 0;
        update_img = true;
        break;

      case 'p':
      case 'P':
        if( --tuple_idx < 0 )
          tuple_idx = calib.numCheckerboards() - 1;
        update_img = true;
        break;
      case ' ':
        calib.setCheckerboardActive( tuple_idx, !calib.isCheckerboardActive(tuple_idx) );
        update_img = true;
        break;
      case 'c':
      case 'C':
        cout << endl<< "Calibrating ..." << endl;
        timer.reset();
        cout << "Average epipolar error :" << (epi_rmse = calib.calibrate()) << endl;
        cout << "Time elapsed: " << timer.elapsedTimeMs() << endl;

        calib.getExtrinsicsParameters(r_mats, t_vecs);

        cout << "Estimated extrinsics : "<<endl;
        for( int k = 0; k < num_cameras; k++ )
          cout << r_mats[k]<<endl<<t_vecs[k]<<endl;

        update_img = true;
        break;

      case 's':
      case 'S':
        if( r_mats.empty() || t_vecs.empty() )
          cout<<"Error saving extrinsics parameter: you should first run the calibration"<<endl;
        else
        {
          // TODO
          for( int k = 0; k < num_cameras; k++ )
          {
            stringstream o_sstr;
            o_sstr<<output_basename;
            o_sstr<<k;
            o_sstr<<".yml";
            cv_ext::write3DTransf(o_sstr.str(), r_mats[k], t_vecs[k]);
            cout<<"Saving extrinsics parameter to "<<o_sstr.str()<<endl;
          }
        }
        break;
      case 27:
        time_to_exit = true;
        break;
    }
  }

  return 0;
}
