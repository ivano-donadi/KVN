#include <cstdio>
#include <string>
#include <sstream>
#include <algorithm>
#include <map>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/rgbd.hpp>
#include "cv_ext/cv_ext.h"
#include "raster_object_model3D.h"
#include "raster_object_model2D.h"
#include "chamfer_matching.h"

#include "apps_utils.h"
#include "automatica_localization.h"

using namespace boost;
namespace po = boost::program_options;
using namespace boost::filesystem;
using namespace std;
using namespace cv;
using namespace cv_ext;


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



struct AppOptions
{
  string model_filename, templates_filename[2], camera_filename[2], stereo_filename,
         imgs_folder_name[2];
  double scale_factor;
  bool has_roi = false;
  cv::Rect roi;
  int top_boundary, bottom_boundary, left_boundary, rigth_boundary;
};

void parseCommandLine( int argc, char **argv, AppOptions &options )
{
  string app_name( argv[0] );

  options.scale_factor = 1.0;
  options.top_boundary = options.bottom_boundary = 
    options.left_boundary = options.rigth_boundary = -1;
  
  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "model_filename,m", po::value<string > ( &options.model_filename )->required(),
    "STL, PLY, OBJ, ...  model file" )
  ( "t0", po::value<string > ( &options.templates_filename[0] )->required(),
    "Camera 0 templates file" )
  ( "t1", po::value<string > ( &options.templates_filename[1] )->required(),
    "Camera 1 templates file" )
  ( "c0", po::value<string > ( &options.camera_filename[0] )->required(),
    "Camera 0 model filename" )
  ( "c1", po::value<string > ( &options.camera_filename[1] )->required(),
    "Camera 1 model filename" )
  ( "sc", po::value<string > ( &options.stereo_filename )->required(),
    "Stereo camera extrinsic parameters filename" )
  ( "f0", po::value<string > ( &options.imgs_folder_name[0] )->required(),
    "Camera 0 input images folder path" )
  ( "f1", po::value<string > ( &options.imgs_folder_name[1] )->required(),
    "Camera 1 input images folder path" )
  ( "scale_factor,s", po::value<double> ( &options.scale_factor ),
    "Scale factor [1]" )
  ( "tb", po::value<int> ( &options.top_boundary ),
    "Optional region of interest: top boundary " )
  ( "bb", po::value<int> ( &options.bottom_boundary ),
    "Optional region of interest: bottom boundary " )
  ( "lb", po::value<int> ( &options.left_boundary ),
  "Optional region of interest: left boundary " )  
  ( "rb", po::value<int> ( &options.rigth_boundary ),
  "Optional region of interest: rigth boundary" );

  po::variables_map vm;

  try
  {
    po::store ( po::parse_command_line ( argc, argv, desc ), vm );

    if ( vm.count ( "help" ) )
    {
      cout << "USAGE: "<<app_name<<" OPTIONS"
                << endl << endl<<desc;                
      exit(EXIT_SUCCESS);
    }

    po::notify ( vm );
  }

  catch ( boost::program_options::required_option& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    cout << "USAGE: "<<app_name<<" OPTIONS"
              << endl << endl<<desc;
    exit(EXIT_FAILURE);
  }

  catch ( boost::program_options::error& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    cout << "USAGE: "<<app_name<<" OPTIONS"
              << endl << endl<<desc;
    exit(EXIT_FAILURE);
  }
}


int main(int argc, char **argv)
{
  AppOptions options;
  parseCommandLine( argc, argv, options );

  cv_ext::PinholeCameraModel cam_models[2];
  for( int i = 0; i < 2; i++ )
  {
    cout << "Loading camera "<<i<<" intrinsic parameters from file : "<<options.camera_filename[i]<< endl;
    cam_models[i].readFromFile ( options.camera_filename[i] );
    cam_models[i].setSizeScaleFactor(options.scale_factor);
  }
  cout << "Scale factor : "<<options.scale_factor<< endl;

  
  cout << "Loading stereo camera extrinsic parameters from file : "<<options.stereo_filename<< endl;
  cv::Mat stereo_r_mat, stereo_t_vec;
  cv_ext::read3DTransf ( options.stereo_filename, stereo_r_mat, stereo_t_vec, cv_ext::ROT_FORMAT_MATRIX );
  Size img_size = cam_models[0].imgSize();

  cv::Rect roi[2];
  
  if( options.top_boundary != -1 || options.bottom_boundary != -1 || 
      options.left_boundary != -1 || options.rigth_boundary != -1 )
  {
    Point tl0(0,0), br0(img_size.width, img_size.height),
          tl1(0,0), br1(img_size.width, img_size.height);
    
    const int bb_l_offset = 0, bb_r_offset = 40;
    if( options.top_boundary != -1 ) { tl0.y = tl1.y = options.top_boundary; }
    if( options.left_boundary != -1 ) { tl0.x = options.left_boundary; tl1.x = img_size.width - options.rigth_boundary + bb_l_offset;  }
    if( options.bottom_boundary != -1 ) { br0.y = br1.y = options.bottom_boundary; }
    if( options.rigth_boundary != -1 ) { br0.x = options.rigth_boundary; br1.x = img_size.width - options.left_boundary + bb_r_offset; }
    
    roi[0] = cv::Rect(tl0, br0);
    roi[1] = cv::Rect(tl1, br1);
  }  
  
  for( int k = 0 ; k < 2; k++ )
    cout << "Loading input images camera "<< k<<" from directory : "<<options.imgs_folder_name[k]<< endl;
  

  vector<string> filelist[2];
  if ( !readFileNamesFromFolder ( options.imgs_folder_name[0], filelist[0] ) ||
       !readFileNamesFromFolder ( options.imgs_folder_name[1], filelist[1] ) )
  {
    cerr<<"Wrong or empty folders"<<endl;
    exit ( EXIT_FAILURE );
  }

  if ( filelist[0].size() != filelist[1].size() )
  {
    cerr<<"Images folder should contain the same number of images, in the same order"<<endl;
    exit ( EXIT_FAILURE );
  }

  
  AutomaticaLocalization loc( cam_models, stereo_r_mat, stereo_t_vec, roi );
  loc.addObj( options.model_filename, options.templates_filename );
  string templates_filename[2];
  templates_filename[0] = string("/home/albe/Datasets/automatica/objects_models/CELL_COVER_left.model");
  templates_filename[1] = string("/home/albe/Datasets/automatica/objects_models/CELL_COVER_right.model");
  loc.addObj( string("/home/albe/Datasets/automatica/objects_models/CELL_COVER.stl"), templates_filename );
  
  loc.initialize();
  
  Mat src_img[2];
  for ( int i = 0; i < int ( filelist[0].size() ); i++ )
  {
    for ( int k = 0; k < 2; k++ )
    {
      src_img[k] = cv::imread ( filelist[k][i] );
      cv::resize(src_img[k],src_img[k],img_size);
      std::cout<<filelist[k][i]<<endl;
    }  
    
    std::vector< ObjIstance > found_obj;
    loc.localize(src_img, found_obj);
  }

  return 0;
}
