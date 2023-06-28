#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "cv_ext/cv_ext.h"

#include "apps_utils.h"

using namespace std;
using namespace cv;
using namespace boost;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

int main( int argc, char **argv )
{
  string app_name( argv[0] ), imgs_folder_name[2], camera_filename[2], 
         stereo_transf_filename, output_basename("rect_"), out_folder_name("./");
  bool show_rectified = false;
  float scale_factor = 1.0f;
  
  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "f0", po::value<string > ( &imgs_folder_name[0] )->required(),
    "Camera 0 input images folder path" )
  ( "f1", po::value<string > ( &imgs_folder_name[1] )->required(),
    "Camera 1 input images folder path" )
  ( "c0", po::value<string > ( &camera_filename[0] )->required(),
    "Camera 0 model filename" )
  ( "c1", po::value<string > ( &camera_filename[1] )->required(),
    "Camera 1 model filename" )  
  ( "stereo_transformation,t", po::value<string > ( &stereo_transf_filename )->required(),
    "YAML file that contains the 3D transformation matrix between the Camera 0 and Camera 1" )
  ( "output_basename,o", po::value<string > ( &output_basename ),
    "Optional output rectified images basic filename [Default: rect_]" )
  ( "fo", po::value<string > ( &out_folder_name ),
    "Optional output rectified camera models and images folder [Default: ./]" )
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
  for( int k = 0; k < 2; k++ )
    cout << "Loading camera "<<k<<" model from file : "<<camera_filename[k]<< endl;
  cout << "Loading 3D transformation between the Camera 0 and Camera 1 from file : "<<stereo_transf_filename<< endl;
  cout << "Output rectified images basic filename  : "<<output_basename<< endl;
  cout << "Output rectified camera models and images folder : "<<out_folder_name<< endl;
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
  
  std::vector< cv_ext::PinholeCameraModel > cam_models(2);
  for( int k = 0; k < 2; k++ )
    cam_models[k].readFromFile(camera_filename[k]);
  
  cv::Mat stereo_r_mat, stereo_t_vec;
  cv_ext::read3DTransf(stereo_transf_filename, stereo_r_mat, stereo_t_vec, cv_ext::ROT_FORMAT_MATRIX);
  
  cv_ext::StereoRectification stereo_rect;
  stereo_rect.setCameraParameters ( cam_models, stereo_r_mat, stereo_t_vec );
  stereo_rect.setImageScaleFacor(scale_factor);
  stereo_rect.update();
  
  auto rect_cam_models = stereo_rect.getCamModels();

  
  boost::filesystem::path out_path(out_folder_name), out_imgs_path[2];
  
  for( int k = 0; k < 2; k++ )
  {
    stringstream output_folder_name;
    output_folder_name<<"imgs_";
    output_folder_name<<k;
    out_imgs_path[k] = out_path;
    out_imgs_path[k] /= output_folder_name.str();
    if ( !boost::filesystem::exists(out_imgs_path[k]) )
      fs::create_directory( out_imgs_path[k] );
    cout<<"Saving rectified camera"<<k<<" images to "<<out_imgs_path[k].string()<<endl;

  }

  for( int k = 0; k < 2; k++ )
  {
    boost::filesystem::path input_camera_path(camera_filename[k]), 
                            output_camera_path = out_path;
    stringstream output_camera_filename;
    output_camera_filename << input_camera_path.stem().string();
    output_camera_filename << "_rectified";
    output_camera_filename << input_camera_path.extension().string();
    output_camera_path /= output_camera_filename.str();
    
    cout<<"Saving rectified camera"<<k<<" model to "<<output_camera_path.string()<<endl;
    rect_cam_models[k].writeToFile(output_camera_path.string());
  }
  
  cout << "Loading and rectfying images ..."<<endl;
  
  Size out_img_size = stereo_rect.getOutputImageSize();
  Mat img_pair ( Size ( 2*out_img_size.width, out_img_size.height ), cv::DataType<Vec3b>::type );
  vector < Mat > src_img(2), rect_img(2);
  int start_col = 0, w = out_img_size.width;
  for ( int k = 0; k < 2; k++, start_col += w )
    rect_img[k] = img_pair.colRange ( start_col, start_col + w );
  
  for( int i = 0; i < int(filelist[0].size()); i++ )
  {
    cout<<"Rectifying image "<<i<<endl;
    vector < string > fn(2);
    for( int k = 0; k < 2; k++ )
    {
      fn[k] = filelist[k][i];
      src_img[k] = imread ( fn[k] );
    }

    stereo_rect.rectifyImagePair ( src_img, rect_img );

    stringstream output_filename;
    output_filename << output_basename;
    output_filename << std::setfill('0') << std::setw(6) << i;
    for( int k = 0; k < 2; k++ )
    {
      fs::path out_img_path = out_imgs_path[k];
      out_img_path /= output_filename.str();
      out_img_path += ".png";
      imwrite(out_img_path.string(), rect_img[k]);
    }

    if( show_rectified )
      cv_ext::showImage(img_pair, "Rectified image");

  }
  
  cout << "...done"<<endl;
  
  return 0;
}
