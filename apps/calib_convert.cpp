#include <boost/program_options.hpp>

#include "cv_ext/cv_ext.h"
#include "io_utils.h"

namespace po = boost::program_options;
using namespace std;
using namespace cv;

int main( int argc, char **argv )
{
  string app_name(argv[0]), in_camera_filename, out_camera_filename;
  int type = 0;

  po::options_description desc("OPTIONS");
  desc.add_options()
      ("help,h", "Print this help messages")
      ("in_cf", po::value<string>(&in_camera_filename)->required(),
       "Input camera parameters file to be converted")
      ("out_cf", po::value<string>(&out_camera_filename)->required(),
       "Converted output camera parameters file")
      ("type", po::value<int>(&type),
       "Conversion type 0: cv_ext to OpenCV, 1: OpenCV to cv_ext,  [default: 0]");

  po::variables_map vm;
  try
  {
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help"))
    {
      cout << "USAGE: " << app_name << " OPTIONS"
           << endl << endl << desc;
      return 0;
    }

    po::notify(vm);
  }

  catch (boost::program_options::required_option &e)
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    cout << "USAGE: " << app_name << " OPTIONS"
         << endl << endl << desc;
    return -1;
  }

  catch (boost::program_options::error &e)
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    cout << "USAGE: " << app_name << " OPTIONS"
         << endl << endl << desc;
    return -1;
  }

  if(type)
  {
    cv::Size image_size;
    cv::Mat camera_matrix, dist_coeffs;
    if( !loadCameraParams( in_camera_filename, image_size, camera_matrix, dist_coeffs ) )
    {
      cout << "Error loading camera filename, exiting"<< endl;
      return -1;
    }

    cv_ext::PinholeCameraModel cam_model(camera_matrix, image_size.width, image_size.height, dist_coeffs );
    cam_model.writeToFile(out_camera_filename);
  }
  else
  {
    cv_ext::PinholeCameraModel cam_model;
    cam_model.readFromFile(in_camera_filename);
    cv::Size image_size(cam_model.imgWidth(), cam_model.imgHeight());

    saveCameraParams( out_camera_filename, image_size,
                      cam_model.cameraMatrix(), cam_model.distorsionCoeff() );
  }

  return 0;
}