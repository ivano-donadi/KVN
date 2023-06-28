#include "apps_utils.h"

#include "cv_ext/cv_ext.h"


#include <iostream>


using namespace std;
namespace po = boost::program_options;


void DefaultOptions::addOptions()
{
  options_.add_options()
  ( "cfg", po::value<std::string > (&cfg_filename ),
    "Optional default configuration file, each line of the file must have the format:\n<cfg_name> = <value>" )
  ( "help,h", "Print this help messages" );
}

void DefaultOptions::print() const
{
  if(!cfg_filename.empty() )
    std::cout << "Loading default configuration from file : "<<cfg_filename<< std::endl;
}
bool DefaultOptions::helpRequired( const po::variables_map &vm ) const
{
  return ( vm.count ( "help" ) );
}
std::string DefaultOptions::cfgFilename( const po::variables_map &vm ) const
{
  if (vm.count("cfg"))
    return vm["cfg"].as<std::string>();
  else
    return std::string();
}

void CADCameraOptions::addOptions()
{
  options_.add_options()
  ( "model_filename,m", po::value<std::string > (&model_filename )->required(),
      "STL, PLY, OBJ, ...  CAD model file" )
  ( "unit,u", po::value<std::string> ( &unit ),
    "Optional unit of measure of the CAD model: [m|cm|mm], default: m" )
  ( "camera_filename,c", po::value<std::string > ( &camera_filename )->required(),
    "A YAML file that stores all the camera parameters in OpenCV-like format" )
  ( "scale_factor,s", po::value<double > ( &scale_factor ),
    "Optional scale factor applied to input images (the calibration parameters are scaled accordingly) [default: 1]" );
}
void CADCameraOptions::print() const
{
  std::cout << "Loading model from file : "<<model_filename<< std::endl;
  std::cout << "unit of measure of the CAD model : "<<unit<< std::endl;
  std::cout << "Loading camera parameters from file : "<<camera_filename<< std::endl;
  std::cout << "Scale factor : "<<scale_factor<< std::endl;
}

void SpaceSamplingOptions::addOptions()
{
  options_.add_options()
  ( "vert_axis,v", po::value<string> ( &vert_axis ),
    "Vertical axis of the CAD model (i.e. the axis perpendicular to the ground): [x|y|z], default: z" )

  ( "rotation_sampling_level,r", po::value<int> ( &rotation_sampling_level ),
    "Rotation subdivision level [1, 2 (default), 3, 4]: the greater this number, the greater the number of rotations sampled around the object" )

  ( "min_d", po::value<double > ( &min_dist ),
    "Minimum distance in meters from the object (it should be positive, default: 2)" )
  ( "max_d", po::value<double > ( &max_dist ),
    "Maximum distance in meters from the object (it should be positive, default: 16)" )
  ( "d_step", po::value<double > ( &dist_step ),
    "Distance sample step in meters (it should be positive, default: 0.5)" )

  ( "min_h", po::value<double > ( &min_height ),
    "Minimum height in meters from the base of the object with respect to the vertical axis (can be negative, default: 0)" )
  ( "max_h", po::value<double > ( &max_height ),
    "Maximum height in meters from the base of the object with respect to the vertical axis (can be negative, default: 0)" )
  ( "h_step", po::value<double > ( &height_step ),
    "Height sample step in meters (it should be positive, default: 0)" )

  ( "min_s", po::value<double > ( &min_soff ),
    "Minimum sideward offset from the object (can be negative, default: 0)" )
  ( "max_s", po::value<double > ( &max_soff ),
    "Maximum sideward offset from the object (can be negative, default: 0)" )
  ( "s_step", po::value<double > ( &soff_step ),
    "sideward offset sample step in meters (it should be positive, default: 0)" );
}

void SpaceSamplingOptions::print() const
{
  std::cout << "Vertical axis of the CAD model : "<<vert_axis<< std::endl;
  std::cout << "Rotation subdivision level : "<<rotation_sampling_level<< std::endl;
  std::cout << "Distance sampling interval [MIN; MAX, STEP] : "<<"["<<min_dist<<", "<<max_dist<<", "<<dist_step<<"]"<<std::endl;
  std::cout << "Height sampling interval [MIN; MAX, STEP] : "<<"["<<min_height<<", "<<max_height<<", "<<height_step<<"]"<<std::endl;
  std::cout << "Sideward sampling interval [MIN; MAX, STEP] : "<<"["<<min_soff<<", "<<max_soff<<", "<<soff_step<<"]"<<std::endl;
}
bool SpaceSamplingOptions::checkData() const
{
  std::string tmp_vert_axis;
  std::transform(vert_axis.begin(), vert_axis.end(), tmp_vert_axis.begin(), ::tolower);
  if( vert_axis.compare("x") && vert_axis.compare("y") && vert_axis.compare("z") )
  {
    std::cerr << "Invalid Vertical axis"<< std::endl;
    return false;
  }

  if ( rotation_sampling_level < 1 || rotation_sampling_level > 4 )
  {
    std::cerr << "Invalid rotation subdivision level "<< endl;
    return false;
  }

  if( !checkInterval( min_dist, max_dist, dist_step, "distance" ) ||
      !checkInterval( min_height, max_height, height_step, "height" ) ||
      !checkInterval( min_soff, max_soff, soff_step, "sideward offset" ) )
    return false;
  else
    return true;
}

void RoIOptions::addOptions()
{
  options_.add_options()
  ( "tb", po::value<int> ( &top_boundary ),
    "Optional region of interest: top boundary " )
  ( "bb", po::value<int> ( &bottom_boundary ),
     "Optional region of interest: bottom boundary " )
  ( "lb", po::value<int> ( &left_boundary ),
    "Optional region of interest: left boundary " )
  ( "rb", po::value<int> ( &rigth_boundary ),
    "Optional region of interest: rigth boundary" );
}

void RoIOptions::print() const
{
  cout<<"Region of interest top-left corner (x,y)  : ["<<left_boundary<<" "<<top_boundary<<"]"<<endl;
  cout<<"Region of interest bottom-right corner  (x,y) : ["<<rigth_boundary<<" "<<bottom_boundary<<"]"<<endl;
}

void BBOffsetOptions::addOptions()
{
  options_.add_options()
  ( "off_x", po::value<double > ( &bb_xoff ),
    "Optional bounding box origin x offset (EXPERIMENTAL)" )
  ( "off_y", po::value<double > ( &bb_yoff ),
    "Optional bounding box origin y offset (EXPERIMENTAL)" )
  ( "off_z", po::value<double > ( &bb_zoff ),
    "Optional bounding box origin z offset (EXPERIMENTAL)" )
  ( "off_w", po::value<double > ( &bb_woff ),
    "Optional bounding box width offset (EXPERIMENTAL)" )
  ( "off_h", po::value<double > ( &bb_hoff ),
    "Optional bounding box height offset (EXPERIMENTAL)" )
  ( "off_d", po::value<double > ( &bb_doff ),
    "Optional bounding box delpth offset (EXPERIMENTAL)" );
}

void BBOffsetOptions::print() const
{
  cout<<"3D bounding box origin offsets [off_X, off_Y, off_Z] : ["<<bb_xoff<<" "<<bb_yoff<<" "<<bb_zoff<<"]"<<endl;
  cout<<"3D bounding box zie offsets [off_width, off_height, off_depth] : ["<<bb_woff<<" "<<bb_hoff<<" "<<bb_doff<<"]"<<endl;
}

void SynLocOptions::addOptions()
{
  options_.add_options()

  ( "id", po::value<int> ( &obj_id )->required(),
     "Object class ID" )
  ( "templates_filename,t", po::value<string > ( &templates_filename )->required(),
    "D2CO templates file" )
  ( "pvnet_home", po::value<string > ( &pvnet_home ),
    "PVNet root directory [default: ../pvnet]" )
  ( "pvnet_model,p", po::value<string > ( &pvnet_model )->required(),
    "PVNet model filename" )
  ( "pvnet_inference_meta,i", po::value<string > ( &pvnet_inference_meta )->required(),
    "PVNet inference metadata filename" )
  ( "model_sampling_step", po::value<double> ( &model_samplig_step ),
    "Model sampling step [default: 0.1 m]" )
  ( "score_threshold", po::value<double> ( &score_threshold ),
    "Matching score threshold [default: 0.6]" )
  ( "ms", po::value<int> ( &matching_step ),
    "Sample step used in the exhaustive matching [default: 4]" )
  ( "nm", po::value<int> ( &num_matches ),
    "Number of best matches to be considered for each matching cell  [default: 5]" );
}
void SynLocOptions::print() const
{
  cout<<"Object class ID "<<obj_id<<endl;
  cout<<"D2CO templates file : "<<templates_filename<<endl;
  cout<<"PVNet root directory : "<<pvnet_home<<endl;
  cout<<"PVNet model filename : "<<pvnet_model<<endl;
  cout<<"PVNet inference metadata filename : "<<pvnet_inference_meta<<endl;
  cout<<"Model sampling step : "<<model_samplig_step<<endl;
  cout<<"Matching score threshold : "<<score_threshold<<endl;
  cout<<"Sample step used in the exhaustive matching : "<<matching_step<<endl;
  cout<<"Number of best matches to be considered for each matching cell : "<<num_matches<<endl;
}

void objectPoseControlsHelp()
{
  std::cout << "Object pose controls"<< std::endl;
  std::cout << "[j-l-k-i-u-o] translate the model along the X-Y-Z axis " << std::endl;
  std::cout << "[a-s-q-e-z-w] handle the rotation through axis-angle notation " << std::endl;  
}

void parseObjectPoseControls ( int key ,cv::Mat &r_vec, cv::Mat &t_vec,
                               double r_inc, double t_inc )
{
  switch( key )
  {
    case 'a':
    case 'A':
      r_vec.at<double>(1,0) += r_inc;
      break;
    case 's':
    case 'S':
      r_vec.at<double>(1,0) -= r_inc;
      break;
    case 'w':
    case 'W':
      r_vec.at<double>(0,0) += r_inc;
      break;
    case 'z':
    case 'Z':
      r_vec.at<double>(0,0) -= r_inc;
      break;
    case 'q':
    case 'Q':
      r_vec.at<double>(2,0) += r_inc;
      break;
    case 'e':
    case 'E':
      r_vec.at<double>(2,0) -= r_inc;
      break;
    case 'i':
    case 'I':
      t_vec.at<double>(1,0) -= t_inc;
      break;
    case 'm':
    case 'M':
      t_vec.at<double>(1,0) += t_inc;
      break;
    case 'j':
    case 'J':
      t_vec.at<double>(0,0) -= t_inc;
      break;
    case 'k':
    case 'L':
      t_vec.at<double>(0,0) += t_inc;
      break;
    case 'u':
    case 'U':
      t_vec.at<double>(2,0) -= t_inc;
      break;
    case 'o':
    case 'O':
      t_vec.at<double>(2,0) += t_inc;
      break;
    default:
      break;
  }
}

bool checkInterval( double min, double max, double step, const std::string &name )
{
  if( min > max || ( max > min && step <= 0 ) )
  {
    if( !name.empty() )
      cerr << "Invalid "<<name<<" interval"<< endl;
    return false;
  }
  else
  {
    return true;
  }

  return true;
}

