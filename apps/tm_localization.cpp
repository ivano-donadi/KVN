#include <cstdio>
#include <string>
#include <sstream>
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "tm_object_localization.h"

#include "io_utils.h"
#include "apps_utils.h"

namespace po = boost::program_options;
using namespace std;
using namespace cv;
using namespace cv_ext;


struct AppOptions
{
  string model_filename, templates_filename, unit,
      camera_filename, imgs_folder;
  double scale_factor;
  int matching_step, num_matches, match_cell_size;
  int top_boundary, bottom_boundary, left_boundary, rigth_boundary;
  double bb_xoff, bb_yoff, bb_zoff, bb_woff, bb_hoff, bb_doff;
};

void parseCommandLine( int argc, char **argv, AppOptions &options )
{
  string app_name( argv[0] );

  options.unit = std::string("m");
  options.scale_factor = 1.0;
  options.matching_step = 4;
  options.num_matches = 5;
  options.match_cell_size = -1;
  options.top_boundary = options.bottom_boundary =
  options.left_boundary = options.rigth_boundary = -1;

  options.bb_xoff = 0; options.bb_yoff = 0; options.bb_zoff = 0;
  options.bb_woff = 0; options.bb_hoff = 0; options.bb_doff = 0;

  po::options_description desc ( "OPTIONS" );
  desc.add_options()
      ( "help,h", "Print this help messages" )
      ( "model_filename,m", po::value<string > ( &options.model_filename )->required(),
        "STL, PLY, OBJ, ...  CAD model file" )
      ( "unit,u", po::value<string> ( &options.unit ),
        "Optional unit of measure of the CAD model: [m|cm|mm], default: m" )
      ( "templates_filename,t", po::value<string > ( &options.templates_filename )->required(),
        "Templates file" )
      ( "camera_filename,c", po::value<string > ( &options.camera_filename )->required(),
        "A YAML file that stores all the camera parameters in OpenCV-like format" )
      ( "imgs_folder,f", po::value<string > ( &options.imgs_folder )->required(),
        "Input images folder path" )
      ( "scale_factor,s", po::value<double> ( &options.scale_factor ),
        "Optional scale factor applied to input images [default: 1]" )
      ( "ms", po::value<int> ( &options.matching_step ),
        "Sample step used in the exhaustive meaning [default: 4]" )
      ( "nm", po::value<int> ( &options.num_matches ),
        "Number of best matches to be considered for each matching cell  [default: 5]" )
      ( "mcs", po::value<int> ( &options.match_cell_size ),
        "Size in pixels of each matching cell (if -1, just search for the best matches for the whoel image).[default: 100]" )
      ( "tb", po::value<int> ( &options.top_boundary ),
        "Optional region of interest: top boundary " )
      ( "bb", po::value<int> ( &options.bottom_boundary ),
        "Optional region of interest: bottom boundary " )
      ( "lb", po::value<int> ( &options.left_boundary ),
        "Optional region of interest: left boundary " )
      ( "rb", po::value<int> ( &options.rigth_boundary ),
        "Optional region of interest: rigth boundary" )

      ( "off_x", po::value<double > ( &options.bb_xoff ),
        "Optional bounding box origin x offset (EXPERIMENTAL)" )
      ( "off_y", po::value<double > ( &options.bb_yoff ),
        "Optional bounding box origin y offset (EXPERIMENTAL)" )
      ( "off_z", po::value<double > ( &options.bb_zoff ),
        "Optional bounding box origin z offset (EXPERIMENTAL)" )
      ( "off_w", po::value<double > ( &options.bb_woff ),
        "Optional bounding box width offset (EXPERIMENTAL)" )
      ( "off_h", po::value<double > ( &options.bb_hoff ),
        "Optional bounding box height offset (EXPERIMENTAL)" )
      ( "off_d", po::value<double > ( &options.bb_doff ),
        "Optional bounding box delpth offset (EXPERIMENTAL)" );

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

  cout << "Loading model from file : "<<options.model_filename<< endl;
  cout << "Loading camera intrinsic parameters from file : "<<options.camera_filename<< endl;
  cout << "Scale factor applied to input images : "<<options.scale_factor<< endl;
  cout << "Sample step used in the exhaustive matching : "<<options.matching_step<< endl;
  cout << "Number of best matches to be considered for each matching cell : "<<options.matching_step<< endl;


  TMObjectLocalization obl_loc;

  obl_loc.setNumMatches(5);
  obl_loc.setScaleFactor(options.scale_factor);
  obl_loc.setCannyLowThreshold(40);

  // TODO
//  if( options.top_boundary != -1 || options.bottom_boundary != -1 ||
//      options.left_boundary != -1 || options.rigth_boundary != -1 )
//  {
//    Point tl(0,0), br(scaled_img_size.width, scaled_img_size.height);
//
//    if( options.top_boundary != -1 ) { tl.y = options.top_boundary; }
//    if( options.left_boundary != -1 ) { tl.x = options.left_boundary;  }
//    if( options.bottom_boundary != -1 ) { br.y = options.bottom_boundary; }
//    if( options.rigth_boundary != -1 ) { br.x = options.rigth_boundary; }
//
//    obl_loc.setRegionOfInterest( cv::Rect(tl, br) );
//  }

  if ( !options.unit.compare("m") )
    obl_loc.setUnitOfMeasure(RasterObjectModel::METER);
  else if ( !options.unit.compare("cm") )
    obl_loc.setUnitOfMeasure(RasterObjectModel::CENTIMETER);
  else if ( !options.unit.compare("mm") )
    obl_loc.setUnitOfMeasure(RasterObjectModel::MILLIMETER);


  cout << "Bounding box origin offset [off_x, off_y, off_z] : ["
       <<options.bb_xoff<<", "<<options.bb_yoff<<", "<<options.bb_zoff<<"]"<< endl;
  cout << "Bounding box size offset [off_width, off_height, off_depth] : ["
       <<options.bb_woff<<", "<<options.bb_hoff<<", "<<options.bb_doff<<"]"<< endl;

  if( options.bb_xoff ||  options.bb_yoff || options.bb_zoff ||
      options.bb_woff || options.bb_hoff || options.bb_doff )
    obl_loc.setBoundingBoxOffset( options.bb_xoff, options.bb_yoff, options.bb_zoff,
                                  options.bb_woff, options.bb_hoff, options.bb_doff );


  obl_loc.enableDisplay(true);

  obl_loc.initialize(options.camera_filename, options.model_filename, options.templates_filename );

  cout << "Loading input images from directory : "<<options.imgs_folder<< endl;

  vector<string> filelist;
  if ( !readFileNamesFromFolder ( options.imgs_folder, filelist ) )
  {
    cerr<<"Wrong or empty folders"<<endl;
    exit ( EXIT_FAILURE );
  }

  cv_ext::BasicTimer timer;
  cv::Mat src_img;
  for ( int i = 0; i < static_cast<int> ( filelist.size() ); i++ )
  {
    std::cout<<"Loading image : "<<filelist[i]<<endl;
    src_img = cv::imread ( filelist[i] );

    timer.reset();
    obl_loc.localize(src_img );
    cout << "Object localization ms: " << timer.elapsedTimeMs() << endl;
  }

  return EXIT_SUCCESS;
}
