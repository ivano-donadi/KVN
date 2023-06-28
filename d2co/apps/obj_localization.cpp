#include <cstdio>
#include <string>
#include <sstream>
#include <algorithm>
#include <map>
#include <boost/program_options.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/rgbd.hpp>
#include "cv_ext/cv_ext.h"
#include "raster_object_model3D.h"
#include "chamfer_matching.h"
#include "chamfer_registration.h"
#include "scoring.h"

#include "apps_utils.h"

namespace po = boost::program_options;
using namespace std;
using namespace cv;
using namespace cv_ext;


struct AppOptions
{
  string model_filename, templates_filename, camera_filename, unit,
         imgs_folder, matching_algorithm;
  double scale_factor;
  int matching_step, num_matches, match_cell_size;
  int top_boundary, bottom_boundary, left_boundary, rigth_boundary;
};

void parseCommandLine( int argc, char **argv, AppOptions &options )
{
  string app_name( argv[0] );

  options.unit = std::string("mm");
  options.scale_factor = 1.0;
  options.matching_step = 4;
  options.num_matches = 5;
  options.match_cell_size = 100;
  options.matching_algorithm = "DCM";
  options.top_boundary = options.bottom_boundary =
  options.left_boundary = options.rigth_boundary = -1;

  po::options_description desc ( "OPTIONS" );
  desc.add_options()
      ( "help,h", "Print this help messages" )
      ( "model_filename,m", po::value<string > ( &options.model_filename )->required(),
        "STL, PLY, OBJ, ...  CAD model file" )
      ( "unit,u", po::value<string> ( &options.unit ),
        "Optional unit of measure of the CAD model: [m|cm|mm], default: mm" )
      ( "templates_filename,t", po::value<string > ( &options.templates_filename )->required(),
        "Templates file" )
      ( "camera_filename,c", po::value<string > ( &options.camera_filename )->required(),
        "Camera model filename" )
      ( "imgs_folder,f", po::value<string > ( &options.imgs_folder )->required(),
        "Input images folder path" )
      ( "scale_factor,s", po::value<double> ( &options.scale_factor ),
        "Scale factor applied to input images [default: 1]" )
      ( "matching_algorithm,a", po::value<string> ( &options.matching_algorithm ),
        "Used matching algorithm, options: [DCM], CM" )
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

  std::transform(options.matching_algorithm.begin(),
                 options.matching_algorithm.end(),
                 options.matching_algorithm.begin(), ::toupper);

  if( options.matching_algorithm.compare("DCM") &&
      options.matching_algorithm.compare("CM") )
  {
    cerr<<"Unrecognized algorithm type ("<<options.matching_algorithm<<")"<<endl;
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char **argv)
{
  AppOptions options;
  parseCommandLine( argc, argv, options );

  cv_ext::PinholeCameraModel cam_model;

  cout << "Loading model from file : "<<options.model_filename<< endl;
  cout << "Loading camera intrinsic parameters from file : "<<options.camera_filename<< endl;
  cout << "Scale factor applied to input images : "<<options.scale_factor<< endl;

  cam_model.readFromFile ( options.camera_filename );
  cam_model.setSizeScaleFactor(options.scale_factor);
  Size scaled_img_size = cam_model.imgSize();

  bool has_roi = false;
  cv::Rect roi;

  if( options.top_boundary != -1 || options.bottom_boundary != -1 ||
      options.left_boundary != -1 || options.rigth_boundary != -1 )
  {
    Point tl(0,0), br(scaled_img_size.width, scaled_img_size.height);

    if( options.top_boundary != -1 ) { tl.y = options.top_boundary; }
    if( options.left_boundary != -1 ) { tl.x = options.left_boundary;  }
    if( options.bottom_boundary != -1 ) { br.y = options.bottom_boundary; }
    if( options.rigth_boundary != -1 ) { br.x = options.rigth_boundary; }

    has_roi = true;
    roi = cv::Rect(tl, br);
  }

  if( has_roi )
  {
    cout << "Setting region of interest : "<<roi<< endl;
    cam_model.setRegionOfInterest(roi);
//  cam_model.enableRegionOfInterest(true);
  }

  RasterObjectModel3DPtr obj_model_ptr = make_shared<RasterObjectModel3D>();
  obj_model_ptr->setCamModel( cam_model );
  obj_model_ptr->setStepMeters ( 0.001 );

  if ( !options.unit.compare("m") )
    obj_model_ptr->setUnitOfMeasure(RasterObjectModel::METER);
  else if ( !options.unit.compare("cm") )
    obj_model_ptr->setUnitOfMeasure(RasterObjectModel::CENTIMETER);
  else if ( !options.unit.compare("mm") )
    obj_model_ptr->setUnitOfMeasure(RasterObjectModel::MILLIMETER);

  if(!obj_model_ptr->setModelFile( options.model_filename ) )
  {
    cout << "Unable to read model file: existing" << endl;
    exit ( EXIT_FAILURE );
  }

  obj_model_ptr->computeRaster();

  // WARNING UGLY WORKAROUND
  obj_model_ptr->cameraModel().enableRegionOfInterest(true);

  DirIdxPointSetVecPtr ts_ptr = make_shared<DirIdxPointSetVec>();
  cout << "Loading templates from file : " << options.templates_filename << endl;
  loadTemplateVector(options.templates_filename, *ts_ptr );

  DistanceTransform dc;
  dc.setDistThreshold(30);
  dc.enableParallelism(true);

  CannyEdgeDetectorUniquePtr edge_detector_ptr( new CannyEdgeDetector() );
  edge_detector_ptr->setLowThreshold(40);
  edge_detector_ptr->setRatio(2);
  edge_detector_ptr->enableRGBmodality(true);

  dc.setEdgeDetector(std::move(edge_detector_ptr));

  const int num_directions = 60;
  const double tensor_lambda = 6.0;
  const bool smooth_tensor = false;

  cout << "Algorithm type : "<<options.matching_algorithm<< endl;

  std::shared_ptr< TemplateMatching<DirIdxPointSet> > tm;

  if( !options.matching_algorithm.compare("DCM") )
    tm = std::make_shared<DirectionalChamferMatchingBase<DirIdxPointSet> >();
  else if( !options.matching_algorithm.compare("CM") )
    tm = std::make_shared<ChamferMatchingBase<DirIdxPointSet> >();

  tm->enableParallelism(true);
  tm->setTemplatesVector( ts_ptr );

  DirectionalChamferRegistration dcr;
  dcr.setNumDirections(60);
  dcr.setObjectModel(obj_model_ptr);

  ObjectTemplatePnP o_pnp(obj_model_ptr->cameraModel());
  o_pnp.fixZTranslation(true);

  GradientDirectionScore scoring;

  cout << "Loading input images from directory : "<<options.imgs_folder<< endl;

  vector<string> filelist;
  if ( !readFileNamesFromFolder ( options.imgs_folder, filelist ) )
  {
    cerr<<"Wrong or empty folders"<<endl;
    exit ( EXIT_FAILURE );
  }

  vector< TemplateMatch > matches;

  Mat display  (scaled_img_size, cv::DataType<Vec3b>::type );
  Mat src_img, resized_img, img_roi;

  cv_ext::BasicTimer timer;

  for ( int i = 0; i < static_cast<int> ( filelist.size() ); i++ )
  {
    std::cout<<"Loading image : "<<filelist[i]<<endl;
    src_img = cv::imread ( filelist[i] );

    if( src_img.empty() )
      continue;

    if( options.scale_factor != 1 )
      cv::resize(src_img, resized_img, scaled_img_size);
    else
      resized_img = src_img.clone();

    if( has_roi )
      img_roi = resized_img(cam_model.regionOfInterest());
    else
      img_roi = resized_img;

    timer.reset();

    ImageTensorPtr dst_map_tensor_ptr = std::make_shared<ImageTensor>();
    dc.computeDistanceMapTensor( img_roi, *dst_map_tensor_ptr, num_directions, tensor_lambda, smooth_tensor);

    cout << "Tensor computation ms: " << timer.elapsedTimeMs() << endl;

    dcr.setInput(dst_map_tensor_ptr);

    timer.reset();

    if( !options.matching_algorithm.compare("DCM") )
    {
      (std::dynamic_pointer_cast<DirectionalChamferMatchingBase<DirIdxPointSet > >(tm))->setInput(dst_map_tensor_ptr);
    }
    else if( !options.matching_algorithm.compare("CM") )
    {
      cv::Mat dst_map;
      dc.computeDistanceMap( img_roi, dst_map );
      (std::dynamic_pointer_cast<ChamferMatchingBase<DirIdxPointSet > >(tm))->setInput(dst_map);
    }

    tm->match(options.num_matches, matches, options.matching_step, options.match_cell_size );

    cout << "Template matching elapsed time ms: " << timer.elapsedTimeMs() << endl;

    int i_m = 0;
    vector<Mat_<double> > r_vec, t_vec;
    multimap<double, int> scores;

    resized_img.copyTo(display);
    Mat draw_img;
    if (has_roi)
    {
      cv::Rect cur_roi = cam_model.regionOfInterest();
      cv::Point dbg_tl = cur_roi.tl(), dbg_br = cur_roi.br();
      dbg_tl.x -= 1;
      dbg_tl.y -= 1;
      dbg_br.x += 1;
      dbg_br.y += 1;
      cv::rectangle(display, dbg_tl, dbg_br, cv::Scalar(255, 255, 255));
      draw_img = display(cur_roi);
    }
    else
      draw_img = display;

    timer.reset();

    ImageGradient im_grad( img_roi );

    for (auto iter = matches.begin(); iter != matches.end(); iter++, i_m++)
    {
      TemplateMatch &match = *iter;

      r_vec.push_back(Mat_<double>(3, 1));
      t_vec.push_back(Mat_<double>(3, 1));

      o_pnp.solve((*ts_ptr)[match.id],match.img_offset, r_vec.back(), t_vec.back());

      dcr.refinePosition((*ts_ptr)[match.id], r_vec.back(), t_vec.back());

      vector<Point2f> refined_proj_pts;
      vector<float> normals;
      obj_model_ptr->setModelView(r_vec.back(), t_vec.back());
      obj_model_ptr->projectRasterPoints(refined_proj_pts, normals);
      double score = scoring.evaluate(im_grad, refined_proj_pts, normals);
      scores.insert(std::pair<double, int>(score, i_m));
    }

    cout << "DCR object registration and scoring elapsed time ms : " << timer.elapsedTimeMs() << endl;

    i_m = 0;

    for (auto &s: scores)
    {
      double score = s.first;
      int idx = s.second;
      vector<Point2f> refined_proj_pts;
      obj_model_ptr->setModelView(r_vec[idx], t_vec[idx]);
      obj_model_ptr->projectRasterPoints(refined_proj_pts);

      if( score > .8 )
      {
        Scalar color;
        if( i_m == static_cast<int>(scores.size()) - 1 )
          color = Scalar(0, 0, 255);
        else if( i_m == static_cast<int>(scores.size()) - 2 )
          color = Scalar(255, 0, 0);
        else if( i_m == static_cast<int>(scores.size()) - 3 )
          color = Scalar(0, 255, 0);
        else
          color = Scalar(255, 255, 255);

        cv_ext::drawPoints(draw_img, refined_proj_pts, color);
      }
      else if( score > .75 )
        cv_ext::drawPoints(draw_img, refined_proj_pts, Scalar(128, 128, 128));

      i_m++;
    }

    cv_ext::showImage(display, "display", true);
  }

  return EXIT_SUCCESS;
}
