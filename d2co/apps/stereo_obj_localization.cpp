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
  string model_filename, templates_filename, unit, camera_filename[2],
         stereo_filename, imgs_folder_name[2];
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

  std::vector < cv_ext::PinholeCameraModel > cam_models(2);
  for( int i = 0; i < 2; i++ )
  {
    cout << "Loading camera "<<i<<" intrinsic parameters from file : "<<options.camera_filename[i]<< endl;
    cam_models[i].readFromFile ( options.camera_filename[i] );
    cam_models[i].setSizeScaleFactor(options.scale_factor);
  }
  cout << "Scale factor applied to input images : "<<options.scale_factor<< endl;

  
  cout << "Loading stereo camera extrinsic parameters from file : "<<options.stereo_filename<< endl;
  cv::Mat stereo_r_mat, stereo_t_vec;
  cv_ext::read3DTransf ( options.stereo_filename, stereo_r_mat, stereo_t_vec, cv_ext::ROT_FORMAT_MATRIX );
  Size img_size = cam_models[0].imgSize();
  

  bool has_roi = false;
  cv::Rect roi[2];
  
  if( options.top_boundary != -1 || options.bottom_boundary != -1 || 
      options.left_boundary != -1 || options.rigth_boundary != -1 )
  {
    Point tl0(0,0), br0(img_size.width, img_size.height),
          tl1(0,0), br1(img_size.width, img_size.height);

    if( options.top_boundary != -1 ) { tl0.y = tl1.y = options.top_boundary/options.scale_factor; }
    if( options.left_boundary != -1 ) { tl0.x = tl1.x = options.left_boundary/options.scale_factor;  }
    if( options.bottom_boundary != -1 ) { br0.y = br1.y = options.bottom_boundary/options.scale_factor; }
    if( options.rigth_boundary != -1 ) { br0.x = br1.x = options.rigth_boundary/options.scale_factor; }
    
    has_roi = true;
    roi[0] = cv::Rect(tl0, br0);
    roi[1] = cv::Rect(tl1, br1);
  }  
  
  cout << "Rectifying images "<<endl;
  
  cv_ext::StereoRectification stereo_rect;
  stereo_rect.setCameraParameters ( cam_models, stereo_r_mat, stereo_t_vec );
  stereo_rect.update();

  auto rect_cam_models = stereo_rect.getCamModels ();
  Point2f stereo_disp = stereo_rect.getCamDisplacement();
  if( has_roi )
  {
    for( int k = 0; k < 2; k++ )
    {
      cout << "Setting region of interest : "<<roi[k]<< endl;  
      rect_cam_models[k].setRegionOfInterest(roi[k]);
//      rect_cam_models[k].enableRegionOfInterest(true);
    }
  }
  
  cout << "Loading model from file : "<<options.model_filename<< endl;  

  vector < RasterObjectModel3DPtr > obj_model_ptrs;
  for( int i = 0; i < 2; i++ )
  {
    obj_model_ptrs.emplace_back( new RasterObjectModel3D() );
    obj_model_ptrs.back()->setCamModel( rect_cam_models[i] );
    obj_model_ptrs.back()->setStepMeters ( 0.001 );

    std::transform(options.unit.begin(), options.unit.end(), options.unit.begin(), ::tolower);
    if ( !options.unit.compare("m") )
      obj_model_ptrs.back()->setUnitOfMeasure(RasterObjectModel::METER);
    else if ( !options.unit.compare("cm") )
      obj_model_ptrs.back()->setUnitOfMeasure(RasterObjectModel::CENTIMETER);
    else if ( !options.unit.compare("mm") )
      obj_model_ptrs.back()->setUnitOfMeasure(RasterObjectModel::MILLIMETER);

    if(!obj_model_ptrs.back()->setModelFile( options.model_filename ) )
    {
      cout << "Unable to read model file: existing" << endl;
      exit ( EXIT_FAILURE );
    }
    obj_model_ptrs.back()->computeRaster();
  }

  // WARNING UGLY WORKAROUND
  for( int k = 0; k < 2; k++ )
    obj_model_ptrs[k]->cameraModel().enableRegionOfInterest(true);

  DirIdxPointSetVecPtr ts_ptrs = make_shared<DirIdxPointSetVec>();
  cout << "Loading templates from file : " << options.templates_filename << endl;
  loadTemplateVector(options.templates_filename, *ts_ptrs );

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

  DirectionalChamferMatchingBase<DirIdxPointSet> dcm;
  dcm.enableParallelism(true);
  dcm.setTemplatesVector( ts_ptrs );

  DirectionalChamferRegistration dcr;
  dcr.setNumDirections(60);
  dcr.setObjectModel(obj_model_ptrs[0]);

  MultiViewsDirectionalChamferRegistration mdcr;
  mdcr.setNumDirections(60);

  cv_ext::vector_Quaterniond  view_r_quats(2);
  cv_ext::vector_Vector3d view_t_vec(2);

  view_r_quats[0].setIdentity();
  view_r_quats[1].setIdentity();
  view_t_vec[0].setZero();
  view_t_vec[1].setZero();

  view_t_vec[1](0) += stereo_disp.x;
  view_t_vec[1](1) += stereo_disp.y;

  mdcr.setObjectModels( obj_model_ptrs, view_r_quats, view_t_vec );

  ObjectTemplatePnP o_pnp(obj_model_ptrs[0]->cameraModel());
  o_pnp.fixZTranslation(true);

  GradientDirectionScore scoring;

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
  
  vector< TemplateMatch > matches;

  Mat img_pair ( Size ( 2*img_size.width, img_size.height ), cv::DataType<Vec3b>::type ),
      display  ( Size ( 2*img_size.width, img_size.height ), cv::DataType<Vec3b>::type );
  std::vector< Mat > src_img(2), rect_img(2), img_roi(2), proc_img(2), h_display(2);
  int start_col = 0, w = img_size.width;
  for ( int k = 0; k < 2; k++, start_col += w )
  {
    rect_img[k] = img_pair.colRange ( start_col, start_col + w );
    h_display[k] = display.colRange ( start_col, start_col + w );
  }

  
  for ( int i = 0; i < static_cast<int> ( filelist[0].size() ); i++ )
  {
    for ( int k = 0; k < 2; k++ )
    {
      std::cout<<"Camera "<< k<<" : loading image : "<<filelist[k][i]<<endl;
      src_img[k] = cv::imread ( filelist[k][i] );
      if( options.scale_factor != 1 )
        cv::resize(src_img[k],src_img[k],img_size);

//      // Unsharp masking
//      cv::Mat smoothed;
//      cv::GaussianBlur(src_img[k], smoothed, cv::Size(0, 0), 3);
//      cv::addWeighted(src_img[k], 1.5, smoothed, -0.5, 0, src_img[k]);

    }
    
    if( src_img[0].empty() || src_img[1].empty() )
      continue;

    stereo_rect.rectifyImagePair ( src_img, rect_img );


    cv_ext::BasicTimer timer;
    for ( int k = 0; k < 2; k++ )
    {
      if( has_roi )
        img_roi[k] = rect_img[k](roi[k]);
      else
        img_roi[k] = rect_img[k];

//#define USE_BILATERAL_FILTER
#if defined(USE_BILATERAL_FILTER)
      cout<<"DEBUG: bilaterl bilter in Canny!"<<endl;
      bilateralFilter(img_roi[k], proc_img[k], -1, 50, 5);
      cout<<"Elapsed time : "<<timer.elapsedTimeMs()<<endl;
#else
      proc_img[k] = img_roi[k];
#endif
    }

    timer.reset();

    vector<ImageTensorPtr> dst_map_tensor_ptrs(2);
    for (int k = 0; k < 2; k++)
    {
      dst_map_tensor_ptrs[k] = std::make_shared<ImageTensor>();
      dc.computeDistanceMapTensor(proc_img[k], *dst_map_tensor_ptrs[k], num_directions, tensor_lambda, smooth_tensor);
    }

    cout << "Stereo tensor computation ms: " << timer.elapsedTimeMs() << endl;

    dcr.setInput(dst_map_tensor_ptrs[0]);
    mdcr.setInput(dst_map_tensor_ptrs);

    timer.reset();
    dcm.setInput(dst_map_tensor_ptrs[0]);
    dcm.match(options.num_matches, matches, options.matching_step, options.match_cell_size );

    cout << "DCM matching elapsed time ms: " << timer.elapsedTimeMs() << endl;


    vector<Mat_<double> > r_vec, t_vec;
    multimap<double, int> scores;

    img_pair.copyTo(display);
    std::vector< Mat > draw_img(2);
    if (has_roi)
    {
      for (int k = 0; k < 2; k++)
      {
        cv::Point dbg_tl = roi[k].tl(), dbg_br = roi[k].br();
        dbg_tl.x -= 1;
        dbg_tl.y -= 1;
        dbg_br.x += 1;
        dbg_br.y += 1;
        cv::rectangle(h_display[k], dbg_tl, dbg_br, cv::Scalar(255, 255, 255));
        draw_img[k] = h_display[k](roi[k]);
      }
    }
    else
    {
      for (int k = 0; k < 2; k++)
        draw_img[k] = h_display[k];
    }


    timer.reset();

    vector<ImageGradient> im_grad;
    im_grad.reserve(2);
    for (int k = 0; k < 2; k++)
      im_grad.emplace_back( proc_img[k] );

    int i_m = 0;
    for (auto iter = matches.begin(); iter != matches.end(); iter++, i_m++)
    {
      TemplateMatch &match = *iter;

      r_vec.push_back(Mat_<double>(3, 1));
      t_vec.push_back(Mat_<double>(3, 1));

      o_pnp.solve((*ts_ptrs)[match.id],match.img_offset, r_vec.back(), t_vec.back());

      dcr.refinePosition((*ts_ptrs)[match.id], r_vec.back(), t_vec.back());
      mdcr.refinePosition((*ts_ptrs)[match.id], r_vec.back(), t_vec.back());

      Mat_<double> right_t_vec(3, 1);
      vector<Point2f> refined_proj_pts[2];
      vector<float> normals[2];

      obj_model_ptrs[0]->setModelView(r_vec.back(), t_vec.back());
      obj_model_ptrs[0]->projectRasterPoints(refined_proj_pts[0], normals[0]);


      right_t_vec(0, 0) = t_vec.back()(0, 0) + stereo_disp.x;
      right_t_vec(1, 0) = t_vec.back()(1, 0) + stereo_disp.y;
      right_t_vec(2, 0) = t_vec.back()(2, 0);
      obj_model_ptrs[1]->setModelView(r_vec.back(), right_t_vec);
      obj_model_ptrs[1]->projectRasterPoints(refined_proj_pts[1], normals[1]);

      double score[2], avg_score;
      score[0] = scoring.evaluate(im_grad[0], refined_proj_pts[0], normals[0]);
      score[1] = scoring.evaluate(im_grad[1], refined_proj_pts[1], normals[1]);

      if( score[0] > .75 && score[1] > .75 )
        avg_score = (score[0] + score[1])/2;
      else
        avg_score = 0;

      scores.insert(std::pair<double, int>(avg_score, i_m));
    }

    cout << "DCR + MDCR object registration and scoring elapsed time ms : " << timer.elapsedTimeMs() << endl;

    i_m = 0;

    for (auto &s: scores)
    {
      double score = s.first;
      int idx = s.second;
      Mat_<double> right_t_vec(3, 1);
      vector<Point2f> refined_proj_pts[2];

      obj_model_ptrs[0]->setModelView(r_vec[idx], t_vec[idx]);
      obj_model_ptrs[0]->projectRasterPoints(refined_proj_pts[0]);

      right_t_vec(0, 0) = t_vec[idx](0, 0) + stereo_disp.x;
      right_t_vec(1, 0) = t_vec[idx](1, 0) + stereo_disp.y;
      right_t_vec(2, 0) = t_vec[idx](2, 0);
      obj_model_ptrs[1]->setModelView(r_vec[idx], right_t_vec);
      obj_model_ptrs[1]->projectRasterPoints(refined_proj_pts[1]);

      if( score >= .8 )
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

        for (int k = 0; k < 2; k++)
          cv_ext::drawPoints(draw_img[k], refined_proj_pts[k], color);

      }
      else if( score > .75 )
      {
        for (int k = 0; k < 2; k++)
          cv_ext::drawPoints(draw_img[k], refined_proj_pts[k], Scalar(128, 128, 128));
      }

      i_m++;
    }

    cv_ext::showImage(display, "display", true);
  }

  return EXIT_SUCCESS;
}
