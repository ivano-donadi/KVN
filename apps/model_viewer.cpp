#include <string>
#include <sstream>
#include <algorithm>
#include <fstream>

#include <time.h>

#include <boost/program_options.hpp>

#include "cv_ext/cv_ext.h"
#include "raster_object_model3D.h"
#include "raster_object_model2D.h"

#include "io_utils.h"
#include "apps_utils.h"

/* TODO
 * -Update render with RoI
 */
#define TEST_DEPTH_MAP 0

using namespace std;
using namespace cv;
namespace po = boost::program_options;

static void quickGuide()
{
  cout << "Use the keyboard to move the object model:" << endl<< endl;

  objectPoseControlsHelp();

  cout << "Visulization modes:"  << endl;
  cout << "[1] to change mode between points and segments mode" << endl;
  cout << "[2] to enable/disable the visulization of the normals" << endl;
  cout << "[3] to enable/disable the visulization of the object mask" << endl;
  cout << "[4] to enable/disable the visulization of the object depth map" << endl;
  cout << "[5] to enable/disable the visulization of the object render (only available if the model has been loaded with \"color\" option)" << endl;
  cout << "[v] to enable/disable the visulization of the object vertices" << endl;
  cout << "[r] to enable/disable the visulization of the object reference frame" << endl;
  cout << "[b] to enable/disable the visulization of the object bounding box" << endl;;
  cout << "[g] to change the light position in a new random position" << endl;
}

int main(int argc, char **argv)
{
  // Initialize random seed
  srand (time(NULL));

  string app_name( argv[0] ), image_filename;
  double  model_samplig_step = 0.1;
  string rgb_color_str;

  po::options_description viewer_options ("Model viewer Options" );
  viewer_options.add_options()
      ( "image_filename,i", po::value<string > ( &image_filename ),
        "Optional background image file" )
      ( "model_samplig_step", po::value<double> ( &model_samplig_step ),
        "Model sampling step [default: 0.1 m]" )
      ( "color", "Try to load also model colors" )
      ( "rgb", po::value<string> ( &rgb_color_str ),
        "Uniform RGB color (in HEX format) to be used to render the model. The --rgb and --color options are mutually exclusive" )
      ( "light", "Enable lighting in model rendering" );

  po::options_description program_options;

  CADCameraOptions model_opt;
  RoIOptions roi_opt;
  DefaultOptions def_opt;

  program_options.add(model_opt.getDescription());
  program_options.add(roi_opt.getDescription());
  program_options.add(viewer_options);
  program_options.add(def_opt.getDescription());

  po::variables_map vm;
  bool has_color = false, has_light = false;
  try
  {
    po::store ( po::parse_command_line ( argc, argv, program_options ), vm );

    std::string cfg_filename = def_opt.cfgFilename(vm);
    if ( !cfg_filename.empty() )
    {
      std::ifstream ifs(cfg_filename.c_str());
      if (ifs)
        store(po::parse_config_file(ifs, program_options), vm);
    }

    if ( def_opt.helpRequired(vm) )
    {
      cout << "USAGE: "<<app_name<<" OPTIONS"
           << endl << endl<<program_options;
      return 0;
    }

    if ( vm.count ( "color" ) || vm.count ( "rgb" ) )
    {
      has_color = true;
      if ( vm.count ( "light" ) )
        has_light = true;
    }

    po::notify ( vm );
  }

  catch ( boost::program_options::required_option& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    cout << "USAGE: "<<app_name<<" OPTIONS"
         << endl << endl<<program_options;
    return -1;
  }

  catch ( boost::program_options::error& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    cout << "USAGE: "<<app_name<<" OPTIONS"
         << endl << endl<<program_options;
    return -1;
  }

  cv::Size image_size;
  cv::Mat camera_matrix, dist_coeffs;
  if( !loadCameraParams(model_opt.camera_filename, image_size, camera_matrix, dist_coeffs ) )
  {
    cout << "Error loading camera filename, exiting"<< endl;
    return -1;
  }

  cv_ext::PinholeCameraModel cam_model(camera_matrix, image_size.width, image_size.height, dist_coeffs );
  cam_model.setSizeScaleFactor(model_opt.scale_factor);
  int img_w = cam_model.imgWidth(), img_h = cam_model.imgHeight();

  bool has_roi = false;
  cv::Rect roi;

  if( roi_opt.top_boundary != -1 || roi_opt.bottom_boundary != -1 ||
      roi_opt.left_boundary != -1 || roi_opt.rigth_boundary != -1 )
  {
    Point tl(0,0), br(img_w, img_h);

    if( roi_opt.top_boundary != -1 ) tl.y = roi_opt.top_boundary;
    if( roi_opt.left_boundary != -1 ) tl.x = roi_opt.left_boundary;
    if( roi_opt.bottom_boundary != -1 ) br.y = roi_opt.bottom_boundary;
    if( roi_opt.rigth_boundary != -1 ) br.x = roi_opt.rigth_boundary;

    has_roi = true;
    roi = cv::Rect(tl, br);
    cam_model.setRegionOfInterest(roi);
    cam_model.enableRegionOfInterest(true);
    roi = cam_model.regionOfInterest();
  }

  RasterObjectModel3D obj_model;

  obj_model.setCamModel( cam_model );
  obj_model.setStepMeters(model_samplig_step);
  obj_model.setRenderZNear(1);
  obj_model.setRenderZFar(20);

  std::transform(model_opt.unit.begin(), model_opt.unit.end(), model_opt.unit.begin(), ::tolower);
  if ( !model_opt.unit.compare("m") )
    obj_model.setUnitOfMeasure(RasterObjectModel::METER);
  else if ( !model_opt.unit.compare("cm") )
    obj_model.setUnitOfMeasure(RasterObjectModel::CENTIMETER);
  else if ( !model_opt.unit.compare("mm") )
    obj_model.setUnitOfMeasure(RasterObjectModel::MILLIMETER);

  if( has_color )
  {
    if( rgb_color_str.length() )
    {
      uint32_t rgb_color;
      std::stringstream ss;
      ss << std::hex << rgb_color_str;
      ss >> rgb_color;

      uchar r = (rgb_color&0XFF0000)>>16, g = (rgb_color&0XFF00)>>8, b = (rgb_color&0XFF);
      obj_model.enableUniformColor(cv::Scalar(r,g,b));
    }
    else
    {
      obj_model.requestVertexColors();
    }
    if( has_light )
      obj_model.requestRenderLighting();
  }

  if(!obj_model.setModelFile(model_opt.model_filename ))
    return -1;

  // obj_model.setMinSegmentsLen(0.01);
  obj_model.computeRaster();

  model_opt.print();
  roi_opt.print();
  cout << "Model sampling step : "<<model_samplig_step<< endl;
  if( !image_filename.empty() )
    cout << "Loading background image from file : "<<image_filename<< endl;
  if(has_roi)
    cout << "Region of interest : "<<roi<< endl;

  has_color = obj_model.vertexColorsEnabled();

  cv_ext::Box3f bb = obj_model.getBoundingBox();
  double diameter = sqrt(bb.width*bb.width + bb.height*bb.height + bb.depth*bb.depth );
  Mat r_vec = (Mat_<double>(3,1) << M_PI,0,0),
      t_vec = (Mat_<double>(3,1) << 0,0,diameter);
  cv::Point3f point_light_pos(1,0,0), light_dir(0,0,-1);

  cv::Mat background_img;
  if(!image_filename.empty())
  {
    background_img = cv::imread(image_filename, cv::IMREAD_COLOR);
    cv::resize(background_img, background_img, Size(img_w, img_h));
  }

  vector<Point2f> raster_pts, middle_points, vertices;
  vector<Vec4f> raster_segs;
  vector<float> raster_normals_dirs;

  bool segment_mode = false, show_normals = false,
      show_mask = false, show_depth = false, show_render = false,
      draw_vertices = false, draw_bb = false, draw_axis = false;

  quickGuide();

  bool exit_now = false;
  while( !exit_now )
  {
    std::cout<<t_vec<<std::endl;
    obj_model.setModelView(r_vec, t_vec);

    Mat background, draw_img;
    if( background_img.empty() )
      background = Mat( Size(img_w,img_h),
                        DataType<Vec3b>::type, CV_RGB( 0,0,0));
    else
      background = background_img.clone();

    if( has_roi )
      draw_img = background(roi);
    else
      draw_img = background;

    if( show_mask )
    {
      draw_img.setTo(cv::Scalar(255,255,255), obj_model.getMask());
    }
    else if( show_depth )
    {
      Mat depth_img_f = obj_model.getModelDepthMap(), depth_img;

#if TEST_DEPTH_MAP

      vector<Point3f> pts = obj_model.getPoints();
      obj_model.projectRasterPoints( raster_pts );
      int useful_pts = 0;
      double avg_abs_diff = 0, avg_diff = 0, max_abs_diff = 0;
      int m_r, m_c;
      for( int i = 0; i < raster_pts.size(); i++ )
      {
        int r = cvRound(raster_pts[i].y), c = cvRound(raster_pts[i].x);
        float d = depth_img_f.at<float>(r,c);

        if( d != -1 )
        {

          double  scene_pt[] = {pts[i].x, pts[i].y, pts[i].z }, transf_pt[3];
          // ceres::AngleAxisRotatePoint uses the rodriguez formula only away from zero
          ceres::AngleAxisRotatePoint((double *)r_vec.data, scene_pt, transf_pt);
          float z = transf_pt[2] + t_vec.at<double>(2);

          float diff = d-z;
          avg_diff += diff;
//           cout<<"("<<r<<","<<c<<") : |" <<d<<" - "<<z<<" | = "<<diff<<endl;

          diff = fabs(diff);
          if( diff > max_abs_diff )
          {
            max_abs_diff = diff;
            m_r = r;
            m_c = c;
          }
          avg_abs_diff += diff;
          useful_pts++;
        }
      }

      depth_img_f.at<float>(m_r,m_c) = 1.0;
      cout<<"Average abs depth diff : "<<avg_abs_diff/useful_pts<<"Average depth diff : "
          <<avg_diff/useful_pts<<" Max abs depth diff  "<<max_abs_diff<<endl;

#endif

      depth_img_f *= 255;
      depth_img_f.convertTo(depth_img,cv::DataType<uchar>::type );
      cv::cvtColor(depth_img, depth_img, cv::COLOR_GRAY2BGR);

      if( background_img.empty() )
        depth_img.copyTo(draw_img);
      else
        depth_img.copyTo(draw_img, obj_model.getMask());
    }
    else if( show_render )
    {
      if( background_img.empty() )
        obj_model.getRenderedModel().copyTo(draw_img);
      else
      {
        Mat render_img = obj_model.getRenderedModel();
        render_img.copyTo(draw_img, obj_model.getMask());
      }
    }
    else
    {
      if( segment_mode )
      {
        if( show_normals )
        {
          obj_model.projectRasterSegments( raster_segs, raster_normals_dirs );

          middle_points.clear();
          middle_points.reserve(raster_segs.size());
          for(int i = 0; i < int(raster_segs.size()); i++)
          {
            Vec4f &seg = raster_segs[i];
            middle_points.push_back(Point2f((seg[0] + seg[2])/2, (seg[1] + seg[3])/2 ));
          }
          cv_ext::drawSegments( draw_img, raster_segs );
          cv_ext::drawNormalsDirections(draw_img, middle_points, raster_normals_dirs );
        }
        else
        {
          obj_model.projectRasterSegments( raster_segs );
          cv_ext::drawSegments( draw_img, raster_segs );
        }
      }
      else
      {
        if( show_normals )
        {
          obj_model.projectRasterPoints( raster_pts, raster_normals_dirs);
          cv_ext::drawNormalsDirections(draw_img, raster_pts, raster_normals_dirs );
        }
        else
        {
          obj_model.projectRasterPoints( raster_pts );
          cv_ext::drawPoints(draw_img, raster_pts );
        }
      }
    }

    if( draw_vertices )
    {
      obj_model.projectVertices( vertices );
      cv_ext::drawCircles(draw_img, vertices, 1, Scalar(255, 0, 255) );
    }

    if( draw_bb )
    {
      vector< Vec4f > proj_bb_segs;
      vector< Point2f > proj_bb_pts;
      obj_model.projectBoundingBox ( proj_bb_segs );
      obj_model.projectBoundingBox ( proj_bb_pts );

      cv_ext::drawSegments( draw_img, proj_bb_segs, Scalar(0, 255, 255) );
      cv_ext::drawCircles(draw_img, proj_bb_pts, 2, Scalar(0, 0, 255) );
    }


    if( draw_axis )
    {
      vector< Vec4f > proj_segs_x, proj_segs_y,  proj_segs_z;
      obj_model.projectAxes ( proj_segs_x, proj_segs_y, proj_segs_z );

      cv_ext::drawSegments( draw_img, proj_segs_x, Scalar(0, 0, 255) );
      cv_ext::drawSegments( draw_img, proj_segs_y, Scalar(0, 255, 0) );
      cv_ext::drawSegments( draw_img, proj_segs_z, Scalar(255, 0, 0) );
    }

    if( has_roi )
    {
      cv::Point dbg_tl = roi.tl(), dbg_br = roi.br();
      dbg_tl.x -= 1; dbg_tl.y -= 1;
      dbg_br.x += 1; dbg_br.y += 1;
      cv::rectangle( background, dbg_tl, dbg_br, cv::Scalar(255,255,255));
    }

    imshow("Test model", background);
    int key = cv_ext::waitKeyboard();

    auto cur_bb = obj_model.getBoundingBox();
    parseObjectPoseControls( key, r_vec, t_vec );

    switch(key)
    {
      case '1':
        segment_mode = !segment_mode;
        break;
      case '2':
        show_normals = !show_normals;
        break;
      case '3':
        show_mask = !show_mask;
        if( show_mask )
          show_depth = show_render = false;
        break;
      case '4':
        show_depth = !show_depth;
        if( show_depth )
          show_mask = show_render = false;
        break;
      case '5':
        if( has_color )
        {
          show_render = !show_render;
          if( show_render )
            show_depth = show_mask = false;
        }
        break;
      case 'v' :
        draw_vertices = !draw_vertices;
        break;
      case 'b' :
        draw_bb = !draw_bb;
        break;
      case 'r':
        draw_axis = !draw_axis;
        break;
      case 'g':
        point_light_pos.x = rand() - RAND_MAX/2;
        point_light_pos.y = rand() - RAND_MAX/2;
        point_light_pos.z = rand() - RAND_MAX/2;

        light_dir.x = rand() - RAND_MAX/2;
        light_dir.y = rand() - RAND_MAX/2;
        light_dir.z = rand() - RAND_MAX/2;

        if(point_light_pos.x || point_light_pos.y || point_light_pos.z )
          point_light_pos /= cv_ext::norm3D(point_light_pos);
        else
          point_light_pos = cv::Point3f(1,0,0);

        point_light_pos *= diameter;

        if(light_dir.x || light_dir.y || light_dir.z )
          light_dir /= cv_ext::norm3D(light_dir);
        else
          light_dir = cv::Point3f(0,0,-1);

        obj_model.setPointLightPos(point_light_pos);
        obj_model.setLightDirection(light_dir);
        break;

      case 156:// Num keyboard 1
        cur_bb.width -= 0.0025;
        obj_model.setBoundingBox(cur_bb);
        break;
      case 153:// Num keyboard 2
        cur_bb.height -= 0.0025;
        obj_model.setBoundingBox(cur_bb);
        break;
      case 155:// Num keyboard 3
        cur_bb.depth -= 0.0025;
        obj_model.setBoundingBox(cur_bb);
        break;
      case 157:// Num keyboard 5
        obj_model.resetBoundingBox();
        break;
      case 149://Bloc Num 7
        cur_bb.x += 0.0025;
        cur_bb.width -= 0.0025;
        obj_model.setBoundingBox(cur_bb);
        break;
      case 151:// Num keyboard 8
        cur_bb.y += 0.0025;
        cur_bb.height -= 0.0025;
        obj_model.setBoundingBox(cur_bb);
        break;
      case 154:// Num keyboard 9
        cur_bb.z += 0.0025;
        cur_bb.depth -= 0.0025;
        obj_model.setBoundingBox(cur_bb);
        break;
        break;

      case cv_ext::KEY_ESCAPE:
        exit_now = true;
        break;
    }
  }

  return 0;
}