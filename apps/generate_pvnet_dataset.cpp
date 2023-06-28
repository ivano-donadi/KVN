#include <string>
#include <sstream>
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>

#include "cv_ext/cv_ext.h"
#include "raster_object_model3D.h"

#include "pvnet_dataset.h"

#include "io_utils.h"
#include "apps_utils.h"
#include "pyhelper.h"

namespace po = boost::program_options;
using namespace std;
using namespace cv;

cv::Mat setRenderedImage(RasterObjectModel3D& obj_model, const cv_ext::PinholeCameraModel& cam_model, const vector< string > bkg_images_names,cv::Mat& render_img){
  if (bkg_images_names.size())
  {
    int bkg_img_idx = bkg_images_names.size() * (static_cast<double>( rand()) / RAND_MAX);
    if (bkg_img_idx == static_cast<int>( bkg_images_names.size()))
      bkg_img_idx--;

    cv::Mat bkg_img = cv::imread(bkg_images_names[bkg_img_idx]);
    if (!bkg_img.empty())
    {
      cv::resize(bkg_img, bkg_img, cv::Size(cam_model.imgWidth(), cam_model.imgHeight()));
      render_img = obj_model.getRenderedModel(bkg_img);
      return bkg_img;
    } else
    {
      render_img = obj_model.getRenderedModel();
      return cv::Mat();
    }
  } else
  {
    render_img = obj_model.getRenderedModel();
    return cv::Mat();
  }
}

int main( int argc, char **argv )
{
  srand(0);
  float distance_ratio_transform = 1.;

  string app_name( argv[0] ), pvnet_home = "../pvnet", dataset_dir, bkg_images_dir;
  int obj_id = 0;
  bool stereo = false;

  po::options_description dataset_options ("Dataset Generator Options" );
  dataset_options.add_options()
      ( "pvnet_home", po::value<string > ( &pvnet_home ),
        "PVNet root directory [default: ../pvnet]" )
      ( "id", po::value<int> ( &obj_id ),
        "Object class ID" )
      ( "dataset_dir,d", po::value<string > ( &dataset_dir )->required(),
        "Output dataset directory" )
      ( "bkg_images_dir,b", po::value<string > ( &bkg_images_dir ),
        "Optional background images directory" )
      ( "stereo", po::value<bool> (&stereo), "Whether to build a mono or stereo dataset");

  po::options_description program_options;

  CADCameraOptions model_opt;
  DefaultOptions def_opt;
  SpaceSamplingOptions ss_opt;

  program_options.add(model_opt.getDescription());
  program_options.add(dataset_options);
  program_options.add(ss_opt.getDescription());
  program_options.add(def_opt.getDescription());

  po::variables_map vm;
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

  if(!ss_opt.checkData())
    return -1;

  model_opt.print();
  cout << "Output dataset directory : "<<dataset_dir<< endl;
  if(!bkg_images_dir.empty())
    cout << "Background images directory : "<<bkg_images_dir<< endl;
  ss_opt.print();
  def_opt.print();

  PVNetDataset ds;
  ds.init(dataset_dir, stereo);

  cv::Point3f point_light_pos(1,0,0), light_dir(0,0,-1);
  cv::Scalar model_color(255,255,255);

  cv::Size image_size;
  cv::Mat camera_matrix, dist_coeffs;
  double baseline = 0.;

  bool loadCameraSuccess = false;
  if(stereo){
    loadCameraSuccess = loadStereoCameraParams(model_opt.camera_filename, image_size, camera_matrix, dist_coeffs, baseline );
  } else {
    loadCameraSuccess = loadCameraParams(model_opt.camera_filename, image_size, camera_matrix, dist_coeffs );
  }
  if( !loadCameraSuccess )
  {
    cout << "Error loading camera filename, exiting"<< endl;
    return -1;
  }

  cv_ext::PinholeCameraModel cam_model(camera_matrix, image_size.width, image_size.height, dist_coeffs );
  cam_model.setSizeScaleFactor(model_opt.scale_factor);

  ds.setCamera(cam_model);

  vector< string > bkg_images_names;
  if( bkg_images_dir.size() )
  {
    cout << "Loading random background images from "<<bkg_images_dir<< endl;
    if( !readFileNamesFromFolder (bkg_images_dir, bkg_images_names ) )
      cerr << "Can't loading random background images"<< endl;
  }

  RasterObjectModel3DPtr obj_model_ptr = std::make_shared<RasterObjectModel3D>();
  RasterObjectModel3D &obj_model = *obj_model_ptr;

  

  double obj_model_scale = 1.0;

  std::transform(model_opt.unit.begin(), model_opt.unit.end(), model_opt.unit.begin(), ::tolower);
  if ( !model_opt.unit.compare("m") )
  {
    obj_model.setUnitOfMeasure(RasterObjectModel::METER);
    obj_model_scale = 1.0;
    distance_ratio_transform = 1.0;
  }
  else if ( !model_opt.unit.compare("cm") )
  {
    obj_model.setUnitOfMeasure(RasterObjectModel::CENTIMETER);
    obj_model_scale = 0.01;
    distance_ratio_transform = 0.01;
  }
  else if ( !model_opt.unit.compare("mm") )
  {
    obj_model.setUnitOfMeasure(RasterObjectModel::MILLIMETER);
    obj_model_scale = 0.001;
    distance_ratio_transform = 0.001;
  }

  obj_model.setCamModel( cam_model );
  obj_model.setStepMeters(0.01); // TODO Check here!
  obj_model.setRenderZNear(0.5*ss_opt.min_dist * distance_ratio_transform);
  obj_model.setRenderZFar(1.5*ss_opt.max_dist * distance_ratio_transform);


  // TODO fix here!
  obj_model.enableUniformColor(model_color);
  obj_model.requestRenderLighting();

  if(!obj_model.setModelFile(model_opt.model_filename ))
    return -1;

  // obj_model.setMinSegmentsLen(0.01);
  obj_model.computeRaster();

  cv_ext::Box3f bb = obj_model.getBoundingBox();
  double diameter = sqrt(bb.width*bb.width + bb.height*bb.height + bb.depth*bb.depth );
  ds.setModel(model_opt.model_filename, obj_id, diameter, obj_model_scale );

  std::vector < cv::Point3f > view_pts;
  cv_ext::createIcosphere(view_pts, ss_opt.rotation_sampling_level );

  std::transform(ss_opt.vert_axis.begin(), ss_opt.vert_axis.end(), ss_opt.vert_axis.begin(), ::tolower);

  Eigen::Vector3d ref_y_axis;
  int coord_idx;
  if(!ss_opt.vert_axis.compare("x"))
  {
    ref_y_axis = Eigen::Vector3d(1.0, 0.0, 0.0);
    coord_idx = 0;
  }
  else if(!ss_opt.vert_axis.compare("y"))
  {
    ref_y_axis = Eigen::Vector3d(0.0, 1.0, 0.0);
    coord_idx = 1;
  }
  else
  {
    ref_y_axis = Eigen::Vector3d(0.0, 0.0, 1.0);
    coord_idx = 2;
  }

  std::cout<<"Generating views..."<<std::endl;
  boost::progress_display show_progress( view_pts.size() );
  for( auto &p : view_pts )
  {
    float vert_coord = *(reinterpret_cast<float *>(&p) + coord_idx);
    if( vert_coord > 0 )
    {
      Eigen::Vector3d z_axis( -p.x, -p.y, -p.z );
      auto x_axis = z_axis.cross(ref_y_axis);
      x_axis /= x_axis.norm();
      auto y_axis = z_axis.cross(x_axis);
      y_axis /= y_axis.norm();

      Eigen::Matrix3d rot_mat;
      rot_mat.col(0) = x_axis;
      rot_mat.col(1) = y_axis;
      rot_mat.col(2) = z_axis;

      rot_mat.transposeInPlace();

      Eigen::Vector3d t_vec;
      Eigen::Vector3d t_vec_scaled;
      t_vec(0) = 0;
      t_vec_scaled(0) = 0;
      
      for( t_vec(2) = ss_opt.min_dist; t_vec(2) <= ss_opt.max_dist; t_vec(2) += ss_opt.dist_step )
      {
        for( t_vec(1) = ss_opt.min_height ; t_vec(1) <= ss_opt.max_height; t_vec(1) += ss_opt.height_step )
        {
          for( t_vec(0) = ss_opt.min_soff ; t_vec(0) <= ss_opt.max_soff; t_vec(0) += ss_opt.soff_step )
          {
            t_vec_scaled = t_vec * distance_ratio_transform;

            model_color[0] = 255.0 * (static_cast<double>(rand()) / RAND_MAX);
            model_color[1] = 255.0 * (static_cast<double>(rand()) / RAND_MAX);
            model_color[2] = 255.0 * (static_cast<double>(rand()) / RAND_MAX);

            if (!model_color[0] && !model_color[1] && !model_color[2])
              model_color = cv::Scalar(128, 128, 128);

            obj_model.setVerticesColor(model_color);

            point_light_pos.x = rand() - RAND_MAX / 2;
            point_light_pos.y = rand() - RAND_MAX / 2;
            point_light_pos.z = rand() - RAND_MAX / 2;

            if (point_light_pos.x || point_light_pos.y || point_light_pos.z)
              point_light_pos /= cv_ext::norm3D(point_light_pos);
            else
              point_light_pos = cv::Point3f(1, 0, 0);

            point_light_pos *= diameter;

            light_dir.x = rand() - RAND_MAX / 2;
            light_dir.y = rand() - RAND_MAX / 2;
            light_dir.z = rand() - RAND_MAX / 2;

            if (light_dir.x || light_dir.y || light_dir.z)
              light_dir /= cv_ext::norm3D(light_dir);
            else
              light_dir = cv::Point3f(0, 0, -1);

            obj_model.setPointLightPos(point_light_pos);
            obj_model.setLightDirection(light_dir);

            obj_model.setModelView(rot_mat, t_vec_scaled);
            cv::Mat render_img, bkg;
            bkg = setRenderedImage(obj_model, cam_model ,bkg_images_names, render_img);
            
            if(stereo) {
              ds.addViewL(rot_mat, t_vec, render_img, obj_model.getMask());
              t_vec_scaled(0) += baseline;
              obj_model.setModelView(rot_mat, t_vec_scaled);
              // use same background for stereo pair
              if(bkg.empty())
                render_img = obj_model.getRenderedModel();
              else
                render_img = obj_model.getRenderedModel(bkg);

              ds.prepare_for_second_stereo_image();
              ds.addViewR(rot_mat, t_vec, render_img, obj_model.getMask());
            } else {
              ds.addView(rot_mat, t_vec, render_img, obj_model.getMask());
            }

            //cv::imshow("render",render_img);
            //cv::waitKey(0);
          }
        }
      }
    }
    ++show_progress;
  }

  std::cout<<std::endl<<"Prepare the dataset for PvNet..."<<std::endl;
  CPyInstance py_env(pvnet_home);
  CPyObject py_module = PyImport_ImportModule("tools.handle_custom_dataset");

  if(py_module)
  {
    CPyObject p_prepare_dataset;
    CPyObject p_arg;
    if (stereo) {
      p_prepare_dataset = PyObject_GetAttrString(py_module, "prepare_dataset_stereo");
      p_arg = PyTuple_New(2);
      PyTuple_SetItem(p_arg, 0, PyUnicode_FromString(dataset_dir.c_str()));
      PyTuple_SetItem(p_arg, 1, PyFloat_FromDouble(baseline));
    } else {
      p_prepare_dataset = PyObject_GetAttrString(py_module, "prepare_dataset");
      p_arg = PyTuple_New(1);
      PyTuple_SetItem(p_arg, 0, PyUnicode_FromString(dataset_dir.c_str()));
    }

    if( !p_prepare_dataset || !PyCallable_Check(p_prepare_dataset) ||
        !PyObject_CallObject(p_prepare_dataset, p_arg) )
    {
      std::cerr<<"Can't call Python function"<<std::endl;
      return -1;
    }
  }
  else
  {
    std::cerr<<"Python module not imported"<<std::endl;
    PyErr_Print();
    return -1;
  }

  std::cout<<"Dataset generated!"<<std::endl;

  return 0;
}