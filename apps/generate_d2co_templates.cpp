#include <string>
#include <sstream>
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>

#include "cv_ext/cv_ext.h"
#include "raster_object_model3D.h"
#include "object_templates_generator.h"

#include "io_utils.h"
#include "apps_utils.h"

namespace po = boost::program_options;
using namespace std;
using namespace cv;

int main( int argc, char **argv )
{
  srand(0);

  string app_name( argv[0] ), templates_filename;

  int num_template_pts = 256, num_template_dirs = 60;
  bool concatenate = false, display = false;

  po::options_description templates_options ("Templates Generator Options" );
  templates_options.add_options()
      ( "templates_filename,t", po::value<string > ( &templates_filename)->required(),
        "Output model templates file (also used as input file if the \"concatenate\" option is set)" )
      ( "concatenate", "If required, concatenate new templates to templates loaded from an input model templates file" )
      ( "num_template_pts,n", po::value<int > ( &num_template_pts ),
        "Maximum number of edge points sampled for each template  (default: 256)" )

      ( "display", "Optionally display the generated templates at the end of the process" );

  po::options_description program_options;

  CADCameraOptions model_opt;
  DefaultOptions def_opt;
  SpaceSamplingOptions ss_opt;
  BBOffsetOptions bbo_opt;

  program_options.add(model_opt.getDescription());
  program_options.add(templates_options);
  program_options.add(ss_opt.getDescription());
  program_options.add(bbo_opt.getDescription());
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

    if ( vm.count ( "display" ) )
      display = true;

    if ( vm.count ( "concatenate" ) )
      concatenate = true;

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
  cout << "Input/Output model templates file : "<<templates_filename<< endl;
  cout << "Maximum number of edge points sampled for each template : "<<num_template_pts<< endl;
  if(concatenate)
    cout << "New templates will be concatenated to templates loaded from the templates file"<< endl;
  ss_opt.print();
  bbo_opt.print();
  def_opt.print();

  cv::Size image_size;
  cv::Mat camera_matrix, dist_coeffs;
  if( !loadCameraParams(model_opt.camera_filename, image_size, camera_matrix, dist_coeffs ) )
  {
    cout << "Error loading camera filename, exiting"<< endl;
    return -1;
  }

  cv_ext::PinholeCameraModel cam_model(camera_matrix, image_size.width, image_size.height, dist_coeffs );
  cam_model.setSizeScaleFactor(model_opt.scale_factor);

  RasterObjectModel3DPtr obj_model_ptr = std::make_shared<RasterObjectModel3D>();
  RasterObjectModel3D &obj_model = *obj_model_ptr;

  obj_model.setCamModel( cam_model );
  obj_model.setStepMeters(0.01); // TODO Check here!
  obj_model.setRenderZNear(0.5*ss_opt.min_dist);
  obj_model.setRenderZFar(1.5*ss_opt.max_dist);

  std::transform(model_opt.unit.begin(), model_opt.unit.end(), model_opt.unit.begin(), ::tolower);
  if ( !model_opt.unit.compare("m") )
    obj_model.setUnitOfMeasure(RasterObjectModel::METER);
  else if ( !model_opt.unit.compare("cm") )
    obj_model.setUnitOfMeasure(RasterObjectModel::CENTIMETER);
  else if ( !model_opt.unit.compare("mm") )
    obj_model.setUnitOfMeasure(RasterObjectModel::MILLIMETER);

  if(!obj_model.setModelFile(model_opt.model_filename ))
    return -1;

  // obj_model.setMinSegmentsLen(0.01);
  obj_model.computeRaster();


  if( bbo_opt.bb_xoff ||  bbo_opt.bb_yoff || bbo_opt.bb_zoff ||
      bbo_opt.bb_woff || bbo_opt.bb_hoff || bbo_opt.bb_doff )
  {
    auto cur_bb = obj_model.getBoundingBox();

    cur_bb.x += bbo_opt.bb_xoff;
    cur_bb.width += bbo_opt.bb_woff - bbo_opt.bb_xoff;
    cur_bb.y += bbo_opt.bb_yoff;
    cur_bb.height += bbo_opt.bb_hoff - bbo_opt.bb_yoff;
    cur_bb.z += bbo_opt.bb_zoff;
    cur_bb.depth += bbo_opt.bb_doff - bbo_opt.bb_zoff;

    obj_model.setBoundingBox(cur_bb);
  }

  DirIdxPointSetVec model_templates;
  if( concatenate )
  {
    cout << "Concatenate new templates to templates loaded from "<<templates_filename<<" (if exists)"<< endl;
    loadTemplateVector( templates_filename, model_templates );
  }

  ObjectTemplateGenerator<DirIdxPointSet> otg;

  otg.enableVerboseMode(true);
  otg.setMaxNumImgPoints(num_template_pts);
  otg.setImgPtsNumDirections(num_template_dirs);
  otg.setImgPointsSpacing(1);
  otg.setTemplateModel(obj_model_ptr);


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
    if( vert_coord > 0 && vert_coord < 0.6 )
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

      for( t_vec(2) = ss_opt.min_dist; t_vec(2) <= ss_opt.max_dist; t_vec(2) += ss_opt.dist_step )
      {
        for( t_vec(1) = ss_opt.min_height ; t_vec(1) <= ss_opt.max_height; t_vec(1) += ss_opt.height_step )
        {
          for( t_vec(0) = ss_opt.min_soff ; t_vec(0) <= ss_opt.max_soff; t_vec(0) += ss_opt.soff_step )
          {
            Eigen::Quaterniond r_quat(rot_mat);
            DirIdxPointSet templ;
            otg.generate(templ, 0, r_quat, t_vec);
            model_templates.push_back(templ);
          }
        }
      }
    }
    ++show_progress;
  }

  // Save templates
  saveTemplateVector( templates_filename, model_templates );

  std::cout<<std::endl<<"Dataset generated!"<<std::endl;

  if( model_templates.size() && display )
  {
    std::cout<<"Displaying the dataset..."<<std::endl;
    int idx = 0;

    bool exit_now = false, view_template = false;
    Point text_org_guide1 (20, 20), text_org_guide2(20, 40), text_org_model (20, 70);
    Scalar text_color(0,0,255);

    Mat draw_img( Size(cam_model.imgWidth(),cam_model.imgHeight()), DataType<Vec3b>::type );
    vector<Vec4f> raster_segs;

    while( !exit_now )
    {
      stringstream sstr_guide1, sstr_guide2, sstr_model;
      sstr_guide1<<"Use [p] and [n] to switch to the previous and to the next templates, [ESC] to exit";
      sstr_guide2<<"[t] to swithch between wireframe and template visualization";
      sstr_model<<"View ";
      sstr_model<<idx + 1;
      sstr_model<<" of ";
      sstr_model<<model_templates.size();

      draw_img.setTo( cv::Scalar( 0,0,0) );

      putText(draw_img, sstr_guide1.str(), text_org_guide1, cv::FONT_HERSHEY_PLAIN, 1,
              text_color, 1, 8);
      putText(draw_img, sstr_guide2.str(), text_org_guide2, cv::FONT_HERSHEY_PLAIN, 1,
              text_color, 1, 8);
      putText(draw_img, sstr_model.str(), text_org_model, cv::FONT_HERSHEY_SIMPLEX, 1,
              text_color, 1, 8);

      if( view_template )
      {
        cv_ext::drawPoints(draw_img, model_templates[idx].proj_pts);
      }
      else
      {
        obj_model.setModelView(model_templates[idx].obj_r_quat, model_templates[idx].obj_t_vec);
        obj_model.projectRasterSegments( raster_segs );
        cv_ext::drawSegments( draw_img, raster_segs );
      }


      cv::imshow("Display", draw_img);
      int key = cv_ext::waitKeyboard();

      switch(key)
      {
        case 'n':
        case 'N':
          idx++;
          if( idx >= static_cast<int>(model_templates.size()) )
            idx = 0;
          break;
        case 'p':
        case 'P':
          idx--;
          if( idx < 0 )
            idx = model_templates.size() - 1;
          break;
        case 't':
        case 'T':
          view_template = !view_template;
          break;

          break;
        case cv_ext::KEY_ESCAPE:
          exit_now = true;
          break;
      }
    }
  }

  return 0;
}


