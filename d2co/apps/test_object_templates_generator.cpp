#include <iostream>
#include <sstream>
#include <string>
#include <ctime>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/iterator/iterator_concepts.hpp>
#include <boost/tokenizer.hpp>
#include <boost/range/iterator_range.hpp>

#include <opencv2/opencv.hpp>

#include "cv_ext/cv_ext.h"
#include "raster_object_model3D.h"
#include "object_templates_generator.h"
#include "views_generator.h"
#include "apps_utils.h"

using namespace std;
using namespace cv;
using namespace boost;
namespace po = boost::program_options;
using namespace boost::filesystem;


int main ( int argc, char **argv )
{
  string app_name ( argv[0] ), model_filename, imgs_folder_name[2], camera_filename[2], stereo_filename;

  double scale_factor = 1.0;

  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "model_filename,m", po::value<string > ( &model_filename )->required(),
    "STL, PLY, OBJ, ...  model file" )
  ( "scale_factor,s", po::value<double> ( &scale_factor ),
    "Scale factor [1]" )
  ( "f0", po::value<string > ( &imgs_folder_name[0] )->required(),
    "Camera 0 input images folder path" )
  ( "f1", po::value<string > ( &imgs_folder_name[1] )->required(),
    "Camera 1 input images folder path" )
  ( "c0", po::value<string > ( &camera_filename[0] )->required(),
    "Camera 0 model filename" )
  ( "c1", po::value<string > ( &camera_filename[1] )->required(),
    "Camera 1 model filename" )
  ( "sc", po::value<string > ( &stereo_filename )->required(),
    "Stereo camera extrinsics filename" );

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

  cout << "Loading model from file : "<<model_filename<< endl;
  for ( int k = 0; k < 2; k++ )
    cout << "Loading camera "<<k<<" images from folder : "<<imgs_folder_name[k]<< endl;
  for ( int k = 0; k < 2; k++ )
    cout << "Loading camera "<<k<<" model from file : "<<camera_filename[k]<< endl;
  cout << "Loading stereo camera extrinsics from file : "<<stereo_filename<< endl;

  vector<string> filelist[2];
  if ( !readFileNamesFromFolder ( imgs_folder_name[0], filelist[0] ) ||
       !readFileNamesFromFolder ( imgs_folder_name[1], filelist[1] ) )
  {
    cerr<<"Wrong or empty folders"<<endl;
    exit ( EXIT_FAILURE );
  }

  if ( filelist[0].size() != filelist[1].size() )
  {
    cerr<<"Images folder should contain the same number of images, in the same order"<<endl;
    exit ( EXIT_FAILURE );
  }

  std::vector < cv_ext::PinholeCameraModel > cam_models(2);
  for ( int k = 0; k < 2; k++ )
  {
    cam_models[k].readFromFile ( camera_filename[k] );
    cam_models[k].setSizeScaleFactor ( scale_factor );
  }

  cv::Mat stereo_r_mat, stereo_t_vec;

  cv_ext::read3DTransf ( stereo_filename, stereo_r_mat, stereo_t_vec, cv_ext::ROT_FORMAT_MATRIX );

  cv_ext::StereoRectification stereo_rect;
  stereo_rect.setCameraParameters ( cam_models, stereo_r_mat, stereo_t_vec );
  stereo_rect.update();

  auto rect_cam_models = stereo_rect.getCamModels();
  cv::Point2f cam_shift = stereo_rect.getCamDisplacement();

  RasterObjectModel3DPtr obj_models[2];

  for ( int k = 0; k < 2; k++ )
  {
    cout<<rect_cam_models[k].cameraMatrix()<<endl;
    cout<<rect_cam_models[k].hasDistCoeff()<<endl;
    obj_models[k] = make_shared<RasterObjectModel3D>();
    obj_models[k]->setCamModel ( rect_cam_models[k] );
    obj_models[k]->setStepMeters ( 0.001 );
    obj_models[k]->setUnitOfMeasure ( RasterObjectModel::MILLIMETER );

    if ( !obj_models[k]->setModelFile ( model_filename ) )
    {
      cerr<<"Can't load object model"<<endl;
      exit ( EXIT_FAILURE );
    }
    // obj_model.setMinSegmentsLen(0.01);
    obj_models[k]->computeRaster();
  }


  ViewsGenerator vg;
  cv_ext::IntervalD ang_range, z_range(0.86, 0.94);

  vg.addDepthRange(z_range);

  vg.setAngStep(M_PI/36);
  vg.setDepthStep(0.02);

  ang_range.start = M_PI - M_PI/18;
  ang_range.end = M_PI;
  vg.addRollPitchRange(ang_range);

//  ang_range.start = 5*M_PI/5;
//  ang_range.end = M_PI;
//  vg.addRollPitchRange(ang_range);

  ang_range.start = 0;
  ang_range.end = M_PI/2;
  vg.addYawRange(ang_range);

  cv_ext::vector_Quaterniond views_r_quats;
  cv_ext::vector_Vector3d views_t_vecs;
  vg.generate(views_r_quats, views_t_vecs);

  cout<<"Templaes : "<<views_r_quats.size()<<endl;

  ObjectTemplateGenerator<DirIdxPointSet> otg;

  otg.enableVerboseMode(true);
  otg.setMaxNumImgPoints(512);
  otg.setImgPtsNumDirections(60);
  otg.setImgPointsSpacing(1);
  otg.setTemplateModel(obj_models[0]);

  DirIdxPointSetVec templates;

  otg.generate(templates, 0, views_r_quats, views_t_vecs);

  saveTemplateVector( string("templates.bin"), templates );

#if TEST_TEMPLATES_SERIALIZATION
  DirIdxPointSetVec test_templates;
  loadTemplateVector( string("templates.bin"), test_templates );

  if( templates.size() != test_templates.size() )
    cout<<"Size!"<<templates.size()<<" "<<test_templates.size()<<endl;

  for( int i = 0; i < templates.size(); i++ )
  {
    if(templates[i].class_id != templates[i].class_id)
      cout<<"class_id [i] = "<<i<<endl;
    if(templates[i].obj_r_quat.w() != templates[i].obj_r_quat.w() ||
       templates[i].obj_r_quat.x() != templates[i].obj_r_quat.x() ||
       templates[i].obj_r_quat.y() != templates[i].obj_r_quat.y() ||
       templates[i].obj_r_quat.z() != templates[i].obj_r_quat.z()  )
      cout<<"obj_r_quat [i] = "<<i<<endl;
    if(templates[i].obj_t_vec != templates[i].obj_t_vec)
      cout<<"obj_t_vec [i] = "<<i<<endl;
    if(templates[i].obj_bbox3d != templates[i].obj_bbox3d)
      cout<<"obj_bbox3d [i] = "<<i<<endl;
    if(templates[i].bbox != templates[i].bbox)
      cout<<"bbox [i] = "<<i<<endl;
    if(templates[i].proj_obj_bbox3d.size() != test_templates[i].proj_obj_bbox3d.size())
      cout<<"proj_obj_bbox3d size [i] = "<<i<<endl;
    for(int j = 0; j < templates[i].proj_obj_bbox3d.size(); j++)
    {
      if(templates[i].proj_obj_bbox3d[j] != test_templates[i].proj_obj_bbox3d[j])
      {
        cout<<"proj_obj_bbox3d[i] = "<<i<<endl;
        break;
      }
    }

    if(templates[i].obj_pts.size() != test_templates[i].obj_pts.size())
      cout<<"obj_pts size [i] = "<<i<<endl;
    for(int j = 0; j < templates[i].obj_pts.size(); j++)
    {
      if(templates[i].obj_pts[j] != test_templates[i].obj_pts[j])
      {
        cout<<"obj_pts[i] = "<<i<<endl;
        break;
      }
    }
    if(templates[i].obj_d_pts.size() != test_templates[i].obj_d_pts.size())
      cout<<"obj_d_pts size [i] = "<<i<<endl;
    for(int j = 0; j < templates[i].obj_d_pts.size(); j++)
    {
      if(templates[i].obj_d_pts[j] != test_templates[i].obj_d_pts[j])
      {
        cout<<"obj_d_pts[i] = "<<i<<endl;
        break;
      }
    }


    if(templates[i].dir_idx.size() != test_templates[i].dir_idx.size())
      cout<<"dir_idx size [i] = "<<i<<endl;
    for(int j = 0; j < templates[i].dir_idx.size(); j++)
    {
      if(templates[i].dir_idx[j] != test_templates[i].dir_idx[j])
      {
        cout<<"dir_idx[i] = "<<i<<endl;
        break;
      }
    }

    if(templates[i].proj_pts.size() != test_templates[i].proj_pts.size())
      cout<<"proj_pts size [i] = "<<i<<endl;
    for(int j = 0; j < templates[i].proj_pts.size(); j++)
    {
      if(templates[i].proj_pts[j] != test_templates[i].proj_pts[j])
      {
        cout<<"proj_pts[i] = "<<i<<endl;
        break;
      }
    }
  }

#endif


  Size img_size = cam_models[0].imgSize();
  Mat img_pair ( Size ( 2*img_size.width, img_size.height ), cv::DataType<Vec3b>::type ), background_img;
  std::vector< Mat > src_img(2), rect_img(2);
  int start_col = 0, w = img_size.width;
  for ( int k = 0; k < 2; k++, start_col += w )
    rect_img[k] = img_pair.colRange ( start_col, start_col + w );

  vector<Point2f> orig_raster_pts[2], cur_raster_pts[2];
  Eigen::Quaterniond r_quat;
  Eigen::Vector3d t_vec[2];

  int ti = 0;
  for ( int i = 0; i < int ( filelist[0].size() ); i++ )
  {
    for ( int k = 0; k < 2; k++ )
    {
      src_img[k] = cv::imread ( filelist[k][i] );
      cv::resize ( src_img[k],src_img[k],img_size );
      std::cout<<filelist[k][i]<<endl;
    }
    stereo_rect.rectifyImagePair ( src_img, rect_img );
    background_img = img_pair.clone();

    int step = 5;
    cv::Point offset(0,0);

    bool exit_now = false, slide = false;
    while ( !exit_now )
    {
      background_img.copyTo ( img_pair );
            
      if ( slide )
      {
        slide = false;

        cv_ext::BasicTimer timer;

        ObjectTemplatePnP o_pnp(rect_cam_models[0]);
        o_pnp.fixZTranslation(true);
        o_pnp.solve( templates[ti], offset, r_quat, t_vec[0] );
        
        cout<<"Elapesed time : "<<timer.elapsedTimeMs()<<endl;

        t_vec[1] = t_vec[0];
        t_vec[1](0) += cam_shift.x;
        t_vec[1](1) += cam_shift.y;
        
        cout<<"---> Estiated "<<t_vec[0].transpose()<<endl;
        
        for ( int k = 0; k < 2; k++ )
        {
          cur_raster_pts[k].clear();

          obj_models[k]->setModelView ( r_quat, t_vec[k] );
          obj_models[k]->projectRasterPoints ( cur_raster_pts[k] );

          cv_ext::drawPoints ( rect_img[k], orig_raster_pts[k] );
          cv_ext::drawPoints ( rect_img[k], cur_raster_pts[k], cv::Scalar ( 0,0,255 ) );
        }
      }
      else
      {
        r_quat = templates[ti].obj_r_quat;

        t_vec[0] = templates[ti].obj_t_vec;
        t_vec[1] = t_vec[0];
        t_vec[1](0) += cam_shift.x;
        t_vec[1](1) += cam_shift.y;

        for ( int k = 0; k < 2; k++ )
        {
          obj_models[k]->setModelView ( r_quat, t_vec[k] );
          obj_models[k]->projectRasterPoints ( orig_raster_pts[k] );
          cv_ext::drawPoints ( rect_img[k], orig_raster_pts[k] );
        }

        cout<<"Orig "<<templates[ti].obj_t_vec.transpose()<<endl;
        cv_ext::drawPoints ( rect_img[0], templates[ti].proj_pts, cv::Scalar ( 0,0,255 ) );

      }

      imshow ( "img_pair", img_pair );
      int key = cv_ext::waitKeyboard();

      switch ( key )
      {

      case '4':
      case 81:
        slide = true;
        offset.x -= step;
        for ( auto &pts:orig_raster_pts )
          for ( auto &p:pts )
            p.x -= step;
        break;
      case '6':
      case 83:
        slide = true;
        offset.x += step;
        for ( auto &pts:orig_raster_pts )
          for ( auto &p:pts )
            p.x += step;
        break;
      case '8':
      case 82:
        slide = true;
        offset.y -= step;
        for ( auto &pts:orig_raster_pts )
          for ( auto &p:pts )
            p.y -= step;
        break;
      case '2':
      case 84:
        slide = true;
        offset.y += step;
        for ( auto &pts:orig_raster_pts )
          for ( auto &p:pts )
            p.y += step;
        break;
      case cv_ext::KEY_ESCAPE:
        exit_now = true;
        break;
      default:
        offset = cv::Point(0,0);
        if(++ti == static_cast<int>(templates.size()))
          return 0;
        break;
      }

    }
  }

  return 0;
}
