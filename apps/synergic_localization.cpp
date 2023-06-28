#include <string>
#include <sstream>
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "cv_ext/conversions.h"
#include "io_utils.h"
#include "apps_utils.h"
#include "pvnet_dataset.h"
#include "metrics.h"

#include "tm_object_localization.h"
#include "pvnet_wrapper.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace std;
using namespace cv;
using namespace cv_ext;

void showGT( RasterObjectModel &obj_model, const cv::Mat &src_img, const Eigen::Matrix3d &gt_r_mat, const Eigen::Vector3d &gt_t_vec)
{
  cv::Mat draw_img;
  if( obj_model.cameraModel().sizeScaleFactor() != 1 )
    cv::resize(src_img, draw_img, obj_model.cameraModel().imgSize());
  else
    draw_img = src_img.clone();

  vector<Point2f> gt_proj_pts;
  obj_model.setModelView(gt_r_mat, gt_t_vec);
  obj_model.projectRasterPoints(gt_proj_pts);
  cv_ext::drawPoints(draw_img, gt_proj_pts, Scalar(0, 0, 255));
  cv_ext::showImage(draw_img, "Ground truth", true, 10);
}

int main(int argc, char **argv)
{
  string app_name( argv[0] ), imgs_folder, gt_fmt = "NONE", gt_folder;
  bool display = false;

  po::options_description evaluation_options ( "Evaluation Options" );
  evaluation_options.add_options()
  ( "imgs_folder,f", po::value<string > ( &imgs_folder )->required(),
    "Input images folder path" )
  ( "gt_fmt", po::value<string> ( &gt_fmt ),
    "Ground truth format, used for performance evaluation  [NONE (default, i.e., no ground truth provided), PVNET, CUSTOM]" )
  ( "gt_folder", po::value<string> ( &gt_folder ),
    "Ground truth folder [default: "", i.e., same folder used for images]" )

  ( "display", "Optionally display localization results" );


//  ( "mcs", po::value<int> ( &match_cell_size ),
//      "Size in pixels of each matching cell (if -1, just search for the best matches for the whoel image).[default: 100]" )

  po::options_description program_options;

  CADCameraOptions model_opt;
  DefaultOptions def_opt;
  BBOffsetOptions bbo_opt;
  SynLocOptions sl_opt;

  program_options.add(model_opt.getDescription());
  program_options.add(sl_opt.getDescription());
  program_options.add(evaluation_options);
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

    po::notify ( vm );
  }

  catch ( boost::program_options::required_option& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    cout << "USAGE: "<<app_name<<" OPTIONS"
    << endl << endl<<program_options;
    exit(EXIT_FAILURE);
  }

  catch ( boost::program_options::error& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    cout << "USAGE: "<<app_name<<" OPTIONS"
    << endl << endl<<program_options;
    exit(EXIT_FAILURE);
  }


  model_opt.print();
  sl_opt.print();

  std::cout<<"Loading images from folder :"<<imgs_folder<<std::endl;

  if( gt_fmt.compare("NONE") )
  {
    if( gt_folder.empty() )
      gt_folder = imgs_folder;
    cout << "Ground truth format : " << gt_fmt << endl;
    cout << "Ground truth folder : " << gt_folder << endl;
  }

  bbo_opt.print();
  def_opt.print();

  TMObjectLocalization obj_loc;

  obj_loc.setNumMatches(5);
  obj_loc.setScaleFactor(model_opt.scale_factor);
  obj_loc.setCannyLowThreshold(40);
  obj_loc.setScoreThreshold(sl_opt.score_threshold);
  obj_loc.setModelSaplingStep(sl_opt.model_samplig_step);

//  if( top_boundary != -1 || bottom_boundary != -1 ||
//      left_boundary != -1 || rigth_boundary != -1 )
//  {
//    Point tl(0,0), br(scaled_img_size.width, scaled_img_size.height);
//
//    if( top_boundary != -1 ) { tl.y = top_boundary; }
//    if( left_boundary != -1 ) { tl.x = left_boundary;  }
//    if( bottom_boundary != -1 ) { br.y = bottom_boundary; }
//    if( rigth_boundary != -1 ) { br.x = rigth_boundary; }
//
//    obl_loc.setRegionOfInterest( cv::Rect(tl, br) );
//  }

  if ( !model_opt.unit.compare("m") )
    obj_loc.setUnitOfMeasure(RasterObjectModel::METER);
  else if ( !model_opt.unit.compare("cm") )
    obj_loc.setUnitOfMeasure(RasterObjectModel::CENTIMETER);
  else if ( !model_opt.unit.compare("mm") )
    obj_loc.setUnitOfMeasure(RasterObjectModel::MILLIMETER);

  // TODO
//  cout << "Bounding box origin offset [off_x, off_y, off_z] : ["
//       <<bb_xoff<<", "<<bb_yoff<<", "<<bb_zoff<<"]"<< endl;
//  cout << "Bounding box size offset [off_width, off_height, off_depth] : ["
//       <<bb_woff<<", "<<bb_hoff<<", "<<bb_doff<<"]"<< endl;
//
//  if( bb_xoff ||  bb_yoff || bb_zoff ||
//      bb_woff || bb_hoff || bb_doff )
//    obl_loc.setBoundingBoxOffset( bb_xoff, bb_yoff, bb_zoff,
//                                  bb_woff, bb_hoff, bb_doff );


  obj_loc.enableDisplay(display);
  obj_loc.initialize(model_opt.camera_filename, model_opt.model_filename, sl_opt.templates_filename );

  PVNetWrapper pv_net_wrapper(sl_opt.pvnet_home);
  pv_net_wrapper.registerObject(0,sl_opt.pvnet_model, sl_opt.pvnet_inference_meta);

  cv_ext::Box3f bb = obj_loc.objectModel()->getBoundingBox();
  double diameter = sqrt(bb.width*bb.width + bb.height*bb.height + bb.depth*bb.depth );

  MaxRtErrorsMetric max_rt_metric;
  Projection2DMetric proj_2d_metric( obj_loc.objectModel()->cameraModel(), obj_loc.objectModel()->vertices() );
  Pose6DMetric pose_6d_metric( obj_loc.objectModel()->cameraModel(), obj_loc.objectModel()->vertices(), diameter );

  double good_max_rt = 0, good_proj_2d = 0, good_pose_6d = 0;


  vector<string> filelist;
  if ( !readFileNamesFromFolder ( imgs_folder, filelist ) )
  {
    cerr<<"Wrong or empty folders"<<endl;
    exit ( EXIT_FAILURE );
  }

  cv_ext::BasicTimer timer;
  cv::Mat src_img, pvnet_img;
  int num_images = 0;
  for ( int i = 0; i < static_cast<int> ( filelist.size() ); i++ )
  {
    timer.reset();
    src_img = cv::imread ( filelist[i] );
    if(src_img.empty())
      continue;
    std::cout<<"Loaded image : "<<filelist[i]<<endl;

    cv::cvtColor(src_img, pvnet_img, cv::COLOR_BGR2RGB);

    num_images++;

    cv::Mat_<double> r_mat, t_vec, r_vec(3,1);
    pv_net_wrapper.localize(pvnet_img,0, r_mat, t_vec);

    cv_ext::rotMat2AngleAxis<double>(r_mat, r_vec);
    obj_loc.refine(src_img, r_vec, t_vec);


    if( !gt_fmt.compare("PVNET") )
    {
      cv_ext::angleAxis2RotMat<double>(r_vec, r_mat );

      Eigen::Matrix3d eig_r_mat;
      Eigen::Vector3d eig_t_vec;

      cv_ext::openCv2Eigen(r_mat,eig_r_mat);
      cv_ext::openCv2Eigen(t_vec,eig_t_vec);

      fs::path img_path(fs::path(filelist[i]).filename().replace_extension("npy"));
      std::string gt_name = std::string("pose") + img_path.string();
      fs::path gt_path = fs::path(gt_folder)/(fs::path(gt_name));

      Eigen::Matrix3d gt_r_mat;
      Eigen::Vector3d gt_t_vec;

      loadPVNetPose(gt_path.string(), gt_r_mat, gt_t_vec );

      Eigen::Quaterniond r_quat(eig_r_mat), gt_r_quat(gt_r_mat);

      max_rt_metric.setGroundTruth(gt_r_quat, gt_t_vec);
      proj_2d_metric.setGroundTruth(gt_r_quat, gt_t_vec);
      pose_6d_metric.setGroundTruth(gt_r_quat, gt_t_vec);

      if(max_rt_metric.performTest(r_quat, eig_t_vec))
      {
        good_max_rt++;
        std::cout<<"good_max_rt : "<<good_max_rt<<" over "<<num_images<<std::endl;
      }

      if(proj_2d_metric.performTest(r_quat, eig_t_vec))
      {
        good_proj_2d++;
        std::cout<<"good_proj_2d : "<<good_proj_2d<<" over "<<num_images<<std::endl;
      }

      if(pose_6d_metric.performTest(r_quat, eig_t_vec))
      {
        good_pose_6d++;
        std::cout<<"good_pose_6d : "<<good_pose_6d<<" over "<<num_images<<std::endl;
      }

//      showGT(*obj_loc.objectModel(), src_img, gt_r_mat, gt_t_vec);
//
//      fs::path pose_path(poses_path_);
//      pose_path /= "pose" + std::to_string(num_views_) + ".npy";
    }
//    else if( !gt_fmt.compare("CUSTOM") )
//    {
//      fs::path img_path(filelist[i]);
//      fs::path gt_path = fs::path(gt_folder)/img_path.filename().replace_extension("yml");
//
//      cv::Mat gt_r_vec, gt_r_mat, gt_t_vec, gt_pivot;
//      cv::FileStorage fs(gt_path.string(), cv::FileStorage::READ);
//
//      if( fs.isOpened() )
//      {
//        fs["rvec"] >> gt_r_vec;
//        fs["tvec"] >> gt_t_vec;
//        fs["Pivot"] >> gt_pivot;
//
//        cv_ext::angleAxis2RotMat<float>(gt_r_vec,gt_r_mat);
//
//        cv::Mat_<float> inv_p_x = cv::Mat_<float>::zeros(cv::Size(3,3)),
//                        inv_p_y = cv::Mat_<float>::zeros(cv::Size(3,3)),
//                        inv_p_z = cv::Mat_<float>::zeros(cv::Size(3,3));
//        //X
//        inv_p_x(0,0) = 1;
//        inv_p_x(1,2) = -1;
//        inv_p_x(2,1) = 1;
////Y
//        inv_p_y(0,2) = 1;
//        inv_p_y(1,1) = 1;
//        inv_p_y(2,0) = -1;
////Z
//        inv_p_z(0,1) = -1;
//        inv_p_z(1,0) = 1;
//        inv_p_z(2,2) = 1;
//
////        gt_t_vec = gt_r_mat*gt_pivot.rowRange(0,3).colRange(3,4) + gt_t_vec;
////        gt_r_mat = gt_r_mat.clone()*gt_pivot.rowRange(0,3).colRange(0,3)*inv_p;
////          gt_t_vec = gt_r_mat*-gt_pivot.rowRange(0,3).colRange(0,3).inv()*gt_pivot.rowRange(0,3).colRange(3,4) + gt_t_vec;
////          gt_r_mat = gt_r_mat*gt_pivot.rowRange(0,3).colRange(0,3).inv()*inv_p_z.inv()*inv_p_x.inv();
//
////        gt_r_mat = gt_pivot.rowRange(0,3).colRange(0,3)*gt_r_mat;
////        gt_t_vec = gt_pivot.rowRange(0,3).colRange(0,3)*gt_t_vec + gt_pivot.rowRange(0,3).colRange(3,4);
//
//        cv_ext::rotMat2AngleAxis<float>(gt_r_mat,gt_r_vec);
//
//        showGT(*obj_loc.objectModel(), src_img, gt_r_mat, gt_t_vec);
//      }
//    }

//    timer.reset();
//    obl_loc.localize(src_img );
    cout << "Object localization ms: " << timer.elapsedTimeMs() << endl;
  }

  std::cout<<"Max RT metric : "<<good_max_rt/num_images<<std::endl;
  std::cout<<"Proj 2D metric : "<<good_proj_2d/num_images<<std::endl;
  std::cout<<"Pose 6D metric : "<<good_pose_6d/num_images<<std::endl;

  return EXIT_SUCCESS;
}
