#include "cv_ext/cv_ext.h"

#include "raster_object_model3D.h"
#include "object_templates_generator.h"
#include "chamfer_matching.h"
#include "chamfer_registration.h"
#include "scoring.h"
#include "metrics.h"

#include "apps_utils.h"

#include <Eigen/Geometry>

#include <cstdio>
#include <string>
#include <sstream>
#include <map>
#include <boost/program_options.hpp>
#include <opencv2/rgbd.hpp>
#include "omp.h"


#define TEST_CM 0
#define TEST_OCM 0
#define TEST_DCM 1
#define TEST_LINE2D 0
#define TEST_LINE2D_GL 0


namespace po = boost::program_options;
using namespace std;
using namespace cv;
using namespace cv_ext;

struct Pose
{
  Pose(Mat r, Mat t, Rect b)
  {
    r_vec = r.clone();
    t_vec = t.clone();
    bb = b;
  };
  Pose(){};
  Mat r_vec, t_vec;
  Rect bb;
};

void drawResponse(const std::vector<cv::linemod::Template>& templates,
                  int num_modalities, cv::Mat& dst, cv::Point offset, int T)
{
  static const cv::Scalar COLORS[5] = { CV_RGB(0, 0, 255),
                                        CV_RGB(0, 255, 0),
                                        CV_RGB(255, 255, 0),
                                        CV_RGB(255, 140, 0),
                                        CV_RGB(255, 0, 0) };

  for (int m = 0; m < num_modalities; ++m)
  {
    cv::Scalar color = COLORS[m];

    for (int i = 0; i < (int)templates[m].features.size(); ++i)
    {
      cv::linemod::Feature f = templates[m].features[i];
      cv::Point pt(f.x + offset.x, f.y + offset.y);
      cv::circle(dst, pt, T / 2, color);
    }
  }
}


void loadLineModDataset( const string input_image_dir, vector<string> &image_names,
                         cv_ext::vector_Quaterniond& r_quats,
                         cv_ext::vector_Vector3d& t_vecs )
{
  Eigen::Matrix3d gt_rot;
  Eigen::Quaterniond r_quat;
  Eigen::Vector3d t_vec;
  
  int image_idx = 0;
  while( true )
  {
//     if(!(image_idx % 100))
//       cout<<"Loading data #"<<image_idx<<endl;

    std::string rot_name = cv::format("%s/rot%d.rot", input_image_dir.c_str(), image_idx),
                tra_name = cv::format("%s/tra%d.tra", input_image_dir.c_str(), image_idx);
    FILE *rot_file = fopen(rot_name.c_str(),"r"),
         *tra_file = fopen(tra_name.c_str(),"r");
    if( rot_file == NULL || tra_file == NULL )
      break;

    double val0,val1;
    CV_Assert(fscanf(rot_file,"%lf %lf", &val0,&val1) == 2);
    CV_Assert(fscanf(rot_file,"%lf %lf %lf", &gt_rot(0,0),&gt_rot(0,1),&gt_rot(0,2)) == 3);
    CV_Assert(fscanf(rot_file,"%lf %lf %lf", &gt_rot(1,0),&gt_rot(1,1),&gt_rot(1,2)) == 3);
    CV_Assert(fscanf(rot_file,"%lf %lf %lf", &gt_rot(2,0),&gt_rot(2,1),&gt_rot(2,2)) == 3);

    CV_Assert(fscanf(tra_file,"%lf %lf", &val0, &val1) == 2);
    CV_Assert(fscanf(tra_file,"%lf %lf %lf", &t_vec(0),&t_vec(1),&t_vec(2)) == 3);

    r_quat = gt_rot;
    t_vec /= 100.0;

    r_quats.push_back(r_quat);
    t_vecs.push_back(t_vec);

    fclose(rot_file);
    fclose(tra_file);

    std::string image_name = cv::format("%s/color%d.jpg", input_image_dir.c_str(), image_idx);
    image_names.push_back(image_name);
    image_idx++;
  }
}

double loadLineModDiamenter( const string diameter_filename )
{
  FILE *diam_file = fopen(diameter_filename.c_str(),"r");

  if( diam_file == NULL )
    return 0;

  double val;
  CV_Assert( fscanf(diam_file,"%lf", &val) == 1 );
  fclose(diam_file);

  return val;
}

struct AppOptions
{
  string dataset_path, object_name, templates_filename;
  int matching_step, num_matches, match_cell_size;
  bool symmetric_object;
};

void parseCommandLine( int argc, char **argv, AppOptions &options )
{
  string app_name( argv[0] );

  options.matching_step = 4;
  options.num_matches = 10;
  options.match_cell_size = -1;
  options.symmetric_object = false;

  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "dataset_path,d", po::value<string > ( &options.dataset_path)->required(),
    "Dataset path" )
  ( "object_name,n", po::value<string > ( &options.object_name)->required(),
    "Object name" )
  ( "templates_filename,t", po::value<string > ( &options.templates_filename),
    "Input model templates file" )
  ( "ms", po::value<int> ( &options.matching_step ),
    "Sample step used in the exhaustive meaning [default: 4]" )
  ( "nm", po::value<int> ( &options.num_matches ),
    "Number of best matches to be considered for each matching cell  [default: 10]" )
  ( "mcs", po::value<int> ( &options.match_cell_size ),
    "Size in pixels of each matching cell (if -1, just search for the best matches for the whoel image).[default: 1]" )
  ( "sym_obj,s", "Set this option if the object is symmetric" );

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

    if ( vm.count ( "sym_obj" ) )
      options.symmetric_object = true;

    po::notify ( vm );
  }

  catch ( boost::program_options::required_option& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    exit(EXIT_FAILURE);
  }

  catch ( boost::program_options::error& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    exit(EXIT_FAILURE);
  }

  cout << "Dataset path : " << options.dataset_path << endl;
  cout << "Object name : " << options.object_name << endl;
  if( !options.templates_filename.empty() )
    cout << "Loading templates from file : " << options.templates_filename << endl;
  cout << "Sample step used in the exhaustive meaning : "<<options.matching_step<< endl;
  cout << "Number of best matches to be considered for each matching cell : "<<options.num_matches<< endl;
  cout << "Size in pixels of each matching cell (if -1, just search for the best matches for the whoel image) : "<<options.match_cell_size<< endl;
}

void logResults(string name, vector<double> results_55, vector<double> results_2dp, vector<double> results_p6d,
                uint64_t total_time, int num_images )
{
  cout<<name<<" avg runtime : "<<total_time /num_images<<endl;

  double results_acc = 0.0;
  cout<<"5cm 5deg = [";
  for(int i = 0; i < int(results_55.size()); i++ )
  {
    results_55[i] /= double(num_images);
    results_acc += results_55[i];
    cout<<results_acc<<" ";
  }
  cout<<"]"<<endl;

  results_acc = 0.0;
  cout<<"2D Projection = [";
  for(int i = 0; i < int(results_2dp.size()); i++ )
  {
    results_2dp[i] /= double(num_images);
    results_acc += results_2dp[i];
    cout<<results_acc<<" ";
  }
  cout<<"]"<<endl;

  results_acc = 0.0;
  cout<<"6D Pose = [";
  for(int i = 0; i < int(results_p6d.size()); i++ )
  {
    results_p6d[i] /= double(num_images);
    results_acc += results_p6d[i];
    cout<<results_acc<<" ";
  }
  cout<<"]"<<endl;
}

int main(int argc, char **argv)
{
  AppOptions options;
  parseCommandLine( argc, argv, options );
  
  const float dist_transform_thresh = 30;
  const int canny_low_thresh = 80, canny_ratio = 2;
  const int num_edge_dirs = 60;
  const bool parallelism_enabled = true, rgb_modality_enabled = true;
  const int max_template_pts = 64;

  string camera_filename(options.dataset_path), model_filename(options.dataset_path),
         input_image_dir(options.dataset_path), diameter_filename(options.dataset_path), test_name;

  camera_filename += "/kinect_linemod_dataset.yml";

  model_filename  += "/";
  model_filename += options.object_name;
  model_filename  += "/mesh.ply";

  input_image_dir  += "/";
  input_image_dir += options.object_name;
  input_image_dir  += "/data/";

  diameter_filename  += "/";
  diameter_filename += options.object_name;
  diameter_filename  += "/distance.txt";

  stringstream test_name_tmp;
  test_name_tmp<<options.object_name;
  test_name_tmp<<"_";

  if( options.templates_filename.empty() )
  {
    test_name_tmp<<"gt_";
    test_name_tmp<<max_template_pts;
  }
  else
    test_name_tmp<<options.templates_filename;

  test_name_tmp<<"_";
  test_name_tmp<<options.matching_step;
  test_name_tmp<<"_";
  test_name_tmp<<options.num_matches;
  if( options.match_cell_size <= 0 )
  {
    test_name_tmp<<"_full";
  }
  else
  {
    test_name_tmp<<options.match_cell_size;
  }

  test_name = test_name_tmp.str();

  cv_ext::PinholeCameraModel cam_model;
  cam_model.readFromFile(camera_filename);

  RasterObjectModel3DPtr obj_model_ptr = std::make_shared<RasterObjectModel3D>();
  RasterObjectModel3D &obj_model = *obj_model_ptr;

  obj_model.setCamModel( cam_model );
  obj_model.setStepMeters(0.005);
  obj_model.setUnitOfMeasure(RasterObjectModel::MILLIMETER);
  obj_model.requestVertexColors();

  if(!obj_model.setModelFile( model_filename ))
    return -1;

  obj_model.computeRaster();

  vector<string> image_names;
  cv_ext::vector_Quaterniond gt_r_quats;
  cv_ext::vector_Vector3d gt_t_vecs;

  loadLineModDataset(input_image_dir, image_names, gt_r_quats, gt_t_vecs);

  DirIdxPointSetVecPtr dir_idx_templ_ptr = make_shared<DirIdxPointSetVec>();

  if( options.templates_filename.empty() )
  {
    const int templ_spacing_step = 1;

    ObjectTemplateGenerator<DirIdxPointSet> otg;

    otg.enableVerboseMode(true);
    otg.setMaxNumImgPoints(max_template_pts);
    otg.setImgPtsNumDirections(num_edge_dirs);
    otg.setImgPointsSpacing(templ_spacing_step);
    otg.setTemplateModel(obj_model_ptr);

    otg.generate(*dir_idx_templ_ptr, 0, gt_r_quats, gt_t_vecs);
  }
  else
    loadTemplateVector( options.templates_filename, *dir_idx_templ_ptr );

  cout<<"Num templates : "<<dir_idx_templ_ptr->size()<<endl;

  ObjectTemplatePnP o_pnp(obj_model_ptr->cameraModel());

  GradientDirectionScore scoring;

  double obj_diameter_cm = loadLineModDiamenter(diameter_filename);

  MaxRtErrorsMetric max_rt_metric;
  Projection2DMetric proj_2d_metric( cam_model, obj_model.vertices(), options.symmetric_object );
  Pose6DMetric pose_6d_metric(cam_model, obj_model.vertices(), 0.01*obj_diameter_cm, options.symmetric_object );

//   // Test draw ground truths
//   for( int i = 0; i < int(image_names.size()); i++ )
//   {
//     Mat src_img = imread ( image_names[i].c_str(), cv::IMREAD_COLOR );
//
//     obj_model.setModelView(dir_idx_templ_ptr->at(i).obj_r_quat, dir_idx_templ_ptr->at(i).obj_t_vec);
//
//     cv_ext::drawPoints(src_img, dir_idx_templ_ptr->at(i).proj_pts, Scalar(0,0,255));
//     cv::rectangle(src_img, dir_idx_templ_ptr->at(i).bbox, CV_RGB(0, 0, 255));
//
//     cv_ext::showImage(src_img);
//   }
  
  
#if TEST_LINE2D || TEST_LINE2D_GL
  
  const int n_threads = ( parallelism_enabled ? omp_get_max_threads() : 1 );  
    
//   cout<<"Linemod models"<<endl;
    const int linemod_matching_threshold = 60;
//  const int T_DEFAULTS[] = {5,8};
  const int T_DEFAULTS[] = {search_step};
  std::vector< Ptr<linemod::Modality> > modalities;

  modalities.push_back(new linemod::ColorGradient(10.0f,63, 55.0f));
  std::vector<int> T_pyramid(T_DEFAULTS, T_DEFAULTS + 1);

  vector<Ptr<linemod::Detector> > lm_detector, lm_detector_gl;
  
  for( int i = 0; i < n_threads; i++ )
  {
    lm_detector.push_back( (new  linemod::Detector(modalities, T_pyramid)) );
    lm_detector_gl.push_back( (new  linemod::Detector(modalities, T_pyramid)) );
  }

  map<string, Pose> pose_map, pose_map_gl;
  map<string, int> lm_detector_map, template_map;
  int num_classes = 0, th_id = 0;

  for( int i = 0; i < int(image_names.size()); i++ )
  {
    std::string class_id = cv::format("rot_%d", num_classes);
    cv::Rect bb, bb_gl;
    vector<Mat> templates, templates_gl;
    obj_model.setModelView ( gt_ts.r_quat[i], gt_ts.t_vec[i] );
    Mat gray_template, src_template = obj_model.getRenderedModel(), src_template_gl;
    cvtColor( src_template, gray_template, cv::COLOR_BGR2GRAY );
    cvtColor( gray_template, src_template_gl, cv::COLOR_GRAY2BGR );
    templates.push_back(src_template);
    templates_gl.push_back(src_template_gl);

//     cv_ext::showImage(src_template,"template");
    
    if ( lm_detector[th_id]->addTemplate(templates, class_id, cv::Mat(), &bb) != -1 &&
         lm_detector_gl[th_id]->addTemplate(templates_gl, class_id, cv::Mat(), &bb_gl) != -1 )
    {
      lm_detector_map[class_id] = th_id;
      template_map[class_id] = i;
//       if(!(num_classes % 100))
//         cout<<"Added template #"<<num_classes<<endl;
      Mat_<double> r_vec, t_vec;
      obj_model.modelView(r_vec, t_vec);
      pose_map[class_id] = Pose(r_vec,t_vec, bb);
      pose_map_gl[class_id] = Pose(r_vec,t_vec, bb_gl);
      num_classes++;

//       Mat src_img = imread ( image_names[i].c_str(), cv::IMREAD_COLOR );
// 
//       vector<Point2f> raster_pts;
// //       obj_model.projectRasterPoints( i, raster_pts );
// //       cv_ext::drawPoints(src_img, raster_pts, Scalar(0,0,255));
//       const std::vector<cv::linemod::Template>& templates_vec = detector[th_id]->getTemplates(class_id, 0);
//       drawResponse(templates_vec,1,src_img,Point(bb.x, bb.y),detector[th_id]->getT(0));
//       cv::rectangle(src_img, bb,CV_RGB(0, 0, 255));
//       cv::rectangle(src_img, bb_gl,CV_RGB(0, 255, 0));
// 
//       cv_ext::showImage(src_img);
      ++th_id;
      th_id %= n_threads;
    }
    else
      cout<<"Failed to add gt template "<<i<<endl;
  }
#endif

  const int num_test_images = image_names.size();
  const int max_num_matches = 10;
  cout<<"Num images : "<<num_test_images<<endl;

  if( options.symmetric_object )
    cout<<"The object is considered symmetric"<<endl;

  vector<double> cm_55(max_num_matches,0.0), ocm_55(max_num_matches,0.0), 
                 dcm_55(max_num_matches,0.0), line_55(max_num_matches,0.0),
                 line_55_gl(max_num_matches,0.0),
                 cm_2dp(max_num_matches,0.0), ocm_2dp(max_num_matches,0.0),
                 dcm_2dp(max_num_matches,0.0), line_2dp(max_num_matches,0.0),
                 line_2dp_gl(max_num_matches,0.0),
                 cm_p6d(max_num_matches,0.0), ocm_p6d(max_num_matches,0.0),
                 dcm_p6d(max_num_matches,0.0), line_p6d(max_num_matches,0.0),
                 line_p6d_gl(max_num_matches,0.0);
                 
  uint64_t avg_timer = 0;
  
#if TEST_CM
  DistanceTransform dc;
  dc.setDistThreshold(dist_transform_thresh);

  CannyEdgeDetectorUniquePtr edge_detector_ptr( new CannyEdgeDetector() );
//   Best so far...
  edge_detector_ptr->setLowThreshold(canny_low_thresh);
  edge_detector_ptr->setRatio(canny_ratio);
  edge_detector_ptr->enableRGBmodality(rgb_modality_enabled);

  dc.enableParallelism( parallelism_enabled );
  dc.setEdgeDetector(std::move(edge_detector_ptr));
  
  ChamferMatchingBase cm;
  cm.setTemplateModel(obj_model_ptr);
  cm.enableParallelism( parallelism_enabled );
  cm.setupExhaustiveMatching( max_template_pts );
  
  avg_timer = 0;
  
  for( int i = 0; i < num_test_images; i++ )
  {
//     if(!(i%100))
//       cout<<"Image "<<i<<" of "<<num_test_images<<endl;
    Mat src_img = imread ( image_names[i].c_str(),cv::IMREAD_COLOR );
    cv_ext::BasicTimer timer;
    Mat dist_map;
    dc.computeDistanceMap(src_img, dist_map);
    cm.setInput( dist_map );
    vector< TemplateMatch > matches;
    cm.match(max_num_matches, matches, search_step);
    avg_timer += timer.elapsedTimeMs();

    int i_m = 0;
    for( auto iter = matches.begin(); iter != matches.end(); iter++, i_m++ )
    {
      TemplateMatch &match = *iter;

      Eigen::Matrix3d est_rot_mat = match.r_quat.toRotationMatrix(), rot_mat_gt;
      cv_ext::exp2RotMat(gt_poses[i].r_vec, rot_mat_gt);

      cv::Point3d p_t_diff( match.t_vec(0) - gt_poses[i].t_vec.at<double>(0),
                            match.t_vec(1) - gt_poses[i].t_vec.at<double>(1),
                            match.t_vec(2) - gt_poses[i].t_vec.at<double>(2) );
      
      double rot_diff = 180.0*cv_ext::rotationDist(est_rot_mat, rot_mat_gt)/M_PI, 
             t_diff  = cv_ext::norm3D(p_t_diff);
             
      if( rot_diff < max_rot_diff_deg && t_diff < max_t_diff )
      {
        cm_55[i_m]++;
        
//         obj_model.setModelView(match.r_quat, match.t_vec);
//         vector<Point2f> proj_pts;
//         obj_model.projectRasterPoints(proj_pts);        
//         cv::Mat display = src_img.clone();
//         cv_ext::drawPoints( display, proj_pts, Scalar(0,255,0) );
//         cv_ext::showImage(display,"display");
        
        break;
      }
    }
  }
  stringstream sstr;
  sstr<<"cm_55";
  sstr<<max_template_pts;
  if( !rgb_modality_enabled )
    sstr<<" gl";
  logResults(sstr.str(), cm_55, avg_timer, num_test_images );
  
#endif
  
#if TEST_OCM
  DistanceTransform dc;
  dc.setDistThreshold(dist_transform_thresh);

  CannyEdgeDetectorUniquePtr edge_detector_ptr( new CannyEdgeDetector() );
//   Best so far...
  edge_detector_ptr->setLowThreshold(canny_low_thresh);
  edge_detector_ptr->setRatio(canny_ratio);
  edge_detector_ptr->enableRGBmodality(rgb_modality_enabled);

//   edge_detector_ptr->setLowThreshold(80);
//   edge_detector_ptr->setRatio(3);

//   LSDEdgeDetectorUniquePtr edge_detector_ptr( new LSDEdgeDetector() );
//   edge_detector_ptr->setPyrNumLevels(2);
//   edge_detector_ptr->setScale(4);

  dc.enableParallelism( parallelism_enabled );
  dc.setEdgeDetector(std::move(edge_detector_ptr));
  
  OrientedChamferMatching ocm;
  ocm.setNumDirections(num_directions);
  ocm.setTemplateModel(obj_model_ptr);
  ocm.enableParallelism( parallelism_enabled );
  ocm.setupExhaustiveMatching( max_template_pts );
  
    
  const double tensor_lambda = 6.0;
  const bool smooth_tensor = false;
 
  avg_timer = 0;
  
  for( int i = 0; i < num_test_images; i++ )
  {
    if(!(i%100))
      cout<<"Image "<<i<<" of "<<num_test_images<<endl;
    Mat src_img = imread ( image_names[i].c_str(),cv::IMREAD_COLOR );
    cv_ext::BasicTimer timer;
    Mat dist_map, dir_map;
    dc.computeDistDirMap(src_img, dist_map, dir_map, num_directions);
//     cout<<"CM elapsed time computeDistanceMapTensor ms : "<<timer.elapsedTimeMs()<<endl;
    ocm.setInput( dist_map, dir_map );
    vector< TemplateMatch > matches;
    ocm.match(max_num_matches, matches, search_step);
    avg_timer += timer.elapsedTimeMs();

    int i_m = 0;
    for( auto iter = matches.begin(); iter != matches.end(); iter++, i_m++ )
    {
      TemplateMatch &match = *iter;

      Eigen::Matrix3d est_rot_mat = match.r_quat.toRotationMatrix(), rot_mat_gt;
      cv_ext::exp2RotMat(gt_poses[i].r_vec, rot_mat_gt);

      cv::Point3d p_t_diff( match.t_vec(0) - gt_poses[i].t_vec.at<double>(0),
                            match.t_vec(1) - gt_poses[i].t_vec.at<double>(1),
                            match.t_vec(2) - gt_poses[i].t_vec.at<double>(2) );
      
      double rot_diff = 180.0*cv_ext::rotationDist(est_rot_mat, rot_mat_gt)/M_PI, 
             t_diff  = cv_ext::norm3D(p_t_diff);
             
      if( rot_diff < max_rot_diff_deg && t_diff < max_t_diff )
      {
        ocm_55[i_m]++;
//         obj_model.setModelView(match.r_quat, match.t_vec);
//         vector<Point2f> proj_pts;
//         obj_model.projectRasterPoints(proj_pts);        
//         cv::Mat display = src_img.clone();
//         cv_ext::drawPoints( display, proj_pts, Scalar(0,255,0) );
//         cv_ext::showImage(display,"display");        
        
        break;
      }
    }
  }
  stringstream sstr;
  sstr<<"ocm_55";
  sstr<<max_template_pts;
  if( !rgb_modality_enabled )
    sstr<<" gl";  
  logResults(sstr.str(), ocm_55, avg_timer, num_test_images );

#endif
  
  
#if TEST_DCM
  DistanceTransform dc;
  dc.setDistThreshold(dist_transform_thresh);

  CannyEdgeDetectorUniquePtr edge_detector_ptr( new CannyEdgeDetector() );
//   Best so far...
  edge_detector_ptr->setLowThreshold(canny_low_thresh);
  edge_detector_ptr->setRatio(canny_ratio);
  edge_detector_ptr->enableRGBmodality(rgb_modality_enabled);

  dc.enableParallelism( parallelism_enabled );
//   dc.setDistType(CV_DIST_L1);
//   dc.setMaskSize(3);
  dc.setEdgeDetector(std::move(edge_detector_ptr));
  
  double tensor_lambda = 6.0;
  const bool smooth_tensor = false;
  
  DirectionalChamferMatchingBase<DirIdxPointSet> dcm;
  dcm.enableParallelism( parallelism_enabled );
  dcm.setTemplatesVector( dir_idx_templ_ptr );

  DirectionalChamferRegistration dcr;
  dcr.setNumDirections(num_edge_dirs);
  dcr.setObjectModel(obj_model_ptr);

  avg_timer = 0;
  
  for( int i = 0; i < num_test_images; i++ )
  {
//     if(!(i%100))
//       cout<<"Image "<<i<<" of "<<num_test_images<<endl;
    Mat src_img = imread ( image_names[i].c_str(),cv::IMREAD_COLOR ), proc_img;
    cv_ext::BasicTimer timer;
    ImageTensorPtr dst_map_tensor_ptr = std::make_shared<ImageTensor>();
    

//     bilateralFilter(src_img, proc_img, -1, 25, 0.5);
    proc_img = src_img;

    dc.computeDistanceMapTensor (proc_img, *dst_map_tensor_ptr, num_edge_dirs, tensor_lambda, smooth_tensor);
    dcm.setInput( dst_map_tensor_ptr );
    dcr.setInput(dst_map_tensor_ptr);
    vector< TemplateMatch > matches;
    dcm.match(options.num_matches, matches, options.matching_step, options.match_cell_size );

    cv_ext::vector_Quaterniond match_r_quats;
    cv_ext::vector_Vector3d match_t_vecs;

    Eigen::Quaterniond match_r_quat;
    Eigen::Vector3d match_t_vec;

    multimap<double, int> scores;

    ImageGradient im_grad( proc_img );

    int i_m = 0;
    for (auto iter = matches.begin(); iter != matches.end(); iter++, i_m++)
    {
      TemplateMatch &match = *iter;
      o_pnp.solve( (*dir_idx_templ_ptr)[match.id], match.img_offset, match_r_quat, match_t_vec);

      dcr.refinePosition( (*dir_idx_templ_ptr)[match.id], match_r_quat, match_t_vec);

      match_r_quats.push_back(match_r_quat);
      match_t_vecs.push_back(match_t_vec);

      vector<Point2f> refined_proj_pts;
      vector<float> normals;
      obj_model_ptr->setModelView(match_r_quat, match_t_vec);
      obj_model_ptr->projectRasterPoints(refined_proj_pts, normals);
      double score = scoring.evaluate(im_grad, refined_proj_pts, normals);
      if( score )
        scores.insert(std::pair<double, int>(1.0/score, i_m));
    }

    avg_timer += timer.elapsedTimeMs();

    i_m = 0;
    bool done_with_55 = false, done_with_2dp = false, done_with_p6d = false;

    max_rt_metric.setGroundTruth( gt_r_quats[i], gt_t_vecs[i] );
    proj_2d_metric.setGroundTruth( gt_r_quats[i], gt_t_vecs[i] );
    pose_6d_metric.setGroundTruth( gt_r_quats[i], gt_t_vecs[i] );

    for( auto iter = scores.begin(); iter != scores.end() && i_m < max_num_matches; iter++, i_m++ )
    {
      int idx = iter->second;
      Eigen::Quaterniond &match_r_quat = match_r_quats[idx];
      Eigen::Vector3d &match_t_vec = match_t_vecs[idx];

//      if( rot_diff < max_rot_diff_deg && t_diff < max_t_diff && avg_pixel_diff > 5 )
//      {
//        cout<<rot_diff<<" "<<t_diff<<" "<<avg_pixel_diff<<" "<<endl;
//
//        vector< cv::Point2f > gt_proj_pts, est_proj_pts;
//        obj_model.setModelView(gt_r_quats[i], gt_t_vecs[i]);
//        obj_model.projectRasterPoints(gt_proj_pts);
//        obj_model.setModelView(match_r_quat, match_t_vec);
//        obj_model.projectRasterPoints(est_proj_pts);
//
//        cv_ext::drawPoints(src_img, gt_proj_pts, Scalar(0,0,255));
//        cv_ext::drawPoints(src_img, est_proj_pts, Scalar(255,0,0));
//
//        cv_ext::showImage(src_img);
//      }

      if( !done_with_55 )
      {
        if( max_rt_metric.performTest(match_r_quat, match_t_vec ) )
        {
          dcm_55[i_m]++;
          done_with_55 = true;
        }
      }

      if( !done_with_2dp )
      {
        if( proj_2d_metric.performTest( match_r_quat, match_t_vec ) )
        {
          dcm_2dp[i_m]++;
          done_with_2dp = true;


//          // WARNING DEBUG CODE
//          if( !done_with_55 )
//          {
//            Eigen::Matrix3d est_rot_mat = match_r_quat.toRotationMatrix(), rot_mat_gt = gt_r_quats[i].toRotationMatrix();
//
//            double rot_diff = 180.0*cv_ext::rotationDist(est_rot_mat, rot_mat_gt)/M_PI,
//                   t_diff  = (match_t_vec - gt_t_vecs[i]).norm();
//
//            cout<<rot_diff<<" "<<t_diff<<" "<<avg_pixel_diff<<" "<<endl;
//            cout<<est_rot_mat<<endl;
//            cout<<rot_mat_gt<<endl;
//
//            vector< cv::Point2f > gt_proj_pts, est_proj_pts;
//            obj_model.setModelView(gt_r_quats[i], gt_t_vecs[i]);
//            obj_model.projectRasterPoints(gt_proj_pts);
//            obj_model.setModelView(match_r_quat, match_t_vec);
//            obj_model.projectRasterPoints(est_proj_pts);
//
//            cv_ext::drawPoints(src_img, gt_proj_pts, Scalar(0,0,255));
//            cv_ext::drawPoints(src_img, est_proj_pts, Scalar(255,0,0));
//
//            cv_ext::showImage(src_img);
//          }

        }
      }

      if( !done_with_p6d )
      {
        if( pose_6d_metric.performTest( match_r_quat, match_t_vec ) )
        {
          dcm_p6d[i_m]++;
          done_with_p6d = true;
        }
      }

      if( done_with_55 && done_with_2dp && done_with_p6d )
        break;
    }
  }

  string log_str("dcm ");
  if( !rgb_modality_enabled )
    log_str += "gl ";
  log_str += test_name;
  logResults(log_str, dcm_55, dcm_2dp, dcm_p6d, avg_timer, num_test_images );
  
#endif

#if TEST_LINE2D
  
  avg_timer = 0;
  
  for( int i = 0; i < num_test_images; i++ )
  {
    Mat src_img = imread ( image_names[i].c_str(), cv::IMREAD_COLOR );

//     if(!(i%100))
//       cout<<"Image "<<i<<" of "<<num_test_images<<endl;
    
    vector<Mat> sources;
    sources.push_back(src_img);
    std::vector<cv::linemod::Match> th_matches[n_threads], matches;

    cv_ext::BasicTimer timer;
    #pragma omp parallel if( parallelism_enabled )
    {
      int i_th = omp_get_thread_num();
      lm_detector[i_th]->match(sources, (float)linemod_matching_threshold, th_matches[i_th] );
    }
    
    // Combine the results
    for( int i_th = 0; i_th < n_threads; i_th++ )
      matches.insert(matches.end(), th_matches[i_th].begin(), th_matches[i_th].end());
    std::stable_sort (matches.begin(), matches.end() );
    
    avg_timer += timer.elapsedTimeMs();
    
    for (int i_m = 0; i_m < (int)matches.size() && i_m < max_num_matches; i_m++)
    {
      cv::linemod::Match m = matches[i_m];
      std::string class_id_gt = cv::format("rot_%d", i);
      Pose p_est = pose_map[m.class_id], p_gt = pose_map[class_id_gt];

      normalizePose( obj_model, cam_model, Point(m.x, m.y), p_est, template_map[m.class_id] );
      
      Eigen::Matrix3d est_rot_mat, rot_mat_gt = gt_ts.r_quat[i].toRotationMatrix();

      cv_ext::exp2RotMat(p_est.r_vec, est_rot_mat);
      
      cv::Point3d p_t_diff( p_est.t_vec.at<double>(0) - gt_ts.t_vec[i](0),
                            p_est.t_vec.at<double>(1) - gt_ts.t_vec[i](1),
                            p_est.t_vec.at<double>(2) - gt_ts.t_vec[i](2) );
      
      double rot_diff = 180.0*cv_ext::rotationDist(est_rot_mat, rot_mat_gt)/M_PI,
             t_diff  = cv_ext::norm3D(p_t_diff);

      if( rot_diff < max_rot_diff_deg && t_diff < max_t_diff )
      {
        line_55[i_m]++;

//         // Draw matching template
//         cv::Mat display = sources[0].clone(); 
//         vector <Point2f> proj_pts;
//         obj_model.projectRasterPoints( template_map[m.class_id], proj_pts );
//         Point2f off_p(m.x - p_est.bb.x, m.y - p_est.bb.y);
//         for( auto &p : proj_pts ) p += off_p;
//         cv_ext::drawPoints( display, proj_pts, Scalar(0,255,0) );
//         cv_ext::showImage(display,"display");

        break;
      }
    }
  }

  logResults("line_55", line_55, avg_timer, num_test_images );

#endif

#if TEST_LINE2D_GL
  
  avg_timer = 0;
  
  for( int i = 0; i < num_test_images; i++ )
  {
//     if(!(i%100))
//       cout<<"Image "<<i<<" of "<<num_test_images<<endl;    
    Mat src_img = imread ( image_names[i].c_str(), cv::IMREAD_COLOR );
    Mat gray_src;
    cvtColor( src_img, gray_src, cv::COLOR_BGR2GRAY );
    cvtColor( gray_src, src_img, cv::COLOR_GRAY2BGR );

    vector<Mat> sources;
    sources.push_back(src_img);
    std::vector<cv::linemod::Match> th_matches[n_threads], matches;

    cv_ext::BasicTimer timer;
    #pragma omp parallel if( parallelism_enabled )
    {
      int i_th = omp_get_thread_num();
      lm_detector_gl[i_th]->match(sources, (float)linemod_matching_threshold, th_matches[i_th] );
    }
    
    // Combine the results
    for( int i_th = 0; i_th < n_threads; i_th++ )
      matches.insert(matches.end(), th_matches[i_th].begin(), th_matches[i_th].end());
    std::stable_sort (matches.begin(), matches.end());
    
    avg_timer += timer.elapsedTimeMs();
    
    for (int i_m = 0; i_m < (int)matches.size() && i_m < max_num_matches; ++i_m)
    {
      cv::linemod::Match m = matches[i_m];
      std::string class_id_gt = cv::format("rot_%d", i);
      Pose p_est = pose_map_gl[m.class_id], p_gt = pose_map_gl[class_id_gt];
      
      normalizePose( obj_model, cam_model, Point(m.x, m.y), p_est, template_map[m.class_id]);
      
      Eigen::Matrix3d est_rot_mat, rot_mat_gt;

      cv_ext::exp2RotMat(p_est.r_vec, est_rot_mat);
      cv_ext::exp2RotMat(p_gt.r_vec, rot_mat_gt);

      cv::Point3d p_t_diff( p_est.t_vec.at<double>(0) - gt_poses[i].t_vec.at<double>(0),
                            p_est.t_vec.at<double>(1) - gt_poses[i].t_vec.at<double>(1),
                            p_est.t_vec.at<double>(2) - gt_poses[i].t_vec.at<double>(2) );
      
      double rot_diff = 180.0*cv_ext::rotationDist(est_rot_mat, rot_mat_gt)/M_PI,
             t_diff  = cv_ext::norm3D(p_t_diff);

      if( rot_diff < max_rot_diff_deg && t_diff < max_t_diff )
      {
        line_55_gl[i_m]++;
//         // Draw matching template
//         cv::Mat display = sources[0].clone(); 
//         vector <Point2f> proj_pts;
//         obj_model.projectRasterPoints( template_map[m.class_id], proj_pts );
//         Point2f off_p(m.x - p_est.bb.x, m.y - p_est.bb.y);
//         for( auto &p : proj_pts ) p += off_p;
//         cv_ext::drawPoints( display, proj_pts, Scalar(0,255,0) );
//         cv_ext::showImage(display,"display");
        break;

      }
    }
  }

  logResults("line_55 gl", line_55_gl, avg_timer, num_test_images );
  
#endif
  
  

  return 0;
}
