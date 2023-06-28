#include <cstdio>
#include <string>
#include <sstream>
#include <map>
#include <iomanip>
#include <boost/program_options.hpp>
#include <opencv2/rgbd.hpp>

#include <opencv2/cnn_3dobj.hpp>
#include <opencv2/features2d.hpp>

#include "omp.h"

#include "cv_ext/cv_ext.h"

#include "raster_object_model3D.h"
#include "raster_object_model2D.h"
#include "chamfer_matching.h"

#include "apps_utils.h"

extern "C"
{
#include "lsd.h"
}


#define PARALLELISM_ENABLED 1

#define RGB_MODALITY_ENABLED 1

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


void normalizePose( const RasterObjectModel3D &obj_model, const PinholeCameraModel &cam_model, const Point &disp, Pose &p, int id )
{
  Point2f off_p(disp.x - p.bb.x, disp.y - p.bb.y);

  if( cv_ext::norm2D(off_p) == 0 )
    return;

  vector <Point3f> obj_pts = obj_model.getPrecomputedPoints(id);
  vector <Point2f> proj_pts;
  
  obj_model.projectRasterPoints( id, proj_pts );
  for( auto &p : proj_pts )
    p += off_p;
  
  cv::Mat r_vec(3,1,cv::DataType<double>::type);
  cv::Mat t_vec(3,1,cv::DataType<double>::type);
  
  cv::solvePnP( obj_pts, proj_pts, cam_model.cameraMatrix(), cam_model.distorsionCoeff(), r_vec, t_vec );
  
  p.r_vec = r_vec;
  p.t_vec = t_vec;
}


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


void loadACCVDataset( const string input_image_dir, vector<string> &image_names,
                       vector<Mat> &gt_r_vecs, vector<Mat> &gt_t_vecs )
{
  Mat gt_r_vec = Mat_<double>(3,1), gt_t_vec = Mat_<double>(3,1);
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

    Eigen::Matrix3d gt_rot;
    Eigen::Vector3d gt_tra;
    double val0,val1;
    CV_Assert(fscanf(rot_file,"%lf %lf", &val0,&val1) == 2);
    CV_Assert(fscanf(rot_file,"%lf %lf %lf", &gt_rot(0,0),&gt_rot(0,1),&gt_rot(0,2)) == 3);
    CV_Assert(fscanf(rot_file,"%lf %lf %lf", &gt_rot(1,0),&gt_rot(1,1),&gt_rot(1,2)) == 3);
    CV_Assert(fscanf(rot_file,"%lf %lf %lf", &gt_rot(2,0),&gt_rot(2,1),&gt_rot(2,2)) == 3);

    CV_Assert(fscanf(tra_file,"%lf %lf", &val0, &val1) == 2);
    CV_Assert(fscanf(tra_file,"%lf %lf %lf", &gt_tra(0),&gt_tra(1),&gt_tra(2)) == 3);

    cv_ext::rotMat2Exp( gt_rot, gt_r_vec );

    gt_t_vec.at<double>(0) = gt_tra(0)/100.0;
    gt_t_vec.at<double>(1) = gt_tra(1)/100.0;
    gt_t_vec.at<double>(2) = gt_tra(2)/100.0;

    gt_r_vecs.push_back(gt_r_vec.clone());
    gt_t_vecs.push_back(gt_t_vec.clone());

    fclose(rot_file);
    fclose(tra_file);

    std::string image_name = cv::format("%s/color%d.jpg", input_image_dir.c_str(), image_idx);
    image_names.push_back(image_name);
    image_idx++;
  }
}

struct AppOptions
{
  string templates_filename, camera_filename, input_image_dir, cnn, caffemodel, 
         mean_file, feature_blob = string("feat"), device = string("GPU");
};

void parseCommandLine( int argc, char **argv, AppOptions &options )
{
  string app_name( argv[0] );

  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "model_filename,m", po::value<string > ( &options.templates_filename )->required(),
    "DXF, STL or PLY model file" )
  ( "camera_filename,c", po::value<string > ( &options.camera_filename )->required(),
    "A YAML file that stores all the camera parameters (see the PinholeCameraModel object)" )
  ( "input_image_dir,d", po::value<string > ( &options.input_image_dir)->required(),
    "input_image_dir" )
  ( "cnn,n", po::value< string > ( &options.cnn)->required(),
    "CNN definition file" )
  ( "caffemodel,f", po::value<string > ( &options.caffemodel)->required(),
    "Caffe model for feature exrtaction" )
  ( "feature_blob", po::value<string > ( &options.feature_blob),
    "Name of layer which will represent as the feature, in this network, ip1 or feat is well" )
  ( "mean_file", po::value<string > ( &options.mean_file),
    "The mean file generated by Caffe from all gallery images" )
  ( "device", po::value<string > ( &options.device ),
    "Device type: CPU or GPU" );
  

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
    exit(EXIT_FAILURE);
  }

  catch ( boost::program_options::error& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    exit(EXIT_FAILURE);
  }

  cout << "Loading model from file : "<<options.templates_filename<< endl;
  cout << "Loading camera parameters from file : "<<options.camera_filename<< endl;
  cout << "Loading input images from directory : "<<options.input_image_dir<< endl;
}

void logResults(string name, vector<double> fbm, uint64_t total_time, int num_images )
{
  double results_acc = 0.0;
  cout<<name<<" =[";
  for(int i = 0; i < fbm.size(); i++ )
  {
    fbm[i] /= double(num_images);
    results_acc += fbm[i];
    cout<<results_acc<<" ";
  }
  cout<<"]"<<" avg runtime : "<<total_time /num_images<<endl;
}

int main(int argc, char **argv)
{
  AppOptions options;
  parseCommandLine( argc, argv, options );
  
  const int search_step = 4, patch_size = 64;
  
  cv_ext::PinholeCameraModel cam_model;
  cam_model.readFromFile(options.camera_filename);

  int width = cam_model.imgWidth(), height = cam_model.imgHeight();

  RasterObjectModel3DPtr obj_model_ptr( new RasterObjectModel3D() );
  RasterObjectModel3D &obj_model = *obj_model_ptr;

  obj_model.setCamModel( cam_model );
  obj_model.setStepMeters(0.005);
  obj_model.setUnitOfMeasure(RasterObjectModel::MILLIMETER);
  obj_model.requestVertexColors();

  if(!obj_model.setModelFile( options.templates_filename ))
    return -1;

  obj_model.computeRaster();


  vector<string> image_names;
  vector<Mat> gt_r_vecs, gt_t_vecs;
  vector<Pose> gt_poses;
  loadACCVDataset( options.input_image_dir, image_names, gt_r_vecs, gt_t_vecs );

  int max_patch_size = 0;
  for( int i = 0; i < int(image_names.size()); i++ )
  {
    obj_model.setModelView(gt_r_vecs[i], gt_t_vecs[i]);
    obj_model.storeModelView();
    vector<Point2f> raster_pts;
    vector<float> raster_normals_dirs;

    obj_model.projectRasterPoints( raster_pts, raster_normals_dirs );
    Point tl(width,height), br(0,0);

    for( int j = 0; j < int(raster_pts.size()); j++ )
    {
      int x = cvRound(raster_pts[j].x), y = cvRound(raster_pts[j].y);

      if( unsigned(x) < unsigned(width) && unsigned(y) < unsigned(height) )
      {
        if( x < tl.x ) tl.x = x;
        if( y < tl.y ) tl.y = y;
        if( x > br.x ) br.x = x;
        if( y > br.y ) br.y = y;
      }
    }
    br.x++;
    br.y++;
    cv::Rect bb(tl,br);
    gt_poses.push_back(Pose(gt_r_vecs[i],gt_t_vecs[i], bb));
    
//     Mat src_template = obj_model.getRenderedModel();
//     if( !RGB_MODALITY_ENABLED )
//       cvtColor( src_template, src_template, cv::COLOR_BGR2GRAY );
    
    if(bb.width > max_patch_size) max_patch_size = bb.width;
    if(bb.height > max_patch_size) max_patch_size = bb.height;
    
  }
  
  float scale = float(patch_size)/max_patch_size; 
  int scaled_search_step = cvRound(scale*search_step), 
      scaled_width = cvRound(scale*width),
      scaled_height = cvRound(scale*height);
      
  if(scaled_search_step < 2) scaled_search_step = 2;
  cout<<"Scale : "<<scale<<endl;
  
  vector<cv::Rect> scaled_bb(image_names.size());
  for( int i = 0; i < image_names.size(); i++ )
  {
    Mat src_img = imread ( image_names[i].c_str(), cv::IMREAD_COLOR );

    gt_poses[i].bb.x -= (max_patch_size - gt_poses[i].bb.width)/2;
    gt_poses[i].bb.y -= (max_patch_size - gt_poses[i].bb.height)/2;
    if(gt_poses[i].bb.x < 0 ) gt_poses[i].bb.x = 0;
    if(gt_poses[i].bb.y < 0 ) gt_poses[i].bb.y = 0;
    if( gt_poses[i].bb.x + max_patch_size > width )
      gt_poses[i].bb.x = width - max_patch_size;
    if( gt_poses[i].bb.y + max_patch_size > height )
      gt_poses[i].bb.y = height - max_patch_size;
    gt_poses[i].bb.width = gt_poses[i].bb.height = max_patch_size;
    
    scaled_bb[i].x = cvRound(scale*gt_poses[i].bb.x);
    scaled_bb[i].y = cvRound(scale*gt_poses[i].bb.y);
    scaled_bb[i].width = patch_size;
    scaled_bb[i].height = patch_size;
  }
  
//   // Test draw ground truths
//   for( int i = 0; i < 30; i++ )
//   {
//     Mat src_img = imread ( image_names[i].c_str(), cv::IMREAD_COLOR ), scaled_img;
//     cv::resize(src_img, scaled_img, Size(), scale, scale);
//     
//     vector<Point2f> raster_pts;
//     obj_model.projectRasterPoints( i, raster_pts );
//     cv_ext::drawPoints(src_img, raster_pts, CV_RGB(0, 0, 255));
//     cv::rectangle(src_img, gt_poses[i].bb,CV_RGB(255, 0, 0));
//     cv_ext::showImage(src_img, "Orig", true, 1);
//     
//     cv::rectangle(scaled_img, scaled_bb[i],CV_RGB(255, 0, 0));
//     cv_ext::showImage(scaled_img, "Scaled");
//     
//   }

  cv::cnn_3dobj::descriptorExtractor descriptor(options.device);
  if( options.mean_file.empty() )
    descriptor.loadNet(options.cnn, options.caffemodel);
  else
    descriptor.loadNet(options.cnn, options.caffemodel, options.mean_file);
  
  std::vector<cv::Mat> img_gallery;
  cv::Mat feature_reference;
    
  for( int i = 0; i < obj_model.numPrecomputedModelsViews(); i++ )
  {
    obj_model.setModelView(i);
    
    Mat src_template = obj_model.getRenderedModel(CV_RGB(128,128,128)), scaled_template;
    cv::resize(src_template, scaled_template, Size(), scale, scale);    
    if( !RGB_MODALITY_ENABLED )
      cvtColor( scaled_template, scaled_template, cv::COLOR_BGR2GRAY );

    Mat patch = scaled_template(scaled_bb[i]);
    img_gallery.push_back(patch);
  }
  
  /* Extract feature from a set of images */
  descriptor.extract(img_gallery, feature_reference, options.feature_blob);

  for(int r = 0; r < feature_reference.rows; r++ )
    cout<<"ROW : "<<r<<feature_reference.row(r)<<endl;

  const double max_rot_diff_deg = 5, max_t_diff = 0.05;
  const int num_test_images = image_names.size();
  const int max_num_matches = 10;
  
  cout<<num_test_images<<endl;

  vector<double> cnn_3dobj_fbm(max_num_matches,0.0);

  vector<cv::Rect> patches_rect;
  
  for( int y = 0; y < scaled_height - patch_size; y+=scaled_search_step)
  {
    for( int x = 0; x < scaled_width - patch_size; x+=scaled_search_step)
      patches_rect.push_back(Rect(x,y,patch_size,patch_size));
  }
  
  uint64_t avg_timer = 0;
  for( int i = 0; i < num_test_images; i++ )
  {
//     if(!(i%100))
//       cout<<"Image "<<i<<" of "<<num_test_images<<endl;    
    Mat src_img = imread ( image_names[i].c_str(), cv::IMREAD_COLOR ), scaled_img;

    cv::resize(src_img, scaled_img, Size(), scale, scale);    
    if( !RGB_MODALITY_ENABLED )
      cvtColor( scaled_img, scaled_img, cv::COLOR_BGR2GRAY );
    
    std::vector<cv::Mat> cur_imgs;
    cur_imgs.reserve(patches_rect.size());
    
    // WARNING DEBUG
    cur_imgs.push_back(scaled_img(scaled_bb[i]));
//     for(auto &bb:patches_rect )
//     {
//       cur_imgs.push_back(scaled_img(bb));  
// //       Mat display = scaled_img.clone();
// //       cv::rectangle(display, bb,CV_RGB(255, 0, 0));
// //       cv_ext::showImage(display, "Scaled");
//     }

    cv_ext::BasicTimer timer;

    cv::Mat cur_feature;
    descriptor.extract(cur_imgs, cur_feature, options.feature_blob);
    /* Initialize a matcher which using L2 distance. */
    cv::BFMatcher matcher(NORM_L2);
    std::vector<std::vector<cv::DMatch> > matches;
    /* Have a KNN match on the target and reference images. */
    matcher.knnMatch(cur_feature, feature_reference, matches, max_num_matches);
    cout<<"cur_feature : "<<cur_feature<<endl;
    avg_timer += timer.elapsedTimeMs();
    cout<<timer.elapsedTimeMs()<<endl;
    
    std::multimap< double, cv::DMatch > best_matches;
    float min_dist = std::numeric_limits< float >::max();
    
    for (size_t ii = 0; ii < matches.size(); ++ii)
    {
      for (size_t jj = 0; jj < matches[ii].size(); ++jj)
      {
//         if( matches[ii][jj].trainIdx == i )
//         {
          float dist = matches[ii][jj].distance;
          if( int(best_matches.size()) < max_num_matches )
          {
            best_matches.insert ( std::pair<double, cv::DMatch >(dist, matches[ii][jj]) );
            min_dist = best_matches.rbegin()->first;
          }
          else if( dist < min_dist )
          {
            best_matches.insert ( std::pair<double, cv::DMatch >(dist, matches[ii][jj]) );
            best_matches.erase(--best_matches.rbegin().base());
            min_dist = best_matches.rbegin()->first;
          }
//         }
      }
    }
    
    for( auto &m:best_matches)
    {
      cout<<"gth_feature : "<<feature_reference.row(i)<<endl;
      cout<<"ref_feature : "<<feature_reference.row(m.second.trainIdx)<<endl;
      cout<<m.second.trainIdx<<" dist "<<m.second.distance<<" "<<norm(feature_reference.row(i) - cur_feature)<<endl;
      Mat display = scaled_img.clone();
//       cv::rectangle(display, patches_rect[m.second.queryIdx],CV_RGB(255, 0, 0));
      cv_ext::showImage(cur_imgs[0], "Patch", true,10);
      cv_ext::showImage(img_gallery[m.second.trainIdx], "Matched patch", true,10);    
      cv_ext::showImage(img_gallery[i], "GT patch");
    }
/*    for (int i_m = 0; i_m < (int)matches.size() && i_m < max_num_matches; i_m++)
    {
      cv::linemod::Match m = matches[i_m];
      std::string class_id_gt = cv::format("rot_%d", i);
      Pose p_est = pose_map[m.class_id], p_gt = pose_map[class_id_gt];

      normalizePose( obj_model, cam_model, Point(m.x, m.y), p_est, template_map[m.class_id] );
      
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
        cnn_3dobj_fbm[i_m]++;

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

   */ 
  }

  logResults("cnn_3dobj_fbm", cnn_3dobj_fbm, avg_timer, num_test_images );
  

  return 0;
}
