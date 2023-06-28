#include <time.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "cv_ext/pinhole_scene_projector.h"
#include "cv_ext/timer.h"
#include "cv_ext/debug_tools.h"

using namespace std;

int main(int argc, char** argv)
{
  srand (time(NULL));
  
  typedef double point_3Dtype;
  typedef float point_2Dtype;

  int num_samples = 1000;  
  if( argc > 1)
    num_samples = atoi(argv[1]);

  cout<<"Using "<<num_samples<<" sample points, precision 3D: "<<sizeof(point_3Dtype)
           <<" bits, precision 2D: "<<sizeof(point_2Dtype)<<" bits"<<endl;
  
  cv::Mat r_vec(cv::Mat_<double>(3,1, 0.0)),
          t_vec(cv::Mat_<double>(3,1, 0.0));
  t_vec.at<double>(2) = 1.0;
  
  //dist_coeff = cv::Mat::zeros(8,1,CV_64F);
  vector< cv::Point3_<point_3Dtype> > sample_pts;
  
  const double max_disp = 5;
  sample_pts.reserve(num_samples);
  double r1 , r2 , r3;
  for(int i = 0; i < num_samples; i++)
  {
    r1 = -max_disp/2 + max_disp*static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    r2 = -max_disp/2 + max_disp*static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    r3 = -0.5 + static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    
    sample_pts.push_back(cv::Point3_<point_3Dtype>(r1, r2, 1.0 + r3));
  }
  
  cv_ext::BasicTimer timer;
  cv_ext::PinholeCameraModel cam_model;
  cam_model.readFromFile("test_camera_calib.yml"); 
    
  cv::Mat camera_matrix = cam_model.cameraMatrix();
  cv::Mat dist_coeff = cam_model.distorsionCoeff();
  
  cv::Mat_<cv::Vec3b> dbg_img(cam_model.imgSize());
  
  vector< cv::Point_<point_2Dtype> > proj_pts_sp_single, proj_pts_sp_parallel;                                          
  // OpenCV does not support different types
  vector< cv::Point_<point_3Dtype> >proj_pts_cv;
  
  cv_ext::PinholeSceneProjector s_proj( cam_model );
  
  s_proj.enableParallelism( true );
  s_proj.setTransformation(r_vec, t_vec );
  
  timer.reset();
  s_proj.projectPoints( sample_pts, proj_pts_sp_parallel );
  cout<<"PinholeSceneProjector projection parallel: elapsedTime :"<<timer.elapsedTimeUs()<<" usec\n";

  // PinholeSceneProjector single thread
  s_proj.enableParallelism( false );
  
  timer.reset();
  s_proj.projectPoints( sample_pts, proj_pts_sp_single );
  cout<<"PinholeSceneProjector projection single: elapsedTime :"<<timer.elapsedTimeUs()<<" usec\n";
  
  // Open cv
  timer.reset();
  cv::projectPoints(sample_pts, r_vec, t_vec, camera_matrix,  dist_coeff, proj_pts_cv );
  cout<<"OpenCV projection: elapsedTime :"<<timer.elapsedTimeUs()<<" usec\n";
  
  bool test_passed = true;
  dbg_img.setTo(cv::Scalar(0));
  int w = cam_model.imgWidth(), h = cam_model.imgHeight(); 
  for(int i = 0; i < num_samples; i++)
  {
    cv::Point_<point_2Dtype> pt_cv = proj_pts_cv[i];

    cv::Point_<point_2Dtype> diff1(proj_pts_sp_parallel[i] - pt_cv);
    cv::Point_<point_2Dtype> diff2(proj_pts_sp_single[i] - pt_cv);
    
    double dist1 = sqrt(diff1.ddot(diff1)), dist2 = sqrt(diff2.ddot(diff2));
    if( dist1 > 1e-7 || dist2 > 1e-7)
    {
      cout<<"Warning!!! PinholeSceneProjector vs OpenCV projected points difference :"<<endl
              <<"parallel : "<<endl<<proj_pts_sp_parallel[i] - pt_cv<<endl
              <<"single thread : "<<endl<<proj_pts_sp_single[i] - pt_cv<<endl;
    
      test_passed = false;
    }
    
    if( pt_cv.x > 0 && pt_cv.y > 0 && pt_cv.x < w - 1 && pt_cv.y < h - 1 )
      dbg_img.at<cv::Vec3b>(cvRound(proj_pts_cv[i].y), cvRound(proj_pts_cv[i].x)) += cv::Vec3b(255,0,0);
    
    if( proj_pts_sp_parallel[i].x > 0 && proj_pts_sp_parallel[i].y > 0 && 
        proj_pts_sp_parallel[i].x < w - 1 && proj_pts_sp_parallel[i].y  < h - 1 )
      dbg_img.at<cv::Vec3b>(cvRound(proj_pts_sp_parallel[i].y), cvRound(proj_pts_sp_parallel[i].x)) += cv::Vec3b(0,255,0);
    
    if( proj_pts_sp_single[i].x > 0 && proj_pts_sp_single[i].y > 0 && 
        proj_pts_sp_single[i].x < w - 1 && proj_pts_sp_single[i].y  < h - 1 )
      dbg_img.at<cv::Vec3b>(cvRound(proj_pts_sp_single[i].y), cvRound(proj_pts_sp_single[i].x)) += cv::Vec3b(0,0,255);
  }
  
  cv_ext::showImage(dbg_img,"Projected points");
  
  
  cv_ext::PinholeCameraModel scaled_cam_model;  
  const double scale_factor = 0.2 + 5*static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
  
  scaled_cam_model = cam_model;
  scaled_cam_model.setSizeScaleFactor(scale_factor);
  cout<<"Set scale factor : "<<scale_factor<<endl;
  
  if( scaled_cam_model != cam_model )
    cout<<"Warning!!! PinholeSceneProjector inequality failed!:"<<endl;
  cv_ext::PinholeSceneProjector scaled_s_proj( scaled_cam_model );
                                
  vector< cv::Point_<point_2Dtype> > proj_pts_sp_scaled;

  double i_scale_factor = 1.0/scale_factor;
  double acc_threshold = sizeof(point_2Dtype)>4?1e-7:1e-3;
  scaled_s_proj.setTransformation(r_vec, t_vec );
  scaled_s_proj.projectPoints( sample_pts, proj_pts_sp_scaled );
  for(int i = 0; i < num_samples; i++)
  {
    cv::Point_<point_2Dtype> pt_scaled;;
    pt_scaled.x = i_scale_factor*double(proj_pts_sp_single[i].x); 
    pt_scaled.y = i_scale_factor*double(proj_pts_sp_single[i].y); 
    cv::Point_<point_2Dtype> diff(pt_scaled - proj_pts_sp_scaled[i]);
    double dist = sqrt(diff.ddot(diff));
    if( dist > acc_threshold )
    {
      cout<<"Warning!!! PinholeSceneProjector with scale factor set does not work properly :"<<endl
               <<"defference : "<<endl<<pt_scaled - proj_pts_sp_scaled[i]<<endl;
    
      test_passed = false;
    }
  }

  vector< cv::Point_<point_2Dtype> > norm_pts_sp_single, norm_pts_sp_parallel, 
                                          dist_pts_sp_single, dist_pts_sp_parallel;
  
  // OpenCV does not support different types
  vector< cv::Point_<point_3Dtype> > norm_pts_cv, dist_pts_cv;
  
  s_proj.enableParallelism( true );
  timer.reset();
  s_proj.normalizePoints( proj_pts_sp_parallel, norm_pts_sp_parallel );
  cout<<"PinholeSceneProjector normalization parallel: elapsedTime :"<<timer.elapsedTimeUs()<<" usec\n";
  
  s_proj.enableParallelism( false );
  timer.reset();
  s_proj.normalizePoints( proj_pts_sp_single, norm_pts_sp_single );
  cout<<"PinholeSceneProjector normalization single: elapsedTime :"<<timer.elapsedTimeUs()<<" usec\n";
  
  timer.reset();
  cv::undistortPoints(proj_pts_cv, norm_pts_cv, camera_matrix, dist_coeff );
  cout<<"OpenCV normalization: elapsedTime :"<<timer.elapsedTimeUs()<<" usec\n";

  s_proj.denormalizePoints( norm_pts_sp_single, dist_pts_sp_single );
  s_proj.denormalizePoints( norm_pts_sp_parallel, dist_pts_sp_parallel );
  // OpenCV does not provde points denormalization
  s_proj.denormalizePoints( norm_pts_cv, dist_pts_cv );
  
  int evaluated_pts = 0, opencv_better_accuracy = 0;
  for(int i = 0; i < num_samples; i++)
  {
    if( proj_pts_sp_single[i].x >= 0 && proj_pts_sp_single[i].y >= 0 && 
        proj_pts_sp_single[i].x < w && proj_pts_sp_single[i].y < h )
    {
      evaluated_pts++;
      cv::Point_<point_2Dtype> diff1(dist_pts_sp_parallel[i] - proj_pts_sp_parallel[i]);
      cv::Point_<point_2Dtype> diff2(dist_pts_sp_single[i] - proj_pts_sp_single[i]);
      cv::Point_<point_3Dtype> diff3(dist_pts_cv[i] - proj_pts_cv[i]);
      
      double dist1 = sqrt(diff1.ddot(diff1)), dist2 = sqrt(diff2.ddot(diff2)), dist3 = sqrt(diff3.ddot(diff3));
      if( dist1 > 1e-4 || dist2 > 1e-4)
      {
        cout<<"Warning!!! PinholeSceneProjector vs OpenCV normalized points difference :"<<endl
                <<"parallel : input point"<<proj_pts_sp_parallel[i]<<" diff "<<endl<<dist_pts_sp_parallel[i] - proj_pts_sp_parallel[i]<<endl
                <<"single thread : input point"<<proj_pts_sp_single[i]<<" diff "<<endl<<dist_pts_sp_single[i] - proj_pts_sp_single[i]<<endl;
      
        test_passed = false;
      }
      
      if( dist3 < dist1 || dist3 < dist2 )
        opencv_better_accuracy++;
    }
  }
  
  cout<<"CV_ext performs a better normalization compared with OpenCV on "<<evaluated_pts - opencv_better_accuracy
      <<" over "<<evaluated_pts<<" points ("<<100*double(evaluated_pts - opencv_better_accuracy)/evaluated_pts<<"%)"<<endl;
  
  cv_ext::PinholeCameraModel roi_cam_model;    
  roi_cam_model = cam_model;
  
  int roi_x = (cam_model.imgWidth() - 1)*static_cast <double> (rand()) / static_cast <double> (RAND_MAX), 
      roi_y = (cam_model.imgHeight() - 1)*static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
  int roi_width = (cam_model.imgWidth() - roi_x)*static_cast <double> (rand()) / static_cast <double> (RAND_MAX), 
      roi_height = (cam_model.imgHeight() - roi_y)*static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
  if( !roi_width ) roi_width=1;
  if( !roi_height ) roi_height=1;
  cv::Rect roi(roi_x, roi_y, roi_width, roi_height);
  roi_cam_model.setRegionOfInterest(roi);
  roi_cam_model.enableRegionOfInterest(true);
  cout<<"Set region of interest : "<<roi_cam_model.regionOfInterest()<<endl;
    
  cv_ext::PinholeSceneProjector roi_s_proj(roi_cam_model);
  roi_s_proj.setTransformation(r_vec, t_vec );
    
  vector< cv::Point_<point_2Dtype> > proj_pts_roi; 
  roi_s_proj.projectPoints( sample_pts, proj_pts_roi );

  for(int i = 0; i < num_samples; i++)
  {
    cv::Point_<point_2Dtype> pt_roi;
    pt_roi.x = proj_pts_sp_single[i].x - double(roi.x); 
    pt_roi.y = proj_pts_sp_single[i].y - double(roi.y);
    
    cv::Point_<point_2Dtype> diff(pt_roi - proj_pts_roi[i]);
    double dist = sqrt(diff.ddot(diff));
    if( dist > acc_threshold )
    {
      cout<<"Warning!!! PinholeSceneProjector with RoI set does not work properly :"<<endl
          <<"Expected point : "<<pt_roi<<" actual point "<<proj_pts_roi[i]<<" dist "<<dist<<endl;
    
      test_passed = false;
    }
  }
  
  if( !test_passed )
    cout<<"Warning!!! Test NOT passed!"<<endl;
  else
    cout<<"OK: Test passed"<<endl;
  
  return 0;
}
