#include <gtest/gtest.h>

#include "cv_ext/base.h"
#include "cv_ext/timer.h"
#include "cv_ext/debug_tools.h"
#include "cv_ext/camera_calibration.h"
#include "cv_ext/conversions.h"
#include "cv_ext/drawing.h"

#include "tests_utils.h"

TEST (CameraCalibrationTest, SingleCameraTest)
{
  const cv::Size img_size(1024, 768);
  const cv::Size board_size(9,7);
  const float s_len = 0.05;
  const int num_boards = 100   ;
  cv::Mat pattern_mask = cv_ext::getStandardPatternMask(board_size);
  pattern_mask.at<uchar>(0,0) = 1;

  const cv_ext::PinholeCameraModel cam_model = sampleCameraModel( img_size );

  cv_ext::CameraCalibration calib_ideal, calib;

  calib_ideal.setBoardData(board_size, s_len, pattern_mask);
  calib_ideal.setMaxNumIter(100);

  calib.setBoardData(board_size, s_len, pattern_mask);
  calib.setMaxNumIter(100);

  double t_std_dev = 0.2, r_std_dev = 0.1;
  cv::Mat cb_img;
  for( int i = 0; i < num_boards; i++ )
  {
    cv::Mat_<double> r_vec(3,1), t_vec(3,1);

    r_vec(0) = cv_ext::sampleGaussian( 0, r_std_dev );
    r_vec(1) = cv_ext::sampleGaussian( 0, r_std_dev );
    r_vec(2) = cv_ext::sampleGaussian( 0, r_std_dev );

    t_vec(0) = cv_ext::sampleGaussian( 0, t_std_dev );
    t_vec(1) = cv_ext::sampleGaussian( 0, t_std_dev );
    t_vec(2) = cv_ext::sampleGaussian( 1.0, t_std_dev );

    auto corner_pts = generateCheckerboardImage(cb_img, cam_model, board_size, s_len, r_vec, t_vec, pattern_mask );

    if( corner_pts.size() )
      calib_ideal.addImage(cb_img, corner_pts );

    calib.addImage(cb_img);
  }

  ASSERT_TRUE(calib_ideal.numCheckerboards() > 0);

  GTEST_COUT<< "Calibrating with ideal corners..."<<std::endl;
  GTEST_COUT<< "...done! RMS Error : "<<calib_ideal.calibrate()<<std::endl;

  cv_ext::PinholeCameraModel est_cam_model = calib_ideal.getCamModel();

  ASSERT_TRUE( fabs( est_cam_model.fx() - cam_model.fx()) < 1.0 );
  ASSERT_TRUE( fabs( est_cam_model.fy() - cam_model.fy()) < 1.0 );
  ASSERT_TRUE( fabs( est_cam_model.cx() - cam_model.cx()) < 1.0 );
  ASSERT_TRUE( fabs( est_cam_model.cy() - cam_model.cy()) < 1.0 );

  ASSERT_TRUE( fabs( est_cam_model.distK0() - cam_model.distK0()) < .001 );
  ASSERT_TRUE( fabs( est_cam_model.distK1() - cam_model.distK1()) < .001 );
  ASSERT_TRUE( fabs( est_cam_model.distK2() - cam_model.distK2()) < .001 );
  ASSERT_TRUE( fabs( est_cam_model.distPx() - cam_model.distPx()) < .001 );
  ASSERT_TRUE( fabs( est_cam_model.distPy() - cam_model.distPy()) < .001 );

  ASSERT_TRUE(calib.numCheckerboards() > 0);

  GTEST_COUT<< "Calibrating with extracted corners..."<<std::endl;
  GTEST_COUT<< "...done! RMS Error : "<<calib.calibrate()<<std::endl;

  est_cam_model = calib.getCamModel();

  ASSERT_TRUE( fabs( est_cam_model.fx() - cam_model.fx()) < 20.0 );
  ASSERT_TRUE( fabs( est_cam_model.fy() - cam_model.fy()) < 20.0 );
  ASSERT_TRUE( fabs( est_cam_model.cx() - cam_model.cx()) < 10.0 );
  ASSERT_TRUE( fabs( est_cam_model.cy() - cam_model.cy()) < 10.0 );

  ASSERT_TRUE( fabs( est_cam_model.distK0() - cam_model.distK0()) < .5 );
  ASSERT_TRUE( fabs( est_cam_model.distK1() - cam_model.distK1()) < .5 );
  ASSERT_TRUE( fabs( est_cam_model.distK2() - cam_model.distK2()) < .5 );
  ASSERT_TRUE( fabs( est_cam_model.distPx() - cam_model.distPx()) < .5 );
  ASSERT_TRUE( fabs( est_cam_model.distPy() - cam_model.distPy()) < .5 );

}

TEST (CameraCalibrationTest, StereoCameraTest)
{
  const cv::Size img_size(1024, 768);
  const cv::Size board_size(9,7);
  const float s_len = 0.05;
  const int num_boards = 100;
  cv::Mat pattern_mask = cv_ext::getStandardPatternMask(board_size);
  pattern_mask.at<uchar>(0,0) = 1;

  std::vector<cv_ext::PinholeCameraModel> cam_models;
  cam_models.reserve(2);
  for( int i = 0; i < 2; i++ )
    cam_models.emplace_back( sampleCameraModel( img_size ) );

  cv::Mat_<double> stereo_r_vec = (cv::Mat_<double>(3,1) << 0,0,0),
                   stereo_t_vec = (cv::Mat_<double>(3,1) << -0.2,0,0);

  stereo_r_vec(0) = stereo_r_vec(0) + cv_ext::sampleGaussian( 0, 0.005 );
  stereo_r_vec(1) = stereo_r_vec(1) + cv_ext::sampleGaussian( 0, 0.005 );
  stereo_r_vec(2) = stereo_r_vec(2) + cv_ext::sampleGaussian( 0, 0.01 );

  stereo_t_vec(0) = stereo_t_vec(0) + cv_ext::sampleGaussian( 0, 0.01 );
  stereo_t_vec(1) = stereo_t_vec(1) + cv_ext::sampleGaussian( 0, 0.01 );
  stereo_t_vec(2) = stereo_t_vec(2) + cv_ext::sampleGaussian( 0, 0.01 );


  cv_ext::StereoCameraCalibration calib_ideal, calib;

  calib_ideal.setBoardData(board_size, s_len, pattern_mask);
  calib_ideal.setCamModels(cam_models);
  calib_ideal.setMaxNumIter(100);

  calib.setBoardData(board_size, s_len, pattern_mask);
  calib.setCamModels(cam_models);
  calib.setMaxNumIter(100);


  double t_std_dev = 0.2, r_std_dev = 0.1;
  std::vector< cv::Mat > cb_imgs(2);
  std::vector< std::vector< cv::Point2f > > corners(2);
  for( int i = 0; i < num_boards; i++ )
  {
    cv::Mat_<double> r_vec(3,1), t_vec(3,1);

    r_vec(0) = cv_ext::sampleGaussian( 0, r_std_dev );
    r_vec(1) = cv_ext::sampleGaussian( 0, r_std_dev );
    r_vec(2) = cv_ext::sampleGaussian( 0, r_std_dev );

    t_vec(0) = cv_ext::sampleGaussian( 0, t_std_dev );
    t_vec(1) = cv_ext::sampleGaussian( 0, t_std_dev );
    t_vec(2) = cv_ext::sampleGaussian( 1.0, t_std_dev );

    cv::Mat pattern_mask = cv_ext::getStandardPatternMask(board_size);
    pattern_mask.at<uchar>(0,0) = 1;
    corners[0] = generateCheckerboardImage(cb_imgs[0], cam_models[0], board_size, s_len, r_vec, t_vec, pattern_mask );

    cv::Mat_<double> tmp_r_mat(3,3), stereo_r_mat(3, 3), tmp_r_vec(3, 1), tmp_t_vec(3, 1);
    cv_ext::angleAxis2RotMat<double>( r_vec, tmp_r_mat);
    cv_ext::angleAxis2RotMat<double>( stereo_r_vec, stereo_r_mat );

    tmp_r_mat = stereo_r_mat*tmp_r_mat;
    tmp_t_vec = stereo_r_mat*t_vec;
    tmp_t_vec += stereo_t_vec;

    cv_ext::rotMat2AngleAxis<double>( tmp_r_mat, tmp_r_vec );

    corners[1] = generateCheckerboardImage( cb_imgs[1], cam_models[1], board_size, s_len, tmp_r_vec, tmp_t_vec, pattern_mask );

    bool add_pts = corners[0].size() && corners[1].size();
    if( add_pts )
     calib_ideal.addImagePair(cb_imgs, corners);

    calib.addImagePair(cb_imgs);
  }

  ASSERT_TRUE(calib_ideal.numCheckerboards() > 0);

  GTEST_COUT<< "Calibrating with ideal corners..."<<std::endl;
  GTEST_COUT<< "...done! Error : "<<calib_ideal.calibrate()<<std::endl;

  cv::Mat est_r_mat, est_t_vec;
  calib_ideal.getExtrinsicsParameters(est_r_mat, est_t_vec );
  cv::Mat_<double> stereo_r_mat(3,3);
  cv_ext::angleAxis2RotMat<double>( stereo_r_vec, stereo_r_mat);

  double r_dist = cv_ext::rotationDist(stereo_r_mat, est_r_mat);
  double t_dist = cv::norm(stereo_t_vec, est_t_vec);

//  std::cout<<r_dist<<" "<<t_dist<<std::endl;

  ASSERT_TRUE( r_dist < 1e-6 );
  ASSERT_TRUE( t_dist < 1e-6 );


  ASSERT_TRUE(calib.numCheckerboards() > 0);

  GTEST_COUT<< "Calibrating with extracted corners..."<<std::endl;
  GTEST_COUT<< "...done! Error : "<<calib.calibrate()<<std::endl;

  calib.getExtrinsicsParameters(est_r_mat, est_t_vec );

  r_dist = cv_ext::rotationDist(stereo_r_mat, est_r_mat);
  t_dist = cv::norm(stereo_t_vec, est_t_vec);

//  std::cout<<r_dist<<" "<<t_dist<<std::endl;


//  cv::Mat_<cv::Vec3b> stereo_display_imgs(cam_models[0].imgHeight(), 2*cam_models[0].imgWidth());
//  std::vector< cv::Mat > display_imgs(2);
//  display_imgs[0] = stereo_display_imgs.colRange(0,cam_models[0].imgWidth());
//  display_imgs[1] = stereo_display_imgs.colRange(cam_models[0].imgWidth(), 2*cam_models[0].imgWidth());
//
//  for(int i = 0; i < calib_ideal.numImagePairs(); i++)
//  {
//    calib.getCornersImagePair(i, display_imgs);
//    cv_ext::showImage(stereo_display_imgs);
//  }


  ASSERT_TRUE( r_dist < 0.001 );
  ASSERT_TRUE( t_dist < 0.001 );

}

TEST (CameraCalibrationTest, MultiStereoCameraTestPair)
{
  const cv::Size img_size(1024, 768);
  const cv::Size board_size(9,7);
  const float s_len = 0.05;
  const int num_boards = 100;
  cv::Mat pattern_mask = cv_ext::getStandardPatternMask(board_size);
  pattern_mask.at<uchar>(0,0) = 1;

  std::vector<cv_ext::PinholeCameraModel> cam_models;
  cam_models.reserve(2);
  for( int i = 0; i < 2; i++ )
    cam_models.emplace_back( sampleCameraModel( img_size ) );

  cv::Mat_<double> stereo_r_vec = (cv::Mat_<double>(3,1) << 0,0,0),
                   stereo_t_vec = (cv::Mat_<double>(3,1) << -0.2,0,0);

  stereo_r_vec(0) = stereo_r_vec(0) + cv_ext::sampleGaussian( 0, 0.005 );
  stereo_r_vec(1) = stereo_r_vec(1) + cv_ext::sampleGaussian( 0, 0.005 );
  stereo_r_vec(2) = stereo_r_vec(2) + cv_ext::sampleGaussian( 0, 0.01 );

  stereo_t_vec(0) = stereo_t_vec(0) + cv_ext::sampleGaussian( 0, 0.01 );
  stereo_t_vec(1) = stereo_t_vec(1) + cv_ext::sampleGaussian( 0, 0.01 );
  stereo_t_vec(2) = stereo_t_vec(2) + cv_ext::sampleGaussian( 0, 0.01 );

  cv_ext::MultiStereoCameraCalibration calib_ideal(2), calib(2);

  calib_ideal.setBoardData(board_size, s_len, pattern_mask);
  calib_ideal.setCamModels(cam_models);
  calib_ideal.setMaxNumIter(100);

  calib.setBoardData(board_size, s_len, pattern_mask);
  calib.setCamModels(cam_models);
  calib.setMaxNumIter(100);

  double t_std_dev = 0.2, r_std_dev = 0.1;
  std::vector< cv::Mat > cb_imgs(2);
  std::vector< std::vector< cv::Point2f > > corners(2);
  for( int i = 0; i < num_boards; i++ )
  {
    cv::Mat_<double> r_vec(3,1), t_vec(3,1);

    r_vec(0) = cv_ext::sampleGaussian( 0, r_std_dev );
    r_vec(1) = cv_ext::sampleGaussian( 0, r_std_dev );
    r_vec(2) = cv_ext::sampleGaussian( 0, r_std_dev );

    t_vec(0) = cv_ext::sampleGaussian( 0, t_std_dev );
    t_vec(1) = cv_ext::sampleGaussian( 0, t_std_dev );
    t_vec(2) = cv_ext::sampleGaussian( 1.0, t_std_dev );

    cv::Mat pattern_mask = cv_ext::getStandardPatternMask(board_size);
    pattern_mask.at<uchar>(0,0) = 1;
    corners[0] = generateCheckerboardImage(cb_imgs[0], cam_models[0], board_size, s_len, r_vec, t_vec, pattern_mask );

    cv::Mat_<double> tmp_r_mat(3,3), stereo_r_mat(3, 3), tmp_r_vec(3, 1), tmp_t_vec(3, 1);
    cv_ext::angleAxis2RotMat<double>( r_vec, tmp_r_mat);
    cv_ext::angleAxis2RotMat<double>( stereo_r_vec, stereo_r_mat );

    tmp_r_mat = stereo_r_mat*tmp_r_mat;
    tmp_t_vec = stereo_r_mat*t_vec;
    tmp_t_vec += stereo_t_vec;

    cv_ext::rotMat2AngleAxis<double>( tmp_r_mat, tmp_r_vec );

    corners[1] = generateCheckerboardImage( cb_imgs[1], cam_models[1], board_size, s_len, tmp_r_vec, tmp_t_vec, pattern_mask );

    bool add_pts = corners[0].size() && corners[1].size();
    if( add_pts )
      calib_ideal.addImageTuple(cb_imgs, corners);

    calib.addImageTuple(cb_imgs);
  }

  ASSERT_TRUE(calib_ideal.numCheckerboards() > 0);

  GTEST_COUT<< "Calibrating with ideal corners..."<<std::endl;
  GTEST_COUT<< "...done! Error : "<<calib_ideal.calibrate()<<std::endl;

  std::vector < cv::Mat > est_r_mats, est_t_vecs;
  calib_ideal.getExtrinsicsParameters(est_r_mats, est_t_vecs );
  cv::Mat_<double> stereo_r_mat(3,3);
  cv_ext::angleAxis2RotMat<double>( stereo_r_vec, stereo_r_mat);

  double r_dist = cv_ext::rotationDist(stereo_r_mat, est_r_mats[1]);
  double t_dist = cv::norm(stereo_t_vec, est_t_vecs[1]);

//  std::cout<<r_dist<<" "<<t_dist<<std::endl;

  ASSERT_TRUE( r_dist < 1e-6 );
  ASSERT_TRUE( t_dist < 1e-6 );


  ASSERT_TRUE(calib.numCheckerboards() > 0);

  GTEST_COUT<< "Calibrating with extracted corners..."<<std::endl;
  GTEST_COUT<< "...done! Error : "<<calib.calibrate()<<std::endl;

  calib.getExtrinsicsParameters(est_r_mats, est_t_vecs );

  r_dist = cv_ext::rotationDist(stereo_r_mat, est_r_mats[1]);
  t_dist = cv::norm(stereo_t_vec, est_t_vecs[1]);

//  std::cout<<r_dist<<" "<<t_dist<<std::endl;

  ASSERT_TRUE( r_dist < 0.001 );
  ASSERT_TRUE( t_dist < 0.001 );

}

TEST (CameraCalibrationTest, MultiStereoCameraTestRig)
{
  const cv::Size img_size(1024, 768);
  const cv::Size board_size(9, 7);
  const float s_len = 0.05;
  const int num_boards = 100;
  const int num_cameras = 8;
  cv::Mat pattern_mask = cv_ext::getStandardPatternMask(board_size);
  pattern_mask.at<uchar>(0,0) = 1;

  std::vector<cv_ext::PinholeCameraModel> cam_models;
  cam_models.reserve(num_cameras);
  for (int i = 0; i < num_cameras; i++)
    cam_models.emplace_back( sampleCameraModel( img_size ) );

  cv_ext::MultiStereoCameraCalibration calib_ideal(num_cameras), calib(num_cameras);

  calib_ideal.setBoardData(board_size, s_len, pattern_mask);
  calib_ideal.setCamModels(cam_models);
  calib_ideal.setMaxNumIter(100);

  calib.setBoardData(board_size, s_len, pattern_mask);
  calib.setCamModels(cam_models);
  calib.setMaxNumIter(100);

  std::vector<cv::Mat_<double> > rig_r_vec(num_cameras), rig_r_mat(num_cameras), rig_t_vec(num_cameras);

  // Build a regular circular array of cameras with distance 1 to the center
  // Compute all the camera rig transformations referred to the the first position,
  // i.e., the reference position, with R = I and t = [0,0,0]'
  cv::Mat_<double> ref_t_vec = (cv::Mat_<double>(3, 1) << 0, 0, 1.0);
  for (int i = 0; i < num_cameras; i++)
  {
    rig_r_vec[i] = (cv::Mat_<double>(3, 1) << 0, i * 2 * M_PI / num_cameras, 0);
    rig_r_mat[i].create(3, 3);
    cv_ext::angleAxis2RotMat<double>( rig_r_vec[i], rig_r_mat[i] );

    rig_t_vec[i] = -rig_r_mat[i] * ref_t_vec + ref_t_vec;
  }

  for( int i = 0; i < num_boards; i++ )
  {
    cv::Mat_<double> r_vec = (cv::Mat_<double>(3, 1) << 0, 0, 0), r_mat(3, 3), t_vec(3, 1);

    r_vec(1) = cv_ext::sampleUniform(-M_PI, M_PI);

    r_vec(0) += cv_ext::sampleGaussian(0, 0.2);
    r_vec(2) += cv_ext::sampleGaussian(0, 0.2);

    cv_ext::angleAxis2RotMat<double>( r_vec, r_mat );

    t_vec(0) = cv_ext::sampleGaussian(0, 0.1);
    t_vec(1) = cv_ext::sampleGaussian(0, 0.1);
    t_vec(2) = cv_ext::sampleGaussian(1.0, 0.05);

    cv::Mat pattern_mask = cv_ext::getStandardPatternMask(board_size);
    pattern_mask.at<uchar>(0,0) = 1;

    std::vector< cv::Mat > cb_imgs(num_cameras);
    std::vector< std::vector< cv::Point2f > > corners(num_cameras);
    cv::Mat_<double> cam_r_mat(3, 3), cam_r_vec(3, 1), cam_t_vec(3, 1);
    for (int j = 0; j < num_cameras; j++)
    {
      cam_r_mat = rig_r_mat[j] * r_mat;
      cv_ext::rotMat2AngleAxis<double>( cam_r_mat, cam_r_vec );
      cam_t_vec = rig_r_mat[j] * t_vec + rig_t_vec[j];
      corners[j] = generateCheckerboardImage(cb_imgs[j], cam_models[j], board_size, s_len, cam_r_vec, cam_t_vec, pattern_mask);
    }

    int valid_cams = 0;
    for( int k = 0; k < num_cameras; k++ )
      valid_cams += static_cast<int>( corners[k].size() ) != 0;

    if( valid_cams > 1 )
      calib_ideal.addImageTuple(cb_imgs, corners);

    calib.addImageTuple(cb_imgs);
  }

  ASSERT_TRUE(calib_ideal.numCheckerboards() > 0);

  GTEST_COUT<< "Calibrating with ideal corners..."<<std::endl;
  GTEST_COUT<< "...done! Error : "<<calib_ideal.calibrate()<<std::endl;

  std::vector<cv::Mat> est_r_mats, est_t_vecs;

  calib_ideal.getExtrinsicsParameters(est_r_mats, est_t_vecs );

  for (int i = 0; i < num_cameras; i++)
  {
    double r_dist = cv_ext::rotationDist(rig_r_mat[i], est_r_mats[i]);
    double t_dist = cv::norm(rig_t_vec[i], est_t_vecs[i]);

//    std::cout<<r_dist<<" "<<t_dist<<std::endl;

    ASSERT_TRUE( r_dist < 1e-6 );
    ASSERT_TRUE( t_dist < 1e-6 );
  }

  ASSERT_TRUE(calib.numCheckerboards() > 0);

  GTEST_COUT<< "Calibrating with extracted corners..."<<std::endl;
  GTEST_COUT<< "...done! Error : "<<calib.calibrate()<<std::endl;

  calib.getExtrinsicsParameters(est_r_mats, est_t_vecs );

  for (int i = 0; i < num_cameras; i++)
  {
    double r_dist = cv_ext::rotationDist(rig_r_mat[i], est_r_mats[i]);
    double t_dist = cv::norm(rig_t_vec[i], est_t_vecs[i]);

//    std::cout<<r_dist<<" "<<t_dist<<std::endl;

    ASSERT_TRUE( r_dist < 0.002 );
    ASSERT_TRUE( t_dist < 0.002 );
  }

//  cv::Mat_<cv::Vec3b> rig_display_imgs(cam_models[0].imgHeight() / 2, cam_models[0].imgWidth());
//  std::vector<cv::Mat> display_imgs(8);
//  int step_x = cam_models[0].imgWidth() / 4, step_y = cam_models[0].imgHeight() / 4;
//  for (int i = 0, start_x = 0; i < 4; i++, start_x += step_x)
//    display_imgs[i] = rig_display_imgs(cv::Rect(start_x, 0, step_x, step_y));
//  for (int i = 4, start_x = 0; i < 8; i++, start_x += step_x)
//    display_imgs[i] = rig_display_imgs(cv::Rect(start_x, step_y, step_x, step_y));
//
//  for(int i = 0; i < calib_ideal.numImageTuples(); i++)
//  {
//    std::vector <cv::Mat> cb_imgs;
//    calib_ideal.getCornersImageTuple(i, cb_imgs);
//    for( int j = 0; j < num_cameras; j++ )
//      cv::resize(cb_imgs[j], display_imgs[j], display_imgs[j].size());
//
//    cv_ext::showImage(rig_display_imgs, "", false);
//  }
}

TEST (CameraCalibrationTest, MultiStereoCameraTestRigMultiBoard)
{
  const cv::Size img_size(1024, 768);
  const cv::Size board_size(7, 13);
  const float s_len = 0.034;
  const int num_views = 50;
  const int num_cameras = 8;
  cv::Mat pattern_mask = cv_ext::getStandardPatternMask(board_size);
  pattern_mask.at<uchar>(0,0) = 1;

  std::vector<cv_ext::PinholeCameraModel> cam_models;
  cam_models.reserve(num_cameras);
  for (int i = 0; i < num_cameras; i++)
    cam_models.emplace_back( sampleCameraModel( img_size ) );

  const int c_pattern_n_boards = 3;
  std::vector< cv::Mat > pattern_masks(c_pattern_n_boards);

  for (int k = 0; k < c_pattern_n_boards; k++)
  {
    pattern_masks[k] = cv_ext::getStandardPatternMask(board_size);
    pattern_masks[k].at<uchar>(0,0) = 1;
  }

  pattern_masks[0].at<uchar>(2,0) = 1;
  pattern_masks[0].at<uchar>(0,2) = 1;

  pattern_masks[1].at<uchar>(4,0) = 1;
  pattern_masks[1].at<uchar>(0,4) = 1;

  pattern_masks[2].at<uchar>(11,3) = 1;
  pattern_masks[2].at<uchar>(9,5) = 1;

  cv_ext::MultiStereoCameraCalibration calib_ideal(num_cameras), calib(num_cameras);

  calib_ideal.setMultipleBoardsData(board_size, s_len, pattern_masks);
  calib_ideal.setCamModels(cam_models);
  calib_ideal.setMaxNumIter(100);

  calib.setMultipleBoardsData(board_size, s_len, pattern_masks);
  calib.setCamModels(cam_models);
  calib.setMaxNumIter(100);

  std::vector<cv::Mat_<double> > rig_r_vec(num_cameras), rig_r_mat(num_cameras), rig_t_vec(num_cameras);

  // Build a regular circular array of cameras with distance 1 to the center
  // Compute all the camera rig transformations referred to the the first position,
  // i.e., the reference position, with R = I and t = [0,0,0]'
  cv::Mat_<double> ref_t_vec = (cv::Mat_<double>(3, 1) << 0, 0, 1.0);
  for (int i = 0; i < num_cameras; i++)
  {
    rig_r_vec[i] = (cv::Mat_<double>(3, 1) << i * 2 * M_PI / num_cameras, 0, 0);
    rig_r_mat[i].create(3, 3);
    cv_ext::angleAxis2RotMat<double>( rig_r_vec[i], rig_r_mat[i] );

    rig_t_vec[i] = -rig_r_mat[i] * ref_t_vec + ref_t_vec;
  }

  const double y_pattern_rot = M_PI/8, x_translation =.3, z_translation = .05;
  std::vector< cv::Mat_<double> > boards_init_r_mat(c_pattern_n_boards),
                                  boards_r_mat(c_pattern_n_boards),
                                  boards_init_t_vec(c_pattern_n_boards),
                                  boards_t_vec(c_pattern_n_boards);

  cv::Mat_<double> boards_r_vec = (cv::Mat_<double>(3, 1) << 0, 0, 0);
  cv_ext::angleAxis2RotMat<double>( boards_r_vec, boards_init_r_mat[0] );
  boards_r_vec(1) = y_pattern_rot;
  cv_ext::angleAxis2RotMat<double>( boards_r_vec, boards_init_r_mat[1] );
  boards_r_vec(1) = -y_pattern_rot;
  cv_ext::angleAxis2RotMat<double>( boards_r_vec, boards_init_r_mat[2] );

  boards_init_t_vec[0] = (cv::Mat_<double>(3, 1) << 0, 0, 0);
  boards_init_t_vec[1] = (cv::Mat_<double>(3, 1) << x_translation, 0, -z_translation);
  boards_init_t_vec[2] = (cv::Mat_<double>(3, 1) << -x_translation, 0, -z_translation);

  cv::Mat_<double> calib_pattern_r_vec = (cv::Mat_<double>(3, 1) << 0, 0, 0), calib_pattern_r_mat(3, 3),
      calib_pattern_t_vec = (cv::Mat_<double>(3, 1) << 0, 0, 1);

  for(int i = 0; i < num_views; i++ )
  {
    calib_pattern_r_vec(0) = i * 2 * M_PI/num_views;
    cv_ext::angleAxis2RotMat<double>( calib_pattern_r_vec, calib_pattern_r_mat );

    for( int j = 0; j < c_pattern_n_boards; j++)
    {
      boards_r_mat[j] = calib_pattern_r_mat*boards_init_r_mat[j];
      boards_t_vec[j] = calib_pattern_r_mat*boards_init_t_vec[j] + calib_pattern_t_vec;
    }


    std::vector< cv::Mat > cb_imgs(num_cameras);
    for (int j = 0; j < num_cameras; j++)
    {
      cb_imgs[j].create(cam_models[j].imgSize(), cv::DataType<uchar>::type);
      cb_imgs[j].setTo(cv::Scalar(128));
    }

    std::vector< std::vector< std::vector< cv::Point2f > > >tmp_corners(c_pattern_n_boards);
    cv::Mat_<double> cam_r_mat(3, 3), cam_r_vec(3, 1), cam_t_vec(3, 1);
    for (int j = 0; j < num_cameras; j++)
    {
      for (int k = 0; k < c_pattern_n_boards; k++)
      {
        cam_r_mat = rig_r_mat[j] * boards_r_mat[k];
        cv_ext::rotMat2AngleAxis<double>(cam_r_mat, cam_r_vec);
        cam_t_vec = rig_r_mat[j] * boards_t_vec[k] + rig_t_vec[j];
        tmp_corners[k].emplace_back(generateCheckerboardImage(cb_imgs[j], cam_models[j], board_size, s_len,
                                                              cam_r_vec, cam_t_vec, pattern_masks[k], false));
      }
    }

    std::vector< std::vector< cv::Point2f > > corners;
    for (int k = 0; k < c_pattern_n_boards; k++)
    {
      int valid_cams = 0;
      for (int j = 0; j < num_cameras; j++)
        valid_cams += static_cast<int>( tmp_corners[k][j].size() ) != 0;

      if( valid_cams > 1 )
        corners.insert(corners.begin(), tmp_corners[k].begin(), tmp_corners[k].end());
    }

    if( corners.size() )
      calib_ideal.addImageTuple(cb_imgs, corners);

    calib.addImageTuple(cb_imgs);
  }

  ASSERT_TRUE(calib_ideal.numCheckerboards() > 0);

  GTEST_COUT<< "Calibrating with ideal corners..."<<std::endl;
  GTEST_COUT<< "...done! Error : "<<calib_ideal.calibrate()<<std::endl;

  std::vector<cv::Mat> est_r_mats, est_t_vecs;

  calib_ideal.getExtrinsicsParameters(est_r_mats, est_t_vecs );

  for (int i = 0; i < num_cameras; i++)
  {
    double r_dist = cv_ext::rotationDist(rig_r_mat[i], est_r_mats[i]);
    double t_dist = cv::norm(rig_t_vec[i], est_t_vecs[i]);

//    std::cout<<r_dist<<" "<<t_dist<<std::endl;

    ASSERT_TRUE( r_dist < 1e-6 );
    ASSERT_TRUE( t_dist < 1e-6 );
  }


  ASSERT_TRUE(calib.numCheckerboards() > 0);

  GTEST_COUT<< "Calibrating with extracted corners..."<<std::endl;
  GTEST_COUT<< "...done! Error : "<<calib.calibrate()<<std::endl;

  calib.getExtrinsicsParameters(est_r_mats, est_t_vecs );

  for (int i = 0; i < num_cameras; i++)
  {
    double r_dist = cv_ext::rotationDist(rig_r_mat[i], est_r_mats[i]);
    double t_dist = cv::norm(rig_t_vec[i], est_t_vecs[i]);

//    std::cout<<r_dist<<" "<<t_dist<<std::endl;

    ASSERT_TRUE( r_dist < 0.002 );
    ASSERT_TRUE( t_dist < 0.002 );
  }

//  cv::Mat_<cv::Vec3b> rig_display_imgs(cam_models[0].imgHeight() / 2, cam_models[0].imgWidth());
//  std::vector<cv::Mat> display_imgs(8);
//  int step_x = cam_models[0].imgWidth() / 4, step_y = cam_models[0].imgHeight() / 4;
//  for (int i = 0, start_x = 0; i < 4; i++, start_x += step_x)
//    display_imgs[i] = rig_display_imgs(cv::Rect(start_x, 0, step_x, step_y));
//  for (int i = 4, start_x = 0; i < 8; i++, start_x += step_x)
//    display_imgs[i] = rig_display_imgs(cv::Rect(start_x, step_y, step_x, step_y));
//
//  for(int i = 0; i < calib.numCheckerboards(); i++)
//  {
//    std::vector <cv::Mat> cb_imgs;
//    calib.getCornersImageTuple(i, cb_imgs);
//    for( int j = 0; j < num_cameras; j++ )
//      cv::resize(cb_imgs[j], display_imgs[j], display_imgs[j].size());
//
//    cv_ext::showImage(rig_display_imgs, "", false);
//  }
}
