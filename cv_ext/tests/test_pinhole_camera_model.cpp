#include "cv_ext/pinhole_camera_model.h"
#include "cv_ext/cv_ext.h"

#include "tests_utils.h"

#include <gtest/gtest.h>
#include <vector>
#include <string>

#include <boost/filesystem.hpp>

// TODO Missing tests: normalize() method, setSizeScaleFactor(), clear(), ...

static void compareCameras( const cv_ext::PinholeCameraModel &cam0,
                            const cv_ext::PinholeCameraModel &cam1,
                            double tolerance = 1e-8 )
{
  ASSERT_TRUE( cam0.imgWidth() == cam1.imgWidth() );
  ASSERT_TRUE( cam0.imgHeight() == cam1.imgHeight() );
  ASSERT_TRUE( fabs( cam0.fx() - cam1.fx() ) < tolerance);
  ASSERT_TRUE( fabs( cam0.fy() - cam1.fy() ) < tolerance);
  ASSERT_TRUE( fabs( cam0.cx() - cam1.cx() ) < tolerance);
  ASSERT_TRUE( fabs( cam0.cy() - cam1.cy() ) < tolerance);
  ASSERT_TRUE( fabs( cam0.distK0() - cam1.distK0() ) < tolerance);
  ASSERT_TRUE( fabs( cam0.distK1() - cam1.distK1() ) < tolerance);
  ASSERT_TRUE( fabs( cam0.distK2() - cam1.distK2() ) < tolerance);
  ASSERT_TRUE( fabs( cam0.distK3() - cam1.distK3() ) < tolerance);
  ASSERT_TRUE( fabs( cam0.distK4() - cam1.distK4() ) < tolerance);
  ASSERT_TRUE( fabs( cam0.distK5() - cam1.distK5() ) < tolerance);
}

template < typename _T > static void testDenormalization( _T tolerance )
{
  const int num_samples = 10000;
  std::vector< cv::Point3_<_T> > sample_pts;
  sample_pts.reserve(num_samples);
  for(int i = 0; i < num_samples; i++)
  {
    sample_pts.template emplace_back( cv_ext::sampleUniform(-4, 4),
                                      cv_ext::sampleUniform(-4, 4),
                                      1.0 );
  }

  cv_ext::PinholeCameraModel cam_model = sampleCameraModel();
  cv::Mat camera_matrix = cv::Mat_<_T>( cam_model.cameraMatrix() );
  cv::Mat dist_coeff = cv::Mat_<_T>( cam_model.distorsionCoeff() );
  // Avoid to use calibration parameters with higher precision
  cam_model = cv_ext::PinholeCameraModel(camera_matrix, cam_model.imgWidth(), cam_model.imgHeight(), dist_coeff);

  std::vector< cv::Point_<_T> > proj_pts(num_samples), proj_pts_nd(num_samples),
                                gt_proj_pts(num_samples), gt_proj_pts_nd(num_samples);

  for(int i = 0; i < num_samples; i++)
  {
    cam_model.denormalize(reinterpret_cast<_T *>(&sample_pts[i]), reinterpret_cast<_T *>(&proj_pts[i]));
    cam_model.denormalizeWithoutDistortion(reinterpret_cast<_T *>(&sample_pts[i]),
                                           reinterpret_cast<_T *>(&proj_pts_nd[i]));
  }

  cv::Mat r_vec(cv::Mat_<_T>(3,1, 0.0)), t_vec(cv::Mat_<_T>(3,1, 0.0));
  cv::projectPoints(sample_pts, r_vec, t_vec, camera_matrix, dist_coeff, gt_proj_pts );
  cv::projectPoints(sample_pts, r_vec, t_vec, camera_matrix, cv::Mat(), gt_proj_pts_nd );

  int w = cam_model.imgWidth(), h = cam_model.imgHeight();
  for(int i = 0; i < num_samples; i++)
  {
    auto gt_pp = gt_proj_pts[i], gt_pp_nd = gt_proj_pts_nd[i];
    if ( gt_pp.x >= 0 && gt_pp.y >= 0 && gt_pp.x <= w - 1 && gt_pp.y <= h - 1)
    {
      ASSERT_TRUE(cv_ext::norm2D(proj_pts[i] - gt_pp) < tolerance);
    }

    if (gt_pp_nd.x >= 0 && gt_pp_nd.y >= 0 && gt_pp_nd.x <= w - 1 && gt_pp_nd.y <= h - 1)
    {
      ASSERT_TRUE(cv_ext::norm2D(proj_pts_nd[i] - gt_pp_nd) < tolerance);
    }
  }
}

template < typename _T > static void testNormalization( _T tolerance )
{
  const int num_samples = 10000;

  cv_ext::PinholeCameraModel cam_model = sampleCameraModel();
  cv::Point_<_T> gt_pt, proj_pt, proj_pt_nd, norm_pt, norm_pt_nd;

  for(int i = 0; i < num_samples; i++)
  {
    gt_pt = cv::Point_<_T>( cv_ext::sampleUniform(-0.5, 0.5), cv_ext::sampleUniform(-0.5, 0.5)  );

    cam_model.template denormalize(reinterpret_cast<_T *>(&gt_pt), reinterpret_cast<_T *>(&proj_pt));
    cam_model.template denormalizeWithoutDistortion(reinterpret_cast<_T *>(&gt_pt), reinterpret_cast<_T *>(&proj_pt_nd));
    cam_model.template normalize(reinterpret_cast<_T *>(&proj_pt), reinterpret_cast<_T *>(&norm_pt));
    cam_model.template normalizeWithoutDistortion(reinterpret_cast<_T *>(&proj_pt_nd), reinterpret_cast<_T *>(&norm_pt_nd));

    ASSERT_TRUE(cv_ext::norm2D(norm_pt - gt_pt) < tolerance);
    ASSERT_TRUE(cv_ext::norm2D(norm_pt_nd - gt_pt) < tolerance);
  }
}

template < typename _T > static void testProjection( _T tolerance )
{
  const int num_samples = 10000;
  const double max_disp = 5;
  std::vector< cv::Point3_<_T> > sample_pts;
  sample_pts.reserve(num_samples);
  for(int i = 0; i < num_samples; i++)
  {
    sample_pts.template emplace_back( cv_ext::sampleUniform(-max_disp, max_disp),
                                      cv_ext::sampleUniform(-max_disp, max_disp),
                                      cv_ext::sampleUniform(0, 2*max_disp) );
  }

  cv_ext::PinholeCameraModel cam_model = sampleCameraModel();
  cv::Mat camera_matrix = cv::Mat_<_T>( cam_model.cameraMatrix() );
  cv::Mat dist_coeff = cv::Mat_<_T>( cam_model.distorsionCoeff() );
  // Avoid to use calibration parameters with higher precision
  cam_model = cv_ext::PinholeCameraModel(camera_matrix, cam_model.imgWidth(), cam_model.imgHeight(), dist_coeff);

//  cv::Mat_<cv::Vec3b> dbg_img(cam_model.imgSize());
//  dbg_img.setTo(0);

  std::vector< cv::Point_<_T> > proj_pts1(num_samples), proj_pts2(num_samples),
                                proj_pts_nd1(num_samples), proj_pts_nd2(num_samples),
                                gt_proj_pts(num_samples), gt_proj_pts_nd(num_samples);
  std::vector< _T > depths(num_samples), depths_nd(num_samples);

  for(int i = 0; i < num_samples; i++)
  {
    cam_model.project(reinterpret_cast<_T *>(&sample_pts[i]), reinterpret_cast<_T *>(&proj_pts1[i]));
    cam_model.projectWithoutDistortion(reinterpret_cast<_T *>(&sample_pts[i]),
                                       reinterpret_cast<_T *>(&proj_pts_nd1[i]));
    cam_model.project(reinterpret_cast<_T *>(&sample_pts[i]), reinterpret_cast<_T *>(&proj_pts2[i]), depths[i]);
    cam_model.projectWithoutDistortion(reinterpret_cast<_T *>(&sample_pts[i]),
                                       reinterpret_cast<_T *>(&proj_pts_nd2[i]), depths_nd[i]);
  }

  cv::Mat r_vec(cv::Mat_<_T>(3,1, 0.0)), t_vec(cv::Mat_<_T>(3,1, 0.0));
  cv::projectPoints(sample_pts, r_vec, t_vec, camera_matrix, dist_coeff, gt_proj_pts );
  cv::projectPoints(sample_pts, r_vec, t_vec, camera_matrix, cv::Mat(), gt_proj_pts_nd );

  int w = cam_model.imgWidth(), h = cam_model.imgHeight();
  for(int i = 0; i < num_samples; i++)
  {
    auto gt_pp = gt_proj_pts[i], gt_pp_nd = gt_proj_pts_nd[i];
    if (gt_pp.x >= 0 && gt_pp.y >= 0 && gt_pp.x <= w - 1 && gt_pp.y <= h - 1)
    {
//      dbg_img.at<cv::Vec3b>(cvRound(proj_pts_cv[i].y), cvRound(proj_pts_cv[i].x)) = cv::Vec3b(0, 0, 255);
//      dbg_img.at<cv::Vec3b>(cvRound(proj_pts[i].y), cvRound(proj_pts[i].x)) = cv::Vec3b(0, 255, 0);
      ASSERT_TRUE(cv_ext::norm2D(proj_pts1[i] - gt_pp) < tolerance);
      ASSERT_TRUE(cv_ext::norm2D(proj_pts2[i] - gt_pp) < tolerance);
      ASSERT_TRUE(depths[i] == sample_pts[i].z);
    }
    if (gt_pp_nd.x >= 0 && gt_pp_nd.y >= 0 && gt_pp_nd.x <= w - 1 && gt_pp_nd.y <= h - 1)
    {
      ASSERT_TRUE(cv_ext::norm2D(proj_pts_nd1[i] - gt_pp_nd) < tolerance);
      ASSERT_TRUE(cv_ext::norm2D(proj_pts_nd2[i] - gt_pp_nd) < tolerance);
      ASSERT_TRUE(depths_nd[i] == sample_pts[i].z);
    }
  }
}


template < typename _T > static void testRTProjection( _T tolerance )
{
  const int num_samples = 10000;
  const double max_disp = 5;
  std::vector< cv::Point3_<_T> > sample_pts;
  sample_pts.reserve(num_samples);
  for(int i = 0; i < num_samples; i++)
  {
    sample_pts.template emplace_back( cv_ext::sampleUniform(-max_disp, max_disp),
                                      cv_ext::sampleUniform(-max_disp, max_disp),
                                      cv_ext::sampleUniform(-max_disp, max_disp) );
  }

  // Sample a rotation and translation
  Eigen::Matrix<_T, 3, 1> t_vec, axis_vec;
  _T angle = cv_ext::sampleUniform(-M_PI, M_PI);
  for( int i = 0; i < 3; i++ )
  {
    axis_vec(i) = cv_ext::sampleUniform(-1.0, 1.0);
    t_vec(i) = cv_ext::sampleUniform(-max_disp, max_disp);
  }
  axis_vec /= axis_vec.norm();
  Eigen::AngleAxis<_T> aa_vec(angle, axis_vec);

  Eigen::Matrix<_T, 3, 3> r_mat = aa_vec.toRotationMatrix();
  Eigen::Quaternion<_T> r_quat(aa_vec);
  Eigen::Matrix<_T, 3, 1> r_vec = axis_vec * angle;

  _T r_quat_array[4] = { r_quat.w(), r_quat.x(), r_quat.y(), r_quat.z() };

  cv::Mat_<_T> r_mat_cv, t_vec_cv;
  cv_ext::eigen2openCv(r_mat, r_mat_cv);
  cv_ext::eigen2openCv(t_vec, t_vec_cv);

  cv_ext::PinholeCameraModel cam_model = sampleCameraModel();
  cv::Mat camera_matrix = cv::Mat_<_T>( cam_model.cameraMatrix() );
  cv::Mat dist_coeff = cv::Mat_<_T>( cam_model.distorsionCoeff() );
  // Avoid to use calibration parameters with higher precision
  cam_model = cv_ext::PinholeCameraModel(camera_matrix, cam_model.imgWidth(), cam_model.imgHeight(), dist_coeff);

  std::vector< cv::Point_<_T> > proj_pts1(num_samples), proj_pts_nd1(num_samples),
                                proj_pts2(num_samples), proj_pts_nd2(num_samples),
                                proj_pts3(num_samples), proj_pts_nd3(num_samples),
                                proj_pts4(num_samples), proj_pts_nd4(num_samples),
                                gt_proj_pts(num_samples), gt_proj_pts_nd(num_samples);

  for(int i = 0; i < num_samples; i++)
  {
    cam_model.template rTProject( r_mat, t_vec, reinterpret_cast<_T *>(&sample_pts[i]),
                                  reinterpret_cast<_T *>(&proj_pts1[i]));
    cam_model.template rTProjectWithoutDistortion( r_mat, t_vec, reinterpret_cast<_T *>(&sample_pts[i]),
                                                   reinterpret_cast<_T *>(&proj_pts_nd1[i]));
    cam_model.template quatRTProject( r_quat_array, t_vec.data(), reinterpret_cast<_T *>(&sample_pts[i]),
                                      reinterpret_cast<_T *>(&proj_pts2[i]));
    cam_model.template quatRTProjectWithoutDistortion( r_quat_array, t_vec.data(), reinterpret_cast<_T *>(&sample_pts[i]),
                                                       reinterpret_cast<_T *>(&proj_pts_nd2[i]));
    cam_model.template quatRTProject( r_quat, t_vec, reinterpret_cast<_T *>(&sample_pts[i]),
                                      reinterpret_cast<_T *>(&proj_pts3[i]));
    cam_model.template quatRTProjectWithoutDistortion( r_quat, t_vec, reinterpret_cast<_T *>(&sample_pts[i]),
                                                       reinterpret_cast<_T *>(&proj_pts_nd3[i]));
    cam_model.template angAxRTProject( r_vec.data(), t_vec.data(), reinterpret_cast<_T *>(&sample_pts[i]),
                                       reinterpret_cast<_T *>(&proj_pts4[i]));
    cam_model.template angAxRTProjectWithoutDistortion( r_vec.data(), t_vec.data(), reinterpret_cast<_T *>(&sample_pts[i]),
                                                        reinterpret_cast<_T *>(&proj_pts_nd4[i]));
  }

  cv::projectPoints(sample_pts, r_mat_cv, t_vec_cv, camera_matrix, dist_coeff, gt_proj_pts );
  cv::projectPoints(sample_pts, r_mat_cv, t_vec_cv, camera_matrix, cv::Mat(), gt_proj_pts_nd );

  int w = cam_model.imgWidth(), h = cam_model.imgHeight();
  for(int i = 0; i < num_samples; i++)
  {
    auto gt_pp = gt_proj_pts[i], gt_pp_nd = gt_proj_pts_nd[i];
    if (gt_pp.x >= 0 && gt_pp.y >= 0 && gt_pp.x <= w - 1 && gt_pp.y <= h - 1)
    {
      ASSERT_TRUE(cv_ext::norm2D(proj_pts1[i] - gt_proj_pts[i]) < tolerance);
      ASSERT_TRUE(cv_ext::norm2D(proj_pts2[i] - gt_proj_pts[i]) < tolerance);
      ASSERT_TRUE(cv_ext::norm2D(proj_pts3[i] - gt_proj_pts[i]) < tolerance);
      ASSERT_TRUE(cv_ext::norm2D(proj_pts4[i] - gt_proj_pts[i]) < tolerance);
    }

    if (gt_pp_nd.x >= 0 && gt_pp_nd.y >= 0 && gt_pp_nd.x <= w - 1 && gt_pp_nd.y <= h - 1)
    {
      ASSERT_TRUE(cv_ext::norm2D(proj_pts_nd1[i] - gt_proj_pts_nd[i]) < tolerance);
      ASSERT_TRUE(cv_ext::norm2D(proj_pts_nd2[i] - gt_proj_pts_nd[i]) < tolerance);
      ASSERT_TRUE(cv_ext::norm2D(proj_pts_nd3[i] - gt_proj_pts_nd[i]) < tolerance);
      ASSERT_TRUE(cv_ext::norm2D(proj_pts_nd4[i] - gt_proj_pts_nd[i]) < tolerance);
    }
  }

  std::vector< _T > depths1(num_samples), depths_nd1(num_samples),
                    depths2(num_samples), depths_nd2(num_samples),
                    depths3(num_samples), depths_nd3(num_samples),
                    depths4(num_samples), depths_nd4(num_samples),
                    depths_gt(num_samples);

  for(int i = 0; i < num_samples; i++)
  {
    cam_model.template rTProject( r_mat, t_vec, reinterpret_cast<_T *>(&sample_pts[i]),
                                  reinterpret_cast<_T *>(&proj_pts1[i]), depths1[i]);
    cam_model.template rTProjectWithoutDistortion( r_mat, t_vec, reinterpret_cast<_T *>(&sample_pts[i]),
                                                   reinterpret_cast<_T *>(&proj_pts_nd1[i]), depths_nd1[i]);
    cam_model.template quatRTProject( r_quat_array, t_vec.data(), reinterpret_cast<_T *>(&sample_pts[i]),
                                      reinterpret_cast<_T *>(&proj_pts2[i]), depths2[i]);
    cam_model.template quatRTProjectWithoutDistortion( r_quat_array, t_vec.data(), reinterpret_cast<_T *>(&sample_pts[i]),
                                                       reinterpret_cast<_T *>(&proj_pts_nd2[i]), depths_nd2[i]);
    cam_model.template quatRTProject( r_quat, t_vec, reinterpret_cast<_T *>(&sample_pts[i]),
                                      reinterpret_cast<_T *>(&proj_pts3[i]), depths3[i]);
    cam_model.template quatRTProjectWithoutDistortion( r_quat, t_vec, reinterpret_cast<_T *>(&sample_pts[i]),
                                                       reinterpret_cast<_T *>(&proj_pts_nd3[i]), depths_nd3[i]);
    cam_model.template angAxRTProject( r_vec.data(), t_vec.data(), reinterpret_cast<_T *>(&sample_pts[i]),
                                       reinterpret_cast<_T *>(&proj_pts4[i]), depths4[i]);
    cam_model.template angAxRTProjectWithoutDistortion( r_vec.data(), t_vec.data(), reinterpret_cast<_T *>(&sample_pts[i]),
                                                        reinterpret_cast<_T *>(&proj_pts_nd4[i]), depths_nd4[i]);

    depths_gt[i] = ( r_mat*Eigen::Map< const Eigen::Matrix< _T, 3, 1 > >( reinterpret_cast<_T *>(&sample_pts[i]) ) + t_vec )(2);
  }

  for(int i = 0; i < num_samples; i++)
  {
    auto gt_pp = gt_proj_pts[i], gt_pp_nd = gt_proj_pts_nd[i];
    if (gt_pp.x >= 0 && gt_pp.y >= 0 && gt_pp.x <= w - 1 && gt_pp.y <= h - 1)
    {
      ASSERT_TRUE(cv_ext::norm2D(proj_pts1[i] - gt_proj_pts[i]) < tolerance);
      ASSERT_TRUE(cv_ext::norm2D(proj_pts2[i] - gt_proj_pts[i]) < tolerance);
      ASSERT_TRUE(cv_ext::norm2D(proj_pts3[i] - gt_proj_pts[i]) < tolerance);
      ASSERT_TRUE(cv_ext::norm2D(proj_pts4[i] - gt_proj_pts[i]) < tolerance);

      ASSERT_TRUE( fabs( depths1[i] - depths_gt[i]) < tolerance );
      ASSERT_TRUE( fabs( depths2[i] - depths_gt[i]) < tolerance );
      ASSERT_TRUE( fabs( depths3[i] - depths_gt[i]) < tolerance );
      ASSERT_TRUE( fabs( depths4[i] - depths_gt[i]) < tolerance );
    }

    if (gt_pp_nd.x >= 0 && gt_pp_nd.y >= 0 && gt_pp_nd.x <= w - 1 && gt_pp_nd.y <= h - 1)
    {
      ASSERT_TRUE(cv_ext::norm2D(proj_pts_nd1[i] - gt_proj_pts_nd[i]) < tolerance);
      ASSERT_TRUE(cv_ext::norm2D(proj_pts_nd2[i] - gt_proj_pts_nd[i]) < tolerance);
      ASSERT_TRUE(cv_ext::norm2D(proj_pts_nd3[i] - gt_proj_pts_nd[i]) < tolerance);
      ASSERT_TRUE(cv_ext::norm2D(proj_pts_nd4[i] - gt_proj_pts_nd[i]) < tolerance);

      ASSERT_TRUE( fabs( depths_nd1[i] - depths_gt[i]) < tolerance );
      ASSERT_TRUE( fabs( depths_nd2[i] - depths_gt[i]) < tolerance );
      ASSERT_TRUE( fabs( depths_nd3[i] - depths_gt[i]) < tolerance );
      ASSERT_TRUE( fabs( depths_nd4[i] - depths_gt[i]) < tolerance );
    }
  }
}

TEST (PinholeCameraModelTest, EqualityOperatorTest)
{
  cv_ext::PinholeCameraModel cam0 = sampleCameraModel(), cam1 = cam0;
  ASSERT_TRUE( cam0 == cam1 );
}

TEST (PinholeCameraModelTest, DenormalizationTest)
{
  for( int i = 0; i < 100; i++ )
    testDenormalization<double>( 1e-6 );
}

TEST (PinholeCameraModelTest, NormalizationTest)
{
  for( int i = 0; i < 100; i++ )
    testNormalization<double>( 1e-5 );
}

TEST (PinholeCameraModelTest, ProjectionTest)
{
  for( int i = 0; i < 100; i++ )
    testProjection<double>( 1e-6 );
}

TEST (PinholeCameraModelTest, RTProjectionTest)
{
  for( int i = 0; i < 100; i++ )
    testRTProjection<double>( 1e-5 );
}


TEST (PinholeCameraModelTest, FilePersistenceTest)
{
  cv_ext::PinholeCameraModel cam0 = sampleCameraModel(), cam1;

  boost::filesystem::path temp_file = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();

  cam0.writeToFile( temp_file.c_str() );
  cam1.readFromFile( temp_file.c_str() );

  compareCameras(cam0, cam1, 1e-8 );
}

TEST (PinholeCameraModelTest, YamlNodePersistenceTest)
{
  cv_ext::PinholeCameraModel cam0 = sampleCameraModel(),
                             cam1 = sampleCameraModel();

  // load cam params into yaml nodes
  YAML::Node root, node_cam0, node_cam1;
  cam0.write(node_cam0);
  cam1.write(node_cam1);
  root["camera0"] = node_cam0;
  root["camera1"] = node_cam1;

  boost::filesystem::path temp_file = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();
  std::ofstream out_file(temp_file.c_str());
  out_file << root;
  out_file.close();

  // load cams from file
  cv_ext::PinholeCameraModel out_cam0, out_cam1;
  YAML::Node new_root = YAML::LoadFile(temp_file.c_str());
  out_cam0.read(new_root["camera0"]);
  out_cam1.read(new_root["camera1"]);

  compareCameras(cam0, out_cam0, 1e-8 );
  compareCameras(cam1, out_cam1, 1e-8 );
}