#include "cv_ext/serialization.h"

#include <fstream>
#include <gtest/gtest.h>

TEST (SerializationTest, WriteRead3DTransfToFileTest)
{
  cv::Mat_<double> in_r_vec0(3,3), in_r_vec1(3,3);
  cv::Mat_<double> in_t_vec0(3,1), in_t_vec1(3,1);
  cv::Mat_<double> out_r_vec0, out_r_vec1;
  cv::Mat_<double> out_t_vec0, out_t_vec1;

  in_r_vec0.setTo(0.3);  in_t_vec0.setTo(0.4);
  in_r_vec1.setTo(0.5);  in_t_vec1.setTo(0.6);

  in_r_vec0.at<double>(0,1) = 0.0;

  YAML::Node root, tf0, tf1;

  cv_ext::write3DTransf(tf0, in_r_vec0, in_t_vec0);
  cv_ext::write3DTransf(tf1, in_r_vec1, in_t_vec1);

  root["transform0"] = tf0;
  root["transform1"] = tf1;

  std::ofstream out_file("test_3D_transf.yml");
  out_file << root;
  out_file.close();

  YAML::Node new_root = YAML::LoadFile("test_3D_transf.yml");

  cv_ext::read3DTransf(new_root["transform0"], out_r_vec0, out_t_vec0, cv_ext::ROT_FORMAT_MATRIX);
  cv_ext::read3DTransf(new_root["transform1"], out_r_vec1, out_t_vec1, cv_ext::ROT_FORMAT_MATRIX);

  cv::Mat diff_r0 = in_r_vec0 != out_r_vec0;
  cv::Mat diff_t0 = in_t_vec0 != out_t_vec0;
  cv::Mat diff_r1 = in_r_vec1 != out_r_vec1;
  cv::Mat diff_t1 = in_t_vec1 != out_t_vec1;

  bool assert_0 = (cv::countNonZero(diff_r0) == 0) && (cv::countNonZero(diff_t0) == 0);
  bool assert_1 = (cv::countNonZero(diff_r1) == 0) && (cv::countNonZero(diff_t1) == 0);

  ASSERT_TRUE( assert_0 && assert_1);
}
