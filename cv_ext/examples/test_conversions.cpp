#include <cstdio>

#include "cv_ext/conversions.h"

using namespace cv_ext;

int main(int argc, char** argv)
{
  cv::Vec<float, 3> vf_r_vec_init, vf_t_vec_init, vf_r_vec, vf_t_vec;
  cv::Vec<double, 3> vd_r_vec_init, vd_t_vec_init, vd_r_vec, vd_t_vec;
  
  cv::RNG rng;
  vd_r_vec_init[0] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform(-1.0, 1.0 );
  vd_r_vec_init[1] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform(-1.0, 1.0 );
  vd_r_vec_init[2] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform(-1.0, 1.0 );
  
  vf_r_vec_init[0] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform( -1.0, 1.0 );
  vf_r_vec_init[1] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform( -1.0, 1.0 );
  vf_r_vec_init[2] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform( -1.0, 1.0 );

  vf_r_vec = vf_r_vec_init;
  vd_r_vec = vd_r_vec_init;
  
  cv::Mat cvf_r_vec, cvf_t_vec, cvf_r_mat, cvf_g_mat, 
	  cvd_r_vec, cvd_t_vec, cvd_r_mat, cvd_g_mat;
  Eigen::Matrix<float, 3, 3> eigenf_r_mat;
  Eigen::Matrix<double, 3, 3> eigend_r_mat;
  Eigen::Matrix<float, 4, 4> eigenf_g_mat;
  Eigen::Matrix<double, 4, 4> eigend_g_mat;
  
  // cv::vec -> cv::Mat
  exp2RotMat<float>( vf_r_vec, cvf_r_mat );
  exp2RotMat<double>( vd_r_vec, cvd_r_mat );
  
  // cv::Mat -> cv::Vec
  rotMat2Exp<float>( cvf_r_mat, vf_r_vec );
  rotMat2Exp<double>( cvd_r_mat, vd_r_vec );    
  
  // cv::Vec -> Eigen::Matrix
  exp2RotMat<float>( vf_r_vec, eigenf_r_mat );
  exp2RotMat<double>( vd_r_vec, eigend_r_mat );

  // Eigen::Matrix -> cv::vec
  rotMat2Exp<float>( eigenf_r_mat, vf_r_vec );
  rotMat2Exp<double>( eigend_r_mat, vd_r_vec );

  // cv::vec -> cv::Mat
  exp2RotMat<float>( vf_r_vec, cvf_r_mat );
  exp2RotMat<double>( vd_r_vec, cvd_r_mat );
  
  // cv::Mat -> cv::Mat
  rotMat2Exp<float>( cvf_r_mat, cvf_r_vec );
  rotMat2Exp<double>( cvd_r_mat, cvd_r_vec );
  
  // cv::Mat -> Eigen::Matrix
  exp2RotMat<float>( cvf_r_vec, eigenf_r_mat );
  exp2RotMat<double>( cvd_r_vec, eigend_r_mat );
  
  // Eigen::Matrix -> cv::Mat
  rotMat2Exp<float>( eigenf_r_mat, cvf_r_vec );
  rotMat2Exp<double>( eigend_r_mat, cvd_r_vec );    

  // cv::Mat -> cv::Mat
  exp2RotMat<float>( cvf_r_vec , cvf_r_mat );
  exp2RotMat<double>( cvd_r_vec , cvd_r_mat );

  // cv::Mat -> cv::Vec
  rotMat2Exp<float>( cvf_r_mat, vf_r_vec );
  rotMat2Exp<double>( cvd_r_mat, vd_r_vec );    

  
  if(fabs(vf_r_vec[0] - vf_r_vec_init[0]) > 1e-7 ||
      fabs(vf_r_vec[1] - vf_r_vec_init[1]) > 1e-7 ||
      fabs(vf_r_vec[2] - vf_r_vec_init[2]) > 1e-7)
  {
    fprintf(stderr, "rotMat2Exp/exp2RotMat : test float failed, residuals: \
                    %.12f %.12f %.12f\n",
           fabs(vf_r_vec[0] - vf_r_vec_init[0]),
           fabs(vf_r_vec[1] - vf_r_vec_init[1]),
           fabs(vf_r_vec[2] - vf_r_vec_init[2]));
  }
  else
    printf("rotMat2Exp/exp2RotMat : test float passed\n");
  
  if(fabs(vd_r_vec[0] - vd_r_vec_init[0]) > 1e-12 ||
      fabs(vd_r_vec[1] - vd_r_vec_init[1]) > 1e-12 ||
      fabs(vd_r_vec[2] - vd_r_vec_init[2]) > 1e-12)
  {
    fprintf(stderr, "rotMat2Exp/exp2RotMat : test double failed, residuals: \
                    %.20f %.20f %.20f\n",
           fabs(vd_r_vec[0] - vd_r_vec_init[0]),
           fabs(vd_r_vec[1] - vd_r_vec_init[1]),
           fabs(vd_r_vec[2] - vd_r_vec_init[2]));
  }
  else
    printf("rotMat2Exp/exp2RotMat : test double passed\n");
  
  vd_r_vec_init[0] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform(-1.0, 1.0 );
  vd_r_vec_init[1] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform(-1.0, 1.0 );
  vd_r_vec_init[2] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform(-1.0, 1.0 );

  vd_t_vec_init[0] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform(-1.0, 1.0 );
  vd_t_vec_init[1] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform(-1.0, 1.0 );
  vd_t_vec_init[2] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform(-1.0, 1.0 );
  
  vf_r_vec_init[0] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform(-1.0, 1.0 );
  vf_r_vec_init[1] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform(-1.0, 1.0 );
  vf_r_vec_init[2] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform(-1.0, 1.0 );

  vf_t_vec_init[0] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform(-1.0, 1.0 );
  vf_t_vec_init[1] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform(-1.0, 1.0 );
  vf_t_vec_init[2] = ( rng.uniform( 0.0, 1.0 ) > 0.5?1:-1)*rng.uniform(-1.0, 1.0 );

  vf_r_vec = vf_r_vec_init;
  vf_t_vec = vf_t_vec_init;
  vd_r_vec = vd_r_vec_init; 
  vd_t_vec = vd_t_vec_init;

  // cv::vec -> cv::Mat
  exp2TransfMat<float>( vf_r_vec, vf_t_vec, cvf_g_mat );
  exp2TransfMat<double>( vd_r_vec, vd_t_vec, cvd_g_mat );
  
  // cv::Mat -> cv::Vec
  transfMat2Exp<float>( cvf_g_mat, vf_r_vec, vf_t_vec );
  transfMat2Exp<double>( cvd_g_mat, vd_r_vec, vd_t_vec );    
  
  // cv::Vec -> Eigen::Matrix
  exp2TransfMat<float>( vf_r_vec, vf_t_vec, eigenf_g_mat );
  exp2TransfMat<double>( vd_r_vec, vd_t_vec, eigend_g_mat );

  // Eigen::Matrix -> cv::vec
  transfMat2Exp<float>( eigenf_g_mat, vf_r_vec, vf_t_vec );
  transfMat2Exp<double>( eigend_g_mat, vd_r_vec, vd_t_vec );

  // cv::vec -> cv::Mat
  exp2TransfMat<float>( vf_r_vec, vf_t_vec, cvf_g_mat );
  exp2TransfMat<double>( vd_r_vec, vd_t_vec, cvd_g_mat );
  
  // cv::Mat -> cv::Mat
  transfMat2Exp<float>( cvf_g_mat, cvf_r_vec, cvf_t_vec );
  transfMat2Exp<double>( cvd_g_mat, cvd_r_vec, cvd_t_vec );
  
  // cv::Mat -> Eigen::Matrix
  exp2TransfMat<float>( cvf_r_vec, cvf_t_vec, eigenf_g_mat );
  exp2TransfMat<double>( cvd_r_vec, cvd_t_vec, eigend_g_mat );
  
  // Eigen::Matrix -> cv::Mat
  transfMat2Exp<float>( eigenf_g_mat, cvf_r_vec, cvf_t_vec );
  transfMat2Exp<double>( eigend_g_mat, cvd_r_vec, cvd_t_vec );    

  // cv::Mat -> cv::Mat
  exp2TransfMat<float>( cvf_r_vec, cvf_t_vec, cvf_g_mat );
  exp2TransfMat<double>( cvd_r_vec, cvd_t_vec, cvd_g_mat );

  // cv::Mat -> cv::Vec
  transfMat2Exp<float>( cvf_g_mat, vf_r_vec, vf_t_vec );
  transfMat2Exp<double>( cvd_g_mat, vd_r_vec, vd_t_vec );        
  
  if(fabs(vf_r_vec[0] - vf_r_vec_init[0]) > 1e-7 ||
      fabs(vf_r_vec[1] - vf_r_vec_init[1]) > 1e-7 ||
      fabs(vf_r_vec[2] - vf_r_vec_init[2]) > 1e-7 ||
      fabs(vf_t_vec[0] - vf_t_vec_init[0]) > 1e-7 ||
      fabs(vf_t_vec[1] - vf_t_vec_init[1]) > 1e-7 ||
      fabs(vf_t_vec[2] - vf_t_vec_init[2]) > 1e-7)
  {
    fprintf(stderr, "transfMat2Exp : test float failed, residuals: \
                    %.12f %.12f %.12f %.12f %.12f %.12f\n",
           fabs(vf_r_vec[0] - vf_r_vec_init[0]),
           fabs(vf_r_vec[1] - vf_r_vec_init[1]),
           fabs(vf_r_vec[2] - vf_r_vec_init[2]),
           fabs(vf_t_vec[0] - vf_t_vec_init[0]),
           fabs(vf_t_vec[1] - vf_t_vec_init[1]),
           fabs(vf_t_vec[2] - vf_t_vec_init[2]));
  }
  else
    printf("transfMat2Exp : test float passed\n");
  
  if(fabs(vd_r_vec[0] - vd_r_vec_init[0]) > 1e-12 ||
      fabs(vd_r_vec[1] - vd_r_vec_init[1]) > 1e-12 ||
      fabs(vd_r_vec[2] - vd_r_vec_init[2]) > 1e-12 ||
      fabs(vd_t_vec[0] - vd_t_vec_init[0]) > 1e-12 ||
      fabs(vd_t_vec[1] - vd_t_vec_init[1]) > 1e-12 ||
      fabs(vd_t_vec[2] - vd_t_vec_init[2]) > 1e-12)
  {
    fprintf(stderr, "transfMat2Exp : test double failed, residuals: \
                    %.20f %.20f %.20f %.20f %.20f %.20f\n",
           fabs(vd_r_vec[0] - vd_r_vec_init[0]),
           fabs(vd_r_vec[1] - vd_r_vec_init[1]),
           fabs(vd_r_vec[2] - vd_r_vec_init[2]),
           fabs(vd_t_vec[0] - vd_t_vec_init[0]),
           fabs(vd_t_vec[1] - vd_t_vec_init[1]),
           fabs(vd_t_vec[2] - vd_t_vec_init[2]));
  }
  else
    printf("transfMat2Exp : test double passed\n");
  
}