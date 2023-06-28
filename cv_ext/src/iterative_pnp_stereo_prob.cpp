#include "cv_ext/iterative_pnp_stereo_prob.h"
#include <opencv2/core/eigen.hpp>

#include <iostream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"



struct PnPReprojectionErrorLeftImage
{
  PnPReprojectionErrorLeftImage ( const cv::Point3f &obj_pt, const cv::Point2f &proj_pt, const float prob) :
    obj_pt_{obj_pt.x, obj_pt.y, obj_pt.z },
    observed_pt_{ proj_pt.x, proj_pt.y },
    prob_{prob}{};


  template <typename _T>
  bool operator() ( const _T* const pos, _T* residuals ) const
  {
    _T obj_pt[3] = { _T ( obj_pt_[0] ), _T ( obj_pt_[1] ), _T ( obj_pt_[2] ) };
    _T proj_pt[2];

    _T transf_pt[3];
    ceres::QuaternionRotatePoint( pos, obj_pt, transf_pt );
    transf_pt[0] += pos[4];
    transf_pt[1] += pos[5];
    transf_pt[2] += pos[6];

    proj_pt[0] = transf_pt[0]/transf_pt[2];
    proj_pt[1] = transf_pt[1]/transf_pt[2];

    _T resx = proj_pt[0] - observed_pt_[0];
    _T resy = proj_pt[1] - observed_pt_[1];

    residuals[0] = resx * _T(prob_);
    residuals[1] = resy * _T(prob_);

    return true;
  }
  
  
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create( const cv::Point3f &obj_pt, const cv::Point2f &proj_pt, const float prob)
  {
    return (new ceres::AutoDiffCostFunction<PnPReprojectionErrorLeftImage, 2, 7 >(
                new PnPReprojectionErrorLeftImage( obj_pt, proj_pt, prob) ) );
  }

private:
  
  double obj_pt_[3], observed_pt_[2];
  double prob_;

};

struct PnPReprojectionErrorRightImage
{
  PnPReprojectionErrorRightImage ( const cv::Point3f &obj_pt, const cv::Point2f &proj_pt, float prob, double baseline) :
    obj_pt_{obj_pt.x, obj_pt.y, obj_pt.z },
    observed_pt_{ proj_pt.x, proj_pt.y },
    prob_{prob},
    baseline_{baseline} {};



  template <typename _T>
  bool operator() ( const _T* const pos, _T* residuals ) const
  {
    _T obj_pt[3] = { _T ( obj_pt_[0] ), _T ( obj_pt_[1] ), _T ( obj_pt_[2] ) };
    _T proj_pt[2];

    _T transf_pt[3];
    ceres::QuaternionRotatePoint( pos, obj_pt, transf_pt );
    transf_pt[0] += pos[4] - _T(baseline_);
    transf_pt[1] += pos[5];
    transf_pt[2] += pos[6];

    proj_pt[0] = transf_pt[0]/transf_pt[2];
    proj_pt[1] = transf_pt[1]/transf_pt[2];

    _T resx = proj_pt[0] - observed_pt_[0];
    _T resy = proj_pt[1] - observed_pt_[1];
    
    residuals[0] = resx * _T(prob_);
    residuals[1] = resy * _T(prob_);
    return true;
  }
  
  
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create( const cv::Point3f &obj_pt, const cv::Point2f &proj_pt, float prob, double baseline)
  {
    return (new ceres::AutoDiffCostFunction<PnPReprojectionErrorRightImage, 2, 7 >(
                new PnPReprojectionErrorRightImage( obj_pt, proj_pt, prob, baseline) ) );
  }

private:
  
  double obj_pt_[3], observed_pt_[2];
  double prob_;
  double baseline_;
};

void IterativePnPStereoProb::setCamModel ( const cv_ext::PinholeCameraModel& cam_model )
{
  cam_model_ = cam_model;
}

void IterativePnPStereoProb::compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<std::vector<cv::Point2f>> &proj_pts, const std::vector<std::vector<cv::Point2f>> &proj_pts_r, const std::vector<std::vector<float>> &probs, const std::vector<std::vector<float>> &probs_r)
{

  auto n_kp = proj_pts.size();
  auto n_rounds = proj_pts[0].size();

  /*std::cout<<"nkp, nrounds "<<n_kp<<", "<<n_rounds<<std::endl;

  for (int i = 0; i < n_kp; i++){
    for (int j = 0; j < n_rounds; j++){
      std::cout<<proj_pts[i][j]<<" - "<<probs[i][j]<<std::endl;
    }
  }

  std::cout<<"---------"<<std::endl;

  for (int i = 0; i < n_kp; i++){
    for (int j = 0; j < n_rounds; j++){
      std::cout<<proj_pts_r[i][j]<<" - "<<probs_r[i][j]<<std::endl;
    }
  }*/

  std::vector<std::vector<cv::Point2f>> norm_proj_pts(n_kp);
  std::vector<std::vector<cv::Point2f>> norm_proj_pts_r(n_kp);

  for (int i = 0; i < static_cast<int>(n_kp); ++i){
    
    const float *proj_pts_p = reinterpret_cast< const float * >(proj_pts[i].data());
    norm_proj_pts[i] = std::vector<cv::Point2f>(n_rounds);
    float *norm_proj_pts_p = reinterpret_cast< float* >(norm_proj_pts[i].data());

    const float *proj_pts_p_r = reinterpret_cast< const float * >(proj_pts_r[i].data());
    norm_proj_pts_r[i] = std::vector<cv::Point2f>(n_rounds);
    float *norm_proj_pts_p_r = reinterpret_cast< float* >(norm_proj_pts_r[i].data());
    
    for (int j = 0; j < static_cast<int>(n_rounds); ++j, proj_pts_p += 2, norm_proj_pts_p += 2, proj_pts_p_r += 2, norm_proj_pts_p_r += 2){
      cam_model_.normalize(proj_pts_p, norm_proj_pts_p);
      cam_model_.normalize(proj_pts_p_r, norm_proj_pts_p_r);
    }

  }

  ceres::Problem problem;
  ceres::Solver::Options options;
  options.max_num_iterations = num_iterations_;
  //options.logging_type = ceres::SILENT;
  options.linear_solver_type = ceres::DENSE_QR;

  ceres::Solver::Summary summary;

  for ( int i = 0; i < static_cast<int>(n_kp); ++i )
  {
    for (int j = 0; j < static_cast<int>(n_rounds); ++j){
      //std::cout<<"---------"<<std::endl;
      //std::cout<<"pij"<<probs[i][j]<<std::endl;
      ceres::CostFunction* cost_function =
        PnPReprojectionErrorLeftImage::Create ( obj_pts[i], norm_proj_pts[i][j], probs[i][j]);
      //std::cout<<"---------"<<std::endl;
      ceres::CostFunction* cost_function_r =
        PnPReprojectionErrorRightImage::Create ( obj_pts[i], norm_proj_pts_r[i][j], probs_r[i][j], baseline_);

      problem.AddResidualBlock ( cost_function, nullptr, transf_.data() );
      problem.AddResidualBlock ( cost_function_r, nullptr, transf_.data() );
    }
  }


  // Ensure cheirality constraint
  problem.SetParameterLowerBound(transf_.data(), 6, 0 );

  ceres::Solve ( options, &problem, &summary );
  
}

void IterativePnPStereoProb::compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<std::vector<cv::Point2f>> &proj_pts, 
                                  const std::vector<std::vector<cv::Point2f>> &proj_pts_r, 
                                  const std::vector<std::vector<float>> &probs, const std::vector<std::vector<float>> &probs_r,
                                  double r_quat[4], double t_vec[3])
{
  compute(obj_pts, proj_pts, proj_pts_r, probs, probs_r);

  r_quat[0] = transf_(0,0);
  r_quat[1] = transf_(1,0);
  r_quat[2] = transf_(2,0);
  r_quat[3] = transf_(3,0);

  t_vec[0] = transf_(4,0);
  t_vec[1] = transf_(5,0);
  t_vec[2] = transf_(6,0);
}

void IterativePnPStereoProb::compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<std::vector<cv::Point2f>> &proj_pts,
                            const std::vector<std::vector<cv::Point2f>> &proj_pts_r, 
                            const std::vector<std::vector<float>> &probs, const std::vector<std::vector<float>> &probs_r,
                            Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec)
{
  compute(obj_pts, proj_pts, proj_pts_r, probs, probs_r);

  cv_ext::quat2EigenQuat(transf_.data(), r_quat );
  
  t_vec( 0,0 ) = transf_(4,0);
  t_vec( 1,0 ) = transf_(5,0);
  t_vec( 2,0 ) = transf_(6,0);
}

void IterativePnPStereoProb::compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<std::vector<cv::Point2f>> &proj_pts, 
                            const std::vector<std::vector<cv::Point2f>> &proj_pts_r, const std::vector<std::vector<float>> &probs, 
                            const std::vector<std::vector<float>> &probs_r, cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec)
{
  compute(obj_pts, proj_pts, proj_pts_r, probs, probs_r);

  double angle_axis[3];
  ceres::QuaternionToAngleAxis<double> ( transf_.data(), angle_axis );

  r_vec ( 0,0 ) = angle_axis[0];
  r_vec ( 1,0 ) = angle_axis[1];
  r_vec ( 2,0 ) = angle_axis[2];

  t_vec ( 0,0 ) = transf_(4,0);
  t_vec ( 1,0 ) = transf_(5,0);
  t_vec ( 2,0 ) = transf_(6,0);
}

py::tuple IterativePnPStereoProb::py_compute(const boost::python::numpy::ndarray& obj_pts_py, const boost::python::numpy::ndarray& proj_pts_py, 
                                         const boost::python::numpy::ndarray& proj_pts_py_r, const boost::python::numpy::ndarray& probs_py,
                                         const boost::python::numpy::ndarray& probs_py_r){
  Eigen::Quaterniond r_quat;
  Eigen::Vector3d t_vec;

  auto obj_pts = read_3d_ndarray(obj_pts_py);
  auto proj_pts = read_2d_ndarray(proj_pts_py);
  auto proj_pts_r = read_2d_ndarray(proj_pts_py_r);
  auto probs = read_1d_ndarray(probs_py);
  auto probs_r = read_1d_ndarray(probs_py_r);
  
  /*std::cout<<"plC++ ";
  for (int i = 0; i < probs.size(); i++){
    std::cout<<"[";
    for (int j = 0; j < probs[i].size(); j++)
      std::cout<<probs[i][j]<<", ";
    std::cout<<"]"<<std::endl;
  
  }
  std::cout<<std::endl;*/

  compute(obj_pts, proj_pts, proj_pts_r, probs, probs_r, r_quat, t_vec);
  Eigen::Matrix3d R = r_quat.normalized().toRotationMatrix();

  auto out_r_mat = r_mat_to_ndarray(R);
  auto out_t_vec = t_vec_to_ndarray(t_vec);

  return py::make_tuple(out_r_mat, out_t_vec);
}

void IterativePnPStereoProb::py_setCamModel (const boost::python::numpy::ndarray& K, int width, int height, float baseline){
  if (width == 0 || height == 0)
    throw std::runtime_error("Image width/height cannot be zero");
  auto K_mat = read_K_matrix(K);
  cv_ext::PinholeCameraModel cam(K_mat, width, height);
  setCamModel(cam);
  baseline_ = (double) baseline;
}

void IterativePnPStereoProb::py_setInitialTransformation(const boost::python::numpy::ndarray& R, const boost::python::numpy::ndarray& t){
  auto R_mat = read_R_matrix(R);
  auto t_vec = read_t_vec(t);

  //std::cout<<"Initial R = "<<R_mat<<std::endl;
  //std::cout<<"Initial t = "<<t_vec<<std::endl;

  Eigen::Quaterniond r_quat(R_mat);
  r_quat.normalize();

  transf_ << r_quat.w(), r_quat.x(), r_quat.y(), r_quat.z(), t_vec(0,0), t_vec(1,0), t_vec(2,0), 0;

}
