#include "cv_ext/iterative_pnp_stereo_variance.h"
#include <opencv2/core/eigen.hpp>

#include <iostream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"



struct PnPReprojectionErrorLeftImageVariance
{
  PnPReprojectionErrorLeftImageVariance ( const cv::Point3f &obj_pt, const cv::Point2f &proj_pt, Eigen::Matrix2f variance) :
    obj_pt_{obj_pt.x, obj_pt.y, obj_pt.z },
    observed_pt_{ proj_pt.x, proj_pt.y },
    variance_{variance} {};


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

    residuals[0] = resx*((_T)variance_(0,0)) + resy*((_T)(variance_(1,0)));
    residuals[1] = resx*((_T)(variance_(0,1))) + resy*((_T)variance_(1,1));

    return true;
  }
  
  
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create( const cv::Point3f &obj_pt, const cv::Point2f &proj_pt, Eigen::Matrix2f variance )
  {
    return (new ceres::AutoDiffCostFunction<PnPReprojectionErrorLeftImageVariance, 2, 7 >(
                new PnPReprojectionErrorLeftImageVariance( obj_pt, proj_pt, variance ) ) );
  }

private:
  
  double obj_pt_[3], observed_pt_[2];
  Eigen::Matrix2f variance_;

};

struct PnPReprojectionErrorRightImageVariance
{
  PnPReprojectionErrorRightImageVariance ( const cv::Point3f &obj_pt, const cv::Point2f &proj_pt, double baseline, Eigen::Matrix2f variance ) :
    obj_pt_{obj_pt.x, obj_pt.y, obj_pt.z },
    observed_pt_{ proj_pt.x, proj_pt.y },
    baseline_{baseline},
    variance_{variance} {};



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
    
    residuals[0] = resx*((_T)variance_(0,0)) + resy*((_T)(variance_(1,0)));
    residuals[1] = resx*((_T)(variance_(0,1)))+resy*((_T)variance_(1,1));
    return true;
  }
  
  
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create( const cv::Point3f &obj_pt, const cv::Point2f &proj_pt, double baseline, Eigen::Matrix2f variance )
  {
    return (new ceres::AutoDiffCostFunction<PnPReprojectionErrorRightImageVariance, 2, 7 >(
                new PnPReprojectionErrorRightImageVariance( obj_pt, proj_pt, baseline, variance ) ) );
  }

private:
  
  double obj_pt_[3], observed_pt_[2];
  double baseline_;
  Eigen::Matrix2f variance_;
};

void IterativePnPStereoVariance::setCamModel ( const cv_ext::PinholeCameraModel& cam_model )
{
  cam_model_ = cam_model;
}

void IterativePnPStereoVariance::compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts, const std::vector<cv::Point2f> &proj_pts_r, const std::vector<Eigen::Matrix2f>& var, const std::vector<Eigen::Matrix2f>& var_r )
{

  std::vector<cv::Point2f> norm_proj_pts(proj_pts.size());
  std::vector<cv::Point2f> norm_proj_pts_r(proj_pts_r.size());

  const float *proj_pts_p = reinterpret_cast< const float * >(proj_pts.data());
  float *norm_proj_pts_p = reinterpret_cast< float* >(norm_proj_pts.data());

  const float *proj_pts_p_r = reinterpret_cast< const float * >(proj_pts_r.data());
  float *norm_proj_pts_p_r = reinterpret_cast< float* >(norm_proj_pts_r.data());

  for ( int i = 0; i < static_cast<int>(proj_pts.size()); ++i, proj_pts_p += 2, norm_proj_pts_p += 2, proj_pts_p_r += 2, norm_proj_pts_p_r += 2){
    cam_model_.normalize(proj_pts_p, norm_proj_pts_p);
    cam_model_.normalize(proj_pts_p_r, norm_proj_pts_p_r);
  }

  ceres::Problem problem;
  ceres::Solver::Options options;
  options.max_num_iterations = num_iterations_;

  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;

  for ( int i = 0; i < static_cast<int>(obj_pts.size()); ++i )
  {
    ceres::CostFunction* cost_function =
        PnPReprojectionErrorLeftImageVariance::Create ( obj_pts[i], norm_proj_pts[i], var[i] );

    ceres::CostFunction* cost_function_r =
        PnPReprojectionErrorRightImageVariance::Create ( obj_pts[i], norm_proj_pts_r[i], baseline_, var_r[i] );

    problem.AddResidualBlock ( cost_function, nullptr, transf_.data() );
    problem.AddResidualBlock ( cost_function_r, nullptr, transf_.data() );
  }


  // Ensure cheirality constraint
  problem.SetParameterLowerBound(transf_.data(), 6, 0 );

  ceres::Solve ( options, &problem, &summary );
  
}

void IterativePnPStereoVariance::compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts, const std::vector<cv::Point2f> &proj_pts_r,
                            double r_quat[4], double t_vec[3], const std::vector<Eigen::Matrix2f>& var, const std::vector<Eigen::Matrix2f>& var_r )
{
  compute(obj_pts, proj_pts, proj_pts_r, var, var_r);

  r_quat[0] = transf_(0,0);
  r_quat[1] = transf_(1,0);
  r_quat[2] = transf_(2,0);
  r_quat[3] = transf_(3,0);

  t_vec[0] = transf_(4,0);
  t_vec[1] = transf_(5,0);
  t_vec[2] = transf_(6,0);
}

void IterativePnPStereoVariance::compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts, const std::vector<cv::Point2f> &proj_pts_r,
                            Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec, const std::vector<Eigen::Matrix2f>& var, const std::vector<Eigen::Matrix2f>& var_r )
{
  compute(obj_pts, proj_pts, proj_pts_r, var, var_r);

  cv_ext::quat2EigenQuat(transf_.data(), r_quat );
  
  t_vec( 0,0 ) = transf_(4,0);
  t_vec( 1,0 ) = transf_(5,0);
  t_vec( 2,0 ) = transf_(6,0);
}

void IterativePnPStereoVariance::compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts, const std::vector<cv::Point2f> &proj_pts_r,
                            cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec, const std::vector<Eigen::Matrix2f>& var, const std::vector<Eigen::Matrix2f>& var_r )
{
  compute(obj_pts, proj_pts, proj_pts_r, var, var_r);

  double angle_axis[3];
  ceres::QuaternionToAngleAxis<double> ( transf_.data(), angle_axis );

  r_vec ( 0,0 ) = angle_axis[0];
  r_vec ( 1,0 ) = angle_axis[1];
  r_vec ( 2,0 ) = angle_axis[2];

  t_vec ( 0,0 ) = transf_(4,0);
  t_vec ( 1,0 ) = transf_(5,0);
  t_vec ( 2,0 ) = transf_(6,0);
}

py::tuple IterativePnPStereoVariance::py_compute(const boost::python::numpy::ndarray& obj_pts_py, const boost::python::numpy::ndarray& proj_pts_py, const boost::python::numpy::ndarray& proj_pts_py_r, const boost::python::numpy::ndarray& var_py, const boost::python::numpy::ndarray& var_py_r){
  Eigen::Quaterniond r_quat;
  Eigen::Vector3d t_vec;

  auto obj_pts = read_3d_ndarray(obj_pts_py);
  auto proj_pts = read_2d_ndarray(proj_pts_py);
  auto proj_pts_r = read_2d_ndarray(proj_pts_py_r);
  auto var = read_variance(var_py);
  auto var_R = read_variance(var_py_r);

  compute(obj_pts, proj_pts, proj_pts_r, r_quat, t_vec, var, var_R);
  Eigen::Matrix3d R = r_quat.normalized().toRotationMatrix();

  auto out_r_mat = r_mat_to_ndarray(R);
  auto out_t_vec = t_vec_to_ndarray(t_vec);

  return py::make_tuple(out_r_mat, out_t_vec);
}

void IterativePnPStereoVariance::py_setCamModel (const boost::python::numpy::ndarray& K, int width, int height, float baseline){
  if (width == 0 || height == 0)
    throw std::runtime_error("Image width/height cannot be zero");
  auto K_mat = read_K_matrix(K);
  cv_ext::PinholeCameraModel cam(K_mat, width, height);
  setCamModel(cam);
  baseline_ = (double) baseline;
}

void IterativePnPStereoVariance::py_setInitialTransformation(const boost::python::numpy::ndarray& R, const boost::python::numpy::ndarray& t){
  auto R_mat = read_R_matrix(R);
  auto t_vec = read_t_vec(t);

  //std::cout<<"Initial R = "<<R_mat<<std::endl;
  //std::cout<<"Initial t = "<<t_vec<<std::endl;

  Eigen::Quaterniond r_quat(R_mat);
  r_quat.normalize();

  transf_ << r_quat.w(), r_quat.x(), r_quat.y(), r_quat.z(), t_vec(0,0), t_vec(1,0), t_vec(2,0), 0;

}
