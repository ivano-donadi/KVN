#include "cv_ext/iterative_pnp.h"

#include <iostream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"


struct PnPReprojectionError
{
  PnPReprojectionError ( const cv::Point3f &obj_pt, const cv::Point2f &proj_pt ) :
    obj_pt_{obj_pt.x, obj_pt.y, obj_pt.z },
    observed_pt_{ proj_pt.x, proj_pt.y } {};


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

    residuals[0] = proj_pt[0] - observed_pt_[0];
    residuals[1] = proj_pt[1] - observed_pt_[1];
    
    return true;
  }
  
  
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create( const cv::Point3f &obj_pt, const cv::Point2f &proj_pt )
  {
    return (new ceres::AutoDiffCostFunction<PnPReprojectionError, 2, 7 >(
                new PnPReprojectionError( obj_pt, proj_pt ) ) );
  }

private:
  
  double obj_pt_[3], observed_pt_[2];

};

template < int _DIM >
struct ConstrainedPnPReprojectionError
{
  ConstrainedPnPReprojectionError ( const cv::Point3f &obj_pt, const cv::Point2f &proj_pt,
                                    double const_val ) :
      obj_pt_{obj_pt.x, obj_pt.y, obj_pt.z },
      observed_pt_{ proj_pt.x, proj_pt.y },
      const_val_(const_val){};

  template <typename _T>
  bool operator() ( const _T *const pos,  _T* residuals ) const;


  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create( const cv::Point3f &obj_pt, const cv::Point2f &proj_pt,
                                      double const_val )
  {
    return (new ceres::AutoDiffCostFunction<ConstrainedPnPReprojectionError, 2, 6 >(
        new ConstrainedPnPReprojectionError( obj_pt, proj_pt, const_val ) ) );
  }

 protected:

  double obj_pt_[3], observed_pt_[2];
  double const_val_;

};

template <> template<typename _T> bool ConstrainedPnPReprojectionError<0> ::
    operator()( const _T *const pos,  _T* residuals ) const
{
  _T obj_pt[3] = {_T(obj_pt_[0]), _T(obj_pt_[1]), _T(obj_pt_[2])};
  _T proj_pt[2];

  _T transf_pt[3];
  ceres::QuaternionRotatePoint(pos, obj_pt, transf_pt);
  transf_pt[0] += _T(const_val_);
  transf_pt[1] += pos[4];
  transf_pt[2] += pos[5];

  proj_pt[0] = transf_pt[0]/transf_pt[2];
  proj_pt[1] = transf_pt[1]/transf_pt[2];

  residuals[0] = proj_pt[0] - observed_pt_[0];
  residuals[1] = proj_pt[1] - observed_pt_[1];

  return true;
}

template <> template<typename _T> bool ConstrainedPnPReprojectionError<1> ::
operator()( const _T *const pos,  _T* residuals ) const
{
  _T obj_pt[3] = {_T(obj_pt_[0]), _T(obj_pt_[1]), _T(obj_pt_[2])};
  _T proj_pt[2];

  _T transf_pt[3];
  ceres::QuaternionRotatePoint(pos, obj_pt, transf_pt);
  transf_pt[0] += pos[4];
  transf_pt[1] += _T(const_val_);
  transf_pt[2] += pos[5];

  proj_pt[0] = transf_pt[0]/transf_pt[2];
  proj_pt[1] = transf_pt[1]/transf_pt[2];

  residuals[0] = proj_pt[0] - observed_pt_[0];
  residuals[1] = proj_pt[1] - observed_pt_[1];

  return true;
}

template <> template<typename _T> bool ConstrainedPnPReprojectionError<2> ::
operator()( const _T *const pos,  _T* residuals ) const
{
  _T obj_pt[3] = {_T(obj_pt_[0]), _T(obj_pt_[1]), _T(obj_pt_[2])};
  _T proj_pt[2];

  _T transf_pt[3];
  ceres::QuaternionRotatePoint(pos, obj_pt, transf_pt);
  transf_pt[0] += pos[4];
  transf_pt[1] += pos[5];
  transf_pt[2] += _T(const_val_);

  proj_pt[0] = transf_pt[0]/transf_pt[2];
  proj_pt[1] = transf_pt[1]/transf_pt[2];

  residuals[0] = proj_pt[0] - observed_pt_[0];
  residuals[1] = proj_pt[1] - observed_pt_[1];

  return true;
}

void IterativePnP::setCamModel ( const cv_ext::PinholeCameraModel& cam_model )
{
  cam_model_ = cam_model;
}

void IterativePnP::compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts )
{
  transf_ << 1.0, 0, 0, 0, 0, 0, 1.0, 0;

  std::vector<cv::Point2f> norm_proj_pts(proj_pts.size());
  const float *proj_pts_p = reinterpret_cast< const float * >(proj_pts.data());
  float *norm_proj_pts_p = reinterpret_cast< float* >(norm_proj_pts.data());

  for ( int i = 0; i < static_cast<int>(proj_pts.size()); ++i, proj_pts_p += 2, norm_proj_pts_p += 2 )
    cam_model_.normalize(proj_pts_p, norm_proj_pts_p);

  ceres::Problem problem;
  ceres::Solver::Options options;
  options.max_num_iterations = num_iterations_;

  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;

  if(fixed_index_ == 0 )
  {
    for ( int i = 0; i < static_cast<int>(obj_pts.size()); ++i )
    {
      ceres::CostFunction* cost_function =
          ConstrainedPnPReprojectionError<0>::Create ( obj_pts[i], norm_proj_pts[i], fixed_value_ );

      problem.AddResidualBlock ( cost_function, nullptr, transf_.data());
    }

    // Ensure cheirality constraint
    problem.SetParameterLowerBound(transf_.data(), 5, 0 );

    ceres::Solve ( options, &problem, &summary );

    transf_(6) = transf_(5);
    transf_(5) = transf_(4);
    transf_(4) = fixed_value_;
  }
  else if( fixed_index_ == 1 )
  {
    for ( int i = 0; i < static_cast<int>(obj_pts.size()); ++i )
    {
      ceres::CostFunction* cost_function =
          ConstrainedPnPReprojectionError<1>::Create ( obj_pts[i], norm_proj_pts[i], fixed_value_ );

      problem.AddResidualBlock ( cost_function, nullptr, transf_.data() );
    }

    // Ensure cheirality constraint
    problem.SetParameterLowerBound(transf_.data(), 5, 0 );

    ceres::Solve ( options, &problem, &summary );

    transf_(6) = transf_(5);
    transf_(5) = fixed_value_;

  }
  else if( fixed_index_ == 2 )
  {
    for ( int i = 0; i < static_cast<int>(obj_pts.size()); ++i )
    {
      ceres::CostFunction* cost_function =
          ConstrainedPnPReprojectionError<2>::Create ( obj_pts[i], norm_proj_pts[i], fixed_value_ );

      problem.AddResidualBlock ( cost_function, nullptr, transf_.data() );
    }
    ceres::Solve ( options, &problem, &summary );

    transf_(6) = fixed_value_;
  }
  else
  {
    for ( int i = 0; i < static_cast<int>(obj_pts.size()); ++i )
    {
      ceres::CostFunction* cost_function =
          PnPReprojectionError::Create ( obj_pts[i], norm_proj_pts[i] );

      problem.AddResidualBlock ( cost_function, nullptr, transf_.data() );
    }


    // Ensure cheirality constraint
    problem.SetParameterLowerBound(transf_.data(), 6, 0 );

    ceres::Solve ( options, &problem, &summary );
  }
}

void IterativePnP::compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts,
                            double r_quat[4], double t_vec[3] )
{
  compute(obj_pts, proj_pts);

  r_quat[0] = transf_(0,0);
  r_quat[1] = transf_(1,0);
  r_quat[2] = transf_(2,0);
  r_quat[3] = transf_(3,0);

  t_vec[0] = transf_(4,0);
  t_vec[1] = transf_(5,0);
  t_vec[2] = transf_(6,0);
}

void IterativePnP::compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts,
                            Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec )
{
  compute(obj_pts, proj_pts);

  cv_ext::quat2EigenQuat(transf_.data(), r_quat );
  
  t_vec( 0,0 ) = transf_(4,0);
  t_vec( 1,0 ) = transf_(5,0);
  t_vec( 2,0 ) = transf_(6,0);
}

void IterativePnP::compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts,
                            cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec )
{
  compute(obj_pts, proj_pts);

  double angle_axis[3];
  ceres::QuaternionToAngleAxis<double> ( transf_.data(), angle_axis );

  r_vec ( 0,0 ) = angle_axis[0];
  r_vec ( 1,0 ) = angle_axis[1];
  r_vec ( 2,0 ) = angle_axis[2];

  t_vec ( 0,0 ) = transf_(4,0);
  t_vec ( 1,0 ) = transf_(5,0);
  t_vec ( 2,0 ) = transf_(6,0);
}

void IterativePnP::constrainsTranslationComponent(int index, double value )
{
  if( index < 0 || index > 2 || ( index == 2 && value <= 0 ) )
  {
    fixed_index_ = -1;
  }
  else
  {
    fixed_index_ = index;
    fixed_value_ = value;
  }
}

void IterativePnP::removeConstraints()
{
  fixed_index_ = -1;
}