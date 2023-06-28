#include "metrics.h"

#include <ceres/rotation.h>

#include <vector>

template<typename _T>
std::vector<_T> removeDuplicates( const std::vector<_T> &src_pts )
{
  std::vector<_T> dst_pts;
  dst_pts.reserve(src_pts.size());

  bool add_elem;
  for( const auto &new_elem : src_pts )
  {
    add_elem = true;
    for( auto &cur_elem : dst_pts )
    {
      if( new_elem == cur_elem )
      {
        add_elem = false;
        break;
      }
    }
    if( add_elem )
      dst_pts.push_back(new_elem);
  }
  return dst_pts;
}

MaxRtErrorsMetric::MaxRtErrorsMetric( double max_rot_err_deg, double max_t_err ) :
    max_rot_err_deg_(max_rot_err_deg),
    max_t_err_(max_t_err)
{}

void MaxRtErrorsMetric::setGroundTruth(Eigen::Quaterniond &gt_r_quat, Eigen::Vector3d &gt_t_vec)
{
  gt_r_quat_ = gt_r_quat;
  gt_t_vec_ = gt_t_vec;
}

bool MaxRtErrorsMetric::performTest(Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec )
{
  Eigen::Matrix3d rot_mat = r_quat.toRotationMatrix(), gt_rot_mat = gt_r_quat_.toRotationMatrix();

  double rot_err = 180.0*cv_ext::rotationDist(rot_mat, gt_rot_mat)/M_PI,
         t_err = (t_vec - gt_t_vec_).norm();

  return ( rot_err < max_rot_err_deg_ && t_err < max_t_err_ );
}

Projection2DMetric::Projection2DMetric( const cv_ext::PinholeCameraModel &cam_model,
                                        const std::vector<cv::Point3f> &model_vtx,
                                        bool symmetric_object, double max_proj_err_pix ) :
    cam_proj_(cam_model),
    model_vtx_(model_vtx),
    symmetric_object_( symmetric_object ),
    max_proj_err_pix_(max_proj_err_pix)
{}

void Projection2DMetric::setGroundTruth(Eigen::Quaterniond &gt_r_quat, Eigen::Vector3d &gt_t_vec)
{
  cam_proj_.setTransformation(gt_r_quat, gt_t_vec);
  cam_proj_.projectPoints(model_vtx_, gt_proj_vtx_ );
}

bool Projection2DMetric::performTest( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec )
{
  std::vector<cv::Point2f> proj_vtx;

  cam_proj_.setTransformation(r_quat, t_vec);
  cam_proj_.projectPoints(model_vtx_, proj_vtx );

  double avg_pixel_diff = 0;

  if( symmetric_object_ )
  {
    for(auto &p : proj_vtx )
    {
      double min_dist = std::numeric_limits<double>::max();
      for(auto &gt_p : gt_proj_vtx_ )
      {
        auto diff_p = p-gt_p;
        double dist = cv::normL2Sqr<float, double>(reinterpret_cast<float *>(&diff_p), 2);
        if(dist < min_dist) min_dist = dist;
      }
      avg_pixel_diff += sqrt(min_dist);
    }
  }
  else
  {
    for (int i = 0; i < static_cast<int>(proj_vtx.size()); i++)
      avg_pixel_diff += cv_ext::norm2D(proj_vtx[i] - gt_proj_vtx_[i]);
  }

  avg_pixel_diff /= proj_vtx.size();
  return  ( avg_pixel_diff < max_proj_err_pix_ );
}

Pose6DMetric::Pose6DMetric(const cv_ext::PinholeCameraModel &cam_model, const std::vector<cv::Point3f> &model_vtx,
                           double obj_diameter, bool symmetric_object, double max_err_percentage ) :
    cam_proj_(cam_model),
    model_vtx_(model_vtx),
    obj_diameter_(obj_diameter),
    symmetric_object_( symmetric_object),
    max_err_percentage_(max_err_percentage)
{}

void Pose6DMetric::setGroundTruth(Eigen::Quaterniond &gt_r_quat, Eigen::Vector3d &gt_t_vec)
{
  gt_vtx_.resize(model_vtx_.size());

  cv::Point3f gt_t( gt_t_vec(0), gt_t_vec(1), gt_t_vec(2) );

  double tmp_gt_q[4];
  cv_ext::eigenQuat2Quat( gt_r_quat, tmp_gt_q );
  float gt_q[4];
  for( int i = 0; i < 4; i++ )
    gt_q[i] = static_cast<float>(tmp_gt_q[i]);

  for( int i = 0; i < static_cast<int>(model_vtx_.size()); i++ )
  {
    ceres::UnitQuaternionRotatePoint( gt_q, reinterpret_cast<const float*>( &(model_vtx_[i]) ),
                                      reinterpret_cast<float*>( &(gt_vtx_[i]) ) );
    gt_vtx_[i] += gt_t;
  }
}

bool Pose6DMetric::performTest( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec )
{
  std::vector<cv::Point3f> vtx(model_vtx_.size());

  cv::Point3f t( t_vec(0), t_vec(1), t_vec(2) );

  double tmp_q[4];
  cv_ext::eigenQuat2Quat( r_quat, tmp_q );
  float q[4];
  for( int i = 0; i < 4; i++ )
    q[i] = static_cast<float>(tmp_q[i]);

  for( int i = 0; i < static_cast<int>(model_vtx_.size()); i++ )
  {
    ceres::UnitQuaternionRotatePoint( q, reinterpret_cast<const float*>( &(model_vtx_[i]) ),
                                      reinterpret_cast<float*>( &(vtx[i]) ) );
    vtx[i] += t;
  }

  double avg_diff = 0;
  if( symmetric_object_ )
  {

    for(auto &p : vtx )
    {
      double min_dist = std::numeric_limits<double>::max();
      for(auto &gt_p : gt_vtx_ )
      {
        auto diff_p = p-gt_p;
        double dist = cv::normL2Sqr<float, double>(reinterpret_cast<float *>(&diff_p), 3);
        if(dist < min_dist) min_dist = dist;
      }
      avg_diff += sqrt(min_dist);
    }
  }
  else
  {
    for( int i = 0; i < static_cast<int>(vtx.size()); i++ )
      avg_diff += cv_ext::norm3D( vtx[i] - gt_vtx_[i] );
  }

  avg_diff /= vtx.size();

  return  ( avg_diff < max_err_percentage_*obj_diameter_ );
}
