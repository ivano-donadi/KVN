#include "object_registration.h"

using namespace std;
using namespace cv;

#include <ceres/rotation.h>

ObjectRegistrationBase::ObjectRegistrationBase()
{
  transf_ << 1.0, 0, 0, 0, 0, 0, 0, 0;
}

void ObjectRegistrationBase::setPos ( const double r_quat[4], const double t_vec[3] )
{
  transf_(0,0) = r_quat[0];
  transf_(1,0) = r_quat[1];
  transf_(2,0) = r_quat[2];
  transf_(3,0) = r_quat[3];

  transf_(4,0) = t_vec[0];
  transf_(5,0) = t_vec[1];
  transf_(6,0) = t_vec[2];
}

void ObjectRegistrationBase::setPos ( const Eigen::Quaterniond& r_quat,
                                      const Eigen::Vector3d& t_vec )
{
  cv_ext::eigenQuat2Quat( r_quat, transf_.data() );
  transf_(4,0) = t_vec[0];
  transf_(5,0) = t_vec[1];
  transf_(6,0) = t_vec[2];
}

void ObjectRegistrationBase::setPos ( const Mat_< double >& r_vec, 
                                      const Mat_< double >& t_vec )
{
  double angle_axis[3] = { r_vec ( 0,0 ) , r_vec ( 1,0 ) , r_vec ( 2,0 ) };
  ceres::AngleAxisToQuaternion<double> ( angle_axis, transf_.data() );

  transf_(4,0) = t_vec ( 0,0 );
  transf_(5,0) = t_vec ( 1,0 );
  transf_(6,0) = t_vec ( 2,0 );
}

void ObjectRegistrationBase::getPos ( double r_quat[4], double t_vec[3] ) const
{
  r_quat[0] = transf_(0,0);
  r_quat[1] = transf_(1,0);
  r_quat[2] = transf_(2,0);
  r_quat[3] = transf_(3,0);

  t_vec[0] = transf_(4,0);
  t_vec[1] = transf_(5,0);
  t_vec[2] = transf_(6,0);
}

void ObjectRegistrationBase::getPos ( Eigen::Quaterniond& r_quat, 
                                Eigen::Vector3d& t_vec ) const
{
  cv_ext::quat2EigenQuat(transf_.data(), r_quat );
  t_vec( 0,0 ) = transf_(4,0);
  t_vec( 1,0 ) = transf_(5,0);
  t_vec( 2,0 ) = transf_(6,0);
}

void ObjectRegistrationBase::getPos ( Mat_< double >& r_vec, 
                                Mat_< double >& t_vec ) const
{
  double angle_axis[3];
  ceres::QuaternionToAngleAxis<double> ( transf_.data(), angle_axis );
  
  r_vec ( 0,0 ) = angle_axis[0];
  r_vec ( 1,0 ) = angle_axis[1];
  r_vec ( 2,0 ) = angle_axis[2];

  t_vec ( 0,0 ) = transf_(4,0);
  t_vec ( 1,0 ) = transf_(5,0);
  t_vec ( 2,0 ) = transf_(6,0);
}

void ObjectRegistration::setObjectModel (const RasterObjectModelPtr& model_ptr )
{
  if( !model_ptr )
    throw runtime_error("Null RasterObjectModel");

  model_ptr_ = model_ptr;
  cam_model_ = model_ptr->cameraModel();

  update_optimizer_ = true;
}

double ObjectRegistration::refinePosition( double r_quat[4], double t_vec[3] )
{
  if( !model_ptr_ )
    throw runtime_error("RasterObjectModel not set");

  setPos( r_quat, t_vec );
  if(update_optimizer_ || !model_ptr_->allVisiblePoints())
  {
    model_ptr_->setModelView(transf_.data(), transf_.block<3,1>(4,0).data());
    updateOptimizer();
    update_optimizer_ = false;
  }
  double res = optimize();
  getPos( r_quat, t_vec );

  return res;
}

double ObjectRegistration::refinePosition ( Eigen::Quaterniond& r_quat,
                                                Eigen::Vector3d& t_vec )
{
  if( !model_ptr_ )
    throw runtime_error("RasterObjectModel not set");

  setPos( r_quat, t_vec );
  if(update_optimizer_ || !model_ptr_->allVisiblePoints())
  {
    model_ptr_->setModelView(transf_.data(), transf_.block<3,1>(4,0).data());
    updateOptimizer();
    update_optimizer_ = false;
  }
  double res = optimize();
  getPos( r_quat, t_vec );

  return res;
}

double ObjectRegistration::refinePosition( Mat_<double> &r_vec, Mat_<double> &t_vec )
{
  if( !model_ptr_ )
    throw runtime_error("RasterObjectModel not set");

  setPos( r_vec, t_vec );
  if(update_optimizer_ || !model_ptr_->allVisiblePoints())
  {
    model_ptr_->setModelView(transf_.data(), transf_.block<3,1>(4,0).data());
    updateOptimizer();
    update_optimizer_ = false;
  }
  double res = optimize();
  getPos( r_vec, t_vec );

  return res;
}

//double ObjectRegistration::refinePosition ( int idx, double r_quat[4], double t_vec[3] )
//{
//  if( !model_ptr_ )
//    throw runtime_error("RasterObjectModel not set");
//
//  Eigen::Quaterniond quat;
//  Eigen::Vector3d t;
//  model_ptr_->modelView(idx, quat, t);
//  setPos ( quat, t );
//
//  if(update_optimizer_ || !model_ptr_->allVisiblePoints())
//  {
//    updateOptimizer(idx);
//    update_optimizer_ = false;
//  }
//
//  double res = optimize();
//  getPos (r_quat, t_vec );
//
//  return res;
//}
//
//double ObjectRegistration::refinePosition ( int idx, Eigen::Quaterniond& r_quat,
//                                               Eigen::Vector3d& t_vec  )
//{
//  if( !model_ptr_ )
//    throw runtime_error("RasterObjectModel not set");
//
//  Eigen::Quaterniond quat;
//  Eigen::Vector3d t;
//  model_ptr_->modelView(idx, quat, t);
//  setPos ( quat, t );
//
//  if(update_optimizer_ || !model_ptr_->allVisiblePoints())
//  {
//    updateOptimizer(idx);
//    update_optimizer_ = false;
//  }
//
//  double res = optimize();
//  getPos (r_quat, t_vec );
//
//  return res;
//}
//
//double ObjectRegistration::refinePosition ( int idx, Mat_< double >& r_vec,
//                                               Mat_< double >& t_vec )
//{
//  if( !model_ptr_ )
//    throw runtime_error("RasterObjectModel not set");
//
//  Eigen::Quaterniond quat;
//  Eigen::Vector3d t;
//  model_ptr_->modelView(idx, quat, t);
//  setPos ( quat, t );
//
//  if(update_optimizer_ || !model_ptr_->allVisiblePoints())
//  {
//    updateOptimizer(idx);
//    update_optimizer_ = false;
//  }
//
//  double res = optimize();
//  getPos (r_vec, t_vec );
//
//  return res;
//}

double ObjectRegistration::getAvgDistance( const double r_quat[4], const double t_vec[3] )
{
  if( !model_ptr_ )
    throw runtime_error("RasterObjectModel not set");

  setPos( r_quat, t_vec );
  if(!model_ptr_->allVisiblePoints())
    model_ptr_->setModelView(transf_.data(), transf_.block<3,1>(4,0).data());

  return avgDistance();
}

double ObjectRegistration::getAvgDistance( const Eigen::Quaterniond& r_quat,
                                           const Eigen::Vector3d& t_vec )
{
  if( !model_ptr_ )
    throw runtime_error("RasterObjectModel not set");

  setPos( r_quat, t_vec );
  if(!model_ptr_->allVisiblePoints())
    model_ptr_->setModelView(transf_.data(), transf_.block<3,1>(4,0).data());

  return avgDistance();
}


double ObjectRegistration::getAvgDistance( const Mat_< double >& r_vec,
                                               const Mat_< double >& t_vec )
{
  if( !model_ptr_ )
    throw runtime_error("RasterObjectModel not set");

  setPos( r_vec, t_vec );
  if(!model_ptr_->allVisiblePoints())
    model_ptr_->setModelView(transf_.data(), transf_.block<3,1>(4,0).data());

  return avgDistance();
}

//double ObjectRegistration::getAvgDistance( int idx )
//{
//  if( !model_ptr_ )
//    throw runtime_error("RasterObjectModel not set");
//
//  Eigen::Quaterniond quat;
//  Eigen::Vector3d t;
//  model_ptr_->modelView(idx, quat, t);
//  setPos ( quat, t );
//
//  return avgDistance(idx);
//}
void MultiViewObjectRegistration::setObjectModels(const std::vector<RasterObjectModel3DPtr> &model_ptrs,
                                                  const cv_ext::vector_Quaterniond &view_r_quats,
                                                  const cv_ext::vector_Vector3d &view_t_vec)
{
  if( model_ptrs.size() != view_r_quats.size() || model_ptrs.size() != view_r_quats.size() )
    throw runtime_error("Models and transformation vectors have different sizes");

  if( !model_ptrs.size() )
    throw runtime_error("Emty RasterObjectModel vector");

  for( auto &m : model_ptrs )
    if( !m )
      throw runtime_error("Null RasterObjectModel");

  model_ptrs_ = model_ptrs;
  cam_models_.reserve(model_ptrs.size());
  for( auto &m : model_ptrs )
    cam_models_.push_back(m->cameraModel());

  view_r_quats_ = view_r_quats;
  view_t_vec_ = view_t_vec;

  update_optimizer_ = true;
}
double MultiViewObjectRegistration::getAvgDistance(const double *r_quat, const double *t_vec)
{
  // TODO
  std::runtime_error("Implement me!!");
  return 0;
}

double MultiViewObjectRegistration::getAvgDistance(const Eigen::Quaterniond &r_quat, const Eigen::Vector3d &t_vec)
{
  // TODO
  std::runtime_error("Implement me!!");
  return 0;
}

double MultiViewObjectRegistration::getAvgDistance(const cv::Mat_<double> &r_vec, const cv::Mat_<double> &t_vec)
{
  // TODO
  std::runtime_error("Implement me!!");
  return 0;
}

double MultiViewObjectRegistration::refinePosition(double *r_quat, double *t_vec)
{
  if( !model_ptrs_.size() )
    throw runtime_error("RasterObjectModel not set");

  setPos( r_quat, t_vec );

  if(update_optimizer_ || !model_ptrs_[0]->allVisiblePoints())
  {
    double view_q[4], q_tot[4], t_tot[3];

    for( int i = 0; i < static_cast<int>(model_ptrs_.size()); i++ )
    {
      cv_ext::eigenQuat2Quat(view_r_quats_[i], view_q );
      ceres::QuaternionProduct( view_q, transf_.data(), q_tot);
      ceres::QuaternionRotatePoint(view_q, transf_.block<3,1>(4,0).data(), t_tot );

      t_tot[0] += view_t_vec_[i](0);
      t_tot[1] += view_t_vec_[i](1);
      t_tot[2] += view_t_vec_[i](2);

      model_ptrs_[i]->setModelView(q_tot, t_tot );
    }

    updateOptimizer();
    update_optimizer_ = false;
  }

  double res = optimize();
  getPos( r_quat, t_vec );

  return res;
}

double MultiViewObjectRegistration::refinePosition(Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec)
{
  if( !model_ptrs_.size() )
    throw runtime_error("RasterObjectModel not set");

  setPos( r_quat, t_vec );

  if(update_optimizer_ || !model_ptrs_[0]->allVisiblePoints())
  {
    double view_q[4], q_tot[4], t_tot[3];

    for( int i = 0; i < static_cast<int>(model_ptrs_.size()); i++ )
    {
      cv_ext::eigenQuat2Quat(view_r_quats_[i], view_q );
      ceres::QuaternionProduct( view_q, transf_.data(), q_tot);
      ceres::QuaternionRotatePoint(view_q, transf_.block<3,1>(4,0).data(), t_tot );

      t_tot[0] += view_t_vec_[i](0);
      t_tot[1] += view_t_vec_[i](1);
      t_tot[2] += view_t_vec_[i](2);

      model_ptrs_[i]->setModelView(q_tot, t_tot );
    }

    updateOptimizer();
    update_optimizer_ = false;
  }

  double res = optimize();
  getPos( r_quat, t_vec );

  return res;
}

double MultiViewObjectRegistration::refinePosition(cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec)
{
  if( !model_ptrs_.size() )
    throw runtime_error("RasterObjectModel not set");

  setPos( r_vec, t_vec );

  if(update_optimizer_ || !model_ptrs_[0]->allVisiblePoints())
  {
    double view_q[4], q_tot[4], t_tot[3];

    for( int i = 0; i < static_cast<int>(model_ptrs_.size()); i++ )
    {
      cv_ext::eigenQuat2Quat(view_r_quats_[i], view_q );
      ceres::QuaternionProduct( view_q, transf_.data(), q_tot);
      ceres::QuaternionRotatePoint(view_q, transf_.block<3,1>(4,0).data(), t_tot );

      t_tot[0] += view_t_vec_[i](0);
      t_tot[1] += view_t_vec_[i](1);
      t_tot[2] += view_t_vec_[i](2);

      model_ptrs_[i]->setModelView(q_tot, t_tot );
    }

    updateOptimizer();
    update_optimizer_ = false;
  }

  double res = optimize();
  getPos( r_vec, t_vec );

  return res;
}
