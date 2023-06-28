#include "chamfer_registration.h"

#include <algorithm>

#include <ceres/ceres.h>
#include "cv_ext/cv_ext.h"

using namespace cv;
using namespace cv_ext;
using namespace std;


class ChamferRegistration::Optimizer
{
public:
  Optimizer(){};
  ~Optimizer(){};
  ceres::Problem problem;
  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ChamferResidual
{
  ChamferResidual ( const cv_ext::PinholeCameraModel &cam_model, const Mat &distance_map,
                    const Point3f &model_pt ) :
    cam_model_ ( cam_model ),
    dist_map_ ( distance_map )
  {
    model_pt_[0] = model_pt.x;
    model_pt_[1] = model_pt.y;
    model_pt_[2] = model_pt.z;
  }

  template <typename _T>
  bool operator() ( const _T* const pos, _T* residuals ) const
  {

    _T model_pt[3] = { _T ( model_pt_[0] ), _T ( model_pt_[1] ), _T ( model_pt_[2] ) };
    _T proj_pt[2];

    cam_model_.quatRTProject ( pos, pos + 4, model_pt, proj_pt );

    if ( proj_pt[0] < _T ( 1 ) ||  proj_pt[1] < _T ( 1 ) ||
         proj_pt[0] > _T ( cam_model_.imgWidth() - 2 ) ||
         proj_pt[1] > _T ( cam_model_.imgHeight() - 2 ) )
    {
      residuals[0] = _T ( 0 );
      return true;
    }

    residuals[0] = cv_ext::bilinearInterp<float, _T> ( dist_map_, proj_pt[0], proj_pt[1] );
    //residuals[0] = cv_ext::getPix<float, _T> ( dist_map_, proj_pt[0], proj_pt[1] );
    return true;
  }

  static ceres::CostFunction* Create ( const cv_ext::PinholeCameraModel &cam_model,
                                       const Mat &dist_map,
                                       const Point3f &model_pt )
  {
    return ( new ceres::AutoDiffCostFunction<ChamferResidual, 1, 7 > (
               new ChamferResidual ( cam_model, dist_map, model_pt ) ) );
  }

  const cv_ext::PinholeCameraModel &cam_model_;
  const Mat &dist_map_;
  double model_pt_[3];
};


void ChamferRegistration::setInput ( const Mat& dist_map )
{
  assert( dist_map.type() == DataType<float>::type );
  assert( dist_map.rows == cam_model_.imgHeight() &&
          dist_map.cols  == cam_model_.imgWidth() );
  
  dist_map_ =  dist_map;
}

double ChamferRegistration::optimize()
{
  if( !optimizer_ptr_ )
    return -1;

  ceres::Solver::Options options;
  options.max_num_iterations = num_optim_iterations_;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = verbose_mode_;

  ceres::Solver::Summary summary;
  ceres::Solve ( options, &optimizer_ptr_->problem, &summary );

  return summary.final_cost;
}

double ChamferRegistration::avgDistance()
{
  int n_pts = 0;
  double avg_dist = 0;

  std::vector<Point2f> proj_pts;

  model_ptr_->projectRasterPoints( proj_pts );

  for( int i = 0; i < int(proj_pts.size()); i++ )
  {
    const Point2f &coord = proj_pts.at(i);
    if( coord.x >= 0)
    {
      n_pts++;
      int x = cvRound(coord.x), y = cvRound(coord.y);
      avg_dist += dist_map_.at<float>( y, x );
    }
  }

  if( n_pts )
    avg_dist /= n_pts;
  else
    avg_dist = std::numeric_limits< float >::max();

  return avg_dist;
}

void ChamferRegistration::updateOptimizer()
{
  const std::vector<Point3f> &model_pts = model_ptr_->getPoints();

  optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );

  for ( int i = 0; i < int(model_pts.size()); ++i )
  {
    ceres::CostFunction* cost_function =
      ChamferResidual::Create ( cam_model_, dist_map_, model_pts[i] );

    optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
  }

  if( ensure_cheirality_constraint_ )
    optimizer_ptr_->problem.SetParameterLowerBound(transf_.data(), 6, 0 );
}

// OrientedChamferRegistration::OrientedChamferRegistration()
// {
//   setNumDirections( 60 );  
// }
// 
// void OrientedChamferRegistration::setNumDirections ( int n )
// {
//   num_directions_ = n;
//   eta_direction_ = double(num_directions_)/M_PI;
// }
// 
// void OrientedChamferRegistration::setInput ( const Mat& dist_map, const cv::Mat &closest_dir_map )
// {
//   if( dist_map.type() != DataType<float>::type ||
//       dist_map.rows != cam_model_.imgHeight()||
//       dist_map.cols  != cam_model_.imgWidth() ||
//       closest_dir_map.type() != DataType<ushort>::type ||
//       closest_dir_map.rows != cam_model_.imgHeight()||
//       closest_dir_map.cols  != cam_model_.imgWidth() )
//     throw invalid_argument("Invalid input data");  
//   
//   dist_map_ =  dist_map;
//   closest_dir_map_ = closest_dir_map;
// }
// 
// double OrientedChamferRegistration::optimize()
// {
//   std::cerr<<"IMPLEMENT ME!!"<<std::endl;
//   return -1;
// }
// 
// double OrientedChamferRegistration::avgDistance( int idx )
// {
// 
//   std::cerr<<"IMPLEMENT ME!!"<<std::endl;
//   return -1;
// }
// 
// void OrientedChamferRegistration::updateOptimizer( int idx )
// {
//   std::cerr<<"IMPLEMENT ME!!"<<std::endl;
// }
// 


struct ICPChamferResidual
{
  ICPChamferResidual ( const cv_ext::PinholeCameraModel &cam_model,
                       const Point3f &model_pt,
                       const Point2f &img_pt ) :
    cam_model_ ( cam_model )
  {
    model_pt_[0] = model_pt.x;
    model_pt_[1] = model_pt.y;
    model_pt_[2] = model_pt.z;
    img_pt_[0] = img_pt.x;
    img_pt_[1] = img_pt.y;
  }

  template <typename _T>
  bool operator() ( const _T* const pos, _T* residuals ) const
  {

    _T model_pt[3] = { _T ( model_pt_[0] ), _T ( model_pt_[1] ), _T ( model_pt_[2] ) };
    _T img_pt[2] = {_T ( img_pt_[0] ), _T ( img_pt_[1] ) }, proj_pt[2];

    cam_model_.quatRTProject ( pos, pos + 4, model_pt, proj_pt );

    if ( proj_pt[0] < _T ( 1 ) ||  proj_pt[1] < _T ( 1 ) ||
         proj_pt[0] > _T ( cam_model_.imgWidth() - 2 ) ||
         proj_pt[1] > _T ( cam_model_.imgHeight() - 2 ) )
    {
      residuals[0] = _T ( 0 );
      residuals[1] = _T ( 0 );
      return true;
    }

    residuals[0] = img_pt[0] - proj_pt[0];
    residuals[1] = img_pt[1] - proj_pt[1];

    return true;
  }

  static ceres::CostFunction* Create ( const cv_ext::PinholeCameraModel &cam_model,
                                       const Point3f &model_pt, const Point2f &img_pt )
  {
    return ( new ceres::AutoDiffCostFunction<ICPChamferResidual, 2, 7 > (
               new ICPChamferResidual ( cam_model, model_pt, img_pt ) ) );
  }

  const cv_ext::PinholeCameraModel &cam_model_;
  double model_pt_[3], img_pt_[2];
};

class ICPChamferRegistration::Optimizer
{
public:
  Optimizer(){};
  ~Optimizer(){};
  ceres::Problem problem;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


void ICPChamferRegistration::setInput ( const Mat& closest_edgels_map )
{  
  assert( closest_edgels_map.type() == DataType<Point2f>::type );
  assert( closest_edgels_map.rows == cam_model_.imgHeight() &&
          closest_edgels_map.cols  == cam_model_.imgWidth() );
  
  closest_edgels_map_ =  closest_edgels_map;
}


double ICPChamferRegistration::avgDistance ()
{
  int n_pts = 0;
  double avg_dist = 0;

  std::vector<Point2f> proj_pts;

  model_ptr_->projectRasterPoints( proj_pts );

  for( int i = 0; i < int(proj_pts.size()); i++ )
  {
    const Point2f &coord = proj_pts.at(i);
    if( coord.x >= 0)
    {
      n_pts++;
      int x = cvRound(coord.x), y = cvRound(coord.y);
      const Point2f &closest_edgel = closest_edgels_map_.at<Point2f>( y, x );
      avg_dist += sqrt( (closest_edgel.x - coord.x)*(closest_edgel.x - coord.x) +
                         (closest_edgel.y - coord.y)*(closest_edgel.y - coord.y) );
    }
  }

  if( n_pts )
    avg_dist /= n_pts;
  else
    avg_dist = std::numeric_limits< float >::max();

  return avg_dist;
}

void ICPChamferRegistration::updateOptimizer ()
{
  model_pts_ = model_ptr_->getPoints();
}

double ICPChamferRegistration::optimize()
{

  double final_cost = -1.0;
  for( int icp_iter = 0; icp_iter < num_icp_iterations_; icp_iter++)
  {
    std::vector<Point2f> img_pts;
    model_ptr_->projectRasterPoints( transf_.data(), transf_.block<3,1>(4,0).data(), img_pts );

    optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );

    int res_size = 0;
    for ( int i = 0; i < int(model_pts_.size()); ++i )
    {
      if( img_pts[i].x >= 0 )
      {
        int x = cvRound(img_pts[i].x), y = cvRound(img_pts[i].y);
        const Point2f &closest_edgel = closest_edgels_map_.at<Point2f>( y, x );
        ceres::CostFunction* cost_function =
          ICPChamferResidual::Create ( cam_model_, model_pts_[i], closest_edgel );

        optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
        res_size++;
      }
    }

    if( ensure_cheirality_constraint_ )
      optimizer_ptr_->problem.SetParameterLowerBound(transf_.data(), 6, 0 );

    if( res_size >= 3 )
    {
      ceres::Solver::Options options;
      options.max_num_iterations = num_optim_iterations_;
      options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
      options.minimizer_progress_to_stdout = verbose_mode_;

      ceres::Solver::Summary summary;
      ceres::Solve ( options, &optimizer_ptr_->problem, &summary );
      final_cost = summary.final_cost;
    }
  }
  return final_cost;
}

class DirectionalChamferRegistration::Optimizer
{
public:
  Optimizer(){};
  ~Optimizer(){};
  ceres::Problem problem;
  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


struct DirectionalChamferResidual
{
  DirectionalChamferResidual ( const cv_ext::PinholeCameraModel &cam_model,
                               const ImageTensorPtr &dist_map_tensor_ptr,
                               const Point3f &model_pt, const Point3f &model_dpt ) :
    cam_model_ ( cam_model ),
    dist_map_tensor_ptr_ ( dist_map_tensor_ptr ),
    dist_map_tensor_( *dist_map_tensor_ptr_ )
  {
    model_pt_[0] = model_pt.x;
    model_pt_[1] = model_pt.y;
    model_pt_[2] = model_pt.z;

    model_dpt_[0] = model_dpt.x;
    model_dpt_[1] = model_dpt.y;
    model_dpt_[2] = model_dpt.z;

    eta_direction_ = double(dist_map_tensor_.depth())/M_PI;

    //imshow("tensor[0]", dist_map_tensor_[23]);
    //waitKey(0);
  }

  template <typename _T>
  bool operator() ( const _T* const pos, _T* residuals ) const
  {

    _T model_pt[3] = { _T ( model_pt_[0] ), _T ( model_pt_[1] ), _T ( model_pt_[2] ) };
    _T model_dpt[3] = { _T ( model_dpt_[0] ), _T ( model_dpt_[1] ), _T ( model_dpt_[2] ) };
    _T proj_pt[2], proj_dpt[2];

    cam_model_.quatRTProject ( pos, pos + 4, model_pt, proj_pt );
    cam_model_.quatRTProject ( pos, pos + 4, model_dpt, proj_dpt );

   if ( proj_pt[0] < _T ( 1 ) ||  proj_pt[1] < _T ( 1 ) ||
        proj_pt[0] > _T ( cam_model_.imgWidth() - 2 ) ||
        proj_pt[1] > _T ( cam_model_.imgHeight() - 2 ) ||
        proj_dpt[0] < _T ( 1 ) ||  proj_dpt[1] < _T ( 1 ) ||
        proj_dpt[0] > _T ( cam_model_.imgWidth() - 2 ) ||
        proj_dpt[1] > _T ( cam_model_.imgHeight() - 2 ) )
    {
      residuals[0] = _T ( 0 );
      return true;
    }

    _T diff[2] = { proj_dpt[0] - proj_pt[0], proj_dpt[1] - proj_pt[1] };
    _T direction;

    if( diff[0] != _T(0) )
      direction = atan( diff[1]/diff[0] );
    else
      direction = _T(-M_PI/2);

    _T z = _T(eta_direction_) * ( direction + _T(M_PI/2) );
    //residuals[0] = cv_ext::tensorGetPix<float, _T> ( dist_map_tensor_, proj_pt[0], proj_pt[1], z );
    residuals[0] = cv_ext::tensorbilinearInterp<float, _T> ( dist_map_tensor_, proj_pt[0], proj_pt[1], z );
    return true;
  }

  static ceres::CostFunction* Create ( const cv_ext::PinholeCameraModel &cam_model,
                                       const ImageTensorPtr &dist_map_tensor_ptr,
                                       const Point3f &model_pt, const Point3f &model_dpt)
  {
    return ( new ceres::AutoDiffCostFunction<DirectionalChamferResidual, 1, 7 > (
               new DirectionalChamferResidual( cam_model, dist_map_tensor_ptr, model_pt, model_dpt ) ) );
  }

  const cv_ext::PinholeCameraModel &cam_model_;
  const ImageTensorPtr dist_map_tensor_ptr_;
  const ImageTensor &dist_map_tensor_;
  double model_pt_[3], model_dpt_[3];
  double eta_direction_;
};

DirectionalChamferRegistration::DirectionalChamferRegistration()
{
  eta_direction_ = double(num_directions_)/M_PI;
}

void DirectionalChamferRegistration::setNumDirections(int n )
{
  num_directions_ = n;
  eta_direction_ = double(num_directions_)/M_PI;
}

void DirectionalChamferRegistration::setInput( const ImageTensorPtr& dist_map_tensor_ptr )
{
  assert( dist_map_tensor_ptr->depth() == num_directions_ );
  assert( dist_map_tensor_ptr->at(0).type() == DataType<float>::type );
  assert( dist_map_tensor_ptr->at(0).rows == cam_model_.imgHeight()&&
          dist_map_tensor_ptr->at(0).cols  == cam_model_.imgWidth() );

  dist_map_tensor_ptr_ = dist_map_tensor_ptr;
}


double DirectionalChamferRegistration::avgDistance( )
{
  std::runtime_error("DirectionalChamferRegistration::avgDistance NOT IMPLEMENETD");
  return 0;
}

double DirectionalChamferRegistration::optimize()
{
  if( !optimizer_ptr_ )
    return -1;

  ceres::Solver::Options options;
  options.max_num_iterations = num_optim_iterations_;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = verbose_mode_;

  ceres::Solver::Summary summary;
  ceres::Solve ( options, &optimizer_ptr_->problem, &summary );

  return summary.final_cost;
}

void DirectionalChamferRegistration::updateOptimizer()
{
  const std::vector<Point3f> &model_pts = model_ptr_->getPoints();
  const std::vector<Point3f> &model_dpts = model_ptr_->getDPoints();

  optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );

  for ( int i = 0; i < int(model_pts.size()); ++i )
  {
    ceres::CostFunction* cost_function =
      DirectionalChamferResidual::Create ( cam_model_, dist_map_tensor_ptr_, model_pts[i], model_dpts[i] );

    optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
  }

  if( ensure_cheirality_constraint_ )
    optimizer_ptr_->problem.SetParameterLowerBound(transf_.data(), 6, 0 );
}

double DirectionalChamferRegistration::refinePosition( const ObjectTemplate &templ,
                                                       Eigen::Quaterniond r_quat,
                                                       Eigen::Vector3d t_vec )
{
  assert( model_ptr_ != nullptr );

  const DirIdxPointSet &dips = dynamic_cast<const DirIdxPointSet&>(templ);

  auto &model_pts = dips.obj_pts;
  auto &model_dpts = dips.obj_d_pts;

  setPos( r_quat, t_vec );

  optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );

  for ( int i = 0; i < int(model_pts.size()); ++i )
  {
    ceres::CostFunction* cost_function =
        DirectionalChamferResidual::Create ( cam_model_, dist_map_tensor_ptr_, model_pts[i], model_dpts[i] );

    optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
  }

  if( ensure_cheirality_constraint_ )
    optimizer_ptr_->problem.SetParameterLowerBound(transf_.data(), 6, 0 );

  double res = optimize();
  getPos( r_quat, t_vec );

  return res;
}

double DirectionalChamferRegistration::refinePosition( const ObjectTemplate &templ,
                                                       cv::Mat_<double> &r_vec,
                                                       cv::Mat_<double> &t_vec)
{
  assert( model_ptr_ != nullptr );

  const DirIdxPointSet &dips = dynamic_cast<const DirIdxPointSet&>(templ);

  auto &model_pts = dips.obj_pts;
  auto &model_dpts = dips.obj_d_pts;

  setPos( r_vec, t_vec );

  optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );

  for ( int i = 0; i < int(model_pts.size()); ++i )
  {
    ceres::CostFunction* cost_function =
        DirectionalChamferResidual::Create ( cam_model_, dist_map_tensor_ptr_, model_pts[i], model_dpts[i] );

    optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
  }

  if( ensure_cheirality_constraint_ )
    optimizer_ptr_->problem.SetParameterLowerBound(transf_.data(), 6, 0 );

  double res = optimize();
  getPos( r_vec, t_vec );

  return res;
}

class HybridDirectionalChamferRegistration::Optimizer
{
public:
  Optimizer(){};
  ~Optimizer(){};
  ceres::Problem problem;
  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

HybridDirectionalChamferRegistration::HybridDirectionalChamferRegistration ()
{
  eta_direction_ = double(num_directions_)/M_PI;
}

void HybridDirectionalChamferRegistration::setNumDirections(int n )
{
  num_directions_ = n;
  eta_direction_ = double(num_directions_)/M_PI;
}

void HybridDirectionalChamferRegistration::setInput ( const ImageTensorPtr& dist_map_tensor_ptr,
                                                  const ImageTensorPtr& edgels_map_tensor_ptr )
{
  assert( dist_map_tensor_ptr->depth() == num_directions_ );
  assert( dist_map_tensor_ptr->at(0).type() == DataType<float>::type );
  assert( dist_map_tensor_ptr->at(0).rows == cam_model_.imgHeight() &&
          dist_map_tensor_ptr->at(0).cols  == cam_model_.imgWidth() );
  assert( edgels_map_tensor_ptr->depth() == num_directions_ );
  assert( edgels_map_tensor_ptr->at(0).type() == DataType<Point2f>::type );
  assert( edgels_map_tensor_ptr->at(0).rows == cam_model_.imgHeight() &&
          edgels_map_tensor_ptr->at(0).cols  == cam_model_.imgWidth() );
  
  dist_map_tensor_ptr_ = dist_map_tensor_ptr;
  edgels_map_tensor_ptr_ = edgels_map_tensor_ptr;
}

double HybridDirectionalChamferRegistration::avgDistance ()
{
  int n_pts = 0;
  double avg_dist = 0;

  std::vector<Point2f> proj_pts;
  std::vector<float> normal_directions;

  model_ptr_->projectRasterPoints( proj_pts, normal_directions );

  ImageTensor &dist_map_tensor = *dist_map_tensor_ptr_;
  for( int i = 0; i < int(proj_pts.size()); i++ )
  {
    const Point2f &coord = proj_pts.at(i);
    if( coord.x >= 0)
    {
      n_pts++;

      float direction = normal_directions[i] + M_PI/2;
      if( direction >= M_PI/2 )
        direction -= M_PI;
      direction += M_PI/2;

      int x = cvRound(coord.x), y = cvRound(coord.y);
      int z = cvRound(eta_direction_*direction);
      z %= num_directions_;

      avg_dist += dist_map_tensor[z].at<float>( y, x );
    }
  }

  if( n_pts )
    avg_dist /= n_pts;
  else
    avg_dist = std::numeric_limits< float >::max();

  return avg_dist;
}

double HybridDirectionalChamferRegistration::optimize()
{
  double final_cost = -1.0;
  ImageTensor &edgels_map_tensor = *edgels_map_tensor_ptr_;

  for( int icp_iter = 0; icp_iter < max_icp_iterations_; icp_iter++)
  {

    d2co_optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );

    for ( int i = 0; i < int(model_pts_.size()); ++i )
    {
      ceres::CostFunction* cost_function =
        DirectionalChamferResidual::Create ( cam_model_, dist_map_tensor_ptr_, model_pts_[i], model_dpts_[i] );

      d2co_optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
    }

    if( ensure_cheirality_constraint_ )
      d2co_optimizer_ptr_->problem.SetParameterLowerBound(transf_.data(), 6, 0 );

    ceres::Solver::Options options;
    //options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = num_optim_iterations_;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = verbose_mode_;

    ceres::Solver::Summary summary;
    ceres::Solve ( options, &d2co_optimizer_ptr_->problem, &summary );

    std::vector<Point2f> img_pts;
    std::vector<float> normal_directions;
    model_ptr_->projectRasterPoints( transf_.data(), transf_.block<3,1>(4,0).data(),
                                     img_pts, normal_directions );

    int res_size = 0;
    for ( int i = 0; i < int(model_pts_.size()); i++  )
    {
      if( img_pts[i].x >= 0 )
      {

        float direction = normal_directions[i] + M_PI/2;
        if( direction >= M_PI/2 )
          direction -= M_PI;
        direction += M_PI/2;

        int x = cvRound(img_pts[i].x), y = cvRound(img_pts[i].y);
        int z = cvRound(eta_direction_*direction);
        z %= num_directions_;

        const Point2f &closest_edgel = edgels_map_tensor[z].at<Point2f>( y, x );

        ceres::CostFunction* cost_function =
          ICPChamferResidual::Create ( cam_model_, model_pts_[i], closest_edgel );

        icp_optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
        res_size++;
      }
    }

    if( ensure_cheirality_constraint_ )
      icp_optimizer_ptr_->problem.SetParameterLowerBound(transf_.data(), 6, 0 );

    if( res_size >= 3 )
    {
      ceres::Solver::Options options;
      //options.linear_solver_type = ceres::DENSE_SCHUR;
      options.max_num_iterations = num_optim_iterations_;
      options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
      options.minimizer_progress_to_stdout = verbose_mode_;

      ceres::Solver::Summary summary;
      ceres::Solve ( options, &icp_optimizer_ptr_->problem, &summary );
      final_cost = summary.final_cost;
    }
    else
      break;
  }
  return final_cost;
}

void HybridDirectionalChamferRegistration::updateOptimizer ()
{
  model_pts_ = model_ptr_->getPoints();
  model_dpts_ = model_ptr_->getDPoints();

//   d2co_optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );
//
//   for ( int i = 0; i < model_pts_.size(); ++i )
//   {
//     ceres::CostFunction* cost_function =
//       DirectionalChamferResidual::Create ( cam_model_, dist_map_tensor_ptr_, model_pts_[i], model_dpts_[i] );
//
//     d2co_optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
//   }
}


class BidirectionalChamferRegistration::Optimizer
{
public:
  Optimizer(){};
  ~Optimizer(){};
  ceres::Problem problem;
  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct BidirectionalChamferResidual
{
  BidirectionalChamferResidual ( const cv_ext::PinholeCameraModel &cam_model,
                                 const ImageTensorPtr &x_dist_map_tensor_ptr,
                                 const ImageTensorPtr &y_dist_map_tensor_ptr,
                                 const Point3f &model_pt, const Point3f &model_dpt ) :
    cam_model_ ( cam_model ),
    x_dist_map_tensor_ptr_ ( x_dist_map_tensor_ptr ),
    y_dist_map_tensor_ptr_ ( y_dist_map_tensor_ptr ),
    x_dist_map_tensor_( *x_dist_map_tensor_ptr_ ),
    y_dist_map_tensor_( *y_dist_map_tensor_ptr_ )
  {
    model_pt_[0] = model_pt.x;
    model_pt_[1] = model_pt.y;
    model_pt_[2] = model_pt.z;

    model_dpt_[0] = model_dpt.x;
    model_dpt_[1] = model_dpt.y;
    model_dpt_[2] = model_dpt.z;

    eta_direction_ = double(x_dist_map_tensor_.depth())/M_PI;
  }

  template <typename _T>
  bool operator() ( const _T* const pos, _T* residuals ) const
  {

    _T model_pt[3] = { _T ( model_pt_[0] ), _T ( model_pt_[1] ), _T ( model_pt_[2] ) };
    _T model_dpt[3] = { _T ( model_dpt_[0] ), _T ( model_dpt_[1] ), _T ( model_dpt_[2] ) };
    _T proj_pt[2], proj_dpt[2];

    cam_model_.quatRTProject ( pos, pos + 4, model_pt, proj_pt );
    cam_model_.quatRTProject ( pos, pos + 4, model_dpt, proj_dpt );

   if ( proj_pt[0] < _T ( 1 ) ||  proj_pt[1] < _T ( 1 ) ||
        proj_pt[0] > _T ( cam_model_.imgWidth() - 2 ) ||
        proj_pt[1] > _T ( cam_model_.imgHeight() - 2 ) ||
        proj_dpt[0] < _T ( 1 ) ||  proj_dpt[1] < _T ( 1 ) ||
        proj_dpt[0] > _T ( cam_model_.imgWidth() - 2 ) ||
        proj_dpt[1] > _T ( cam_model_.imgHeight() - 2 ) )
    {
      residuals[0] = _T ( 0 );
      residuals[1] = _T ( 0 );
      return true;
    }

    _T diff[2] = { proj_dpt[0] - proj_pt[0], proj_dpt[1] - proj_pt[1] };
    _T direction;

    if( diff[0] != _T(0) )
      direction = atan( diff[1]/diff[0] );
    else
      direction = _T(-M_PI/2);

    _T z = _T(eta_direction_) * ( direction + _T(M_PI/2) );
    //residuals[0] = cv_ext::tensorGetPix<float, _T> ( dist_map_tensor_, proj_pt[0], proj_pt[1], z );
    residuals[0] = cv_ext::tensorbilinearInterp<float, _T> ( x_dist_map_tensor_, proj_pt[0], proj_pt[1], z );
    residuals[1] = cv_ext::tensorbilinearInterp<float, _T> ( y_dist_map_tensor_, proj_pt[0], proj_pt[1], z );

    return true;
  }

  static ceres::CostFunction* Create ( const cv_ext::PinholeCameraModel &cam_model,
                                       const ImageTensorPtr &x_dist_map_tensor_ptr,
                                       const ImageTensorPtr &y_dist_map_tensor_ptr,
                                       const Point3f &model_pt, const Point3f &model_dpt)
  {
    return ( new ceres::AutoDiffCostFunction<BidirectionalChamferResidual, 2, 7 > (
               new BidirectionalChamferResidual( cam_model, x_dist_map_tensor_ptr, y_dist_map_tensor_ptr,
                                                 model_pt, model_dpt ) ) );
  }

  const cv_ext::PinholeCameraModel &cam_model_;
  const ImageTensorPtr x_dist_map_tensor_ptr_, y_dist_map_tensor_ptr_;
  const ImageTensor &x_dist_map_tensor_, y_dist_map_tensor_;
  double model_pt_[3], model_dpt_[3];
  double eta_direction_;
};

BidirectionalChamferRegistration::BidirectionalChamferRegistration()
{
  eta_direction_ = double(num_directions_)/M_PI;
}

void BidirectionalChamferRegistration::setNumDirections(int n )
{
  num_directions_ = n;
  eta_direction_ = double(num_directions_)/M_PI;
}

void BidirectionalChamferRegistration ::setInput ( const ImageTensorPtr& x_dist_map_tensor_ptr,
                                               const ImageTensorPtr& y_dist_map_tensor_ptr )
{
  assert( x_dist_map_tensor_ptr->depth() == num_directions_ );
  assert( x_dist_map_tensor_ptr->at(0).type() == DataType<float>::type );
  assert( x_dist_map_tensor_ptr->at(0).rows == cam_model_.imgHeight() &&
          x_dist_map_tensor_ptr->at(0).cols  == cam_model_.imgWidth() );
  assert( y_dist_map_tensor_ptr->depth() == num_directions_ );
  assert( y_dist_map_tensor_ptr->at(0).type() == DataType<float>::type );
  assert( y_dist_map_tensor_ptr->at(0).rows == cam_model_.imgHeight() &&
          y_dist_map_tensor_ptr->at(0).cols  == cam_model_.imgWidth() );
  
  x_dist_map_tensor_ptr_ = x_dist_map_tensor_ptr;
  y_dist_map_tensor_ptr_ = y_dist_map_tensor_ptr;
}

double BidirectionalChamferRegistration ::avgDistance()
{
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
  return -1;
}


double BidirectionalChamferRegistration ::optimize()
{
  if( !optimizer_ptr_ )
    return -1;

  ceres::Solver::Options options;
  //options.linear_solver_type = ceres::DENSE_SCHUR;
  options.max_num_iterations = num_optim_iterations_;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = verbose_mode_;

  ceres::Solver::Summary summary;
  ceres::Solve ( options, &optimizer_ptr_->problem, &summary );

  return summary.final_cost;
}

void BidirectionalChamferRegistration ::updateOptimizer()
{
  const std::vector<Point3f> &model_pts = model_ptr_->getPoints();
  const std::vector<Point3f> &model_dpts = model_ptr_->getDPoints();

  optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );

  for ( int i = 0; i < int(model_pts.size()); ++i )
  {
    ceres::CostFunction* cost_function =
      BidirectionalChamferResidual::Create ( cam_model_, x_dist_map_tensor_ptr_, y_dist_map_tensor_ptr_,
                                             model_pts[i], model_dpts[i] );

    optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
  }

  if( ensure_cheirality_constraint_ )
    optimizer_ptr_->problem.SetParameterLowerBound(transf_.data(), 6, 0 );
}

class ICPDirectionalChamferRegistration::Optimizer
{
public:
  Optimizer(){};
  ~Optimizer(){};
  ceres::Problem problem;
  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

ICPDirectionalChamferRegistration::ICPDirectionalChamferRegistration ()
{
  eta_direction_ = double(num_directions_)/M_PI;
}

void ICPDirectionalChamferRegistration::setNumDirections(int n )
{
  num_directions_ = n;
  eta_direction_ = double(num_directions_)/M_PI;
}

void ICPDirectionalChamferRegistration::setInput ( const ImageTensorPtr& edgels_map_tensor_ptr )
{
  assert( edgels_map_tensor_ptr->depth() == num_directions_ );
  assert( edgels_map_tensor_ptr->at(0).type() == DataType<Point2f>::type );
  assert( edgels_map_tensor_ptr->at(0).rows == cam_model_.imgHeight() &&
          edgels_map_tensor_ptr->at(0).cols  == cam_model_.imgWidth() );

  edgels_map_tensor_ptr_ = edgels_map_tensor_ptr;
}

double ICPDirectionalChamferRegistration::avgDistance ()
{
  int n_pts = 0;
  double avg_dist = 0;

  std::vector<Point2f> proj_pts;
  std::vector<float> normal_directions;
  model_ptr_->projectRasterPoints( proj_pts, normal_directions );

  ImageTensor &edgels_map_tensor = *edgels_map_tensor_ptr_;
  for( int i = 0; i < int(proj_pts.size()); i++ )
  {
    const Point2f &coord = proj_pts.at(i);
    if( coord.x >= 0)
    {
      n_pts++;

      float direction = normal_directions[i] + M_PI/2;
      if( direction >= M_PI/2 )
        direction -= M_PI;
      direction += M_PI/2;

      int x = cvRound(coord.x), y = cvRound(coord.y);
      int z = cvRound(eta_direction_*direction);
      z %= num_directions_;

      const Point2f &closest_edgel = edgels_map_tensor[z].at<Point2f>( y, x );
      avg_dist += sqrt( (closest_edgel.x - coord.x)*(closest_edgel.x - coord.x) +
                        (closest_edgel.y - coord.y)*(closest_edgel.y - coord.y) );

    }
  }

  if( n_pts )
    avg_dist /= n_pts;
  else
    avg_dist = std::numeric_limits< float >::max();

  return avg_dist;
}

void ICPDirectionalChamferRegistration::updateOptimizer ()
{
  model_pts_ = model_ptr_->getPoints();
}

double ICPDirectionalChamferRegistration::optimize()
{
  double final_cost = -1.0;
  ImageTensor &edgels_map_tensor = *edgels_map_tensor_ptr_;

  for( int icp_iter = 0; icp_iter < num_icp_iterations_; icp_iter++)
  {

    std::vector<Point2f> img_pts;
    std::vector<float> normal_directions;


    model_ptr_->projectRasterPoints( transf_.data(), transf_.block<3,1>(4,0).data(),
                                     img_pts, normal_directions );

    optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );

    int res_size = 0;
    for ( int i = 0; i < int(model_pts_.size()); ++i )
    {
      if( img_pts[i].x >= 0 )
      {

        float direction = normal_directions[i] + M_PI/2;
        if( direction >= M_PI/2 )
          direction -= M_PI;
        direction += M_PI/2;

        int x = cvRound(img_pts[i].x), y = cvRound(img_pts[i].y);
        int z = cvRound(eta_direction_*direction);
        z %= num_directions_;

        const Point2f &closest_edgel = edgels_map_tensor[z].at<Point2f>( y, x );

        ceres::CostFunction* cost_function =
          ICPChamferResidual::Create ( cam_model_, model_pts_[i], closest_edgel );

        optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
        res_size++;
      }
    }

    if( ensure_cheirality_constraint_ )
      optimizer_ptr_->problem.SetParameterLowerBound(transf_.data(), 6, 0 );

    if( res_size >= 3 )
    {
      ceres::Solver::Options options;
      //options.linear_solver_type = ceres::DENSE_SCHUR;
      options.max_num_iterations = num_optim_iterations_;
      options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
      options.minimizer_progress_to_stdout = verbose_mode_;

      ceres::Solver::Summary summary;
      ceres::Solve ( options, &optimizer_ptr_->problem, &summary );
      final_cost = summary.final_cost;
    }
  }
  return final_cost;
}

class MultiViewsDirectionalChamferRegistration::Optimizer
{
public:
  Optimizer(){};
  ~Optimizer(){};
  ceres::Problem problem;
  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


struct MultiViewsDirectionalChamferResidual
{
  MultiViewsDirectionalChamferResidual(const cv_ext::PinholeCameraModel &cam_model,
                                       const ImageTensorPtr &dist_map_tensor_ptr,
                                       const Point3f &model_pt, const Point3f &model_dpt,
                                       const Eigen::Quaterniond &view_q, const Eigen::Vector3d &view_t) :
      cam_model_(cam_model),
      dist_map_tensor_ptr_(dist_map_tensor_ptr),
      dist_map_tensor_(*dist_map_tensor_ptr)
  {
    model_pt_[0] = model_pt.x;
    model_pt_[1] = model_pt.y;
    model_pt_[2] = model_pt.z;

    model_dpt_[0] = model_dpt.x;
    model_dpt_[1] = model_dpt.y;
    model_dpt_[2] = model_dpt.z;

    eta_direction_ = double(dist_map_tensor_.depth()) / M_PI;

    view_q_[0] = view_q.w();
    view_q_[1] = view_q.x();
    view_q_[2] = view_q.y();
    view_q_[3] = view_q.z();
    view_t_[0] = view_t(0);
    view_t_[1] = view_t(1);
    view_t_[2] = view_t(2);
  }

  template<typename _T>
  bool operator()(const _T *const pos, _T *residuals) const
  {
    _T model_pt[3] = {_T(model_pt_[0]), _T(model_pt_[1]), _T(model_pt_[2])};
    _T model_dpt[3] = {_T(model_dpt_[0]), _T(model_dpt_[1]), _T(model_dpt_[2])};
    _T proj_pt[2], proj_dpt[2];

    _T view_q[4] = {_T(view_q_[0]), _T(view_q_[1]), _T(view_q_[2]), _T(view_q_[3])}, q_tot[4];
    _T t_tot[3];

    ceres::QuaternionProduct(view_q, pos, q_tot);
    ceres::QuaternionRotatePoint(view_q, pos + 4, t_tot);
    t_tot[0] += _T(view_t_[0]);
    t_tot[1] += _T(view_t_[1]);
    t_tot[2] += _T(view_t_[2]);

    cam_model_.quatRTProject(q_tot, t_tot, model_pt, proj_pt);
    cam_model_.quatRTProject(q_tot, t_tot, model_dpt, proj_dpt);

    if (proj_pt[0] < _T(1) || proj_pt[1] < _T(1) ||
        proj_pt[0] > _T(cam_model_.imgWidth() - 2) ||
        proj_pt[1] > _T(cam_model_.imgHeight() - 2) ||
        proj_dpt[0] < _T(1) || proj_dpt[1] < _T(1) ||
        proj_dpt[0] > _T(cam_model_.imgWidth() - 2) ||
        proj_dpt[1] > _T(cam_model_.imgHeight() - 2))
    {
      residuals[0] = _T(0);
      return true;
    }

    _T diff[2] = {proj_dpt[0] - proj_pt[0], proj_dpt[1] - proj_pt[1]};
    _T direction;

    if (diff[0] != _T(0))
      direction = atan(diff[1] / diff[0]);
    else
      direction = _T(-M_PI / 2);

    _T z = _T(eta_direction_) * (direction + _T(M_PI / 2));
    residuals[0] = cv_ext::tensorGetPix<float, _T>(dist_map_tensor_, proj_pt[0], proj_pt[1], z);
    //residuals[0] = cv_ext::tensorbilinearInterp<float, _T> ( dist_map_tensor_, proj_pt[0], proj_pt[1], z );

    return true;
  }

  static ceres::CostFunction* Create ( const cv_ext::PinholeCameraModel &cam_model,
                                       const ImageTensorPtr &dist_map_tensor_ptr,
                                       const Point3f &model_pt, const Point3f &model_dpt,
                                       const Eigen::Quaterniond &view_q, const Eigen::Vector3d &view_t)
  {
    return ( new ceres::AutoDiffCostFunction<MultiViewsDirectionalChamferResidual, 1, 7 > (
               new MultiViewsDirectionalChamferResidual( cam_model, dist_map_tensor_ptr, model_pt, model_dpt, view_q, view_t ) ) );
  }

  const cv_ext::PinholeCameraModel &cam_model_;
  const ImageTensorPtr dist_map_tensor_ptr_;
  const ImageTensor & dist_map_tensor_;
  double model_pt_[3], model_dpt_[3];
  double eta_direction_;
  double view_q_[4], view_t_[3];
};

MultiViewsDirectionalChamferRegistration::MultiViewsDirectionalChamferRegistration()
{
  eta_direction_ = double(num_directions_)/M_PI;
}

void MultiViewsDirectionalChamferRegistration::setNumDirections(int n )
{
  num_directions_ = n;
  eta_direction_ = double(num_directions_)/M_PI;
}

void MultiViewsDirectionalChamferRegistration::
  setInput( const std::vector < cv_ext::ImageTensorPtr > &dist_map_tensor_ptrs )
{
  dist_map_tensor_ptrs_ = dist_map_tensor_ptrs;
}


double MultiViewsDirectionalChamferRegistration::avgDistance()
{
  // TODO
  return -1;
//   int n_pts = 0;
//   double avg_dist = 0;
//
//   std::vector<Point2f> proj_pts;
//   std::vector<float> normal_directions;
//
//   if( idx < 0 )
//     model_ptr_->projectRasterPoints( proj_pts, normal_directions );
//   else
//     model_ptr_->projectRasterPoints( idx, proj_pts, normal_directions );
//
//   std::vector<Mat > &dist_map_tensor = *dist_map_tensor_ptr_;
//   for( int i = 0; i < proj_pts.size(); i++ )
//   {
//     const Point2f &coord = proj_pts.at(i);
//     if( coord.x >= 0)
//     {
//       n_pts++;
//
//       float direction = normal_directions[i] + M_PI/2;
//       if( direction >= M_PI/2 )
//         direction -= M_PI;
//       direction += M_PI/2;
//
//       int x = cvRound(coord.x), y = cvRound(coord.y);
//       int z = cvRound(eta_direction_*direction);
//       z %= num_directions_;
//
//       avg_dist += dist_map_tensor[z].at<float>( y, x );
//     }
//   }
//
//   if( n_pts )
//     avg_dist /= n_pts;
//   else
//     avg_dist = std::numeric_limits< float >::max();
//
//   return avg_dist;
}


double MultiViewsDirectionalChamferRegistration::optimize()
{
  if( !optimizer_ptr_ )
    return -1;

  ceres::Solver::Options options;
  //options.linear_solver_type = ceres::DENSE_SCHUR;
  options.max_num_iterations = num_optim_iterations_;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = verbose_mode_;

  ceres::Solver::Summary summary;
  ceres::Solve ( options, &optimizer_ptr_->problem, &summary );

  return summary.final_cost;
}
  
void MultiViewsDirectionalChamferRegistration::updateOptimizer()
{
  optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );

  for( int i = 0; i < static_cast<int>(model_ptrs_.size()); i++ )
  {
    const std::vector<Point3f> &model_pts = model_ptrs_[i]->getPoints();
    const std::vector<Point3f> &model_dpts = model_ptrs_[i]->getDPoints();

    for ( int k = 0; k < static_cast<int>(model_pts.size()); k++ )
    {
      ceres::CostFunction* cost_function =
        MultiViewsDirectionalChamferResidual::Create ( cam_models_[i], dist_map_tensor_ptrs_[i],
                                                       model_pts[k], model_dpts[k],
                                                       view_r_quats_[i], view_t_vec_[i] );
      optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
    }
  }

  if( ensure_cheirality_constraint_ )
    optimizer_ptr_->problem.SetParameterLowerBound(transf_.data(), 6, 0 );

}
double MultiViewsDirectionalChamferRegistration::refinePosition( const ObjectTemplate &templ,
                                                                 cv::Mat_<double> &r_vec,
                                                                 cv::Mat_<double> &t_vec )
{
  assert( model_ptrs_.size() );

  const DirIdxPointSet &dips = dynamic_cast<const DirIdxPointSet&>(templ);

  auto &model_pts = dips.obj_pts;
  auto &model_dpts = dips.obj_d_pts;

  setPos( r_vec, t_vec );

  optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );

  for( int i = 0; i < static_cast<int>(model_ptrs_.size()); i++ )
  {
    for ( int k = 0; k < static_cast<int>(model_pts.size()); k++ )
    {
      ceres::CostFunction* cost_function =
          MultiViewsDirectionalChamferResidual::Create ( cam_models_[i], dist_map_tensor_ptrs_[i],
                                                         model_pts[k], model_dpts[k],
                                                         view_r_quats_[i], view_t_vec_[i] );
      optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
    }
  }

  if( ensure_cheirality_constraint_ )
    optimizer_ptr_->problem.SetParameterLowerBound(transf_.data(), 6, 0 );

  double res = optimize();
  getPos( r_vec, t_vec );

  return res;
}