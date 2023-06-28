#include "direct_matching.h"
/*
#include <ceres/ceres.h>

void computeGradientMagnitudePyr ( const cv::Mat& src_img, ScaledImagesListPtr &g_mag_pyr_ptr,
                                   unsigned int pyr_levels, double smooth_std )
{
  cv_ext::ImageStatisticsPtr img_stats_ptr =
    cv_ext::ImageStatistics::createImageStatistics ( src_img, false, pyr_levels, 2.0 );

  g_mag_pyr_ptr = ScaledImagesListPtr( new std::vector< ScaledImage >() );
  
  cv::Mat in_g_img =  img_stats_ptr->getEigenMagnitudesImage ( 0 ), 
          out_g_img(in_g_img.rows, in_g_img.cols, cv::DataType<float>::type);
  cv::normalize(in_g_img, out_g_img, 0, 1, cv::NORM_MINMAX);
  if( smooth_std > 0.0 )
    cv::GaussianBlur(out_g_img, out_g_img,cv::Size(0,0),smooth_std);    
  out_g_img = 1.0f - out_g_img;
  
  ScaledImage scaled_img;
  scaled_img.img = out_g_img;
  scaled_img.scale = img_stats_ptr->getPyrScale(0);
  
  g_mag_pyr_ptr->push_back( scaled_img );
  
  for ( unsigned int i = 0 ; i < pyr_levels; i++ )
  {
    in_g_img =  img_stats_ptr->getEigenMagnitudesImage ( i );
    cv::Mat g_img(in_g_img.rows, in_g_img.cols, cv::DataType<float>::type), g_img_sat;
    cv::normalize(in_g_img, g_img, 0, 255, cv::NORM_MINMAX);
    
    int hist_bins = 256;
    const int hist_sizes[] = { hist_bins };
    float hist_range0[] = { 0, float(hist_bins) };
    const float* hist_ranges[] = { hist_range0 };
    cv::Mat hist;
    int channels[] = {0};

    cv::calcHist( &g_img, 1, channels, cv::Mat(), // do not use mask
                  hist, 1, hist_sizes, hist_ranges, true, // the histogram is uniform
                  false );
                
    float hist_acc = 0, hist_thresh = 0.90*float(g_img.total());
    int threas_bin;
    for( threas_bin = 0; threas_bin < hist_bins; threas_bin++ )
    {
      hist_acc += hist.at<float>(threas_bin, 0);
      if(hist_acc >= hist_thresh)
        break;
    }
    if(threas_bin < 2) threas_bin = 2;
    float sat_exp = 1.0;
    do{ sat_exp += 0.01; } while(pow(float(threas_bin),sat_exp) < 255.0);
    if(sat_exp > 2.0) sat_exp = 2.0;
    //std::cout<<sat_exp<<std::endl;
    cv::pow(g_img,sat_exp, g_img_sat);
  
    cv::threshold(g_img_sat, g_img_sat, 255.0, 0.0, cv::THRESH_TRUNC );
    
    if( smooth_std > 0.0 )
      cv::GaussianBlur(g_img_sat, g_img_sat,cv::Size(0,0),smooth_std);    
    
    cv::normalize(g_img_sat, g_img_sat, 0, 1.0, cv::NORM_MINMAX);
    g_img_sat = 1.0 - g_img_sat;
    
    scaled_img.img = g_img_sat;
    scaled_img.scale = img_stats_ptr->getPyrScale(i);
    
    g_mag_pyr_ptr->push_back( scaled_img );
//     cv_ext::showImage(in_g_img, "input");
//     cv_ext::showImage(g_img_sat, "sat");
  }
    
//   for ( int i = 0 ; i < g_mag_pyr_ptr->size(); i++ )
//   {
//     std::cout<<g_mag_pyr_ptr->at(i).scale<<std::endl;
//     cv_ext::showImage(g_mag_pyr_ptr->at(i).img, "sat");
//   }
}


class DirectMatching::Optimizer
{
public:
  Optimizer() {};
  ~Optimizer(){};
  ceres::Problem problem;
};
 
struct DirectResidual
{
  DirectResidual ( const cv_ext::PinholeCameraModel &cam_model, const cv::Mat &residual_map,
                    const cv::Point3f &model_pt ) :
    cam_model_ ( cam_model ),
    residual_map_ ( residual_map )
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

    //residuals[0] = cv_ext::bilinearInterp<float, _T> ( dist_map_, x, y );
    residuals[0] = cv_ext::getPix<float, _T> ( residual_map_, proj_pt[0], proj_pt[1] );
    return true;
  }

  static ceres::CostFunction* Create ( const cv_ext::PinholeCameraModel &cam_model,
                                       const cv::Mat &res_map,
                                       const cv::Point3f &model_pt )
  {
    return ( new ceres::AutoDiffCostFunction<DirectResidual, 1, 7 > (
               new DirectResidual ( cam_model, res_map, model_pt ) ) );
  }

  const cv_ext::PinholeCameraModel &cam_model_;
  const cv::Mat &residual_map_;
  double model_pt_[3];
};

DirectMatching::DirectMatching () :
  TemplateMatching ()
{
}

void DirectMatching::setInput ( const ScaledImagesListPtr &imgs_list_ptr )
{
  imgs_list_ptr_ = imgs_list_ptr;
  for( int i = 0; i < int(imgs_list_ptr_->size()); i++)
  {
    scaled_cam_models_.push_back(cam_model_);
    scaled_cam_models_.back().setSizeScaleFactor(imgs_list_ptr->at(i).scale);
  }
}

void DirectMatching::match(int num_best_matches, std::vector< TemplateMatch >& matches, int image_step)
{
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
}

void DirectMatching::updateOptimizer( int idx )
{
  const std::vector<cv::Point3f> &model_pts = ( idx < 0 )?model_ptr_->getPoints():model_ptr_->getPrecomputedPoints(idx);
  
  optimizer_pyr_ptr_.clear();
  
  for( int i = 0; i < int(imgs_list_ptr_->size()); i++)
  {
    optimizer_pyr_ptr_.push_back( std::shared_ptr< Optimizer > ( new Optimizer () ) );
    Optimizer &optimizer = *(optimizer_pyr_ptr_.back());
    for ( int j = 0; j < int(model_pts.size()); ++j )
    {
      ceres::CostFunction* cost_function =
        DirectResidual::Create ( scaled_cam_models_[i], imgs_list_ptr_->at(i).img, model_pts[j] );

      optimizer.problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
    }
  }
}

void DirectMatching::setupExhaustiveMatching()
{
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
//   exit(-1);
}

// TODO
double DirectMatching::avgDistance( int idx )
{
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
  exit(-1);
}

double DirectMatching::optimize()
{
  if( optimizer_pyr_ptr_.empty() )
    return - 1;
  
  ceres::Solver::Options options;
  //options.linear_solver_type = ceres::DENSE_SCHUR;
  options.max_num_iterations = optimizer_num_iterations_;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = verbouse_mode_;

  double final_cost = 0;
  for( int i = int(imgs_list_ptr_->size()) - 1; i >= 0; i-- )
  {
    ceres::Solver::Summary summary;
    ceres::Solve ( options, &(optimizer_pyr_ptr_[i]->problem), &summary );  
    if( !i )
      final_cost = summary.final_cost;
  }
  
  return final_cost;
}*/
