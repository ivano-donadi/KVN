/*
 * d2co - Direct Directional Chamfer Optimization
 *
 *  Copyright (c) 2020, Alberto Pretto <alberto.pretto@flexsight.eu>
 *                      Marco Imperoli <marco.imperoli@flexsight.eu>
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <stdexcept>
#include <opencv2/opencv.hpp>

#include "cv_ext/cv_ext.h"

class PatchMatchFlow
{
public:
  PatchMatchFlow( const cv_ext::PinholeCameraModel &cam_model ) :
   cam_model_(cam_model),
   aperture_(35),
   search_radius_(15),
   pyr_num_levels_(3),
   pyr_scale_factor_(2.0),
   g_filter_sigma_(6.0),
   max_searched_pixel_(50),
   dist_tolerance_(3.0)
//    ,
//    merge_tolerance_(1.0)
  {
    h_aperture_ = aperture_/2;
    dist_tolerance2_ = dist_tolerance_*dist_tolerance_;
  };
   
  ~PatchMatchFlow() {};
 
  int patchAperture() const { return aperture_; };
  int search_radius() const { return search_radius_; };
  int pyrNumLevels() const { return pyr_num_levels_; };
  double pyrScaleFactor() const { return pyr_scale_factor_; };
  double gaussianFilterSigma() const { return g_filter_sigma_; };
  int maxSearchedPixels() const { return max_searched_pixel_; };
  double distanceTolerance() const { return dist_tolerance_; };
  
  void setPatchAperture( int aperture )
  { 
    if( aperture < 3 ) aperture_ = 3;
    else if( aperture%2 == 0 ) aperture_ = aperture + 1;
    h_aperture_ = aperture_/2;
  };
  
  void setSearch_radius( int search_radius )
  { 
    search_radius_ = search_radius; 
    if( search_radius_ < 1 ) search_radius_ = 1;
  };
  
  void setPyrNumLevels( int pyr_levels )
  { 
    pyr_num_levels_ = pyr_levels; 
    if(pyr_num_levels_ < 1 ) pyr_num_levels_ = 1;
  };
  
  void setPyrScaleFactor( double pyr_scale_factor )
  { 
    pyr_scale_factor_ = pyr_scale_factor; 
    if( pyr_scale_factor_ < 1.0 ) pyr_scale_factor_ = 1.0;
  };
  
  void setGaussianFilterSigma( double sigma )
  {
    g_filter_sigma_ = sigma;
    
    if( g_filter_sigma_ < 0) g_filter_sigma_ = 1.0;
  }
  
  void setMaxSearchedPixels( int val ) { max_searched_pixel_ = val; };
  
  void setDistanceTolerance( double val )
  { 
    dist_tolerance_ = val; 
    if( dist_tolerance_ < 0)
      dist_tolerance_ = -dist_tolerance_;
    dist_tolerance2_ = dist_tolerance_*dist_tolerance_;
  };
    
  template< typename _T > void compute( const cv::Mat &src_ref, const cv::Mat &src_cur, cv::Mat &dst_flow );
  
private:
  
  cv_ext::PinholeCameraModel cam_model_;
  
  int aperture_, h_aperture_;
  int search_radius_;
  int pyr_num_levels_;
  double pyr_scale_factor_;
  double g_filter_sigma_;
  int max_searched_pixel_;
  double dist_tolerance_, dist_tolerance2_;
  
  std::vector< cv_ext::PinholeCameraModel > cam_models_pyr_;
  cv_ext::ImagePyramid ref_pyr_, cur_pyr_;
  
  std::vector<double> g_filter_;
  
  cv::Mat index_mat_;
  std::vector< std::multimap< double,cv::Point2f > > costs_maps_;
  
  void preparePyramids( const cv::Mat &src_ref, const cv::Mat &src_cur );
  template< typename _T > void directMatch( int i );
  void createFilter();
  void extractFlow( int i, cv::Mat& flow );
  
  template< typename _T > inline double pixDist( const _T &val1, const _T &val2 );
  template< typename _T > inline double computeCost( const cv::Mat &bimg0, const cv::Mat &bimg1, 
                                                     cv::Point &p1,  cv::Point &p2 );
  template< typename _T > inline void computeCosts( const cv::Mat &bimg0, const cv::Mat &bimg1, cv::Point &p, 
                                                    std::multimap< double,cv::Point2f > &costs_map );
  template< typename _T > inline void computeRandomizedCosts( const cv::Mat &bimg0, const cv::Mat &bimg1, cv::Point &p, 
                                                              std::multimap< double,cv::Point2f > &costs_map );
  
  void getSearchLimits( const cv::Point &p, const int &width, const int &height, 
                        cv::Point &min, cv::Point &max );
};


/* Implementation */

template< typename _T > void PatchMatchFlow :: compute( const cv::Mat &src_ref, 
                                                        const cv::Mat &src_cur, 
                                                        cv::Mat &dst_flow )
{
  if( src_ref.type() != cv::DataType<_T>::type || src_cur.type() != cv::DataType<_T>::type  ||
     !src_ref.rows || !src_ref.cols || src_ref.rows != src_cur.rows || src_ref.cols != src_cur.cols )
    throw std::invalid_argument("Input images should have same size (=/= 0) \
                                 and same type (the one in the template paramater)");

  cv_ext:: stimer;
  stimer.reset();
  
  preparePyramids( src_ref, src_cur );
  
  // WARNING Debug code
  int dbg_scale = pyr_num_levels_ - 1; 
  directMatch< _T>( dbg_scale );
  extractFlow( dbg_scale, dst_flow );      
  
  std::cout<<"Patch Flow time : "<<stimer.elapsedTimeMs()<<std::endl;
}

void PatchMatchFlow :: preparePyramids( const cv::Mat &src_ref, const cv::Mat &src_cur )
{
  ref_pyr_.setImage(src_ref, pyr_num_levels_, src_ref.type(), true, pyr_scale_factor_ );
  cur_pyr_.setImage(src_cur, pyr_num_levels_, src_cur.type(), true, pyr_scale_factor_ );
  cam_models_pyr_.resize(pyr_num_levels_, cam_model_ );
  
  for( int i = 0; i < pyr_num_levels_; i++ )
    cam_models_pyr_[i].setSizeScaleFactor( ref_pyr_.getScale(i) );
}

template< typename _T > void PatchMatchFlow :: directMatch( int i )
{ 
  const cv::Mat &ref_img = ref_pyr_.getImageAt( i ), &cur_img = cur_pyr_.getImageAt( i );
  
  int width = ref_img.cols, height = ref_img.rows;
  index_mat_ = cv::Mat( height, width, cv::DataType<int>::type, cv::Scalar(-1));
  
  // Constructs a larger image to fit both the image and the border
  cv::Mat bref_img( height + h_aperture_*2, width + h_aperture_*2, cv::DataType<_T>::type ),
          bcur_img( height + h_aperture_*2, width + h_aperture_*2, cv::DataType<_T>::type );
      
  // Copy with border 
  cv::copyMakeBorder(ref_img, bref_img, h_aperture_, h_aperture_, 
                     h_aperture_, h_aperture_, cv::BORDER_REPLICATE);
  cv::copyMakeBorder(cur_img, bcur_img, h_aperture_, h_aperture_, 
                     h_aperture_, h_aperture_, cv::BORDER_REPLICATE);

  createFilter();
  
  int index = 0;
  costs_maps_.resize( cur_img.total() );
  
  int search_area_total = 2*search_radius_ + 1;
  search_area_total *= search_area_total;
  
  if( max_searched_pixel_ <= 0 || search_area_total <= max_searched_pixel_)
  {
    // Complete search (i.e., consider all pixel in the search area)
    for( int y = 0; y < height; y++)
    {
      std::cout<<y<<" ";
      cv_ext:: stimer;
      stimer.reset();
      int *imat_p = index_mat_.ptr< int >(y);
      for( int x = 0; x < width; x++, imat_p++)
      {
        cv::Point p1( x, y );
        *imat_p = index;
        computeCosts<_T>( bref_img, bcur_img, p1, costs_maps_[index++]);
      }
      std::cout<<stimer.elapsedTimeMs()<<std::endl;
    }
  }
  else
  {
    // Randomized search (consider only a subset of max_searched_pixel_ random pixel)
    for( int y = 0; y < height; y++)
    {
      std::cout<<y<<" ";
      cv_ext:: stimer;
      stimer.reset();
      int *imat_p = index_mat_.ptr< int >(y);
      for( int x = 0; x < width; x++, imat_p++)
      {
        cv::Point p1( x, y );
        *imat_p = index;
         computeRandomizedCosts<_T>( bref_img, bcur_img, p1, costs_maps_[index++]);
      }
      std::cout<<stimer.elapsedTimeMs()<<std::endl;
    }
  }
  
  costs_maps_.resize( index );
}

void PatchMatchFlow :: createFilter()
{  
  std::vector<double> filt;
  filt.resize(aperture_);
  
  g_filter_.resize( aperture_*aperture_ );
  double s = 0.;
  for (int x = -h_aperture_, i = 0; x <= h_aperture_; x++, i++)
  {
    filt[i] = (float)exp(-(double(x)*double(x))/(2*g_filter_sigma_*g_filter_sigma_));
    s += filt[i];
  }
  
  s = 1./s;
  for (int i = 0; i < aperture_; i++)
  {
    filt[i] *= s;
  }

  s = 0;
  for (int i = 0; i < aperture_; i++)
  {
    for (int j = 0; j < aperture_; j++)
    {
      g_filter_[aperture_*j + i] = filt[i]*filt[j];
      s += g_filter_[aperture_*j + i];
    }
  }
  
  s = 1./s;
  for (int i = 0; i < aperture_*aperture_; i++)
  {
    g_filter_[i] *= s;
  }
  
//   cv::Mat test_filter(cv::Size(aperture_, aperture_), cv::DataType<double>::type, g_filter_.data(), aperture_*sizeof(double));
//   cv_ext::showDebugImage(test_filter, "test_filter");
}

template< typename _T > inline double PatchMatchFlow:: pixDist( const _T &val0, const _T &val1 )
{
  double diff = double(val0) - double(val1);
  return diff*diff;
}

// template< typename _Tp, int m, int n > static 
//   inline double patchMatchDist( const cv::Matx<_Tp, m, n> &val1, const cv::Matx<_Tp, m, n> &val2 )
// {
//   cv::Matx<double, m, n> diff = val1 - val2;
//   return cv::normL2Sqr<double, double>(diff.val, m*n);
// }



template< typename _T > inline 
  double PatchMatchFlow::computeCost( const cv::Mat &bimg0, const cv::Mat &bimg1, 
                                      cv::Point &p0,  cv::Point &p1 )
{
  double cost = 0;
  for( int y = 0; y < aperture_; y++)
  {
    const _T *img0_p = bimg0.ptr<_T>(p0.y - h_aperture_),
             *img1_p = bimg1.ptr<_T>(p1.y - h_aperture_);
    
    img0_p += (p0.x - h_aperture_);
    img1_p += (p1.x - h_aperture_);
    int filt_idx = aperture_*y;
    for( int x = 0; x < aperture_; x++, img0_p++, img1_p++, filt_idx++ )
      cost += g_filter_[filt_idx]*pixDist(*img0_p, *img1_p);
  }
  
  return cost;
}

template< typename _T > inline 
  void PatchMatchFlow:: computeCosts( const cv::Mat &bimg0, const cv::Mat &bimg1, cv::Point &p, 
                                      std::multimap< double,cv::Point2f > &costs_map )
{
  costs_map.clear();
  cv::Point min, max;
  getSearchLimits( p, bimg0.cols - 2*h_aperture_, bimg0.rows - 2*h_aperture_, min, max );
  cv::Point p0 = cv::Point(p.x + h_aperture_, p.y + h_aperture_ );
  
//   if(p.x == 134 && p.y == 9)
//   {
//     cv::Mat dbg_img0 = bimg0.clone();
//     cv::rectangle(dbg_img0, cv::Point(p0.x - h_aperture_, p0.y - h_aperture_), 
//                   cv::Point(p0.x + h_aperture_, p0.y + h_aperture_), cv::Scalar(255, 255, 255));
//     cv_ext::showDebugImage(dbg_img0,"cur");
//   }
    
//   double min_val = 10000000000000000;      
//   cv::Point min_p;
  _T val0 = bimg0.at<_T>(p0.y, p0.x);
  
  for( int y = min.y; y <= max.y; y++)
  {
    const _T *val1_p = bimg1.ptr< _T >(y + h_aperture_);
    val1_p += (min.x + h_aperture_);
    for( int x = min.x; x <= max.x; x++, val1_p++ )
    { 

      //std::cout<<p2;
      if( pixDist( val0, *val1_p ) <= dist_tolerance2_)
      {
        cv::Point p1 = cv::Point(x + h_aperture_, y + h_aperture_ );
        double cost = computeCost<_T>( bimg0, bimg1, p0, p1 );
        costs_map.insert( std::pair< double, cv::Point2f >(cost, cv::Point2f(x, y)) );
      }
      
//       if(p.x == 134 && p.y == 9)
//       {
//         if(cost <  min_val )
//         {
//           min_val = cost;
//           min_p = p1;
//         }
//         std::cout<<cost<<" "<<min_val<<" "<<std::endl;;
//         cv::Mat dbg_img1 = bimg1.clone();
//         
//         cv::line(dbg_img1, p0, p0, cv::Scalar(0, 0, 0), 4 );
//         cv::line(dbg_img1, p1, p1, cv::Scalar(255, 255, 255), 4 );
//         cv::rectangle(dbg_img1, cv::Point(p1.x - h_aperture_, p1.y - h_aperture_), 
//                       cv::Point(p1.x + h_aperture_, p1.y + h_aperture_), cv::Scalar(255, 255, 255));
//         cv::rectangle(dbg_img1, cv::Point(min_p.x - h_aperture_, min_p.y - h_aperture_), 
//               cv::Point(min_p.x + h_aperture_, min_p.y + h_aperture_), cv::Scalar(0,0,0));
//         
//         cv_ext::showDebugImage(dbg_img1,"ref");   
//       }
      
      
    }
  }
  
//   cv_ext::showDebugImage(bimg1, "bimg1");
}

template< typename _T > inline 
void PatchMatchFlow::computeRandomizedCosts ( const cv::Mat& bimg0, const cv::Mat& bimg1, cv::Point& p, 
                                              std::multimap< double, cv::Point2f >& costs_map )
{
  costs_map.clear();
  cv::Point min, max;
  getSearchLimits( p, bimg0.cols - 2*h_aperture_, bimg0.rows - 2*h_aperture_, min, max );
  cv::Point p0 = cv::Point(p.x + h_aperture_, p.y + h_aperture_ );
  
//   cv::Mat dbg_img0 = bimg0.clone();
//   cv::Mat dbg_img1 = bimg1.clone();
//   cv::rectangle(dbg_img0, cv::Point(p0.x - h_aperture_, p0.y - h_aperture_), 
//                 cv::Point(p0.x + h_aperture_, p0.y + h_aperture_), cv::Scalar(255, 255, 255));
//   cv_ext::showDebugImage(dbg_img0,"cur", true, 10);
//   cv::line(dbg_img1, p0, p0, cv::Scalar(0, 0, 0), 4 );
  
  int interval_x = max.x - min.x, interval_y = max.y - min.y;
  _T val0 = bimg0.at<_T>(p0.y, p0.x);
  for(int i = 0; i < max_searched_pixel_; i++)
  {
    int x = min.x + rand()%interval_x, y = min.y + rand()%interval_y;
    cv::Point p1 = cv::Point(x + h_aperture_, y + h_aperture_ );
    
//     cv::line(dbg_img1, p1, p1, cv::Scalar(255, 255, 255), 1 );
    
    _T val1 = bimg1.at<_T>(p1.y, p1.x);
    //std::cout<<p2;
    if( pixDist( val0, val1 ) <= dist_tolerance2_)
    {
      double cost = computeCost<_T>( bimg0, bimg1, p0, p1 );
      costs_map.insert( std::pair< double, cv::Point2f >(cost, cv::Point2f(x, y)) );
    }
  }
  
//   cv_ext::showDebugImage(dbg_img1,"ref", true, 10);   
}

void PatchMatchFlow::extractFlow( int i, cv::Mat &flow )
{
  flow = cv::Mat(ref_pyr_.getImageAt( i ).size(), cv::DataType<cv::Point2f>::type);
  int width = ref_pyr_.getImageAt( i ).cols, height = ref_pyr_.getImageAt( i ).rows;
  for( int y = 0; y < height; y++)
  {
    int *imat_p = index_mat_.ptr< int >(y);
    cv::Point2f *flow_p = flow.ptr< cv::Point2f >(y);
    
    for( int x = 0; x < width; x++, imat_p++, flow_p++)
    {
//         cv::Mat dbg_img1 = scaled_ref_imgs[i].clone();
//         cv::Mat dbg_img2 = scaled_cur_imgs[i].clone();
      
      cv::Point2f p(x,y);
      std::multimap< double,cv::Point2f > &cost_map = costs_maps_.at(*imat_p);
      *flow_p = cost_map.begin()->second - p;
    }
  }
}

void PatchMatchFlow::getSearchLimits( const cv::Point &p, const int &width, const int &height, 
                                      cv::Point &min, cv::Point &max )
{
  min.x = std::max(0, p.x - search_radius_ );
  min.y = std::max(0, p.y - search_radius_ );
  max.x = std::min( width - 1, p.x + search_radius_ );
  max.y = std::min( height - 1, p.y + search_radius_ );
}
