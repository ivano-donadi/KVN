#include <stdexcept>
#include <algorithm>
#include <limits>
#include <boost/concept_check.hpp>

#include <omp.h>

#include "cv_ext/types.h"
#include "cv_ext/image_pyramid.h"
#include "cv_ext/hdr.h"

using namespace cv_ext;

template <typename _TPixDepth> HDR<_TPixDepth>::HDR() :
  parallelism_enabled_ ( false )
{
  pix_levels_ = std::numeric_limits<_TPixDepth>::max() + 1;
}

template <typename _TPixDepth> void HDR<_TPixDepth>::computeTriangleWeights( std::vector<double> &weights ) const
{
  weights.resize ( pix_levels_ );
  int h = pix_levels_ / 2;

  double s = 0.0;

  for ( int i = 0; i < h; i++ )
    s += ( weights[i] = double ( i + 1 ) );

  for ( int i = h; i < pix_levels_; i++ )
    s += ( weights[i] = double ( pix_levels_ - i ) );

  s = 1.0/s;

  for ( int i = 0; i < pix_levels_; i++ )
    weights[i] *= s;
}

template <typename _TPixDepth> void HDR<_TPixDepth>::computeGaussianWeights( std::vector<double> &weights, 
                                                             double sigma ) const
{
  weights.resize ( pix_levels_ );

  int h = pix_levels_/2;
  
  double s = 0.;
  for ( int x = -h, i = 0; i < pix_levels_; x++, i++ )
    s += ( weights[i] = exp ( -double ( x*x ) / ( 2.0*sigma*sigma ) ) );

  s = 1./s;
  for ( int i = 0; i < pix_levels_; i++ )
    weights[i] *= s;
}

template <typename _TPixDepth> void HDR<_TPixDepth>::computeTriangleWeights( std::vector<double> &gl_weights, 
                                                             std::vector<cv::Vec3d> &bgr_weights ) const
{
  computeTriangleWeights( gl_weights );
  bgr_weights.resize ( pix_levels_ );
  
  for ( int i = 0; i < pix_levels_; i++ )
    bgr_weights[i] = cv::Vec3f::all(gl_weights[i]);
}

template <typename _TPixDepth> void HDR<_TPixDepth>::computeGaussianWeights( std::vector<double> &gl_weights, 
                                                             std::vector<cv::Vec3d> &bgr_weights, 
                                                             double sigma ) const
{
  computeGaussianWeights(gl_weights, sigma );
  bgr_weights.resize ( pix_levels_ );

  for ( int i = 0; i < pix_levels_; i++ )
    bgr_weights[i] = cv::Vec3f::all(gl_weights[i]);
}

template <typename _TPixDepth> 
  void HDR<_TPixDepth>::computeLinearResponse( std::vector<double> &gl_responses, 
                                       std::vector<cv::Vec3d> &bgr_responses ) const
{
  gl_responses.resize ( pix_levels_ );
  bgr_responses.resize ( pix_levels_ );
  
  for ( int i = 0; i < pix_levels_; i++ )
  {
    gl_responses[i] = double ( i + 1 );
    bgr_responses[i] = cv::Vec3f::all(gl_responses[i]);
  }
}


template <typename _TPixDepth>
void HDR<_TPixDepth>::checkData ( const std::vector<cv::Mat> &imgs, const std::vector<float> &exp_times ) const
{
  if ( imgs.size() < 3 || imgs.size() != exp_times.size() )
    throw std::invalid_argument ( "Input images vector and exposure times vector should have same size >= 3" );

  if ( ( imgs[0].channels() != 1 && imgs[0].channels() != 3 ) || imgs[0].depth() != cv::DataType<_TPixDepth>::depth )
    throw std::invalid_argument ( "Unsupported image: only grey level or RGB images are supported and with same \
                                   type as the one in the template paramater (uchar or ushort)" );

  for ( int i = 0 ; i < int(imgs.size()); i++ )
  {
    if ( imgs[i].size() != imgs[0].size() || imgs[i].type() != imgs[0].type() || exp_times[i] <= 0 )
      throw std::invalid_argument ( "Input images should have the same type an size and the exposure \
                                     times should be greater than zero" );
  }
}

template <typename _TPixDepth>
void HDR<_TPixDepth>::checkData ( const std::vector<cv::Mat> &imgs ) const
{
  if ( imgs.size() < 3 )
    throw std::invalid_argument ( "Input images vector should have size >= 3" );

  if ( ( imgs[0].channels() != 1 && imgs[0].channels() != 3 ) || imgs[0].depth() != cv::DataType<_TPixDepth>::depth )
    throw std::invalid_argument ( "Unsupported image: only grey level or RGB images are supported and with same \
                                   type as the one in the template paramater (uchar or ushort)" );

  for ( int i = 0 ; i < int(imgs.size()) ; i++ )
  {
    if ( imgs[i].size() != imgs[0].size() || imgs[i].type() != imgs[0].type())
      throw std::invalid_argument ( "Input images should have the same type an size" );
  }
}


template <typename _TPixDepth> DebevecHDR<_TPixDepth>::DebevecHDR() 
{
  this->computeTriangleWeights( gl_weights_, bgr_weights_ );
  this->computeLinearResponse( gl_responses_, bgr_responses_ );
  computeLogResponses();
}

template <typename _TPixDepth> void DebevecHDR<_TPixDepth>::setResponses ( const std::vector<double> &responses )
{
  if ( int(responses.size()) != this->pix_levels_ || int(gl_weights_.size()) != this->pix_levels_ )
    throw std::invalid_argument ( "Wrong responses size (it must be equal to \
                                  the number of possible pixel values)" );

  gl_responses_ = responses;
  bgr_responses_.resize ( this->pix_levels_ );
  
  for ( int i = 0; i < this->pix_levels_; i++ )
    bgr_responses_[i] = cv::Vec3f::all(gl_responses_[i]);
}

template <typename _TPixDepth> void DebevecHDR<_TPixDepth>::setWeights ( const std::vector<double> &weights )
{
  if ( int(weights.size()) != this->pix_levels_ )
    throw std::invalid_argument ( "Wrong weights size (it must be equal to \
                                   the number of possible pixel values)" );

  gl_weights_ = weights;
  bgr_weights_.resize ( this->pix_levels_ );
  
  for ( int i = 0; i < this->pix_levels_; i++ )
    bgr_weights_[i] = cv::Vec3f::all(gl_weights_[i]);
}

template <typename _TPixDepth> void DebevecHDR<_TPixDepth>::setResponses ( const std::vector<cv::Vec3d> &responses )
{
  if ( int(responses.size()) != this->pix_levels_ || int(gl_weights_.size()) != this->pix_levels_ )
    throw std::invalid_argument ( "Wrong responses size (it must be equal to \
                                  the number of possible pixel values)" );

  bgr_responses_ = responses;
  gl_responses_.resize ( this->pix_levels_ );
  
  for ( int i = 0; i < this->pix_levels_; i++ )
    gl_responses_[i] = (bgr_responses_[i](0) + bgr_responses_[i](1) + bgr_responses_[i](2))/3.0;
}

template <typename _TPixDepth> void DebevecHDR<_TPixDepth>::setWeights ( const std::vector<cv::Vec3d> &weights )
{
  if ( int(weights.size()) != this->pix_levels_ )
    throw std::invalid_argument ( "Wrong weights size (it must be equal to \
                                   the number of possible pixel values)" );

  bgr_weights_ = weights;
  gl_weights_.resize ( this->pix_levels_ );
  
  for ( int i = 0; i < this->pix_levels_; i++ )
    gl_weights_[i] = (bgr_weights_[i](0) + bgr_weights_[i](1) + bgr_weights_[i](2))/3.0;
}

template <typename _TPixDepth> void DebevecHDR<_TPixDepth>::setWeights ( WeightFunctionType type )
{

  switch ( type )
  {
  case cv_ext::GAUSSIAN_WEIGHT:
    this->computeGaussianWeights( gl_weights_, bgr_weights_, double( this->pix_levels_ / 2 ) );
    break;
  case cv_ext::TRIANGLE_WEIGHT:
    this->computeTriangleWeights( gl_weights_, bgr_weights_ );
    break;
  default:
    this->computeTriangleWeights( gl_weights_, bgr_weights_ );
    break;
  }
}

template <typename _TPixDepth> void DebevecHDR<_TPixDepth>::computeLogResponses()
{
  ln_gl_responses_.resize ( this->pix_levels_ );
  ln_bgr_responses_.resize ( this->pix_levels_ );
  
  for ( int i = 0; i < this->pix_levels_; i++ )
  {
    ln_gl_responses_[i] = log ( gl_responses_[i] );
    ln_bgr_responses_[i] = cv::Vec3f::all(ln_gl_responses_[i]);
  }
}

template <typename _TPixDepth>
  void DebevecHDR<_TPixDepth>::merge ( const std::vector<cv::Mat> &imgs, cv::Mat &hdr_img ) const
{
  std::vector<float> exp_times( imgs.size() );
  for( int i = 0; i < int(imgs.size()); i++)
    exp_times[i] = 1.0;
  this->merge( imgs, exp_times, hdr_img );
}

template <typename _TPixDepth>
  void DebevecHDR<_TPixDepth>::merge ( const std::vector<cv::Mat> &imgs, const std::vector<float> &exp_times,
                               cv::Mat &hdr_img ) const
{
  this->checkData ( imgs, exp_times );

  const int n_images = imgs.size();
  const int width = imgs[0].cols, height  = imgs[0].rows;
  const int n_channels = imgs[0].channels();

  // Compute natural logarithm of exposure times
  std::vector < double > ln_exp_times ( exp_times.size() );
  for ( int i = 0; i < n_images; i++ )
    ln_exp_times[i] = log ( exp_times[i] );

  if( n_channels == 1 )
  {
    hdr_img = cv::Mat ( cv::Size ( width, height ), cv::DataType<float>::type );
  
    #pragma omp parallel for if( this->parallelism_enabled_ )
    for ( int y = 0; y < height; y++ )
    {
      float *hdr_p = hdr_img.ptr<float> ( y );
      std::vector< const _TPixDepth *> imgs_p ( n_images );

      for ( int i = 0; i < n_images; i++ )
        imgs_p[i] = imgs[i].ptr<_TPixDepth> ( y );

      for ( int x = 0; x < width; x++, hdr_p++ )
      {
        double hdr_val = 0, eta = 0.0;

        // For each exposure
        for ( int i = 0; i < n_images; i++ )
        {
          double ln_exp_time = ln_exp_times[i];
          _TPixDepth ldr_val = *imgs_p[i]++;

          hdr_val += gl_weights_[ldr_val] * ( ln_gl_responses_[ldr_val] - ln_exp_time );
          eta += gl_weights_[ldr_val];
        }

        if ( eta )
          hdr_val *= ( 1.0/eta );
        else
          hdr_val = 0;

        *hdr_p = float ( exp ( hdr_val ) );
      }
    }
  }
  else
  {
    hdr_img = cv::Mat ( cv::Size ( width, height ), cv::DataType<cv::Vec3f>::type );
    
    #pragma omp parallel for if( this->parallelism_enabled_ )
    for ( int y = 0; y < height; y++ )
    {
      cv::Vec3f *hdr_p = hdr_img.ptr< cv::Vec3f > ( y );
      std::vector< const cv::Vec<_TPixDepth, 3>* > imgs_p ( n_images );

      for ( int i = 0; i < n_images; i++ )
        imgs_p[i] = imgs[i].ptr< cv::Vec<_TPixDepth, 3> > ( y );

      for ( int x = 0; x < width; x++, hdr_p++ )
      {
        double hdr_val0  = 0, hdr_val1  = 0, hdr_val2  = 0, 
               eta = 0, weight;

        // For each exposure
        for ( int i = 0; i < n_images; i++ )
        {
          double ln_exp_time = ln_exp_times[i];
          cv::Vec<_TPixDepth, 3> ldr_val = *imgs_p[i]++;

          weight = (bgr_weights_[ldr_val(0)](0) + 
                    bgr_weights_[ldr_val(1)](1) + 
                    bgr_weights_[ldr_val(2)](2) ) / 3.0;
                    
          hdr_val0 += weight * ( ln_bgr_responses_[ldr_val(0)](0) - ln_exp_time );
          hdr_val1 += weight * ( ln_bgr_responses_[ldr_val(1)](1) - ln_exp_time );
          hdr_val2 += weight * ( ln_bgr_responses_[ldr_val(2)](2) - ln_exp_time );
          
          eta +=weight;
        }

        if ( eta )
        {
          eta = 1.0/eta;
          hdr_val0 *= eta;
          hdr_val1 *= eta;
          hdr_val2 *= eta;
        }
        else
          hdr_val0 = hdr_val1 = hdr_val2 = 0;

        *hdr_p = cv::Vec3f( float( exp ( hdr_val0 ) ), float( exp ( hdr_val1 ) ), float( exp ( hdr_val2 ) ) );
      }
    }
  }
}


template <typename _TPixDepth>
  MertensHDR<_TPixDepth>::MertensHDR() :
    contrast_exp_(1.0), 
    saturation_exp_(1.0), 
    exposedness_exp_(1.0)
{
  exposedness_sigma_ = this->pix_levels_*0.2;
  this->computeGaussianWeights( exposedness_weights_, exposedness_sigma_ );
}

template <typename _TPixDepth>
  void MertensHDR<_TPixDepth>::setExposednessSigma( double sigma )
{
  exposedness_sigma_ = sigma;
  this->computeGaussianWeights( exposedness_weights_, exposedness_sigma_ );
}
  
template <typename _TPixDepth>
  void MertensHDR<_TPixDepth>::merge ( const std::vector<cv::Mat> &imgs, cv::Mat &fusion ) const
{
  this->checkData ( imgs );

  const int n_images = imgs.size();
  const int width = imgs[0].cols, height  = imgs[0].rows;
  const int n_channels = imgs[0].channels();
  
  std::vector<cv::Mat> weights( n_images );
  cv::Mat weights_sum = cv::Mat::zeros(cv::Size ( width, height ), cv::DataType<float>::type );
  
  for ( int i = 0; i < n_images; i++ )
    weights[i] = cv::Mat ( cv::Size ( width, height ), cv::DataType<float>::type );
  
  int pyr_max_level = int(floor(log2((width < height)?width:height))) - 1;
  std::vector<cv::Mat> res_pyr(pyr_max_level + 1);
      
  if( n_channels == 1 )
  {
    #pragma omp parallel for if( this->parallelism_enabled_ )
    for ( int y = 1; y < height - 1; y++ )
    {
      /* Using simple laplacian filter kernel:
      * 
      * 0  1  0
      * 1 -4  1
      * 0  1  0
      * 
      * where:
      * 
      * ----------   up_pix_p    ----------
      * left_pix_p   cur_pix_p   right_pix_p
      * ----------   down_pix_p  ----------
      */
      
      std::vector< const _TPixDepth* > cur_pix_p( n_images ), up_pix_p( n_images ), 
                               down_pix_p( n_images ), left_pix_p( n_images ),
                               right_pix_p( n_images );
                               
      std::vector< float* > w_p( n_images );
      float *w_sum_p;
      
      for ( int i = 0; i < n_images; i++ )
      {
        right_pix_p[i] = imgs[i].ptr< _TPixDepth > ( y );
        left_pix_p[i] = right_pix_p[i]++;
        cur_pix_p[i] = right_pix_p[i]++;
        up_pix_p[i] = imgs[i].ptr< _TPixDepth > ( y - 1 );
        down_pix_p[i] = imgs[i].ptr< _TPixDepth > ( y + 1 );
        up_pix_p[i]++;
        down_pix_p[i]++;
        
        w_p[i] = weights[i].ptr< float > ( y );
        w_p[i]++;
      }
      
      w_sum_p = weights_sum.ptr< float > ( y );
      w_sum_p++;
      
      for ( int x = 1; x < width - 1; x++, w_sum_p++ )
      {
        // For each exposure
        for ( int i = 0; i < n_images; i++ )
        { 
          const _TPixDepth &cur_gl_val = *cur_pix_p[i]++,
                   &left_gl_val = *left_pix_p[i]++,
                   &right_gl_val = *right_pix_p[i]++,
                   &up_gl_val = *up_pix_p[i]++,
                   &down_gl_val = *down_pix_p[i]++;

          double contrast = fabs( double(left_gl_val) + double(right_gl_val) + 
                                  double(up_gl_val) + double(down_gl_val) 
                                  - 4.0 * double(cur_gl_val) );
          double exposedness = exposedness_weights_[cur_gl_val];
          
          contrast = pow(contrast, contrast_exp_);
          exposedness = pow(exposedness, exposedness_exp_);
          
          if( contrast < DBL_EPSILON ) contrast = DBL_EPSILON; 
          if( exposedness < DBL_EPSILON ) exposedness = DBL_EPSILON; 
          
          *w_p[i] = contrast * exposedness;
          *w_sum_p += *w_p[i]++;
        }
      }
    }
    
    // Complete the first and last rows and cols for the weights matrix
    fillWeightsBorders( weights, weights_sum );    

    #pragma omp parallel for if( this->parallelism_enabled_ )
    for ( int i = 0; i < n_images; i++ )
    {
      weights[i] /= weights_sum;
        
      // For each exposure, generate a pyramid for both images and weights
      cv_ext::ImagePyramid img_pyr( imgs[i], pyr_max_level + 1, cv::DataType<float>::type, true ),
                           weights_pyr( weights[i], pyr_max_level + 1, cv::DataType<float>::type, true );
          
      for(int l = 0; l < pyr_max_level; l++) 
      {
        // Extract the "high frequency" components
        cv::Mat h_img;
        cv::pyrUp( img_pyr[l + 1], h_img, img_pyr[l].size());
        img_pyr[l] -= h_img;
      }
      
      for(int l = 0; l <= pyr_max_level; l++) 
      {
        for ( int y = 0; y < img_pyr[l].rows; y++ )
        {
          float *img_p = img_pyr[l].ptr< float > ( y );
          float *w_p = weights_pyr[l].ptr< float > ( y );
          for ( int x = 0; x < img_pyr[l].cols; x++, img_p++, w_p++ )
            *img_p *= (*w_p);
        }
        
        #pragma omp critical
        {         
          if(res_pyr[l].empty())
            res_pyr[l] = img_pyr[l];
          else
            res_pyr[l] += img_pyr[l];
        }
      }
    }
      
    for(int l = pyr_max_level; l > 0; l--)
    {
      cv::Mat h_img;
      cv::pyrUp(res_pyr[l], h_img, res_pyr[l - 1].size());
      res_pyr[l - 1] += h_img;
    }

    fusion = cv::Mat(cv::Size(width, height), cv::DataType<float>::type);
    res_pyr[0].copyTo(fusion);
  }
  else
  {
    #pragma omp parallel for if( this->parallelism_enabled_ )
    for ( int y = 1; y < height - 1; y++ )
    {
      /* Using simple laplacian filter kernel:
      * 
      * 0  1  0
      * 1 -4  1
      * 0  1  0
      * 
      * where:
      * 
      * ----------   up_pix_p    ----------
      * left_pix_p   cur_pix_p   right_pix_p
      * ----------   down_pix_p  ----------
      */
      
      std::vector< const cv::Vec<_TPixDepth, 3>* > cur_pix_p( n_images ), up_pix_p( n_images ), 
                                          down_pix_p( n_images ), left_pix_p( n_images ),
                                          right_pix_p( n_images );
      std::vector< float* > w_p( n_images );
      float *w_sum_p;
      
      for ( int i = 0; i < n_images; i++ )
      {
        right_pix_p[i] = imgs[i].ptr< cv::Vec<_TPixDepth, 3> > ( y );
        left_pix_p[i] = right_pix_p[i]++;
        cur_pix_p[i] = right_pix_p[i]++;
        up_pix_p[i] = imgs[i].ptr< cv::Vec<_TPixDepth, 3> > ( y - 1 );
        down_pix_p[i] = imgs[i].ptr< cv::Vec<_TPixDepth, 3> > ( y + 1 );
        up_pix_p[i]++;
        down_pix_p[i]++;
        
        w_p[i] = weights[i].ptr< float > ( y );
        w_p[i]++;
      }
      
      w_sum_p = weights_sum.ptr< float > ( y );
      w_sum_p++;
      
      for ( int x = 1; x < width - 1; x++, w_sum_p++ )
      {
        // For each exposure
        for ( int i = 0; i < n_images; i++ )
        { 
          const cv::Vec<_TPixDepth, 3> &cur_pix = *cur_pix_p[i]++,
                               &left_pix = *left_pix_p[i]++,
                               &right_pix = *right_pix_p[i]++,
                               &up_pix = *up_pix_p[i]++,
                               &down_pix = *down_pix_p[i]++;

          double cur_gl_val = double(cur_pix(0) + cur_pix(1) + cur_pix(2))/3.0,
                 left_gl_val = double(left_pix(0) + left_pix(1) + left_pix(2))/3.0,
                 right_gl_val = double(right_pix(0) + right_pix(1) + right_pix(2))/3.0,
                 up_gl_val = double(up_pix(0) + up_pix(1) + up_pix(2))/3.0,
                 down_gl_val = double(down_pix(0) + down_pix(1) + down_pix(2))/3.0;
                
          double contrast = fabs( left_gl_val + right_gl_val + up_gl_val + down_gl_val - 4.0 * cur_gl_val );
          
          double saturation = (cur_pix(0) - cur_gl_val)*(cur_pix(0) - cur_gl_val);
          saturation += (cur_pix(1) - cur_gl_val)*(cur_pix(1) - cur_gl_val);
          saturation += (cur_pix(2) - cur_gl_val)*(cur_pix(2) - cur_gl_val);
          saturation = sqrt(saturation);

          
          double exposedness = ( exposedness_weights_[cur_pix(0)] * 
                                 exposedness_weights_[cur_pix(1)] * 
                                 exposedness_weights_[cur_pix(2)] );
          
          if( contrast < DBL_EPSILON ) contrast = DBL_EPSILON;
          if( saturation < DBL_EPSILON ) saturation = DBL_EPSILON;
          if( exposedness < DBL_EPSILON ) exposedness = DBL_EPSILON;
          
          contrast = pow(contrast, contrast_exp_);
          saturation = pow(saturation, saturation_exp_);
          exposedness = pow(exposedness, exposedness_exp_);
          
          *w_p[i] = contrast * saturation * exposedness;
          *w_sum_p += *w_p[i]++;
        }
      }
    }
    
    // Complete the first and last rows and cols for the weights matrix
    fillWeightsBorders( weights, weights_sum );
    
    #pragma omp parallel for if( this->parallelism_enabled_ )
    for ( int i = 0; i < n_images; i++ )
    {
      weights[i] /= weights_sum;
        
      // For each exposure, generate a pyramid for both images and weights
      cv_ext::ImagePyramid img_pyr( imgs[i], pyr_max_level + 1, cv::DataType<cv::Vec3f>::type, true ),
                           weights_pyr( weights[i], pyr_max_level + 1, cv::DataType<float>::type, true );
          
      for(int l = 0; l < pyr_max_level; l++) 
      {
        // Extract the "high frequency" components
        cv::Mat h_img;
        cv::pyrUp( img_pyr[l + 1], h_img, img_pyr[l].size());
        img_pyr[l] -= h_img;
      }
      
      for(int l = 0; l <= pyr_max_level; l++) 
      {
        for ( int y = 0; y < img_pyr[l].rows; y++ )
        {
          cv::Vec3f *img_p = img_pyr[l].ptr< cv::Vec3f > ( y );
          float *w_p = weights_pyr[l].ptr< float > ( y );
          for ( int x = 0; x < img_pyr[l].cols; x++, img_p++, w_p++ )
          {
            (*img_p)(0) *= (*w_p);
            (*img_p)(1) *= (*w_p);
            (*img_p)(2) *= (*w_p);
          }
        }
        
        #pragma omp critical
        {         
          if(res_pyr[l].empty())
            res_pyr[l] = img_pyr[l];
          else
            res_pyr[l] += img_pyr[l];
        }
      }
    }
      
    for(int l = pyr_max_level; l > 0; l--)
    {
      cv::Mat h_img;
      cv::pyrUp(res_pyr[l], h_img, res_pyr[l - 1].size());
      res_pyr[l - 1] += h_img;
    }

    fusion = cv::Mat(cv::Size(width, height), cv::DataType<cv::Vec3f>::type);
    res_pyr[0].copyTo(fusion);

  }
}
 
template <typename _TPixDepth>
  void MertensHDR<_TPixDepth>::merge ( const std::vector<cv::Mat> &imgs, const std::vector<float> &exp_times,
                               cv::Mat &fusion ) const
{
  this->merge( imgs, fusion );
}

template <typename _TPixDepth>
  void MertensHDR<_TPixDepth>::fillWeightsBorders( std::vector<cv::Mat> &weights, cv::Mat &weights_sum ) const
{
  const int n_images = weights.size();
  const int width = weights[0].cols, height  = weights[0].rows;
  
  // Complete the first and last rows and cols for the weights matrix
  weights_sum.col(0) = cv::Mat::zeros(height, 1, cv::DataType<float>::type );
  weights_sum.col(width - 1)  = cv::Mat::zeros(height, 1, cv::DataType<float>::type );
  weights_sum.row(0)  = cv::Mat::zeros(1, width, cv::DataType<float>::type );
  weights_sum.row(height - 1)  = cv::Mat::zeros(1, width, cv::DataType<float>::type );
    
  for ( int i = 0; i < n_images; i++ )
  {
    
    cv::Mat w_first_col = weights[i].col(0), w_last_col =  weights[i].col(width - 1), 
            w_first_row = weights[i].row(0), w_last_row =  weights[i].row(height - 1);
            
    weights[i].col(1).copyTo(w_first_col);
    weights[i].col(width - 2).copyTo(w_last_col);
    weights[i].row(1).copyTo(w_first_row);
    weights[i].row(height - 2).copyTo(w_last_row);
    
    weights_sum.col(0) += w_first_col;
    weights_sum.col(width - 1) += w_last_col;
    weights_sum.row(0) += w_first_row;
    weights_sum.row(height - 1) += w_last_row;
    
    weights_sum.at<float>(0, 0) -= weights[i].at<float>(0, 0);
    weights_sum.at<float>(0, width - 1) -= weights[i].at<float>(0, width - 1);
    weights_sum.at<float>(height - 1, 0) -= weights[i].at<float>(height - 1, 0);
    weights_sum.at<float>(height - 1, width - 1) -= weights[i].at<float>(height - 1, width - 1);
  }    
}

void Tonemap::checkData ( const cv::Mat &hdr_img, const cv::Mat &ldr_img ) const
{
  if ( !hdr_img.rows || !hdr_img.cols || ( hdr_img.channels() != 1 && hdr_img.channels() != 3 ) ||
       hdr_img.depth() != cv::DataType< float >::depth )
    throw std::invalid_argument ( "Unsupported image: size must be greater than zero and \
                                   only grey level or RGB float images are supported " );
}

void GammaTonemap::compute ( const cv::Mat& hdr_img, cv::Mat& ldr_img ) const 
{
  checkData ( hdr_img, ldr_img );

  const int n_channels = hdr_img.channels();
  if( n_channels == 1 )
    ldr_img = cv::Mat ( hdr_img.size(), cv::DataType< float >::type );
  else
    ldr_img = cv::Mat ( hdr_img.size(), cv::DataType< cv::Vec3f >::type );
  
  double min, max, range, g = 1.0f / gamma_;

  cv::minMaxLoc ( hdr_img, &min, &max );
  range = max - min;

  if ( range > DBL_EPSILON )
    ldr_img = ( hdr_img - min ) / range;
  else
    hdr_img.copyTo ( ldr_img );

  cv::pow ( ldr_img, g, ldr_img );

  ldr_img *= this->max_level_;
}


#define CV_EXT_INSTANTIATE_DebevecHDR(_T) template class DebevecHDR<_T>;
#define CV_EXT_INSTANTIATE_MertensHDR(_T) template class MertensHDR<_T>;

CV_EXT_INSTANTIATE( DebevecHDR, CV_EXT_UINT_TYPES )
CV_EXT_INSTANTIATE( MertensHDR, CV_EXT_UINT_TYPES )
