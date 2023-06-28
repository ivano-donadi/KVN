#include <stdexcept>

#include "cv_ext/image_pyramid.h"
#include "cv_ext/memory.h"

using namespace cv;
using namespace cv_ext;
using namespace std;

template class cv_ext::ImagePyramidBase<MEM_ALIGN_NONE>;
template class cv_ext::ImagePyramidBase<MEM_ALIGN_16>;
template class cv_ext::ImagePyramidBase<MEM_ALIGN_32>;
template class cv_ext::ImagePyramidBase<MEM_ALIGN_64>;
template class cv_ext::ImagePyramidBase<MEM_ALIGN_128>;
template class cv_ext::ImagePyramidBase<MEM_ALIGN_256>;
template class cv_ext::ImagePyramidBase<MEM_ALIGN_512>;

template < MemoryAlignment _align >
  ImagePyramidBase<_align >::ImagePyramidBase(const cv::Mat &img, int pyr_levels, int data_type, bool deep_copy  )
{
  create( img, pyr_levels, data_type, deep_copy );
}

template < MemoryAlignment _align >
  void ImagePyramidBase<_align >::create(const Mat &img, int pyr_levels, int data_type, bool deep_copy )
{
  if( pyr_levels < 1 )
    throw invalid_argument ( "ImagePyramidBase::Number of level must be greater or equal than 1" );
  
  scale_factor_ = 2.0; 
  gaussian_pyr_ = true;
  num_levels_ = pyr_levels;
  interp_type_ = cv_ext::INTERP_NEAREST_NEIGHBOR;

  buildPyr( img, data_type, deep_copy );
}

template < MemoryAlignment _align >
  void ImagePyramidBase<_align >::createCustom (const Mat& img, int pyr_levels, double pyr_scale_factor,
                                                int data_type, InterpolationType interp_type, bool deep_copy )
{
  
  if ( pyr_scale_factor <= 1.0 || pyr_scale_factor > 2.0 )
    throw invalid_argument ( "ImagePyramidBase::Pyramid scale factor must be in the range (1,2]" );
  
  if( pyr_levels < 1 )
    throw invalid_argument ( "ImagePyramidBase::Number of level must be greater or equal than 1" );

  scale_factor_ = pyr_scale_factor; 
  gaussian_pyr_ = false;
  num_levels_ = pyr_levels;
  interp_type_ = interp_type;

  buildPyr( img, data_type, deep_copy );
}

template < MemoryAlignment _align >
  double ImagePyramidBase<_align >::getScale (int level ) const
{
  if( level < 0 || level >= num_levels_ )
    throw invalid_argument("ImagePyramidBase::Scale index out of range");
  
  return scales_[level];
}

template < MemoryAlignment _align >
  void ImagePyramidBase<_align >::buildPyr(const Mat& img, int data_type, bool deep_copy )
{    
  pyr_imgs_.clear();
  pyr_imgs_.reserve( num_levels_ );
  scales_.resize( num_levels_ );
  scales_[0] = 1.0;

  for ( int i = 1; i < num_levels_; i++ )
    scales_[i] = scales_[i-1]*scale_factor_;

  if( data_type < 0 || img.type() == data_type )
  {
    pyr_imgs_.emplace_back( img, deep_copy );
  }
  else
  {
    pyr_imgs_.emplace_back(img.size(), data_type );
    img.convertTo( pyr_imgs_[0], data_type );
  }

  int last_level = num_levels_ - 1;

  // Scale down to obtain the required pyramid level
  if( gaussian_pyr_ )
  {
    // Use pyrDown() for scale factor 2
    for ( int i = 1; i <= last_level; i++ )
    {
      pyr_imgs_.emplace_back();
      pyrDown(pyr_imgs_[i-1], pyr_imgs_[i]);
    }
  }
  else
  {
    // ... otherwise, use resize()

    int interp_type;
    switch( interp_type_ )
    {
      case cv_ext::INTERP_NEAREST_NEIGHBOR:
        interp_type = INTER_NEAREST;
        break;
      case cv_ext::INTERP_BILINEAR:
        interp_type = INTER_LINEAR;
        break;
      case cv_ext::INTERP_BICUBIC:
        interp_type = INTER_CUBIC;
        break;
      default:
        interp_type = INTER_NEAREST;
        break;
    }
    
    for ( int i = 1; i <= last_level; i++ )
    {
      pyr_imgs_.emplace_back();
      double cur_scale = 1.0/scales_[i];
      Size cur_size( cvRound(img.cols*cur_scale), cvRound(img.rows*cur_scale));
      resize(pyr_imgs_[i-1], pyr_imgs_[i], cur_size, 0, 0, interp_type );
    }
  }
}
