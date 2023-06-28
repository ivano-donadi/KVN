/*
 * cv_ext - openCV EXTensions
 *
 *  Copyright (c) 2020, Alberto Pretto <alberto.pretto@flexsight.eu>
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

#include <vector>
#include <opencv2/opencv.hpp>

#include "cv_ext/types.h"
#include "cv_ext/aligned_mat.h"

namespace cv_ext
{
/**
 * @brief ImagePyramidBase is an utility class used to produce a gaussian or a general image pyramid
 *        from an input image.
 *
 * @tparam _align The internal data will be aligned at the requested boundary, see cv_ext::MemoryAlignment
 */
template < MemoryAlignment _align > class ImagePyramidBase
{
public:

  /**
   * @brief Default constructor: it creates an empty pyramid
   *
   * Use the create() or createCustom() methods to build an image pyramid
   */  
  ImagePyramidBase() = default;

  ImagePyramidBase (ImagePyramidBase&& other ) = default;

  ImagePyramidBase (const ImagePyramidBase& other ) = delete;
  ImagePyramidBase& operator= (const ImagePyramidBase& other ) = delete;

  /**
   * @brief Constructor that builds a gaussian image pyramid from an input image
   *
   * @param[in] img Input image used to build the pyramid
   * @param[in] pyr_levels Numbers of pyramid levels, it should be greater or equal than 1.
   * @param[in] data_type Desired pyramid elements type as provided by the OpenCV template "trait" class
   *                      cv::DataType< T >::type, where T a basic type (char, float, ,...).
   *                      If data_type is negative, the pyramid images will have the same type as the input image
   * @param[in] deep_copy If true, make a deep copy of the input image as level 0 of the pyramid
   *
   * A gaussian pyramid is built smoothing with a gaussian filter the input image and then subsampling
   * by a scale factor of 2 the smoothed image
   *
   * @note The number of channels of each pyramid images is the the same as the input image img
   */
  ImagePyramidBase(const cv::Mat &img, int pyr_levels, int data_type = -1, bool deep_copy = true );

  /**
   * @brief Builds a gaussian image pyramid from an input image
   *
   * @param[in] img Input image used to build the pyramid
   * @param[in] pyr_levels Numbers of pyramid levels, it should be greater or equal than 1.
   * @param[in] data_type Desired pyramid elements type as provided by the OpenCV template "trait" class
   *                      cv::DataType< T >::type, where T a basic type (char, float, ,...).
   *                      If data_type is negative, the pyramid images will have the same type as the input image
   * @param[in] deep_copy If true, make a deep copy of the input image as level 0 of the pyramid
   *
   * A gaussian pyramid is built smoothing with a gaussian filter the input image and then subsampling
   * by a scale factor of 2 the smoothed image
   *
   * @note The number of channels of each pyramid images is the the same as the input image img
   */  
  void create( const cv::Mat &img, int pyr_levels, int data_type = -1, bool deep_copy = true );

  /**
   * @brief Builds a general image pyramid with user defined scale factor from an input image
   *
   * @param[in] img Input image used to build the pyramid
   * @param[in] pyr_levels Numbers of pyramid levels, it should be greater or equal than 1.
   * @param[in] pyr_scale_factor Desired scale factor between two consecutive levels
   * @param[in] data_type Desired pyramid elements type as provided by the OpenCV template "trait" class
   *                      cv::DataType< T >::type, where T a basic type (char, float, ,...).
   *                      If data_type is negative, the pyramid images will have the same type as the input image
   * @param[in] interp_type Interpolation type used to resize the image between two consecutive levels
   * @param[in] deep_copy If true, make a deep copy of the input image as level 0 of the pyramid
   *
   * A general image pyramid is built resizing the input image with the given scale factor and the given
   * interpolation type
   *
   * @note The number of channels of each pyramid images is the the same as the input image img
   */   
  void createCustom( const cv::Mat &img, int pyr_levels, double pyr_scale_factor, int data_type = -1,
                     InterpolationType interp_type = cv_ext::INTERP_NEAREST_NEIGHBOR, bool deep_copy = true  );
  
  /**
   * @brief Provides the numbers of pyramid levels
   */  
  int numLevels()  const { return num_levels_; };
  
  /**
   * @brief Provides the scale factor between two consecutive pyramid levels
   */  
  double scaleFactor() const { return scale_factor_; };

  /**
   * @brief Provides a reference to the image associated to a desired pyramid level
   * 
   * @param[in] pyr_level Desired pyramid level
   */  
  cv::Mat& operator[]( int pyr_level ){  return pyr_imgs_[pyr_level]; };
  
  /**
   * @brief Provides a reference to the image associated to a desired pyramid level
   * 
   * @param[in] pyr_level Desired pyramid level
   */  
  cv::Mat& at( int pyr_level ){  return pyr_imgs_[pyr_level]; };

  /**
   * @brief Provides a reference to the image associated to a desired pyramid level
   *
   * @param[in] pyr_level Desired pyramid level
   */
  const cv::Mat& operator[]( int pyr_level ) const {  return pyr_imgs_[pyr_level]; };

  /**
   * @brief Provides a reference to the image associated to a desired pyramid level
   *
   * @param[in] pyr_level Desired pyramid level
   */
  const cv::Mat& at( int pyr_level ) const {  return pyr_imgs_[pyr_level]; };

  /**
   * @brief Provides the scale associated to a desired pyramid level
   * 
   * @param[in] pyr_level Desired pyramid level
   */    
  double getScale( int pyr_level ) const;
  
private:
  
  void buildPyr( const cv::Mat& img, int data_type, bool deep_copy );

  int num_levels_ = 0;
  double scale_factor_ = 0.0;

  std::vector< cv_ext::AlignedMatBase<_align> > pyr_imgs_;
  std::vector<double> scales_; 
  bool gaussian_pyr_ = true;
  InterpolationType interp_type_ = cv_ext::INTERP_NEAREST_NEIGHBOR;
};

typedef ImagePyramidBase<CV_EXT_DEFAULT_ALIGNMENT> ImagePyramid;

}
