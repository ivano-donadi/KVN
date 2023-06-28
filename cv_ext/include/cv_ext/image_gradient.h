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

#include <memory>
#include <mutex>
#include "cv_ext/image_pyramid.h"
#include <Eigen/Dense>

namespace cv_ext
{
/**
 * @brief ImageGradientBase is an utility class used to produce the partial derivatives, the gradient magnitude and
 *        the gradient orientation from an input image.
 *        It can provide also a gaussian pyramid representation of such operators, and it
 *        allows to specify the memory alignment (16, 32, 64, .. bytes).
 *
 * @tparam _align The internal data will be aligned at the requested boundary, see cv_ext::MemoryAlignment
 *
 * @note The operators (e.g., the gradient magnitede) has computed "on demand" when the corresponding getX() method
 *       is called.
 * @warning All GetX() methods are thread safe, while other methods are not
 */
template < MemoryAlignment _align > class ImageGradientBase
{
public:

  /**
   * @brief Default constructor: it creates an empty object
   *
   * Use the method create() to setup the object with an input image
   */
  ImageGradientBase() = default;

  /**
   * @brief Constructor that setups the object from an input image
   *
   * It calls create(), so refer to this methods for details on usage and parameters
   */
  ImageGradientBase( const cv::Mat &img, unsigned int pyr_levels = 1, double pyr_scale_factor = 2.0,
                     bool deep_copy = true, bool bgr_color_order = true );


  /**
   * @brief Setup the object from an input image
   *
   * @param[in] img Input image from which the derivatives are computed
   * @param[in] pyr_levels Numbers of pyramid levels, it should be greater or equal than 1.
   * @param[in] pyr_scale_factor Desired scale factor between two consecutive pyramid levels
   * @param[in] deep_copy If true, make a copy of the input image and set it at level 0 of the intensities images pyramid
   * @param[in] bgr_color_order If true, consider the color of an input RGB images encoded with the reverse order (BGR)
   *
   * Setup the object in order to provide the partial derivatives, the gradient magnitude and
   * the gradient orientation of an input image, possibly in a pyramidal representation (see pyr_levels and
   * pyr_scale_factor parameters). If pyr_scale_factor is equal to 2.0, a standard Gaussian pyramid is computed.
   *
   * @note Partial derivatives, gradient magnitude and gradient orientation images are provided as single channel,
   * single precision floating-point image.
   *
   * @warning If deep_copy is false and cv::Mat img has not proper memory alignment, this method raises an exception.
   */
  void create( const cv::Mat &img, unsigned int pyr_levels = 1, double pyr_scale_factor = 2.0,
               bool deep_copy = true, bool bgr_color_order = true );

  /**
   * @brief Move constructor
   *
   * @param [in] other moved object
   */
  ImageGradientBase (ImageGradientBase &&other );

  ImageGradientBase(const ImageGradientBase &other ) = delete;
  ImageGradientBase & operator=(const ImageGradientBase &other ) = delete;

  virtual ~ImageGradientBase() = default;

  /**
   * @brief Provides the numbers of pyramid levels
   * */
  int numPyrLevels() const { return pyr_num_levels_; };

  /**
   * @brief Provides the scale factor between two consecutive pyramid levels
   */
  double pyrScaleFactor() const { return pyr_scale_factor_; };

  /**
   * @brief If enabled, 3x3 Scharr kernels are used to calculate the derivatives in place of the Sobel kernels.
   */
  void enableScharrOperator( bool enable ){ use_scharr_operator_ = enable; };

  /**
   * @brief If enabled, the magnitude image is computed in a quickest way, i.e., by using the L1 in place of the L2 norm
   */
  void enableFastMagnitude ( bool enable ){ fast_magnitude_ = enable; };

  /**
   * @brief Return true if 3x3  Scharr kernels are used to calculate the derivatives in place of the Sobel kernels.
   */
  bool usingScharrOperator(){ return use_scharr_operator_; };

  /**
   * @brief Return true if the magnitude image is computed in a quickest way, by using the L1 in place of the L2 norm
   */
  bool usingFastMagnitude (){ return fast_magnitude_; };

/**
 * @brief Provides the scale associated to a desired pyramid level
 *
 * @param[in] pyr_level Desired pyramid level
 */
  double getPyrScale( int pyr_level ) const { return gl_pyr_.getScale( pyr_level ); };

  /**
   * @brief Provides a reference to the input (possibly converted to) grey-level image at a desired pyramid level
   *
   * @param[in] pyr_level Desired pyramid level
   *
   * The output image is single channel, single precision floating-point cv::Mat, with memory aligned as specified
   * by the template parameter _align.
   * @note The grey-level [min,max] range is the same as the input image, so it'is not automatically normalized
   * between 0 and 1
   */
  const cv::Mat &getIntensities( int scale_index = 0 );

  /**
   * @brief Provides a reference to the derivative of the the input image along the X direction, at a desired pyramid
   *        level
   *
   * @param[in] pyr_level Desired pyramid level
   *
   * The derivatives are computed by default by using the 3x3 Scharr kernels, or the Sobel kernels 
   * (see the enableScharrOperator() method).
   * The output image is single channel, single precision floating-point cv::Mat, with memory aligned as specified
   * by the template parameter _align.
   */
  const cv::Mat &getGradientX( int scale_index = 0 );

  /**
   * @brief Provides a reference to the derivative of the the input image along the Y direction, at a desired pyramid
   *        level
   *
   * @param[in] pyr_level Desired pyramid level
   *
   * The derivatives are computed by default by using the 3x3 Scharr kernels, or the Sobel kernels 
   * (see the enableScharrOperator() method).
   * The output image is single channel, single precision floating-point cv::Mat, with memory aligned as specified
   * by the template parameter _align.
   */
  const cv::Mat &getGradientY( int scale_index = 0 );

  /**
   * @brief Provides a reference to the gradient direction image, at a desired pyramid level
   *
   * @param[in] pyr_level Desired pyramid level
   *
   * The gradient direction is an angle (in radians) in the range [-pi/2,pi/2] computed using an approximated atan()
   * function applied to each X,Y image derivative component. When both X and Y components are 0,
   * the direction is set to 0. The output image is single channel, single precision floating-point cv::Mat, with
   * memory aligned as specified by the template parameter _align.
   */
  const cv::Mat &getGradientDirections( int scale_index = 0 );

  /**
   * @brief Provides a reference to the gradient orientation image, at a desired pyramid level
   *
   * @param[in] pyr_level Desired pyramid level
   *
   * The gradient orientation is an angle (in radians) in the range [-pi,pi] computed using an approximated atan2()
   * function applied to each X,Y image derivative component. When both X and Y components are 0, the orientation
   * is set to 0. The output image is single channel, single precision floating-point cv::Mat, with memory aligned
   * as specified by the template parameter _align.
   */
  const cv::Mat &getGradientOrientations( int scale_index = 0 );

  /**
   * @brief Provides a reference to a matrix with the directions of the greater eigenvectors of the Harris matrices
   *
   * @param[in] pyr_level Desired pyramid level
   *
   * The eigen direction is an angle (in radians) in the range [-pi/2,pi/2] computed using an approximated atan()
   * function applied to the greater eigenvectors of the Harris matrices extracted from the image derivative. The Harris
   * matrix is the 2x2 covariance matrix of the distribution of the derivatives dx,dy computed over a KxK block.
   * Here K is fixed to 5. When both the eigenvetor components are 0, the direction is set to 0.
   * The output image is single channel, single precision floating-point cv::Mat, with memory aligned as specified
   * by the template parameter _align.
   */
  const cv::Mat &getEigenDirections( int scale_index = 0 );

  /**
   * @brief Provides a reference to a matrix with the orientation of the greater eigenvectors of the Harris matrices
   *
   * @param[in] pyr_level Desired pyramid level
   *
   * The eigen orientation is an angle (in radians) in the range [-pi,pi] computed using an approximated atan2()
   * function applied to the greater eigenvectors of the Harris matrices extracted from the image derivative. The Harris
   * matrix is the 2x2 covariance matrix of the distribution of the derivatives dx,dy computed over a KxK block.
   * Here K is fixed to 5. When both the eigenvetor components are 0, the orientation is set to 0.
   * The output image is single channel, single precision floating-point cv::Mat, with memory aligned as specified
   * by the template parameter _align.
   */
  const cv::Mat &getEigenOrientations( int scale_index = 0 );

  /**
   * @brief Provides a reference to the gradient magnitude image, at a desired pyramid level
   *
   * @param[in] pyr_level Desired pyramid level
   *
   * The magnitude is computed by default as the sum of absolute values of the X,Y derivatives,
   * otherwise the L2 norm of such components is used (see the method enableFastMagnitude()).
   * The output image is single channel, single precision floating-point cv::Mat, with memory aligned as specified
   * by the template parameter _align.
   */
  const cv::Mat &getGradientMagnitudes( int scale_index = 0 );

  /**
   * @brief Provides a vector of intensity values extracted from the input (possibly converted to) grey-level image,
   *        extracted at desired locations and a desired pyramid level
   *
   * @param[in] coords Desired pixel locations
   * @param[in] pyr_level Desired pyramid level
   * @param[in] type Interpolation type used to look-up the values
   *
   * If it is not possible to look-up a value (e.g., a coordinate is outside the image) the corresponding value
   * is set to OUT_OF_IMG_VAL (i.e., to std::numeric_limits<float>::max(), defined in the standard header <limits>).
   * @note The grey-level [min,max] range is the same as the input image, so it'is not automatically normalized
   * between 0 and 1
   */
  std::vector<float> getIntensities( const std::vector<cv::Point2f> &coords, int scale_index = 0,
                                     InterpolationType type = INTERP_NEAREST_NEIGHBOR );

  /**
   * @brief Provides a vector of values extracted from the derivative of the the input image along the X direction,
   *        extracted at desired locations and a desired pyramid level.
   *
   * @param[in] coords Desired pixel locations
   * @param[in] pyr_level Desired pyramid level
   * @param[in] type Interpolation type used to look-up the values
   *
   * If it is not possible to look-up a value (e.g., a coordinate is outside the image) the corresponding value
   * is set to OUT_OF_IMG_VAL (i.e., to std::numeric_limits<float>::max(), defined in the standard header <limits>).
   * The derivatives are computed by default by using the 3x3 Scharr kernels, or the Sobel kernels 
   * (see the enableScharrOperator() method).
   */
  std::vector<float> getGradientX( const std::vector<cv::Point2f> &coords, int scale_index = 0,
                                   InterpolationType type = INTERP_NEAREST_NEIGHBOR );

  /**
   * @brief Provides a vector of values extracted from the derivative of the the input image along the Y direction,
   *        extracted at desired locations and a desired pyramid level
   *
   * @param[in] coords Desired pixel locations
   * @param[in] pyr_level Desired pyramid level
   * @param[in] type Interpolation type used to look-up the values
   *
   * If it is not possible to look-up a value (e.g., a coordinate is outside the image) the corresponding value
   * is set to OUT_OF_IMG_VAL (i.e., to std::numeric_limits<float>::max(), defined in the standard header <limits>).
   * The derivatives are computed by default by using the 3x3 Scharr kernels, or the Sobel kernels 
   * (see the enableScharrOperator() method).
   */
  std::vector<float> getGradientY( const std::vector<cv::Point2f> &coords, int scale_index = 0,
                                   InterpolationType type = INTERP_NEAREST_NEIGHBOR );

  /**
   * @brief Provides a vector of values extracted from the gradient direction image, extracted at desired locations
   *        and a desired pyramid level.
   *
   * @param[in] coords Desired pixel locations
   * @param[in] pyr_level Desired pyramid level
   * @param[in] type Interpolation type used to look-up the values
   *
   * If it is not possible to look-up a value (e.g., a coordinate is outside the image) the corresponding value
   * is set to OUT_OF_IMG_VAL (i.e., to std::numeric_limits<float>::max(), defined in the standard header <limits>).
   * The gradient direction is an angle (in radians) in the range [-pi/2,pi/2] computed using an approximated atan()
   * function applied to each X,Y image derivative component. When both X and Y components are 0,
   * the direction is set to 0.
   */
  std::vector<float> getGradientDirections( const std::vector<cv::Point2f> &coords, int scale_index = 0,
                                            InterpolationType type = INTERP_NEAREST_NEIGHBOR  );

  /**
   * @brief Provides a vector of values extracted from the gradient orientation image, extracted at desired locations
   *        and a desired pyramid level.
   *
   * @param[in] coords Desired pixel locations
   * @param[in] pyr_level Desired pyramid level
   * @param[in] type Interpolation type used to look-up the values
   *
   * If it is not possible to look-up a value (e.g., a coordinate is outside the image) the corresponding value
   * is set to OUT_OF_IMG_VAL (i.e., to std::numeric_limits<float>::max(), defined in the standard header <limits>).
   * The gradient orientation is an angle (in radians) in the range [-pi, pi] computed using an approximated atan2()
   * function applied to each X,Y image derivative component (default). When both X and Y components are 0,
   * the orientation is set to 0.
   */
  std::vector<float> getGradientOrientations( const std::vector<cv::Point2f> &coords, int scale_index = 0,
                                              InterpolationType type = INTERP_NEAREST_NEIGHBOR  );

  /**
   * @brief Provides a vector of values extracted from a matrix with the directions of the greater eigenvectors of
   * the Harris matrices, extracted at desired locations and a desired pyramid level.
   *
   * @param[in] coords Desired pixel locations
   * @param[in] pyr_level Desired pyramid level
   * @param[in] type Interpolation type used to look-up the values
   *
   * If it is not possible to look-up a value (e.g., a coordinate is outside the image) the corresponding value
   * is set to OUT_OF_IMG_VAL (i.e., to std::numeric_limits<float>::max(), defined in the standard header <limits>).
   * The eigen direction is an angle (in radians) in the range [-pi/2,pi/2] computed using an approximated atan()
   * function applied to the greater eigenvectors of the Harris matrices extracted from the image derivative. The Harris
   * matrix is the 2x2 covariance matrix of the distribution of the derivatives dx,dy computed over a KxK block.
   * Here K is fixed to 5. When both the eigenvetor components are 0, the direction is set to 0.
   */
  std::vector<float> getEigenDirections( const std::vector<cv::Point2f> &coords, int scale_index = 0,
                                         InterpolationType type = INTERP_NEAREST_NEIGHBOR  );

  /**
   * @brief Provides a vector of values extracted from a matrix with the orientations of the greater eigenvectors of
   * the Harris matrices, extracted at desired locations and a desired pyramid level.
   *
   * @param[in] coords Desired pixel locations
   * @param[in] pyr_level Desired pyramid level
   * @param[in] type Interpolation type used to look-up the values
   *
   * If it is not possible to look-up a value (e.g., a coordinate is outside the image) the corresponding value
   * is set to OUT_OF_IMG_VAL (i.e., to std::numeric_limits<float>::max(), defined in the standard header <limits>).
   * The eigen orientation is an angle (in radians) in the range [-pi,pi] computed using an approximated atan2()
   * function applied to the greater eigenvectors of the Harris matrices extracted from the image derivative. The Harris
   * matrix is the 2x2 covariance matrix of the distribution of the derivatives dx,dy computed over a KxK block.
   * Here K is fixed to 5. When both the eigenvetor components are 0, the orientation is set to 0.
   */
  std::vector<float> getEigenOrientations( const std::vector<cv::Point2f> &coords, int scale_index = 0,
                                           InterpolationType type = INTERP_NEAREST_NEIGHBOR  );

  /**
   * @brief Provides a vector of values extracted from the gradient magnitude image, extracted at desired locations and
   *        a desired pyramid level.
   *
   * @param[in] coords Desired pixel locations
   * @param[in] pyr_level Desired pyramid level
   * @param[in] type Interpolation type used to look-up the values
   *
   * If it is not possible to look-up a value (e.g., a coordinate is outside the image) the corresponding value
   * is set to OUT_OF_IMG_VAL (i.e., to std::numeric_limits<float>::max(), defined in the standard header <limits>).
   * The magnitude is computed by default as the sum of absolute values of the X,Y derivatives,
   * otherwise the L2 norm of such components is used (see the method enableFastMagnitude()).
   */
  std::vector<float> getGradientMagnitudes( const std::vector<cv::Point2f> &coords, int scale_index = 0,
                                            InterpolationType type = INTERP_NEAREST_NEIGHBOR  );

  const float OUT_OF_IMG_VAL = std::numeric_limits<float>::max();

private:

  void setImage( const cv::Mat &img, bool deep_copy, bool bgr_color_order );
  void computeGradientImages( int scale_index );
  void computeGradientDirections( int scale_index );
  void computeGradientOrientations( int scale_index );
  void computeEigenDirections( int scale_index );
  void computeEigenOrientations( int scale_index );
  void computeGradientMagnitude( int scale_index );

  void computeDirections( const cv::Mat &vx, const cv::Mat &vy, cv::Mat &dst );
  void computeOrientations( const cv::Mat &vx, const cv::Mat &vy, cv::Mat &dst );
  void computeEigenVector(int scale_index, cv::Mat &vx, cv::Mat &vy );


  template <typename T> std::vector<float>
    getResultFromCordsNN( const cv_ext::AlignedMatBase <_align > &res_img,
                          const std::vector<cv::Point2f> &coords ) const;
  template <typename T> std::vector<float>
    getResultFromCordsBL( const cv_ext::AlignedMatBase <_align > &res_img,
                          const std::vector<cv::Point2f> &coords ) const;
  template <typename T> std::vector<float>
    getResultFromCordsBC( const cv_ext::AlignedMatBase <_align > &res_img,
                          const std::vector<cv::Point2f> &coords ) const;
  
  std::recursive_mutex mutex_;

  int pyr_num_levels_ = 0;
  double pyr_scale_factor_ = 0.0;
  bool use_scharr_operator_ = true;
  bool fast_magnitude_ = true;

  ImagePyramidBase <_align > gl_pyr_;

  std::vector< cv_ext::AlignedMatBase <_align > > dx_imgs_, dy_imgs_;
  std::vector< cv_ext::AlignedMatBase <_align > > gradient_dir_imgs_;
  std::vector< cv_ext::AlignedMatBase <_align > > gradient_ori_imgs_;
  std::vector< cv_ext::AlignedMatBase <_align > > eigen_dir_imgs_;
  std::vector< cv_ext::AlignedMatBase <_align > > eigen_ori_imgs_;
  std::vector< cv_ext::AlignedMatBase <_align > > gradient_mag_imgs_;

};

typedef ImageGradientBase<CV_EXT_DEFAULT_ALIGNMENT> ImageGradient;

}
