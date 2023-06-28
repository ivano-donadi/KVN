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

namespace cv_ext
{
/**
 * @brief Abstract base class for High Dynamic Range Imaging classes. Any class inheriting from
 *        HDR must provide implementations for both the merge() methods, in order 
 *        to construct a HDR image from a sequence of (low dynamic range, LDR) images taken with different 
 *        exposure values.
 *        The template argument _TPixDepth represents the single or three channel pixel type 
 *        of the input LDR images. Currently, only CV_EXT_UINT_PIXEL_DEPTH_TYPES types 
 *        are supported (see cv_ext/types.h).
 */
template < typename _TPixDepth > class HDR
{

public:

  HDR();
  ~HDR() {};

  /**
   * @brief Enable/disable parallelism for the projected algorithm
   *
   * @param enable If true, algorithms are run in a parallel region,
   *               otherwise it is run as a single thread
   */
  void enableParallelism ( bool enable )
  {
    parallelism_enabled_ = enable;
  };

  /**
   * @brief Return true is the algorithms are run in a parallel region
   */
  bool isParallelismEnabled() const
  {
    return parallelism_enabled_;
  };

  /**
   * @brief Any class inheriting from HDR must provide implementations for this functions,
   *        in order to construct a HDR image from a sequence of LDR images taken with different 
   *        exposure values. 
   *
   * @param imgs The sequence of input LDR images.
   * @param hdr_img Output HDR image
   * 
   * The input LDR images should be one or three channels with _TPixDepth pixel type. 
   * The output HDR image will have the same number of channels, with float pixel type.
   * This function is useful when the exposure times are unnecessary for the implemented algorithm, 
   * or when it is necessary to automatically estimate the exposure times 
   */
  virtual void merge ( const std::vector<cv::Mat> &imgs, cv::Mat &hdr_img ) const = 0;

  /**
   * @brief Any class inheriting from HDR must provide implementations for this functions,
   *        in order to construct a HDR image from a sequence of LDR images taken with different 
   *        exposure values. 
   *
   * @param imgs The sequence of input LDR images.
   * @param exp_times The sequence of exposure times, one for each image
   * @param hdr_img Output HDR image
   * 
   * The input LDR images should be one or three channels with _TPixDepth pixel type. 
   * The output HDR image will have the same number of channels, with float pixel type.
   */
  virtual void merge ( const std::vector<cv::Mat> &imgs, const std::vector<float> &exp_times,
                       cv::Mat &hdr_img ) const = 0;

protected:

  void computeTriangleWeights( std::vector<double> &weights ) const;
  void computeGaussianWeights( std::vector<double> &weights, double sigma  ) const;
  
  void computeTriangleWeights( std::vector<double> &gl_weights, std::vector<cv::Vec3d> &bgr_weights ) const;
  void computeGaussianWeights( std::vector<double> &gl_weights, std::vector<cv::Vec3d> &bgr_weights, 
                               double sigma  ) const;
  void computeLinearResponse( std::vector<double> &gl_responses, std::vector<cv::Vec3d> &bgr_responses ) const;

  void checkData ( const std::vector<cv::Mat> &imgs ) const;
  void checkData ( const std::vector<cv::Mat> &imgs, const std::vector<float> &exp_times ) const;
  
  bool parallelism_enabled_;
  int pix_levels_;

};

/**
 * @brief Implementation of the algorithm presented in:
 * 
 *        P. Debevec, J. Malik, “Recovering High Dynamic Range Radiance Maps from Photographs”, 
 *        Proceedings of ACM SIGGRAPH, 1997, 369 - 378.
 * 
 *        This algorithm basically composes the HDR image computing each pixel as a weighted sum of the same 
 *        pixel seen at n different exposure times.
 *        
 *        hdr_val(u,v) = eta * ( w[ ldr_val(u,v,0) ]*log(res[ ldr_val(u,v,0) ] / time(0) ) + ...
 *                               w[ ldr_val(u,v,n-1) ]*log(res[ ldr_val(u,v,n-1) ] / time(n-1) ) )
 * 
 *        where w[.] is a weights vector, res[.] is a responses vector, time(.) is the exposure time and
 *        eta is a normalization parameter.
 *        By default, DebevecHDR uses triangle weights and linear responses.
 * 
 *        The resulting HDR image requires tonemapping before being processed.
 * 
 *        The template argument _TPixDepth represents the single or three channel pixel type 
 *        of the input LDR images. Currently, only uchar and ushort types are supported.
 */
template <typename _TPixDepth> class DebevecHDR : public HDR<_TPixDepth>
{

public:

  DebevecHDR();
  ~DebevecHDR() {};
  
  /**
   * @brief Set a custom responses vector (one channel case)
   *
   * @param responses The input responses vector, the size should match the pixel type _TPixDepth, 
   *                  e.g. 256 if _TPixDepth == uchar
   */  
  void setResponses ( const std::vector<double> &responses );
  
  /**
   * @brief Set a custom weights vector (one channel case)
   *
   * @param weights The input weights vector, the size should match the pixel type _TPixDepth, 
   *                e.g. 256 if _TPixDepth == uchar
   */  
  void setWeights ( const std::vector<double> &weights );
  
  /**
   * @brief Set a custom responses vector (three channel case)
   *
   * @param responses The input responses vector, the size should match the pixel type _TPixDepth, 
   *                  e.g. 256 if _TPixDepth == uchar
   */  
  void setResponses ( const std::vector<cv::Vec3d> &responses );
  
  /**
   * @brief Set a custom weights vector (three channel case)
   *
   * @param weights The input weights vector, the size should match the pixel type _TPixDepth, 
   *                e.g. 256 if _TPixDepth == uchar
   */  
  void setWeights ( const std::vector<cv::Vec3d> &weights );
  
  /**
   * @brief Set a standard weights vector
   *
   * @param weights The weights function type (e.g., GAUSSIAN_WEIGHT), see WeightFunctionType
   */  
  void setWeights ( WeightFunctionType type );


  /**
   * @brief Construct a HDR image from a sequence of LDR images using the Debevec algorithm,
   *        considering constast exposure times
   *
   * @param imgs The sequence of input LDR images.
   * @param hdr_img Output HDR image
   * 
   * The input LDR images should be one or three channels with _TPixDepth pixel type. 
   * The output HDR image will have the same number of channels, with float pixel type.
   * The resulting HDR image requires tonemapping before being processed.
   */
  virtual void merge ( const std::vector<cv::Mat> &imgs, cv::Mat &hdr_img ) const;
  
  /**
   * @brief Construct a HDR image from a sequence of LDR images taken with different 
   *        exposure values using the Debevec algorithm. 
   *
   * @param imgs The sequence of input LDR images.
   * @param exp_times The sequence of exposure times, one for each image
   * @param hdr_img Output HDR image
   * 
   * The input LDR images should be one or three channels with _TPixDepth pixel type. 
   * The output HDR image will have the same number of channels, with float pixel type.
   * The resulting HDR image requires tonemapping before being processed.
   */
  virtual void merge ( const std::vector<cv::Mat> &imgs, const std::vector<float> &exp_times,
                       cv::Mat &hdr_img ) const;
  
private:

  void computeLogResponses();
  
  std::vector<double> gl_responses_, ln_gl_responses_, gl_weights_;
  std::vector<cv::Vec3d> bgr_responses_, ln_bgr_responses_, bgr_weights_;
  
};

/**
 * @brief Implementation of the algorithm presented in:
 * 
 *        T. Mertens, J. Kautz, F. Van Reeth, “Exposure Fusion”, 
 *        Proceedings of the 15th Pacific Conference on Computer Graphics 
 *        and Applications, 2007, 382 - 390.
 * 
 *        This algorithm basically composes the HDR image computing each pixel as a weighted sum of the same 
 *        pixel seen at n different exposure times, without considering the exposure times:
 *        
 *        hdr_val(u,v) = eta * ( w(u,v,0) * ldr_val(u,v,0) + ...+ w(u,v,n-1) * ldr_val(u,v,n-1) )
 * 
 *        where w[.] is computed as a composition of three factors:
 * 
 *        w(u,v,x) = contrast(u,v,x)^w_cont * saturation(u,v,x)^w_sat * exposedness(u,v,x)^w_expos
 * 
 *        where contrast, saturation and exposedness are three quality measures computed pixel-wise 
 *        at each exposure, and w_cont, w_sat and w_expos are three "weightig" exponents (see 
 *        setContrastExponent(), setSaturationExponent() and setExposednessExponent() ).
 *        Contrast is computed using the lapliacian operator,  saturation depends on the standard 
 *        deviation within the R, G and B channels, and exposedness is a gaussain weighting factor with 
 *        standard deviation sigma that depends on the pixel value.
 *        
 * 
 *        Multiresolution blending is then used to reduce the "seams" effect.
 * 
 *        By default, MertensHDR uses w_cont = w_sat = w_expos = 0 and sigma = 0.2 * pix_levels. 
 *        In case of single channel images, the saturation measure is not used, since it 
 *        depends on the color channel distribution.
 * 
 *        The template argument _TPixDepth represents the single or three channel pixel type 
 *        of the input LDR images. Currently, only uchar and ushort types are supported.
 */
template <typename _TPixDepth> class MertensHDR : public HDR<_TPixDepth>
{

public:

  MertensHDR();
  ~MertensHDR() {};

  /**
   * @brief Provide the current "weightig" exponent for the contrast quality measure
   */
  double contrastExponent() const { return contrast_exp_; };
  
  /**
   * @brief Provide the current "weightig" exponent for the saturation quality measure
   */  
  double saturationExponent() const{ return saturation_exp_; };

  /**
   * @brief Provide the current "weightig" exponent for the exposedness quality measure
   */  
  double exposednessExponent() const { return exposedness_exp_; };
  
  /**
   * @brief Set the "weightig" exponent for the contrast quality measure
   *
   * @param val The "weightig" exponent
   */  
  void setContrastExponent( double val ) { contrast_exp_ = val; };

  /**
   * @brief Set the "weightig" exponent for the saturation quality measure
   *
   * @param val The "weightig" exponent
   */  
  void setSaturationExponent( double val ) { saturation_exp_ = val; };

  /**
   * @brief Set the "weightig" exponent for the exposedness quality measure
   *
   * @param val The "weightig" exponent
   */  
  void setExposednessExponent( double val ) { exposedness_exp_ = val; };

  /**
   * @brief Provide the current standard deviation used to compute the gaussain weighting factor 
   *        that represents the exposedness quality measure
   */
  double exposednessSigma() const { return exposedness_sigma_; };

  /**
   * @brief Set the standard deviation used to compute the gaussain weighting factor 
   *        that represents the exposedness quality measure
   * 
   * @param sigma The standard deviation
   */
  void setExposednessSigma( double sigma );

  
  /**
   * @brief Construct a HDR image from a sequence of LDR images taken with different 
   *        exposure values using the Mertens algorithm. 
   *
   * @param imgs The sequence of input LDR images.
   * @param hdr_img Output HDR image
   * 
   * The input LDR images should be one or three channels with _TPixDepth pixel type. 
   * The output HDR image will have the same number of channels, with float pixel type.
   * The resulting HDR image may not require tonemapping before being processed.
   */
  virtual void merge ( const std::vector<cv::Mat> &imgs, cv::Mat &hdr_img ) const;

  /**
   * @brief Construct a HDR image from a sequence of LDR images taken with different 
   *        exposure values using the Mertens algorithm. 
   *
   * @param imgs The sequence of input LDR images.
   * @param exp_times The sequence of exposure times, not considered since Mertens algorithm
   *                  does not exploit times
   * @param hdr_img Output HDR image
   * 
   * The input LDR images should be one or three channels with _TPixDepth pixel type. 
   * The output HDR image will have the same number of channels, with float pixel type.
   * The resulting HDR image may not require tonemapping before being processed.
   */
  virtual void merge ( const std::vector<cv::Mat> &imgs, const std::vector<float> &exp_times,
                       cv::Mat &hdr_img ) const;
                       
private:
  
  void fillWeightsBorders( std::vector<cv::Mat> &weights, cv::Mat &weights_sum ) const;
  
  double exposedness_sigma_;
  std::vector<double> exposedness_weights_;
  
  double contrast_exp_, saturation_exp_, exposedness_exp_;
};



class Tonemap
{

/**
 * @brief Abstract base class for tonemapping classes. Any class inheriting from
 *        Tonemap must provide implementations for the compute() method, in order 
 *        to construct a remapped LDR image from an input, possibly HDR, image.
 *        Both the input and output image should be one or three channels 
 *        with float pixel type, the highest value of the resulting LDR images is 
 *        represented by the protected variable max_level_ 
 *        (see setMaxLevel() function)
 */
public:

  Tonemap() : max_level_( 255 ){};
  ~Tonemap() {};

  /**
   * @brief Any class inheriting from Tonemap must provide an implementation for this functions,
   *        in order to construct a remapped LDR image from an input, possibly HDR, image.
   *
   * @param hdr_img The input image.
   * @param hdr_img Output LDR image
   * 
   * Both the input and output image should be one or three channels 
   * with float pixel type, the highest value of the resulting LDR images is 
   * represented by the protected variable max_level_ 
   * (see setMaxLevel() function)
   */
  virtual void compute( const cv::Mat &hdr_img, cv::Mat &ldr_img ) const = 0;

  /**
   * @brief Provide the current highest value of the resulting LDR images
   */
  unsigned int maxLevel() const { return max_level_; };

  /**
   * @brief Set highest value of the resulting LDR images
   * 
   * @param max_level The pixel value
   */
  void setMaxLevel( unsigned int max_level ){ max_level_ = max_level; };  

  
protected:
  
  void checkData ( const cv::Mat &hdr_img, const cv::Mat &ldr_img ) const;
  unsigned int max_level_;
};

/**
 * @brief Implementation of a simple tonemap algorithm based on gamma correction, see for details:
 *         
 *        http://en.wikipedia.org/wiki/Gamma_correction
 */
class GammaTonemap : public Tonemap
{

public:

  GammaTonemap() {};
  ~GammaTonemap() {};

  /**
   * @brief Construct a remapped LDR image from an input, possibly HDR, image, using gamma correction.
   *
   * @param hdr_img The input image.
   * @param hdr_img Output LDR image
   * 
   * Both the input and output image should be one or three channels 
   * with float pixel type, the highest value of the resulting LDR images is 
   * represented by the protected variable max_level_ 
   * (see setMaxLevel() function)
   */
  virtual void compute( const cv::Mat &hdr_img, cv::Mat &ldr_img ) const;

  /**
   * @brief Provide the current exponent used in the gamma correction tonemapping
   */
  double gamma() const { return gamma_; };
  
  /**
   * @brief Set the exponent used in the gamma correction tonemapping
   * 
   * @param g The exponent
   */
  void setGamma( double g ) { gamma_ = g; };

private:
  
  double gamma_;
  
};

}