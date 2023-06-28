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

#include "cv_ext/macros.h"

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <random>

namespace cv_ext
{

static std::random_device rand_dev;
static std::mt19937 engine ( rand_dev() );

/**
 * @brief Draw a value from a Gaussian probability density
 *
 * @param[in] mean Density mean
 * @param[in] stddev Density standard deviation
 *
 * @return The sampled value
 * */
static inline double sampleGaussian( double mean, double stddev )
{
  std::normal_distribution<double> norm_distrib(mean, stddev);
  return norm_distrib(engine);
}

/**
 * @brief Draw a value from a uniform probability density
 *
 * @param[in] min Minimum value of the uniform distribution
 * @param[in] max Maximum value of the uniform distribution
 *
 * @return The sampled value
 * */
static inline double sampleUniform( double min, double max )
{
  std::uniform_real_distribution<> unif_distrib(min, max); ;
  return unif_distrib(engine);
}

/**
 * @brief Simple but efficient function to round a positive real value to nearest integer value
 *
 * @tparam _T Data type (e.g., float, double, ...)
 * 
 * @param[in] val The value to be rounded, it should be >= 0
 * @return The rounded value
 *
 * \note For negative input values, the returned results will be wrong
 */
template < typename _T > inline int roundPositive ( _T val )
{
  return ((int)(val + 0.5f));
}

/**
 * @brief Simple but efficient function to round a negative real value to nearest integer value
 *
 * @tparam _T Data type (e.g., float, double, ...)
 * 
 * @param[in] val The value to be rounded, it should be <= 0
 * @return The rounded value
 *
 * \note For positive input values, the returned results will be wrong
 */
template < typename _T > inline int roundNegative ( _T val )
{
  return ((int)(val - 0.5f));
}

/**
 * @brief BUG Returns an approximation ot the arc tangent of x
 *
 * @param[in] x Value whose arc tangent is computed.
 * @return Principal arc tangent of x (approximatred)
 *
 * This is a fast arctan approximation taken from the paper:
 *
 * Rajan, S. Sichun Wang Inkol, R. Joyal, A., "Efficient approximations for the arctangent function", May 2006
 *
 * The code comes from the webpage: http://nghiaho.com/?p=997
 */
// inline double fastAtan( double x )
// {
//   // 0.78539816339744830962 =~ M_PI/4
//   return 0.7853 * x - x * ( fabs(x) - 1) * (0.2447 + 0.0663*fabs(x) );
// }


/**
 * @brief Normalize a general input angle between -M_PI and M_PI
 *
 * @tparam _T Data type (e.g., float, double, ...)
 * 
 * @param[in] ang The input angle
 * @return The normalized angle, [ -M_PI, M_PI )
 */
template < typename _T > inline _T normalizeAngle ( _T ang )
{
  if ( ang >= _T ( -M_PI ) && ang < _T ( M_PI ) )
    return ang;

  _T multiplier = floor ( ang / ( _T ( 2.0*M_PI ) ) );
  ang = ang - _T ( 2*M_PI ) *multiplier;
  if ( ang >= _T ( M_PI ) )
    ang = ang - _T ( 2.0*M_PI );

  if ( ang < _T ( -M_PI ) )
    ang = ang + _T ( 2.0*M_PI );

  return ang;
}

/**
 * @brief Provide the L2-norm of a 2D point
 *
 * @tparam _TPoint2D The 2D point type (e.g., cv::Point2d ).  It should have x and y fields, 
 *                   and basic constructors like _TPoint2D(_T x, _T y) where _T represents the 
 *                   data type (e.g., float, double, ...)
 * 
 * @param[in] pt The input point
 * @return The L2-norm
 */
template<typename _TPoint2D> static inline
double norm2D(const _TPoint2D& pt)
{
  return std::sqrt( double(pt.x)*pt.x + double(pt.y)*pt.y );
}

/**
 * @brief Provide the L2-norm of a 3D point
 *
 * @tparam _TPoint3D The 3D point type (e.g., cv::Point3d ).  It should have x, y and z fields, 
 *                   and basic constructors like _TPoint2D(_T x, _T y, _T z) where _T represents the 
 *                   data type (e.g., float, double, ...)
 * 
 * @param[in] pt The input point
 * @return The L2-norm
 */
template<typename _TPoint3D> static inline
double norm3D(const _TPoint3D& pt)
{
  return std::sqrt( double(pt.x)*pt.x + double(pt.y)*pt.y + double(pt.z)*pt.z );
}

/**
 * @brief Normalize a input 2D point to be in the unit sphere
 *
 * @tparam _TPoint2D The 2D point type (e.g., cv::Point2d ).  It should have x and y fields, 
 *                   and basic constructors like _TPoint2D(_T x, _T y) where _T represents the 
 *                   data type (e.g., float, double, ...)
 * 
 * @param[in] src_pt The input point
 * 
 * @return The normalized point
 */
template<typename _TPoint2D> static inline
  _TPoint2D normalize2DPoint(const _TPoint2D& src_pt )
{
  _TPoint2D dst_pt;
  double eta = 1.0/cv_ext::norm2D(src_pt);
  dst_pt.x = eta*src_pt.x;
  dst_pt.y = eta*src_pt.y;
  return dst_pt;
}

/**
 * @brief Normalize a input 3D point to be in the unit sphere
 *
 * @tparam _TPoint3D The 3D point type (e.g., cv::Point3d ).  It should have x, y and z fields, 
 *                   and basic constructors like _TPoint2D(_T x, _T y, _T z) where _T represents the 
 *                   data type (e.g., float, double, ...)
 * 
 * @param[in] src_pt The input point
 * 
 * @return The normalized point
 */
template<typename _TPoint3D> static inline
  _TPoint3D normalize3DPoint(const _TPoint3D& src_pt )
{
  _TPoint3D dst_pt;
  double eta = 1.0/cv_ext::norm3D(src_pt);
  dst_pt.x = eta*src_pt.x;
  dst_pt.y = eta*src_pt.y;
  dst_pt.z = eta*src_pt.z;
  return dst_pt;
}

/**
 * @brief This function computes the distance in radians between two rotations represented by 3x3 rotation matrices
 *
 * @tparam _T Data type (e.g., float, double, ...)
 * 
 * @param[in] r_mat0 Orthogonal 3x3 matrix representing the first rotation
 * @param[in] r_mat1 Orthogonal 3x3 matrix representing the second rotation
 * 
 * @return The computed distance
 */
template<typename _T> inline
  _T rotationDist( const Eigen::Matrix<_T, 3, 3> &r_mat0, const Eigen::Matrix<_T, 3, 3> &r_mat1 )
{
  _T trace = (r_mat0*r_mat1.transpose()).trace();
  if( trace > 3.0 )
    trace = 3.0;
  return acos( 0.5*( trace - 1.0 ) );
}

/**
 * @brief This function computes the distance in radians between two rotations represented by 3x3 rotation matrices
 * *
 * @param[in] r_mat0 Orthogonal 3x3 openCV matrix representing the first rotation
 * @param[in] r_mat1 Orthogonal 3x3 openCV matrix representing the second rotation
 *
 * @return The computed distance
 */
static inline double rotationDist( const cv::Mat r_mat0, const cv::Mat r_mat1 )
{
  cv_ext_assert( r_mat0.rows == 3 && r_mat0.cols == 3);
  cv_ext_assert( r_mat1.rows == 3 && r_mat1.cols == 3);

  double tr = cv::trace(r_mat0*r_mat1.t())[0];
  if( tr > 3.0 )
    tr = 3.0;

  return acos( 0.5*( tr - 1.0 ) );
}

}
