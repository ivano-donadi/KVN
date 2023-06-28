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

#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <boost/concept_check.hpp>
#include <cv_ext/interpolations.h>


namespace cv_ext
{
/**
 * @brief Provides a vector of the points that belong to a line segment 
 *        connecting two points. The line is computed using the
 *        bresenham algorithm, with tickness = 1
 * 
 * @param[in] p0 Line segment start point
 * @param[in] p1 Line segment end point
 * @param[out] line_pts Output line points
 */  
void getLinePoints( const cv::Point &p0, const cv::Point &p1, std::vector<cv::Point> &line_pts );

/**
 * @brief Object that represents a "directional" integral image of a single channel input image,
 *        along a specified direction (orientation) d, where d should be [-M_PI/2, M_PI/2).
 *        In a directional integral an image each pixel (x,y) represents the sum of all previous pixel
 *        of the line with direction d the pixel (x,y) belongs to.
 *        This object can be used to quickly compute the sum of values in segments of given direction
 * .
 *        Typenames _Tin and _Tout represent the input image and integral image pixel depths, respectively
 *        (e.g., _Tin can be uchar, float, double, ..., see CV_EXT_PIXEL_DEPTH_TYPES in cv_ext/types.h,
 *               _Tout can be uint32_t, uint64_t, float or double )
 */
template <typename _Tin, typename _Tout> class DirectionalIntegralImage
{
public:

 /**
  * @brief Constructor that internally builds the directional integral image with size (width+1, hight+1),
  *        where (width, hight) the size of the input image.
  *
  * @param[in] src_img Input, source image
  * @param[in] direction Direction (in image reference frame) used to compute the inegral image
  *                      (it should be [-M_PI/2, M_PI/2), but a normalization is performed internally)
  *
  * \note The integral image is computed following a line computed using the
  * bresenham algorithm, see getLinePoints() function
  */
  DirectionalIntegralImage( const cv::Mat& src_img, double direction = 0 );
  ~DirectionalIntegralImage(){};
  
 /**
  * @brief Provides the direction associated with the directional integral image
  */
  double direction() const { return direction_; };

 /**
  * @brief Provides the line points offsets to be used to compute the summations
  *
  * @param[out] line_pts Output line points
  */
  void getLinePattern( std::vector<cv::Point> &pattern ) const;

 /**
  * @brief Return true if the line pattern used to compute the summations mainly traverses the x axis,
  *        otherwise it mainly traverses the x axis
  *
  * If true, the direction is in [-M_PI/, M_PI/4), otherwise the direction is outside this interval.
  */
  bool isXMajor() const{ return x_major_; };

 /**
  * @brief Return the sum of values in a segment (along a specific direction, see direction())
  *        computed using the directional integral image.
  *
  * @param[in] p0 Segment first point
  * @param[in] p1 Segment second point
  *
  * One of the two points (the one with the smaller x or y coordinate, it depends on direction, see isXMajor())
  * will be prometed as segment starting point. The end point will be computed as a sort of "nearest neighbour" of the
  * other points along a specific direction (see direction())
  *
  * The template type _TPoint2D represents the point coordinates type (e.g., cv::Point2i ).
  * It should have x and y fields.
  * \note Floating point coordinates will be converted to integer by truncation
  */
  template < typename _TPoint2D > inline _Tout getSegmentSum( _TPoint2D &p0, _TPoint2D &p1 ) const;

 /**
  * @brief Return the sum of values in a segment (along a specific direction, see direction())
  *        computed using the directional integral image.
  *
  * @param[in] line_seg Four components vector, containing the coordinates (x0,y0) of the first segment point
  *                     followed by the coordinates (x1,y1) of the second segment point
  *
  * One of the two points (the one with the smaller x or y coordinate, it depends on direction, see isXMajor())
  * will be prometed as segment starting point. The end point will be computed as a sort of "nearest neighbour" of the
  * other points along a specific direction (see direction())
  *
  * The template type _TVec4 represents the 4D vector type (e.g., cv::Vec4i ).
  * It should have the operator [].
  * \note Floating point coordinates will be converted to integer by truncation
  */
  template < typename _TVec4 > inline _Tout getSegmentSum( const _TVec4 &line_seg ) const;

 /**
  * @brief Return the sum of values in a segment (along a specific direction, see direction())
  *        computed using the directional integral image.
  *
  * @param[in] p0_x X coordinate of the first segment point
  * @param[in] p0_y Y coordinate of the first segment point
  * @param[in] p1_x X coordinate of the second segment point
  * @param[in] p1_y Y coordinate of the second segment point
  *
  * One of the two points (the one with the smaller x or y coordinate, it depends on direction, see isXMajor())
  * will be prometed as segment starting point. The end point will be computed as a sort of "nearest neighbour" of the
  * other points along a specific direction (see direction())
  *
  * The template type _T represents the coordinates type (e.g., uint32_t ).
  * \note Floating point coordinates will be converted to integer by truncation
  */
  template < typename _T > inline _Tout getSegmentSum( _T p0_x, _T p0_y, _T p1_x, _T p1_y ) const;

 /**
  * @brief Return the sum of values in a segment (along a specific direction, see direction())
  *        computed using the directional integral image.
  *
  * @param[in] p0 Segment start segment point
  * @param[in] len Segment length
  *
  * The end point will be computed along a specific direction (see direction())
  *
  * The template type _TPoint2D represents the point coordinates type (e.g., cv::Point2i ).
  * It should have x and y fields.
  * \note Floating point coordinates will be converted to integer by truncation
  */
  template < typename _TPoint2D > inline _Tout getSegmentSum( _TPoint2D p0, int len ) const;

 /**
  * @brief Return the sum of values in a segment (along a specific direction, see direction())
  *        computed using the directional integral image.
  *
  * @param[in] p0_x X coordinate of the start segment point
  * @param[in] p0_y Y coordinate of the start segment point
  * @param[in] len Segment length
  *
  * The end point will be computed along a specific direction (see direction())
  *
  * The template type _T represents the coordinates type (e.g., uint32_t ).
  * \note Floating point coordinates will be converted to integer by truncation
  */
  template < typename _T > inline _Tout getSegmentSum( _T p0_x, _T p0_y, int len ) const;

private:

  //! Core function used to compoute the directional integral image
  void computeIntegralImage( const cv::Mat& src_img );

  //! The directional integral image
  cv::Mat int_img_;
  //! Input image size
  int w_, h_;
  //! Offset used for image look-up
  int offset_x_, offset_y_;
  //! Direction (in image reference frame) used to compute the inegral image
  double direction_;
  //! True if the line pattern used to compute the summations mainly traverses the x axis
  bool x_major_;
  //! The line points offset to be used to compute the summations
  std::vector<cv::Point> line_pattern_;
  //! Look-up table used to quickly retrive the x or y coordinates of the segments' second point
  std::vector<int> lut_;
};

// Some implementations

template < typename _Tin, typename _Tout > template < typename _TPoint2D > inline
  _Tout DirectionalIntegralImage<_Tin, _Tout >::getSegmentSum( _TPoint2D &p0, _TPoint2D &p1 ) const
{
  return getSegmentSum( p0.x, p0.y, p1.x, p1.y );
}

template < typename _Tin, typename _Tout > template < typename _TVec4 > inline
  _Tout DirectionalIntegralImage<_Tin, _Tout >::getSegmentSum( const _TVec4 &line_seg ) const
{
  return getSegmentSum( line_seg[0], line_seg[1], line_seg[2], line_seg[3] );
}

template < typename _Tin, typename _Tout > template < typename _T > inline
  _Tout DirectionalIntegralImage<_Tin, _Tout >::getSegmentSum( _T p0_x, _T p0_y, _T p1_x, _T p1_y ) const
{
  if( x_major_ )
  {
    if( p1_x > p0_x )
    {
      p0_y += offset_y_;
      p1_x++;
      p1_y = p0_y + lut_[p1_x] - lut_[p0_x];
      return int_img_.at<_Tout> ( p1_y, p1_x ) - int_img_.at<_Tout> ( p0_y, p0_x);
    }
    else
    {
      p1_y += offset_y_;
      p0_x++;
      p0_y = p1_y + lut_[p0_x] - lut_[p1_x];
      return int_img_.at<_Tout> ( p0_y, p0_x) - int_img_.at<_Tout> ( p1_y, p1_x );
    }
  }
  else
  {
    if( p1_y > p0_y )
    {
      p0_x += offset_x_;
      p1_y++;
      p1_x = p0_x + lut_[p1_y] - lut_[p0_y];
      return int_img_.at<_Tout> ( p1_y, p1_x ) - int_img_.at<_Tout> ( p0_y, p0_x);
    }
    else
    {
      p1_x += offset_x_;
      p0_y++;
      p0_x = p1_x + lut_[p0_y] - lut_[p1_y];
      return int_img_.at<_Tout> ( p0_y, p0_x) - int_img_.at<_Tout> ( p1_y, p1_x );
    }
  }
}

template < typename _Tin, typename _Tout > template < typename _TPoint2D > inline
  _Tout DirectionalIntegralImage<_Tin, _Tout >::getSegmentSum( _TPoint2D p0, int len ) const
{
  return getSegmentSum( p0.x, p0.y, len );
}

template < typename _Tin, typename _Tout > template < typename _T > inline
  _Tout DirectionalIntegralImage<_Tin, _Tout >::getSegmentSum( _T p0_x, _T p0_y, int len ) const
{
  if( x_major_ )
  {
    p0_y += offset_y_;
    _T p1_x = p0_x + len, p1_y = p0_y + lut_[p1_x] - lut_[p0_x];
    return int_img_.at<_Tout> ( p1_y, p1_x ) - int_img_.at<_Tout> ( p0_y, p0_x);
  }
  else
  {
    p0_x += offset_x_;
    _T p1_y = p0_y + len, p1_x = p0_x+ lut_[p1_y] - lut_[p0_y];
    return int_img_.at<_Tout> ( p1_y, p1_x ) - int_img_.at<_Tout> ( p0_y, p0_x);
  }
}
}
