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

namespace cv_ext
{
  
/**
* @brief Draw a set of points in an image provided in input
*           The template type _TPoint2D represent a 2D point (e.g., cv::Point2d ).
* 
* @param[in,out] Img The image where the points will drawn 
* @param[in] pts Input points vector
* @param[in] color Points color
* @param[in] scale The input points coordinates will be scaled by this scale factor before drawing (e.g, scale*x, scale*y)
*/  
template < typename _TPoint2 > void drawPoints( cv::Mat &img, const std::vector< _TPoint2 > &pts,
                                                cv::Scalar color = cv::Scalar( 255,255,255 ),
                                                double scale = 1.0 );
/**
* @brief Draw a set of normals directions in an image provided in input
*        The template type _TPoint2D represent a 2D point (e.g., cv::Point2d ), _T the normals type 
*        (i.e., double or float)
* 
* @param[in,out] Img The image where the normals will drawn 
* @param[in] poses Input normal directions poses
* @param[in] normals_dirs Input points normals directions (it should be [-pi/2, pi/2])
* @param[in] dir_length Normals' segments length 
* @param[in] color Points/segment color
* @param[in] scale The input points coordinates will be scaled by this scale factor before drawing (e.g, scale*x, scale*y)
*/  
template < typename _TPoint2, typename _T > 
  void drawNormalsDirections( cv::Mat &img, const std::vector< _TPoint2 > &poses, 
                              const std::vector< _T > &normals_dirs, 
                              cv::Scalar color = cv::Scalar( 255,255,255 ),int normal_length = 10,
                              double scale = 1.0 );
  
/**
* @brief Draw a set of segments in an image provided in input
*        The template type _TVec4 represent a 4D vector (e.g., cv::Point2d ) with the segment
*        stored as x1 = vec[0], y1 = vec[2], x2 = vec[2] and y2 = vec[3]
* 
* @param[in,out] Img The image where the segments will drawn 
* @param[in] pts Input segments
* @param[in] color Segments color
* @param[in] tickness Tickness of the segments
* @param[in] scale The input segments coordinates will be scaled by this scale factor before drawing (e.g, scale*x, scale*y)
*/  
template <typename _TVec4> void drawSegments( cv::Mat &img, const std::vector< _TVec4 > &segments,
                                              cv::Scalar color = cv::Scalar( 255,255,255 ), 
                                              int tickness = 1, double scale = 1.0 );

/**
* @brief Draw a set of circles in an image provided in input
*        The template type _TPoint2D represent a 2D point (e.g., cv::Point2d ).
* 
* @param[in,out] Img The image where the circels will drawn 
* @param[in] pts Input points vector (i.e., the centers of the circles)
* @param[in] radius Radius of the circles.
* @param[in] color Points color
* @param[in] scale The input points coordinates will be scaled by this scale factor before drawing (e.g, scale*x, scale*y)
*/  
template < typename _TPoint2 > void drawCircles( cv::Mat &img, const std::vector< _TPoint2 > &pts,
                                                 int radius = 1, cv::Scalar color = cv::Scalar( 255,255,255 ),
                                                 double scale = 1.0 );

/**
* @brief Provide an image with depicted sample motion vectors of the input dense optical flow
* 
* @param[in] flow The dense optical flow matrix, with one motion vector for each pixel (2 channels, type float)
* @param[in] img Image to be used as background for the resulting optical flow image (e.g., the reference image 
*            from which the flow has been computed). It must have the same size as flow, with one or three 
*            channels.
* @param[in] step Sample step, in pixel (i.e, drawDenseOptFlow() will draw a motion vector each step pixel)
* @param[in] color Motion vectors color
* @param[in] mask If not empty, it must have the same size as flow, with one channel and type char.
*             The motion vectors are drawn only for non-zero pixels in the mask.
*/
cv::Mat drawDenseOptFlow ( const cv::Mat &flow, const cv::Mat &img, int step,
                           cv::Scalar color = cv::Scalar( 255,255,255 ), const cv::Mat &mask = cv::Mat() );

/**
* @brief Provide an image composed by two images placed side by side (or one above the other) with depicted
*        the matches between points computed using the input dense optical flow matrix
*
* @param[in] flow The dense optical flow matrix, with one motion vector for each pixel (2 channels, type float)
* @param[in] img0 The reference image from which the flow has been computed). It must have the same size as flow, with one or three
*            channels.
* @param[in] img1 The second image from which the flow has been computed. It must have the same size as flow, with one or three
*            channels.
* @param[in] step Sample step, in pixel (i.e, drawDenseOptFlow() will draw a motion vector each step pixel)
* @param[in] color Motion vectors color
* @param[in] side_by_side if true, the two input images are placed side by side, one above the other otherwise.
* @param[in] mask If not empty, it must have the same size as flow, with one channel and type char.
*             The motion vectors are drawn only for non-zero pixels in the mask.
*/
cv::Mat drawDenseOptFlow( const cv::Mat &flow, const cv::Mat &img0, const cv::Mat &img1,
                          int step, cv::Scalar color, bool side_by_side = true,
                          const cv::Mat &mask = cv::Mat() );
}