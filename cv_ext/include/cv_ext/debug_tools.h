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
#include <Eigen/Geometry>
#include <limits.h>
#include <cstdlib>

#include "cv_ext/types.h"

namespace cv_ext
{
enum SpecialKeys
{
  KEY_SPACE = 32,
  KEY_ENTER = 10,
  KEY_BS = 65288,
  KEY_CANC = 65535,
  KEY_ESCAPE = 27,
  KEY_UP = 65362,
  KEY_DOWN = 65364,
  KEY_RIGHT = 65363,
  KEY_LEFT = 65361,
  KEY_PAGE_UP = 65365,
  KEY_PAGE_DOWN = 65366,
  KEY_F1 = 65470,
  KEY_F2 = 65471,
  KEY_F3 = 65472,
  KEY_F4 = 65473,
  KEY_F5 = 65474,
  KEY_F6 = 65475,
  KEY_F7 = 65476,
  KEY_F8 = 65477,
  KEY_F9 = 65478,
  KEY_F10 = 65479,
  KEY_F11 = 65480,
  KEY_F12 = 65481,
  KEY_HOME = 65360,
  KEY_END = 65367
};

/**
* @brief Waits for a pressed key, same as cv::waitKey() but fixed
*
* @param[in] delay Delay in milliseconds. 0 is the special value that means "forever"
*
* @returns The code of the pressed key (e.g., an ascii character or a special character,
*          see SpecialKeys= or -1 if no key was pressed before the specified time had elapsed.
*/
int waitKeyboard( int delay = 0 );

/**
* @brief Provides a random RGB color.
*
* @returns A 3D random vector representing the 3 color components.
*/
cv::Vec3b randRGBColor();

/**
* @brief Map an scalar value into an RGB color, creating a "rainbow like" color gradient
*        from blue (smallest value) to red (biggest value)
*
* @tparam _T Type of the value to be mapped
* 
* @param[in] val The value to be mapped
* @param[in] max_val The maximum allowed value. All values gratest than max_val will
*                    be mapped to the same color as max_val
*
* @returns A 3D vector representing the 3 color components, in the BGR color format used by opencv
*/
template < typename _T > cv::Vec3b mapValue2RGB( _T val, _T max_val );

/**
* @brief Map a single channel image into an RGB image, creating a "rainbow like" color gradient
*        from blue (smallest value) to red (biggest value)
*
* @tparam _T Type of the single channel image be mapped
* 
* @param[in] src_mat The simgle channel image to be mapped
* @param[out] dst_img An output BGR image
* @param[in] max_val The maximum allowed value. All values gratest than max_val will
*                    be mapped to the same color as max_val
* 
* This function internally uses the functon mapValue2RGB()
*/
template < typename _T > void mapMat2RGB( const cv::Mat &src_mat, cv::Mat &dst_img, double max_val );

/**
* @brief Simple image viewer
* 
* @param[in] img The input image
* @param[in] win_name Window names
* @param[in] normalize If true, normalize each channel of the image between 0 and 255 (i.e., min pix value -> 0, 
*                  max pixel value -> 255)
* @param[in] sleep If sleep == 0, block the application and wait for escape key press. If sleep > 0, wait for
*                  block the application for sleep milliseconds
*/
void showImage ( const cv::Mat &img, const std::string &win_name = "", bool normalize = true, int sleep = 0 );

/**
 * @brief Create an image composed by a regular grid of sub-images (cells)
 *
 * @param[in] grid_size The grid size
 * @param[in] cell_size The size of each sub-image (cells) in the grid
 * @param[in] type Desired output matrix type
 * @param[out] cells A vector of cv::Mat each one representing a cell of the grid, in row-major order
 *
 *
* @return The cv::Mat representing the image grid
*/
cv::Mat createImageGrid(cv::Size grid_size, cv::Size cell_size, int type, std::vector< cv::Mat > &cells );

/**
* @brief Simple point cloud viewer. 
* 
* @tparam _TPoint3D Type of the 3D point (compliant types defined in CV_EXT_3D_POINT_TYPES, 
*                   see cv_ext/types.h)
*
* @param[in] points Input 3D points
* @param[in] win_name Window names
* @param[in] show_axes Show x,y,z axes
* @param[in] axes_transf Axes transformation matrixes vector
* 
* \note If show_axes is true and axes_transf is empty, show a single coordinate system in the origin.
*       If show_axes is true and axes_transf is not empty, show a coordinate system for each transformation 
*/
template < typename _TPoint3D >
  void show3DPoints ( const std::vector<_TPoint3D> &points, 
                      const std::string &win_name = "", bool show_axes = false, 
                      const vector_Isometry3d &axes_transf = vector_Isometry3d() );
}
