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

#include <string>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

namespace cv_ext
{
 
enum RotationFormat{ROT_FORMAT_MATRIX, ROT_FORMAT_AXIS_ANGLE};

/** @brief Check if a filename has .yaml or .yml extension, lowercase or uppercase. 
 *         Otherwise, it adds a ".yml" extension 
 * 
 * @param[in] filename A input string
 * 
 * @return The input string, with added the ".yml" extension if necessary
 */
std::string generateYAMLFilename( const std::string &filename );

/** @brief Read from file a 3D rigid body transformation, i.e. a rotation and a translation
 * 
 * @param[out] rotation A 3x1 rotation vector or a 3x3 rotation matrix, depending on the rf parameter
 * @param[out] translation A 3x1 translation vector 
 * @param[in] rf Rotation format
 *
 * @return true if successful, false otherwise
 *
 * If rf is set to ROT_FORMAT_AXIS_ANGLE (default), the rotation will returned by means a 3x1 rotation 
 * vector (in Axis-Angle representation), while if rf is set to ROT_FORMAT_MATRIX the rotation will 
 * returned by means a 3x3 rotation matrix
 */
bool read3DTransf( const std::string &filename, cv::Mat &rotation, cv::Mat &translation,
                   RotationFormat rf = ROT_FORMAT_AXIS_ANGLE );

/** @brief Read from a YAML::Node a 3D rigid body transformation, i.e. a rotation and a translation
 *
 * @param[in] in_node A YAML::Node containing the 3D transform data
 * @param[out] rotation A 3x1 rotation vector or a 3x3 rotation matrix, depending on the rf parameter
 * @param[out] translation A 3x1 translation vector
 * @param[in] rf Rotation format
 *
 * If rf is set to ROT_FORMAT_AXIS_ANGLE (default), the rotation will returned by means a 3x1 rotation
 * vector (in Axis-Angle representation), while if rf is set to ROT_FORMAT_MATRIX the rotation will
 * returned by means a 3x3 rotation matrix
 */
void read3DTransf( const YAML::Node &in_node, cv::Mat &rotation, cv::Mat &translation,
                   RotationFormat rf = ROT_FORMAT_AXIS_ANGLE );

/** @brief Write to file a 3D rigid body transformation, i.e. a rotation and a translation
 * 
 * @param[in] rotation A 3x1 or a 1x3 rotation vector, or a 3x3 rotation matrix
 * @param[in] translation A 3x1 or a 1x3 translation vector
 *
 * @return true if successful, false otherwise
 */
bool write3DTransf( const std::string &filename, const cv::Mat &rotation, const cv::Mat &translation );

/** @brief Write to YAML::Node a 3D rigid body transformation, i.e. a rotation and a translation
 *
 * @param[out] out_node A YAML::Node output of the read operation
 * @param[in] rotation A 3x1 or a 1x3 rotation vector, or a 3x3 rotation matrix
 * @param[in] translation A 3x1 or a 1x3 translation vector
 */
void write3DTransf( YAML::Node& out_node, const cv::Mat &rotation, const cv::Mat &translation );

}