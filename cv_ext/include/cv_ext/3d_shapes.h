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
#include <Eigen/Geometry>
#include "cv_ext/types.h"

namespace cv_ext
{
 
/**
  * @brief Align the selected rot_axis with the input vector vec and provide a
  *        quaternion that represents a rotation around this axis.
  *
  * @tparam _TPoint3D A 3D point type (compliant types defined in CV_EXT_3D_POINT_TYPES, see cv_ext/types.h)
  * 
  * @param[in] vec Input 3D point, i.e. the (direction) vector
  * @param[out] quat_rotation The quaternion representig the computed rotation
  * @param[in] ang The angle in radians used in the rotation around the selected axis
  * @param[in] rot_axis The coordindate axis to be aligned with the input vector
  * 
  * This function first compute the rotation to align the selected axis (rot_axis) with the 
  * input vector vec, then rotate around the selected axis.
  */
template < typename _TPoint3D > 
  void rotationAroundVector( const _TPoint3D &vec, Eigen::Quaterniond &quat_rotation, double ang, 
                              CoordinateAxis rot_axis = COORDINATE_Z_AXIS );
  
/**
  * @brief Align the selected rot_axis with the input vector vec and provide a set of 
  *        num_rotations quaternions that represents uniformly distributed rotations around this axis.
  *
  * @tparam _TPoint3D A 3D point type (compliant types defined in CV_EXT_3D_POINT_TYPES, see cv_ext/types.h)
  * 
  * @param[in] vec Input 3D point, i.e. the (direction) vector
  * @param[out] quat_rotations The set of quaternions representig the sampled rotations
  * @param[in] num_rotations Number of uniformly distributed rotations around the input vector
  * @param[in] rot_axis The coordindate axis to be aligned with the input vector
  * 
  * This function first compute the rotation to align the selected axis (rot_axis) with the 
  * input vector vec, then sample around the selected axis a number of uniformly distributed 
  * num_rotations rotations.
  */
template < typename _TPoint3D > 
  void sampleRotationsAroundVector( const _TPoint3D &vec, vector_Quaterniond &quat_rotations, 
                                    int num_rotations, CoordinateAxis rot_axis = COORDINATE_Z_AXIS );
  

/**
  * @brief Align the selected rot_axis with the input vectors vecs and provide a set of 
  *        num_rotations quaternions that represents uniformly distributed rotations around each aligned axis.
  * 
  * @tparam _TPoint3D A 3D point type (compliant types defined in CV_EXT_3D_POINT_TYPES, see cv_ext/types.h)
  * 
  * @param[in] vecs Input 3D points, i.e. the (direction) vectors
  * @param[out] quat_rotations The set of quaternions representig the sampled rotations
  * @param[in] num_rotations Number of uniformly distributed rotations around each input vector
  * @param[in] rot_axis The coordindate axis to be aligned with the input vectors
  * 
  * This function first compute the rotation to align the selected axis (rot_axis) with each of the 
  * input vectors vecs, then sample around the selected axis a number of uniformly distributed
  * num_rotations rotations.
  */
template < typename _TPoint3D > 
  void sampleRotationsAroundVectors( const std::vector< _TPoint3D > &vecs, 
                                     vector_Quaterniond &quat_rotations, 
                                     int num_rotations, CoordinateAxis rot_axis = COORDINATE_Z_AXIS );

  /**
  * @brief Align the selected rot_axis with the input vector vec and provide a set of 
  *        quaternions that represents uniformly distributed rotations around this axis
  *        from angle ang0 to angle ang1.
  *
  * @tparam _TPoint3D A 3D point type (compliant types defined in CV_EXT_3D_POINT_TYPES, see cv_ext/types.h)
  * 
  * @param[in] vec Input 3D point, i.e. the (direction) vector
  * @param[out] quat_rotations The set of quaternions representig the sampled rotations
  * @param[in] ang0 Start angle: the angle in radians from which rotations around the selected axis begin
  * @param[in] ang1 End angle: the last angle in radians of the rotations around the selected axis
  * @param[in] ang_step Angle step between rotations
  * @param[in] rot_axis The coordindate axis to be aligned with the input vector
  * 
  * This function first compute the rotation to align the selected axis (rot_axis) with the 
  * input vector vec, then sample around the selected axis a number of uniformly distributed 
  * num_rotations rotations.
  */
template < typename _TPoint3D > 
  void sampleRotationsAroundVector( const _TPoint3D &vec, vector_Quaterniond &quat_rotations, 
                                    double ang0, double ang1, double ang_step,
                                    CoordinateAxis rot_axis = COORDINATE_Z_AXIS );
  

/**
  * @brief Align the selected rot_axis with the input vectors vecs and provide a set of 
  *        quaternions that represents uniformly distributed rotations around each aligned axis,
  *        from angle ang0 to angle ang1.
  * 
  * @tparam _TPoint3D A 3D point type (compliant types defined in CV_EXT_3D_POINT_TYPES, see cv_ext/types.h)
  * 
  * @param[in] vecs Input 3D points, i.e. the (direction) vectors
  * @param[out] quat_rotations The set of quaternions representig the sampled rotations
  * @param[in] ang0 Start angle: the angle in radians from which rotations around the selected axis begin
  * @param[in] ang1 End angle: the last angle in radians of the rotations around the selected axis
  * @param[in] ang_step Angle step between rotations
  * @param[in] rot_axis The coordindate axis to be aligned with the input vectors
  * 
  * This function first compute the rotation to align the selected axis (rot_axis) with each of the 
  * input vectors vecs, then sample around the selected axis a number of uniformly distributed
  * num_rotations rotations.
  */
template < typename _TPoint3D > 
  void sampleRotationsAroundVectors( const std::vector< _TPoint3D > &vecs, 
                                     vector_Quaterniond &quat_rotations, 
                                     double ang0, double ang1, double ang_step,
                                     CoordinateAxis rot_axis = COORDINATE_Z_AXIS );


/**
  * @brief Construct a spherical point cloud with points uniformly distributed in the 
  *        whole surface.
  * 
  * @tparam _TPoint3D A 3D point type (compliant types defined in CV_EXT_3D_POINT_TYPES, see cv_ext/types.h)
  * 
  * @param[out] sphere_points Output 3D cloud
  * @param[in] n_iter Num splitting iterations, if n_iter is 0, a basic, 12 vertices icosphere is returned
  * @param[in] radius Sphere radius
  * 
  * @return The the actual angle step (in radians) between points of the resulting icosphere 
  * 
  * The sphere ("icosphere") is constructed starting from a an icosahedron 
  * (i.e., a polyhedron with 20 triangular faces, 30 edges and 12 vertices, see:
  *  http://en.wikipedia.org/wiki/Icosahedron ) by recursively splitting each triangle 
  * into 4 smaller triangles for n_iter iterations.
  */
template < typename _TPoint3D > 
  void createIcosphere( std::vector<_TPoint3D> &sphere_points, int n_iter, double radius = 1.0 );

/**
  * @brief Construct a spherical point cloud with points uniformly distributed in the 
  *        whole surface.
  * 
  * @tparam _TPoint3D A 3D point type (compliant types defined in CV_EXT_3D_POINT_TYPES, see cv_ext/types.h)
  * 
  * @param[out] sphere_points Output 3D cloud
  * @param[in] n_iter Num splitting iterations, if n_iter is 0, a basic, 12 vertices icosphere is returned
  * @param[out] iter_num_pts Vector of n_iter + 1 elements that provides, for each splitting iteration, 
  *                          the number of added points
  * @param[out] iter_ang_steps Vector of n_iter + 1 elements that provides, for each splitting iteration, 
  *                            the angle step between neighbours points for such iteration
  * @param[in] radius Sphere radius
  * 
  * @return The the actual angle step (in radians) between points of the resulting icosphere 
  * 
  * The sphere ("icosphere") is constructed starting from a an icosahedron 
  * (i.e., a polyhedron with 20 triangular faces, 30 edges and 12 vertices, see:
  *  http://en.wikipedia.org/wiki/Icosahedron ) by recursively splitting each triangle 
  * into 4 smaller triangles for n_iter iterations.
  */
template < typename _TPoint3D > 
  void createIcosphere( std::vector<_TPoint3D> &sphere_points, int n_iter, 
                        std::vector<int> &iter_num_pts, std::vector<double> &iter_ang_steps,
                        double radius = 1.0 );
  
/**
  * @brief Construct a spherical point cloud with points uniformly distributed in the 
  *        whole surface. 
  * 
  * @tparam _TPoint3D A 3D point type (compliant types defined in CV_EXT_3D_POINT_TYPES, see cv_ext/types.h)
  * 
  * @param[out] sphere_points Output 3D cloud
  * @param[in] max_ang_step Max angle step (in radians) between points of the icosphere
  * @param[in] radius Sphere radius
  * 
  * @return The the actual angle step (in radians) between points of the resulting icosphere 
  * 
  * The sphere ("icosphere") is constructed starting from a an icosahedron 
  * (i.e., a polyhedron with 20 triangular faces, 30 edges and 12 vertices, see:
  *  http://en.wikipedia.org/wiki/Icosahedron ) by recursively splitting each triangle 
  * into 4 smaller triangles. The number of splitting iterations is computed such as the 
  * the angle step between icosphere points is no greater than max_ang_step.
  */
template < typename _TPoint3D > 
  double createIcosphereFromAngle( std::vector<_TPoint3D> &sphere_points, double max_ang_step, double radius = 1.0 );
  
/**
  * @brief Construct the spherical polar caps point cloud with 
  *        points uniformly distributed in the whole surface. 
  *        
  * 
  * 
  * @tparam _TPoint3D A 3D point type (compliant types defined in CV_EXT_3D_POINT_TYPES, see cv_ext/types.h)
  * 
  * @param[out] cap_points Output 3D cloud
  * @param[in] max_ang_step Max angle step (in radians) between points of the icosphere
  * @param[in] latitude_angle Cutting absolute latitude angle, in radians.
  * @param[in] radius Sphere radius
  * @param[in] only_north_cap If true, create only the upper pole of the sphere
  * 
  * @return The the actual angle step (in radians) between points of the provided polar cap/s
  * 
  * See createIcosphere() and createIcosphereFromAngle() for further details about the icosphere
  * \note If latitude_angle is 0, createIcospherePolarCap() provide a complete icosphere. If 
  * latitude_angle is pi/2, createIcospherePolarCapFromAngle() provides zero or just two points in the poles (or only
  * one if only_north_cap is set to true), depending on the required max angle step.
  */
template < typename _TPoint3D > 
  double createIcospherePolarCapFromAngle( std::vector<_TPoint3D> &cap_points, double max_ang_step, 
                                           double latitude_angle, double radius = 1.0, bool only_north_cap = false );
}
