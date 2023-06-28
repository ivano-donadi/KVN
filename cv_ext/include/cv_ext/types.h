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

#include <stdint.h>
#include <vector>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#ifdef CV_EXT_USE_PCL
#include <pcl/point_types.h>
#endif

#define CV_EXT_REAL_TYPES \
  (double)                \
  (float)

#define CV_EXT_UINT_TYPES \
  (uint8_t)               \
  (uint16_t)              \
  (uint32_t)              \
  (uint64_t)

#define CV_EXT_PIXEL_DEPTH_TYPES \
  CV_EXT_UINT_TYPES  \
  CV_EXT_REAL_TYPES
  
#define CV_EXT_2D_REAL_POINT_TYPES \
  (cv::Point2f)                    \
  (cv::Point2d)

#ifdef CV_EXT_USE_PCL
#define CV_EXT_3D_REAL_POINT_TYPES \
(cv::Point3f)                    \
(cv::Point3d)                      \
PCL_XYZ_POINT_TYPES
#else
#define CV_EXT_3D_REAL_POINT_TYPES \
(cv::Point3f)                    \
(cv::Point3d)
#endif

#define CV_EXT_2D_INT_POINT_TYPES \
  (cv::Point2i)

#define CV_EXT_3D_INT_POINT_TYPES \
(cv::Point3i)

#define CV_EXT_2D_POINT_TYPES \
  CV_EXT_2D_REAL_POINT_TYPES  \
  CV_EXT_2D_INT_POINT_TYPES
  
#define CV_EXT_3D_POINT_TYPES \
  CV_EXT_3D_REAL_POINT_TYPES  \
  CV_EXT_3D_INT_POINT_TYPES

#define CV_EXT_4D_VECTOR_TYPES \
  (cv::Vec4i)                  \
  (cv::Vec4f)                  \
  (cv::Vec4d)                  \
  (Eigen::Vector4i)            \
  (Eigen::Vector4f)            \
  (Eigen::Vector4d)


#define CV_EXT_INSTANTIATE_MACRO(r, NAME, TYPE) \
  BOOST_PP_CAT(CV_EXT_INSTANTIATE_, NAME)(TYPE)

/** @brief Provide explicit instantiations of some templates for the types given in the 
 *         sequence TYPES_SEQ, to be called at the end of the function/class definitions 
 *         (e.g., at the end of the .cpp file)
 * 
 * When writing:
 * 
 * CV_EXT_INSTANTIATE( ClassName, CV_EXT_FLOATING_POINT_TYPES )
 * 
 * the macro will expand:
 * 
 *  CV_EXT_INSTANTIATE_ClassName( float )
 *  CV_EXT_INSTANTIATE_ClassName( double )
 * 
 * so CV_EXT_INSTANTIATE_ClassName should be defined somewhere, before CV_EXT_INSTANTIATE(...), 
 * e.g. :
 * 
 * CV_EXT_INSTANTIATE_ClassName( TYPE ) \
 *   template class ClassName< TYPE >;
 * 
 *  (the semicolon at the end of the macro is mandatory, due to BOOST_PP_SEQ_FOR_EACH(...) )
 */
#define CV_EXT_INSTANTIATE(NAME, TYPES_SEQ )                        \
  BOOST_PP_SEQ_FOR_EACH(CV_EXT_INSTANTIATE_MACRO, NAME, TYPES_SEQ )


namespace cv_ext
{
enum CoordinateAxis { COORDINATE_X_AXIS, COORDINATE_Y_AXIS, COORDINATE_Z_AXIS };
enum InterpolationType { INTERP_NEAREST_NEIGHBOR, INTERP_BILINEAR, INTERP_BICUBIC };
enum WeightFunctionType { TRIANGLE_WEIGHT, GAUSSIAN_WEIGHT };

typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > vector_Vector3f;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vector_Vector3d;
typedef std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > vector_Vector4f;
typedef std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > vector_Vector4d;
typedef std::vector<Eigen::Affine3f, Eigen::aligned_allocator<Eigen::Affine3f> > vector_Affine3f;
typedef std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d> > vector_Affine3d;
typedef std::vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f> > vector_Isometry3f;
typedef std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> > vector_Isometry3d;
typedef std::vector<Eigen::AngleAxisf, Eigen::aligned_allocator<Eigen::AngleAxisf> > vector_AngleAxisf;
typedef std::vector<Eigen::AngleAxisd, Eigen::aligned_allocator<Eigen::AngleAxisd> > vector_AngleAxisd;
typedef std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf> > vector_Quaternionf;
typedef std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > vector_Quaterniond;

/** @brief Template class specifying an user defined i.

@tparam _T The domain (i.e., basics data type) where the interval is defined 
*/
template < typename _T > class Interval
{
public:
  
  Interval(){};
  Interval( _T start, _T end): start(start), end(end){};
  _T size() const { return end - start; };
  /** @brief This method returns true if the object defines an empty, zero size interval */
  inline bool empty() const { return start == end; };
  /** @brief This method returns true if the object defines an interval that includes the whole domain fo the type _T */
  inline bool full() const 
  { 
    return start == std::numeric_limits< _T >::lowest() && end == std::numeric_limits< _T >::max();
  };
  /** @brief This method sets an interval to define the whole domain fo type _T */
  inline void setFull()
  { 
    start = std::numeric_limits< _T >::lowest();
    end = std::numeric_limits< _T >::max();
  };
  
  /** @brief This method returns true if the value val is contained in the interval */
  inline bool within( _T val ){ return val >= start && val <= end; };
  
  _T start = _T(0), end = _T(0);
};


template < typename _T > inline bool operator == ( const Interval< _T >& i1, const Interval< _T >& i2 )
{
  return i1.start == i2.start && i1.end == i2.end;
}

template < typename _T > inline std::ostream& operator<<( std::ostream& os, const Interval< _T > in )  
{
  os <<"[ " << in.start << ", " << in.end << " ]";
  return os;  
}  

typedef Interval<int> IntervalI;
typedef Interval<float> IntervalF;
typedef Interval<double> IntervalD;


/** @brief Template class for 3D boxes

Inspired by the OpenCV cv::Rect_ class for 2D rectangles, this class represent 
3D boxes with faces aligned with the the x, y and/or z axes. 
It is described by the following parameters:
- Coordinates of the two extreme vertices 
- Coordinates the origin (i.e., the "smallest" extreme point) 
  plus width, height, and depth

\note Differently from the cv:::Rect_ object, all boundaries of the box are inclusive, i.e., 
      a box contains both its extreme vertices

For convenience, some Box aliases ara available, e.g.: cv_ext::Box3i, cv_ext::Box3f, and cv_ext::Box3d.
*/
template< typename _T > class Box3
{
public:

  /** @brief  Various constructors */
  Box3(){};
  Box3(_T x, _T y, _T z,_T width, _T height, _T depth );
  Box3(const cv::Point3_<_T>& org,_T width, _T height, _T depth );
  Box3(const cv::Point3_<_T> &pt1, const cv::Point3_<_T> &pt2);
  Box3(const Box3 &b );

  Box3& operator= ( const Box3& b );
  
  /** @brief Provide the "smallest" vertex 
   *
   * @return The vertex composed by the smallest coordinates */
  cv::Point3_<_T> minVertex() const;
  
  /** @brief Provide the "biggest" vertex 
   *
   * @return The vertex composed by the biggest coordinates 
   */
  cv::Point3_<_T> maxVertex() const;

  /** @brief Provide the width, height, depth of the box. 
   *
   * @return A 3D point whose coordinates x,y, and z represents the width, 
   *         height, depth of the box, respectively.
   */
  cv::Point3_<_T> size() const;

  /** @brief Provide the volume (width*height*depth) of the box */
  _T volume() const;

  /** @brief Return true in case of a zero size box */
  bool empty() const;

  /** @brief Checks whether the box contains a point */
  bool contains(const cv::Point3_<_T>& pt) const;
  
  /** @brief Provides the 8 box vertices
   * 
   * @param[out] v_vec A vector that contains the 3D vertices
   * 
   * The vertices are returned with the following order: 
   * (x,y,z), (x + width,y,z), (x + width,y + height,z), (x,y + height,z), 
   * (x,y,z + depth), (x + width,y,z + depth), (x + width,y + height,z + depth), (x,y + height,z + depth) 
   */
  void vertices( std::vector< cv::Point3_<_T> > &v_vec ) const;

  /** @brief Provides the 8 box vertices
   *
   * @returns A vector that contains the 3D vertices
   *
   * The vertices are returned with the following order:
   * (x,y,z), (x + width,y,z), (x + width,y + height,z), (x,y + height,z),
   * (x,y,z + depth), (x + width,y,z + depth), (x + width,y + height,z + depth), (x,y + height,z + depth)
   */
  std::vector< cv::Point3_<_T> > vertices() const;

  _T x = 0, y = 0, z = 0, width = 0, height = 0, depth = 0;

};

typedef Box3<int> Box3i;
typedef Box3<float> Box3f;
typedef Box3<double> Box3d;

template < typename _T > inline std::ostream& operator<<( std::ostream& os, const Box3< _T > b )
{
  os <<"Origin: [ " << b.x << ", " << b.y << ", " << b.z << " ] Size: [ " 
     << b.width << ", " << b.height << ", " << b.depth << " ]";
  return os;  
}

template< typename _T > bool operator== (const Box3<_T> &lhs, const Box3<_T> &rhs)
{
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z &&
      lhs.width == rhs.width && lhs.height == rhs.height && lhs.depth == rhs.depth;
}

template< typename _T > bool operator!= (const Box3<_T> &lhs, const Box3<_T> &rhs)
{
  return !(lhs == rhs);
}



/* Box3 implementation */

template< typename _T > inline
  Box3<_T>::Box3( _T x, _T y, _T z, _T width, _T height, _T depth )
  : x(x), y(y), z(z), width(width), height(height), depth(depth) {}

template< typename _T > inline
  Box3<_T>::Box3(const cv::Point3_< _T >& org, _T width, _T height, _T depth)
  : x(org.x), y(org.y), z(org.z), width(width), height(height), depth(depth) {}

template< typename _T > inline
  Box3<_T>::Box3(const cv::Point3_< _T >& pt1, const cv::Point3_< _T >& pt2)
{
  x = std::min(pt1.x, pt2.x);
  y = std::min(pt1.y, pt2.y);
  z = std::min(pt1.z, pt2.z);
  width = std::max(pt1.x, pt2.x) - x;
  height = std::max(pt1.y, pt2.y) - y;
  depth = std::max(pt1.z, pt2.z) - z;
}

template< typename _T > inline
  Box3<_T>::Box3(const Box3& b)
  : x(b.x), y(b.y), z(b.z), width(b.width), height(b.height), depth(b.depth) {}


template< typename _T > inline
  Box3<_T>& Box3<_T>::operator=(const Box3& b)
{
  x = b.x;
  y = b.y;
  z = b.z;
  width = b.width;
  height = b.height;
  depth = b.depth;
  
  return *this;
}

template< typename _T > inline
  cv::Point3_<_T> Box3<_T>::minVertex() const
{
  return cv::Point3_<_T>(x,y,z);
}

template< typename _T > inline
  cv::Point3_<_T> Box3<_T>::maxVertex() const
{
  return cv::Point3_<_T>(x + width,y + height,z + depth);
}

template< typename _T > inline
  cv::Point3_<_T> Box3<_T>::size() const
{
  return cv::Point3_<_T>(width,height,depth);
}

template< typename _T > inline
  _T Box3<_T>::volume() const
{
  return width * height * depth;
}

template< typename _T > inline
  bool Box3<_T>::empty() const
{
  return width <= 0 || height <= 0 || depth <= 0;
}


template< typename _T > inline
  bool Box3<_T>::contains( const cv::Point3_<_T>& pt ) const
{
  return x <= pt.x && pt.x <= x + width && 
         y <= pt.y && pt.y <= y + height && 
         z <= pt.z && pt.z <= z + depth;
}

template< typename _T > inline
  void Box3<_T>::vertices( std::vector< cv::Point3_<_T> > &v_vec ) const
{
  v_vec.resize(8);
  v_vec[0].x = x;           v_vec[0].y = y;            v_vec[0].z = z;
  v_vec[1].x = x + width;   v_vec[1].y = y;            v_vec[1].z = z;
  v_vec[2].x = x + width;   v_vec[2].y = y + height;   v_vec[2].z = z;
  v_vec[3].x = x;           v_vec[3].y = y + height;   v_vec[3].z = z;
  v_vec[4].x = x;           v_vec[4].y = y;            v_vec[4].z = z + depth;
  v_vec[5].x = x + width;   v_vec[5].y = y;            v_vec[5].z = z + depth;
  v_vec[6].x = x + width;   v_vec[6].y = y + height;   v_vec[6].z = z + depth;
  v_vec[7].x = x;           v_vec[7].y = y + height;   v_vec[7].z = z + depth;
}

template< typename _T > inline
std::vector< cv::Point3_<_T> > Box3<_T>::vertices() const
{
  std::vector< cv::Point3_<_T> > v_vec;
  vertices(v_vec);
  return v_vec;
}


}
