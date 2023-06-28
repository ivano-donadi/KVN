/*
 * d2co - Direct Directional Chamfer Optimization
 *
 *  Copyright (c) 2020, Alberto Pretto <alberto.pretto@flexsight.eu>
 *                      Marco Imperoli <marco.imperoli@flexsight.eu>
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

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <fstream>

#include <Eigen/Dense>

#include "cv_ext/cv_ext.h"

struct PointSet;
struct DirIdxPointSet;

/** @brief Vector typedef */
/**@{*/ 
typedef std::vector<PointSet, Eigen::aligned_allocator<PointSet> > PointSetVec;
typedef std::vector<DirIdxPointSet, Eigen::aligned_allocator<DirIdxPointSet> > DirIdxPointSetVec;
/**@}*/


/** @brief Shared pointer typedef */
/**@{*/ 
typedef std::shared_ptr< PointSetVec > PointSetVecPtr;
typedef std::shared_ptr< const PointSetVec > PointSetVecConstPtr;
typedef std::shared_ptr< DirIdxPointSetVec > DirIdxPointSetVecPtr;
typedef std::shared_ptr< const DirIdxPointSetVec > DirIdxPointSetVecConstPtr;
/**@}*/

struct ObjectTemplate
{
  virtual ~ObjectTemplate() = 0;

  uint32_t class_id;
  Eigen::Quaterniond obj_r_quat;
  Eigen::Vector3d obj_t_vec;
  cv_ext::Box3f obj_bbox3d;
  std::vector <cv::Point2f> proj_obj_bbox3d;

  virtual void binaryRead( std::ifstream& in);  
  virtual void binaryWrite( std::ostream& out) const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PointSet : public ObjectTemplate
{
  ~PointSet() override = default;

  std::vector< cv::Point3f > obj_pts;
  std::vector< cv::Point > proj_pts;
  cv::Rect bbox;
  
  void binaryRead ( std::ifstream& in) override;
  void binaryWrite( std::ostream& out) const override;
};

struct DirIdxPointSet : public PointSet
{
  ~DirIdxPointSet() override = default;

  std::vector< cv::Point3f > obj_d_pts;
  std::vector<int> dir_idx;
  
  void binaryRead( std::ifstream& in)  override;
  void binaryWrite( std::ostream& out)  const override;
};

/** @brief Class used to estimate the position of an object given a template and an image offset
 *
 *  @details Object templates (see ObjectTemplate class and its derived classes) are generated from a specific
 *  object view (rotation + translation) projected in the image plane. If this projection is translated of a 2D
 *  offset in the image plane, such rotation and translation obviously change. The new rotation and tranlsation
 *  can be retrieved by calling the ObjectTemplatePnP::solve() method.
 */
class ObjectTemplatePnP
{
 public:

  /** @brief Default object constructor
   *
   *  @note A camera model should be set with the setCamModel() method
   */
  ObjectTemplatePnP() = default;

  /** @brief Object constructor
   *
   *  @param[in] cam_model Camera intrinsic camera parameters used to estimate the object position
   */
  ObjectTemplatePnP( const cv_ext::PinholeCameraModel &cam_model );

  /** @brief Set the camera parameters
   *
   *  @param[in] cam_model Camera intrinsic camera parameters used to estimate the object position
   */
  void setCamModel ( const cv_ext::PinholeCameraModel& cam_model );

  /** @brief If enabled, preserve the in the estimation the original object depth
   *
   *  @param[in] enabled Flag used to enable/disable the functinality
   *
   *  Setting enabled to true will Enabling this function, the position of the object is estimated
   *  letting the original depth of the object template unchanged (i.e., the z translation component)
   *
   *  @note Defaults false
   */
  void fixZTranslation( bool enabled ) { fix_z_ = enabled; };

  /** @brief Retrieved the new object position given a translated version of an object template
   *
   *  @param[in] obj_templ Input object template
   *  @param[in] img_offset X,Y translation of the original projection of obj_templ
   *  @param[out] r_quat Output object rotation (quaternion representation)
   *  @param[out] t_vec Output object translation
   */
  void solve( const ObjectTemplate &obj_templ, const cv::Point &img_offset,
              double r_quat[4], double t_vec[3] );

  /** @brief Retrieved the new object position given a translated version of an object template
   *
   *  @param[in] obj_templ Input object template
   *  @param[in] img_offset X,Y translation of the original projection of obj_templ
   *  @param[out] r_quat Output object rotation (quaternion representation)
   *  @param[out] t_vec Output object translation
   */
  void solve( const ObjectTemplate &obj_templ, const cv::Point &img_offset,
              Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec );

  /** @brief Retrieved the new object position given a translated version of an object template
   *
   *  @param[in] obj_templ Input object template
   *  @param[in] img_offset X,Y translation of the original projection of obj_templ
   *  @param[out] r_vec Output object rotation (axis angle representation)
   *  @param[out] t_vec Output object translation
   */
  void solve( const ObjectTemplate &obj_templ, const cv::Point &img_offset,
              cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec );

 private:

  IterativePnP ipnp_;
  bool fix_z_ = false;
};

template <typename _T> bool loadTemplateVector( const std::string &filename, std::vector<_T, Eigen::aligned_allocator<_T> > &tv );
template <typename _T> bool saveTemplateVector( const std::string &filename, const std::vector<_T, Eigen::aligned_allocator<_T> > &tv );

/* Some basic implementations */

template<typename _T>
bool loadTemplateVector(const std::string &filename, std::vector<_T, Eigen::aligned_allocator<_T> > &tv)
{
  tv.clear();

  std::ifstream in_file(filename, std::ifstream::in | std::ofstream::binary );
  if( in_file.is_open() )
  {
    size_t num_templates;
    in_file.read( reinterpret_cast<char*>(&num_templates), sizeof(size_t));

    tv.resize(num_templates);

    for( auto &t : tv )
      t.binaryRead(in_file);

    in_file.close();
    return true;
  }
  else
    return false;
}

template <typename _T>
bool saveTemplateVector ( const std::string &filename, const std::vector<_T, Eigen::aligned_allocator<_T> > &tv )
{
  std::ofstream out_file(filename,  std::ofstream::out | std::ofstream::binary );
  if( out_file.is_open() )
  {
    size_t num_templates = tv.size();
    out_file.write( reinterpret_cast<const char*>( &num_templates ), sizeof( size_t ) );

    for( auto &t : tv )
      t.binaryWrite(out_file);

    out_file.close();

    return true;
  }
  else
    return false;
}
