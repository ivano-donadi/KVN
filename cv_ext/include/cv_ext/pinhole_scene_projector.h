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

#include <omp.h>
#include <stdexcept>
#include <boost/concept_check.hpp>
#include <boost/shared_ptr.hpp>
#include <Eigen/Geometry>

#include "cv_ext/conversions.h"
#include "cv_ext/pinhole_camera_model.h"

namespace cv_ext
{

/** @brief Shared pointer typedef */
typedef boost::shared_ptr< class PinholeSceneProjector > PinholeSceneProjectorPtr;

/**
 * @brief Utility class used for perspective projection (using the pinhole model)
 *  
 * It is based on the PinholeCameraModel object. It can deals with occlusions. 
 * 
 */
class PinholeSceneProjector
{
public:
  
  /**
   * @brief Constructor: just stores the parameters
   * 
   * @param[in] cam_model The camera model object
   */
  PinholeSceneProjector( const PinholeCameraModel &cam_model ) :
    cam_model_(cam_model),
    parallelism_enabled_(false) 
  {
    r_mat_.setIdentity();
    t_vec_.setZero();
  };

  //! @brief Provide the camera model object used for projection
  PinholeCameraModel cameraModel() const { return cam_model_; };
  
  /**
   * @brief Enable/disable parallelism for the projected algorithm
   * 
   * @param[in] enable If true, projection algorithm is run in a parallel region,
   *                   otherwise it is run as a single thread
   */
  void enableParallelism( bool enable ){ parallelism_enabled_ = enable; };
  
  //! @brief Return true is the projection algorithm is run in a parallel region
  bool isParallelismEnabled() const { return parallelism_enabled_; };

  /**
 * @brief Set current rigid body transformation (quaternion rotation + translation)
 *
 * @param[in] r_mat 3x3 rotation matrix
 * @param[in] t_vec Translation vector
 *
 * The rigid body transformation map points from the scene frame to the camera frame
 */
  void setTransformation( const Eigen::Matrix3d &r_mat, const Eigen::Vector3d &t_vec )
  {
    r_mat_ = r_mat;
    t_vec_ = t_vec;
  };

  /**
   * @brief Set the current rigid body transformation (quaternion rotation + translation)
   * 
   * @param[in] r_quat Quaternion that represents the rotation. It is not assumed that r_quat has unit norm, but
    *                  it is assumed that the norm is non-zero.
   * @param[in] t_vec Translation vector
   * 
   * The rigid body transformation map points from the scene frame to the camera frame
   */
  void setTransformation(const double r_quat[4], const double t_vec[3] )
  {
    Eigen::Quaterniond eigen_r_quat;
    cv_ext::quat2EigenQuat(r_quat, eigen_r_quat);
    r_mat_ = eigen_r_quat.toRotationMatrix();
    t_vec_[0] = t_vec[0]; t_vec_[1] = t_vec[1]; t_vec_[2] = t_vec[2];
  };
  
  /**
   * @brief Set current rigid body transformation (quaternion rotation + translation)
   * 
   * @param[in] r_quat Quaternion that represents the rotation. It is not assumed that r_quat has unit norm, but
    *                  it is assumed that the norm is non-zero.
   * @param[in] t_vec Translation vector
   * 
   * The rigid body transformation map points from the scene frame to the camera frame
   */
  void setTransformation(const Eigen::Quaterniond &r_quat, const Eigen::Vector3d &t_vec )
  {
    r_mat_ = r_quat.toRotationMatrix();
    t_vec_ = t_vec;
  };
  
  /**
   * @brief Set current rigid body transformation (axis-angle or quaternion rotation + translation)
   * 
   * @param[in] r_vec 3 by 1 rotation vector, in exponential notation (axis-angle),
   *                  or 4 x 1 quaternion that represents the rotation. 
   *                  It is not assumed that r_quat has unit norm, but
    *                 it is assumed that the norm is non-zero.
   * @param[in] t_vec 3 by 1 translation vector
   * 
   * The rigid body transformation map points from the scene frame to the camera frame
   */
  void setTransformation(const cv::Mat_<double> &r_vec, const cv::Mat_<double> &t_vec)
  {
    cv_ext_assert( r_vec.channels() == 1 && r_vec.rows == 3 && r_vec.cols == 1 );
    cv_ext_assert( t_vec.channels() == 1 && t_vec.rows == 3 && t_vec.cols == 1 );

    ceres::AngleAxisToRotationMatrix(reinterpret_cast<double *>(r_vec.data), r_mat_.data());
    t_vec_ = Eigen::Map<Eigen::Vector3d>(reinterpret_cast<double *>(t_vec.data));
  };

  /**
   * @brief Computes the ideal point coordinates observed by a normalized (canonical) camera from 
   *        a vector of points observed by the real camera.
   * 
   * @tparam _TPoint2D The image coordinates type (e.g., cv::Point2d ). It should have x and y fields, and a basic 
   *                   constructor like _TPoint2D(T _x, T _y).
   * 
   * @param[in] dist_pts Input 2D real image points
   * @param[out] norm_pts Output 2D ideal points
   */
  template < typename _TPoint2D > 
    void normalizePoints( const std::vector< _TPoint2D > &dist_pts, 
                          std::vector< _TPoint2D > &norm_pts ) const;
   

  /**
   * @brief Computes the real pixel coordinates observed by a  by the real camera from 
   *        a vector of points observed by the normalized (canonical) camera.
   * 
   * @tparam _TPoint2D The image coordinates type (e.g., cv::Point2d ). It should have x and y fields, and a basic 
   *                   constructor like _TPoint2D(T _x, T _y).
   * 
   * @param[in] norm_pts Input 2D ideal points
   * @param[out] dist_pts Output 2D real image points
   */
  template < typename _TPoint2D > 
    void denormalizePoints( const std::vector< _TPoint2D > &norm_pts, 
                            std::vector< _TPoint2D > &dist_pts ) const;
                            
  /**
   * @brief Projects a vector of points given the 
   *        transformation previously provided with setTransformation()
   * 
   * @tparam _TPoint3D The 3D scene points type (e.g., cv::Point3d ). It should have x, y and z fields, 
   *                   and a basic constructor like _TPoint3D(T _x, T _y, T _z).   
   * @tparam _TPoint2D The image coordinates type (e.g., cv::Point2d ). It should have x and y fields, 
   *                   and a basic constructor like _TPoint2D(T _x, T _y).   
   * 
   * @param[in] scene_pts Input 3D scene points
   * @param[out] img_pts Output 2D image points
   * @param[in] check_occlusions If true, this function will take into account the occlusions between points
   * 
   * If check_occlusions is set to true, the coordinates of occluded points are set to (-1, -1)
   */
  template < typename _TPoint3D, typename _TPoint2D > 
    void projectPoints( const std::vector< _TPoint3D > &scene_pts, 
                        std::vector< _TPoint2D > &img_pts,
                        bool check_occlusions = false ) const ;

  /**
   * @brief Projects a vector of points given the
   *        transformation previously provided with setTransformation().
   * 
   * @tparam _TPoint3D The 3D scene points type (e.g., cv::Point3d ). It should have x, y and z fields, and a basic 
   *                   constructor like _TPoint3D(T _x, T _y, T _z).   
   * @tparam _TPoint2D The image coordinates type (e.g., cv::Point2d ). It should have x and y fields, and a basic 
   *                   constructor like _TPoint2D(T _x, T _y).   
   *
   * @param[in] scene_pts Input 3D scene points
   * @param[out] img_pts Output 2D image points
   * @param[out] occ_mask Output (binary) image mask with highlighted the current positions of
   *                      the occlusions
   *
   * This function deals with occlusions, providing in output an occlusion mask.
   * The coordinates of occluded points are set to (-1, -1)
   */
  template < typename _TPoint3D, typename _TPoint2D >
    void projectPoints( const std::vector< _TPoint3D > &scene_pts,
                        std::vector< _TPoint2D > &img_pts,
                        cv::Mat &occ_mask ) const ;

  /**
   * @brief Projects a vector of 3D segments (represented by theirs start and end points)
   *        given the transformation previously provided with setTransformation().
   * 
   * @tparam _TSeg3D The 3D scene segments type (e.g., cv::Vec6f ). with 6 elementsand accessible with the [x] operator
   * @tparam _TSeg2D The 2D scene segments type (e.g., cv::Vec4f ). with 4 elements and accessible with the [x] operator
   * 
   * @param[in] scene_segs Input 3D scene segments
   * @param[out] img_segs Output 2D image segments
   */
  template < typename _TSeg3D, typename _TSeg2D > 
    void projectSegments( const std::vector< _TSeg3D > &scene_segs, 
                          std::vector< _TSeg2D > &img_segs ) const ;
                          
  /**
   * @brief Unprojects a vector of image points along with their depths
   *        into a vector of 3D scene points, assuming the points are in the camera reference frame.
   * 
   * @tparam _TPoint3D The 3D scene points type (e.g., cv::Point3d ). It should have x, y and z fields, and a basic 
   *                   constructor like _TPoint3D(T _x, T _y, T _z).   
   * @tparam _TPoint2D The image coordinates type (e.g., cv::Point2d ). It should have x and y fields, and a basic 
   *                   constructor like _TPoint2D(T _x, T _y).   
   * 
   * @param[in] img_pts Input 2D image points
   * @param[in] depths Input z coordinates (depths) of the points in the camera reference frame: it should have the
   *                   same size of img_pts
   * @param[out] scene_pt Output 3D scene points
   */
  template < typename _TPoint2D, typename _TPoint3D, typename _T > 
    void unprojectPoints( const std::vector< _TPoint2D > &img_pts,
                          const std::vector< _T > &depths,
                          std::vector< _TPoint3D > &scene_pts ) const;
                        
private:

  template < typename _TPoint3D, typename _TPoint2D >
    void projectPoints( const std::vector< _TPoint3D > &scene_pts,
                        std::vector< _TPoint2D > &img_pts,
                        bool check_occlusions, cv::Mat occ_mask ) const;
  
  PinholeCameraModel cam_model_;

  Eigen::Matrix3d r_mat_;
  Eigen::Vector3d t_vec_;

  bool parallelism_enabled_;
};
  
/* Implementation */

template < typename _TPoint2D > void PinholeSceneProjector:: 
  normalizePoints( const std::vector< _TPoint2D > &dist_pts, std::vector< _TPoint2D > &norm_pts ) const
{   
  int pts_size = dist_pts.size();
  norm_pts.resize(pts_size);
    
  if( cam_model_.hasDistCoeff() )
  {    
    #pragma omp parallel for if( parallelism_enabled_ )
    for( int i = 0; i < pts_size; i++)
    {
      const double dist_pt[2] = { dist_pts[i].x, dist_pts[i].y };
      double norm_pt[2];
      cam_model_.normalize( dist_pt, norm_pt );
      norm_pts[i] = _TPoint2D( norm_pt[0], norm_pt[1]);
    }
  }
  else
  {
    #pragma omp parallel for if( parallelism_enabled_ )
    for( int i = 0; i < pts_size; i++)
    {
      const double dist_pt[2] = { dist_pts[i].x, dist_pts[i].y };
      double norm_pt[2];
      cam_model_.normalizeWithoutDistortion( dist_pt, norm_pt );
      norm_pts[i] = _TPoint2D( norm_pt[0], norm_pt[1]);
    }
  }
}

template < typename _TPoint2D > void PinholeSceneProjector:: 
  denormalizePoints( const std::vector< _TPoint2D > &norm_pts, std::vector< _TPoint2D > &dist_pts ) const
{   
  int pts_size = norm_pts.size();
  dist_pts.resize(pts_size);
    
  if( cam_model_.hasDistCoeff() )
  {    
    #pragma omp parallel for if( parallelism_enabled_ )
    for( int i = 0; i < pts_size; i++)
    {
      const double norm_pt[2] = { norm_pts[i].x, norm_pts[i].y };
      double dist_pt[2];
      cam_model_.denormalize( norm_pt, dist_pt );
      dist_pts[i] = _TPoint2D( dist_pt[0], dist_pt[1]);
    }
  }
  else
  {
    #pragma omp parallel for if( parallelism_enabled_ )
    for( int i = 0; i < pts_size; i++)
    {
      const double norm_pt[2] = { norm_pts[i].x, norm_pts[i].y };
      double dist_pt[2];
      cam_model_.denormalizeWithoutDistortion( norm_pt, dist_pt );
      dist_pts[i] = _TPoint2D( dist_pt[0], dist_pt[1]);
    }
  }
}

template < typename _TPoint3D, typename _TPoint2D > void PinholeSceneProjector:: 
  projectPoints( const std::vector< _TPoint3D > &scene_pts, 
                 std::vector< _TPoint2D > &img_pts,
                 bool check_occlusions ) const
{
  projectPoints( scene_pts, img_pts,check_occlusions, cv::Mat() );
}

template < typename _TPoint3D, typename _TPoint2D > void PinholeSceneProjector::
    projectPoints( const std::vector< _TPoint3D > &scene_pts, std::vector< _TPoint2D > &img_pts, 
                   cv::Mat &occ_mask ) const 
{
  occ_mask = cv::Mat_<uchar>(cam_model_.imgHeight(), cam_model_.imgWidth(), uchar(0));
  projectPoints( scene_pts, img_pts, true, occ_mask );
}

template < typename _TPoint3D, typename _TPoint2D > void PinholeSceneProjector::
    projectPoints( const std::vector< _TPoint3D > &scene_pts, std::vector< _TPoint2D > &img_pts,
                   bool check_occlusions, cv::Mat occ_mask ) const
{
  int pts_size = scene_pts.size();
  img_pts.resize(pts_size);

  if( check_occlusions )
  {
    std::vector<double> scene_pts_z;
    scene_pts_z.resize(pts_size);

    if( cam_model_.hasDistCoeff() )
    {
      #pragma omp parallel for if( parallelism_enabled_ )
      for( int i = 0; i < pts_size; i++)
      {
        const double scene_pt[3] = { scene_pts[i].x, scene_pts[i].y, scene_pts[i].z };
        double proj_pt[2], depth;
        cam_model_.rTProject( r_mat_, t_vec_, scene_pt, proj_pt, depth );
        img_pts[i] = _TPoint2D( proj_pt[0], proj_pt[1]);
        scene_pts_z[i] = depth;
      }
    }
    else
    {
      #pragma omp parallel for if( parallelism_enabled_ )
      for( int i = 0; i < pts_size; i++)
      {
        const double scene_pt[3] = { scene_pts[i].x, scene_pts[i].y, scene_pts[i].z };
        double proj_pt[2], depth;
        cam_model_.rTProjectWithoutDistortion(r_mat_, t_vec_, scene_pt, proj_pt, depth );
        img_pts[i] = _TPoint2D( proj_pt[0], proj_pt[1]);
        scene_pts_z[i] = depth;
      }
    }

    cv::Mat index_mask = cv::Mat_<int>(cam_model_.imgHeight(), cam_model_.imgWidth(), -1);
    bool has_mask = !occ_mask.empty();

    for( int i = 0; i < pts_size; i++)
    {
      if( img_pts[i].x >= 0 && img_pts[i].y >= 0 && 
          img_pts[i].x < cam_model_.imgWidth() && img_pts[i].y < cam_model_.imgHeight())
      {
        int x = round(img_pts[i].x), y = round(img_pts[i].y);
        int selected_idx = index_mask.at<int>(y,x);
        if( selected_idx < 0 )
          index_mask.at<int>(y,x) = i;
        else if( scene_pts_z[i] != scene_pts_z[selected_idx] )
        {
          // Occlusion
          if( scene_pts_z[i] < scene_pts_z[selected_idx] )
          {
            img_pts[selected_idx] = _TPoint2D( -1, -1 );
            index_mask.at<int>(y,x) = i;
          }
          else
            img_pts[i] = _TPoint2D( -1, -1 );
          
          if( has_mask )
            occ_mask.at<uchar>(y,x) = 255;
        }
      }
    }
  }
  else
  {
    if( cam_model_.hasDistCoeff() )
    {
      #pragma omp parallel for if( parallelism_enabled_ )
      for( int i = 0; i < pts_size; i++)
      {
        const double scene_pt[3] = { scene_pts[i].x, scene_pts[i].y, scene_pts[i].z };
        double proj_pt[2];
        cam_model_.rTProject( r_mat_, t_vec_, scene_pt, proj_pt );
        img_pts[i] = _TPoint2D( proj_pt[0], proj_pt[1]);
      }
    }
    else
    {
      #pragma omp parallel for if( parallelism_enabled_ )
      for( int i = 0; i < pts_size; i++)
      {
        const double scene_pt[3] = { scene_pts[i].x, scene_pts[i].y, scene_pts[i].z };
        double proj_pt[2];
        cam_model_.rTProjectWithoutDistortion(r_mat_, t_vec_, scene_pt, proj_pt );
        img_pts[i] = _TPoint2D( proj_pt[0], proj_pt[1]);
      }
    }
  }
}

template < typename _TSeg3D, typename _TSeg2D > 
    void PinholeSceneProjector::projectSegments( const std::vector< _TSeg3D > &scene_segs, 
                                                 std::vector< _TSeg2D > &img_segs ) const
{
  int segs_size = scene_segs.size();
  img_segs.resize(segs_size);
  
  if( cam_model_.hasDistCoeff() )
  {    
    #pragma omp parallel for if( parallelism_enabled_ )
    for( int i = 0; i < segs_size; i++)
    {
      const _TSeg3D &scene_seg = scene_segs[i];
      const double scene_pt0[3] = { scene_seg[0], scene_seg[1], scene_seg[2] }, 
                   scene_pt1[3] = { scene_seg[3], scene_seg[4], scene_seg[5] };
      double proj_pt0[2], proj_pt1[2];
      cam_model_.rTProject( r_mat_, t_vec_, scene_pt0, proj_pt0 );
      cam_model_.rTProject( r_mat_, t_vec_, scene_pt1, proj_pt1 );
      img_segs[i] =  _TSeg2D( proj_pt0[0], proj_pt0[1], proj_pt1[0], proj_pt1[1] );
    }
  }
  else
  {
    #pragma omp parallel for if( parallelism_enabled_ )
    for( int i = 0; i < segs_size; i++)
    {
      const _TSeg3D &scene_seg = scene_segs[i];
      const double scene_pt0[3] = { scene_seg[0], scene_seg[1], scene_seg[2] }, 
                   scene_pt1[3] = { scene_seg[3], scene_seg[4], scene_seg[5] };
      double proj_pt0[2], proj_pt1[2];
      cam_model_.rTProjectWithoutDistortion( r_mat_, t_vec_, scene_pt0, proj_pt0 );
      cam_model_.rTProjectWithoutDistortion( r_mat_, t_vec_, scene_pt1, proj_pt1 );
      img_segs[i] = _TSeg2D( proj_pt0[0], proj_pt0[1], proj_pt1[0], proj_pt1[1] );
    }
  }
}

template < typename _TPoint2D, typename _TPoint3D, typename _T > void PinholeSceneProjector::
  unprojectPoints( const std::vector< _TPoint2D > &img_pts,
                   const std::vector< _T > &depths,
                   std::vector< _TPoint3D > &scene_pts ) const
{
  if( img_pts.size() != depths.size() )
    throw std::invalid_argument("Image points and depth vectors should have the same size");  
  
  int pts_size = img_pts.size();
  scene_pts.resize(pts_size);
    
  if( cam_model_.hasDistCoeff() )
  {    
    #pragma omp parallel for if( parallelism_enabled_ )
    for( int i = 0; i < pts_size; i++)
    {
      const double img_pt[2] = { img_pts[i].x, img_pts[i].y };
      double scene_pt[3];
      cam_model_.unproject( img_pt, double(depths[i]), scene_pt );
      scene_pts[i] = _TPoint3D( scene_pt[0], scene_pt[1], scene_pt[2]);

    }
  }
  else
  {
    #pragma omp parallel for if( parallelism_enabled_ )
    for( int i = 0; i < pts_size; i++)
    {
      const double img_pt[2] = { img_pts[i].x, img_pts[i].y };
      double scene_pt[3];
      cam_model_.unprojectWithoutDistortion( img_pt, double(depths[i]), scene_pt );
      scene_pts[i] = _TPoint3D( scene_pt[0], scene_pt[1], scene_pt[2]);
    }
  }  
}

}