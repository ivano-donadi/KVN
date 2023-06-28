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

// Fix reset() bug
#pragma once

#include <limits>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <ceres/rotation.h>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include "cv_ext/conversions.h"

namespace cv_ext
{

/**
 * @brief PinholeCameraModel represents a Pinhole model camera. It stores 
 *        the camera intrinsic parameters and it provides functions for 3D point
 *        normalization and denormalization, projection and unprojection, etc.
 */
class PinholeCameraModel
{
public:
  
   /**
   * @brief Default (empty) constructor: canonical camera model, focal len = 1, center = (0,0), no distortion
   */
   PinholeCameraModel(){};

  /**
   * @brief Constructor: just checks and stores inputs
   *
   * @param[in] camera_matrix Linear intrinsic parameters of the camera: it is a 3 by 3 matrix
   * @param[in] img_width Width in pixel of the original (unscaled) image
   * @param[in] img_height Height in pixel of the original (unscaled) image
   * @param[in] dist_coeff Distortion coefficients, i.e. non-linear intrinsic parameters
   *                       of the camera: it could be a 4 to 8 by 1 matrix
   *
   * The scale factor, the region of interest and the accuracy used in the termination criteria
   * are set to their default values (see reset()).
   */
   PinholeCameraModel( const cv::Mat &camera_matrix, int img_width, int img_height,
                       const cv::Mat &dist_coeff = cv::Mat() );

  /**
   * @brief Constructor: just checks and stores inputs
   *
   * @param[in] fx Focal lenght in pixels in the x direction
   * @param[in] fy Focal lenght in pixels in the x direction
   * @param[in] cx X coordinate of the camera principal point in pixel
   * @param[in] cy Y coordinate of the camera principal point in pixel
   * @param[in] img_width Width in pixel of the original (unscaled) image
   * @param[in] img_height Height in pixel of the original (unscaled) image
   *
   * @param[in] dist_k0 Radial distortion first polynomial coefficient
   * @param[in] dist_k1 Radial distortion second polynomial coefficient
   * @param[in] dist_px Tangential distortion X coefficient
   * @param[in] dist_py Tangential distortion Y coefficient
   * @param[in] dist_k2 Radial distortion third polynomial coefficient
   * @param[in] dist_k3 Radial distortion fouth polynomial coefficient
   * @param[in] dist_k4 Radial distortion fifth polynomial coefficient
   * @param[in] dist_k5 Radial distortion sixth polynomial coefficient
   *
   * The scale factor, the region of interest and the accuracy used in the termination criteria
   * are set to their default values (see reset()).
   */
  PinholeCameraModel( double fx, double fy, double cx, double cy, int img_width, int img_height,
                      double dist_k0, double dist_k1, double dist_px, double dist_py,
                      double dist_k2 = 0, double dist_k3 = 0, double dist_k4 = 0, double dist_k5 = 0 );

  /**
   * @brief Equality operator
   *
   * @param[in] l A PinholeCameraModel object
   * @param[in] r Another PinholeCameraModel 
   * 
   * \note The current scale factor, the region of interest and the accuracy used
   *       in the termination criteria (see set setTerminationEpsilon()) are not considered
   */
  friend bool operator==(const PinholeCameraModel& l, const PinholeCameraModel& r)
  {
    if( l.orig_img_width_ != r.orig_img_width_ ||
        l.orig_img_height_ != r.orig_img_height_ ||
        l.has_dist_coeff_ != r.has_dist_coeff_ ||
        l.orig_fx_ != r.orig_fx_ ||
        l.orig_fy_ != r.orig_fy_ ||
        l.orig_cx_ != r.orig_cx_ ||
        l.orig_cy_ != r.orig_cy_ ||
        l.dist_px_ != r.dist_px_ ||
        l.dist_py_ != r.dist_py_ ||
        l.dist_k0_ != r.dist_k0_ ||
        l.dist_k1_ != r.dist_k1_ ||
        l.dist_k2_ != r.dist_k2_ ||
        l.dist_k3_ != r.dist_k3_ ||
        l.dist_k4_ != r.dist_k4_ ||
        l.dist_k5_ != r.dist_k5_ )
      return false;
    else
      return true;
  }

  /**
   * @brief Inequality operator
   *
   * @param[in] l A PinholeCameraModel object
   * @param[in] r Another PinholeCameraModel object
   * 
   * \note The current scale factor, the region of interest and the accuracy used
   *       in the termination criteria (see set setTerminationEpsilon()) are not considered
   */
  friend bool operator!=(const PinholeCameraModel& l, const PinholeCameraModel& r){ return !(l == r); };

  /**
   * @brief Read all the PinholeCameraModel object members from a yaml-cpp YAML::Node
   *
   * @param[in] in The yaml-cpp YAML::Node
   *
   * The scale factor, the region of interest and the accuracy used in the termination criteria
   * are set to their default values (see reset()).
   */
  void read(const YAML::Node &in );

  /**
   * @brief Write all the PinholeCameraModel object members to a yaml-cpp YAML::Node
   *
   * @param[in] out The yaml-cpp YAML::Node
   *
   * \note The current scale factor, the current region of interest and the accuracy used
   *       in the termination criteria (see set setTerminationEpsilon()) are not saved
   */
  void write( YAML::Node &out ) const;

  /**
   * @brief Read all the PinholeCameraModel object members from a YAML file
   *
   * @param[in] filename The filename: the file extension should be "yml" or "yaml",
   *                     otherwise the extension is automatically added
   * @return true if successful, false otherwise
   *
   * The scale factor, the region of interest and the accuracy used in the termination criteria 
   * are set to their default values (see reset()).
   */
  bool readFromFile( const std::string &filename );

  /**
   * @brief Write all the PinholeCameraModel object members to a YAML file
   *
   * @param[in] filename The filename: the file extension should be "yml" or "yaml",
   *                     otherwise the extension is automatically added
   * @return true if successful, false otherwise
   * 
   * \note The current scale factor, the current region of interest and the accuracy used
   *       in the termination criteria (see set setTerminationEpsilon()) are not saved   
   */
  bool writeToFile( const std::string &filename ) const;

  //! @brief True if distortion coefficients are not zero
  inline bool hasDistCoeff() const { return has_dist_coeff_; };

  /**
   * @brief Provide the original image width
   * 
   * The current scale and/or RoI are not not take into account.
   */ 
  inline int origImgWidth() const { return orig_img_width_; };

  /**
   * @brief Provide the original image height
   * 
   * The current scale and/or RoI are not not take into account.
   */ 
  inline int origImgHeight() const { return orig_img_height_; };
  
  /**
   * @brief Provide the (scaled) image width
   * 
   * If the region of interest is enabled (see enableRegionOfInterest() and 
   * setRegionOfInterest()), the returned width equals the current RoI width
   */ 
  inline int imgWidth() const { return img_width_; };

  /**
   * @brief Provide the (scaled) image height
   * 
   * If the region of interest is enabled (see enableRegionOfInterest() and 
   * setRegionOfInterest()), the returned height equals the current RoI height
   */ 
  inline int imgHeight() const { return img_height_; };

  /**
   * @brief Provide the (scaled) size (width, height) of the image
   *
   * If the region of interest is enabled (see enableRegionOfInterest() and
   * setRegionOfInterest()), the returned size equals the current RoI size
   */
  inline cv::Size imgSize() const { return cv::Size(img_width_, img_height_); };
  
  //! @brief Provide the current size scale factor
  double sizeScaleFactor() const { return size_scale_factor_; };

  /**
   * @brief Provide the current, possibly scaled  (see setSizeScaleFactor()) image
   * region of interest (see setRegionOfInterest())
   */
  cv::Rect regionOfInterest() const { return roi_; };

  /**
   * @brief Provide the current image region of interest (see setRegionOfInterest())
   *
   * The current scale and/or RoI are not not take into account.
   */
  cv::Rect origRegionOfInterest() const { return orig_roi_; };

  /**
   * @brief Return true is the region of interest is enabled (see enableRegionOfInterest() and 
   *        setRegionOfInterest())
   */
  bool regionOfInterestEnabled() const { return roi_enabled_; };
  
  /**
   * @brief Provide the desired accuracy used in the termination criteria, 
   *        e.g., in the point normalization
   *
   **/
  double terminationEpsilon() const { return term_epsilon_; };
   
  //! @brief Provide the (scaled) X focal lenght 
  inline const double &fx() const { return fx_; };
  //! @brief Provide the (scaled) Y focal lenght 
  inline const double &fy() const { return fy_; };
  /**
   * @brief Provide the (scaled) X coordinate of the image projection center
   * 
   * If the region of interest is enabled (see enableRegionOfInterest() and 
   * setRegionOfInterest()) the image projection center will have an offset that depends
   * on the current RoI 
   */
  inline const double &cx() const { return cx_; };
  /**
   * @brief Provide the (scaled) Y coordinate of the image projection center
   * 
   * If the region of interest is enabled (see enableRegionOfInterest() and 
   * setRegionOfInterest()) the image projection center will have an offset that depends
   * on the current RoI 
   */
  inline const double &cy() const { return cy_; };
  
  /**
   * @brief Provide the (scaled) 3 by 3 image camera matrix
   * 
   * If the region of interest is enabled (see enableRegionOfInterest() and 
   * setRegionOfInterest()) the image projection center (components (0,2) and (1,2) of the
   * camera matrix) will have an offset that depends on the current RoI 
   */
  cv::Mat_<double> cameraMatrix() const { return (cv::Mat_<double>(3,3)  << fx_, 0,   cx_,
                                                                            0,   fy_, cy_,
                                                                            0,   0,   1); };
                                                                   
  //! @brief Provide the distortion coefficient Px
  inline const double &distPx() const { return dist_px_; };
  //! @brief Provide the distortion coefficient Py
  inline const double &distPy() const { return dist_py_; };
  
  //! @brief Provide the distortion coefficient K0
  inline const double &distK0() const { return dist_k0_; };
  //! @brief Provide the distortion coefficient K1
  inline const double &distK1() const { return dist_k1_; };
  //! @brief Provide the distortion coefficient K2
  inline const double &distK2() const { return dist_k2_; };
  //! @brief Provide the distortion coefficient K3
  inline const double &distK3() const { return dist_k3_; };
  //! @brief Provide the distortion coefficient K4
  inline const double &distK4() const { return dist_k4_; };
  //! @brief Provide the distortion coefficient K5
  inline const double &distK5() const { return dist_k5_; };
  
  //! @brief Provide an 8 by 1 distortion coefficients matrix
  cv::Mat_<double> distorsionCoeff() const
  { 
    return (cv::Mat_<double>(8,1) <<   dist_k0_, dist_k1_, 
                                       dist_px_, dist_py_, 
                                       dist_k2_, dist_k3_, 
                                       dist_k4_, dist_k5_);
  };
  
  /**
   * @brief Set a new size scale factor
   *        
   * @param[in] scale_factor The scale factor
   * 
   * The camera matrix, the image size and the RoI (see setRegionOfInterest()) will be scaled accordling 
   * For instance, a scale of 2 means that the width and height will be halved with respect 
   * to the original image, so the scaled image has 1/4 pixels of the original.
   * \note The scale factor must be greater than 0.
   * 
   */
  void setSizeScaleFactor( double scale_factor );
  
  /**
   * @brief Set an image region of interest (RoI). 
   *        
   * @param[in] roi Region of interest
   * 
   * Define a new image that is a rectangular portion of the original image. 
   * If the region of interest is enabled (see enableRegionOfInterest()) the PinholeCameraModel 
   * object will works with a new image plane represented by the provided RoI. For instance, 
   * the point projection operations will provided outputs with a 2D offset defined
   * by top-left corner of the rectangle that defines the region of interest.
   * The region of interest should have non zero size, and it should be inside the image.
   * \note The region of interest set with this function refers to the original, not scaled, image 
   *       (see origImgWidth() and origImgHeight() to query the original image size): 
   *       to query the current (possibly scaled) RoI, used regionOfInterest().
   *       If the scale is changed, the region of interest changes accordingly.
   * \warning The RoI is NOT enabled by default: you should explicitly call enableRegionOfInterest(true) 
   *          to enable it
   */
  void setRegionOfInterest( cv::Rect& roi );

  /**
   * @brief Enable/disable the region of interest
   * 
   * @param[in] enable If true, PinholeCameraModel object will works with a new image plane represented 
   *                   by the provided RoI (see setRegionOfInterest())
   */  
  void enableRegionOfInterest( bool enable );
  
  /**
   * @brief Set the accuracy used in the termination criteria, 
   *        e.g., in the point normalization   
   * 
   * @param[in] eps desired accuracy
   **/
   void setTerminationEpsilon( double eps ){ term_epsilon_ = eps; };

  /**
   * @brief Reset the scale factor, the region of interest and the accuracy used
   *       in the termination criteria to their default values.
   * 
   * Default values: scale = 1, region of interest(not enabled) = the whole image, 
   * termination epsilon = 1e-7
   **/   
   void reset();
   
  /**
   * @brief Computes the real pixel coordinates of an ideal point observed from 
   *        the normalized (canonical) camera.
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] src_pt Input 2D ideal point
   * @param[out] dest_pt Output 2D real image point
   * 
   * \warning No checks are made to verify whether the output pixel is inside the image
   */    
  template < typename _T >
  inline void denormalize( const _T src_pt[2], _T dest_pt[2] ) const;
  
  /**
   * @brief Computes the real pixel coordinates of a ideal point observed from 
   *        the normalized (canonical) camera. The distortion parameters are not used.
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] src_pt Input 2D ideal point
   * @param[out] dest_pt Output 2D real image point
   * 
   * \warning No checks are made to verify whether the output pixel is inside the image.
   * Use this function for efficiency reasons if the distortion parameters are zero or negligible.
   */
  template < typename _T >
  inline void denormalizeWithoutDistortion( const _T src_pt[2], _T dest_pt[2] ) const;
                                                                  
  /**
   * @brief Computes the ideal point coordinates in the the normalized (canonical) camera from 
   *        a point observed from the real camera.
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] src_pt Input 2D real image point
   * @param[out] dest_pt Output 2D ideal point
   * 
   * Note about the used undistortion method:
   * 
   * The radial-tangential distortion of a real camera can be basically 
   * modeled as a function that depends on the undistorted point position p, i.e.:
   * 
   * p_dist = f(p,p) = radial(p)*p + tangential(p).
   * 
   * It is not possible to analytically invert this function, so we need to find an approximation. 
   * Differently from other implementations (e.g., OpenCV) that use a first order approximation 
   * to iterativelly estimate the normalized point, we employ a full second order approximation, iterativelly
   * solving the function:
   * 
   * p_cur' = ( I - J(p_cur, p_dist) )^-1 * (inv_f(p_cur, p_dist) - J(p_cur, p_dist) * p_cur;
   * 
   * Where p_cur is the current estimation, p_cur' is the next estimation, inv_f is the inverse 
   * of the function f computed in p_cur and p_dist and J its Jacobian.
   * For the details, take a look in the code (functions denormalize() and 
   * computeNormalizationFunction()). We have found that this method provides more accurate results.
   * 
   * \warning No checks are made to verify whether the input pixel is inside the image
   */
  template < typename _T >
  inline void normalize( const _T src_pt[2], _T dest_pt[2] ) const;
  
  
  /**
   * @brief Computes the ideal point coordinates in the the normalized (canonical) camera from 
   *        a point observed from the real camera. The distortion parameters are not used.
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] src_pt Input 2D real image point
   * @param[out] dest_pt Output 2D ideal point
   * 
   * \warning No checks are made to verify whether the input pixel is inside the image.
   * Use this function for efficiency reasons if the distortion parameters are zero or negligible.
   */  
  template < typename _T >
  inline void normalizeWithoutDistortion( const _T src_pt[2], _T dest_pt[2] ) const;

  /**
   * @brief Projects a single 3D scene point in the image plane, assuming
   *        the point in the camera reference frame.
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] scene_pt Input 3D scene point
   * @param[out] img_pt Output 2D image point
   */    
  template< typename _T >
  inline void project( const _T scene_pt[3], _T img_pt[2] ) const;
                                               
  /**
   * @brief Projects a single 3D scene point in the image plane, assuming
   *        the point in the camera reference frame.
   *        The distortion parameters are not used.
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] scene_pt Input 3D scene point
   * @param[out] img_pt Output 2D image point 
   * 
   * \warning No checks are made to verify whether the input pixel is inside the image.
   * \note Use this function for efficiency reasons if the distortion parameters are zero or negligible.
   */ 
  template< typename _T >
  inline void projectWithoutDistortion( const _T scene_pt[3], _T img_pt[2] ) const;

  /**
   * @brief Projects a single 3D scene point in the image plane, assuming
   *        the point in the camera reference frame.
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] scene_pt Input 3D scene point
   * @param[out] img_pt Output 2D image point
   * @param[out] depth Output z coordinate of the point in the camera reference frame
   * 
   * \warning No checks are made to verify whether the input pixel is inside the image.
   */    
  template< typename _T >
  inline void project( const _T scene_pt[3], _T img_pt[2], _T &depth ) const;
  
                                               
  /**
   * @brief Projects a single 3D scene point in the image plane, assuming
   *        the point in the camera reference frame. 
   *        The distortion parameters are not used.
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] scene_pt Input 3D scene point
   * @param[out] img_pt Output 2D image point 
   * @param[out] depth Output z coordinate of the point in the camera reference frame
   * 
   * \warning No checks are made to verify whether the input pixel is inside the image.
   * \note Use this function for efficiency reasons if the distortion parameters are zero or negligible.
   */ 
  template< typename _T >
  inline void projectWithoutDistortion( const _T scene_pt[3], _T img_pt[2], _T &depth ) const;

  // TODO Add documentation
  template < typename _T >
  inline void rTProject( const Eigen::Matrix< _T, 3, 3 > &r_mat,
                         const Eigen::Matrix< _T, 3, 1 > &t_vec,
                         const _T scene_pt[3], _T img_pt[2] ) const;

  // TODO Add documentation
  template < typename _T >
  inline void rTProjectWithoutDistortion( const Eigen::Matrix< _T, 3, 3 > &r_mat,
                                          const Eigen::Matrix< _T, 3, 1 > &t_vec,
                                          const _T scene_pt[3], _T img_pt[2] ) const;

  /**
   * @brief Projects a single 3D scene point in the image plane, using the 
   *        transformation provided in input. The rotation is expressed using the
   *        angle-axis notation.
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] r_vec Rotation vector, in exponential notation (angle-axis)
   * @param[in] t_vec Translation vector
   * @param[in] scene_pt Input 3D scene point
   * @param[out] img_pt Output 2D image point
   * 
   * \warning No checks are made to verify whether the input pixel is inside the image.
   */
  template < typename _T >
  inline void angAxRTProject( const _T r_vec[3], const _T t_vec[3],
                              const _T scene_pt[3], _T img_pt[2] ) const;
  
  /**
   * @brief Projects a single 3D scene point in the image plane, using the 
   *        transformation provided in input. The rotation is expressed using the
   *        angle-axis notation, the distortion parameters are not used.
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] r_vec Rotation vector, in exponential notation (angle-axis)
   * @param[in] t_vec Translation vector
   * @param[in] scene_pt Input 3D scene point
   * @param[out] img_pt Output 2D image point
   * 
   * \note Use this function for efficiency reasons if the distortion parameters are zero or negligible.
   * \warning No checks are made to verify whether the input pixel is inside the image.
   */
  template < typename _T >
  inline void angAxRTProjectWithoutDistortion( const _T r_vec[3], const _T t_vec[3],
                                               const _T scene_pt[3], _T img_pt[2] ) const;
  
  /**
   * 
   * @brief Projects a single 3D scene point in the image plane, using the 
   *        transformation provided in input. The rotation is expressed using a
   *        quaternion
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] r_quat Quaternion that represents the rotation. 
   * @param[in] t_vec Translation vector
   * @param[in] scene_pt Input 3D scene point
   * @param[out] img_pt Output 2D image point
   * 
   * \note It is not assumed that r_quat has unit norm, but it is assumed that the norm is non-zero.
   * \warning No checks are made to verify whether the input pixel is inside the image.
   */
  template < typename _T >
  inline void quatRTProject( const _T r_quat[4], const _T t_vec[3],
                             const _T scene_pt[3], _T img_pt[2] ) const;

  /**
   * 
   * @brief Projects a single 3D scene point in the image plane, using the 
   *        transformation provided in input. The rotation is expressed using a
   *        quaternion, the distortion parameters are not used.
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] r_quat Quaternion that represents the rotation. 
   * @param[in] t_vec Translation vector
   * @param[in] scene_pt Input 3D scene point
   * @param[out] img_pt Output 2D image point
   * 
   * \warning No checks are made to verify whether the input pixel is inside the image.
   * \note IfIt is not assumed that r_quat has unit norm, but it is assumed that the norm is non-zero.
   * Use this function for efficiency reasons if the distortion parameters are zero or negligible.
   */
  template < typename _T >
  inline void quatRTProjectWithoutDistortion( const _T r_quat[4], const _T t_vec[3],
                                              const _T scene_pt[3], _T img_pt[2] ) const;
                                                                      
  /**
   * 
   * @brief Projects a single 3D scene point in the image plane, using the 
   *        transformation provided in input. The rotation is expressed using a
   *        quaternion
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] r_quat Quaternion that represents the rotation. 
   * @param[in] t_vec Translation vector
   * @param[in] scene_pt Input 3D scene point
   * @param[out] img_pt Output 2D image point
   * 
   * \warning No checks are made to verify whether the input pixel is inside the image.
   */
  template < typename _T >
  inline void quatRTProject( const Eigen::Quaternion<_T> &r_quat,
                             const Eigen::Matrix< _T, 3, 1 > &t_vec,
                             const _T scene_pt[3], _T img_pt[2]) const;

  /**
   * 
   * @brief Projects a single 3D scene point in the image plane, using the 
   *        transformation provided in input. The rotation is expressed using a
   *        quaternion, the distortion parameters are not used.
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] r_quat Quaternion that represents the rotation. 
   * @param[in] t_vec Translation vector
   * @param[in] scene_pt Input 3D scene point
   * @param[out] img_pt Output 2D image point
   * 
   * \warning No checks are made to verify whether the input pixel is inside the image.
   * Use this function for efficiency reasons if the distortion parameters are zero or negligible.
   */
  template < typename _T >
  inline void quatRTProjectWithoutDistortion( const Eigen::Quaternion<_T> &r_quat,
                                              const Eigen::Matrix< _T, 3, 1 > &t_vec,
                                              const _T scene_pt[3], _T img_pt[2]) const;

  // TODO Add documentation
  template < typename _T >
  inline void rTProject( const Eigen::Matrix< _T, 3, 3 > &r_mat,
                         const Eigen::Matrix< _T, 3, 1 > &t_vec,
                         const _T scene_pt[3], _T img_pt[2],
                         _T &depth ) const;

  // TODO Add documentation
  template < typename _T >
  inline void rTProjectWithoutDistortion( const Eigen::Matrix< _T, 3, 3 > &r_mat,
                                          const Eigen::Matrix< _T, 3, 1 > &t_vec,
                                          const _T scene_pt[3], _T img_pt[2], _T &depth ) const;

  /**
   * @brief Projects a single 3D scene point in the image plane, using the 
   *        transformation provided in input. The rotation is expressed using the
   *        angle-axis notation.
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] r_vec Rotation vector, in exponential notation (angle-axis)
   * @param[in] t_vec Translation vector
   * @param[in] scene_pt Input 3D scene point
   * @param[out] img_pt Output 2D image point
   * @param[out] depth Output z coordinate of the point in the camera reference frame
   * 
   * \warning No checks are made to verify whether the input pixel is inside the image.
   */
  template < typename _T >
  inline void angAxRTProject( const _T r_vec[3], const _T t_vec[3],
                              const _T scene_pt[3], _T img_pt[2], _T &depth ) const;
  
  /**
   * @brief Projects a single 3D scene point in the image plane, using the 
   *        transformation provided in input. The rotation is expressed using the
   *        angle-axis notation, the distortion parameters are not used.
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] r_vec Rotation vector, in exponential notation (angle-axis)
   * @param[in] t_vec Translation vector
   * @param[in] scene_pt Input 3D scene point
   * @param[out] img_pt Output 2D image point
   * @param[out] depth Output z coordinate of the point in the camera reference frame
   * 
   * \warning No checks are made to verify whether the input pixel is inside the image.
   * \note Use this function for efficiency reasons if the distortion parameters are zero or negligible.
   */
  template < typename _T >
  inline void angAxRTProjectWithoutDistortion( const _T r_vec[3], const _T t_vec[3],
                                               const _T scene_pt[3], _T img_pt[2], _T &depth ) const;
  
  /**
   * 
   * @brief Projects a single 3D scene point in the image plane, using the 
   *        transformation provided in input. The rotation is expressed using a
   *        quaternion
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] r_quat Quaternion that represents the rotation. 
   * @param[in] t_vec Translation vector
   * @param[in] scene_pt Input 3D scene point
   * @param[out] img_pt Output 2D image point
   * @param[out] depth Output z coordinate of the point in the camera reference frame
   * 
   * \note If the point is projected outside the image, or it is not possible to project 
   * the point or it is occluded, output coordinates and depth are set to -1.
   * It is not assumed that r_quat has unit norm, but it is assumed that the norm is non-zero.
   */
  template < typename _T >
  inline void quatRTProject( const _T r_quat[4], const _T t_vec[3],
                             const _T scene_pt[3], _T img_pt[2], _T &depth ) const;

  /**
   * 
   * @brief Projects a single 3D scene point in the image plane, using the 
   *        transformation provided in input. The rotation is expressed using a
   *        quaternion, the distortion parameters are not used.
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] r_quat Quaternion that represents the rotation. 
   * @param[in] t_vec Translation vector
   * @param[in] scene_pt Input 3D scene point
   * @param[out] img_pt Output 2D image point
   * @param[out] depth Output z coordinate of the point in the camera reference frame
   * 
   * \warning No checks are made to verify whether the input pixel is inside the image.
   * \note It is not assumed that r_quat has unit norm, but it is assumed that the norm is non-zero.
   * Use this function for efficiency reasons if the distortion parameters are zero or negligible.
   */
  template < typename _T >
  inline void quatRTProjectWithoutDistortion( const _T r_quat[4], const _T t_vec[3],
                                              const _T scene_pt[3], _T img_pt[2], _T &depth ) const;
                                                                      
  /**
   * 
   * @brief Projects a single 3D scene point in the image plane, using the 
   *        transformation provided in input. The rotation is expressed using a
   *        quaternion
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] r_quat Quaternion that represents the rotation. 
   * @param[in] t_vec Translation vector
   * @param[in] scene_pt Input 3D scene point
   * @param[out] img_pt Output 2D image point
   * @param[out] depth Output z coordinate of the point in the camera reference frame
   * 
   * \warning No checks are made to verify whether the input pixel is inside the image.
   */
  template < typename _T >
  inline void quatRTProject( const Eigen::Quaternion<_T> &r_quat,
                             const Eigen::Matrix< _T, 3, 1 > &t_vec,
                             const _T scene_pt[3], _T img_pt[2], _T &depth ) const;

  /**
   * 
   * @brief Projects a single 3D scene point in the image plane, using the 
   *        transformation provided in input. The rotation is expressed using a
   *        quaternion, the distortion parameters are not used.

   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] r_quat Quaternion that represents the rotation. 
   * @param[in] t_vec Translation vector
   * @param[in] scene_pt Input 3D scene point
   * @param[out] img_pt Output 2D image point
   * @param[out] depth Output z coordinate of the point in the camera reference frame
   * 
   * \warning No checks are made to verify whether the input pixel is inside the image.
   * Use this function for efficiency reasons if the distortion parameters are zero or negligible.
   */
  template < typename _T >
  inline void quatRTProjectWithoutDistortion( const Eigen::Quaternion<_T> &r_quat,
                                              const Eigen::Matrix< _T, 3, 1 > &t_vec,
                                              const _T scene_pt[3], _T img_pt[2], _T &depth ) const;


  /**
   * @brief Unprojects a single image point along with its depth
   *        into a 3D scene point, assuming the point is the camera reference frame.

   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] img_pt Input 2D image point
   * @param[in] depth Input z coordinate of the point in the camera reference frame
   * @param[in] scene_pt Output 3D scene point
   * 
   * \note See normalize() function
   * \warning No checks are made to verify whether the input pixel is inside the image
   */    
  template< typename _T >
  inline void unproject( const _T img_pt[2], const _T &depth,
                         _T scene_pt[3] ) const;
  
                                               
  /**
   * @brief Unprojects a single image point along with its depth
   *        into a 3D scene point, assuming the point is the camera reference frame.
   *        The distortion parameters are not used.
   * 
   * @tparam _T Data type (e.g., float, double, ...)
   * 
   * @param[in] img_pt Input 2D image point
   * @param[in] depth Input z coordinate of the point in the camera reference frame
   * @param[out] scene_pt Output 3D scene point
   * 
   * \warning No checks are made to verify whether the input pixel is inside the image
   * Use this function for efficiency reasons if the distortion parameters are zero or negligible.
   */ 
  template< typename _T >
  inline void unprojectWithoutDistortion( const _T img_pt[2], const _T &depth, _T scene_pt[3] ) const;
                                                                  
private:

  void updateCurrentParamters();
  template < typename _T > inline void invertMat2X2( _T m[4] ) const;
  template < typename _T > inline void computeNormalizationFunction( const _T dist_pt[2], _T cur_pt[2], _T f[2], _T j[4] ) const;
  
  int orig_img_width_ = 1.0, orig_img_height_ = 1.0, 
      img_width_ = 1.0, img_height_ = 1.0;
  double size_scale_factor_ = 1.0;
  cv::Rect orig_roi_= cv::Rect(0,0,1,1), roi_= cv::Rect(0,0,1,1);
  bool has_dist_coeff_ = false, roi_enabled_ = false;
  double orig_fx_ = 1.0, orig_fy_ = 1.0, orig_cx_ = 0.0, orig_cy_ = 0.0, 
         fx_ = 1.0, fy_ = 1.0, inv_fx_ = 1.0, inv_fy_ = 1.0, cx_ = 0.0, cy_ = 0.0;
  double dist_px_ = 0.0, dist_py_ = 0.0, 
         dist_k0_ = 0.0, dist_k1_ = 0.0, dist_k2_ = 0.0, dist_k3_ = 0.0, dist_k4_ = 0.0, dist_k5_ = 0.0;

  double term_epsilon_ = 1e-7;
};


/**  Utility function, see PinholeCameraModel::read() */
inline void read(const YAML::Node &in, PinholeCameraModel &obj )
{
  obj.read(in);
}

/**  Utility function, see PinholeCameraModel::write() */
inline void write( YAML::Node &out, const PinholeCameraModel &obj )
{
  obj.write(out);
}

/**  Utility function, see PinholeCameraModel::readFromFile() */
inline bool readFromFile( const std::string &filename, PinholeCameraModel &obj )
{
  return obj.readFromFile(filename);
}

/**  Utility function, see PinholeCameraModel::writeToFile() */
inline bool writeToFile( const std::string &filename, const PinholeCameraModel &obj )
{
  return obj.writeToFile(filename);
}

/** Redirect to to an output stream a readable output of an instance of PinholeCameraModel
 *
 * If a size scale factor other than one has been set  (see setSizeScaleFactor()) or a region of interest (RoI) is enabled
 * (see enableRegionOfInterest() and  setRegionOfInterest()), the written parameters refer to a camera associated with
 * the -scaled- RoI
 *
 * Usage example;
 *
 * PinholeCameraModel model;
 * ...
 * std::cout<<mode<<std::endl;
 **/
inline std::ostream& operator<<(std::ostream& os, const PinholeCameraModel& obj)
{
  std::cout<<"===Intrinsic Parameters==="<<std::endl;

  os<<"Image size (width, height) : ["<<obj.imgWidth()<<", "<<obj.imgHeight()<<"]"<<std::endl;
  os<<"Focal lenghts (in pixels size, along x and y directions) : ["<<obj.fx()<<", "<<obj.fy()<<"]"<<std::endl;
  os<<"Principal point (x,y) : ["<<obj.cx()<<", "<<obj.cy()<<"]"<<std::endl;

  std::cout<<"===Distortion Coefficients==="<<std::endl;

  os<<"k0 : "<<obj.distK0()<<std::endl;
  os<<"k1 : "<<obj.distK1()<<std::endl;
  os<<"k2 : "<<obj.distK2()<<std::endl;
  os<<"k3 : "<<obj.distK3()<<std::endl;
  os<<"k4 : "<<obj.distK4()<<std::endl;
  os<<"k5 : "<<obj.distK5()<<std::endl;

  os<<"px : "<<obj.distPx()<<std::endl;
  os<<"py : "<<obj.distPy()<<std::endl;

  return os;
}

/* Implementation */
  
//template < typename _T >
//inline void PinholeCameraModel::denormalize( const _T src_pt[2],
//                                             _T dest_pt[2] ) const
//{
//  const _T &x = src_pt[0], &y = src_pt[1];
//
//  _T x2 = x * x, y2 = y * y, xy_dist, r2, r4, r6;
//  _T l_radial, h_radial, radial, tangential_x, tangential_y;
//
//  r2 = x2 + y2;
//  r6 = r2 * ( r4 = r2 * r2 );
//
//  l_radial = _T(1.0) + _T(dist_k0_) * r2 + _T(dist_k1_) * r4 + _T(dist_k2_) * r6;
//  h_radial = _T(1.0)/(_T(1.0) + _T(dist_k3_)*r2 + _T(dist_k4_)*r4 + _T(dist_k5_)*r6);
//  radial = l_radial * h_radial;
//  xy_dist = _T(2.0) * x * y;
//  tangential_x = xy_dist * _T(dist_px_) + _T(dist_py_) * ( r2 + _T(2.0) * x2 );
//  tangential_y = xy_dist * _T(dist_py_) + _T(dist_px_) * ( r2 + _T(2.0) * y2 );
//
//  _T x_dist = x * radial + tangential_x;
//  _T y_dist = y * radial + tangential_y;
//
//  dest_pt[0] = x_dist * _T(fx_) + _T(cx_);
//  dest_pt[1] = y_dist * _T(fy_) + _T(cy_);
//}

template < typename _T >
inline void PinholeCameraModel::denormalize( const _T src_pt[2],
                                             _T dest_pt[2] ) const
{
  const _T &x = src_pt[0], &y = src_pt[1];

  _T x2 = x * x, y2 = y * y, xy_dist, r2, r4, r6;
  _T l_radial, h_radial, radial, tangential_x, tangential_y;

  r2 = x2 + y2;
  r6 = r2 * ( r4 = r2 * r2 );

  l_radial = 1.0 + dist_k0_ * r2 + dist_k1_ * r4 + dist_k2_ * r6;
  h_radial = 1.0/(1.0 + dist_k3_*r2 + dist_k4_*r4 + dist_k5_*r6);
  radial = l_radial * h_radial;
  xy_dist = 2.0 * x * y;
  tangential_x = xy_dist * dist_px_ + dist_py_ * ( r2 + 2.0 * x2 );
  tangential_y = xy_dist * dist_py_ + dist_px_ * ( r2 + 2.0 * y2 );

  _T x_dist = x * radial + tangential_x;
  _T y_dist = y * radial + tangential_y;

  dest_pt[0] = x_dist * fx_ + cx_;
  dest_pt[1] = y_dist * fy_ + cy_;
}

template < typename _T >
inline void PinholeCameraModel::denormalizeWithoutDistortion( const _T src_pt[2],
                                                              _T dest_pt[2] ) const
{
  dest_pt[0] = src_pt[0] * fx_ + cx_;
  dest_pt[1] = src_pt[1] * fy_ + cy_;
}

template < typename _T >
inline void PinholeCameraModel::normalize( const _T src_pt[2],
                                             _T dest_pt[2] )  const
{
  _T dist_pt[2], prev_pt[2], f[2], j[4];

  dist_pt[0] = ( src_pt[0] - _T(cx_) ) * _T(inv_fx_);
  dist_pt[1] = ( src_pt[1] - _T(cy_) ) * _T(inv_fy_);

  prev_pt[0] = dest_pt[0] = dist_pt[0];
  prev_pt[1] = dest_pt[1] = dist_pt[1];

  for(int iter = 0; iter < 5; iter++)
  {
    computeNormalizationFunction( dist_pt, dest_pt, f, j );

    _T tmp_u = f[0] - j[0]*dest_pt[0] - j[1]*dest_pt[1];
    _T tmp_v = f[1] - j[2]*dest_pt[0] - j[3]*dest_pt[1];

    j[0] = _T(1.0) - j[0];
    j[1] = -j[1];
    j[2] = -j[2];
    j[3] = _T(1.0) - j[3];

    invertMat2X2( j );

    dest_pt[0] = j[0]*tmp_u + j[1]*tmp_v;
    dest_pt[1] = j[2]*tmp_u + j[3]*tmp_v;

    // Exit if convergence
    if( std::abs<_T>(prev_pt[0] - dest_pt[0]) < _T(term_epsilon_) &&
        std::abs<_T>(prev_pt[1] - dest_pt[1]) < _T(term_epsilon_) )
      break;

    prev_pt[0] = dest_pt[0];
    prev_pt[1] = dest_pt[1];
  }
}

template < typename _T >
inline void PinholeCameraModel::normalizeWithoutDistortion( const _T src_pt[2],
                                                            _T dest_pt[2] )  const
{
  dest_pt[0] = ( src_pt[0] - cx_ ) * inv_fx_;
  dest_pt[1] = ( src_pt[1] - cy_ ) * inv_fy_;
}

template< typename _T >
inline void PinholeCameraModel::project( const _T scene_pt[3], _T img_pt[2] ) const
{
  _T inv_z = 1.0/scene_pt[2];
  _T norm_img_pt[2] = {scene_pt[0]*inv_z, scene_pt[1]*inv_z};

  denormalize( norm_img_pt, img_pt );
}

template< typename _T >
inline void PinholeCameraModel::projectWithoutDistortion( const _T scene_pt[3], _T img_pt[2] ) const
{
  _T inv_z = 1.0/scene_pt[2];
  _T norm_img_pt[2] = {scene_pt[0]*inv_z, scene_pt[1]*inv_z};
  
  denormalizeWithoutDistortion( norm_img_pt, img_pt );
}

template< typename _T >
inline void PinholeCameraModel::project( const _T scene_pt[3],
                                         _T img_pt[2], _T &depth ) const
{
  depth = scene_pt[2];
  project( scene_pt, img_pt );
}

template< typename _T >
inline void PinholeCameraModel::projectWithoutDistortion( const _T scene_pt[3], _T img_pt[2],
                                                          _T &depth ) const
{
  depth = scene_pt[2];
  projectWithoutDistortion( scene_pt, img_pt );
}

template < typename _T >
inline void PinholeCameraModel::rTProject( const Eigen::Matrix< _T, 3, 3 > &r_mat,
                                           const Eigen::Matrix< _T, 3, 1 > &t_vec,
                                           const _T scene_pt[3], _T img_pt[2] ) const
{
  Eigen::Matrix< _T, 3, 1 > transf_pt = r_mat*Eigen::Map< const Eigen::Matrix< _T, 3, 1 > >(scene_pt) + t_vec;
  project( transf_pt.data(), img_pt );
}

template < typename _T >
inline void PinholeCameraModel::rTProjectWithoutDistortion( const Eigen::Matrix< _T, 3, 3 > &r_mat,
                                                            const Eigen::Matrix< _T, 3, 1 > &t_vec,
                                                            const _T scene_pt[3], _T img_pt[2] ) const
{
  Eigen::Matrix< _T, 3, 1 > transf_pt = r_mat*Eigen::Map< const Eigen::Matrix< _T, 3, 1 > >(scene_pt) + t_vec;
  projectWithoutDistortion( transf_pt.data(), img_pt );
}

template < typename _T >
inline void PinholeCameraModel::angAxRTProject( const _T r_vec[3], const _T t_vec[3],
                                                const _T scene_pt[3], _T img_pt[2] ) const
{
  _T transf_pt[3];
  // ceres::AngleAxisRotatePoint uses the rodriguez formula only away from zero
  ceres::AngleAxisRotatePoint(r_vec, scene_pt, transf_pt);
  for( int i = 0; i < 3; i++ )
    transf_pt[i] += t_vec[i];

  project( transf_pt, img_pt );
}


template < typename _T >
inline void PinholeCameraModel::angAxRTProjectWithoutDistortion( const _T r_vec[3], const _T t_vec[3],
                                                                 const _T scene_pt[3], _T img_pt[2] ) const
{
  _T transf_pt[3];
  // ceres::AngleAxisRotatePoint uses the rodriguez formula only away from zero
  ceres::AngleAxisRotatePoint(r_vec, scene_pt, transf_pt );
  for( int i = 0; i < 3; i++ ) transf_pt[i] += t_vec[i];

  projectWithoutDistortion( transf_pt, img_pt );
}


template < typename _T >
inline void PinholeCameraModel::quatRTProject( const _T r_quat[4], const _T t_vec[3],
                                               const _T scene_pt[3], _T img_pt[2] ) const
{
  _T transf_pt[3];
  ceres::QuaternionRotatePoint( r_quat, scene_pt, transf_pt );
  for( int i = 0; i < 3; i++ ) transf_pt[i] += t_vec[i];

  project( transf_pt, img_pt );
}

template < typename _T >
inline void PinholeCameraModel::quatRTProjectWithoutDistortion( const _T r_quat[4], const _T t_vec[3],
                                                                const _T scene_pt[3], _T img_pt[2] ) const
{
  _T transf_pt[3];
  ceres::QuaternionRotatePoint( r_quat, scene_pt, transf_pt );
  for( int i = 0; i < 3; i++ ) transf_pt[i] += t_vec[i];

  projectWithoutDistortion( transf_pt, img_pt );
}


template < typename _T >
inline void PinholeCameraModel::quatRTProject ( const Eigen::Quaternion< _T >& r_quat,
                                                const Eigen::Matrix< _T, 3, 1  >& t_vec,
                                                const _T scene_pt[3], _T img_pt[2] ) const
{
  Eigen::Matrix< _T, 3, 1 > transf_pt = r_quat*Eigen::Map< const Eigen::Matrix< _T, 3, 1 > >(scene_pt) + t_vec;
  project( transf_pt.data(), img_pt );
}


template < typename _T >
inline void PinholeCameraModel::quatRTProjectWithoutDistortion ( const Eigen::Quaternion< _T >& r_quat,
                                                                 const Eigen::Matrix< _T, 3, 1 >& t_vec,
                                                                 const _T scene_pt[3], _T img_pt[2] ) const
{
  Eigen::Matrix< _T, 3, 1 > transf_pt = r_quat*Eigen::Map< const Eigen::Matrix< _T, 3, 1 > >(scene_pt) + t_vec;
  projectWithoutDistortion( transf_pt.data(), img_pt );
}

template < typename _T >
inline void PinholeCameraModel::rTProject( const Eigen::Matrix< _T, 3, 3 > &r_mat,
                                           const Eigen::Matrix< _T, 3, 1 > &t_vec,
                                           const _T scene_pt[3], _T img_pt[2],
                                           _T &depth ) const
{
  Eigen::Matrix< _T, 3, 1 > transf_pt = r_mat*Eigen::Map< const Eigen::Matrix< _T, 3, 1 > >(scene_pt) + t_vec;
  project( transf_pt.data(), img_pt, depth );
}

template < typename _T >
inline void PinholeCameraModel::rTProjectWithoutDistortion( const Eigen::Matrix< _T, 3, 3 > &r_mat,
                                                            const Eigen::Matrix< _T, 3, 1 > &t_vec,
                                                            const _T scene_pt[3], _T img_pt[2],
                                                            _T &depth ) const
{
  Eigen::Matrix< _T, 3, 1 > transf_pt = r_mat*Eigen::Map< const Eigen::Matrix< _T, 3, 1 > >(scene_pt) + t_vec;
  projectWithoutDistortion( transf_pt.data(), img_pt, depth );
}


template < typename _T >
inline void PinholeCameraModel::angAxRTProject( const _T r_vec[3], const _T t_vec[3],
                                                const _T scene_pt[3], _T img_pt[2],
                                                _T &depth ) const
{
  _T transf_pt[3];
  // ceres::AngleAxisRotatePoint uses the rodriguez formula only away from zero
  ceres::AngleAxisRotatePoint(r_vec, scene_pt, transf_pt );
  for( int i = 0; i < 3; i++ )
    transf_pt[i] += t_vec[i];
  
  project( transf_pt, img_pt, depth );
}

template < typename _T >
inline void PinholeCameraModel::angAxRTProjectWithoutDistortion( const _T r_vec[3], const _T t_vec[3],
                                                                 const _T scene_pt[3], _T img_pt[2],
                                                                 _T &depth ) const
{
  _T transf_pt[3];
  // ceres::AngleAxisRotatePoint uses the rodriguez formula only away from zero
  ceres::AngleAxisRotatePoint(r_vec, scene_pt, transf_pt );
  for( int i = 0; i < 3; i++ )
    transf_pt[i] += t_vec[i];
  
  projectWithoutDistortion( transf_pt, img_pt, depth );
}

template < typename _T >
inline void PinholeCameraModel::quatRTProject( const _T r_quat[4], const _T t_vec[3],
                                               const _T scene_pt[3], _T img_pt[2],
                                               _T &depth ) const
{
  _T transf_pt[3];
  ceres::QuaternionRotatePoint( r_quat, scene_pt, transf_pt );
  for( int i = 0; i < 3; i++ )
    transf_pt[i] += t_vec[i];
  
  project( transf_pt, img_pt, depth );
}

template < typename _T >
inline void PinholeCameraModel::quatRTProjectWithoutDistortion( const _T r_quat[4], const _T t_vec[3],
                                                                const _T scene_pt[3], _T img_pt[2],
                                                                _T &depth ) const
{
  _T transf_pt[3];
  ceres::QuaternionRotatePoint( r_quat, scene_pt, transf_pt );
  for( int i = 0; i < 3; i++ )
    transf_pt[i] += t_vec[i];
  
  projectWithoutDistortion( transf_pt, img_pt, depth );
}

template < typename _T >
inline void PinholeCameraModel::quatRTProject ( const Eigen::Quaternion< _T >& r_quat,
                                                const Eigen::Matrix< _T, 3, 1  >& t_vec,
                                                const _T scene_pt[3], _T img_pt[2], _T& depth ) const
{
  Eigen::Matrix< _T, 3, 1 > transf_pt = r_quat*Eigen::Map< const Eigen::Matrix< _T, 3, 1 > >(scene_pt) + t_vec;
  project( transf_pt.data(), img_pt, depth );
}

template < typename _T >
inline void PinholeCameraModel::quatRTProjectWithoutDistortion ( const Eigen::Quaternion< _T >& r_quat,
                                                                 const Eigen::Matrix< _T, 3, 1  >& t_vec,
                                                                 const _T scene_pt[3], _T img_pt[2],
                                                                 _T& depth ) const
{
  Eigen::Matrix< _T, 3, 1 > transf_pt = r_quat*Eigen::Map< const Eigen::Matrix< _T, 3, 1 > >(scene_pt) + t_vec;
  projectWithoutDistortion( transf_pt.data(), img_pt, depth );
}


template < typename _T >
inline void PinholeCameraModel::invertMat2X2( _T m[4] ) const
{
  _T inv_det = 1.0/(m[0]*m[3] - m[1]*m[2]), tmp = m[0];
  m[0] = inv_det*m[3];
  m[1] *= -inv_det;
  m[2] *= -inv_det;
  m[3] = inv_det*tmp;
}

template< typename _T >
inline void PinholeCameraModel::unproject( const _T img_pt[2], const _T &depth,
                                           _T scene_pt[3] ) const
{
  _T norm_img_pt[2];
  normalize( img_pt, norm_img_pt );
  scene_pt[0] = norm_img_pt[0]*depth;
  scene_pt[1] = norm_img_pt[1]*depth;
  scene_pt[2] = depth;
}

template< typename _T >
inline void PinholeCameraModel::unprojectWithoutDistortion( const _T img_pt[2], const _T &depth,
                                                            _T scene_pt[3] ) const
{
  _T norm_img_pt[2];
  normalizeWithoutDistortion( img_pt, norm_img_pt );
  scene_pt[0] = norm_img_pt[0]*depth;
  scene_pt[1] = norm_img_pt[1]*depth;
  scene_pt[2] = depth;
}

template < typename _T >
inline void PinholeCameraModel::computeNormalizationFunction( const _T dist_pt[2], _T cur_pt[2],
                                                              _T f[2], _T j[4] ) const
{
  _T &x = cur_pt[0], &y = cur_pt[1];
  const _T &dist_x = dist_pt[0], &dist_y = dist_pt[1];
  
  _T x2 = x * x, y2 = y * y,
     xyd, r2, r4, r6, l_radial,// h_radial, 
     il_radial, il_radial2, ih_radial, tangential_x, tangential_y,
     d_l_radial_dx, d_l_radial_dy, d_ih_radial_dx, d_ih_radial_dy,
     d_tangential_x_dx, d_tangential_x_dy, d_tangential_y_dx, d_tangential_y_dy;
        
  r2 = x2 + y2;
  r6 = r2 * ( r4 = r2 * r2 );
  
  l_radial = _T(1.0) + _T(dist_k0_)*r2 + _T(dist_k1_) * r4 + _T(dist_k2_) * r6;
  ih_radial = _T(1.0) + _T(dist_k3_)*r2 + _T(dist_k4_)*r4 + _T(dist_k5_)*r6;
  
  il_radial = _T(1.0)/l_radial;
//   h_radial = _T(1.0)/ih_radial;
  
  il_radial2 = _T(1.0)/(l_radial*l_radial);
  
  d_l_radial_dx = _T(6.0)*_T(dist_k2_)*x*r4 + _T(4.0)*_T(dist_k1_)*x*r2 + _T(2.0)*_T(dist_k0_)*x;
  d_l_radial_dy = _T(6.0)*_T(dist_k2_)*y*r4 + _T(4.0)*_T(dist_k1_)*y*r2 + _T(2.0)*_T(dist_k0_)*y;
  
  d_ih_radial_dx = _T(6.0)*_T(dist_k5_)*x*r4 + _T(4.0)*_T(dist_k4_)*x*r2 + _T(2.0)*_T(dist_k3_)*x;
  d_ih_radial_dy = _T(6.0)*_T(dist_k5_)*y*r4 + _T(4.0)*_T(dist_k4_)*y*r2 + _T(2.0)*_T(dist_k3_)*y;
  
  xyd = _T(2.0)*x*y;
  tangential_x = xyd*_T(dist_px_) + _T(dist_py_)*( r2 + _T(2.0)*x2 );
  tangential_y = xyd*_T(dist_py_) + _T(dist_px_)*( r2 + _T(2.0)*y2 );
  
  d_tangential_x_dx = _T(2.0)*_T(dist_px_)*y + _T(6.0)*_T(dist_py_)*x;
  d_tangential_x_dy = d_tangential_y_dx = _T(2.0)*_T(dist_py_)*y + _T(2.0)*_T(dist_px_)*x;
  d_tangential_y_dy = _T(6.0)*_T(dist_px_)*y + _T(2.0)*_T(dist_py_)*x;
  
  f[0] = ( dist_x - tangential_x )*ih_radial*il_radial;
  f[1] = ( dist_y - tangential_y )*ih_radial*il_radial;
  
  j[0] = d_ih_radial_dx*( dist_x - tangential_x )*il_radial + 
        -d_tangential_x_dx*ih_radial*il_radial - 
        ( dist_x - tangential_x )*d_l_radial_dx*ih_radial*il_radial2;
       
  j[1] = d_ih_radial_dy*( dist_x - tangential_x )*il_radial + 
        -d_tangential_x_dy* ih_radial*il_radial - 
        d_l_radial_dy*( dist_x - tangential_x )*ih_radial*il_radial2;
  
  j[2] = d_ih_radial_dx*( dist_y - tangential_y )*il_radial +
         -d_tangential_y_dx * ih_radial * il_radial - 
         d_l_radial_dx*( dist_y - tangential_y )*ih_radial*il_radial2;
        
  j[3] = d_ih_radial_dy*(dist_y - tangential_y )*il_radial + 
         -d_tangential_y_dy * ih_radial * il_radial - 
         d_l_radial_dy*(dist_y- tangential_y )*ih_radial*il_radial2;

}

}
