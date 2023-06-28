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


/* TODO
 * Clean up and switch everithing to float
 *
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <boost/thread/mutex.hpp>

#include "cv_ext/cv_ext.h"

/** @brief Shared pointer typedef */
typedef std::shared_ptr< class RasterObjectModel > RasterObjectModelPtr;

class RasterObjectModel
{
public:
  
  RasterObjectModel();
  ~RasterObjectModel(){};

  enum UoM{ MICRON = 1000000,
            CENTIMILLIMETER = 100000, 
            DECIMILLIMETER = 10000, 
            MILLIMETER = 1000, 
            CENTIMETER = 100, 
            METER = 1 };
            
  
  UoM unitOfMeasure() const { return unit_meas_; };
  cv::Point3f origOffset()const { return orig_offset_; }
  double stepMeters() const { return step_; };
  double minSegmentsLen() const { return min_seg_len_; }

//  void savePrecomputedModelsViews( std::string filename ) const;
//  void loadPrecomputedModelsViews( std::string filename );
//
//  int numPrecomputedModelsViews() const {return precomputed_rq_view_.size(); };
  /**
   * @brief Provide the camera model object used to project the raster to an
   *        image plane
   */
  cv_ext::PinholeCameraModel cameraModel() const { return cam_model_; }

  /**
   * @brief Provide the camera model object used to project the raster to an
   *        image plane
   */
  cv_ext::PinholeCameraModel & cameraModel() { return cam_model_; }
  
  void setUnitOfMeasure( UoM val );
  void setCentroidOrigOffset();
  void setBBCenterOrigOffset();
  void setOrigOffset( cv::Point3f offset );
  void setStepMeters( double s );
  void setMinSegmentsLen( double len );
  void setCamModel( const cv_ext::PinholeCameraModel &cam_model );
    
  virtual bool setModelFile( const std::string& filename ) = 0;
  virtual bool allVisiblePoints() const = 0;
  virtual void computeRaster() = 0;
  virtual void update() = 0;

// TODO Add documentation
  void setBoundingBox( cv_ext::Box3f &bbox )
  {
    bbox_ = bbox;
    custom_bbox_ = bbox_ != orig_bbox_;
    update();
  };

// TODO Add documentation
  void resetBoundingBox()
  {
    bbox_ = orig_bbox_;
    custom_bbox_ = false;
    update();
  };

  /**
 * // TODO Update documentation
 * @brief Provide the object 3D bounding box, represented in the object reference frame
 */
  cv_ext::Box3f getBoundingBox() const { return bbox_; };

  /**
   * // TODO Update documentation
   * @brief Provide the object 3D bounding box, represented in the object reference frame
   */
  cv_ext::Box3f getOrigBoundingBox() const { return orig_bbox_; };


  // TODO Update documentation
  void setModelView( const Eigen::Matrix3d &r_mat, const Eigen::Vector3d &t_vec );

  /**
   * @brief Set the transformation that transforms points from the model frame to the camera frame 
   * 
   * @param[in] r_quat Rotation quaternion
   * @param[in] t_vec Translation vector
   */
  void setModelView( const double r_quat[4], const double t_vec[3] );

  /**
   * @brief Set the transformation that transforms points from the model frame to the camera frame 
   * 
   * @param[in] r_quat Rotation quaternion
   * @param[in] t_vec Translation vector
   */
  void setModelView( const Eigen::Quaterniond &r_quat, const Eigen::Vector3d &t_vec );
  
  /**
   * @brief Set the transformation that transforms points from the model frame to the camera frame 
   * 
   * @param[in] r_vec 3 by 1 rotation vector, in exponential notation (angle-axis)
   * @param[in] t_vec Translation vector
   */
  void setModelView( const cv::Mat_<double> &r_vec, const cv::Mat_<double> &t_vec );



  // TODO Update documentation
  void modelView( Eigen::Matrix3d &r_mat, Eigen::Vector3d &t_vec );

  
  /**
   * @brief Provide the current transformation that transforms points from the model frame to the camera frame 
   * 
   * @param[out] r_quat Rotation quaternion
   * @param[out] t_vec Translation vector
   */
  void modelView( double r_quat[4], double t_vec[3] ) const;
  
  /**
   * @brief Provide the current transformation that transforms points from the model frame to the camera frame 
   * 
   * @param[out] r_quat Rotation quaternion
   * @param[out] t_vec Translation vector
   */
  void modelView( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec ) const;

 /**
   * @brief Provide the current transformation that transforms points from the model frame to the camera frame 
   * 
   * @param[out] r_vec 3 by 1 rotation vector, in exponential notation (angle-axis)
   * @param[out] t_vec Translation vector
   */
  void modelView( cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec ) const;


  virtual const std::vector<cv::Point3f> &getPoints( bool only_visible_points = true ) const = 0;
  virtual const std::vector<cv::Point3f> &getDPoints( bool only_visible_points = true) const = 0;
  virtual const std::vector<cv::Vec6f> &getSegments( bool only_visible_segments = true ) const = 0;
  virtual const std::vector<cv::Point3f> &getDSegments( bool only_visible_segments = true ) const = 0;

  
  /**
   * @brief Project the raster points on the image plane, 
   *        given the current model view ( see setModelView() )
   * 
   * @param[out] proj_pts Output projected points
   * @param[in] only_visible_points If false, project also hidden, possinly occluded points
   * 
   * \note If a point is projected outside the image, or it is not possible to 
   * project the point or it is occluded, its coordinates are set to (-1, -1)
   */
  void projectRasterPoints( std::vector<cv::Point2f> &proj_pts,
                            bool only_visible_points = true ) const;

  // TODO Add documentation
  void projectRasterPoints(  const Eigen::Matrix3d &r_mat, const Eigen::Vector3d &t_vec,
                             std::vector<cv::Point2f> &proj_pts,
                             bool only_visible_points = true ) const;

  // TODO Add documentation
  void projectRasterPoints(  const double r_quat[4], const double t_vec[3],
                             std::vector<cv::Point2f> &proj_pts,
                             bool only_visible_points = true ) const;

  // TODO Add documentation
  void projectRasterPoints(  const Eigen::Quaterniond &r_quat, const Eigen::Vector3d &t_vec,
                             std::vector<cv::Point2f> &proj_pts,
                             bool only_visible_points = true ) const;

  // TODO Add documentation
  void projectRasterPoints(  const cv::Mat_<double> &r_vec, const cv::Mat_<double> &t_vec,
                             std::vector<cv::Point2f> &proj_pts,
                             bool only_visible_points = true ) const;
  
  /**
   * @brief Project the raster points and theirs normal directions on the image plane, 
   *        given the current model view ( see setModelView() )
   * 
   * @param[out]proj_pts Output projected points
   * @param[out]normal_directions Output normal directions in the image plane
   * @param[in] only_visible_points I false, project also hidden, possinly occluded points
   */                      
  void projectRasterPoints( std::vector<cv::Point2f> &proj_pts,
                            std::vector<float> &normal_directions,
                            bool only_visible_points = true ) const;

  // TODO Add documentation
  void projectRasterPoints(  const Eigen::Matrix3d &r_mat, const Eigen::Vector3d &t_vec,
                             std::vector<cv::Point2f> &proj_pts,
                             std::vector<float> &normal_directions,
                             bool only_visible_points = true ) const;

  // TODO Add documentation
  void projectRasterPoints( const double r_quat[4], const double t_vec[3],
                            std::vector<cv::Point2f> &proj_pts,
                            std::vector<float> &normal_directions,
                            bool only_visible_points = true ) const;

  // TODO Add documentation
  void projectRasterPoints( const Eigen::Quaterniond &r_quat, const Eigen::Vector3d &t_vec,
                            std::vector<cv::Point2f> &proj_pts,
                            std::vector<float> &normal_directions,
                            bool only_visible_points = true ) const;

  // TODO Add documentation
  void projectRasterPoints( const cv::Mat_<double> &r_vec, const cv::Mat_<double> &t_vec,
                            std::vector<cv::Point2f> &proj_pts,
                            std::vector<float> &normal_directions,
                            bool only_visible_points = true ) const;

                            
  /**
   * @brief Project the raster segments on the image plane, 
   *        given the current model view ( see setModelView() )
   * 
   * @param[out]proj_segs Output projected segments
   * @param[in] only_visible_segments If false, project also hidden, possinly occluded segments
   *
   * \note If a segment is projected outside the image, or it is not possible to 
   * project the segment or it is occluded, all its coordinates are set to -1
   */
  void projectRasterSegments( std::vector<cv::Vec4f> &proj_segs,
                              bool only_visible_segments = true ) const;
 
  
  /**
   * @brief Project the raster segments and theirs normal directions  on the image plane, 
   *        given the current model view ( see setModelView() )
   * 
   * @param[out]proj_segs Output projected segments
   * @param[out]normal_directions Output normal directions in the image plane
   * @param[in] only_visible_segments If false, project also hidden, possinly occluded segments
   *
   * \note If a segment is projected outside the image, or it is not possible to 
   * project the segment or it is occluded, all its coordinates are set to -1
   * and the normal directions is set to std::numeric_limits< float >::max();
   */                      
  void projectRasterSegments( std::vector<cv::Vec4f> &proj_segs,
                              std::vector<float> &normal_directions,
                              bool only_visible_segments = true  ) const;


  /**
   * @brief Provide the original 3D model vertices
   */
  const std::vector<cv::Point3f> & vertices() const { return  vertices_; };

  /**
   * @brief Project the original 3D model vertices into the image plane
   *
   * @param[out]proj_vtx Output projected points
   *
   * It provide a projection of original 3D model vertices into the image plane
   * given the current model view ( see setModelView() ).
   */
  void projectVertices( std::vector<cv::Point2f> &proj_vtx ) const;

  
  /**
   * @brief Project the 8 bounding box vertices into the image plane
   * 
   * @param[out]proj_bb_segs Output projected points that compose the bounding box
   * 
   * It provide a projection of a bounding box enclosing the object, by means of the set of 
   * vertices that delimit such box, projected on the image plane given the current model 
   * view ( see setModelView() ).
   */  
  void projectBoundingBox( std::vector<cv::Point2f> &proj_bb_pts ) const;

  /**
   * @brief Debug function useful to draw the object bounding box into the image plane.
   * 
   * @param[out]proj_bb_segs Output projected segments that compose the bounding box
   * 
   * It provide a projection of a bounding box encloing the object, by means of the set of 
   * segments that compose such box, projected on the image plane given the current model 
   * view ( see setModelView() ).
   */  
  void projectBoundingBox( std::vector<cv::Vec4f> &proj_bb_segs ) const;
  
  /**
   * @brief Debug function useful to draw the axes of a reference frame into the image plane.
   * 
   * @param[out]proj_segs_x Output projected segments that compose the X axis
   * @param[out]proj_segs_y Output projected segments that compose the Y axis
   * @param[out]proj_segs_z Output projected segments that compose the Z axis
   * @param[in] lx X axis lenght (in meters)
   * @param[in] ly X axis lenght (in meters)
   * @param[in] lz X axis lenght (in meters)
   * @param[in] radius Axes radius (in meters)
   * 
   * It provide the axes of the reference frame of the object by menas a sequence of edges for each axes, 
   * projected on the image plane given the current model view ( see setModelView() )
   * If the axis lenght is set to zero, that axis is not considered
   */  
  void projectAxes( std::vector<cv::Vec4f> &proj_segs_x, 
                    std::vector<cv::Vec4f> &proj_segs_y, 
                    std::vector<cv::Vec4f> &proj_segs_z, 
                    double lx = 0.1, double ly  = 0.1, double lz = 0.1, double radius = 0.001 ) const;

  
  virtual cv::Mat getMask();

protected:

  cv_ext::PinholeCameraModel cam_model_;
  Eigen::Matrix3f view_r_mat_;
  Eigen::Vector3f view_t_vec_;

  UoM unit_meas_;
  enum OrigOffsetType
  {
    USER_DEFINED_ORIG_OFFSET,
    CENTROID_ORIG_OFFSET,
    BOUNDING_BOX_CENTER_ORIG_OFFSET
  };
  
  OrigOffsetType centroid_orig_offset_;
  cv::Point3f orig_offset_;
  double step_;
  double epsilon_;
  double min_seg_len_;

  std::vector<cv::Point3f> vertices_;
  cv_ext::Box3f orig_bbox_, bbox_;
  bool custom_bbox_ = false;
    
//  cv_ext::vector_Quaterniond precomputed_rq_view_;
//  cv_ext::vector_Vector3d precomputed_t_view_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
