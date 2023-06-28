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

#include "cv_ext/pinhole_camera_model.h"

namespace cv_ext
{

class StereoRectification
{
public:
    
  /** @brief Set the parameters that define the stereo pair 
   * 
   * @param[in] cam_models Camera models of each camera (i.e., the intrincs of the two cameras)
   * @param[in] r_mat Rotation matrix between the first and the second camera
   * @param[in] t_vec Translation vector between the two cameras
   */
  void setCameraParameters( const std::vector < PinholeCameraModel > &cam_models,
                            const cv::Mat &r_mat, const cv::Mat &t_vec );
  
  /** @brief Set the image scale factor. 
   * 
   * @param[in] scale_factor The scaling parameter
   * 
   * The output rectified image will be rescaled using this scale factor 
   */
  void setImageScaleFacor( double  scale_factor );
  
  /** @brief If enable is set to true, the left to right minimum disparity will be zero
   * 
   * @todo Double check this!
   */
  void setZeroDisparity( bool enable ){ zero_disp_ = enable; };
  
  /** @brief Set the parameter used to scale the rectified images. 
   * 
   * @param[in] val The scaling parameter
   * 
   * The val parameter should be between 0 and 1: 0 (the default value) means that in the rectified images
   * all pixel are valid (no black areas after rectification, getRegionsOfInterest() return region of interest 
   * that match the image size); 1 means that the rectified images is resized and shifted so 
   * that all the pixels from the original images are retained in the rectified images.
   */
  void setScalingParameter( double val );
  
  /** @brief Update the rectification map 
   * 
   * This method should be called after setting all the parameters (setCameraParameters(), 
   * setScalingParameter(), ...)
   */
  void update();
  
  /** @brief Provide the resulting disparity-to-depth mapping matrix
   * 
   * @param[out] map A 4x4 disparity-to-depth mapping matrix
   * 
   * @note This method should be called after setting the stereo camera parameters with setCameraParameters()
   *       otherwise an empty matrix will be returned.
   */
  void getDisp2DepthMap( cv::Mat &map ){ map = disp2depth_mat_; };
  
  /** @brief Provide the resulting inverse disparity-to-depth mapping matrix
   * 
   * @param[out] map A 4x4 disparity-to-depth mapping matrix
   * 
   * This map considers is computed considering the inverse transformation between the two cameras. 
   * (E.g., right to left in place to left to right)
   * @note This method should be called after setting the stereo camera parameters with setCameraParameters()
   *       otherwise an empty matrix will be returned.
   */
  void getInvDisp2DepthMap( cv::Mat &map ){ map = inv_disp2depth_mat_; };
  
  /** @brief Provide a pair of region of interest (roi) representing the areas of the rectified image where
   *         all pixels are valid
   * 
   * @return Output vector of the two rois
   * 
   * @note This method should be called after setting the stereo camera parameters with setCameraParameters()
   *       otherwise two undefined roi will be returned
   */
  std::vector <cv::Rect> getRegionsOfInterest();

  /** @brief Rectify an input stereo pair images
   * 
   * @param[in] imgs Input image pair
   * @param[out] corners_img Output rectified image pair
   *
   * @note This method should be called after setting the stereo camera parameters with setCameraParameters()
   *       otherwise two empty images will be returned.
   */    
  void rectifyImagePair( const std::vector< cv::Mat > &imgs, std::vector< cv::Mat > &rect_imgs );
  
  /** @brief Provide the camera models of the rectified stereo images
   * 
   * @return The new camera models
   * 
   * @note Call this method after update(), othervise it will
   *       provide just two default PinholeCameraModel objects.
   */
  std::vector< PinholeCameraModel > getCamModels();
  
  /** @brief Provide the size of the output rectified image.
   * 
   * The rectified image will be rescaled using the scale factor set with the setScalingParameter()
   * method, this method will provide the resulting size.
   * @note Call this method after update(), othervise it will provide a (-1, -1) size.
   */
  cv::Size getOutputImageSize(){ return out_image_size_; };

  
  /** @brief Provide the (x,y) displacement between the two rectified cameras
   * 
   * @return The X,Y displacement
   * 
   * @note Call this method after update(), othervise it will
   *       provides just a (0,0) displacement.
   */
  cv::Point2f getCamDisplacement();
  
private:
  
  cv::Size image_size_ = cv::Size(-1,-1), out_image_size_ = cv::Size(-1,-1);
  cv::Mat camera_matrices_[2], dist_coeffs_[2];
  cv::Mat r_mat_, t_vec_;  
  cv::Mat rect_map_[2][2];
  cv::Mat disp2depth_mat_, inv_disp2depth_mat_;
  cv::Rect rect_roi_[2];
  cv::Mat proj_mat_[2];
  
  double scaling_param_ = 0, scale_factor_ = 1.0;
  bool zero_disp_ = false;
};

}