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
#include <vector>
#include <utility>
#include <list>

#include "cv_ext/pinhole_camera_model.h"
#include "cv_ext/stereo_camera.h"

/* TODO
 *
 *  -MultiStereoCameraCalibration : check connections between cameras. Connected graph?
 */

namespace cv_ext
{
// TODO Add documentation
cv::Mat getStandardPatternMask(cv::Size board_size, bool origin_black = true );

/** @brief Abstract base class for calibration objects */       
class CameraCalibrationBase
{
public:

  /** @brief Object constructor
   * 
   * @param[in] cache_max_size Max number of calibration images kept in memory before start to cache them to disk
   * @param[in] tmp_folder Path of the temporary directory used to cache the calibration images
   * 
   * The constructor creates a new directory inside tmp_folder to temporarily cache the calibration images.
   * If tmp_folder is an empty string, the constructor will use the system default temporary directory (e.g., /tmp)
   */
  CameraCalibrationBase( int cache_max_size, const std::string &tmp_folder );
  
  /** @brief Object destructor
   * 
   * The destructor deletes all the images temporarily cached (see the constructor)
   */
  virtual ~CameraCalibrationBase() = 0;

  // TODO Update this documentation
  void setBoardData( const cv::Size &board_size, double square_size );
  /** @brief Set the size of the checkerboard and the size of its squares if a single board is used
   *
   * @param[in] board_size Internal corners size (width X height)
   * @param[in] square_size Square size in the defined unit (e.g., meters)
   * @param[in] origin_white_circle True if the board has a white circle at the origin of the reference frame
   *
   * If a single checkerboard is used for calibration, the size in internal corners (width X height) and the
   * square size in the defined unit are required.
   * @note Horizontal and vertical sizes must differ.
   * It is generally recommended to use a checkerboard with a specific pattern of white or black circles
   * drawn inside specific squares on the checkerboard, see for example setMultipleBoardsData() to define a general
   * pattern. Setting the parameter origin_white_circle to true, enables to accept only checkerboards that have
   * a white circle in the internal black square corresponding to the origin of their reference frame.
   * @note This method is mutually exclusive with respect to setMultipleBoardsData().
   **/
  void setBoardData( const cv::Size &board_size, double square_size, const cv::Mat &pattern_mask );

  /** @brief Set the size of the checkerboard(s), the size of its squares and the pattern masks
   *
   * @param[in] board_size Internal corners size (width X height)
   * @param[in] square_size Square size in the defined unit (e.g., meters)
   * @param[in] pattern_masks Vector of matrices, one for boards, representing unique pattern of white or black circles
   *
   * It is possible to use multiple checkerboards for calibration, which can be framed at the same time,
   * each one with the same corners and square size. To distinguish them, they should be characterized by an
   * unique pattern of white or black circles drawn inside specific black or white (internal) squares, respectively.
   * The circles patterns are defined by means of matrices with size that equals the number of internal squares of
   * the checkerboards (that is, board size minus one, in both dimensions). Each elements should be either 0
   * (no circles in the corresponding square), 1 (a white circle in the corresponding square, that should be black),
   * 2 (a black circle in the corresponding square, that should be white). if none of the framed checkerboards does
   * not contain one of defined patterns, no checkerboards will be detected.
   * Hence, besides the size in internal corners (width X height) and the square size in the defined unit, it is also
   * necessary to provide in input a vector of matrices, one for board, representing unique patterns of white or
   * black circles.
   * It is possible to use this method also for calibrations with a single checkerboard with a custom circles pattern.
   * @note Horizontal and vertical sizes must differ. The matrices in the pattern_masks must be different from
   * each other, and they must not be symmetrical to each other.
   * @note This method is mutually exclusive with respect to setBoardData().
   */
  void setMultipleBoardsData( const cv::Size &board_size, double square_size,
                              const std::vector< cv::Mat > &pattern_masks );

  /** @brief Provides the size of the checkerboard(s), i.e. the size in internal corners (width X height) */
  cv::Size boardSize(){ return board_size_; };

  /** @brief Provides the size of a checkerboard(s) square in the defined unit (e.g., meters) */
  float squareSize(){ return square_size_; };

  /** @brief Provides the vector of matrices, one for board, representing unique patterns of white or black circles
   *
   * Checkerboards can be characterized by an unique pattern of white or black circles drawn inside
   * specific black or white (internal) squares, respectively. This method returns all the matrices that represent
   * such patterns (see setMultipleBoardsData()).
   * @note This method returns an empty vector when using a single checkerboard without any pattern, see the
   * setBoardData() method.
   **/
  std::vector< cv::Mat > patternMasks(){ return pattern_masks_; };

  //! Returns true if multiple checkerboards are used (see also setMultipleBoardsData() and patternMasks())
  bool usingMultipleCheckerboards() { return pattern_masks_.size() > 1; };

  //! Returns the number of unique patterns of white or black circles (see also setMultipleBoardsData() and patternMasks())
  int numPatternMasks() { return pattern_masks_.size(); };

  /** @brief Set the number of levels of the gaussian image pyramid used to extract corners
   * 
   * @param[in] pyr_num_levels Numbers of pyramid levels, it should be greater or equal than 1.
   * 
   * Build a gaussian pyramid of the image and start to extract the corners from the higher
   * level of a gaussian pyramid. For large images, it may be useful to use a 2 or 3 levels pyramid.
   * Set to 1 by default
   */
  void setPyramidNumLevels( int pyr_num_levels ) { pyr_num_levels_ = pyr_num_levels; };

  /** @brief Provides the number of levels of the gaussian image pyramid used to extract corners,
   *         see setPyramidNumLevels() */
  int pyramidNumLevels() { return pyr_num_levels_; };

  /** @brief Provides the size of the images used for calibration
   * 
   * A (-1,-1) size is returnend if no images have been added
   */  
  cv::Size imagesSize(){ return image_size_; }

  /** @brief Provides the OpenCV image type of the images used for calibration
   *
   * See cv::Mat::type()
   * A -1 type is returnend if no images have been added
   */
  int imagesType(){ return image_type_; }

  /** @brief Set the the maximum number of iterations of the calibration algorithm */
  void setMaxNumIter( int num ){ max_iter_ = num; };

  /** @brief Provides the the maximum number of iterations of the calibration algorithm */
  int maxNumIter(){ return max_iter_; };

  /** @brief Pure virtual method that should be overridden in derived classes
   * 
   * It should run the calibration given the loadaed/selected images.
   * 
   * @return Some metric about the calibration
   */
  virtual double calibrate() = 0;
  
  /** @brief Pure virtual method that should be overridden in derived classes.
   *
   *  It should clear all the loaded images and the chache 
   **/
  virtual void clear() = 0;
  
protected:
  
  void clearDiskCache();
  void cachePutImage( int i, const cv::Mat &img );
  cv::Mat cacheGetImage( int i );
  std::string getCacheFilename( int i );

  //! Extract corners from an image, in case only one checkerboard is used
  bool findCorners ( const cv::Mat &img, std::vector<cv::Point2f> &img_pts );
  //! Extract corners from an image, in case multiple checkerboards are used
  bool findCorners ( const cv::Mat &img, std::vector< std::vector<cv::Point2f> > &img_pts_vec );

  void getCornersImage ( int cached_image_idx, const std::vector<cv::Point2f> &corners, cv::Mat &dst_img,
                         cv::Size size, float scale_factor );
  void getCornersDistribution ( const std::vector< std::vector<cv::Point2f> > &corners,
                                const std::vector<bool> &masks,
                                float kernel_stdev, cv::Mat &dst_img, cv::Size size, float scale_factor );

  cv::Size board_size_ = cv::Size(-1,-1);
  double square_size_ = 0.0f;
  int max_iter_ = 30;

  cv::Size image_size_ = cv::Size(-1,-1);
  int image_type_;

private:

  void patternMasksSetup();
  int matchCirclesPatterns( const cv::Mat &img, std::vector<cv::Point2f> &img_pts );
  unsigned char sampleSquareCenter( const cv::Mat &img, const std::vector<cv::Point2f> &img_pts, int square_x, int square_y );

  /** List used to implement a very simple, linear-time access cache */
  std::list< std::pair<int, cv::Mat> > images_cache_;

  int cache_max_size_;
  std::string cache_folder_;

  int pyr_num_levels_ = 1;
  std::vector< cv::Mat > pattern_masks_;
  std::vector<cv::Point2f> square_corners_, test_pts_;
};

/** @brief Calibrate a single camera
 *
 *  CameraCalibration estimates the intrinsic parameters (both K and distortion parameters) of a cameras from a
 *  set of images of one or more checkerboards
 *  It is possible to frame more than one checkerboard at the same time, each one with the same board and square
 *  size, but with different patterns of circles inside the checkerboard squares, see the setMultipleBoardsData()
 *  method.
 **/
class CameraCalibration : public CameraCalibrationBase
{
public:

  /** @brief Object constructor
   * 
   * @param[in] cache_max_size Max number of calibration images kept in memory before start to cache them to disk
   * @param[in] tmp_folder Path of the temporary directory used to cache the calibration images
   * 
   * The constructor creates a new directory inside tmp_folder to temporarily cache the calibration images.
   * If tmp_folder is an empty string, the constructor will use the system default temporary dir (e.g., /tmp)
   */
  explicit CameraCalibration( int cache_max_size = 100, const std::string &tmp_folder = std::string() );
  
  /** @brief Object destructor
   * 
   * The destructor deletes all the images temporarily cached (see the constructor)
   */
  virtual ~CameraCalibration() = default;

  /** @brief If enabled, the calibration assumes zero tangential distortion
   *
   * Set to false by default
   */
  void setZeroTanDist( bool enable ){ zero_tan_dist_ = enable; };

  /** @brief Return true if the calibration assumes zero tangential distortion  */
  bool zeroTanDist(){ return zero_tan_dist_; };

  /** @brief If enabled, the calibration uses a provided camera model as initial guess
 *
 * Set to false by default
 */
  void setUseIntrinsicGuess( bool enable ){ use_intrinsic_guess_ = enable; };

  /** @brief Return true if the calibration uses a provided camera model as initial guess
   */
  bool useIntrinsicGuess(){ return use_intrinsic_guess_; };

  /** @brief If enabled, consider in the calibration only fy as a free parameter, with fx/fy = 1
   *
   * Set to false by default
   */
  void setFixAspectRatio( bool enable ){ fix_aspect_ratio_ = enable; };

  /** @brief Return true if the calibration considers only fy as a free parameter, with fx/fy = 1
   */
  bool fixAspectRatio(){ return fix_aspect_ratio_; };

  /** @brief If enabled, the principal point is not changed during the global optimization
   *
   * Set to false by default
   */
  void setFixPrincipalPoint( bool enable ){ fix_principal_point_ = enable; };

  /** @brief Return true if the principal point is not changed during the global optimization
   */
  void fixPrincipalPoint( bool enable ){ fix_principal_point_ = enable; };

  /** @brief Set a previously computed camera model
   * 
   * @param[in] cam_model Input camera model
   * 
   * If setUseIntrinsicGuess() is set to true, this model will be used as an initial guess in the calibration
   */
  void setCamModel( const PinholeCameraModel &model );
  
  /** @brief Provides the resulting camera model
   *
   * @return The estimated model
   *
   * @note If no calibration has been performed, or no models have been set with setCamModel(),
   *       this method provides a default PinholeCameraModel object
   */
  PinholeCameraModel getCamModel();
  
  /** Add an image of a checkerboard (or more checkerboards) and extract corners
   * 
   * @param[in] img A reference to a one or three channels image
   *
   * @return True if the image is valid and at least one checkerboard has been successfully extracted, false otherwise
   */
  bool addImage( const cv::Mat &img );

  /** Add an image of a checkerboard and the pre-extracted corners.
   *
   * @param[in] img A reference to a one or three channels image
   * @param[in] corner_pts A vector of corner points extracted from the image
   */
  void addImage( const cv::Mat &img, const std::vector<cv::Point2f> &corners_pts );

  /** Add an image of a set of checkerboards and the pre-extracted corners.
   *
   * @param[in] img A reference to a one or three channels image
   * @param[in] corner_pts A vector of vector (one for each checkerboard) of corner points extracted from the image
   */
  void addImage( const cv::Mat &img, const std::vector< std::vector<cv::Point2f> > &corners_pts );

  /** @brief Load and add an image of a checkerboard (or more checkerboards) and extract corners
   * 
   * @param[in] filename Path of image file to be loaded.
   * 
   * @return True if the image is valid and at least one checkerboard has been successfully extracted, false otherwise
   */  
  bool addImageFile ( const std::string &filename );

  /** @brief Provides the number of added images
   *
   * See also addImage() and addImageFile()
   */
  int numImages_(){ return num_images_; };

  /** @brief Provides the number of checkerboards successfully extracted from images
   *
   * See also addImage() and addImageFile()
   */
  int numCheckerboards(){ return num_cb_; };

  /** @brief Provides the number of active checkerboards, i.e. the ones actually used for calibration
   *
   * See also setCheckerboardActive() and calibrate()
   */
  int numActiveCheckerboards(){ return num_active_cb_; };
  
  /** @brief Set whether an extracted checkerboard will be used in the calibration process
   * 
   * @param[in] i Index of the extracted checkerboard, from 0 to numCheckerboards() - 1
   * @param[in] active If false, the checkerboard with index i will not be used for calibration
   * 
   * By default, all the extracted checkerboard are used for calibration.
   */
  void setCheckerboardActive( int i, bool active );
  
  /** Return true if an extracted checkerboard will be used in the calibration process
   * 
   * @param[in] i Index of the extracted checkerboard, from 0 to numCheckerboards() - 1
   */
  bool isCheckerboardActive( int i ){ return cb_masks_[i]; };
  
  /** @brief Perform the calibration.
   * 
   * @return The root mean squared re-projection error, computed only on the checkerboards marked as active
   * 
   * The calibration is performed using only the checkerboards marked as active (see setCheckerboardActive())
   * If no checkerboards have been extracted (see addImage() and addImageFile()), or all checkerboards have been marked
   * as not active, this method will return an infinite value
   */
  double calibrate() override;

  /** @brief Provides a root mean squared reprojection error for the i-th checkerboard
   * 
   * @param[in] i Index of the image, from 0 to numCheckerboards() - 1
   * 
   * If the last calibration has not been performed using the required checkerboard,
   * this method will return an infinite value
   */
  double getReprojectionError( int i ){ return cb_errors_[i]; };

  /** @brief Provides the extracted corners for the ith checkerboard
   *
   * @param[in] i Index of the checkerboard, from 0 to numCheckerboards() - 1
   *
   * @return Vector with the extracted corners
   */
  std::vector<cv::Point2f> getCorners( int i );

  /** @brief Provides a possibly scaled image with drawn the detected checkerboard corners
   * 
   * @param[in] i Index of the checkerboard, from 0 to numCheckerboards() - 1
   * @param[out] corners_img Output image with represented the extracted corners
   * @param[in] size Size of the output image; if zero, the image is scaled according to the scale factor parameter
   * @param[in] scale_factor The output image scale factor, used if size is zero; it should be >= 1
   */  
  void getCornersImage( int i, cv::Mat &corners_img, cv::Size size = cv::Size(0,0), float scale_factor = 1.0f );

  /** @brief Provides a possibly scaled image that depicts a qualitative representation of the
   *  non-normalized density of the checkerboard corners
   * 
   * @param[in] kernel_stdev Standard deviation of the Gaussian kernel used in the density 
   *                         estimation
   * @param[out] corner_dist Output corner distribution image  (one channel, depth CV_32F)
   * @param[in] size Size of the output image; if zero, the image is scaled according to the scale factor parameter
   * @param[in] scale_factor The output image scale factor, it should be >= 1
   * 
   * getCornersDistribution() considers the corners extracted from each checkerboard (see addImage() and
   * addImageFile()) marked as active (see setCheckerboardActive()).
   * The density is obtained by means of kernel density estimation, i.e. a simplified version 
   * of the Parzen–Rosenblatt window method, using a non-normalized Gaussian kernel (kernel(0,0) = 1) 
   * with standard deviation kernel_stdev.
   */  
  void getCornersDistribution( float kernel_stdev, cv::Mat &corner_dist, cv::Size size = cv::Size(0,0),
                               float scale_factor = 1.0f );
  
  /** @brief Provides a possibly scaled image of checkerboad, undistorted using the current calibration parameters
   * 
   * @param[in] i Index of the image, from 0 to numImages() - 1
   * @param[out] und_img Output undistorted image
   * @param[in] size Size of the output image; if zero, the image is scaled according to the scale factor parameter
   * @param[in] scale_factor The output image scale factor, it should be >= 1
   */
  void getUndistortedImage( int i, cv::Mat &und_img, cv::Size size = cv::Size(0,0),
                            float scale_factor = 1.0f );

  /** @brief Clear all the loaded images and the chache **/
  void clear();

private:

  bool use_intrinsic_guess_ = false,
       fix_aspect_ratio_ = false,
       zero_tan_dist_ = false,
       fix_principal_point_ = false;

  int num_images_ = 0, num_cb_ = 0, num_active_cb_ = 0;
  cv::Mat camera_matrix_, dist_coeffs_;
  
  std::vector< std::vector<cv::Point2f> > cb_corners_;
  std::vector<bool> cb_masks_;
  std::vector<double> cb_errors_;
  std::vector< int > cb_img_map_;
};

/** @brief Calibrate a stereo pair
 *
 *  StereoCameraCalibration can estimates the extrinsics and (if required) also the intrinsic parameters of
 *  the two cameras (both K and distortion parameters) from a set of images pairs of one or more checkerboards.
 *  It is possible to frame more than one checkerboard at the same time, each one with the same board and square
 *  size, but with different patterns of circles inside the checkerboard squares, see the setMultipleBoardsData()
 *  method.
 **/
class StereoCameraCalibration : public CameraCalibrationBase
{ 
public:
  /** @brief Object constructor
   * 
   * @param[in] cache_max_size Max number of calibration images kept in memory before start to cache them to disk
   * @param[in] tmp_folder Path of the temporary directory used to cache the calibration images
   * 
   * The constructor creates a new directory inside tmp_folder to temporarily cache the calibration images.
   * If tmp_folder is an empty string, the constructor will use the system default temporary dir (e.g., /tmp)
   */
  explicit StereoCameraCalibration( int cache_max_size = 100, const std::string &tmp_folder = std::string() );
  
  /** @brief Object destructor
   * 
   * The destructor deletes all the images temporarily cached (see the constructor)
   */
  virtual ~StereoCameraCalibration() = default;

  /** @brief If enabled, the calibration assumes zero tangential distortion
   *
   * Set to false by default
   */
  void setZeroTanDist( bool enable ){ zero_tan_dist_ = enable; };

  /** @brief Return true if the calibration assumes zero tangential distortion  */
  bool zeroTanDist(){ return zero_tan_dist_; };

  /** @brief If enabled, the calibration uses a provided camera model as initial guess
 *
 * Set to false by default
 */
  void setUseIntrinsicGuess( bool enable ){ use_intrinsic_guess_ = enable; };

  /** @brief Return true if the calibration uses a provided camera model as initial guess
   */
  bool useIntrinsicGuess(){ return use_intrinsic_guess_; };

  /** @brief If enabled, consider in the calibration only fy as a free parameter, with fx/fy = 1
   *
   * Set to false by default
   */
  void setFixAspectRatio( bool enable ){ fix_aspect_ratio_ = enable; };

  /** @brief Return true if the calibration considers only fy as a free parameter, with fx/fy = 1
   */
  bool fixAspectRatio(){ return fix_aspect_ratio_; };

  /** @brief If enabled, the principal point is not changed during the global optimization
   *
   * Set to false by default
   */
  void setFixPrincipalPoint( bool enable ){ fix_principal_point_ = enable; };

  /** @brief Return true if the principal point is not changed during the global optimization
   */
  void fixPrincipalPoint( bool enable ){ fix_principal_point_ = enable; };

  /** @brief If enabled, enforce the focal lenghts to be the same for both camera
   *
   * This method has effect if no camera model has been provided in input (see setCamModels()) or
   * setUseIntrinsicGuess() is set to true.
   * Set to false by default
   */
  void setForceSameFocalLenght( bool enable ){ force_same_focal_lenght_ = enable; }; 

  /** @brief Return true if the calibration enforces the focal lenghts to be the same for both camera, 
   *         see setForceSameFocalLenght() */
  bool forceSameFocalLenght(){ return force_same_focal_lenght_; }; 
  
  /** @brief Set previously computed camera models
   * 
   * @param[in] cam_models Input vector of two camera models
   * 
   * Before calibrate a stereo camera, it is recommended to calibrate the cameras 
   * individually using CameraCalibration an to provide the resulting camera parameters to 
   * StereoCameraCalibration using setCamModels().
   * If setUseIntrinsicGuess() is set to true, this models will be used as an initial guess in 
   * the calibration, otherwise these parameters will be keep fixed and the calibration will 
   * estimate only the extrinsics parameters between the cameras.
   */  
  void setCamModels( const std::vector < PinholeCameraModel > &cam_models );
 
  /** @brief Provides the resulting camera models
   *
   * @return A vector with the estimated camera models
   * 
   * @note If no calibration has been performed, or no models have been set with setCamModels(),
   *       this method provides two default PinholeCameraModel objects.
   */
  std::vector < PinholeCameraModel > getCamModels();
  
  /** @brief Provides the resulting extrinsic parameters
   * 
   * @param[out] r_mat Rotation matrix between the first and the second camera
   * @param[out] t_vec Translation vector between the two cameras
   * 
   * @note If no calibration has been performed, this method provides two empty matrices
   */
  void getExtrinsicsParameters( cv::Mat &r_mat, cv::Mat &t_vec );


  /** Add a pair of images of a checkerboard (or more checkerboards) acquired by a stereo pair
   * 
   * @param[in] imgs A vector of two one or three channels images.
   * 
   * @return True if the images are valid and at least one checkerboard has been
   *         successfully extracted in both images, false otherwise.
   *
   * Both cameras should frame the same checkerboard(s).
   */  
  bool addImagePair( const std::vector< cv::Mat > &imgs );

  /** Add a pair of images of a checkerboard (or more checkerboards) acquired by a stereo pair, and the pre-extracted corners.
   *
   * @param[in] imgs A vector of two one or three channels images.
   * @param[in] corner_pts A vector of vectors of corner points extracted from the image pair
   *
   * @note The corners_pts vector should contain at least two vectors of corner points in case a single
   * checkerboard is framed, or a number of vectors that is a multiple of 2 in case more than one checkerboard
   * is framed at a time
   */
  void addImagePair( const std::vector< cv::Mat > &imgs, const std::vector< std::vector<cv::Point2f> > &corners_pts );

  /** @brief Load and add a pair of images of a checkerboard (or more checkerboards) acquired by a stereo pair
   * 
   * @param[in] filenames A vector with the paths of the two image files to be loaded.
   * 
   * @return True if the images are valid and the checkerboard(s) has been
   *         successfully extracted in both images, false otherwise
   *
   * Both cameras should frame the same checkerboard(s).
   */   
  bool addImagePairFiles ( const std::vector< std::string > &filenames );

  /** @brief Provides the number of images pairs successfully added
   *
   * See also the addImagePair() or addImagePairFiles() methods
   */
  int numImagePairs_(){ return num_image_pairs_; };

  /** @brief Provides the number of checkerboards successfully extracted from images pairs
   *
   * See also the addImagePair() or addImagePairFiles() methods
   */
  int numCheckerboards(){ return num_cb_; };

  /** @brief Provides the number of active checkerboards, i.e. the ones actually used for calibration
   *
   * See also the setCheckerboardActive() and calibrate() methods
   */
  int numActiveCheckerboards(){ return num_active_cb_; };

  /** @brief Set whether an extracted checkerboard will be used in the calibration process
   *
   * @param[in] i Index of the extracted checkerboard, from 0 to numCheckerboards() - 1
   * @param[in] active If false, the checkerboard with index i will not be used for calibration
   *
   * By default, all checkerboards are used for calibration.
   */
  void setCheckerboardActive( int i, bool active );

  /** Return true if an extracted checkerboard will be used in the calibration process
   *
   * @param[in] i Index of the extracted checkerboard, from 0 to numCheckerboards() - 1
   */
  bool isCheckerboardActive( int i ){ return cb_masks_[i]; };
  
  /** @brief Perform the calibration.
   * 
   * @return The root mean squared (RMS) distance between the extracted corners and estimated epipolar lines.
   *
   * The calibration is performed using only the checkerboards marked as active (see setCheckerboardActive())
   * If no images pairs have been successfully added (see addImagePair() and addImagePairFiles()), or
   * all checkerboards have been marked as not active, this method will return an infinite value.
   * The returned RMS error is computed using the epipolar geometry constraint: m2^t*F*m1=0, where the fundamental
   * matrix F is computed using the estimated relative transformation between cameras.
   * @note The RMSE is computed considering  only on the images pairs marked as active.
   */  
  double calibrate() override;

  /** @brief Provides the RMS distance between the extracted points and estimated epipolar lines for the i-th checkerboard
   * 
   * @param[in] i Index of the checkerboard, from 0 to numCheckerboards() - 1
   * 
   * If the last calibration has not been performed using the required checkerboard,
   * this method will return an infinite value
   */
  double getEpipolarError( int i ){ return cb_errors_[i]; };

  /** @brief Provides the extracted corners for the ith checkerboard
   *
   * @param[in] i Index of the checkerboard, from 0 to numCheckerboards() - 1
   *
   * @return Vectors of two vector with the extracted corners (one for each image of tha pair)
   */
  std::vector< std::vector< cv::Point2f > > getCornersPair( int i );

  /** @brief Provides a pair of possibly scaled images with drawn the detected checkerboard corners
   * 
   * @param[in] i Index of the checkerboard, from 0 to numCheckerboards() - 1
   * @param[out] corners_img Output image pair with represented the extracted corners
   * @param[in] size Size of the output image; if zero, the image is scaled according to the scale factor parameter
   * @param[in] scale_factor The output image scale factor, it should be >= 1 
   */  
  void getCornersImagePair( int i, std::vector< cv::Mat > &corners_imgs, cv::Size size = cv::Size(0,0),
                            float scale_factor = 1.0f );
  
  /** @brief Provides a pair of possibly scaled images that depicts a qualitative representation of the
   *  non-normalized density of the checkerboard corners
   * 
   * @param[in] kernel_stdev Standard deviation of the Gaussian kernel used in the density 
   *                         estimation
   * @param[out] corner_dists Output corner distribution images (one channel, depth CV_32F)
   * @param[in] size Size of the output image; if zero, the image is scaled according to the scale factor parameter
   * @param[in] scale_factor The output image scale factor, it should be >= 1
   * 
   * getCornersDistribution() considers the corners extracted from each added image pair
   * (see addImagePair() and addImagePairFiles()) marked as active (see setCheckerboardActive()).
   * The density is obtained by means of kernel density estimation, i.e. a simplified version 
   * of the Parzen–Rosenblatt window method, using a non-normalized Gaussian kernel (kernel(0,0) = 1) 
   * with standard deviation kernel_stdev
   */  
  void getCornersDistribution( float kernel_stdev, std::vector< cv::Mat > &corner_dists,
                               cv::Size size = cv::Size(0,0), float scale_factor = 1.0f );
  
  /** @brief Provides a pair of possibly scaled rectified images
   * 
   * @param[in] i Index of the image pair to be rectified, from 0 to numImagePairs() - 1
   * @param[out] corners_img Output rectified image pair
   * @param[in] size Size of the output image; if zero, the image is scaled according to the scale factor parameter
   * @param[in] scale_factor The output image scale factor
   *
   * This method internally uses the StereoRectification() object, can can be called only after calibrate(),
   * otherwise two empty images will be returned.
   */  
  void getRectifiedImagePair( int i, std::vector< cv::Mat > &rect_imgs, cv::Size size = cv::Size(0,0),
                              float scale_factor = 1.0f );

  /** @brief Clear all the loaded images and the chache **/
  void clear();
  
private:
  
  int num_image_pairs_ = 0, num_cb_ = 0, num_active_cb_ = 0;

  bool use_intrinsic_guess_ = false,
       fix_aspect_ratio_ = false,
       zero_tan_dist_ = false,
       fix_principal_point_ = false,
       force_same_focal_lenght_ = false;

  cv::Mat camera_matrices_[2], dist_coeffs_[2];
  cv::Mat r_mat_, t_vec_;
  
  std::vector< std::vector<cv::Point2f> > cb_corners_[2];
  std::vector<bool> cb_masks_;
  std::vector<double> cb_errors_;
  std::vector< int > cb_img_map_;

  cv_ext::StereoRectification stereo_rect_;
};

/** @brief Calibrate a multi-stereo system (i.e., a N-cameras rig)
 *
 *  MultiStereoCameraCalibration estimates the extrinsics parameters of the cameras (i.e., the rigid body
 *  transformations between the first camera and each other camera in the rig) from a set of images tuples of one or
 *  more checkerboards.
 *  It is possible to frame more than one checkerboard at the same time, each one with the same board and square
 *  size, but with different patterns of circles inside the checkerboard squares, see the setMultipleBoardsData()
 *  method.
 */
class MultiStereoCameraCalibration : public CameraCalibrationBase
{
 public:

  /** @brief Object constructor
   *
   * @param[in] num_cameras Number of cameras included in the rig
   * @param[in] cache_max_size Max number of calibration images kept in memory before start to cache them to disk
   * @param[in] tmp_folder Path of the temporary directory used to cache the calibration images
   *
   * The constructor creates a new directory inside tmp_folder to temporarily cache the calibration images.
   * If tmp_folder is an empty string, the constructor will use the system default temporary dir (e.g., /tmp)
   */
  explicit MultiStereoCameraCalibration( int num_cameras, int cache_max_size = 100,
                                         const std::string tmp_folder = std::string() );

  /** @brief Object destructor
   *
   * The destructor deletes all the images temporarily cached (see the constructor)
   */
  virtual ~MultiStereoCameraCalibration() = default;

  /** @brief Set the previously computed camera models, on for each camera of the rig
   *
   * @param[in] cam_models Input vector of camera models
   *
   * MultiStereoCameraCalibration only estimates the extrinsics parameters between the cameras.
   * The intrinsic parameters of each camera (both K and distortion parameters) should be estimated
   * in advance, e.g. by using the CameraCalibration object.
   */
  void setCamModels( const std::vector< PinholeCameraModel > &cam_models );

  /** @brief Set the alpha parameter used in the Huber loss function during the final calibration refinement */
  void setHuberLossAlpha( double alpha ){ huber_loss_alpha_ = alpha; }

  /** @brief Provides the alpha parameter used in the Huber loss function during the final calibration refinement */
  double huberLossAlpha(){ return huber_loss_alpha_; }

  /** @brief Provides the resulting extrinsic parameters
   *
   * @param[out] r_mats Vector of the rotation matrices between the first and the other cameras
   * @param[out] t_vecs Vector of the translation vectors between the first and the other cameras
   *
   * The extinisc parameters of a N-cameras rig are represented by the N-1 rigid body transformations between the
   * first camera and the other N-1 cameras in the rig.
   * @note If no calibration has been performed, this method provides empty vectors.
   */
  void getExtrinsicsParameters( std::vector< cv::Mat > &r_mats, std::vector< cv::Mat > &t_vecs );

  /** Add a tuple of images of a checkerboard (or more checkerboards) acquired at the same time by a N-cameras rig.
   *
   * @param[in] imgs A vector of one or three channels images, with size equal to the number of cameras of the rig
   *
   * @return True if the images are valid and the checkerboard(s) has been
   *         successfully extracted in at least two images of the rig, false otherwise
   *
   * At least two cameras should frame the checkerboard.
   */
  bool addImageTuple(const std::vector<cv::Mat> &imgs );

  /** Add a tuple of images of a checkerboard (or more checkerboards) acquired at the same time by a N-cameras rig.
   *
   * @param[in] imgs A vector of one or three channels images, with size equal to the number of cameras of the rig
   * @param[in] corner_pts A vector of vectors of corner points extracted from the images
   *
   * @note The corners_pts vector should contain at least N vectors of corner points (N number of cameras of the rig)
   * in case a single checkerboard is framed, or a number of vectors that is a multiple of N in case more than one
   * checkerboard is framed at a time
   */
  void addImageTuple(const std::vector<cv::Mat > &imgs, const std::vector<std::vector<cv::Point2f> > &corners_pts );

  /** @brief Load and add a tuple of images of a checkerboard (or more checkerboards) acquired at the same time by the N-cameras rig.
   *
   * @param[in] filenames A vector with the paths of images files to be loaded, with size equal to the number of cameras
   *
   * @return True if the images are valid and the checkerboard(s) has been
   *         successfully extracted in at least two images, false otherwise.
   *
   * At least two cameras should frame the checkerboard(s).
   */
  bool addImageTupleFiles ( const std::vector< std::string > &filenames );

  /** @brief Provides the number of images tuples successfully added
   *
   * See also the addImageTuple() or addImageTupleFiles() methods
   */
  int numImageTuples_(){ return num_tuples_; };

  /** @brief Provides the number of checkerboards successfully extracted from images tuples
   *
   * See also the addImageTuple() or addImageTupleFiles() methods
   */
  int numCheckerboards(){ return num_cb_; };

  /** @brief Provides the number of active checkerboards, i.e. the ones actually used for calibration
   *
   * See also the setCheckerboardActive() and calibrate() methods
   */
  int numActiveCheckerboards(){ return num_active_cb_; };

  /** @brief Set whether an extracted checkerboard will be used in the calibration process
   *
   * @param[in] i Index of the extracted checkerboard, from 0 to numCheckerboards() - 1
   * @param[in] active If false, the checkerboard with index i will not be used for calibration
   *
   * By default, all checkerboards are used for calibration.
   */
  void setCheckerboardActive( int i, bool active );

  /** Return true if an extracted checkerboard will be used in the calibration process
   *
   * @param[in] i Index of the extracted checkerboard, from 0 to numCheckerboards() - 1
   */
  bool isCheckerboardActive( int i ){ return cb_masks_[i]; };

  /** @brief Perform the calibration.
   *
   * @return The root mean squared (RMS) distance between the extracted points and estimated epipolar lines.
   *
   * The calibration is performed using only the checkerboards marked as active (see setCheckerboardActive())
   * If no image tuples have been successfully added (see addImageTuple() and addImageTupleFiles()),
   * or all checkerboards have been marked as not active, this method will return an infinite value.
   * The returned RMS error is computed using the epipolar geometry constraint: m2^t*F*m1=0, where the fundamental
   * matrices F are computed using the estimated relative transformations between cameras.
   * @note The RMSE is computed considering  only on the image tuples marked as active.
   */
  double calibrate() override;

  /** @brief Provides the RMS distance between the extracted points and estimated epipolar lines for the i-th checkerboard
   *
   * @param[in] i Index of the image tuple, from 0 to numCheckerboards() - 1
   *
   * If the last calibration has not been performed using the required image checkerboard,
   * this method will return an infinite value
   */
  double getEpipolarError( int i ){ return cb_errors_[i]; };

  /** @brief Provides a tuple of possibly scaled images with drawn the detected checkerboard corners
   *
   * @param[in] i Index of the image pair, from 0 to numImageTuples() - 1
   * @param[out] corners_img Output image tuple with represented the extracted corners
   * @param[in] size Size of the output image; if zero, the image is scaled according to the scale factor parameter
   * @param[in] scale_factor The output image scale factor, it should be >= 1
   */
  void getCornersImageTuple( int i, std::vector< cv::Mat > &corners_imgs, cv::Size size = cv::Size(0,0),
                             float scale_factor = 1.0f );

  /** @brief Provides a tuple of possibly scaled images that depicts a qualitative representation of the
   *  non-normalized density of the checkerboard corners
   *
   * @param[in] kernel_stdev Standard deviation of the Gaussian kernel used in the density
   *                         estimation
   * @param[out] corner_dists Output corner distribution images (one channel, depth CV_32F)
   * @param[in] size Size of the output image; if zero, the image is scaled according to the scale factor parameter
   * @param[in] scale_factor The output image scale factor, it should be >= 1
   *
   * getCornersDistribution() considers the corners extracted from each checkerboard
   * (see addImageTuple() and addImageTupleFiles()) marked as active (see setCheckerboardActive()).
   * The density is obtained by means of kernel density estimation, i.e. a simplified version
   * of the Parzen–Rosenblatt window method, using a non-normalized Gaussian kernel (kernel(0,0) = 1)
   * with standard deviation kernel_stdev
   */
  void getCornersDistribution( float kernel_stdev, std::vector< cv::Mat > &corner_dists, cv::Size size = cv::Size(0,0),
                               float scale_factor = 1.0f );

  /** @brief Clear all the loaded images and the chache **/
  void clear();

 private:

  int num_cameras_;

  int num_tuples_ = 0, num_cb_ = 0, num_active_cb_ = 0;
  std::vector< cv_ext::PinholeCameraModel > cam_models_;

  std::vector< cv::Mat > r_mats_, t_vecs_;
  double huber_loss_alpha_ = 1.0;

  std::vector< std::vector< std::vector<cv::Point2f> > > cb_corners_;
  std::vector<bool> cb_masks_;
  std::vector<double> cb_errors_;
  std::vector< int > cb_img_map_;
};

}
