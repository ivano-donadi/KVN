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

#include <memory>
#include <vector>

#include "cv_ext/cv_ext.h"

typedef std::shared_ptr< class EdgeDetector > EdgeDetectorPtr;
typedef std::shared_ptr< class LSDEdgeDetector > LSDEdgeDetectorPtr;
typedef std::shared_ptr< class CannyEdgeDetector > CannyEdgeDetectorPtr;
typedef std::unique_ptr< class EdgeDetector > EdgeDetectorUniquePtr;
typedef std::unique_ptr< class LSDEdgeDetector > LSDEdgeDetectorUniquePtr;
typedef std::unique_ptr< class CannyEdgeDetector > CannyEdgeDetectorUniquePtr;

class EdgeDetector
{
public:
  EdgeDetector() :
    white_bg_(false),
    num_directions_(1),
    map_size_(cv::Size(0,0)){};

  virtual ~EdgeDetector(){};

  bool whiteBackgroundEnabled(){ return white_bg_; };
  int numEdgeDirections(){ return num_directions_; };

  void enableWhiteBackground( bool enable  ){ white_bg_ = enable; };
  void setNumEdgeDirections( int n ){ num_directions_ = (n >= 1 ? ( (n > 0xFFFF) ? 0xFFFF : n) : 1); };
  // TODO Check mask
  void setMask( cv::Mat mask ){ mask_ = mask; };

  virtual void setImage( const cv::Mat &src_img ) = 0;
  virtual void getEdgeMap( cv::Mat &edge_map ) = 0;
  virtual void getEdgeDirectionsMap( cv::Mat& edge_dir_map ) = 0;
  virtual void getDirectionalEdgeMap( int i_dir, cv::Mat &edge_map ) = 0;
  
protected:
  bool white_bg_;
  int num_directions_;
  cv::Size map_size_;
  cv::Mat mask_;
};

class LSDEdgeDetector : public EdgeDetector
{
public:
  LSDEdgeDetector() :
  pyr_num_levels_ (1),
  scale_(1.0),
  sigma_scale_(0.6),
  quant_threshold_(0.5){};

  int pyrNumLevels(){ return pyr_num_levels_; };
  double scale(){ return scale_; };
  double sigmaScale(){ return sigma_scale_; };
  double quantizationThreshold() {return quant_threshold_; };

  void setPyrNumLevels( int num_levels ){ pyr_num_levels_ = num_levels; };
  void setScale( double s ){ scale_ = s; };
  void setSigmaScale( double s ){ sigma_scale_ = s; };
  void setQuantizationThreshold( double q ) {quant_threshold_ = q; };

  virtual void setImage( const cv::Mat &src_img );
  virtual void getEdgeMap( cv::Mat &edge_map );
  virtual void getEdgeDirectionsMap( cv::Mat& edge_dir_map );
  virtual void getDirectionalEdgeMap( int i_dir, cv::Mat &edge_map );
  
private:
  cv::Mat checkImage( const cv::Mat &src_img);
  void extractEdgesPyr( const cv::Mat &src_img, std::vector< cv::Vec4f > &segments );
  void extractEdges( const cv::Mat &src_img, std::vector< cv::Vec4f > &segments, float scale = 1.0 );
  void computeEdgesNormalsDirections( const std::vector< cv::Vec4f > &segments,
                                      std::vector<float> &normals );

  int pyr_num_levels_;
  double scale_, sigma_scale_, quant_threshold_;

  std::vector< cv::Vec4f > segments_;
  std::vector< std::vector<cv::Vec4f> > dir_segments_;
};

class CannyEdgeDetector : public EdgeDetector
{
public:
  CannyEdgeDetector() : low_threshold_(20), ratio_(3), use_rgb_(false){};

  int lowThreshold(){ return low_threshold_; };
  int ratio(){ return ratio_; };

  void setLowThreshold( int th){ if( th > 1 && th <= 100) low_threshold_ = th; };
  void setRatio( int r ){ ratio_ = r; };
  
  /**
   * @brief Enable/disable the "RGB modality" of the edge detector
   * 
   * @param[in] enable If true, apply Canny to each image channel
   * 
   * If enable is true, in case of an input RGB image, the Canny edge detector is applied to each image channel (RGB) 
   * and then each resulting edge map is merged (OR operator) into an unique edge map
   * \note In case of grey level input images, the conventional Canny edge detector is applied even if the "RGB modality" 
   *       is enabled.
   */  
  void enableRGBmodality( bool enable ){ use_rgb_ = enable; };
  virtual void setImage( const cv::Mat &src_img );
  virtual void getEdgeMap( cv::Mat &edge_map );
  virtual void getEdgeDirectionsMap( cv::Mat& edge_dir_map );
  virtual void getDirectionalEdgeMap( int i_dir, cv::Mat &edge_map );

protected:

  virtual cv::Mat checkImage( const cv::Mat &src_img);
  virtual void computePointsNormalsDirections( std::vector< cv::Point > &points,
                                               std::vector<float> &normals );

  int low_threshold_;
  int ratio_;
  bool use_rgb_;

  cv::Mat edge_map_;

  std::vector< std::vector<cv::Point> > dir_points_;
};