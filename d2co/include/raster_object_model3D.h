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

#include "raster_object_model.h"

/** @brief Shared pointer typedef */
typedef std::shared_ptr< class RasterObjectModel3D > RasterObjectModel3DPtr;

class RasterObjectModel3D : public RasterObjectModel
{
public:
  
  RasterObjectModel3D();
  ~RasterObjectModel3D();
    
  virtual bool setModelFile( const std::string& filename );
  virtual bool allVisiblePoints() const{ return false; };
  virtual void computeRaster();
  virtual void update();
  
  virtual const std::vector<cv::Point3f> &getPoints( bool only_visible_points = true ) const;
  virtual const std::vector<cv::Point3f> &getDPoints( bool only_visible_points = true) const;
  virtual const std::vector<cv::Vec6f> &getSegments( bool only_visible_segments = true ) const;
  virtual const std::vector<cv::Point3f> &getDSegments( bool only_visible_segments = true ) const;
  
  void setRenderZNear( float dist ){ render_z_near_ = dist; }
  void setRenderZFar( float dist ){ render_z_far_ = dist; }
  void setNormalEpsilon( float epsilon ){ normal_epsilon2_ = epsilon*epsilon; };

  /**
   * @brief This method request to load and use the vertex color information (if available)
   * 
   * After loading the model, it is possible to check if the vertex color information has been loaded
   * by calling the vertexColorsEnabled() method
   *
   * The methods requestVertexColors() and enableUniformColor() are mutually exclusive
   * @note This method should be called before computerRaster().
   */
  void requestVertexColors ()
  {
    vertex_color_ = cv::Scalar(-1);
    has_color_ = true;
  };

  /**
   * @brief This method request to render the model by using an uniform color
   *
   * @param[in] color An RGB color (each component in the range [0,255])
   *
   * the color can be changed later with setVerticesColor()
   * The methods requestVertexColors() and enableUniformColor() are mutually exclusive
   * @note This method should be called before computerRaster().
   */
  void enableUniformColor ( cv::Scalar color )
  {
    vertex_color_ = prev_vertex_color_ = color;
    has_color_ = true;
  };

  /**
   * @brief Set a new color used to uniformly render the model
   *
   * @param[in] color An RGB color (each component in the range [0,255])
   *
   * @note If enableUniformColor() was not called in advance, this method has no effect
   */
  void setVerticesColor ( cv::Scalar color )
  {
    vertex_color_ = color;
  };

  /**
   * @brief This method request to use lighting in model rendering
   *
   * @param[in] enable If true, add basig shading to the rendered object
   * 
   * @note Lighting in rendering is actually activeted only 
   *       if vertex color has been enabled (see enableVertexColors())
   */
  void requestRenderLighting ()
  {
    light_on_ = true;
  };
  
  /**
   * @brief Set the position of the point light
   *
   * @param[in] pos 3D light position
   * 
   * Two point light are used: placed on position pos, the other one at the position -pos
   * The lighting has to be enabled, see requestRenderLighting()
   */  
  void setPointLightPos( cv::Point3f pos)
  {
    point_light_pos_ = pos;
  };
  
  /**
   * @brief Set the direction of the directional light
   *
   * @param[in] dir 3D light direction
   * 
   * The lighting has to be enabled, see requestRenderLighting()
   */  
  void setLightDirection( cv::Point3f dir)
  {
     light_dir_ = dir;
  };

  
  float renderZNear() const { return render_z_near_; };
  float renderZFar() const { return render_z_far_; };
  float normalEpsilon() const { return std::sqrt(normal_epsilon2_); };
  
  /**
   * @brief Return true if vertex color information is used
   */
  bool vertexColorsEnabled() const { return has_color_; }
  
  /**
   * @brief Provide the uniform vertex color possibly set with setVerticesColor()
   * 
   * @return A RGB color (each component in the range [0,255]), or (-1,-1,-1) if no color has been
   *         set with setVerticesColor()
   */  
  cv::Scalar getVerticesColor()
  {
    return vertex_color_;
  }
  
  /**
   * @brief Return true if lighting in model rendering is enabled
   */
  bool renderLightingEnabled () { return light_on_; };
  
  /**
   * @brief Provide the current position of the point light
   */  
  cv::Point3f getLightPos() { return point_light_pos_; };
 
  /**
   * @brief Provide the current direction of the directional light
   */  
  cv::Point3f getLightDir() { return light_dir_; };

  /**
   * @brief Provide a depth map of the model
   */  
  cv::Mat getModelDepthMap();

  /**
   * @brief Provide an image with the colored render of the model
   *
   * @param background_color An RGB background image
   *
   * @note This method will return an empty image if the vertex color is not enabled (see
   *       requestVertexColors() or setVerticesColor())
   */
  cv::Mat getRenderedModel( const cv::Mat &background_img = cv::Mat() );

  /**
   * @brief Provide an image with the colored render of the model
   * 
   * @param background_color The RGB background color (each component in the range [0,255]) 
   * 
   * @note This method will return an empty image if the vertex color is not enabled (see 
   *       requestVertexColors() or setVerticesColor())
   */
  cv::Mat getRenderedModel( const cv::Scalar &background_color );
  
  virtual cv::Mat getMask();


  
private:

  bool initOpenGL();
  void createShader();
  void loadMesh();
  void clearMask();

  inline void addSegment(cv::Point3f &p0, cv::Point3f &p1 );
  inline void addVisibleSegment(cv::Point3f &p0, cv::Point3f &p1 );
  inline float glDepthBuf2Depth( float d );
  bool checkPointOcclusion( cv::Point3f& p );

  bool has_color_ = false, light_on_ = false;
  cv::Scalar vertex_color_ = cv::Scalar(-1), prev_vertex_color_ = cv::Scalar(-1);
  cv::Point3f point_light_pos_ = cv::Point3f(1,0,0),
              light_dir_ = cv::Point3f(0,0,-1);
  std::vector<cv::Point3f> pts_, vis_pts_, *vis_pts_p_;
  std::vector<cv::Point3f> d_pts_, vis_d_pts_, *vis_d_pts_p_;

  std::vector<cv::Vec6f> segs_, vis_segs_, *vis_segs_p_;
  std::vector<cv::Point3f> d_segs_, vis_d_segs_, *vis_d_segs_p_;
    
  /* Pimpl idiom */
  class MeshModel; 
  std::unique_ptr< MeshModel > mesh_model_ptr_;
  
  cv::Size render_win_size_ = cv::Size ( 0, 0 );
  float render_z_near_ = 0.05f, 
        render_z_far_ = 5.0f;
  float depth_buffer_epsilon_ = 0.001f, 
        normal_epsilon2_ = 0.4f*0.4f;

  std::vector<float> depth_buf_data_;
  std::vector<cv::Vec3f> normal_buf_data_;
  std::vector<cv::Vec3b> color_buf_data_;

  cv::Mat depth_buf_, normal_buf_, color_buf_;

  float depth_transf_a_, depth_transf_b_, depth_transf_c_;
  bool raster_updated_ = false, raster_initiliazed_ = false;

  cv::Mat model_mask_;
  std::vector<cv::Point> mask_pts;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
