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
#include <stdexcept>

#include "cv_ext/cv_ext.h"
#include "raster_object_model3D.h"
#include "object_templates.h"


template <typename _T> class ObjectTemplateGeneratorBase
{
public:
    
  virtual ~ObjectTemplateGeneratorBase() = 0;

  void enableVerboseMode( bool enable ){ verbose_ = enable; };

  virtual void generate( _T &templ, uint32_t class_id,
                         const Eigen::Quaterniond &r_quat, const Eigen::Vector3d &t_vec) = 0;

  void generate( std::vector<_T, Eigen::aligned_allocator<_T> > &obj_templates, uint32_t class_id,
                 const cv_ext::vector_Quaterniond &r_quats, 
                 const cv_ext::vector_Vector3d &t_vecs, 
                 bool concatenate = false ); 
protected:
    
  bool verbose_ = false;

};


template<typename _T> class ObjectTemplateGenerator : public ObjectTemplateGeneratorBase<_T> {};

template <> class ObjectTemplateGenerator<ObjectTemplate>
  : public ObjectTemplateGeneratorBase<ObjectTemplate>
{
public:

 ~ObjectTemplateGenerator<ObjectTemplate>() override = default;

  virtual void setTemplateModel( const RasterObjectModel3DPtr &model_ptr ){ model_ptr_ = model_ptr; };

  void generate( ObjectTemplate &templ, uint32_t class_id,
                 const Eigen::Quaterniond &r_quat, const Eigen::Vector3d &t_vec) override;
                         
protected:

  RasterObjectModel3DPtr model_ptr_;
};


template <> class ObjectTemplateGenerator<PointSet> 
  : public ObjectTemplateGeneratorBase<PointSet>, public ObjectTemplateGenerator<ObjectTemplate>
{
public:

  ~ObjectTemplateGenerator() override = default;

  using ObjectTemplateGeneratorBase<PointSet>::enableVerboseMode;
  using ObjectTemplateGeneratorBase<PointSet>::generate;
  //using ObjectTemplateGenerator<ObjectTemplate>::setTemplateModel;

  void setTemplateModel( const RasterObjectModel3DPtr &model_ptr ) override;

  /** @brief Max number of image points for each template */
  void setMaxNumImgPoints( double num_pts ){ max_num_pts_ = num_pts; };
  /** @brief Minimum spacing between image points */
  void setImgPointsSpacing( double spacing ){ if( spacing >= 0 ) img_pts_spacing_ = spacing; };
  
  void generate( PointSet &templ, uint32_t class_id,
                 const Eigen::Quaterniond &r_quat, const Eigen::Vector3d &t_vec) override;
protected:

  cv::Rect selectRoundImagePoints(const std::vector<cv::Point2f> &in_pts, std::vector<cv::Point> &out_pts,
                                  std::vector<int> &selected_idx );

  int max_num_pts_ = -1;
  int img_pts_spacing_ = 0;
  
  cv::Mat template_mask_;
};

template <> class ObjectTemplateGenerator<DirIdxPointSet>
  : public ObjectTemplateGeneratorBase<DirIdxPointSet>, ObjectTemplateGenerator<PointSet>
{
public:

  ~ObjectTemplateGenerator() override = default;

  using ObjectTemplateGeneratorBase<DirIdxPointSet>::enableVerboseMode;
  using ObjectTemplateGeneratorBase<DirIdxPointSet>::generate;
  using ObjectTemplateGenerator<PointSet>::setTemplateModel;
  using ObjectTemplateGenerator<PointSet>::setMaxNumImgPoints;
  using ObjectTemplateGenerator<PointSet>::setImgPointsSpacing;

  /** @brief Number of discretized directions for the image points */
  void setImgPtsNumDirections ( int n ) { if (n >= 1) img_pts_num_directions_ = n; };
  
  void generate( DirIdxPointSet &templ, uint32_t class_id,
                 const Eigen::Quaterniond &r_quat, const Eigen::Vector3d &t_vec) override;
protected:

  int img_pts_num_directions_ = 1;
};



template <typename _T> 
  void ObjectTemplateGeneratorBase< _T >:: generate( std::vector<_T, Eigen::aligned_allocator<_T> > &obj_templates, uint32_t class_id,
                                                     const cv_ext::vector_Quaterniond &r_quats,
                                                     const cv_ext::vector_Vector3d &t_vecs,
                                                     bool concatenate )
  
{
  if( r_quats.size() != t_vecs.size() )
   throw std::invalid_argument("ObjectTemplateGeneratorBase<_T> :: generate() r_quats and t_vecs should have the same size"); 
  
  int start_idx, templ_size;
  
  if( concatenate )
  {
    start_idx = obj_templates.size();
    templ_size = obj_templates.size() + r_quats.size();
  }
  else
  {
    start_idx = 0;
    templ_size = r_quats.size();   
  }
  
  obj_templates.resize( templ_size );

  for( int i = start_idx, j = 0; i < templ_size; i++, j++ )
  {
    if( verbose_ && !(i%100) )
      std::cout<<"Generated "<<i<<" of "<<templ_size - 1<<" object templates"<<std::endl;
    generate ( obj_templates[i], class_id, r_quats[j], t_vecs[j] );
  }
}

