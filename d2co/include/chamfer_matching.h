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
#include <map>
#include <opencv2/opencv.hpp>

#include "cv_ext/cv_ext.h"
#include "template_matching.h"
#include "object_templates.h"
#include "distance_transforms.h"

/* TODO
 * 
 * -Remove fast_XXX_map_
 */
template <typename _T> class ChamferMatchingBase : public TemplateMatching<_T>
{
public:
  
  virtual ~ChamferMatchingBase(){};
  
  void  setInput( const cv::Mat &dist_map );
  
  void match( int num_best_matches, std::vector< TemplateMatch > &matches,
              int image_step = -1, int match_cell_size = -1 ) const override;
  float templateDist( const _T &object_template ) const override;

 protected:

  using TemplateMatching<_T>::verbose_mode_;
  using TemplateMatching<_T>::parallelism_enabled_;
  using TemplateMatching<_T>::tv_ptr_;

private:
 
  cv::Mat dist_map_, fast_dist_map_;
};

// class OrientedChamferMatching : public TemplateMatching<DirPointSet>
// {
// public:
//   
//   virtual ~OrientedChamferMatching(){};  
//     
//   void  setInput( const cv::Mat& dist_map, const cv::Mat& closest_dir_map ) const override;
//   
//   virtual void match( int num_best_matches, std::vector< TemplateMatch > &matches, int image_step = -1 ) const override;
//   virtual float templateDist( const DirIdxPointSet &object_template ) const override;
// 
// 
// private:
//     
//   cv::Mat dist_map_, closest_dir_map_, fast_dist_map_;
// };

template <typename _T> class DirectionalChamferMatchingBase : public TemplateMatching<_T>
{
public:
  
  DirectionalChamferMatchingBase() = default;
  virtual ~DirectionalChamferMatchingBase() override = default;

  void setInput( const cv_ext::ImageTensorPtr& dist_map_tensor_ptr );

  void match( int num_best_matches, std::vector< TemplateMatch > &matches,
              int image_step = -1, int match_cell_size = -1 ) const override;
  float templateDist( const _T &object_template ) const override;

 protected:

  using TemplateMatching<_T>::verbose_mode_;
  using TemplateMatching<_T>::parallelism_enabled_;
  using TemplateMatching<_T>::tv_ptr_;

private:
  
  cv_ext::ImageTensorPtr dist_map_tensor_ptr_;
  cv_ext::ImageTensor fast_dist_map_tensor_;
};

typedef ChamferMatchingBase<PointSet> ChamferMatching;
typedef DirectionalChamferMatchingBase<DirIdxPointSet> DirectionalChamferMatching;