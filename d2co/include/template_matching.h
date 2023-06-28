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
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>


struct TemplateMatch
{
  TemplateMatch(){};
  TemplateMatch( int id,  double dist, const cv::Point &offset ) :
    id(id),
    distance(dist),
    img_offset(offset)
    {};

  int id;
  double distance;
  cv::Point img_offset;
};


//   /**
//    * @brief TODO Update it _ Normalize an input match by setting its actual 6D position and no image offset
//    *
//    * @param m An input TemplateMatch
//    * 
//    * Given an input TemplateMatch instance, this function estimates the actual location of the object 
//    * as if it had no image offset. The r_quat and t_vec are set accordling, while img_offset is set to (0,0). 
//    * If the match has no image offset (i.e., img_offset is (0,0)) r_quat and t_vec are not changed.
//    */  
//   void normalizeMatch( const ObjectTemplate &templ, TemplateMatch &m);
// 

template <typename _T> class TemplateMatching
{
public:
  
  virtual ~TemplateMatching(){};

  void enableVerboseMode( bool enable ) { verbose_mode_ = enable; };
  
  /**
   * @brief Enable/disable parallelism for some/all algorithm
   *
   * @param enable If true, some algorithms are run in a parallel region,
   *               otherwise it is run as a single thread
   * 
   * \note If parallelism is enabled, some results may slightly change due to the data processing order
   */
  void enableParallelism ( bool enable )
  {
    parallelism_enabled_ = enable;
  };

  /**
   * @brief Return true is the algorithms are run in a parallel region
   */
  bool isParallelismEnabled() const
  {
    return parallelism_enabled_;
  };

  virtual void setTemplatesVector( std::shared_ptr< std::vector<_T, Eigen::aligned_allocator<_T> > > &tv_ptr ){ tv_ptr_ = tv_ptr; };
  const std::vector<_T, Eigen::aligned_allocator<_T> > templatesVector() const { return *tv_ptr_;};
  
  virtual void match( int num_best_matches, std::vector< TemplateMatch > &matches,
                      int image_step, int match_cell_size ) const = 0;
  virtual float templateDist( const _T &object_template ) const = 0;

protected:

  double verbose_mode_ = false;
  bool parallelism_enabled_ = false;
  std::shared_ptr< std::vector<_T, Eigen::aligned_allocator<_T> > > tv_ptr_;
};
