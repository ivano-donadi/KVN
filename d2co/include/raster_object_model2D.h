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
typedef std::shared_ptr< class RasterObjectModel2D > RasterObjectModel2DPtr;

class RasterObjectModel2D : public RasterObjectModel
{
public:
  
  RasterObjectModel2D();
  ~RasterObjectModel2D(){};
    
  virtual bool setModelFile( const std::string &filename );
  virtual bool allVisiblePoints() const{ return true; };
  virtual void computeRaster();
  virtual void update();

  virtual const std::vector<cv::Point3f> &getPoints( bool only_visible_points = true ) const;
  virtual const std::vector<cv::Point3f> &getDPoints( bool only_visible_points = true) const;
  virtual const std::vector<cv::Vec6f> &getSegments( bool only_visible_segments = true ) const;
  virtual const std::vector<cv::Point3f> &getDSegments( bool only_visible_segments = true ) const;

protected:

private:
  
  inline void addLine( cv::Point3f &p0, cv::Point3f &p1 );
  inline void addCircleArc( cv::Point3f &center, float radius, float start_ang, float end_ang );
  inline void addEllipseArc( cv::Point3f &center, cv::Point3f &major_axis_ep,
                             float minor_major_ratio, float start_ang, float end_ang );

  std::vector<cv::Point3f> pts_;
  std::vector<cv::Point3f> d_pts_;

  std::vector<cv::Vec6f> segs_;
  std::vector<cv::Point3f> d_segs_;
  
  /* Pimpl idiom */
  class CadModel; 
  std::shared_ptr< CadModel > cad_ptr_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
