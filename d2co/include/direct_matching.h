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

// #include <memory>
// #include <vector>
// #include <opencv2/opencv.hpp>
// 
// #include "cv_ext/cv_ext.h"
// #include "raster_object_model.h"
// #include "template_matching.h"
// 
// struct ScaledImage { cv::Mat img; double scale; };
// typedef std::shared_ptr< std::vector< ScaledImage > > ScaledImagesListPtr;
// 
// void computeGradientMagnitudePyr( const cv::Mat& src_img, ScaledImagesListPtr &g_mag_pyr_ptr,
//                                   unsigned int pyr_levels, double smooth_std = 1.0 );
// 
// class DirectMatching : public TemplateMatching<PointSet>
// {
// public:
//   
//   DirectMatching();
//   virtual ~DirectMatching(){};  
// 
//   void setInput( const ScaledImagesListPtr &img_pyr_ptr );
//   
//   virtual void match( int num_best_matches, std::vector< TemplateMatch > &matches, int image_step = -1 ) const override;
//   virtual float templateDist( const DirIdxPointSet &object_template ) const override;
// 
// protected:
//   
// 
//   
// private:
//   
// //   ScaledImagesListPtr imgs_list_ptr_;
// //   std::vector<cv_ext::PinholeCameraModel> scaled_cam_models_;
// };
