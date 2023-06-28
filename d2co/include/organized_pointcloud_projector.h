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

#ifndef ORGANIZED_POINTCLOUD_PROJECTOR_H
#define ORGANIZED_POINTCLOUD_PROJECTOR_H

#include <memory>

#include <pcl/common/common_headers.h>
#include <opencv2/opencv.hpp>

#include "cv_ext/conversions.h"
#include "cv_ext/pinhole_scene_projector.h"
#include "cv_ext/debug_tools.h"

/** TODO
 * 
 */

template < typename T > class OrganizedPointCloudProjector
{
public:
  
  virtual ~OrganizedPointCloudProjector(){};
    
  static std::shared_ptr< OrganizedPointCloudProjector< pcl::PointXYZ> >
    createFromDepth( const cv::Mat &depth_img, const cv_ext::PinholeCameraModel &cam_model,
                     float min_depth = 0.01f, float max_depth = std::numeric_limits<float>::max() )
  {
    if( depth_img.depth() != cv::DataType<float>::type || depth_img.channels() != 1 ||
        !depth_img.rows || !depth_img.cols )
      throw std::invalid_argument("Invalid depth image");
  
    return std::shared_ptr< OrganizedPointCloudProjector< pcl::PointXYZ > >
      ( new OrganizedPointCloudProjector< pcl::PointXYZ > ( depth_img, cv::Mat(), cam_model, 
                                                            min_depth, max_depth ) );
  };
  
  static std::shared_ptr< OrganizedPointCloudProjector< pcl::PointXYZRGB > >
    createFromDepth( const cv::Mat &depth_img, const cv::Mat &rgb_img, 
                     const cv_ext::PinholeCameraModel &cam_model,
                     float min_depth = 0.01f, float max_depth = std::numeric_limits<float>::max() )
  {
    if( depth_img.depth() != cv::DataType<float>::type || depth_img.channels() != 1 ||
        !depth_img.rows || !depth_img.cols ||
        rgb_img.depth() != cv::DataType<uchar>::type || rgb_img.channels() != 3 ||
        rgb_img.rows != depth_img.rows || rgb_img.cols != depth_img.cols )
      throw std::invalid_argument("Invalid input images");
    
    return std::shared_ptr< OrganizedPointCloudProjector< pcl::PointXYZRGB > >
      ( new OrganizedPointCloudProjector< pcl::PointXYZRGB > ( depth_img, rgb_img, cam_model, 
                                                               min_depth, max_depth ) );
  };
  
  const cv::Mat & getOrganizedOpenCvCloud() const { return org_opencv_cloud_mat_; };
  const cv::Mat & getDepthMask() const {return depth_mask_; };

  const typename pcl::PointCloud< T >::Ptr getUnorganizedPCLCloud() const { return unorg_pcl_cloud_ptr_; };  
  const std::vector<cv::Point3f> & getUnorganizedOpenCvCloud() const{ return unorg_opencv_cloud_vec_; };  
  
  std::shared_ptr< std::vector<cv::Point2f> > projectCloud( const Eigen::Matrix4d &transformation,
                                                              bool discard_occlusions = false, 
                                                              unsigned int sample_step = 1,
                                                              const cv::Mat &cloud_mask = cv::Mat() ) const;
  
private:
  
  OrganizedPointCloudProjector( const cv::Mat &depth_img, const cv::Mat &rgb_img, 
                       const cv_ext::PinholeCameraModel &cam_model,
                       float min_depth, float max_depth )
  : cam_model_(cam_model), min_depth_(min_depth), max_depth_(max_depth)
  {  
    depth2PointCloud( depth_img, rgb_img, unorg_pcl_cloud_ptr_ );
  };
  
//   void depth2PointCloud( const cv::Mat &depth_img, const cv::Mat &rgb_img, 
//                          pcl::PointCloud< T >::Ptr &pc_ptr )
//   {
//     throw std::runtime_error("Unsopported cloud");
//   };
  
  void depth2PointCloud( const cv::Mat &depth_img, const cv::Mat &rgb_img, 
                         pcl::PointCloud< pcl::PointXYZ >::Ptr &pc_ptr );
  void depth2PointCloud( const cv::Mat &depth_img, const cv::Mat &rgb_img, 
                         pcl::PointCloud< pcl::PointXYZRGB >::Ptr &pc_ptr );
  
  cv::Mat org_opencv_cloud_mat_;
  cv::Mat depth_mask_;
  
  typename pcl::PointCloud< T >::Ptr unorg_pcl_cloud_ptr_;
  std::vector<cv::Point3f> unorg_opencv_cloud_vec_; 

  cv_ext::PinholeCameraModel cam_model_;
  
  float min_depth_, max_depth_;
};

template< typename T > void OrganizedPointCloudProjector<T> :: 
  depth2PointCloud( const cv::Mat &depth_img, const cv::Mat &rgb_img,
                    pcl::PointCloud< pcl::PointXYZ >::Ptr &pc_ptr )
{
  pc_ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  auto &pcl_points_vec = unorg_pcl_cloud_ptr_->points;
  
  org_opencv_cloud_mat_ = cv::Mat(depth_img.rows, depth_img.cols, cv::DataType<cv::Point3f>::type );
  depth_mask_ = cv::Mat(depth_img.rows, depth_img.cols, cv::DataType<uchar>::type );
  
  pcl_points_vec.resize( depth_img.total() );
  unorg_opencv_cloud_vec_.resize( depth_img.total() );
  
  int i_pt = 0;
  for (int y = 0; y < depth_img.rows; y++)
  {
    const float *depth = depth_img.ptr<float>(y);
    cv::Point3f *organized_cloud = org_opencv_cloud_mat_.ptr<cv::Point3f>(y);
    uchar *mask = depth_mask_.ptr<uchar>(y);
    
    for (int x = 0; x < depth_img.cols; x++, depth++, organized_cloud++, mask++)
    {
      float d = *depth;
      if ( d >= min_depth_ && d <= max_depth_ )
      {
        float img_pt[2] = {x, y}, scene_pt[3];
        // Optimize here
        if( cam_model_.hasDistCoeff() )
          cam_model_.unproject(img_pt, d, scene_pt );
        else
          cam_model_.unprojectWithoutDistortion(img_pt, d, scene_pt );
        
        pcl_points_vec[i_pt].x = unorg_opencv_cloud_vec_[i_pt].x = scene_pt[0];
        pcl_points_vec[i_pt].y = unorg_opencv_cloud_vec_[i_pt].y = scene_pt[1];
        pcl_points_vec[i_pt].z = unorg_opencv_cloud_vec_[i_pt].z = scene_pt[2];
        
        *organized_cloud = unorg_opencv_cloud_vec_[i_pt];
        *mask = 255;
        i_pt++;
      }
      else
      {
        *organized_cloud = cv::Point3f(NAN, NAN, NAN );
        *mask = 0;
      }
    }
  }
  
  pcl_points_vec.resize(i_pt);
  unorg_opencv_cloud_vec_.resize(i_pt);
  
  pc_ptr->width = (int) pcl_points_vec.size();
  pc_ptr->height = 1;
}

template< typename T > void OrganizedPointCloudProjector<T> ::
  depth2PointCloud( const cv::Mat &depth_img, const cv::Mat &rgb_img, 
                    pcl::PointCloud< pcl::PointXYZRGB >::Ptr &pc_ptr )
{
  pc_ptr = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
  auto &pcl_points_vec = pc_ptr->points;
  
  org_opencv_cloud_mat_ = cv::Mat(depth_img.rows, depth_img.cols, cv::DataType<cv::Point3f>::type );
  depth_mask_ = cv::Mat(depth_img.rows, depth_img.cols, cv::DataType<uchar>::type );
  
  pcl_points_vec.resize( depth_img.total() );
  unorg_opencv_cloud_vec_.resize( depth_img.total() );
  
  int i_pt = 0;
  for (int y = 0; y < depth_img.rows; y++)
  {
    const float *depth = depth_img.ptr<float>(y);
    cv::Point3f *organized_cloud = org_opencv_cloud_mat_.ptr<cv::Point3f>(y);
    uchar *mask = depth_mask_.ptr<uchar>(y);
    const uchar *bgr = rgb_img.ptr<uchar>(y);
    uchar r, g, b;
    
    for ( int x = 0; x < depth_img.cols; x++, depth++, organized_cloud++, mask++, bgr+=3 )
    {
      float d = *depth;
      if ( d >= min_depth_ && d <= max_depth_ )
      {
        float img_pt[2] = {x, y}, scene_pt[3];
        // Optimize here
        if( cam_model_.hasDistCoeff() )
          cam_model_.unproject(img_pt, d, scene_pt );
        else
          cam_model_.unprojectWithoutDistortion(img_pt, d, scene_pt );
                 
        b = bgr[0];
        g = bgr[1];
        r = bgr[2];
        
        uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                        static_cast<uint32_t>(g) << 8 | 
                        static_cast<uint32_t>(b));

        pcl_points_vec[i_pt].x = unorg_opencv_cloud_vec_[i_pt].x = scene_pt[0];
        pcl_points_vec[i_pt].y = unorg_opencv_cloud_vec_[i_pt].y = scene_pt[1];
        pcl_points_vec[i_pt].z = unorg_opencv_cloud_vec_[i_pt].z = scene_pt[2];
        
        pcl_points_vec[i_pt].rgb = *reinterpret_cast<float*>(&rgb);
        
        *organized_cloud = unorg_opencv_cloud_vec_[i_pt];
        *mask = 255;
        i_pt++;
      }
      else
      {
        *organized_cloud = cv::Point3f(NAN, NAN, NAN );
        *mask = 0;
      }
    }
  }
  
  pcl_points_vec.resize(i_pt);
  unorg_opencv_cloud_vec_.resize(i_pt);
  
  pc_ptr->width = (int) pcl_points_vec.size();
  pc_ptr->height = 1;
};

// template< typename T > std::shared_ptr< std::vector<cv::Point3f> >
//   OrganizedPointCloudProjector<T> :: getUnorganizedOpenCvCloud( const cv::Mat &cloud_mask,
//                                                                 int sample_step ) const 
// {
//   if( cloud_mask.depth() != cv::DataType<uchar>::type || cloud_mask.channels() != 1 ||
//       cloud_mask.rows != org_opencv_cloud_mat_.rows || cloud_mask.cols != org_opencv_cloud_mat_.cols )
//     throw std::invalid_argument("Invalid image mask"); 
// 
//   std::shared_ptr< std::vector<cv::Point3f> > cv_points_ptr( new std::vector<cv::Point3f> );
//   std::vector<cv::Point3f> &cv_points = *(cv_points_ptr.get());
//     
//   cv_points.resize( org_opencv_cloud_mat_.total() );
//   
//   int i_pt = 0;
//   for (int y = 0; y < org_opencv_cloud_mat_.rows; y += sample_step)
//   {
//     cv::Point3f *organized_cloud = org_opencv_cloud_mat_.ptr<cv::Point3f>(y);
//     uchar *mask = cloud_mask.ptr<uchar>(y);
//     
//     for (int x = 0; x < organized_cloud.cols; x += sample_step, 
//                                               organized_cloud += sample_step, 
//                                               mask += sample_step)
//     {
//       if(*mask)
//         cv_points[i_pt++] = *organized_cloud;
//     }
//   }
//   cv_points.resize(i_pt);
//   return cv_points_ptr;
// }

template< typename T > std::shared_ptr< std::vector<cv::Point2f> >
  OrganizedPointCloudProjector<T> :: projectCloud( const Eigen::Matrix4d &transformation, 
                                                   bool discard_occlusions,
                                                   unsigned int sample_step,
                                                   const cv::Mat &cloud_mask ) const 
{
  if( !cloud_mask.empty() &&
      ( cloud_mask.depth() != cv::DataType<uchar>::type || cloud_mask.channels() != 1 ||
        cloud_mask.rows != org_opencv_cloud_mat_.rows || cloud_mask.cols != org_opencv_cloud_mat_.cols) )
    throw std::invalid_argument("Invalid image mask"); 
  
  std::shared_ptr< std::vector<cv::Point2f> > coords_ptr( new std::vector<cv::Point2f> );
  std::vector<cv::Point2f> &coords = *(coords_ptr.get());
  
  cv::Mat r_vec, t_vec;
  cv_ext::transfMat2Exp<double>(transformation, r_vec, t_vec );
  
  cv_ext::PinholeSceneProjector scene_projector( cam_model_ );
  scene_projector.setTransformation( r_vec, t_vec );
  
  if( !cloud_mask.empty() )
  {
    std::vector<cv::Point3f> points3d;
      
    points3d.resize( org_opencv_cloud_mat_.total() );
    
    int i_pt = 0;
    for (int y = 0; y < org_opencv_cloud_mat_.rows; y += sample_step)
    {
      const cv::Point3f *organized_cloud_p = org_opencv_cloud_mat_.ptr<cv::Point3f>(y);
      const uchar *mask_p = cloud_mask.ptr<uchar>(y);
      
      for (int x = 0; x < org_opencv_cloud_mat_.cols; x += sample_step, 
                                                      organized_cloud_p += sample_step, 
                                                      mask_p += sample_step)
      {
        if(*mask_p)
          points3d[i_pt++] = *organized_cloud_p;
      }
    }
    points3d.resize(i_pt);
 
    scene_projector.projectPoints( points3d, coords, discard_occlusions );
    // if(discard_occlusions)
    //   cv_debug::showDebugImage(scene_projector.getOcclusionsMask(), "occ mask");
  } 
  else
  {
    scene_projector.projectPoints( unorg_opencv_cloud_vec_, coords, discard_occlusions );
    // if(discard_occlusions)
    //   cv_debug::showDebugImage(scene_projector.getOcclusionsMask(), "occ mask");
  }

  return coords_ptr;
};

#endif // ORGANIZED_POINTCLOUD_PROJECTOR_H
