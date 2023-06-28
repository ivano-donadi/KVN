#include <iostream>
#include <stdexcept>
#include <opencv2/flann.hpp>

#include "object_templates_generator.h"

using namespace std;
using namespace cv;
using namespace cv_ext;

static inline int getDirectionIndex( float direction, int num_directions )
{
  float dir = direction + M_PI/2;
  if( dir >= M_PI/2 )
    dir -= M_PI;
  dir += M_PI/2;

  int i_dir = cv_ext::roundPositive(dir*double(num_directions)/M_PI);
  i_dir %= num_directions;
  return i_dir;
}

static inline void setRegion( Mat &m, int cx, int cy, int spacing_size, uchar val )
{
  int mx = cx + spacing_size, my = cy + spacing_size;
  for( int y = cy - spacing_size; y <= my; y++ )
    for( int x = cx - spacing_size; x <= mx; x++ )
      m.at<uchar>(y,x) = val;
}

template <typename _T> vector<_T> extractSubset( const vector<_T> &in_vec, const vector<int> &sel_idx  )
{
  vector<_T> out_vec;
  out_vec.reserve(sel_idx.size());
  for( auto &idx:sel_idx)
    out_vec.push_back(in_vec[idx]);
  return out_vec;
}

template <> ObjectTemplateGeneratorBase<ObjectTemplate>::~ObjectTemplateGeneratorBase(){}

void ObjectTemplateGenerator< ObjectTemplate >::generate ( ObjectTemplate& templ, uint32_t class_id, 
                                                           const Eigen::Quaterniond& r_quat, 
                                                           const Eigen::Vector3d& t_vec )
{
  if( model_ptr_ == nullptr )

    std::runtime_error("ObjectTemplateGenerator : Object model not set");

  templ.class_id = class_id;
  templ.obj_r_quat = r_quat;
  templ.obj_t_vec = t_vec;
  templ.obj_bbox3d = model_ptr_->getBoundingBox();

  cv_ext::PinholeSceneProjector proj( model_ptr_->cameraModel() );
  proj.setTransformation( r_quat, t_vec );  
  proj.projectPoints( templ.obj_bbox3d.vertices(), templ.proj_obj_bbox3d );
}

void ObjectTemplateGenerator< PointSet >::setTemplateModel( const RasterObjectModel3DPtr &model_ptr )
{
  ObjectTemplateGenerator< ObjectTemplate >::setTemplateModel( model_ptr );

  Size mask_size( model_ptr_->cameraModel().imgWidth() + 2*img_pts_spacing_, 
                  model_ptr_->cameraModel().imgHeight() + 2*img_pts_spacing_ );  
  template_mask_.create(mask_size, DataType<uchar>::type);
  template_mask_.setTo(Scalar(255));
}

template <> ObjectTemplateGeneratorBase<PointSet>::~ObjectTemplateGeneratorBase(){}

cv::Rect ObjectTemplateGenerator<PointSet>::selectRoundImagePoints( const vector<Point2f> &in_pts,
                                                                    vector<Point> &out_pts,
                                                                    vector<int> &selected_idx )
{
  int w = model_ptr_->cameraModel().imgWidth(), h = model_ptr_->cameraModel().imgHeight();
  Point tl(w,h), br(0,0);

  std::vector< cv::Point > tmp_pts;
  vector<int> tmp_idx;

  int num_pts = 0;
  for(int i = 0; i < static_cast<int>(in_pts.size()); i++ )
  {
    int x = cvRound(in_pts[i].x), y = cvRound(in_pts[i].y),
            mask_x = x + img_pts_spacing_, mask_y = y + img_pts_spacing_;

    if( static_cast<unsigned>(x) < static_cast<unsigned>(w) &&
        static_cast<unsigned>(y) < static_cast<unsigned>(h) &&
        template_mask_.at<uchar>(mask_y,mask_x) )
    {
      if( x < tl.x ) tl.x = x;
      if( y < tl.y ) tl.y = y;
      if( x > br.x ) br.x = x;
      if( y > br.y ) br.y = y;

      tmp_pts.emplace_back(x,y);
      tmp_idx.push_back(i);

      setRegion( template_mask_, mask_x, mask_y, img_pts_spacing_, 0 );
      num_pts++;
    }
  }

  // cv::Rect assumes that the top and left boundary of the rectangle are inclusive, while the
  // right and bottom boundaries are not
  br.x++;
  br.y++;

  for(int i = 0; i < static_cast<int>(tmp_pts.size()); i++ )
  {
    int mask_x = tmp_pts[i].x + img_pts_spacing_, mask_y = tmp_pts[i].y + img_pts_spacing_;
    setRegion( template_mask_, mask_x, mask_y, img_pts_spacing_, 255 );
  }

  if( max_num_pts_ > 0 && num_pts > max_num_pts_ )
  {
    // Taken from selectScatteredFeatures() in OpenCV linemod implementation
    float distance = static_cast<float>(tmp_pts.size()) / static_cast<float>(max_num_pts_) + 1.0f;

    out_pts.clear();
    out_pts.reserve(max_num_pts_);
    selected_idx.clear();
    selected_idx.reserve(max_num_pts_);

    float distance_sq = distance * distance;
    int i = 0;
    while (int(selected_idx.size()) < max_num_pts_)
    {
      const Point2f &new_p = in_pts[tmp_idx[i]];

      bool keep = true;
      for ( int j = 0; (j < (int)selected_idx.size()) && keep; ++j )
      {
        const Point2f &p = in_pts[selected_idx[j]];
        keep = (new_p.x - p.x)*(new_p.x - p.x) + (new_p.y - p.y)*(new_p.y - p.y) >= distance_sq;
      }
      if (keep)
      {
        out_pts.push_back(tmp_pts[i]);
        selected_idx.push_back(tmp_idx[i]);
      }

      if (++i == (int)tmp_idx.size())
      {
        // Start back at beginning, and relax required distance
        i = 0;
        distance -= 1.0f;
        distance_sq = distance * distance;
      }
    }
  }
  else
  {
    out_pts = tmp_pts;
    selected_idx = tmp_idx;
  }

  return Rect(tl,br);
}

void ObjectTemplateGenerator< PointSet >::generate ( PointSet& templ, uint32_t class_id,
                                                     const Eigen::Quaterniond& r_quat, const Eigen::Vector3d& t_vec )
{
  ObjectTemplateGenerator< ObjectTemplate >::generate(templ, class_id, r_quat, t_vec);

  model_ptr_->setModelView ( r_quat, t_vec );
  vector<Point3f> obj_pts = model_ptr_->getPoints(true);;
  vector<Point2f> proj_pts;
  model_ptr_->projectRasterPoints ( proj_pts, true );

  vector<int> selected_idx;
  templ.bbox = selectRoundImagePoints( proj_pts, templ.proj_pts, selected_idx);
  templ.obj_pts = extractSubset( obj_pts, selected_idx ) ;

  // DEBUG CODE
//   Mat dbg_img( template_mask_.size(), DataType<cv::Vec3b>::type, Scalar(0));
//   cv_ext::drawPoints(dbg_img, template_pts, Scalar(255,255,255));
//   cv_ext::drawPoints(dbg_img, templ.proj_obj_bbox3d, Scalar(0,0,255));
//   cv_ext::showImage(dbg_img);
  // END DEBUG CODE
}

template <> ObjectTemplateGeneratorBase<DirIdxPointSet>::~ObjectTemplateGeneratorBase(){}

void ObjectTemplateGenerator< DirIdxPointSet >::generate ( DirIdxPointSet& templ, uint32_t class_id, 
                                                           const Eigen::Quaterniond& r_quat,
                                                           const Eigen::Vector3d& t_vec )
{
  ObjectTemplateGenerator< ObjectTemplate >::generate(templ, class_id, r_quat, t_vec);

  model_ptr_->setModelView ( r_quat, t_vec );
  vector<Point3f> obj_pts = model_ptr_->getPoints(true);
  vector<Point3f> obj_d_pts = model_ptr_->getDPoints(true);
  vector<Point2f> proj_pts;
  vector<float> norm_dirs;
  model_ptr_->projectRasterPoints ( proj_pts, norm_dirs, true );

  vector<int> selected_idx;
  templ.bbox = selectRoundImagePoints( proj_pts, templ.proj_pts, selected_idx);
  templ.obj_pts = extractSubset( obj_pts, selected_idx );
  templ.obj_d_pts = extractSubset( obj_d_pts, selected_idx );

  templ.dir_idx.clear();
  templ.dir_idx.reserve(selected_idx.size());
  for( auto &idx:selected_idx)
    templ.dir_idx.push_back(getDirectionIndex( norm_dirs[idx], img_pts_num_directions_ ));

  // DEBUG CODE  
//   Mat dbg_img( template_mask_.size(), DataType<cv::Vec3b>::type, Scalar(0));
//   cv_ext::drawPoints(dbg_img, template_pts, Scalar(255,255,255));
//   cv_ext::showImage(dbg_img);
//   for( int i_dir = 0; i_dir < num_directions_; i_dir++ )
//   {
//     Mat dbg_img( template_mask_.size(), DataType<cv::Vec3b>::type, Scalar(0));
//     vector< Point > dir_pts;
//     cv_ext::drawPoints(dbg_img, template_pts, Scalar(255,255,255));
//     for( int j_p = 0; j_p < int(template_pts.size()); j_p++ )
//     {
//       if( templ.dir_idx[j_p] == i_dir )
//         dir_pts.push_back(template_pts[j_p]);
//     }
//     cv_ext::drawPoints(dbg_img, dir_pts, Scalar(0,0,255));
//     cv_ext::showImage(dbg_img);
//   }
  // END DEBUG CODE
}
