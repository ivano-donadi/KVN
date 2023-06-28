#include "object_templates.h"

using namespace std;

ObjectTemplate::~ObjectTemplate() {}

void ObjectTemplate::binaryRead ( ifstream& in )
{
  in.read(reinterpret_cast<char*>(&class_id), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&obj_r_quat.coeffs()), 4*sizeof(double));
  in.read(reinterpret_cast<char*>(obj_t_vec.data()), 3*sizeof(double));
  in.read(reinterpret_cast<char*>(&obj_bbox3d), sizeof(cv_ext::Box3f));
  proj_obj_bbox3d.resize(8);
  in.read(reinterpret_cast<char*>(&proj_obj_bbox3d[0]), 8*sizeof(cv::Point2f));
}

void ObjectTemplate::binaryWrite ( ostream& out ) const
{
  out.write(reinterpret_cast<const char*>(&class_id), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&obj_r_quat.coeffs()), 4*sizeof(double));
  out.write(reinterpret_cast<const char*>(obj_t_vec.data()), 3*sizeof(double));
  out.write(reinterpret_cast<const char*>(&obj_bbox3d), sizeof(cv_ext::Box3f));
  out.write(reinterpret_cast<const char*>(&proj_obj_bbox3d[0]), 8*sizeof(cv::Point2f));
}

void PointSet::binaryRead ( ifstream& in )
{
  ObjectTemplate::binaryRead ( in );
  
  size_t num_pts;
  in.read(reinterpret_cast<char*>(&num_pts), sizeof(size_t));
  obj_pts.resize(num_pts);
  proj_pts.resize(num_pts);
  in.read(reinterpret_cast<char*>(&obj_pts[0]), num_pts*sizeof(cv::Point3f));
  in.read(reinterpret_cast<char*>(&proj_pts[0]), num_pts*sizeof(cv::Point));
  in.read(reinterpret_cast<char*>(&bbox), sizeof(cv::Rect));
}

void PointSet::binaryWrite ( ostream& out ) const
{
  ObjectTemplate::binaryWrite ( out );

  size_t num_pts = obj_pts.size();
  out.write(reinterpret_cast<const char*>(&num_pts), sizeof(size_t));
  out.write(reinterpret_cast<const char*>(&obj_pts[0]), num_pts*sizeof(cv::Point3f));
  out.write(reinterpret_cast<const char*>(&proj_pts[0]), num_pts*sizeof(cv::Point));
  out.write(reinterpret_cast<const char*>(&bbox), sizeof(cv::Rect));
}

void DirIdxPointSet::binaryRead ( ifstream& in )
{
  PointSet::binaryRead ( in );
  obj_d_pts.resize(obj_pts.size());
  dir_idx.resize(obj_pts.size());
  in.read(reinterpret_cast<char*>(&obj_d_pts[0]), obj_pts.size()*sizeof(cv::Point3f));
  in.read(reinterpret_cast<char*>(&dir_idx[0]), obj_pts.size()*sizeof(int));
}

void DirIdxPointSet::binaryWrite ( ostream& out ) const
{
  PointSet::binaryWrite ( out );

  out.write(reinterpret_cast<const char*>(&obj_d_pts[0]), obj_pts.size()*sizeof(cv::Point3f));
  out.write(reinterpret_cast<const char*>(&dir_idx[0]), dir_idx.size()*sizeof(int));
}


ObjectTemplatePnP::ObjectTemplatePnP( const cv_ext::PinholeCameraModel &cam_model )
{
  setCamModel(cam_model);
}

void ObjectTemplatePnP::setCamModel(const cv_ext::PinholeCameraModel &cam_model)
{
  ipnp_.setCamModel(cam_model);
}

void ObjectTemplatePnP::solve( const ObjectTemplate &obj_templ, const cv::Point &img_offset,
                               double r_quat[4], double t_vec[3] )
{
  if( fix_z_ )
    ipnp_.constrainsTranslationComponent(2, obj_templ.obj_t_vec(2));
  else
    ipnp_.removeConstraints();

  vector<cv::Point2f> proj_bb_pts;
  proj_bb_pts.reserve(obj_templ.proj_obj_bbox3d.size());
  for (auto &p : obj_templ.proj_obj_bbox3d)
  {
    proj_bb_pts.emplace_back(p.x + static_cast<float>(img_offset.x),
                             p.y + static_cast<float>(img_offset.y));
  }

  ipnp_.compute(obj_templ.obj_bbox3d.vertices(), proj_bb_pts, r_quat, t_vec);
}

void ObjectTemplatePnP::solve( const ObjectTemplate &obj_templ, const cv::Point &img_offset,
                               Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec )
{
  if( fix_z_ )
    ipnp_.constrainsTranslationComponent(2, obj_templ.obj_t_vec(2));
  else
    ipnp_.removeConstraints();

  vector<cv::Point2f> proj_bb_pts;
  proj_bb_pts.reserve(obj_templ.proj_obj_bbox3d.size());
  for (auto &p : obj_templ.proj_obj_bbox3d)
  {
    proj_bb_pts.emplace_back(p.x + static_cast<float>(img_offset.x),
                             p.y + static_cast<float>(img_offset.y));
  }

  ipnp_.compute(obj_templ.obj_bbox3d.vertices(), proj_bb_pts, r_quat, t_vec);
}

void ObjectTemplatePnP::solve( const ObjectTemplate &obj_templ, const cv::Point &img_offset,
                               cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec )
{
  if( fix_z_ )
    ipnp_.constrainsTranslationComponent(2, obj_templ.obj_t_vec(2));
  else
    ipnp_.removeConstraints();

  vector<cv::Point2f> proj_bb_pts;
  proj_bb_pts.reserve(obj_templ.proj_obj_bbox3d.size());
  for (auto &p : obj_templ.proj_obj_bbox3d)
  {
    proj_bb_pts.emplace_back(p.x + static_cast<float>(img_offset.x),
                             p.y + static_cast<float>(img_offset.y));
  }

  ipnp_.compute(obj_templ.obj_bbox3d.vertices(), proj_bb_pts, r_vec, t_vec);
}
