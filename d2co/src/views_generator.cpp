#include "views_generator.h"

using namespace std;
using namespace cv;
using namespace cv_ext;

void ViewsGenerator::generate( vector_Quaterniond& r_quats, cv_ext::vector_Vector3d& t_vecs )
{  
  r_quats.clear();
  t_vecs.clear();

  if( rp_ranges_.empty() || yaw_ranges_.empty() || z_ranges_.empty() || ang_step_ <= 0 || z_step_ <= 0 )
    return;
  
  vector< Point3f > sphere_pts;
  vector< double > yaws;
  vector< double > zs;
  
  std::vector <cv::Point3f> tmp_pts;
  
  cv_ext::createIcosphereFromAngle( tmp_pts, ang_step_ );
  for( auto &rpr : rp_ranges_ )
  {
    if(rpr.full())
    {
      sphere_pts = tmp_pts;
      break;
    }
    else
    {      
      for( auto &pt : tmp_pts )
      {
        double dist_z = std::sqrt(pt.x*pt.x + pt.y*pt.y);
        double ang = atan2(dist_z, pt.z);
        if( rpr.within(ang) )
          sphere_pts.push_back(pt);
      }
    }
  }
  
  for( auto &yr : yaw_ranges_ )
  {
    if(yr.full())
    {
      int n_step = ceil( (2*M_PI)/ang_step_ );
      double yaw_step = (2*M_PI)/n_step;
      double yaw = -M_PI;
      for( int is = 0; is < n_step; is++, yaw += yaw_step )
        yaws.push_back(yaw);
      break;
    }
    else if( !yr.size() )
      yaws.push_back(yr.start);
    else
    {            
      double ang_size = yr.size();
      int n_step = ceil(ang_size/ang_step_);
      double yaw_step = ang_size/n_step;
      double yaw = yr.start;
      for( int is = 0; is < n_step; is++, yaw += yaw_step )
        yaws.push_back(yaw);  
    }
  }
  
  for( auto &zr : z_ranges_ )
  {
    if( zr.full() )
    {
      throw std::invalid_argument("Z range can't be full"); 
    }
    double z_size = zr.size();
    if( !z_size )
      zs.push_back(zr.start);
    else
    {
      int n_step = ceil(z_size/z_step_);
      double z_step = z_size/n_step;
      double z = zr.start;
      
      for( int is = 0; is <= n_step; is++, z += z_step )
        zs.push_back(z);  
    }
  }
  
  int views_size = sphere_pts.size() * yaws.size() * zs.size();
  
  r_quats.reserve( views_size );
  t_vecs.reserve( views_size );
  
  Eigen::Quaterniond quat;
  
  for(auto &v : sphere_pts )
  {
    for(auto &yaw : yaws )
    {
      for(auto &z : zs )
      { 
        cv_ext::rotationAroundVector(v, quat, yaw);
        r_quats.push_back(quat);
        t_vecs.push_back(Eigen::Vector3d(0,0,z));
      }
    }
  }
}

void ViewsGenerator::reset()
{
  rp_ranges_.clear();
  yaw_ranges_.clear();
  z_ranges_.clear();
  ang_step_ = z_step_ = 0;
}

