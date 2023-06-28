#include <iostream>
#include <boost/concept_check.hpp>
#include <boost/thread/locks.hpp>
 
#include <dime/Input.h>
#include <dime/Output.h>
#include <dime/Model.h>
#include <dime/entities/Vertex.h>
#include <dime/sections/Section.h>
#include <dime/sections/EntitiesSection.h>
#include <dime/entities/Entity.h>
#include <dime/entities/Line.h>
#include <dime/entities/Polyline.h>
#include <dime/entities/Circle.h>
#include <dime/entities/Arc.h>
#include <dime/entities/Ellipse.h>

#include "raster_object_model2D.h"

/* TODO
 * Deals with shared end/start points normal orientations
 * Implement discretizeEllipseArc() (and function for the other entities)
 * WARNING Fix the bug for closest objets
 * 
 */
class RasterObjectModel2D::CadModel 
{
public:
  CadModel(){};
  ~CadModel(){};
  dimeModel model;
};

RasterObjectModel2D ::RasterObjectModel2D()
{
  cad_ptr_ = std::shared_ptr< CadModel > ( new CadModel () );
  std::cout<<"WARNING: bounding box computation not implemented!"<<std::endl;
}

bool RasterObjectModel2D::setModelFile( const std::string &filename )
{
//   boost::lock_guard<boost::mutex> lock(mutex_);
  
  dimeInput dime_in;
  if (!dime_in.setFile(filename.data()))
  {
    std::cerr<<"Error opening: "<<filename<<std::endl;
    return false;
  }

  /* Try reading the file */
  if (!cad_ptr_->model.read(&dime_in))
  {
    std::cerr<<"CAD file read error in line: "<<dime_in.getFilePosition()<<std::endl;
    return false;
  }
  return true;
}

void RasterObjectModel2D::computeRaster()
{
  pts_.clear();
  d_pts_.clear();
  segs_.clear();
  d_segs_.clear();
  
  dimeEntitiesSection *entity_sec = (dimeEntitiesSection*)cad_ptr_->model.findSection("ENTITIES");
  if( entity_sec == NULL )
    return;
  
  float multiplier = 1.0f/unit_meas_;
  
  int n_entities = entity_sec->getNumEntities();
  for(int i = 0; i < n_entities; i++)
  {
    dimeEntity *entity = entity_sec->getEntity(i);
    const std::string ent_name(entity->getEntityName());
    
    if(!ent_name.compare("LINE"))
    {
      dimeLine *line_entity = dynamic_cast<dimeLine*>( entity );
      if( line_entity )
      {
        const dimeVec3f &c0 = line_entity->getCoords(0),
                        &c1 = line_entity->getCoords(1);
                        
        cv::Point3f p0(c0.x, c0.y, c0.z),
                    p1(c1.x, c1.y, c1.z);
        p0 *= multiplier; p1 *= multiplier;
        
        addLine(p0, p1);
      }
    }
    else if(!ent_name.compare("POLYLINE"))
    {
      // WARNING Complete me!!
      dimePolyline *polyline_entity = dynamic_cast<dimePolyline*>( entity );
      if( polyline_entity )
      {
        int surface_type = polyline_entity->getSurfaceType();
        int polyline_type = polyline_entity->getType();
        switch( polyline_type )
        {
          case dimePolyline::POLYLINE:
            std::cout<<"Warning : Polyline type POLYLINE not yet implemented \n";
            break;
          case dimePolyline::POLYFACE_MESH:
            std::cout<<"Warning : Polyline type POLYFACE_MESH not yet implemented \n";                        
            break;
          case dimePolyline::POLYGON_MESH:
            std::cout<<"Warning : Polyline type POLYGON_MESH not yet implemented \n";
            break;
        }

//         switch( surface_type )
//         {
//           case dimePolyline::NONE:
//             std::cout<<"NONE\n";
//             break;
//           case dimePolyline::QUADRIC_BSPLINE:
//             std::cout<<"QUADRIC_BSPLINE\n";                        
//             break;
//           case dimePolyline::CUBIC_BSPLINE:
//             std::cout<<"CUBIC_BSPLINE\n";
//             break;
//           case dimePolyline::BEZIER:
//             std::cout<<"BEZIER\n";
//             break;
//         }

        const int &n_vertices = polyline_entity->getNumCoordVertices(),
                  &n_faces = polyline_entity->getNumIndexVertices();
 
        
        for( int i = 0; i < n_faces ; i++)
        {
          const dimeVertex *index_vertex_p = polyline_entity->getIndexVertex(i);
          const int &n_indices = index_vertex_p->numIndices();
          
          for (int j = 0; j < n_indices - 1; j++)
          {
            const int &idx_0 = index_vertex_p->getIndex(j),
                      &idx_1 = index_vertex_p->getIndex( j + 1 );
          
            if( idx_0 < n_vertices && idx_1 < n_vertices )
            {
              const dimeVec3f &c0 = polyline_entity->getCoordVertex(idx_0)->getCoords(),
                              &c1 = polyline_entity->getCoordVertex(idx_1)->getCoords();
                        
              cv::Point3f p0(c0.x, c0.y, c0.z),
                          p1(c1.x, c1.y, c1.z);
              p0 *= multiplier; p1 *= multiplier;
              
              addLine(p0, p1);
            }
          }
          const int &idx_0 = index_vertex_p->getIndex(n_indices),
                    &idx_1 = index_vertex_p->getIndex( 0 );
        
          if( idx_0 < n_vertices && idx_1 < n_vertices )
          {
            const dimeVec3f &c0 = polyline_entity->getCoordVertex(idx_0)->getCoords(),
                            &c1 = polyline_entity->getCoordVertex(idx_1)->getCoords();
                      
            cv::Point3f p0(c0.x, c0.y, c0.z),
                        p1(c1.x, c1.y, c1.z);
            p0 *= multiplier; p1 *= multiplier;
            
            addLine(p0, p1);
          }
        }
      }
    }
    else if(!ent_name.compare("CIRCLE"))
    {
      dimeCircle *circle_entity = dynamic_cast<dimeCircle*>(entity);
      if( circle_entity )
      {
        const dimeVec3f &c = circle_entity->getCenter();
        cv::Point3f center(c.x, c.y, c.z);
        center *= multiplier;
        float radius = multiplier*circle_entity->getRadius();
        addCircleArc( center, radius, 0, 2*M_PI );
      }
    }
    else if(!ent_name.compare("ARC"))
    {
      dimeArc *arc_entity = dynamic_cast<dimeArc*>(entity);
      if( arc_entity )
      {
        dimeVec3f c;
        arc_entity->getCenter(c);
        cv::Point3f center(c.x, c.y, c.z);
        center *= multiplier;
        float radius = multiplier*arc_entity->getRadius(), 
              s_ang = M_PI*arc_entity->getStartAngle()/180.0,
              e_ang = M_PI*arc_entity->getEndAngle()/180.0;
        // std::cout<<center<<" "<<arc_entity->getStartAngle()<< " "<< arc_entity->getEndAngle()<<std::endl;
        addCircleArc( center, radius, s_ang, e_ang );
      }
    }
    else if(!ent_name.compare("ELLIPSE"))
    {
      dimeEllipse *ellipse_entity = dynamic_cast<dimeEllipse*>(entity);
      if( ellipse_entity )
      {
        const dimeVec3f &c = ellipse_entity->getCenter(),
                        &ep = ellipse_entity->getMajorAxisEndpoint();
        cv::Point3f center(c.x, c.y, c.z), major_axis_ep(ep.x, ep.y, ep.z);
                    
        center *= multiplier;
        major_axis_ep *= multiplier;
        
        float minor_major_ratio = ellipse_entity->getMinorMajorRatio(), 
              s_ang = M_PI*ellipse_entity->getStartParam()/180,
              e_ang = M_PI*ellipse_entity->getEndParam()/180;
              
        addEllipseArc( center, major_axis_ep, minor_major_ratio, s_ang, e_ang );
      }
    }
    else
    {
      std::cout<<"WARNING RasterObjectModel2D::not supported entity : "<<ent_name<<std::endl;
    }
  }
  
  if( centroid_orig_offset_ == CENTROID_ORIG_OFFSET)
  {
    cv::Point3f mean(0,0,0);
    for( int i = 0; i < int(pts_.size()); i++ )
      mean += pts_[i];
    
    mean.x /= float(pts_.size());
    mean.y /= float(pts_.size());
    mean.z /= float(pts_.size());
    
    orig_offset_ = -mean;
  }
  else if( centroid_orig_offset_ == BOUNDING_BOX_CENTER_ORIG_OFFSET )
  {
    cv::Point3f min(std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max());
    cv::Point3f max = -min;
    for( int i = 0; i < int(pts_.size()); i++ )
    {
      if(pts_[i].x > max.x)  max.x = pts_[i].x; else if(pts_[i].x < min.x) min.x = pts_[i].x;
      if(pts_[i].y > max.y)  max.y = pts_[i].y; else if(pts_[i].y < min.y) min.y = pts_[i].y;
      if(pts_[i].z > max.z)  max.z = pts_[i].z; else if(pts_[i].z < min.z) min.z = pts_[i].z;
    }
    
    orig_offset_ = -cv::Point3f( min.x + ( max.x - min.x )/2.0f,
                                 min.y + ( max.y - min.y )/2.0f,
                                 min.z + ( max.z - min.z )/2.0f);
  }
  
  std::cout<<"orig_offset_ : "<<orig_offset_<<std::endl;
  
  for( int i = 0; i < int(pts_.size()); i++ )
  {
    pts_[i] += orig_offset_;
    d_pts_[i] += orig_offset_;
  }
  
  for( int i = 0; i < int(segs_.size()); i++)
  {
    cv::Vec6f &s = segs_[i];
    s[0] += orig_offset_.x;
    s[1] += orig_offset_.y;
    s[2] += orig_offset_.z;
    s[3] += orig_offset_.x;
    s[4] += orig_offset_.y;
    s[5] += orig_offset_.z;
    d_segs_[i] += orig_offset_;
  }
}

void RasterObjectModel2D::update()
{
  // All visible points: do nothing
  return;
}

inline void RasterObjectModel2D::addLine( cv::Point3f &p0, cv::Point3f &p1 )
{
  cv::Point3f dir = p1 - p0;
  float len = std::sqrt(dir.dot(dir));
  if(!len)
    return;
    
  int n_steps = len/step_;
  cv::Point3f dp(p0 + epsilon_*dir); 
  if( len >= min_seg_len_ )
  {
    segs_.push_back( cv::Vec6f(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z) );
    d_segs_.push_back(dp);
  }
  
  if( !n_steps )
  {
    // Push at least the first point
    pts_.push_back(p0);
    d_pts_.push_back(dp);
  }
  else
  {
    float dist_step = 1.0/n_steps;
      
    for(int i = 0; i <= n_steps; i++)
    {
      pts_.push_back(p0 + (i*dist_step)*dir);
      d_pts_.push_back(p0 + (i*dist_step + epsilon_)*dir);
    }
  }
}

inline void RasterObjectModel2D::addCircleArc( cv::Point3f &center, float radius,
                                               float start_ang, float end_ang )
{
  float ang_step = step_/radius;
  
  float ang = start_ang, ang_diff;
  if( start_ang < end_ang )
    ang_diff = end_ang - start_ang;
  else if( start_ang == end_ang )
    return;
  else
    ang_diff = 2*M_PI-(start_ang - end_ang);
  
  int n_ang_step = ang_diff/ang_step;
  
  if( !n_ang_step )
  {
    // Push at least the first point
    cv::Point3f p(radius*cos(ang),radius*sin(ang), 0), 
                dp(radius*cos(ang + epsilon_),radius*sin(ang + epsilon_), 0);
    p += center;
    dp += center;
    pts_.push_back(p);
    d_pts_.push_back(dp);
    if( step_ >= min_seg_len_ )
    {
      segs_.push_back(cv::Vec6f( p.x, p.y, p.z, p.x, p.y, p.z ));
      d_segs_.push_back(dp);
    }
  }
  else
  {
    cv::Point3f prev_p, first_p;
    
    for(int i = 0; i <= n_ang_step; i++)
    {
      cv::Point3f p (radius*cos(ang),radius*sin(ang), 0), 
                  dp(radius*cos(ang + epsilon_),radius*sin(ang + epsilon_), 0);
      p += center;
      dp += center;
      pts_.push_back(p);
      d_pts_.push_back(dp);
     
      if( step_ >= min_seg_len_ )
      {
        if(i)
        {
          segs_.push_back(cv::Vec6f( prev_p.x, prev_p.y, prev_p.z, p.x, p.y, p.z ));
          d_segs_.push_back(p);
        }
        else
          first_p = p;
        
        prev_p = p;
      }
      ang += ang_step;
    }
    // In case, close the circle 
    if( step_ >= min_seg_len_  && ang_diff >= float(2*M_PI) - std::numeric_limits< float >::epsilon() )
    {
      segs_.push_back(cv::Vec6f( prev_p.x, prev_p.y, prev_p.z, first_p.x, first_p.y, first_p.z ));
      d_segs_.push_back(first_p);
    }
  }
}

inline void RasterObjectModel2D::addEllipseArc( cv::Point3f &center, cv::Point3f &major_axis_ep, 
                                                float minor_major_ratio, float start_ang, float end_ang )
{
  throw "RasterObjectModel2D::discretizeEllipseArc() Implement me!!";
  
// int ModelEntityEllipseArc::ComputeRaster(vector<Vector3>* raster, int startindex, double scale)
// {
//         int steps = (double)GetLength() * scale;
//         double rads = (d_AngleEnd - d_AngleStart);
//         double step = rads / steps;
// 
//         double a = GetRadiusX();
//         double b = GetRadiusY();
// 
//         for (double angle = d_AngleStart; angle <= d_AngleEnd; angle += step)
//         {
//                 double x = cos(angle);
//                 double y = sin(angle);
//                 raster->push_back((o_XAxis - o_Center) * x + (o_YAxis - o_Center) * y + o_Center);
//                 startindex++;
//         }
// 
//         return startindex;
// }

}

const std::vector< cv::Point3f >& RasterObjectModel2D::getPoints( bool only_visible_points ) const
{
  return pts_;
}

const std::vector< cv::Point3f >& RasterObjectModel2D::getDPoints( bool only_visible_points ) const
{
  return d_pts_;
}

const std::vector< cv::Vec6f >& RasterObjectModel2D::getSegments( bool only_visible_segments ) const
{
  return segs_;
}

const std::vector< cv::Point3f >& RasterObjectModel2D::getDSegments( bool only_visible_segments ) const
{
  return d_segs_;
}
