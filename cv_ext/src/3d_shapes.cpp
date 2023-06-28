#include <cmath>
#include <iostream>
  #include <stdexcept>
#include <boost/concept_check.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Geometry> 


#include "cv_ext/base.h"
#include "cv_ext/types.h"
#include "cv_ext/3d_shapes.h"

static inline Eigen::Matrix3d computeXRot( double theta_x )
{
  Eigen::Matrix3d rot;
  rot << 1.0,                   0.0,                   0.0,
         0.0,                   cos ( theta_x ),       -sin ( theta_x ),
         0.0,                   sin ( theta_x ),      cos ( theta_x );
  return rot;
}

static inline Eigen::Matrix3d computeYRot( double theta_y )
{
  Eigen::Matrix3d rot;
  rot << cos ( theta_y ),       0.0,                   sin ( theta_y ),
         0.0,                   1.0,                   0.0,
         -sin ( theta_y ),       0.0,                   cos ( theta_y );
  return rot;
}

static inline Eigen::Matrix3d computeZRot( double theta_z )
{
  Eigen::Matrix3d rot;
  rot << cos ( theta_z ),       -sin ( theta_z ),       0.0,
         sin ( theta_z ),      cos ( theta_z ),       0.0,
         0.0,                   0.0,                   1.0;
  return rot;
}

template < typename _TPoint3D > 
  static void sampleRotations( const _TPoint3D &vec, cv_ext::vector_Quaterniond &quat_rotations, 
                               double from_ang, double to_ang, double ang_step, 
                               cv_ext::CoordinateAxis rot_axis )
{  
  Eigen::Vector3d n_vec( double(vec.x), double(vec.y), double(vec.z) ), axis_vec, r_vec;
  // Normalize the input vector to unit vector
  n_vec /= n_vec.norm();
  Eigen::Matrix3d init_rot, final_rot, tot_rot;
  
  switch( rot_axis )
  {
    case cv_ext::COORDINATE_X_AXIS:
      axis_vec = Eigen::Vector3d( 1, 0, 0 );
      break;
      
    case cv_ext::COORDINATE_Y_AXIS:
      axis_vec = Eigen::Vector3d( 0, 1, 0 );
      break;
      
    default:
    case cv_ext::COORDINATE_Z_AXIS:
      axis_vec = Eigen::Vector3d( 0, 0, 1 );
      break;
  }

  // Compute the rotation to align the selected axis (axis_vec) with the input vector vec
  // Case 1: Input vector has same direction of axis_vec
  if( (n_vec - axis_vec).norm() <= std::numeric_limits< double>::epsilon() )
    init_rot = Eigen::Matrix3d::Identity();
  // Case 2: Input vector has opposite direction of axis_vec 
  else if( (n_vec + axis_vec).norm() <= std::numeric_limits< double>::epsilon() )
  {
    switch( rot_axis )
    {
      case cv_ext::COORDINATE_X_AXIS:
        init_rot << -1,  0,  0, 
                     0, -1,  0, 
                     0,  0,  1;
        break;
        
      case cv_ext::COORDINATE_Y_AXIS:
        init_rot <<  1,  0,  0, 
                     0, -1,  0, 
                     0,  0, -1;
        break;
        
      default:
      case cv_ext::COORDINATE_Z_AXIS:
        init_rot << -1,  0,  0, 
                     0,  1,  0, 
                     0,  0, -1;
        break;
    }
  }
  // Case 3: general case
  else
  {
    // The cross product is defined by the formula v1 x v2 = ||v1|| ||v2|| sin( theta )
    r_vec = axis_vec.cross(n_vec);
    double theta_0 = std::asin(r_vec.norm());
    r_vec /= r_vec.norm();
    Eigen::Matrix3d rot_0, rot_1;
    rot_0 = Eigen::AngleAxisd(theta_0, r_vec);
    rot_1 = Eigen::AngleAxisd(M_PI - theta_0, r_vec);
    if( (rot_0*axis_vec - n_vec).norm()< (rot_1*axis_vec - n_vec).norm() )
      init_rot = rot_0;
    else
      init_rot = rot_1;
  }
  
  from_ang = cv_ext::normalizeAngle(from_ang);
  to_ang = cv_ext::normalizeAngle(to_ang);
  
  if( from_ang > to_ang ) 
    from_ang -= 2*M_PI;
  
  int num_step = 0;
  if( from_ang != to_ang && ang_step != 0 )
  {
    double ang_diff = to_ang - from_ang;
    num_step = cvRound(ang_diff/ang_step);
    if( num_step < 1 ) num_step = 1;
    ang_step = ang_diff/double(num_step);
  }
  
  double theta = from_ang;
  // Rotate around the selected axes
  for( int i = 0; i <= num_step; i++, theta+= ang_step )
  {
    final_rot =  Eigen::AngleAxisd(theta, n_vec);
    tot_rot = final_rot*init_rot;
    quat_rotations.push_back( Eigen::Quaterniond(tot_rot) );
  }
}

template < typename _TPoint3D > 
  static void sampleRotations( const _TPoint3D &vec, cv_ext::vector_Quaterniond &quat_rotations, 
                               int num_rotations, cv_ext::CoordinateAxis rot_axis )
{
  if( num_rotations < 1 )
    throw std::invalid_argument("num_rotations should be >= 1");
  
  if( num_rotations == 1 )
    sampleRotations( vec, quat_rotations, 0, 0, 0, rot_axis );
  else
  {
    double from_ang = 0, ang_step = 2*M_PI/num_rotations, to_ang = 2*M_PI - ang_step;
    sampleRotations( vec, quat_rotations, from_ang, to_ang, ang_step, rot_axis );
  }
}

template < typename _TPoint3D > 
  void cv_ext::rotationAroundVector( const _TPoint3D& vec, Eigen::Quaterniond& quat_rotation, double ang, 
                                     cv_ext::CoordinateAxis rot_axis )
{
  cv_ext::vector_Quaterniond quat_rotations;
  sampleRotations( vec, quat_rotations, ang, ang, 0, rot_axis );
  quat_rotation = quat_rotations[0];
}
  

template < typename _TPoint3D > 
  void cv_ext::sampleRotationsAroundVector( const _TPoint3D& vec, cv_ext::vector_Quaterniond& quat_rotations, 
                                            int num_rotations, cv_ext::CoordinateAxis rot_axis )
{
  quat_rotations.clear();
  sampleRotations( vec, quat_rotations, num_rotations, rot_axis );
}

template < typename _TPoint3D > 
  void cv_ext::sampleRotationsAroundVectors( const std::vector< _TPoint3D> &vecs, cv_ext::vector_Quaterniond &quat_rotations, 
                                             int num_rotations, CoordinateAxis rot_axis )
{
  quat_rotations.clear();
  for( int i = 0; i < int(vecs.size()); i++ )
    sampleRotations( vecs[i], quat_rotations, num_rotations, rot_axis );
}

template < typename _TPoint3D > 
  void cv_ext::sampleRotationsAroundVector( const _TPoint3D &vec, cv_ext::vector_Quaterniond &quat_rotations, 
                                            double ang0, double ang1, double ang_step,
                                            cv_ext::CoordinateAxis rot_axis )
{
  quat_rotations.clear();
  sampleRotations( vec, quat_rotations, ang0, ang1, ang_step, rot_axis );
}

    
template < typename _TPoint3D > 
  void cv_ext::sampleRotationsAroundVectors( const std::vector< _TPoint3D >& vecs, cv_ext::vector_Quaterniond& quat_rotations, 
                                             double ang0, double ang1, double ang_step, cv_ext::CoordinateAxis rot_axis )
{
  quat_rotations.clear();
  for( int i = 0; i < int(vecs.size()); i++ )
    sampleRotations( vecs[i], quat_rotations, ang0, ang1, ang_step, rot_axis );
}

// Icosphere implementaion inspired by: 
// http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html

static const int ICOSPHERE_NUM_VERTEXS = 12;
static const int ICOSPHERE_NUM_FACES = 20;

/* icosphereAngStep() provides the actual angle step of an icosphere obtained by applying 
 * n_iter iterations.
*/ 
double icosphereAngStep( const int n_iter )
{
  
  double ang_step = 2.0*M_PI/5.0;
  for( int iter = 1; iter <= n_iter; iter++ )
    ang_step /= 2.0;
  return ang_step;
}

/* icosphereNumIterations() provides the number of iterations (output parameter n_iter) required
 * to obtain an icosphere with angle step between points no greater than max_ang_step. 
 * This function provide also the actual angle step of the resulting icosphere (output parameter ang_step).
*/ 
void icosphereNumIterations( const double max_ang_step, int &n_iter, double &ang_step )
{
  n_iter = 0;
  ang_step = 2.0*M_PI/5.0;
  while( ang_step > max_ang_step )
  {
    n_iter++;
    ang_step /= 2.0;
  }
}

struct IcosphereTriangle
{ 
  IcosphereTriangle(){};
  IcosphereTriangle( int v0_idx, int v1_idx, int v2_idx) :
    v0_idx(v0_idx), v1_idx(v1_idx), v2_idx(v2_idx){};
  int v0_idx, v1_idx, v2_idx;   
};

template < typename _TPoint3D > 
  static inline IcosphereTriangle computeInscribedTriangle( const IcosphereTriangle& triangle, 
                                                            std::vector< _TPoint3D > &sphere_points,
                                                            cv::Mat &idx_mask )
{
  IcosphereTriangle inscribed;
  _TPoint3D new_pt;
  
  // Checks if the point has already been inserted
  if( idx_mask.at<int32_t>( triangle.v0_idx, triangle.v1_idx ) < 0 && 
      idx_mask.at<int32_t>( triangle.v1_idx, triangle.v0_idx ) < 0 )
  {
    // New point: compute the point ...
    new_pt.x = 0.5* (sphere_points[triangle.v0_idx].x + sphere_points[triangle.v1_idx].x );
    new_pt.y = 0.5* (sphere_points[triangle.v0_idx].y + sphere_points[triangle.v1_idx].y );
    new_pt.z = 0.5* (sphere_points[triangle.v0_idx].z + sphere_points[triangle.v1_idx].z );
    new_pt = cv_ext::normalize3DPoint(new_pt);
    // ... add the index to the triangle...
    inscribed.v0_idx = sphere_points.size();
    // ... add the point to the point vector ...
    sphere_points.push_back( new_pt );
    // ... and update the mask
    idx_mask.at<int32_t>( triangle.v0_idx, triangle.v1_idx ) = inscribed.v0_idx;
    idx_mask.at<int32_t>( triangle.v1_idx, triangle.v0_idx ) = inscribed.v0_idx;
  }
  else
  {
    // Otherwise, just retrive and store the index
    inscribed.v0_idx = idx_mask.at<int32_t>( triangle.v0_idx, triangle.v1_idx );
  }
  
  // Checks if the point has already been inserted
  if( idx_mask.at<int32_t>( triangle.v1_idx, triangle.v2_idx ) < 0 && 
      idx_mask.at<int32_t>( triangle.v2_idx, triangle.v1_idx ) < 0 )
  {
    // New point: compute the point ...
    new_pt.x = 0.5*( sphere_points[triangle.v1_idx].x + sphere_points[triangle.v2_idx].x );
    new_pt.y = 0.5*( sphere_points[triangle.v1_idx].y + sphere_points[triangle.v2_idx].y );
    new_pt.z = 0.5*( sphere_points[triangle.v1_idx].z + sphere_points[triangle.v2_idx].z );
    new_pt = cv_ext::normalize3DPoint(new_pt);
    // ... add the index to the triangle...
    inscribed.v1_idx = sphere_points.size();
    // ... add the point to the point vector ...
    sphere_points.push_back( new_pt );
    // ... and update the mask
    idx_mask.at<int32_t>( triangle.v1_idx, triangle.v2_idx ) = inscribed.v1_idx;
    idx_mask.at<int32_t>( triangle.v2_idx, triangle.v1_idx ) = inscribed.v1_idx;
  }
  else
  {
    // Otherwise, just retrive and store the index
    inscribed.v1_idx = idx_mask.at<int32_t>( triangle.v1_idx, triangle.v2_idx );
  }
  
  // Checks if the point has already been inserted
  if( idx_mask.at<int32_t>( triangle.v2_idx, triangle.v0_idx ) < 0 && 
      idx_mask.at<int32_t>( triangle.v0_idx, triangle.v2_idx ) < 0 )
  {
    // New point: compute the point ...
    new_pt.x = 0.5*( sphere_points[triangle.v2_idx].x + sphere_points[triangle.v0_idx].x );
    new_pt.y = 0.5*( sphere_points[triangle.v2_idx].y + sphere_points[triangle.v0_idx].y );
    new_pt.z = 0.5*( sphere_points[triangle.v2_idx].z + sphere_points[triangle.v0_idx].z );
    new_pt = cv_ext::normalize3DPoint(new_pt);
    // ... add the index to the triangle...
    inscribed.v2_idx = sphere_points.size();
    // ... add the point to the point vector ...
    sphere_points.push_back( new_pt );
    // ... and update the mask
    idx_mask.at<int32_t>( triangle.v2_idx, triangle.v0_idx ) = inscribed.v2_idx;
    idx_mask.at<int32_t>( triangle.v0_idx, triangle.v2_idx ) = inscribed.v2_idx;
  }
  else
  {
    // Otherwise, just retrive and store the index
    inscribed.v2_idx = idx_mask.at<int32_t>( triangle.v2_idx, triangle.v0_idx );
  }
  
  return inscribed;
}

template < typename _TPoint3D > 
  static void icosphereRefinement( const std::vector< IcosphereTriangle > &src_triangles, 
                                   std::vector< IcosphereTriangle > &dst_triangles, 
                                   std::vector< _TPoint3D > &sphere_points )
{
  int src_triangles_size = src_triangles.size(),
      dst_triangles_size = 4*src_triangles_size,
      pts_size = sphere_points.size();
  
  sphere_points.reserve( sphere_points.size() + 3*src_triangles_size/2 );
  
  dst_triangles.clear();
  dst_triangles.reserve ( dst_triangles_size );
  
  cv::Mat idx_mask(pts_size, pts_size, cv::DataType< int32_t >::type, cv::Scalar(-1) );
  
  for ( int i=0; i < src_triangles_size; i++ )
  {
    const IcosphereTriangle &src_triangle = src_triangles[i];
    IcosphereTriangle inscribed = computeInscribedTriangle< _TPoint3D > ( src_triangle, sphere_points, idx_mask );
    
    // Split the old triangle in 4 smaller triangles, starting from the inscribed one
    dst_triangles.push_back( inscribed );
    dst_triangles.push_back( IcosphereTriangle( inscribed.v0_idx, inscribed.v1_idx, src_triangle.v1_idx) );
    dst_triangles.push_back( IcosphereTriangle( src_triangle.v2_idx, inscribed.v1_idx, inscribed.v2_idx ) );
    dst_triangles.push_back( IcosphereTriangle( src_triangle.v0_idx, inscribed.v0_idx, inscribed.v2_idx ) );
  }
}


template < typename _TPoint3D > 
  void cv_ext::createIcosphere( std::vector<_TPoint3D> &sphere_points, int n_iter, 
                                std::vector<int> &iter_num_pts, std::vector<double> &iter_ang_steps,
                                double radius )
{
  iter_num_pts.clear();
  iter_ang_steps.clear();
  
  double tau = ( 1.0 + std::sqrt ( 5.0 ) ) /2.0;

  // Inizialization vertexs of icosahedron
  sphere_points.resize ( ICOSPHERE_NUM_VERTEXS );

  sphere_points[0].x = -1.0;  sphere_points[0].y = tau;  sphere_points[0].z = 0.0;
  sphere_points[1].x = 1.0;   sphere_points[1].y = tau;  sphere_points[1].z = 0.0;
  sphere_points[2].x = -1.0;  sphere_points[2].y = -tau; sphere_points[2].z = 0.0;
  sphere_points[3].x = 1.0;   sphere_points[3].y = -tau; sphere_points[3].z = 0.0;

  sphere_points[4].x = 0.0;   sphere_points[4].y = -1.0; sphere_points[4].z = tau;
  sphere_points[5].x = 0.0;   sphere_points[5].y = 1.0;  sphere_points[5].z = tau;
  sphere_points[6].x = 0.0;   sphere_points[6].y = -1.0; sphere_points[6].z = -tau;
  sphere_points[7].x = 0.0;   sphere_points[7].y = 1.0;  sphere_points[7].z = -tau;

  sphere_points[8].x = tau;   sphere_points[8].y = 0.0;  sphere_points[8].z = -1.0;
  sphere_points[9].x = tau;   sphere_points[9].y = 0.0;  sphere_points[9].z = 1.0;
  sphere_points[10].x = -tau; sphere_points[10].y = 0.0; sphere_points[10].z = -1.0;
  sphere_points[11].x = -tau; sphere_points[11].y = 0.0; sphere_points[11].z = 1.0;

  for ( int i=0; i < ICOSPHERE_NUM_VERTEXS; i++ )
    sphere_points[i] = cv_ext::normalize3DPoint(sphere_points[i]);
  
  std::vector< IcosphereTriangle > triangles_buf[2];
  std::vector< IcosphereTriangle > &triangles = triangles_buf[0];
  
  triangles.resize ( ICOSPHERE_NUM_FACES );

  // 5 faces around vertex 0
  triangles[0] = IcosphereTriangle( 0, 11, 5 );
  triangles[1] = IcosphereTriangle( 0, 5, 1 );
  triangles[2] = IcosphereTriangle( 0, 1, 7 );
  triangles[3] = IcosphereTriangle( 0, 7, 10 );
  triangles[4] = IcosphereTriangle( 0, 10, 11 );

  // 5 adjacent faces
  triangles[5] = IcosphereTriangle( 1, 5, 9 );
  triangles[6] = IcosphereTriangle( 5, 11, 4 );
  triangles[7] = IcosphereTriangle( 11, 10, 2 );
  triangles[8] = IcosphereTriangle( 10, 7, 6 );
  triangles[9] = IcosphereTriangle( 7, 1, 8 );

  //5 faces around point 3
  triangles[10] = IcosphereTriangle( 3, 9, 4 );
  triangles[11] = IcosphereTriangle( 3, 4, 2 );
  triangles[12] = IcosphereTriangle( 3, 2, 6 );
  triangles[13] = IcosphereTriangle( 3, 6, 8 );
  triangles[14] = IcosphereTriangle( 3, 8, 9 );

  //5 adjacent faces
  triangles[15] = IcosphereTriangle( 4, 9, 5 );
  triangles[16] = IcosphereTriangle( 2, 4, 11 );
  triangles[17] = IcosphereTriangle( 6, 2, 10 );
  triangles[18] = IcosphereTriangle( 8, 6, 7 );
  triangles[19] = IcosphereTriangle( 9, 8, 1 );

  double ang_step = 2.0*M_PI/5.0;
  iter_num_pts.push_back(ICOSPHERE_NUM_VERTEXS);
  iter_ang_steps.push_back(ang_step);
  
  for ( int ref_i=1; ref_i <= n_iter; ref_i++ )
  {
    std::vector< IcosphereTriangle > &src_triangles = triangles_buf[!(ref_i%2)],
                                     &dst_triangles = triangles_buf[ref_i%2];
   
    int prev_points_size = sphere_points.size();
    icosphereRefinement< _TPoint3D >( src_triangles, dst_triangles, sphere_points );
    
    iter_num_pts.push_back(sphere_points.size() - prev_points_size);
    iter_ang_steps.push_back(ang_step /= 2);    
  }
  
  if( radius != 1.0)
  {
    for( int i = 0; i < int(sphere_points.size()); i++ )
    {
      sphere_points[i].x *= radius;
      sphere_points[i].y *= radius;
      sphere_points[i].z *= radius;
    }
  }  
}

template < typename _TPoint3D > 
  void cv_ext::createIcosphere( std::vector<_TPoint3D> &sphere_points, int n_iter, double radius )
{
  std::vector<int> iter_num_pts;
  std::vector<double> iter_ang_steps;
  
  createIcosphere( sphere_points, n_iter, iter_num_pts, iter_ang_steps, radius );
}

template < typename _TPoint3D > 
  double cv_ext::createIcosphereFromAngle( std::vector< _TPoint3D > &sphere_points, 
                                  double max_ang_step, double radius )
{
  if( max_ang_step <= 0 || radius <= 0 )
    throw std::invalid_argument("max_ang_step and radius should be > 0");
  
  int n_iter;
  double ang_step;
  icosphereNumIterations( max_ang_step, n_iter, ang_step );
  createIcosphere( sphere_points, n_iter, radius );

  return ang_step;
}

template < typename _TPoint3D > 
  double cv_ext::createIcospherePolarCapFromAngle( std::vector<_TPoint3D> &cap_points, double max_ang_step, 
                                        double latitude_angle, double radius, bool only_north_cap )
{
  std::vector<_TPoint3D> sphere_points;
  double ang_step = createIcosphereFromAngle( sphere_points, max_ang_step, radius );
  if( latitude_angle < 0 )
    latitude_angle = 0;
  else if( latitude_angle >= M_PI/2 )
    latitude_angle = M_PI/2;
  
  int pts_size = sphere_points.size();

  cap_points.clear();
  cap_points.reserve( pts_size );
  
  for( int i = 0; i < pts_size; i++)
  {
    double dist_z = std::sqrt(sphere_points[i].x*sphere_points[i].x + sphere_points[i].y*sphere_points[i].y);
    if( atan(sphere_points[i].z/dist_z) >= latitude_angle )
      cap_points.push_back(sphere_points[i]);
    if( !only_north_cap && atan(sphere_points[i].z/dist_z) <= -latitude_angle )
      cap_points.push_back(sphere_points[i]);
  }
  return ang_step;
}

  
#define CV_EXT_INSTANTIATE_rotationAroundVector(_T) \
template void cv_ext::rotationAroundVector ( const _T &vec, Eigen::Quaterniond &quat_rotation, \
                                             double ang, cv_ext::CoordinateAxis rot_axis );
#define CV_EXT_INSTANTIATE_sampleRotationsAroundVector(_T) \
template void cv_ext::sampleRotationsAroundVector( const _T &vec, cv_ext::vector_Quaterniond &quat_rotations, \
                                                   int num_rotations, cv_ext::CoordinateAxis rot_axis ); \
template void cv_ext::sampleRotationsAroundVector( const _T &vec, cv_ext::vector_Quaterniond &quat_rotations, \
                                                   double ang0, double ang1, double ang_step, \
                                                   cv_ext::CoordinateAxis rot_axis );
#define CV_EXT_INSTANTIATE_sampleRotationsAroundVectors(_T) \
template void cv_ext::sampleRotationsAroundVectors( const std::vector< _T> &vecs, \
                                                    cv_ext::vector_Quaterniond &quat_rotations, \
                                                    int num_rotations, cv_ext::CoordinateAxis rot_axis ); \
template void cv_ext::sampleRotationsAroundVectors( const std::vector< _T> &vecs, \
                                                    cv_ext::vector_Quaterniond &quat_rotations, \
                                                    double ang0, double ang1, double ang_step, \
                                                    cv_ext::CoordinateAxis rot_axis );
#define CV_EXT_INSTANTIATE_createIcosphere(_T) \
template void cv_ext::createIcosphere( std::vector<_T> &sphere_points, int n_iter, double radius ); \
template void cv_ext::createIcosphere( std::vector<_T> &sphere_points, int n_iter, \
                                       std::vector<int> &iter_num_pts, std::vector<double> &iter_ang_steps, \
                                       double radius );
#define CV_EXT_INSTANTIATE_createIcosphereFromAngle(_T) \
template double cv_ext::createIcosphereFromAngle( std::vector<_T> &sphere_points, double max_ang_step, double radius );
#define CV_EXT_INSTANTIATE_createIcospherePolarCapFromAngle(_T) \
template double cv_ext::createIcospherePolarCapFromAngle( std::vector<_T> &cap_points, double max_ang_step, \
                                                 double latitude_angle, double radius, bool only_north_cap );

CV_EXT_INSTANTIATE( rotationAroundVector, CV_EXT_3D_POINT_TYPES )
CV_EXT_INSTANTIATE( sampleRotationsAroundVector, CV_EXT_3D_POINT_TYPES )
CV_EXT_INSTANTIATE( sampleRotationsAroundVectors, CV_EXT_3D_POINT_TYPES )
CV_EXT_INSTANTIATE( createIcosphere, CV_EXT_3D_POINT_TYPES )
CV_EXT_INSTANTIATE( createIcosphereFromAngle, CV_EXT_3D_POINT_TYPES )
CV_EXT_INSTANTIATE( createIcospherePolarCapFromAngle, CV_EXT_3D_POINT_TYPES )
