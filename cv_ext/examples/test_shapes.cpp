#include <iostream>
#include "cv_ext/3d_shapes.h"
#include "cv_ext/debug_tools.h"

using namespace cv_ext;
using namespace Eigen;

int main(int argc, char** argv)
{  
  double max_ang_step = M_PI/12, ang_step;
  
  std::vector <cv::Point3f> points_icosphere, points_caps;

  ang_step = createIcosphereFromAngle( points_icosphere, max_ang_step );
  std::cout<<"Num sphere points : "<<points_icosphere.size()<<" angle step : required "<<max_ang_step<<" actual "<<ang_step<<std::endl;
  show3DPoints( points_icosphere );
  
  max_ang_step *= 2;
  
  ang_step = createIcospherePolarCapFromAngle( points_caps, max_ang_step, M_PI/4, 1.0, true );
  std::cout<<"Num cap points : "<<points_caps.size()<<" angle step : required "<<max_ang_step<<" actual "<<ang_step<<std::endl;
  show3DPoints( points_caps, "points_caps", true );
  
  
  vector_Quaterniond quat_rotations;
  sampleRotationsAroundVectors( points_caps, quat_rotations, 18, COORDINATE_Z_AXIS );
  vector_Isometry3d axes_transf;
  for( int i = 0; i < int(quat_rotations.size()); i++ )
  {
    Isometry3d transf;
    transf.fromPositionOrientationScale( Vector3d::Zero(), quat_rotations[i], Vector3d::Ones());
    axes_transf.push_back(transf);
  }
  show3DPoints( points_caps, "rotations", true, axes_transf );
      
  ang_step = createIcospherePolarCapFromAngle( points_caps, max_ang_step, M_PI/2, 1, true );
  std::cout<<"Num cap points : "<<points_caps.size()<<" angle step : required "<<max_ang_step<<" actual "<<ang_step<<std::endl;
  
  sampleRotationsAroundVectors( points_caps, quat_rotations, 6, COORDINATE_Z_AXIS );
  axes_transf.clear();
  for( int i = 0; i < int(quat_rotations.size()); i++ )
  {
    Isometry3d transf;
    transf.fromPositionOrientationScale( Vector3d::Zero(), quat_rotations[i], Vector3d::Ones());
    axes_transf.push_back(transf);
  }
  show3DPoints( points_caps, "rotations", true, axes_transf );
  
  ang_step = createIcospherePolarCapFromAngle( points_caps, max_ang_step, M_PI/6, 1, true );
  std::cout<<"Num cap points : "<<points_caps.size()<<" angle step : required "<<max_ang_step<<" actual "<<ang_step<<std::endl;
  
  sampleRotationsAroundVectors( points_caps, quat_rotations, -M_PI/6, M_PI/6, ang_step, COORDINATE_Z_AXIS );
  axes_transf.clear();
  for( int i = 0; i < int(quat_rotations.size()); i++ )
  {
    Isometry3d transf;
    transf.fromPositionOrientationScale( Vector3d::Zero(), quat_rotations[i], Vector3d::Ones());
    axes_transf.push_back(transf);
  }
  show3DPoints( points_caps, "rotations", true, axes_transf );
  
  return 0;
}
