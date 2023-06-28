#pragma once

#include "cv_ext/cv_ext.h"
#include "raster_object_model3D.h"

#include <memory>

class TMObjectLocalization
{
 public:

  TMObjectLocalization();
  ~TMObjectLocalization();

  void setScaleFactor( double s);
  void setCannyLowThreshold( int th );
  void setScoreThreshold( double th );
  void setUnitOfMeasure( RasterObjectModel3D::UoM unit );
  void setModelSaplingStep( double step );
  void setRegionOfInterest( const cv::Rect &roi );
  void setBoundingBoxOffset( double xoff, double yoff, double zoff,
                             double woff, double hoff, double doff );
  void setNumMatches( int n );
  void enableDisplay( bool enabled );

  bool initialize( std::string cam_model_fn,
                   std::string cad_model_fn,
                   std::string templates_fn );

  RasterObjectModel3DPtr objectModel();

  bool localize( cv::Mat src_img );
  bool refine( const cv::Mat &src_img, cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec );

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};