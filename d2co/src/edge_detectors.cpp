#include "edge_detectors.h"

extern "C"
{
#include "lsd.h"
}

using namespace cv;
using namespace std;

void LSDEdgeDetector::setImage( const cv::Mat &src_img )
{
  Mat gl_img = checkImage( src_img );
  map_size_ = gl_img.size();

  segments_.clear();
  dir_segments_.clear();

  if( pyr_num_levels_ > 1)
    extractEdgesPyr( gl_img, segments_ );
  else
    extractEdges( gl_img, segments_ );

  if( num_directions_ > 1 )
  {
    dir_segments_.resize(num_directions_);
    vector< float > normals;
    computeEdgesNormalsDirections( segments_, normals );

    double eta_dir = double(num_directions_)/M_PI;

    for( int i = 0; i < int(segments_.size()); i++ )
    {
      float direction = normals[i];

      direction += M_PI/2;
      if( direction >= M_PI/2 )
        direction -= M_PI;
      direction += M_PI/2;

      int i_dir = cv_ext::roundPositive(eta_dir*direction);
      i_dir %= num_directions_;
      dir_segments_[i_dir].push_back( segments_[i] );
    }
  }
}

void LSDEdgeDetector::getEdgeMap( cv::Mat &edge_map )
{
  edge_map = Mat( map_size_, DataType<uchar>::type, (white_bg_?Scalar(255):Scalar(0)));
  cv_ext::drawSegments( edge_map, segments_, (white_bg_?Scalar(0):Scalar(255)), 1);
  if( !mask_.empty() )
  {
    if(white_bg_)
      edge_map = edge_map|(~mask_);
    else
      edge_map = edge_map&mask_;
  }
}

void LSDEdgeDetector::getEdgeDirectionsMap( Mat& edge_dir_map )
{
  std::cerr<<"WARNING: NOT tested LSDEdgeDetector::getEdgeDirectionsMap() mehod!"<<endl;
  edge_dir_map = Mat_<ushort>( map_size_, 0 );
  
  if( num_directions_ > 1 )
  {
    for( int i = 0; i < num_directions_; i++ )
      cv_ext::drawSegments( edge_dir_map, dir_segments_[i], Scalar(i), 1);
    if ( !mask_.empty() )
      edge_dir_map.setTo(Scalar(0), ~mask_);
  }
}

void LSDEdgeDetector::getDirectionalEdgeMap( int i_dir, cv::Mat &edge_map )
{

  if( i_dir >= num_directions_ )
    throw invalid_argument("Out of range direction index");

  if( num_directions_ > 1 )
  {
    edge_map = Mat( map_size_, DataType<uchar>::type, (white_bg_?Scalar(255):Scalar(0)));
    cv_ext::drawSegments( edge_map, dir_segments_[i_dir], (white_bg_?Scalar(0):Scalar(255)), 1);
    if( !mask_.empty() )
    {
      if( white_bg_)
        edge_map = edge_map|(~mask_);
      else
        edge_map = edge_map&mask_;
    }
  }
  else
    getEdgeMap(edge_map);
}

Mat LSDEdgeDetector::checkImage( const Mat &src_img )
{
  if( src_img.depth() != DataType<uchar>::depth || !src_img.rows || !src_img.cols ||
      ( src_img.channels() != 1 && src_img.channels() != 3 ) )
    throw invalid_argument("Invalid input image");

  Mat gl_img;
  if( src_img.channels() == 3 )
    cvtColor(src_img, gl_img, COLOR_BGR2GRAY);
  else
    gl_img = src_img;
  
  return gl_img;
}

void LSDEdgeDetector::extractEdgesPyr( const Mat &src_img, vector< Vec4f > &segments )
{
  cv_ext::ImagePyramid img_pyr( src_img, pyr_num_levels_ );
  
  for ( int i = 0; i < img_pyr.numLevels(); i++ )
  { 
    const Mat scaled_img = img_pyr.at(i);
    extractEdges( scaled_img, segments, img_pyr.getScale(i) );
  }
}

void LSDEdgeDetector::extractEdges( const Mat &src_img, vector< Vec4f > &segments, float scale )
{
  double *dbl_img = new double[src_img.cols*src_img.rows];

  for ( int y = 0; y < src_img.rows; y++ )
    for ( int x=0; x < src_img.cols; x++ )
      dbl_img[ x + y * src_img.cols ] = ( double ) src_img.at<uchar> ( y,x );

  int n_segments;

  double *dbl_segments = LineSegmentDetection ( &n_segments, dbl_img, src_img.cols,src_img.rows,
                                                scale_,//1.0 /*scale*/,
                                                sigma_scale_,//1.0 /*sigma_scale*/,
                                                quant_threshold_,//0.95 /* quant*/, //original 0.5
                                                22.5,//22.5 /* ang_th */,
                                                0.0 /*log_eps */,
                                                0.5 /* density_th */,
                                                1024,//1024 /* n_bins */,
                                                NULL, NULL, NULL );

  delete [] dbl_img;

  segments.reserve(segments.size() + n_segments);
  if( scale > 1.0 )
  {
    for ( int i = 0; i < n_segments; i++ )
    {
      Vec4f s;
      s[0] = dbl_segments[7*i]*scale;
      s[1] = dbl_segments[7*i + 1]*scale;
      s[2] = dbl_segments[7*i + 2]*scale;
      s[3] = dbl_segments[7*i + 3]*scale;
      segments.push_back(s);
    }
  }
  else
  {
    for ( int i = 0; i < n_segments; i++ )
    {
      Vec4f s;
      s[0] = dbl_segments[7*i];
      s[1] = dbl_segments[7*i + 1];
      s[2] = dbl_segments[7*i + 2];
      s[3] = dbl_segments[7*i + 3];
      segments.push_back(s);
    }      
  }
  delete dbl_segments;
}
  
void LSDEdgeDetector::computeEdgesNormalsDirections( const vector< Vec4f > &segments, 
                                                     vector<float> &normals )
{
  double m;
  int seg_size = segments.size();
  normals.resize(seg_size);

  for(int i = 0; i < seg_size; i++)
  {
    
    const float &x1 = segments[i][0], &y1 = segments[i][1],
                &x2 = segments[i][2], &y2 = segments[i][3];

    // Normal direction
    double cos_ang = x2-x1, sin_ang = y2-y1;
    if( sin_ang )
      m = -atan(cos_ang/sin_ang);
    else
      m = -M_PI/2;

    normals[i] = m;
  }
}

void CannyEdgeDetector::setImage(const Mat& src_img)
{
  Mat img = checkImage( src_img );
  map_size_ = img.size();

  dir_points_.clear();

  int kernel_size = -1;
  
  if( img.channels() == 3 && use_rgb_ )
  {
    vector<Mat> img_channels, c_edge_maps(3);
    img_channels.reserve(3);
    cv::split(img, img_channels);
    for( int i = 0; i < 3; i ++)
    {
      cv::GaussianBlur(img_channels[i], img_channels[i], cv::Size(0, 0), 1);
      cv::Canny( img_channels[i], c_edge_maps[i], low_threshold_, low_threshold_*ratio_, kernel_size, true );
    }
    edge_map_ = c_edge_maps[0] | c_edge_maps[1] | c_edge_maps[2];
  }
  else
  {
    cv::GaussianBlur(img, img, cv::Size(0, 0), 1);
    cv::Canny( img, edge_map_, low_threshold_, low_threshold_*ratio_, kernel_size, true );
  }
  
  if( !mask_.empty() )
    edge_map_ = edge_map_&mask_;

  if( num_directions_ > 1 )
  {
    vector< Point > points;
    vector< float > normals;
    computePointsNormalsDirections(points, normals);

    dir_points_.resize(num_directions_);
    double eta_dir = double(num_directions_)/M_PI;

    for( int i = 0; i < int(points.size()); i++ )
    {
      float direction = normals[i];

      direction += M_PI/2;
      if( direction >= M_PI/2 )
        direction -= M_PI;
      direction += M_PI/2;

      int i_dir = cv_ext::roundPositive(eta_dir*direction);
      i_dir %= num_directions_;
      dir_points_[i_dir].push_back( points[i] );
    }
  }
}

void CannyEdgeDetector::getEdgeMap(Mat& edge_map)
{
  if(white_bg_)
    edge_map = uchar(255) - edge_map_;
  else
    edge_map = edge_map_.clone();
}

void CannyEdgeDetector::getEdgeDirectionsMap( Mat& edge_dir_map )
{
  edge_dir_map = Mat_<ushort>( map_size_, 0 );
  
  if( num_directions_ > 1 )
  {
    for( int i = 0; i < num_directions_; i++ )
    {
      vector<cv::Point> &points = dir_points_[i];
      for(int j = 0; j < int(points.size()); j++)
        edge_dir_map.at<ushort>(points[j].y,points[j].x) = i;     
    }
  }
}

void CannyEdgeDetector::getDirectionalEdgeMap(int i_dir, Mat& edge_map)
{
  if( i_dir >= num_directions_ )
    throw invalid_argument("Out of range direction index");

  if( num_directions_ > 1 )
  {
    edge_map = Mat( edge_map_.size(), DataType<uchar>::type, (white_bg_?Scalar(255):Scalar(0)));
    vector<cv::Point> &points = dir_points_[i_dir];
    uchar color = (white_bg_?0:255);
    for(int i = 0; i< int(points.size()); i++)
      edge_map.at<uchar>(points[i].y,points[i].x) = color;
  }
  else
    getEdgeMap(edge_map);
}

Mat CannyEdgeDetector::checkImage(const Mat& src_img)
{
  if( src_img.depth() != DataType<uchar>::depth || !src_img.rows || !src_img.cols ||
      ( src_img.channels() != 1 && src_img.channels() != 3 ) )
    throw invalid_argument("Invalid input image");

  Mat dst_img;
  if( src_img.channels() == 3 && !use_rgb_ )
    cvtColor(src_img, dst_img, COLOR_BGR2GRAY);
  else
    dst_img = src_img;

  return dst_img;
}

void CannyEdgeDetector::computePointsNormalsDirections( vector< Point > &points, vector< float >& normals)
{
  Mat eigen_mat;
  cornerEigenValsAndVecs(edge_map_, eigen_mat, 5, 3 );

  points.reserve(edge_map_.total());
  normals.reserve(edge_map_.total());

  vector<Point> zero_points;
  int w = edge_map_.cols, h = edge_map_.rows;
  for( int y = 0; y < h; y++)
  {
    const uchar *img_p = edge_map_.ptr<uchar>(y);
    const Vec6f *eigen_p = eigen_mat.ptr<Vec6f>(y);

    for( int x = 0; x < w; x++, img_p++, eigen_p++)
    {
      if(*img_p)
      {
        points.push_back(Point(x,y));
        // The first eigenvalue is always greater than or equal to the second one
        if( (*eigen_p)[2] != 0.0f )
          normals.push_back(atan((*eigen_p)[3]/(*eigen_p)[2]));
        else
          normals.push_back(M_PI/2);
      }
    }
  }
}