#include <stdexcept>

#include "cv_ext/drawing.h"
#include "cv_ext/image_inspection.h"
#include "cv_ext/types.h"

using namespace cv;

template < typename _TPoint2 > void cv_ext::drawPoints(Mat &img, const std::vector< _TPoint2 > &pts,
                                                       Scalar color, double scale )
{
  drawCircles( img, pts, 0, color, scale );
}


template < typename _TPoint2, typename _T  > 
  void cv_ext::drawNormalsDirections( Mat &img, const std::vector< _TPoint2 > &pts,
                                      const std::vector< _T > &normals_dirs, Scalar color,
                                      int normal_length, double scale )
{
  float dx,dy;
  Point p1, p2;
  int pts_size = pts.size(), width = img.cols, height = img.rows;
  for(int i = 0; i< pts_size; i++)
  {
    const float &xc = pts[i].x, &yc = pts[i].y;
    
    dx = normal_length*cos(normals_dirs[i]);
    dy = normal_length*sin(normals_dirs[i]);
    
    p1.x = cvRound(scale*xc-dx/2);
    p1.y = cvRound(scale*yc-dy/2);
    p2.x = cvRound(scale*xc+dx/2);
    p2.y = cvRound(scale*yc+dy/2);

    if ( unsigned(p1.x) < unsigned(width) && unsigned(p1.y) < unsigned(height) &&
         unsigned(p2.x) < unsigned(width) && unsigned(p2.y) < unsigned(height) )
      line(img, p1, p2, color, 1, 8, 0);
  }
}

template <typename _TVec4> void cv_ext::drawSegments( Mat &img, const std::vector< _TVec4 > &segments,
                                                      Scalar color, int tickness, double scale )
{
  Point2i pt0, pt1;
  int seg_size = segments.size();
  scale = 1./scale;
  for(int i = 0; i < seg_size; i++)
  {
    const _TVec4 &s = segments[i];
    pt0 = Point(cvRound(scale*s[0]),cvRound(scale*s[1]));
    pt1 = Point(cvRound(scale*s[2]),cvRound(scale*s[3]));

    line(img, pt0, pt1, color, tickness);
  }
}

// template <typename _TVec4> void cv_ext::drawSegmentsWithNormalsDirections( Mat &img, Mat &norm_dir_img,
//                                                                            const std::vector< _TVec4 > &segments,
//                                                                            Scalar color, double scale )
// {
//   Point2i pt0, pt1;
//   int seg_size = segments.size();
//   scale = 1./scale;
//   std::vector<Point> line_pts;
//   for(int i = 0; i < seg_size; i++)
//   {
//     const _TVec4 &s = segments[i];
//     pt0 = Point(cvRound(scale*s[0]),cvRound(scale*s[1]));
//     pt1 = Point(cvRound(scale*s[2]),cvRound(scale*s[3]));
// 
//     getLinePoints( pt0, pt1, line_pts );
//     line(img, pt0, pt1, color, tickness);
//   }
// }

template < typename _TPoint2 > void cv_ext::drawCircles( Mat &img, const std::vector< _TPoint2 > &pts,
                                                         int radius, Scalar color, double scale )
{
  int pts_size = pts.size(), width = img.cols, height = img.rows;
  for(int i = 0; i< pts_size; i++)
  {
    int xc = cvRound(scale*pts[i].x), yc = cvRound(scale*pts[i].y);
    if ( unsigned(xc) < unsigned(width) && unsigned(yc) < unsigned(height) )
      circle( img, Point(xc,yc),radius, color,-1,8 );
  }
}

Mat cv_ext::drawDenseOptFlow( const Mat &flow, const Mat &img, int step,
                                  Scalar color, const Mat &mask )
{
  if( flow.depth() != DataType<float>::type || !flow.rows || !flow.cols || flow.channels() != 2 ||
      img.rows != flow.rows || img.cols != flow.cols || (img.channels() != 1 && img.channels() != 3) ||
      (!mask.empty() && ( mask.depth() != DataType<uchar>::type || mask.channels() != 1 ||
         mask.rows != flow.rows || mask.cols != flow.cols ) ) )
    throw std::invalid_argument("Unsopported  or incompatible images");
 
  Mat flow_img(flow.rows, flow.cols, DataType< Vec< uchar, 3 > >::type );

  Mat tmp_img = img;

  if( img.channels() == 1 )
  {
    if( img.type() != DataType<uchar>::type )
      img.convertTo(tmp_img, DataType< uchar >::type);
    cvtColor(tmp_img, flow_img, COLOR_GRAY2BGR );
  }
  else
  {
    if( img.type() != DataType< Vec3b >::type )
      img.convertTo(tmp_img, DataType< Vec3b >::type);
    tmp_img.copyTo(flow_img);
  }
  
  if( !mask.empty() )
  {
    for(int y = 0; y < flow_img.rows; y += step)
    {
      const uchar *mask_p = mask.ptr<uchar>(y);
      const Point2f *flow_p = flow.ptr<Point2f>(y);
      
      for(int x = 0; x < flow_img.cols; x += step, mask_p += step, flow_p += step ) 
      {
        if( *mask_p )
        {
          const Point2f &fxy = *flow_p;
          line(flow_img, Point(x,y),
          Point( cvRound(x + fxy.x), cvRound(y +fxy.y) ), color);
        }
      }
    }
  }
  else
  {
    for(int y = 0; y < flow_img.rows; y += step)
    {
      const Point2f *flow_p = flow.ptr<Point2f>(y);
      
      for(int x = 0; x < flow_img.cols; x += step, flow_p += step ) 
      {
        const Point2f &fxy = *flow_p;
        line(flow_img, Point(x,y),
                  Point( cvRound(x + fxy.x), cvRound(y +fxy.y) ), color);
      }
    }
  }
  
  return flow_img;
}

Mat cv_ext::drawDenseOptFlow( const Mat &flow, const Mat &img0, const Mat &img1,
                                  int step, Scalar color, bool side_by_side, const Mat &mask )
{
  if( flow.depth() != DataType<float>::type || !flow.rows || !flow.cols || flow.channels() != 2 ||
      img0.rows != flow.rows || img0.cols != flow.cols || (img0.channels() != 1 && img0.channels() != 3) ||
      img1.rows != flow.rows || img1.cols != flow.cols || (img1.channels() != 1 && img1.channels() != 3) ||
      (!mask.empty() && ( mask.depth() != DataType<uchar>::type || mask.channels() != 1 ||
         mask.rows != flow.rows || mask.cols != flow.cols ) ) )
    throw std::invalid_argument("Unsopported or incompatible images");

  Mat flow_img( (side_by_side?1:2)*img0.rows, (side_by_side?2:1)*img0.cols, DataType< Vec3b >::type );

  Mat tmp_img0 = img0, tmp_img1 = img1;

  if( img0.channels() == 1 )
  {
    if( img0.type() != DataType<uchar>::type )
      img0.convertTo(tmp_img0, DataType< uchar >::type);

    if( img1.type() != DataType<uchar>::type )
      img1.convertTo(tmp_img1, DataType< uchar >::type);

    if( side_by_side )
    {
      cvtColor(tmp_img0, flow_img.colRange(0,img0.cols), COLOR_GRAY2BGR );
      cvtColor(tmp_img1, flow_img.colRange(img0.cols,2*img0.cols), COLOR_GRAY2BGR );
    }
    else
    {
      cvtColor(tmp_img0, flow_img.rowRange(0,img0.rows), COLOR_GRAY2BGR );
      cvtColor(tmp_img1, flow_img.rowRange(img0.rows,2*img0.rows), COLOR_GRAY2BGR );
    }
  }
  else
  {
    if( img0.type() != DataType< Vec3b >::type )
      img0.convertTo(tmp_img0, DataType< Vec3b >::type);

    if( img1.type() != DataType< Vec3b >::type )
      img1.convertTo(tmp_img1, DataType< Vec3b >::type);

    if( side_by_side )
    {
      tmp_img0.copyTo(flow_img.colRange(0,img0.cols));
      tmp_img1.copyTo(flow_img.colRange(img0.cols,2*img0.cols));
    }
    else
    {
      tmp_img0.copyTo(flow_img.rowRange(0,img0.rows));
      tmp_img1.copyTo(flow_img.rowRange(img0.rows,2*img0.rows));
    }
  }

  if( !mask.empty() )
  {
    for(int y = 0; y < flow.rows; y += step)
    {
      const uchar *mask_p = mask.ptr<uchar>(y);
      const Point2f *flow_p = flow.ptr<Point2f>(y);

      for(int x = 0; x < flow.cols; x += step, mask_p += step, flow_p += step )
      {
        if( *mask_p )
        {
          const Point2f &fxy = *flow_p;
          line(flow_img, Point(x,y),
                  Point( cvRound( (side_by_side?img0.cols:0) + x + fxy.x),
                         cvRound( (side_by_side?0:img0.rows) + y +fxy.y) ), color);
        }
      }
    }
  }
  else
  {
    for(int y = 0; y < flow.rows; y += step)
    {
      const Point2f *flow_p = flow.ptr<Point2f>(y);

      for(int x = 0; x < flow.cols; x += step, flow_p += step )
      {
        const Point2f &fxy = *flow_p;
        line(flow_img, Point(x,y),
                  Point( cvRound( (side_by_side?img0.cols:0) + x + fxy.x),
                         cvRound( (side_by_side?0:img0.rows) + y +fxy.y) ), color);
      }
    }
  }
  return flow_img;
}


#define CV_EXT_INSTANTIATE_drawPoints(_T) \
template void cv_ext::drawPoints(Mat &img, const std::vector< _T > &pts, \
                                 Scalar color, double scale );

#define CV_EXT_INSTANTIATE_drawNormalsDirections(_T) \
template void cv_ext::drawNormalsDirections( Mat &img, const std::vector< _T > &pts, \
                                             const std::vector< float > &normals_dirs, Scalar color, \
                                             int normal_length, double scale ); \
template void cv_ext::drawNormalsDirections( Mat &img, const std::vector< _T > &pts, \
                                             const std::vector< double > &normals_dirs, Scalar color, \
                                             int normal_length, double scale );
                                         
#define CV_EXT_INSTANTIATE_drawSegments(_T) \
template void cv_ext::drawSegments( Mat &img, const std::vector< _T > &segments, \
                                    Scalar color, int tickness, double scale );
                        
#define CV_EXT_INSTANTIATE_drawCircles(_T) \
template void cv_ext::drawCircles( Mat &img, const std::vector< _T > &pts, \
                                   int radius, Scalar color, double scale );
                                   
CV_EXT_INSTANTIATE( drawPoints, CV_EXT_2D_POINT_TYPES )
CV_EXT_INSTANTIATE( drawNormalsDirections, CV_EXT_2D_POINT_TYPES )
CV_EXT_INSTANTIATE( drawSegments, CV_EXT_4D_VECTOR_TYPES )
CV_EXT_INSTANTIATE( drawCircles, CV_EXT_2D_POINT_TYPES )