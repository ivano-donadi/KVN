#include "cv_ext/image_inspection.h"
#include "cv_ext/base.h"
#include "cv_ext/types.h"
#include "cv_ext/debug_tools.h"

using namespace std;
using namespace cv;

static void getDirectionalLinePattern( int width, int height, double &direction,
                                       vector<Point> &line_pattern, bool &x_major )
{
  direction = cv_ext::normalizeAngle(direction);
  if( direction >= M_PI/2 )
    direction -= M_PI;
  else if ( direction < -M_PI/2 )
    direction += M_PI;
  
  Point p0(0,0), p1;
  if( direction >= -M_PI/4 && direction < M_PI/4 )
  {
    x_major = true;
    p0 = Point(0,0);
    p1 = Point(width - 1, (width-1)*tan(direction));
  }
  else
  {
    x_major = false;
    if( direction > 0 )
      p1 = Point((height-1)*1.0/tan(direction), height - 1);
    else if( direction > -M_PI/2 )
      p1 = Point( (height-1)*1.0/tan(direction + M_PI), height - 1);
    else
      p1 = Point(0, height - 1);
  }

  cv_ext::getLinePoints( p0, p1, line_pattern );
}

void cv_ext :: getLinePoints( const Point &p0, const Point &p1,
                              vector<Point> &line_pts )
{
  int dx =  abs(p0.x-p1.x), sx = p0.x<p1.x ? 1 : -1;
  int dy = -abs(p0.y-p1.y), sy = p0.y<p1.y ? 1 : -1;
  int err = dx+dy, e2;
  int line_size = max<int>(dx,-dy) + 1;
  
  line_pts.clear();
  line_pts.reserve(line_size);
  
  Point p = p0;
  for( int i = 0; i < line_size; i++)
  {
    line_pts.push_back(p);
    e2 = 2*err;
    if (e2 >= dy) { err += dy; p.x += sx; } 
    if (e2 <= dx) { err += dx; p.y += sy; }
  }
}

template < typename _Tin, typename _Tout>
  cv_ext::DirectionalIntegralImage<_Tin, _Tout >::DirectionalIntegralImage( const Mat& src_img, double direction ) :
  offset_x_(0),
  offset_y_(0),
  direction_(direction)
{
  computeIntegralImage( src_img );
  
  lut_.resize(line_pattern_.size());
  if( x_major_ )
  {
    for( size_t i = 0; i < line_pattern_.size(); i++)
      lut_[i] = line_pattern_[i].y;
  }
  else
  {
    for( size_t i = 0; i < line_pattern_.size(); i++)
      lut_[i] = line_pattern_[i].x;
  }
}

template < typename _Tin, typename _Tout>
  void cv_ext::DirectionalIntegralImage<_Tin, _Tout >::computeIntegralImage( const Mat& src_img )
{
  w_ = src_img.cols; h_ = src_img.rows;
  int_img_ = Mat ( Size ( w_ + 1, h_ + 1 ), DataType<_Tout>::type );
  Mat sub_int_img;

  // Get the "line pattern", i.e. the line points offsets to be used to compute the summations
  getDirectionalLinePattern( w_ + 1, h_ + 1, direction_, line_pattern_, x_major_ );

//   Mat dbg_image(int_img_.size(), DataType<Vec3b>::type);
//   dbg_image.setTo(0);
//   Mat sub_dbg_img;

  // First and last point of the line pattern that could belong to the source image
  // (the last point could belong only to the directional integral image)
  Point &p0 = line_pattern_[0], &p1 = line_pattern_[line_pattern_.size() - 2];

  if( x_major_ )
  {
    int pattern_offset_y = p1.y - p0.y, init_y, end_y;

    // Line orientation is between -M_PI/4 and 0
    if( pattern_offset_y < 0 )
    {
      offset_y_ = 1;
      init_y = 0;
      end_y = h_ - pattern_offset_y;
      // Set to zero the first row and the last column
      int_img_.row(0).setTo(0);
      int_img_.col(w_).setTo(0);
      sub_int_img = int_img_(Rect(0,1,w_,h_));
//       sub_dbg_img = dbg_image(Rect(0,1,w_,h_));
    }
    else
    // Line orientation is between 0 and M_PI/4
    {
      init_y = -pattern_offset_y;
      end_y = h_;
      // Set to zero the last row and the last column
      int_img_.row(h_).setTo(0);
      int_img_.col(w_).setTo(0);
      sub_int_img = int_img_(Rect(0,0,w_,h_));
//       sub_dbg_img = dbg_image(Rect(0,0,w_,h_));
    }

    // Move the line pattern (line_pts) top to down
    for(int iy = init_y; iy < end_y; iy++)
    {
      // Restart the summation for each line
      _Tout sum = 0;
      // Index of the last pattern element that lies inside the image
      int last_i = -1;

      for( int i = 0; i < int(line_pattern_.size()) - 1; i++ )
      {
        Point &p = line_pattern_[i];
        int &x = p.x, y = p.y + iy;
        if ( unsigned(x) < unsigned(w_) && unsigned(y) < unsigned(h_) )
        {
          sub_int_img.at<_Tout>(y, x) = sum;
//           sub_dbg_img.at<Vec3b>(y, x) = sum?map2RGB(uint32_t(sum), 0x1FFFF):Vec3b(0,0,0);
          sum += _Tout(src_img.at<_Tin>(y, x));
          last_i = i;
        }
      }
      // Add the last pixel in the integral image that represents the sum of the whole line
      Point &p = line_pattern_[last_i + 1];
      int x = p.x, y = p.y + iy + offset_y_;
      int_img_.at<_Tout>(y, x) = sum;
//       dbg_image.at<Vec3b>(y, x) = sum?map2RGB(uint32_t(sum), 0x1FFFF):Vec3b(0,0,0);
    }
  }
  else
  {
    int pattern_offset_x = p1.x - p0.x, init_x, end_x;

    // Line orientation is between -M_PI/2 and -M_PI/4 or
    if(pattern_offset_x < 0)
    {
      init_x = 0;
      end_x = w_ - pattern_offset_x;
      offset_x_ = 1;
      // Set to zero the last row and the first column
      int_img_.row(h_).setTo(0);
      int_img_.col(0).setTo(0);
      sub_int_img = int_img_(Rect(1,0,w_,h_));
//       sub_dbg_img = dbg_image(Rect(1,0,w_,h_));
    }
    // Line orientation is between M_PI/4 and M_PI/2 or
    else
    {
      init_x = -pattern_offset_x;
      end_x = w_;
      // Set to zero the last row and the last column
      int_img_.row(h_).setTo(0);
      int_img_.col(w_).setTo(0);
      sub_int_img = int_img_(Rect(0,0,w_,h_));
//       sub_dbg_img = dbg_image(Rect(0,0,w_,h_));
    }

    // Move the line pattern (line_pts) left to right
    for(int ix = init_x; ix < end_x; ix++)
    {
      // Restart the summation for each line
      _Tout sum = 0;
      // Index of the last pattern element that lies inside the image
      int last_i = -1;

      for( int i = 0; i < int(line_pattern_.size()) - 1; i++ )
      {
        Point &p = line_pattern_[i];
        int x = p.x + ix, y = p.y;
        if ( unsigned(x) < unsigned(w_) && unsigned(y) < unsigned(h_) )
        {
          sub_int_img.at<_Tout>(y, x) = sum;
//           sub_dbg_img.at<Vec3b>(y, x) = sum?map2RGB(uint32_t(sum), 0x1FFFF):Vec3b(0,0,0);
          sum += _Tout(src_img.at<_Tin>(y, x));
          last_i = i;
        }
      }
      // Add the last pixel in the integral image that represents the sum of the whole line
      Point &p = line_pattern_[last_i + 1];
      int x = p.x + ix + offset_x_, y = p.y;
      int_img_.at<_Tout>(y, x) = sum;
//       dbg_image.at<Vec3b>(y, x) = sum?map2RGB(uint32_t(sum), 0x1FFFF):Vec3b(0,0,0);
    }
  }
//   cv_ext::showImage(dbg_image);
}

template < typename _Tin, typename _Tout>
  void cv_ext::DirectionalIntegralImage<_Tin, _Tout >::getLinePattern( std::vector<cv::Point> &pattern ) const
{
  pattern = line_pattern_;
  pattern.resize(pattern.size() - 1);
}

#define CV_EXT_INSTANTIATE_DirectionalIntegralImage(_T) \
template class cv_ext::DirectionalIntegralImage< _T, uint32_t >; \
template class cv_ext::DirectionalIntegralImage< _T, uint64_t >; \
template class cv_ext::DirectionalIntegralImage< _T, float >; \
template class cv_ext::DirectionalIntegralImage< _T, double >;

CV_EXT_INSTANTIATE( DirectionalIntegralImage, CV_EXT_PIXEL_DEPTH_TYPES )