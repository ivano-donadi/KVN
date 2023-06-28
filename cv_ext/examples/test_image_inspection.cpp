#include "cv_ext/image_inspection.h"
#include "cv_ext/timer.h"
#include "cv_ext/debug_tools.h"
#include "omp.h"

#include <time.h>

using namespace std;
using namespace cv;

int main  ()
{
  // Initialize random seed
  srand (time(NULL));

  // Test image size
  const int w = 1280, h = 1024;
  // Num test segments
  const int n_segments = 1000000;
  // Max segments len
  const int max_seg_len = 200;
  // Num directions used to compute the summations
  const int n_directions = 60;
  // Compute a random direction for each segment
  const bool rand_direction = true;
  // If rand_direction is false, compute n_segments_per_dir for each direction
  int n_segments_per_dir = round(double(n_segments)/n_directions);

  // Create a random test image (all pixels are randomly chosen)
  Mat test_img = Mat ( Size ( w, h ), DataType<uchar>::type );
  for( int r = 0; r < test_img.rows; r++ )
  {
    uchar *img_p = test_img.ptr<uchar>(r);
    for( int c = 0; c < test_img.cols; c++, img_p++ )
      *img_p = rand()%255;
  }

//   cv_ext::showImage(test_img, "Test Image - Type ESC to continue");
  
  vector<Vec4i> segments(n_segments);
  vector<int> segs_i_dir(n_segments), segs_i_pat(n_segments),
              segs_len(n_segments), segs_offset(n_segments);
  vector<double> segs_sum(n_segments), int_segs_sum_v1(n_segments), int_segs_sum_v2(n_segments);

  cv_ext::BasicTimer timer;
  vector<cv_ext::DirectionalIntegralImage<uchar,double> > dir_int_imgs;
  dir_int_imgs.reserve(n_directions);

  timer.reset();
  // Create the integral images (i.e., the DirectionalIntegralImage objects, for n_directions discretized directions
  // in the interval [-M_PI/2, M_PI/2) )
  for( double direction = -M_PI/2; direction <= M_PI/2; direction += M_PI/double(n_directions))
    dir_int_imgs.push_back( cv_ext::DirectionalIntegralImage<uchar,double>( test_img, direction ) );
  
  cout<<"Elapsed time to compute "<<dir_int_imgs.size()<<" ["<<w<<"X"<<h<<"] integral images : "
      <<timer.elapsedTimeMs()<<endl;

  // Collect all the line patterns used to compute the directional integral images
  vector< vector<Point> > line_patterns(n_directions);
  vector< bool > x_major_dir(n_directions);
  for( int i = 0; i < n_directions; i++ )
  {
    dir_int_imgs[i].getLinePattern( line_patterns[i] );
    x_major_dir[i] = dir_int_imgs[i].isXMajor();
  }

  // Sample n_segments sample segments with random direction and random lenght
  // (use the line patterns used to compute the directional integral images)
  int i_dir = -1;
  for( int i = 0; i < n_segments; i++ )
  {
    bool outside = true;
    Point p0, p1;
    int len, i_pat, offset;

    if( rand_direction )
      // Sample a discretized direction
      i_dir = rand()%n_directions;
    else
    {
      if( !(i%n_segments_per_dir))
        i_dir++;
    }

    while( outside )
    {
      // Sample an index of the line pattern
      i_pat = rand()%line_patterns[i_dir].size();
      // Sample a lenght up to max_seg_len
      len = rand()%max_seg_len + 1;
      if(i_pat + len >= int(line_patterns[i_dir].size()))
        continue;

      p0 = line_patterns[i_dir][i_pat];
      p1 = line_patterns[i_dir][i_pat + len - 1];

      // Sample the line pattern offset
      if( x_major_dir[i_dir] )
      {
        offset = rand()%(2*h) - h/2;
        p0.y += offset;
        p1.y += offset;
      }
      else
      {
        offset = rand()%(2*w) - w/2;
        p0.x += offset;
        p1.x += offset;
      }

      if( p0.x >= 0 && p0.y >= 0 && p0.x < w && p0.y < h &&
          p1.x >= 0 && p1.y >= 0 && p1.x < w && p1.y < h )
        outside = false;
    }

    segments[i] = Vec4i( p0.x, p0.y, p1.x, p1.y );
    segs_i_dir[i] = i_dir;
    segs_len[i] = len;
    segs_i_pat[i] = i_pat;
    segs_offset[i] = offset;
  }

  timer.reset();
  for( int i = 0; i < n_segments; i++ )
  {
    double seg_sum = 0;
    int i_dir = segs_i_dir[i];
    for( int j = segs_i_pat[i]; j < segs_i_pat[i] + segs_len[i]; j++ )
    {
      Point p = line_patterns[i_dir][j];
      if( x_major_dir[i_dir] )
        p.y += segs_offset[i];
      else
        p.x += segs_offset[i];

      seg_sum += test_img.at<uchar>(p.y,p.x);
    }
    segs_sum[i] = seg_sum;
  }
  cout<<"Elapsed time to compute segment sum for "<<n_segments<<" segments : "<<timer.elapsedTimeMs()<<endl;


  timer.reset();
  // Compute the segment sum using DirectionalIntegralImage with segment's start and end points
  for( int i = 0; i < n_segments; i++ )
  {
    int i_dir = segs_i_dir[i];
    int_segs_sum_v1[i] = dir_int_imgs[i_dir].getSegmentSum(segments[i]);
  }
  cout<<"Elapsed time to compute segment sum with integral images for "
      <<n_segments<<" segments (first version): "<<timer.elapsedTimeMs()<<endl;


  timer.reset();
  // Compute the segment sum using DirectionalIntegralImage with segment's start point and lenght
  for( int i = 0; i < n_segments; i++ )
  {
    int i_dir = segs_i_dir[i];
    int len = segs_len[i];
    int_segs_sum_v2[i] = dir_int_imgs[i_dir].getSegmentSum(segments[i][0], segments[i][1], len);
  }
  cout<<"Elapsed time to compute segment sum with integral images for "
      <<n_segments<<" segments (second version): "<<timer.elapsedTimeMs()<<endl;

  // Check the computed summations
  for( int i = 0; i < n_segments; i++ )
  {
    if(int_segs_sum_v1[i] != segs_sum[i] )
      cout<<"WARNING!!! DirectionalIntegralImage provides a sum of "<<int_segs_sum_v1[i]
                <<" for a segment with actual sum of "<<segs_sum[i]
                <<" (segment index :"<<i
                <<", direction index :"<<segs_i_dir[i]<<" )"<<endl;

    if(int_segs_sum_v2[i] != segs_sum[i] )
      cout<<"WARNING!!! DirectionalIntegralImage provides a sum of "<<int_segs_sum_v2[i]
                <<" for a segment with actual sum of "<<segs_sum[i]
                <<" (segment index :"<<i
                <<", direction index :"<<segs_i_dir[i]<<" )"<<endl;
  }

  return 0;
}
