#include "distance_transforms.h"

#include <utility>

#include "omp.h"
//
extern "C"
{
#include "lsd.h"
}

#if defined(D2CO_USE_SSE) || defined(D2CO_USE_AVX)
#include <immintrin.h>
#endif

using namespace cv;
using namespace cv_ext;
using namespace std;

void DistanceTransform::setEdgeDetector(EdgeDetectorUniquePtr ed)
{
  edge_detector_ = std::move(ed);
}

void DistanceTransform::computeDistanceMap( const Mat &src_img, Mat &dist_map )
{
  Mat edge_map;
  edge_detector_->enableWhiteBackground(true);
  edge_detector_->setImage(src_img);
  edge_detector_->getEdgeMap( edge_map);

  cv::distanceTransform ( edge_map, dist_map, dist_type_, mask_size_ );
  
  if( dist_thresh_ > 0 )
    dist_map.setTo(dist_thresh_, dist_map > dist_thresh_ );
}

void DistanceTransform::computeDistanceMap( const Mat &src_img, Mat &dist_map,
                                            Mat &closest_edgels_map)
{
  Mat edge_map;
  edge_detector_->enableWhiteBackground(true);
  edge_detector_->setImage( src_img );
  edge_detector_->getEdgeMap( edge_map );

  distanceMapClosestEdgels( edge_map, dist_map, closest_edgels_map);
}

void DistanceTransform::computeDistDirMap ( const Mat& src_img, Mat& dist_map, Mat& closest_dir_map, int num_directions )
{
  Mat edge_map, edge_dir_map;
  edge_detector_->enableWhiteBackground(true);
  edge_detector_->setNumEdgeDirections(num_directions);
  edge_detector_->setImage( src_img );
  edge_detector_->getEdgeMap( edge_map );
  edge_detector_->getEdgeDirectionsMap( edge_dir_map );

  distanceMapClosestEdgelsDir( edge_map, edge_dir_map, dist_map, closest_dir_map);
}

void DistanceTransform::computeDistanceMapTensor( const Mat &src_img,
                                                  ImageTensor &dist_map_tensor,
                                                  int num_directions, double lambda, bool smooth_tensor )
{
  edge_detector_->enableWhiteBackground(true);
  edge_detector_->setNumEdgeDirections(num_directions);
  edge_detector_->setImage(src_img);

//  Mat edge_map;
//  edge_detector_->getEdgeMap(edge_map);
//  cv_ext::showImage(edge_map, "edge map");

  dist_map_tensor.create(src_img.rows, src_img.cols, num_directions, DataType<float>::type );

//   static uint64_t timer1 = 0, timer2 = 0, counter = 1;
//   cv_ext::BasicTimer timer;
  #pragma omp parallel for schedule(static) if( parallelism_enabled_ )
  for( int i = 0; i < num_directions; i++ )
  {
    Mat edge_map;
    edge_detector_->getDirectionalEdgeMap(i,edge_map);
    cv::distanceTransform (edge_map, dist_map_tensor[i], dist_type_, mask_size_ );
    if( dist_thresh_ > 0 )
      dist_map_tensor[i].setTo(dist_thresh_, dist_map_tensor[i] > dist_thresh_ );
    //cv_ext::showImage(dist_map_tensor[i], "Segments");
  }

//   cout<<"distanceTransform elapsed time ms : "<<(timer1+=timer.elapsedTimeMs())/counter<<endl;
//   timer.reset();
  
  int w = src_img.cols, h = src_img.rows;

  double eta_dir = double(num_directions)/M_PI;
  double ang_step = 1.0/eta_dir;
  const float ang_penalty = lambda * ang_step;
  //cout<<"ang_penalty : "<<ang_penalty<<endl;
  
  const int n_threads = ( parallelism_enabled_ ? omp_get_max_threads() : 1 );  

  int th_h = h/n_threads;
  if( th_h < 1) th_h = 1;
  
  vector< std::pair<int,int> > row_ranges(n_threads);
  int cur_r = 0;
  for( int i = 0; i < n_threads - 1; i++ )
  {
    row_ranges[i].first = cur_r;
    cur_r += th_h;
    row_ranges[i].second = cur_r;
  }
  
  row_ranges[n_threads - 1].first = cur_r;
  row_ranges[n_threads - 1].second = h;
  
  #pragma omp parallel if( parallelism_enabled_ )
  {
    int i_th = omp_get_thread_num();
    // Forward recursion
    bool first_pass = true;
    for( int i = 0, prev_i = num_directions - 1; i < num_directions; i++, prev_i++ )
    {
      prev_i %= num_directions;

      Mat prev_map = dist_map_tensor[prev_i].rowRange(row_ranges[i_th].first, row_ranges[i_th].second),
          cur_map = dist_map_tensor[i].rowRange(row_ranges[i_th].first, row_ranges[i_th].second);
       
#if defined(D2CO_USE_AVX)
          
      __m256 cand_dist, ang_p = _mm256_set1_ps(ang_penalty), prev_dist,cur_dist, dist_mask;
      float *prev_map_p = (float *)prev_map.data,
            *cur_map_p = (float *)cur_map.data, 
            *end_map_p = cur_map_p + (cur_map.step1()*cur_map.rows);
        
      for( ; cur_map_p != end_map_p; cur_map_p += 8, prev_map_p += 8 )
      {
        prev_dist = _mm256_load_ps (prev_map_p);
        cur_dist = _mm256_load_ps (cur_map_p);
        cand_dist = _mm256_add_ps(prev_dist, ang_p);
        dist_mask = _mm256_cmp_ps(cur_dist, cand_dist, _CMP_GT_OS);
       _mm256_maskstore_ps(cur_map_p, _mm256_castps_si256(dist_mask), cand_dist);      
      }
      
#elif defined(D2CO_USE_SSE)
      
      __m128 cand_dist, ang_p = _mm_set1_ps(ang_penalty); 
      __m128 *prev_map_p = (__m128 *)prev_map.data, 
             *cur_map_p = (__m128 *)cur_map.data,  
             *end_map_p = cur_map_p + (cur_map.step1()*cur_map.rows)/4; 
            
      for( ; cur_map_p != end_map_p; cur_map_p++, prev_map_p++ ) 
      { 
        cand_dist = _mm_add_ps(*prev_map_p, ang_p); 
        *cur_map_p = _mm_min_ps(*cur_map_p, cand_dist);           
      }
      
#else
      for( int y = 0; y < th_h; y++)
      {
        float *prev_map_p = prev_map.ptr<float>(y),
              *cur_map_p = cur_map.ptr<float>(y);
        for( int x = 0; x < w; x++, prev_map_p++, cur_map_p++ )
        {
          float new_dist = *prev_map_p + ang_penalty;
          if(*cur_map_p > new_dist )
            *cur_map_p = new_dist;
        }
      }
#endif

      if( i == num_directions - 1 && first_pass )
      {
        i = -1;
        first_pass = false;
      }
    }

    // Backward recursion
    first_pass = true;
    for( int i = num_directions - 1, next_i = 0; i >= 0; i--, next_i-- )
    {
      if( next_i < 0 )
        next_i = num_directions - 1;
      
      Mat cur_map = dist_map_tensor[i].rowRange(row_ranges[i_th].first, row_ranges[i_th].second),
          next_map = dist_map_tensor[next_i].rowRange(row_ranges[i_th].first, row_ranges[i_th].second);
          
#if defined(D2CO_USE_AVX)
      
      __m256 cand_dist, ang_p = _mm256_set1_ps(ang_penalty), next_dist,cur_dist, dist_mask;
      float *next_map_p = (float *)next_map.data,
            *cur_map_p = (float *)cur_map.data, 
            *end_map_p = cur_map_p + (cur_map.step1()*cur_map.rows);
        
      for( ; cur_map_p != end_map_p; cur_map_p += 8, next_map_p += 8 )
      {
        next_dist = _mm256_load_ps (next_map_p);
        cur_dist = _mm256_load_ps (cur_map_p);
        cand_dist = _mm256_add_ps(next_dist, ang_p);
        dist_mask = _mm256_cmp_ps(cur_dist, cand_dist, _CMP_GT_OS);
       _mm256_maskstore_ps(cur_map_p, _mm256_castps_si256(dist_mask), cand_dist);      
      }
            
#elif defined(D2CO_USE_SSE)
      
      __m128 cand_dist, ang_p = _mm_set1_ps(ang_penalty); 
      __m128 *next_map_p = (__m128 *)next_map.data, 
             *cur_map_p = (__m128 *)cur_map.data,  
             *end_map_p = cur_map_p + (cur_map.step1()*cur_map.rows)/4; 
            
      for( ; cur_map_p != end_map_p; cur_map_p++, next_map_p++ ) 
      { 
        cand_dist = _mm_add_ps(*next_map_p, ang_p); 
        *cur_map_p = _mm_min_ps(*cur_map_p, cand_dist);           
      }
      
#else

      for( int y = 0; y < th_h; y++)
      {
        float *cur_map_p = cur_map.ptr<float>(y),
              *next_map_p = next_map.ptr<float>(y);
        for( int x = 0; x < w; x++, cur_map_p++, next_map_p++ )
        {
          float new_dist = *next_map_p + ang_penalty;
          if(*cur_map_p > new_dist )
            *cur_map_p = new_dist;
        }
      }
      
#endif    


      if( i == 0 && first_pass )
      {
        i = num_directions;
        first_pass = false;
      }
    }
  }
//   cout<<"Recursion elapsed time ms : "<<(timer2+=timer.elapsedTimeMs())/counter<<endl;
//   counter++;
  
  if( smooth_tensor )
    smoothTensor(dist_map_tensor);
}

void DistanceTransform::computeDistanceMapTensor( const Mat &src_img,
                                                  ImageTensor &dist_map_tensor,
                                                  ImageTensor &edgels_map_tensor,
                                                  int num_directions, double lambda, bool smooth_tensor )
{
  edge_detector_->enableWhiteBackground(true);
  edge_detector_->setNumEdgeDirections(num_directions);
  edge_detector_->setImage(src_img);

  dist_map_tensor.create(src_img.rows, src_img.cols, num_directions, DataType<float>::type );
  edgels_map_tensor.create(src_img.rows, src_img.cols, num_directions, DataType<Point2f>::type );

  #pragma omp parallel for schedule(static) if( parallelism_enabled_ )
  for( int i = 0; i < num_directions; i++ )
  {
    Mat edge_map;
    edge_detector_->getDirectionalEdgeMap(i,edge_map);
    distanceMapClosestEdgels(edge_map, dist_map_tensor[i], edgels_map_tensor[i] );
  }

  int w = src_img.cols, h = src_img.rows;
  double eta_dir = double(num_directions)/M_PI;
  double ang_step = 1.0/eta_dir;
  const float ang_penalty = lambda * ang_step;
  //cout<<"ang_penalty : "<<ang_penalty<<endl;

  // Forward recursion
  bool first_pass = true;
  for( int i = 0, prev_i = num_directions - 1; i < num_directions; i++, prev_i++ )
  {
    prev_i %= num_directions;
    #pragma omp parallel for schedule(static) if( parallelism_enabled_ )
    for( int y = 0; y < h; y++)
    {
      float *prev_map_p = dist_map_tensor[prev_i].ptr<float>(y),
            *cur_map_p = dist_map_tensor[i].ptr<float>(y);
      Point2f *prev_points_map_p = edgels_map_tensor[prev_i].ptr<Point2f>(y),
              *cur_points_map_p = edgels_map_tensor[i].ptr<Point2f>(y);

      for( int x = 0; x < w; x++, prev_map_p++, cur_map_p++, prev_points_map_p++, cur_points_map_p++ )
      {
        float new_dist = *prev_map_p + ang_penalty;
        if( *cur_map_p > new_dist )
        {
          *cur_map_p = new_dist;
          *cur_points_map_p = *prev_points_map_p;
        }
      }
    }

    if( i == num_directions - 1 && first_pass )
    {
      i = -1;
      first_pass = false;
    }
  }

  // Backward recursion
  first_pass = true;
  for( int i = num_directions - 1, next_i = 0; i >= 0; i--, next_i-- )
  {

    if( next_i < 0 )
      next_i = num_directions - 1;
    #pragma omp parallel for schedule(static) if( parallelism_enabled_ )
    for( int y = 0; y < h; y++)
    {
      float *cur_map_p = dist_map_tensor[i].ptr<float>(y),
            *next_map_p = dist_map_tensor[next_i].ptr<float>(y);
      Point2f *cur_points_map_p = edgels_map_tensor[i].ptr<Point2f>(y),
              *next_points_map_p = edgels_map_tensor[next_i].ptr<Point2f>(y);
      for( int x = 0; x < w; x++, cur_map_p++, next_map_p++, cur_points_map_p++, next_points_map_p++ )
      {
        float new_dist = *next_map_p + ang_penalty;
        if(*cur_map_p > new_dist )
        {
          *cur_map_p = new_dist;
          *cur_points_map_p = *next_points_map_p;
        }
      }
    }
    if( i == 0 && first_pass )
    {
      i = num_directions;
      first_pass = false;
    }
  }

  if( smooth_tensor )
    smoothTensor( dist_map_tensor );
}

void DistanceTransform::computeDistanceMapTensor( const Mat &src_img,
                                                  ImageTensor &dist_map_tensor,
                                                  ImageTensor &x_dist_map_tensor,
                                                  ImageTensor &y_dist_map_tensor,
                                                  ImageTensor &edgels_map_tensor,
                                                  int num_directions, double lambda, bool smooth_tensor )
{
  DistanceTransform::computeDistanceMapTensor( src_img, dist_map_tensor, edgels_map_tensor,
                                               num_directions, lambda, smooth_tensor );

  x_dist_map_tensor.create(src_img.rows, src_img.cols, num_directions, DataType<float>::type );
  x_dist_map_tensor.create(src_img.rows, src_img.cols, num_directions, DataType<float>::type );
    
  int w = src_img.cols, h = src_img.rows;

#pragma omp parallel for schedule(static) if( parallelism_enabled_ )
  for( int i = 0 ; i < num_directions; i++ )
  {
    for( int y = 0; y < h; y++)
    {
      float *x_map_p = x_dist_map_tensor[i].ptr<float>(y),
            *y_map_p = y_dist_map_tensor[i].ptr<float>(y);
      Point2f *points_map_p = edgels_map_tensor[i].ptr<Point2f>(y);
      for( int x = 0; x < w; x++, x_map_p++, y_map_p++, points_map_p++ )
      {
        *x_map_p = (*points_map_p).x - float( x );
        *y_map_p = (*points_map_p).y - float( y );
      }
    }

//     cv_ext::showImage(x_dist_map_tensor[i], "x_dst_map_tensor[i]");
//     cv_ext::showImage(y_dist_map_tensor[i], "y_dst_map_tensor[i]");
  }

  if( smooth_tensor )
  {
    smoothTensor(x_dist_map_tensor);
    smoothTensor(y_dist_map_tensor);
  }
}

void DistanceTransform::distanceMapClosestEdgels(const Mat& edge_map, Mat& dist_map, Mat& closest_edgels_map)
{
  Mat labels_img(edge_map.size(), DataType<u_int32_t>::type );
  closest_edgels_map = Mat(edge_map.size(), DataType<Point2f>::type );

  // The labels_img matrix groups all the pixel nearest to a edgels with a common label (ie., an ID), 
  // shared with the edgels itself  
  cv::distanceTransform(edge_map, dist_map, labels_img, dist_type_, mask_size_, DIST_LABEL_PIXEL );
  if( dist_thresh_ > 0 )
    dist_map.setTo(dist_thresh_, dist_map > dist_thresh_ );

  int labels_size = edge_map.total() - countNonZero(edge_map) + 1;
  vector< Point2f> edgels(labels_size);

  // Retreive the labels of each edgel...
  for( int y = 0; y < labels_img.rows; y++)
  {
    const uchar *em_row_p = edge_map.ptr<uchar>(y);
    const u_int32_t *label_row_p = labels_img.ptr<u_int32_t>(y);
    for( int x = 0; x < labels_img.cols; x++, label_row_p++, em_row_p++ )
    {
      if( !(*em_row_p) )
        edgels[*label_row_p] = Point2f(x,y);
    }
  }

  for( int y = 0; y < labels_img.rows; y++)
  {
    Point2f *points_row_p = closest_edgels_map.ptr<Point2f>(y);
    const u_int32_t *label_row_p = labels_img.ptr<u_int32_t>(y);
    for( int x = 0; x < labels_img.cols; x++, label_row_p++, points_row_p++ )
      *points_row_p = edgels[*label_row_p];
  }

// DEBUG CODE
//     for( int y = 0; y < labels_img.rows; y++)
//     {
//       const Point2f *points_row_p = closest_edgels_map.ptr<Point2f>(y);
//       for( int x = 0; x < labels_img.cols; x++, points_row_p++ )
//       {
//         Mat dbg_img = seg_img.clone();
//         const Point2f &closest_edgels = *points_row_p;
//         line(dbg_img, Point(x,y), Point(closest_edgels.x, closest_edgels.y),
//                  Scalar(128,128,128));
//         cv_ext::showImage(dbg_img, "seg_img");
//       }
//     }
//   cv_ext::showImage(seg_img, "seg_img");
// END DEDUG CODE
}

void DistanceTransform::distanceMapClosestEdgelsDir ( const Mat &edge_map, const Mat &edge_dir_map,
                                                      Mat &dist_map, Mat &closest_edgels_dir_map )
{
  Mat labels_img(edge_map.size(), DataType<u_int32_t>::type );
  closest_edgels_dir_map = Mat_<ushort>(edge_map.size());

  // The labels_img matrix groups all the pixel nearest to a edgels with a common label (ie., an ID), 
  // shared with the edgels itself  
  cv::distanceTransform(edge_map, dist_map, labels_img, dist_type_, mask_size_, DIST_LABEL_PIXEL );
  if( dist_thresh_ > 0 )
    dist_map.setTo(dist_thresh_, dist_map > dist_thresh_ );

  int labels_size = edge_map.total() - countNonZero(edge_map) + 1;
  vector<ushort> edgels(labels_size);

  // Retreive the direction for of each edgel...
  for( int y = 0; y < labels_img.rows; y++)
  {
    const uchar *em_row_p = edge_map.ptr<uchar>(y);
    const ushort *ed_row_p = edge_dir_map.ptr<ushort>(y);
    const u_int32_t *label_row_p = labels_img.ptr<u_int32_t>(y);
    for( int x = 0; x < labels_img.cols; x++, label_row_p++, em_row_p++, ed_row_p++ )
    {
      if( !(*em_row_p) )
        edgels[*label_row_p] = *ed_row_p;
    }
  }

  for( int y = 0; y < labels_img.rows; y++)
  {
    ushort *ced_row_p = closest_edgels_dir_map.ptr<ushort>(y);
    const u_int32_t *label_row_p = labels_img.ptr<u_int32_t>(y);
    for( int x = 0; x < labels_img.cols; x++, label_row_p++, ced_row_p++ )
      *ced_row_p = edgels[*label_row_p];
  }
    
//   // DEBUG CODE
//   cv_ext::showImage(edge_map, "edge_map");
//   cv_ext::showImage(edge_dir_map, "edge_dir_map");
//   cv_ext::showImage(closest_edgels_dir_map, "closest_edgels_dir_map");
//   // END DEBUG CODE
}

void DistanceTransform::smoothTensor( ImageTensor &dist_map_tensor )
{
  int num_directions = dist_map_tensor.depth();
  ImageTensor smoothed_tensor( dist_map_tensor.rows(), dist_map_tensor.rows(),
                               num_directions, DataType<float>::type );

  #pragma omp parallel for if( parallelism_enabled_ )
  for( int i = 0; i < num_directions; i++ )
  {
    int prev_i = i - 1, next_i = i + 1;
    if( i == 0 )
      prev_i = num_directions - 1;
    else if(i == num_directions - 1)
      next_i = 0;

    smoothed_tensor[i] =  0.25*dist_map_tensor.at(prev_i) +
                          0.5*dist_map_tensor.at(i) +
                          0.25*dist_map_tensor.at(next_i);
  }
  dist_map_tensor = std::move(smoothed_tensor);
}
