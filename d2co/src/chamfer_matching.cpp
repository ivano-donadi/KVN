#include "chamfer_matching.h"

#include <omp.h>
#include <algorithm>
#include <stdexcept>

#include "cv_ext/cv_ext.h"

#if defined(D2CO_USE_SSE) || defined(D2CO_USE_AVX)
#include <immintrin.h>
#elif defined(D2CO_USE_NEON)
#include <arm_neon.h>
#endif

using namespace cv;
using namespace cv_ext;
using namespace std;

template class ChamferMatchingBase<PointSet>;
template class ChamferMatchingBase<DirIdxPointSet>;
template class DirectionalChamferMatchingBase<DirIdxPointSet>;

#define CHAMFER_MATCHING_FAST_MAP 1

#if CHAMFER_MATCHING_FAST_MAP
#if ( defined(D2CO_USE_SSE) || defined(D2CO_USE_NEON) )
static cv_ext::MemoryAlignment mem_align = cv_ext::MEM_ALIGN_16;
#elif defined(D2CO_USE_AVX)
static cv_ext::MemoryAlignment mem_align = cv_ext::MEM_ALIGN_32;
#else
static cv_ext::MemoryAlignment mem_align = cv_ext::MEM_ALIGN_NONE;
#endif
#endif

template <typename _T>
  void ChamferMatchingBase<_T>::setInput (const Mat& dist_map )
{
  dist_map_ =  dist_map;
  
#if CHAMFER_MATCHING_FAST_MAP  
  // TODO Switch each map to ushort type?
  dist_map_.convertTo(fast_dist_map_, cv::DataType<ushort>::type);
#endif
}

template <typename _T>
  void ChamferMatchingBase<_T>::match(int num_best_matches, vector<TemplateMatch >& matches,
                                      int image_step, int match_cell_size ) const
{
  assert( !dist_map_.empty() );

  const int n_threads = ( parallelism_enabled_ ? omp_get_max_threads() : 1 );

  vector< multimap< double, TemplateMatch > > best_matches(n_threads);
  vector< float > min_dists( n_threads, std::numeric_limits< float >::max());
  const int w = dist_map_.cols, h = dist_map_.rows;

#if CHAMFER_MATCHING_FAST_MAP && ( defined(D2CO_USE_SSE) || defined(D2CO_USE_AVX) || defined(D2CO_USE_NEON) )
//   cv_ext::BasicTimer timer;
    
  vector<Mat> step_map;
  step_map.reserve(image_step);

  int step_map_w = cvCeil(static_cast<float>(w)/image_step), step_map_h = h;

  for( int j = 0; j < image_step; j++ )
    step_map.push_back(cv_ext::AlignedMat(step_map_h, step_map_w, DataType<ushort>::type, mem_align));
  
  for( int y = 0; y < h; y++)
  {
    const ushort *map_p = fast_dist_map_.ptr<ushort>(y);
    vector <ushort *> step_map_p(image_step);
    for( int j = 0; j < image_step; j++ )
      step_map_p[j] = step_map[j].ptr<ushort>(y);
    
    for( int x = 0; x < w; x++, map_p++)
      *step_map_p[x%image_step]++ = *map_p;
  }
//   cout<<"OBJ_REC_FAST_SSE data creation Elapsed time ms : "<<timer.elapsedTimeMs()<<endl;

#endif

  #pragma omp parallel for schedule(dynamic) if( parallelism_enabled_ )
  for( int templ_i = 0; templ_i < static_cast<int>(tv_ptr_->size()); templ_i++ )
  {
    int th_id = omp_get_thread_num();
    auto &th_best_matches = best_matches[th_id];
    auto &min_dist = min_dists[th_id];
    auto &templ = tv_ptr_->at(templ_i);
    
    vector<Point> proj_pts = templ.proj_pts;

    // No x,y exhaustive matching, just the templates in their original position
    if( image_step < 1 )
    {
      float dist = templateDist(templ);

      if( int(th_best_matches.size()) < num_best_matches )
      {
        th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist,Point(0,0))) );
        min_dist = th_best_matches.rbegin()->first;
      }
      else if( dist < min_dist )
      {
        th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist, Point(0,0))) );
        th_best_matches.erase(--th_best_matches.rbegin().base());
        min_dist = th_best_matches.rbegin()->first;
      }
    }
    else
    {
      Rect &bb = templ.bbox;
      Point tl = bb.tl();
      for( int i_p = 0; i_p < static_cast<int>(proj_pts.size()); i_p++ )
        proj_pts[i_p] -= tl;

      int dists_w = (w - bb.width)/image_step + 1, dists_h = (h - bb.height)/image_step + 1;      
      Mat_<float> avg_dists(dists_h, dists_w);
      
#if CHAMFER_MATCHING_FAST_MAP

      cv_ext::AlignedMat c_dists(dists_h, dists_w, DataType<ushort>::type, cv::Scalar(0) );

#if defined(D2CO_USE_AVX)

      __m256i cur_dist, acc_dist, *c_dists_p, *dist_map_p;
      int vec_c_dists_w = c_dists.step1()/16;
      for( int i_p = 0; i_p < static_cast<int>(proj_pts.size()); i_p++ )
      {
        int x = proj_pts[i_p].x;
        Mat &dist_map = step_map[x%image_step];

        for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
        {
          c_dists_p = (__m256i *)c_dists.ptr<ushort>(c_dists_y);
          dist_map_p = (__m256i *)(dist_map.ptr<ushort>(y) + x/image_step);

          for(int vec_c_dists_x = 0; vec_c_dists_x < vec_c_dists_w; dist_map_p++, c_dists_p++, vec_c_dists_x++ )
          {
            cur_dist = _mm256_loadu_si256(dist_map_p);
            acc_dist = _mm256_load_si256(c_dists_p);
            acc_dist = _mm256_adds_epu16(acc_dist, cur_dist);
            _mm256_store_si256(c_dists_p, acc_dist );
          }
        }
      }

#elif defined(D2CO_USE_SSE)

      __m128i cur_dist, acc_dist, *c_dists_p, *dist_map_p;
      int vec_c_dists_w = c_dists.step1()/8;
      for( int i_p = 0; i_p < static_cast<int>(proj_pts.size()); i_p++ )
      {
        int x = proj_pts[i_p].x;
        Mat &dist_map = step_map[x%image_step];

        for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
        {
          c_dists_p = (__m128i *)c_dists.ptr<ushort>(c_dists_y);
          dist_map_p = (__m128i *)(dist_map.ptr<ushort>(y) + x/image_step);

          for(int vec_c_dists_x = 0; vec_c_dists_x < vec_c_dists_w; dist_map_p++, c_dists_p++, vec_c_dists_x++ )
          {
            cur_dist = _mm_loadu_si128(dist_map_p);
            acc_dist = _mm_load_si128(c_dists_p);
            acc_dist = _mm_adds_epu16(acc_dist, cur_dist);
            _mm_store_si128(c_dists_p, acc_dist );
          }
        }
      }

#elif defined(D2CO_USE_NEON)

      uint16x8_t cur_dist, acc_dist;
      uint16_t* c_dists_p, *dist_map_p;
      int vec_c_dists_w = c_dists.step1()/8;
      for( int i_p = 0; i_p < static_cast<int>(proj_pts.size()); i_p++ )
      {
        int x = proj_pts[i_p].x;
        Mat &dist_map = step_map[x%image_step];

        for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
        {
          c_dists_p = (uint16_t *)c_dists.ptr<ushort>(c_dists_y);
          dist_map_p = (uint16_t *)(dist_map.ptr<ushort>(y) + x/image_step);

          for(int vec_c_dists_x = 0; vec_c_dists_x < vec_c_dists_w; dist_map_p+=8, c_dists_p+=8, vec_c_dists_x++ )
          {
            cur_dist = vld1q_u16(dist_map_p); // Unaligned
            acc_dist = vld1q_u16(c_dists_p);
            acc_dist = vaddq_u16(acc_dist, cur_dist);
            vst1q_u16(c_dists_p, acc_dist );
          }
        }
      }

#else

      for( int i_p = 0; i_p < static_cast<int>(proj_pts.size()); i_p++ )
      {
        for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
        {
          ushort *dists_p = c_dists.ptr<ushort>(c_dists_y);
          const ushort *dist_map_p = fast_dist_map_.ptr<ushort>(y) + proj_pts[i_p].x;

          for(int c_dists_x = 0; c_dists_x < dists_w; dist_map_p += image_step, dists_p++, c_dists_x++ )
            *dists_p += *dist_map_p;
        }
      }

#endif

      float norm_factor = 1.0f/proj_pts.size();
      c_dists.convertTo(avg_dists, cv::DataType< float >::type, norm_factor);   

#else // CHAMFER_MATCHING_FAST_MAP

      avg_dists.setTo(Scalar(0));
      for( int i_p = 0; i_p < static_cast<int>(proj_pts.size()); i_p++ )
      {
        for( int y = proj_pts[i_p].y, avg_dists_y = 0; avg_dists_y < dists_h; y += image_step, avg_dists_y++ )
        {
          float *dists_p = avg_dists.ptr<float>(avg_dists_y);
          const float *dist_map_p = dist_map_.ptr<float>(y) + proj_pts[i_p].x;

          for(int avg_dists_x = 0; avg_dists_x < dists_w; dist_map_p += image_step, dists_p++, avg_dists_x++ )
            *dists_p += *dist_map_p;
        }
      }

      avg_dists *= 1.0f/proj_pts.size();
      
#endif
      
      for(int avg_dists_y = 0; avg_dists_y < dists_h; avg_dists_y++ )
      {
        float *dists_p = avg_dists.ptr<float>(avg_dists_y);
        for(int avg_dists_x = 0; avg_dists_x < dists_w; avg_dists_x++, dists_p++ )
        {
          float dist = *dists_p;
          if( int(th_best_matches.size()) < num_best_matches )
          {
            Point offset( avg_dists_x*image_step, avg_dists_y*image_step);
            offset -= bb.tl();
            th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist, offset)) );
            min_dist = th_best_matches.rbegin()->first;
          }
          else if( dist < min_dist )
          {
            Point offset( avg_dists_x*image_step, avg_dists_y*image_step);
            offset -= bb.tl();
            th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist, offset)) );
            th_best_matches.erase(--th_best_matches.rbegin().base());
            min_dist = th_best_matches.rbegin()->first;
          }
        }
      }      
    }
  }
  
  for( int i = 1; i < n_threads; i++ )
    best_matches[0].insert(best_matches[i].begin(), best_matches[i].end());

  matches.clear();
  matches.reserve(num_best_matches);
  int num_matches = 0;
  for( auto iter = best_matches[0].begin();
       iter != best_matches[0].end() && num_matches < num_best_matches;
       iter++, num_matches++ )

    matches.push_back(iter->second);

  
//   for( auto iter = best_pts.begin(); iter != best_pts.end(); iter++ )
//   {
//     Mat dbg_img2 = dbg_img.clone();
// //     Mat dbg_img(cam_model_.imgSize(), DataType<uchar>::type, Scalar(255));
//     vector< Point > &proj_pts = iter->second;
//     for( auto &p : proj_pts )
//       cout<<p;
//     cout<<endl;
//     cv_ext::drawPoints(dbg_img2, proj_pts, Scalar(0,0,255));
//     cv_ext::showImage(dbg_img2);
//   }
}

template <typename _T>
  float ChamferMatchingBase<_T>::templateDist(const _T &object_template ) const
{
  double avg_dist = 0;
  const std::vector<cv::Point> &proj_pts = object_template.proj_pts;
  for( int i = 0; i < static_cast<int>(proj_pts.size()); i++ )
#if CHAMFER_MATCHING_FAST_MAP
    avg_dist += fast_dist_map_.at<ushort>( proj_pts[i].y, proj_pts[i].x );
#else  
    avg_dist += dist_map_.at<float>( proj_pts[i].y, proj_pts[i].x );
#endif    
    

  if( proj_pts.size() )
    avg_dist /= proj_pts.size();
  else
    avg_dist = std::numeric_limits< float >::max();

  return float(avg_dist);
}

// void OrientedChamferMatching::setInput ( const Mat& dist_map, const cv::Mat &closest_dir_map )
// {
//   if( dist_map.type() != DataType<float>::type ||
//       dist_map.rows != cam_model_.imgHeight()||
//       dist_map.cols  != cam_model_.imgWidth() ||
//       closest_dir_map.type() != DataType<ushort>::type ||
//       closest_dir_map.rows != cam_model_.imgHeight()||
//       closest_dir_map.cols  != cam_model_.imgWidth() )
//     throw invalid_argument("Invalid input data");  
//   
//   dist_map_ =  dist_map;
//   closest_dir_map_ = closest_dir_map;
// 
//   #if CHAMFER_MATCHING_FAST_MAP
//   // TODO Switch each map to ushort type?
//   dist_map_.convertTo(fast_dist_map_, cv::DataType<ushort>::type);  
// #endif
// 
// }
// 
// void OrientedChamferMatching::match(int num_best_matches, vector< TemplateMatch >& matches, int image_step)
// {
//   const int n_threads = ( parallelism_enabled_ ? omp_get_max_threads() : 1 );
// 
//   vector< multimap< double, TemplateMatch > > best_matches(n_threads);
//   vector< float > min_dists( n_threads, std::numeric_limits< float >::max());
//   const int w = cam_model_.imgWidth(), h = cam_model_.imgHeight();
//   
// #if CHAMFER_MATCHING_FAST_MAP && ( defined(D2CO_USE_SSE) || defined(D2CO_USE_AVX) )
//   
//   // TODO
//   std::cerr<<"IMPLEMENT ME!!"<<std::endl;
// 
// #endif
// 
//   #pragma omp parallel for schedule(dynamic) if( parallelism_enabled_ )
//   for( int templ_i = 0; templ_i < static_cast<int>(tv_ptr_->size()); templ_i++ )
//   {
//     int th_id = omp_get_thread_num();
//     auto &th_best_matches = best_matches[th_id];
//     float &min_dist = min_dists[th_id];
//     Template &templ = tv_ptr_->at(templ_i);
//     
//     vector<Point> proj_pts = templ.proj_pts;
//     const vector<int> &i_dirs = templ.dir_idx;
// 
//     // No x,y exhaustive matching, just the templates in their original position
//     if( image_step < 1 )
//     {
//       float dist = templateDist(proj_pts, i_dirs);
// 
//       if( int(th_best_matches.size()) < num_best_matches )
//       {
//         th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist, Point(0,0))) );
//         min_dist = th_best_matches.rbegin()->first;
//       }
//       else if( dist < min_dist )
//       {
//         th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist, Point(0,0))) );
//         th_best_matches.erase(--th_best_matches.rbegin().base());
//         min_dist = th_best_matches.rbegin()->first;
//       }
//     }
//     else
//     {
//       Rect &bb = templ.bbox;
//       Point tl = bb.tl();
//       for( int i_p = 0; i_p < static_cast<int>(proj_pts.size()); i_p++ )
//         proj_pts[i_p] -= tl;
// 
//       int dists_w = (w - bb.width)/image_step + 1, dists_h = (h - bb.height)/image_step + 1;      
//       Mat_<float> avg_dists(dists_h, dists_w);
//       
// #if CHAMFER_MATCHING_FAST_MAP
// 
// #if defined(D2CO_USE_SSE)
//       
//   // TODO
//   std::cerr<<"IMPLEMENT ME!!"<<std::endl;
//   
// //       int aligned_c_dists_w = dists_w - (dists_w%8), c_dists_data_step = aligned_c_dists_w + ((dists_w%8)?8:0);
// // 
// //       // TODO Move this away (avoid to relocate for each template)?      
// //       void *c_dists_data = _mm_malloc (dists_h * c_dists_data_step * sizeof(ushort), 16);
// //       Mat c_dists(dists_h, dists_w, DataType<ushort>::type, c_dists_data, c_dists_data_step * sizeof(ushort));      
// //       memset(c_dists_data, 0, dists_h * c_dists_data_step * sizeof(ushort));
// //       
// //       __m128i map_dist128i, avg_dist128i, *c_dists128i_p, *dist_map128i_p;
// //       int vec_c_dists_w = aligned_c_dists_w/8;
// //       for( int i_p = 0; i_p < proj_pts.size(); i_p++ )
// //       {
// //         int x = proj_pts[i_p].x;
// //         Mat &dist_map = step_tensor[x%image_step][i_dirs[i_p]];
// // 
// //         for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
// //         {
// //           c_dists128i_p = (__m128i *)c_dists.ptr<ushort>(c_dists_y);
// //           dist_map128i_p = (__m128i *)(dist_map.ptr<ushort>(y) + x/image_step);  
// // 
// //           for(int vec_c_dists_x = 0; vec_c_dists_x < vec_c_dists_w; dist_map128i_p++, c_dists128i_p++, vec_c_dists_x++ )
// //           {
// //             map_dist128i = _mm_loadu_si128(dist_map128i_p);
// //             avg_dist128i = _mm_load_si128(c_dists128i_p);
// //             avg_dist128i = _mm_adds_epu16(avg_dist128i, map_dist128i);
// //             _mm_store_si128(c_dists128i_p, avg_dist128i );
// //           }
// //           ushort *dist_map_p = (ushort *)dist_map128i_p, *dists_p = (ushort *)c_dists128i_p;
// //           for( int c_dists_x = aligned_c_dists_w; c_dists_x < dists_w; 
// //                dist_map_p++, dists_p++, c_dists_x++ )
// //             *dists_p += *dist_map_p;
// //         }
// //       }
//       
// #else // defined(D2CO_USE_SSE)
// 
//       Mat c_dists( dists_h, dists_w, DataType<ushort>::type, Scalar(0));
//       for( int i_p = 0; i_p < static_cast<int>(proj_pts.size()); i_p++ )
//       {
//         for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
//         {
//           ushort *dists_p = c_dists.ptr<ushort>(c_dists_y);
//           ushort *dist_map_p = fast_dist_map_.ptr<ushort>(y) + proj_pts[i_p].x;
// 
//           for(int c_dists_x = 0; c_dists_x < dists_w; dist_map_p += image_step, dists_p++, c_dists_x++ )
//             *dists_p += *dist_map_p;
//         }
//       }
// 
//       for( int i_p = 0; i_p < static_cast<int>(proj_pts.size()); i_p++ )
//       {
//         ushort point_dir = i_dirs[i_p];
//         for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
//         {
//           ushort *dists_p = c_dists.ptr<ushort>(c_dists_y);
//           ushort *dir_map_p = closest_dir_map_.ptr<ushort>(y) + proj_pts[i_p].x;
// 
//           for(int c_dists_x = 0; c_dists_x < dists_w; dir_map_p += image_step, dists_p++, c_dists_x++ )
//           {
//             ushort ang_diff = (point_dir > *dir_map_p)?(point_dir - *dir_map_p):(*dir_map_p - point_dir);
//             *dists_p += ang_diff;
//           }
//         }
//       }
//       
// #endif
// // WARNING Commented out for debug
// //       float norm_factor = 1.0f/proj_pts.size();
// //       c_dists.convertTo(avg_dists, cv::DataType< float >::type, norm_factor);
//       
// #if defined(D2CO_USE_SSE)
//       // TODO Move from here to top
// // WARNING Commented out for debug      
// //       _mm_free(c_dists_data);
// #endif      
// 
// #else // CHAMFER_MATCHING_FAST_MAP
// 
//
//       avg_dists.setTo(Scalar(0));
//       for( int i_p = 0; i_p < proj_pts.size(); i_p++ )
//       {
//         int point_dir = i_dirs[i_p];
//         for( int y = proj_pts[i_p].y, avg_dists_y = 0; avg_dists_y < dists_h; y += image_step, avg_dists_y++ )
//         {
//           float *dists_p = avg_dists.ptr<float>(avg_dists_y);
//           
//           float *dist_map_p = dist_map_.ptr<float>(y) + proj_pts[i_p].x;
//           ushort *dir_map_p = closest_dir_map_.ptr<ushort>(y) + proj_pts[i_p].x;
// 
//           for(int avg_dists_x = 0; avg_dists_x < dists_w; dist_map_p += image_step, 
//               dir_map_p += image_step, dists_p++, avg_dists_x++ )
//           {
//             float ang_diff = point_dir - *dir_map_p;
//             *dists_p += *dist_map_p + (ang_diff<0)?-ang_diff:ang_diff;
//             }
//         }
//       }
// 
//       avg_dists *= 1.0f/proj_pts.size();
//       
// #endif
//       
//       for(int avg_dists_y = 0; avg_dists_y < dists_h; avg_dists_y++ )
//       {
//         float *dists_p = avg_dists.ptr<float>(avg_dists_y);
//         for(int avg_dists_x = 0; avg_dists_x < dists_w; avg_dists_x++, dists_p++ )
//         {
//           float dist = *dists_p;
//           if( int(th_best_matches.size()) < num_best_matches )
//           {
//             Point offset( avg_dists_x*image_step, avg_dists_y*image_step);
//             offset -= bb.tl();
// 
//             th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist, offset)) );
//             //             best_pts[dist] = proj_pts;
//             min_dist = th_best_matches.rbegin()->first;
//             //             cout<<"INSERTED"<<endl;
//           }
//           else if( dist < min_dist )
//           {
//             Point offset( avg_dists_x*image_step, avg_dists_y*image_step);
//             offset -= bb.tl();
//             th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist, offset)) );
//             //             best_pts[dist] = proj_pts;
//             th_best_matches.erase(--th_best_matches.rbegin().base());
//             //             best_pts.erase(--best_pts.rbegin().base());
//             min_dist = th_best_matches.rbegin()->first;
//             //             cout<<"INSERTED 2"<<endl;
//           }
//         }
//       }      
//     }
//   }
// 
// 
//   
// #if CHAMFER_MATCHING_FAST_MAP && ( defined(D2CO_USE_SSE) || defined(D2CO_USE_AVX) )
//   // TODO Move from here to top
// // WARNING Commented out for debug
// //    _mm_free(step_tensor_data);
// #endif
//   
//   for( int i = 1; i < n_threads; i++ )
//     best_matches[0].insert(best_matches[i].begin(), best_matches[i].end());
// 
//   matches.clear();
//   matches.reserve(num_best_matches);
//   int num_matches = 0;
//   for( auto iter = best_matches[0].begin();
//        iter != best_matches[0].end() && num_matches < num_best_matches;
//        iter++, num_matches++ )
// 
//     matches.push_back(iter->second);
// 
// //   for( auto iter = best_pts.begin(); iter != best_pts.end(); iter++ )
// //   {
// //     Mat dbg_img2 = dbg_img.clone();
// // //     Mat dbg_img(cam_model_.imgSize(), DataType<uchar>::type, Scalar(255));
// //     vector< Point > &proj_pts = iter->second;
// //     for( auto &p : proj_pts )
// //       cout<<p;
// //     cout<<endl;
// //     cv_ext::drawPoints(dbg_img2, proj_pts, Scalar(0,0,255));
// //     cv_ext::showImage(dbg_img2);
// //   }
// }
// 
// float OrientedChamferMatching::templateDist ( const vector< Point >& proj_pts, 
//                                               const vector< int >& dirs )
// {
//   std::cerr<<"IMPLEMENT ME!!"<<std::endl;
//   return -1;
// }

template <typename _T>
  void DirectionalChamferMatchingBase<_T>::setInput(const ImageTensorPtr& dist_map_tensor_ptr )
{       
  if( dist_map_tensor_ptr->at(0).type() != DataType<float>::type )
    throw invalid_argument("Invalid input data");  
  
  dist_map_tensor_ptr_ = dist_map_tensor_ptr;
  
#if CHAMFER_MATCHING_FAST_MAP
  int num_directions = dist_map_tensor_ptr->depth();
  // TODO Switch each map to ushort type?
  fast_dist_map_tensor_.create(dist_map_tensor_ptr->at(0).rows, dist_map_tensor_ptr->at(0).cols, 
                               num_directions, cv::DataType<ushort>::type);
  
  for( int i = 0; i < num_directions; i++ )
    dist_map_tensor_ptr_->at(i).convertTo(fast_dist_map_tensor_[i], cv::DataType<ushort>::type);
#endif
}

template <typename _T>
  void DirectionalChamferMatchingBase<_T>::match(int num_best_matches, vector<TemplateMatch >& matches,
                                                 int image_step, int match_cell_size ) const
{
  assert( dist_map_tensor_ptr_ != nullptr && dist_map_tensor_ptr_->depth() != 0 );
  
  const int n_threads = ( parallelism_enabled_ ? omp_get_max_threads() : 1 );

  const int w = dist_map_tensor_ptr_->at(0).cols, h = dist_map_tensor_ptr_->at(0).rows;
  vector< cv_ext::AlignedMat > avg_dists;
  avg_dists.reserve( n_threads );
  for( int i_th = 0; i_th < n_threads; i_th++ )
    avg_dists.emplace_back( h/image_step + 1, w/image_step + 1, DataType<float>::type );

  cv::Mat_<int> match_grid_map( h/image_step + 1, w/image_step + 1 );
  int match_grid_num_cells;

  if( match_cell_size > 0 )
  {
    if( match_cell_size < image_step )
      match_cell_size = image_step;
    int rel_match_grid_step = match_cell_size/image_step;
    int match_grid_w= match_grid_map.cols/rel_match_grid_step + (match_grid_map.cols%rel_match_grid_step ? 1 : 0),
        match_grid_h= match_grid_map.rows/rel_match_grid_step + (match_grid_map.cols%rel_match_grid_step ? 1 : 0);
    for( int r = 0; r < match_grid_map.rows; r++ )
    {
      int *mgm_p = match_grid_map.ptr<int>(r);
      int grid_r = r/rel_match_grid_step;
      for( int c = 0; c < match_grid_map.cols; c++ )
        *mgm_p++ = grid_r*match_grid_w + c/rel_match_grid_step;
    }
    match_grid_num_cells = match_grid_w*match_grid_h;
  }
  else
  {
    match_grid_map.setTo(Scalar(0));
    match_grid_num_cells = 1;
  }

  vector < vector< multimap< double, TemplateMatch > > > best_matches( n_threads );
  vector < vector< float > > min_dists( n_threads );

  for( int i_th = 0; i_th < n_threads; i_th++ )
  {
    best_matches[i_th].resize(match_grid_num_cells);
    min_dists[i_th].resize( match_grid_num_cells, std::numeric_limits< float >::max() );
  }

#if CHAMFER_MATCHING_FAST_MAP

  vector< cv_ext::AlignedMat > c_dists;
  c_dists.reserve( n_threads );
  for( int i_th = 0; i_th < n_threads; i_th++ )
    c_dists.emplace_back( h/image_step + 1, w/image_step + 1, DataType<ushort>::type );

#if ( defined(D2CO_USE_SSE) || defined(D2CO_USE_AVX) || defined(D2CO_USE_NEON) )

//   cv_ext::BasicTimer timer;
  // TODO Move this directly in conversion
  int num_dirs = dist_map_tensor_ptr_->depth();
  int step_tensor_w = cvCeil(static_cast<float>(w)/image_step), step_tensor_h = h;
  vector<ImageTensor> step_tensor(image_step);
  for( auto &tensor : step_tensor )
    tensor.create (step_tensor_h, step_tensor_w, num_dirs, cv::DataType<ushort>::type );

  // TODO Use vectorization
  for( int i = 0; i < num_dirs; i++ )
  {
    const Mat &orig_mat_tensor = fast_dist_map_tensor_.at(i);

    for( int y = 0; y < h; y++)
    {
      const ushort *orig_p = orig_mat_tensor.ptr<ushort>(y);
      vector <ushort *> step_p(image_step);
      for( int j = 0; j < image_step; j++ )
        step_p[j] = step_tensor[j][i].ptr<ushort>(y);
      
      for( int x = 0; x < w; x++, orig_p++)
        *step_p[x%image_step]++ = *orig_p;
    }
  }

//   cout<<"OBJ_REC_FAST_SSE data creation Elapsed time ms : "<<timer.elapsedTimeMs()<<endl;

#endif

#endif

  #pragma omp parallel for schedule(dynamic) if( parallelism_enabled_ )
  for( int templ_i = 0; templ_i < static_cast<int>(tv_ptr_->size()); templ_i++ )
  {
    int th_id = omp_get_thread_num();
    auto &th_best_matches = best_matches[th_id];
    auto &min_dist = min_dists[th_id];
    auto &th_avg_dists = avg_dists[th_id];
    auto &templ = tv_ptr_->at(templ_i);

    vector<Point> proj_pts = templ.proj_pts;
    const vector<int> &i_dirs = templ.dir_idx;

    // No x,y exhaustive matching, just the templates in their original position
    if( image_step < 1 )
    {
      float dist = templateDist(templ);
      int match_cell_idx = match_grid_map.at<int>(templ.bbox.y,templ.bbox.x);
      if( int(th_best_matches.size()) < num_best_matches )
      {
        th_best_matches[match_cell_idx].insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist, Point(0,0))) );
        min_dist[match_cell_idx] = th_best_matches[match_cell_idx].rbegin()->first;
      }
      else if( dist < min_dist[match_cell_idx] )
      {
        th_best_matches[match_cell_idx].insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist, Point(0,0))) );
        th_best_matches[match_cell_idx].erase(--th_best_matches[match_cell_idx].rbegin().base());
        min_dist[match_cell_idx] = th_best_matches[match_cell_idx].rbegin()->first;
      }
    }
    else
    {
      Rect &bb = templ.bbox;
      Point tl = bb.tl();
      for( int i_p = 0; i_p < static_cast<int>(proj_pts.size()); i_p++ )
        proj_pts[i_p] -= tl;

      const int dists_w = (w - bb.width)/image_step + 1, dists_h = (h - bb.height)/image_step + 1;

#if CHAMFER_MATCHING_FAST_MAP

      auto &c_dists_th = c_dists[th_id];
      c_dists_th(cv::Rect(0,0, dists_w, dists_h)).setTo(Scalar(0));

#if defined(D2CO_USE_AVX)

      __m256i cur_dist, acc_dist, *c_dists_p, *dist_map_p;
      int vec_c_dists_w = c_dists_th.step1()/16;
      for( int i_p = 0; i_p < static_cast<int>(proj_pts.size()); i_p++ )
      {
        int x = proj_pts[i_p].x;
        Mat &dist_map = step_tensor[x%image_step][i_dirs[i_p]];

        for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
        {
          c_dists_p = (__m256i *)c_dists_th.ptr<ushort>(c_dists_y);
          dist_map_p = (__m256i *)(dist_map.ptr<ushort>(y) + x/image_step);

          for(int vec_c_dists_x = 0; vec_c_dists_x < vec_c_dists_w; dist_map_p++, c_dists_p++, vec_c_dists_x++ )
          {
            cur_dist = _mm256_loadu_si256(dist_map_p);
            acc_dist = _mm256_load_si256(c_dists_p);
            acc_dist = _mm256_adds_epu16(acc_dist, cur_dist);
            _mm256_store_si256(c_dists_p, acc_dist );
          }
        }
      }

#elif defined(D2CO_USE_SSE)

      __m128i cur_dist, acc_dist, *c_dists_p, *dist_map_p;
      int vec_c_dists_w = c_dists_th.step1()/8;
      for( int i_p = 0; i_p < static_cast<int>(proj_pts.size()); i_p++ )
      {
        int x = proj_pts[i_p].x;
        Mat &dist_map = step_tensor[x%image_step][i_dirs[i_p]];

        for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
        {
          c_dists_p = (__m128i *)c_dists_th.ptr<ushort>(c_dists_y);
          dist_map_p = (__m128i *)(dist_map.ptr<ushort>(y) + x/image_step);  

          for(int vec_c_dists_x = 0; vec_c_dists_x < vec_c_dists_w; dist_map_p++, c_dists_p++, vec_c_dists_x++ )
          {
            cur_dist = _mm_loadu_si128(dist_map_p);
            acc_dist = _mm_load_si128(c_dists_p);
            acc_dist = _mm_adds_epu16(acc_dist, cur_dist);
            _mm_store_si128(c_dists_p, acc_dist );
          }
        }
      }

#elif defined(D2CO_USE_NEON)

      uint16x8_t cur_dist, acc_dist;
      uint16_t* c_dists_p, *dist_map_p;
      int vec_c_dists_w = c_dists_th.step1()/8;
      for( int i_p = 0; i_p < static_cast<int>(proj_pts.size()); i_p++ )
      {
        int x = proj_pts[i_p].x;
        Mat &dist_map = step_tensor[x%image_step][i_dirs[i_p]];

        for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
        {
          c_dists_p = (uint16_t *)c_dists_th.ptr<ushort>(c_dists_y);
          dist_map_p = (uint16_t *)(dist_map.ptr<ushort>(y) + x/image_step);  

          for(int vec_c_dists_x = 0; vec_c_dists_x < vec_c_dists_w; dist_map_p+=8, c_dists_p+=8, vec_c_dists_x++ )
          {
            cur_dist = vld1q_u16(dist_map_p); // Unaligned
            acc_dist = vld1q_u16(c_dists_p);
            acc_dist = vaddq_u16(acc_dist, cur_dist);
            vst1q_u16(c_dists_p, acc_dist );
          }
        }
      }
      
#else

      for( int i_p = 0; i_p < static_cast<int>(proj_pts.size()); i_p++ )
      {
        const Mat &dist_map = fast_dist_map_tensor_[i_dirs[i_p]];

        for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
        {
          ushort *dists_p = c_dists_th.ptr<ushort>(c_dists_y);
          const ushort *dist_map_p = dist_map.ptr<ushort>(y) + proj_pts[i_p].x;

          for(int c_dists_x = 0; c_dists_x < dists_w; dist_map_p += image_step, dists_p++, c_dists_x++ )
            *dists_p += *dist_map_p;
        }
      }
#endif

      float norm_factor = 1.0f/proj_pts.size();
      for (int avg_dists_y = 0; avg_dists_y < dists_h; avg_dists_y++)
      {
        const ushort *i_dists_p = c_dists_th.ptr<ushort>(avg_dists_y);
        float *o_dists_p = th_avg_dists.ptr<float>(avg_dists_y);
        for (int avg_dists_x = 0; avg_dists_x < dists_w; avg_dists_x++, i_dists_p++, o_dists_p++ )
          *o_dists_p = norm_factor*static_cast<float>(*i_dists_p);
      }
#else // CHAMFER_MATCHING_FAST_MAP

      th_avg_dists(cv::Rect(0,0, dists_w, dists_h)).setTo(Scalar(0));
      for( int i_p = 0; i_p < static_cast<int>(proj_pts.size()); i_p++ )
      {
        Mat &dist_map = (*dist_map_tensor_ptr_)[i_dirs[i_p]];

        for( int y = proj_pts[i_p].y, avg_dists_y = 0; avg_dists_y < dists_h; y += image_step, avg_dists_y++ )
        {
          float *dists_p = th_avg_dists.ptr<float>(avg_dists_y);
          const float *dist_map_p = dist_map.ptr<float>(y) + proj_pts[i_p].x;

          for(int avg_dists_x = 0; avg_dists_x < dists_w; dist_map_p += image_step, dists_p++, avg_dists_x++ )
            *dists_p += *dist_map_p;
        }
      }

      th_avg_dists *= 1.0f/proj_pts.size();
      
#endif
      
      for(int avg_dists_y = 0; avg_dists_y < dists_h; avg_dists_y++ )
      {
        const float *dists_p = th_avg_dists.ptr<float>(avg_dists_y);
        for(int avg_dists_x = 0; avg_dists_x < dists_w; avg_dists_x++, dists_p++ )
        {
          float dist = *dists_p;
          int match_cell_idx = match_grid_map.at<int>(avg_dists_y,avg_dists_x);
          if( int(th_best_matches[match_cell_idx].size()) < num_best_matches )
          {
            Point offset( avg_dists_x*image_step, avg_dists_y*image_step);
            offset -= bb.tl();

            th_best_matches[match_cell_idx].insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist, offset)) );
            min_dist[match_cell_idx] = th_best_matches[match_cell_idx].rbegin()->first;
          }
          else if( dist < min_dist[match_cell_idx] )
          {
            Point offset( avg_dists_x*image_step, avg_dists_y*image_step);
            offset -= bb.tl();
            th_best_matches[match_cell_idx].insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist, offset)) );
            th_best_matches[match_cell_idx].erase(--th_best_matches[match_cell_idx].rbegin().base());
            min_dist[match_cell_idx] = th_best_matches[match_cell_idx].rbegin()->first;
          }
        }
      }      
    }
  }
  
  for( int i_th = 1; i_th < n_threads; i_th++ )
    for( int j = 0; j < match_grid_num_cells; j++ )
      best_matches[0][j].insert(best_matches[i_th][j].begin(), best_matches[i_th][j].end());

  matches.clear();
  matches.reserve(num_best_matches*match_grid_num_cells);
  for( int j = 0; j < match_grid_num_cells; j++ )
  {
    int num_matches = 0;
    for( auto iter = best_matches[0][j].begin(); iter != best_matches[0][j].end() && num_matches < num_best_matches;
         iter++, num_matches++ )
      matches.push_back(iter->second);
  }
}

template <typename _T>
  float DirectionalChamferMatchingBase<_T>::templateDist(const _T &object_template ) const
{
  double avg_dist = 0;
  const std::vector<cv::Point> &proj_pts = object_template.proj_pts;
  const std::vector<int> &i_dirs = object_template.dir_idx;
  
#if CHAMFER_MATCHING_FAST_MAP
  for( int i = 0; i < static_cast<int>(proj_pts.size()); i++ )  
    avg_dist += fast_dist_map_tensor_[i_dirs[i]].at<ushort>( proj_pts[i].y, proj_pts[i].x );
#else
  ImageTensor &dist_map_tensor = *dist_map_tensor_ptr_;
  for( int i = 0; i < static_cast<int>(proj_pts.size()); i++ )
    avg_dist += dist_map_tensor[i_dirs[i]].at<float>( proj_pts[i].y, proj_pts[i].x );
#endif

  if( proj_pts.size() )
    avg_dist /= proj_pts.size();
  else
    avg_dist = std::numeric_limits< float >::max();

  return float(avg_dist);
}
