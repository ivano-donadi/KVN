#include <stdexcept>
#include <cmath>

#include "cv_ext/image_skeleton.h"

static void shrinkIntersections( const cv::Mat &skeleton, cv::Mat &dst )
{
  cv::Mat mask = cv::Mat::zeros(skeleton.size(), cv::DataType<uchar>::type);//skeleton.clone();
  dst = skeleton.clone();
  
  for( int y = 1; y < skeleton.rows - 1; y++)
  {
    /* Image patch:
    * p9 p2 p3
    * p8 p1 p4
    * p7 p6 p5
    */
    const uchar *s1_p, *s2_p, *s3_p, *s4_p, *s5_p, *s6_p, *s7_p, *s8_p, *s9_p;
    const uchar *src_row_up_p = skeleton.ptr<uchar>(y-1), 
                *src_row_p = skeleton.ptr<uchar>(y), 
                *src_row_down_p = skeleton.ptr<uchar>(y+1);
    
    uchar *m1_p, *m2_p, *m3_p, *m4_p, *m5_p, *m6_p, *m7_p, *m8_p, *m9_p;
    uchar *mask_row_up_p = mask.ptr<uchar>(y-1), 
          *mask_row_p = mask.ptr<uchar>(y), 
          *mask_row_down_p = mask.ptr<uchar>(y+1);
                
    
    s2_p = src_row_up_p++;   
    s1_p = src_row_p++;      
    s6_p = src_row_down_p++; 
    
    s3_p = src_row_up_p++;   
    s4_p = src_row_p++;      
    s5_p = src_row_down_p++; 
    
    m2_p = mask_row_up_p++;   
    m1_p = mask_row_p++;      
    m6_p = mask_row_down_p++; 
    
    m3_p = mask_row_up_p++;   
    m4_p = mask_row_p++;      
    m5_p = mask_row_down_p++; 
    
     
    for( int x = 1; x < skeleton.cols - 1; x++ )
    {  
      s9_p = s2_p; 
      s8_p = s1_p; 
      s7_p = s6_p; 
      
      s2_p = s3_p; 
      s1_p = s4_p; 
      s6_p = s5_p; 
      
      s3_p = src_row_up_p++;
      s4_p = src_row_p++;
      s5_p = src_row_down_p++;
      
      m9_p = m2_p; 
      m8_p = m1_p; 
      m7_p = m6_p; 
      
      m2_p = m3_p; 
      m1_p = m4_p; 
      m6_p = m5_p; 
      
      m3_p = mask_row_up_p++;
      m4_p = mask_row_p++;
      m5_p = mask_row_down_p++;
      
      if( *s1_p )
      {
        if( *s2_p ) (*m2_p)++;
        if( *s3_p ) (*m3_p)++;
        if( *s4_p ) (*m4_p)++;
        if( *s5_p ) (*m5_p)++;
        if( *s6_p ) (*m6_p)++;
        if( *s7_p ) (*m7_p)++;
        if( *s8_p ) (*m8_p)++;
        if( *s9_p ) (*m9_p)++;
      }
    }
  }
  
  for( int y = 1; y < skeleton.rows - 1; y++)
  {
    // WARNING Unused s_p pointer
    const uchar *s_p = skeleton.ptr<uchar>(y);
    uchar *m_p = mask.ptr<uchar>(y);
    uchar *d_p = dst.ptr<uchar>(y);
    s_p++;
    m_p++;
    d_p++;
    
    for( int x = 1; x < skeleton.cols - 1; x++, s_p++, m_p++, d_p++ )
    {  
      if( *m_p >= 5 )
      {
        if( mask.at<uchar>( y, x - 1) >= 5 || mask.at<uchar>( y, x + 1) >= 5 ||
            mask.at<uchar>( y - 1, x) >= 5 || mask.at<uchar>( y + 1, x) >= 5 ||
            mask.at<uchar>( y - 1, x - 1) >= 5 || mask.at<uchar>( y + 1, x + 1) >= 5 ||
            mask.at<uchar>( y - 1, x + 1) >= 5 || mask.at<uchar>( y + 1, x - 1) >= 5 )
        {
          *d_p = 0;
          *m_p = 0;

          mask.at<uchar>( y, x - 1) = 0; 
          mask.at<uchar>( y, x + 1) = 0;
          mask.at<uchar>( y - 1, x) = 0;
          mask.at<uchar>( y + 1, x) = 0;
          mask.at<uchar>( y - 1, x - 1) = 0;
          mask.at<uchar>( y + 1, x + 1) = 0;
          mask.at<uchar>( y - 1, x + 1) = 0;
          mask.at<uchar>( y + 1, x - 1) = 0;           
        }
      }
    }
  } 
}

static void findNeighborNodes( int src_node_idx, int x, int y, cv::Mat &mask, cv::Mat &index_mat,
                               std::vector<cv::Point2f> &nodes, 
                               std::vector< std::vector<int> > &edges )
{
  if( !mask.at<uchar>(y,x) )
    return;
    
  mask.at<uchar>(y,x) = 0;
  //bool neighbor_node = false;
  for( int tx = std::max<int>(0, x - 1); tx <= std::min<int>(mask.cols, x + 1); tx++)
  {
    for( int ty = std::max<int>(0, y - 1); ty <= std::min<int>(mask.rows, y + 1); ty++)
    {
      if( (tx != x || ty != y ) && mask.at<uchar>(ty,tx))
      {
        int neighbor_idx = index_mat.at<int32_t>(ty, tx);
        if( neighbor_idx >= 0 && neighbor_idx != src_node_idx )
        {
          bool add_neighbor = true;
          std::vector<int> &neighbors = edges[src_node_idx];
          for( int i = 0; i < int(neighbors.size()); i++)
            if(neighbors[i] == neighbor_idx)
              add_neighbor = false;

          if( add_neighbor )
          {
            edges[src_node_idx].push_back(neighbor_idx);
            edges[neighbor_idx].push_back(src_node_idx);
          }
          return;
        }
      }
    }
  }

  for( int tx = std::max<int>(0, x - 1); tx <= std::min<int>(mask.cols, x + 1); tx++)
    for( int ty = std::max<int>(0, y - 1); ty <= std::min<int>(mask.rows, y + 1); ty++)
      if( (tx != x || ty != y ) && mask.at<uchar>(ty,tx))
        findNeighborNodes(src_node_idx, tx, ty, mask, index_mat, nodes, edges);
}

void cv_ext::morphThinning( const cv::Mat& src, cv::Mat& dst, bool binarize, uchar thresh )
{
  if( src.depth() != cv::DataType<uchar>::type || src.channels() != 1 ||
    !src.rows || !src.cols )
    throw std::invalid_argument("Unsopported input image");
  
  if( &src == &dst )
    throw std::invalid_argument("Unsopported in-place call: using input image as destination image: ");

  cv::Mat bin_img;
  
  if( binarize )
  {
    bin_img = cv::Mat(src.size(), cv::DataType<uchar>::type); 
    cv::threshold(src, bin_img, double(thresh), 1, cv::THRESH_BINARY);
  }
  else
    bin_img = src;
  
  dst = bin_img.clone(); 
    
  int n_deleted;
  do
  {
    n_deleted = 0;
    for( int iter = 0; iter < 2; iter++)
    {
      for( int y = 1; y < bin_img.rows - 1; y++)
      {
        /* Image patch:
        * p9 p2 p3
        * p8 p1 p4
        * p7 p6 p5
        */
        uchar p1, p2, p3, p4, p5, p6, p7, p8, p9;
        const uchar *row_up_p = bin_img.ptr<uchar>(y-1), 
                    *row_p = bin_img.ptr<uchar>(y), 
                    *row_down_p = bin_img.ptr<uchar>(y+1);
              
        uchar *dst_p = dst.ptr<uchar>(y);
        dst_p++;
        
        p2 = *row_up_p++;
        p1 = *row_p++;
        p6 = *row_down_p++;
        
        p3 = *row_up_p++;
        p4 = *row_p++;
        p5 = *row_down_p++;
        
        for( int x = 1; x < bin_img.cols - 1; x++, dst_p++ )
        {  
          p9 = p2;
          p8 = p1;
          p7 = p6;
          
          p2 = p3;
          p1 = p4;
          p6 = p5;
          
          p3 = *row_up_p++;
          p4 = *row_p++;
          p5 = *row_down_p++;
          
          if( p1 )
          {
            int patterns_01  = (p2 == 0 && p3 ) + (p3 == 0 && p4 ) + 
                               (p4 == 0 && p5 ) + (p5 == 0 && p6 ) + 
                               (p6 == 0 && p7 ) + (p7 == 0 && p8 ) +
                               (p8 == 0 && p9 ) + (p9 == 0 && p2 );
                               
            int nonzero_neighbors  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int cond_1 = (iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8));
            int cond_2 = (iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8));
            
            if (patterns_01 == 1 && (nonzero_neighbors >= 2 && nonzero_neighbors <= 6) && 
                cond_1 == 0 && cond_2 == 0)
            {
              *dst_p = 0;
              n_deleted++;
            }
          }
        }
      }
      dst.copyTo(bin_img);
    }
  }
  while( n_deleted );
  
  dst *= 255;
}


void cv_ext::graphExtraction( const cv::Mat &skeleton, std::vector<cv::Point2f> &nodes,
                              std::vector< std::vector<int> > &edges,
                              bool find_leafs, int min_dist )
{
  if( skeleton.depth() != cv::DataType<uchar>::type || skeleton.channels() != 1 ||
    !skeleton.rows || !skeleton.cols )
    throw std::invalid_argument("Unsopported input image");
  
  if( min_dist < 1)
  {
    std::cout<<"graphExtraction() : Warning! min_dist < 1, using default value 1"<<std::endl; 
    min_dist = 1;
  }

  cv::Mat index_mat = cv::Mat(skeleton.size(), cv::DataType<int>::type, cv::Scalar(-1)),
          mask(skeleton.size(), cv::DataType<uchar>::type), tmp_skel;

  shrinkIntersections( skeleton, tmp_skel );
  
  // Extract nodes
  nodes.clear();
  std::vector<float> n_nodes;
  
  for( int y = 1; y < tmp_skel.rows - 1; y++)
  {
    /* Image patch:
    * p9 p2 p3
    * p8 p1 p4
    * p7 p6 p5
    */
    
    const uchar *s1_p, *s2_p, *s3_p, *s4_p, *s5_p, *s6_p, *s7_p, *s8_p, *s9_p;
    const uchar *src_row_up_p = tmp_skel.ptr<uchar>(y-1), 
                *src_row_p = tmp_skel.ptr<uchar>(y), 
                *src_row_down_p = tmp_skel.ptr<uchar>(y+1);
    
    int32_t *index_mat_p = index_mat.ptr<int32_t>(y);
    index_mat_p++;
    
    s2_p = src_row_up_p++;   
    s1_p = src_row_p++;      
    s6_p = src_row_down_p++; 
    
    s3_p = src_row_up_p++;   
    s4_p = src_row_p++;      
    s5_p = src_row_down_p++; 
     
    for( int x = 1; x < tmp_skel.cols - 1; x++, index_mat_p++ )
    {  
      s9_p = s2_p; 
      s8_p = s1_p; 
      s7_p = s6_p; 
      
      s2_p = s3_p; 
      s1_p = s4_p; 
      s6_p = s5_p; 
      
      s3_p = src_row_up_p++;
      s4_p = src_row_p++;
      s5_p = src_row_down_p++;
          
      if( *s1_p )
      {
        int n_patterns;
        if( *s2_p)
        {
          n_patterns  = (*s2_p && *s3_p == 0) + (*s3_p && *s4_p == 0) + 
                        (*s4_p && *s5_p == 0) + (*s5_p && *s6_p == 0) + 
                        (*s6_p && *s7_p == 0) + (*s7_p && *s8_p == 0) +
                        (*s8_p && *s9_p == 0);
        }
        else
        {
          n_patterns  = (*s2_p == 0 && *s3_p) + (*s3_p == 0 && *s4_p) + 
                        (*s4_p == 0 && *s5_p) + (*s5_p == 0 && *s6_p) + 
                        (*s6_p == 0 && *s7_p) + (*s7_p == 0 && *s8_p) +
                        (*s8_p == 0 && *s9_p);  
        }
        if( n_patterns >= 3 || (find_leafs && n_patterns == 1 ))
        {
          bool add_node = true;
          int node_index;
          // Ensure minimum distance between nodes
          for( int tx = std::max<int>(0, x - min_dist); tx <= std::min<int>(tmp_skel.cols, x + min_dist); tx++)
            for( int ty = std::max<int>(0, y - min_dist); ty <= std::min<int>(tmp_skel.rows, y + min_dist); ty++)
              if( (tx != x || ty != y ) && (node_index = index_mat.at<int32_t>(ty,tx)) >= 0)
              {
                *index_mat_p = node_index;
                nodes[node_index] += cv::Point2f(tx,ty);
                n_nodes[node_index] += 1.0f;
                add_node = false;
              }
          
          if( add_node )
          {
            *index_mat_p = nodes.size();
            nodes.push_back(cv::Point2f(x,y));
            n_nodes.push_back(1.0f);
          }
        }
      }
    }
  }

  for( int i = 0; i < int(nodes.size()); i++)
  {
    nodes[i].x /= n_nodes[i];
    nodes[i].y /= n_nodes[i];
  }
  
  // Extract topology (edges)
  mask = tmp_skel.clone();
  edges.clear();
  edges.resize(nodes.size());
    
  for( int i = 0; i < int(nodes.size()); i++)
  {
    int x = cvRound(nodes[i].x), y = cvRound(nodes[i].y);
    findNeighborNodes(i, x, y, mask, index_mat, nodes, edges);
  }
}
