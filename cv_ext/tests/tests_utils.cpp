#include "tests_utils.h"

#include "cv_ext/pinhole_scene_projector.h"
#include "cv_ext/drawing.h"
#include "cv_ext/image_inspection.h"

bool identicalMats( const cv::Mat &m1, const cv::Mat &m2 )
{
  if( m1.size() != m2.size() || m1.type() != m2.type() )
    return false;

  cv::Mat cmp_m;
  cv::compare(m1,m2,cmp_m, cv::CMP_NE);
  return cv::countNonZero(cmp_m) == 0;
}

bool quasiIdenticalMats( const cv::Mat &m1, const cv::Mat &m2, double epsilon )
{
  if( m1.size() != m2.size() || m1.type() != m2.type() || m1.channels() != 1 )
    return false;

  cv::Mat diff_m;
  cv::Mat d_m1, d_m2;
  m1.convertTo(d_m1, cv::DataType<double>::type);
  m2.convertTo(d_m2, cv::DataType<double>::type);
  cv::absdiff( d_m1,d_m2,diff_m );
  return cv::countNonZero( diff_m > epsilon ) == 0;
}

cv_ext::PinholeCameraModel sampleCameraModel( cv::Size img_size, bool no_distortion )
{
  if( no_distortion )
  {
    return cv_ext::PinholeCameraModel( img_size.width + cv_ext::sampleGaussian( 0, 5 ),
                                       img_size.width + cv_ext::sampleGaussian( 0, 5 ),
                                       img_size.width / 2 + cv_ext::sampleGaussian( 0, 5 ),
                                       img_size.height / 2 + cv_ext::sampleGaussian( 0, 5 ),
                                       img_size.width, img_size.height, 0, 0, 0, 0, 0 );
  }
  else
  {
    return cv_ext::PinholeCameraModel( img_size.width + cv_ext::sampleGaussian( 0, 5 ),
                                       img_size.width + cv_ext::sampleGaussian( 0, 5 ),
                                       img_size.width / 2 + cv_ext::sampleGaussian( 0, 5 ),
                                       img_size.height / 2 + cv_ext::sampleGaussian( 0, 5 ),
                                       img_size.width, img_size.height,
                                       cv_ext::sampleGaussian( 0, 0.1 ),
                                       cv_ext::sampleGaussian( 0, 0.1 ),
                                       cv_ext::sampleGaussian( 0, 0.001 ),
                                       cv_ext::sampleGaussian( 0, 0.001 ),
                                       cv_ext::sampleGaussian( 0, 0.001 ) );
  }
}


std::vector< cv::Point2f > generateCheckerboardImage( cv::Mat &cb_img, const cv_ext::PinholeCameraModel &cam_model,
                                                      const cv::Size &board_size, float square_len,
                                                      const cv::Mat_<double> &r_vec, const cv::Mat_<double> &t_vec,
                                                      cv::Mat pattern_mask, bool create_new_image )
{
  cv::Size actual_size = board_size;
  actual_size.width += 2;
  actual_size.height += 2;
  int num_corners = actual_size.width*actual_size.height;
  std::vector<cv::Point3f> corners(num_corners);
  float x, y = -square_len*(actual_size.height - 1)/2;
  for( int r = 0; r < actual_size.height; r++, y += square_len )
  {
    x = -square_len*(actual_size.width - 1)/2;
    for( int c = 0; c < actual_size.width; c++, x += square_len )
    {
      int idx = r*actual_size.width + c;
      corners[idx].x = x;
      corners[idx].y = y;
      corners[idx].z = 0;
    }
  }

  if( create_new_image || cb_img.empty() || cb_img.size() != cam_model.imgSize() ||
      cb_img.type() != cv::DataType<uchar>::type )
  {
    cb_img.create(cam_model.imgSize(), cv::DataType<uchar>::type);
    cb_img.setTo(cv::Scalar(128));
  }

  std::vector<cv::Point2f> proj_corners;
  cv_ext::PinholeSceneProjector scene_proj( cam_model );
  scene_proj.setTransformation(r_vec, t_vec);
  scene_proj.projectPoints(corners, proj_corners);

  cv::Mat_<double> r_mat(3,3);
  cv::Mat_<double> z_vec = (cv::Mat_<double>(3,1) << 0,0,1.0);
  cv_ext::angleAxis2RotMat<double>( r_vec, r_mat );
  double dot_p = z_vec.dot(r_mat.col(2));

  for( int r = 0; r < actual_size.height - 1; r++ )
  {
    for( int c = 0; c < actual_size.width  - 1; c++ )
    {
      cv::Scalar color = ( dot_p <= 0.1 || (r+c)%2 )?255:0;
      int idx[4] = { r*actual_size.width + c, r*actual_size.width + c+1,
                     (r+1)*actual_size.width + c+1, (r+1)*actual_size.width + c };
      std::vector< cv::Point > poly_pts(4);
      for( int i = 0; i < 4; i++ )
        poly_pts[i] = cv::Point(cvRound(proj_corners[idx[i]].x), cvRound(proj_corners[idx[i]].y));

      cv::fillConvexPoly(cb_img, poly_pts, color );
    }
  }

  if( !pattern_mask.empty() )
  {
    corners.resize(4);
    float orig_x = -square_len*(actual_size.width - 1)/2 + 4*square_len/3,
          orig_y = -square_len*(actual_size.height - 1)/2 + 4*square_len/3,
          offset_x = 0, offset_y = 0;

    corners[0].x = orig_x; corners[0].y = orig_y;
    corners[1].x = orig_x + square_len/3; corners[1].y = orig_y;
    corners[2].x = orig_x + square_len/3; corners[2].y = orig_y + square_len/3;
    corners[3].x = orig_x; corners[3].y = orig_y + square_len/3;

    cv::Point3f pattern_offset(0,0,0);

    std::vector<cv::Point3f> pattern_corners(4);
    for (int r = 0; r < pattern_mask.rows; r++, pattern_offset.y += square_len )
    {
      pattern_offset.x = 0;
      const uchar *m = pattern_mask.ptr<const uchar>(r);
      for (int c = 0; c < pattern_mask.cols; c++, m++, pattern_offset.x += square_len )
      {
        if(*m)
        {
          for( int i = 0; i < 4; i++ )
            pattern_corners[i] = corners[i] + pattern_offset;

          scene_proj.projectPoints(pattern_corners, proj_corners);

          std::vector< cv::Point > poly_pts(4);
          for( int i = 0; i < 4; i++ )
            poly_pts[i] = cv::Point(cvRound(proj_corners[i].x), cvRound(proj_corners[i].y));

          cv::fillConvexPoly(cb_img, poly_pts, (*m == 1)?cv::Scalar(255):cv::Scalar(0) );
        }
      }
    }
  }

  std::vector<cv::Point2f> proj_internal_corners;
  if ( dot_p > 0.1 )
  {
    // Check that all the projected corners are inside the image
    bool good_corners = true;
    int idx[4] = { actual_size.width + 1, actual_size.width + actual_size.width - 2,
                   (actual_size.height - 2) * actual_size.width + 1,
                   (actual_size.height - 2) * actual_size.width + actual_size.width - 2 };

    for( int i = 0; i < 4; i++)
    {
      auto p = proj_corners[idx[i]];
      if(p.x < 0 || p.y < 0 || p.x > cam_model.imgWidth() - 1 || p.y > cam_model.imgHeight() - 1)
      {
        good_corners = false;
        break;
      }
    }

    if( good_corners )
    {
      proj_internal_corners.reserve(board_size.width*board_size.height);
      for( int r = 1; r < actual_size.height - 1; r++ )
      {
        for (int c = 1; c < actual_size.width - 1; c++)
        {
          int i = r * actual_size.width + c;
          proj_internal_corners.push_back(proj_corners[i]);
        }
      }
    }
  }

  return proj_internal_corners;
}