#include "scoring.h"

#include <limits>

using namespace std;
using namespace cv_ext;

double GradientDirectionScore::evaluate ( ImageGradient &im_grad,
                                          vector<cv::Point2f> &im_pts,
                                          const vector<float> &normal_directions ) const
{
  vector<float> g_dir =  im_grad.getGradientDirections ( im_pts );
//  vector<float> g_dir =  im_grad.getEigenDirections ( im_pts );

//  cv_ext::showImage(im_grad.getGradientDirections (), "getGradientDirections", true, 10);
//  cv_ext::showImage(im_grad.getGradientMagnitudes(), "getGradientMagnitudes", true, 10);

  vector<float> g_mag =  im_grad.getGradientMagnitudes ( im_pts, 0, INTERP_BILINEAR );

//  cv::Mat eigen_mat;
//  vector<float> g_dir;
//  g_dir.reserve(g_mag.size());
//  cv::cornerEigenValsAndVecs(im_grad.getIntensities(), eigen_mat, 5, 3 );
//
//  for( auto & p : im_pts )
//  {
//    int x = cvRound(p.x), y = cvRound(p.y);
//    // The first eigenvalue is always greater than or equal to the second one
//    if( (eigen_mat.at<cv::Vec6f>(y,x))[2] != 0.0f )
//      g_dir.push_back(atan((eigen_mat.at<cv::Vec6f>(y,x))[3]/(eigen_mat.at<cv::Vec6f>(y,x))[2]));
//    else
//      g_dir.push_back(M_PI/2);
//  }


  if ( !g_dir.size() || !g_mag.size() )
    return 0;

//  cv::Mat display = im_grad.getIntensities().clone();
//  cv_ext::drawPoints(display,im_pts);
//  int n_outlier = 0;

  double score = 0;
  for ( int i = 0; i < int(g_dir.size()); i++ )
  {
    float &direction = g_dir[i], magnitude = g_mag[i];
    if ( direction == im_grad.OUT_OF_IMG_VAL )
      score += OUTER_POINT_SCORE;
    else if (magnitude > grad_mag_thresh_)
      score += abs ( cos ( static_cast<double>( direction ) - normal_directions[i] ) );
//    else
//    {
//      display.at<float>(im_pts[i].y, im_pts[i].x) = 0;
//      n_outlier++;
//    }
  }

//  std::cout<<"Score : "<<score/g_dir.size()<<" outliers "<<n_outlier<<" over "<<g_dir.size()<<std::endl;
//  cv_ext::showImage(display);

  return score/g_dir.size();
}
