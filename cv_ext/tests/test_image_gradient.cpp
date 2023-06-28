#include <gtest/gtest.h>

#include "cv_ext/image_gradient.h"
#include "cv_ext/memory.h"
#include "cv_ext/interpolations.h"
#include "cv_ext/timer.h"

#include "tests_utils.h"

#include <opencv2/core/hal/hal.hpp>

using namespace cv_ext;

// From OpenCV corner.cpp source
static void eigen2x2( const float* cov, float* dst, int n )
{
  for( int j = 0; j < n; j++ )
  {
    double a = cov[j*3];
    double b = cov[j*3+1];
    double c = cov[j*3+2];

    double u = (a + c)*0.5;
    double v = std::sqrt((a - c)*(a - c)*0.25 + b*b);
    double l1 = u + v;
    double l2 = u - v;

    double x = b;
    double y = l1 - a;
    double e = fabs(x);

    if( e + fabs(y) < 1e-4 )
    {
      y = b;
      x = l1 - c;
      e = fabs(x);
      if( e + fabs(y) < 1e-4 )
      {
        e = 1./(e + fabs(y) + FLT_EPSILON);
        x *= e, y *= e;
      }
    }

    double d = 1./std::sqrt(x*x + y*y + DBL_EPSILON);
    dst[6*j] = (float)l1;
    dst[6*j + 2] = (float)(x*d);
    dst[6*j + 3] = (float)(y*d);

    x = b;
    y = l2 - a;
    e = fabs(x);

    if( e + fabs(y) < 1e-4 )
    {
      y = b;
      x = l2 - c;
      e = fabs(x);
      if( e + fabs(y) < 1e-4 )
      {
        e = 1./(e + fabs(y) + FLT_EPSILON);
        x *= e, y *= e;
      }
    }

    d = 1./std::sqrt(x*x + y*y + DBL_EPSILON);
    dst[6*j + 1] = (float)l2;
    dst[6*j + 4] = (float)(x*d);
    dst[6*j + 5] = (float)(y*d);
  }
}

// From OpenCV corner.cpp source
static void calcEigenValsVecs( const cv::Mat& _cov, cv::Mat& _dst )
{
  cv::Size size = _cov.size();
  if( _cov.isContinuous() && _dst.isContinuous() )
  {
    size.width *= size.height;
    size.height = 1;
  }

  for( int i = 0; i < size.height; i++ )
  {
    const float* cov = _cov.ptr<float>(i);
    float* dst = _dst.ptr<float>(i);

    eigen2x2(cov, dst, size.width);
  }
}

template < MemoryAlignment _align > static void testAlignment()
{
  const int num_levels = 5, vec_size = 3;
  cv::Mat m1(cv::Size(317,199), cv::DataType<ushort>::type);
  fillRandom<ushort>(m1);

  cv_ext::ImageGradientBase<_align> ig1( m1, num_levels );
  cv::Mat test_m[5];
  for( int i = 0; i < num_levels; i++ )
  {
    test_m[0] = ig1.getIntensities( i );
    test_m[1] = ig1.getGradientX( i );
    test_m[2] = ig1.getGradientY( i );
    test_m[3] = ig1.getGradientDirections( i );
    test_m[4] = ig1.getGradientMagnitudes( i );

    for( auto &m : test_m )
    {
      ASSERT_TRUE(CV_EXT_IS_ALIGNED(m.data, _align));
      ASSERT_EQ(m.step[0]%_align,0);
      ASSERT_TRUE(CV_EXT_IS_ALIGNED(m.data, _align));
      ASSERT_EQ(m.step[0]%_align,0);
    }
  }

  cv::Mat tmp_m = ig1.getIntensities( 0 );
  cv_ext::ImageGradientBase<_align> ig2;

  ig2.create( tmp_m, num_levels, 1.33, false );

  for( int i = 0; i < num_levels; i++ )
  {
    test_m[0] = ig2.getIntensities( i );
    test_m[1] = ig2.getGradientX( i );
    test_m[2] = ig2.getGradientY( i );
    test_m[3] = ig2.getGradientDirections( i );
    test_m[4] = ig2.getGradientMagnitudes( i );

    for( auto &m : test_m )
    {
      ASSERT_TRUE(CV_EXT_IS_ALIGNED(m.data, _align));
      ASSERT_EQ(m.step[0]%_align,0);
      ASSERT_TRUE(CV_EXT_IS_ALIGNED(m.data, _align));
      ASSERT_EQ(m.step[0]%_align,0);
    }
  }

  std::vector< cv_ext::ImageGradientBase<_align> > ig_vec;
  ig_vec.reserve(vec_size);
  for( int i = 0; i < vec_size; i++ )
    ig_vec.emplace_back( m1, num_levels );

  for( int i = 0; i < vec_size; i++ )
  {
    for( int j = 0; j < num_levels; j++ )
    {
      test_m[0] = ig_vec[i].getIntensities( j );
      test_m[1] = ig_vec[i].getGradientX( j );
      test_m[2] = ig_vec[i].getGradientY( j );
      test_m[3] = ig_vec[i].getGradientDirections( j );
      test_m[4] = ig_vec[i].getGradientMagnitudes( j );

      for( auto &m : test_m )
      {
        ASSERT_TRUE(CV_EXT_IS_ALIGNED(m.data, _align));
        ASSERT_EQ(m.step[0]%_align,0);
        ASSERT_TRUE(CV_EXT_IS_ALIGNED(m.data, _align));
        ASSERT_EQ(m.step[0]%_align,0);
      }
    }
  }
}


template < typename _T > static void testOperators()
{
  const cv::Size im_size(3840, 2160);
  cv::Mat input_m(im_size, cv::DataType<_T>::type);
  fillRandom<_T>(input_m);
  ImageGradient ig;
  ig.create( input_m );
  ig.enableScharrOperator(false);
  ig.enableFastMagnitude(true);
  cv::Mat_<float> m, dx,dy;
  input_m.convertTo(m, CV_32F );
  cv::Sobel(m,dx,CV_32F,1,0);
  cv::Sobel(m,dy,CV_32F,0,1);

  ASSERT_TRUE(identicalMats(ig.getGradientX(),dx));
  ASSERT_TRUE(identicalMats(ig.getGradientY(),dy));

  ig.create( m );
  ig.enableScharrOperator(true);

  cv::Scharr(m,dx,CV_32F,1,0);
  cv::Scharr(m,dy,CV_32F,0,1);

  ASSERT_TRUE(identicalMats(ig.getGradientX(),dx));
  ASSERT_TRUE(identicalMats(ig.getGradientY(),dy));

  cv::Mat_<float> grad_o, grad_d;
  grad_o.create( im_size );
  for( int r = 0; r < im_size.height; r++ )
  {
    const float *dx_p =  dx.ptr<const float>(r), *dy_p =  dy.ptr<const float>(r);
    float *grad_o_p =  grad_o.ptr<float>(r);
    for( int c = 0; c < im_size.width; c++, dx_p++, dy_p++, grad_o_p++ )
    {
      if( *dx_p || *dy_p )
        *grad_o_p = atan2(*dy_p, *dx_p );
      else
        *grad_o_p = 0;
    }
  }

  cv_ext::BasicTimer timer;
  timer.reset();
  const cv::Mat &ig_grad_o = ig.getGradientOrientations();
  GTEST_COUT<< "Elapsed time getGradientOrientations() : " << timer.elapsedTimeMs()<< " ms" << std::endl;

  ASSERT_TRUE(quasiIdenticalMats(ig_grad_o,grad_o, 5e-3 ));

  grad_d.create( im_size );
  for( int r = 0; r < im_size.height; r++ )
  {
    const float *dx_p =  dx.ptr<const float>(r), *dy_p =  dy.ptr<const float>(r);
    float *grad_d_p = grad_d.ptr<float>(r);
    for( int c = 0; c < im_size.width; c++, dx_p++, dy_p++, grad_d_p++ )
    {
      if( *dx_p )
        *grad_d_p = atan( *dy_p / *dx_p );
      else if( *dy_p > 0 )
        *grad_d_p = M_PI/2;
      else if( *dy_p < 0 )
        *grad_d_p = -M_PI/2;
      else
        *grad_d_p = 0;
    }
  }

  timer.reset();
  const cv::Mat &ig_grad_d = ig.getGradientDirections();
  GTEST_COUT<< "Elapsed time getGradientDirections() : " << timer.elapsedTimeMs()<< " ms" << std::endl;

  ASSERT_TRUE(quasiIdenticalMats(ig_grad_d,grad_d, 5e-3 ));

  cv::Mat cov( im_size, CV_32FC3 ), eigenv( im_size, CV_32FC(6) );
  for( int r = 0; r < im_size.height; r++ )
  {
    const float *dx_p =  dx.ptr<const float>(r), *dy_p =  dy.ptr<const float>(r);
    float* cov_data = cov.ptr<float>(r);

    for(int c = 0; c < im_size.width; c++, dx_p++, dy_p++ )
    {
      cov_data[c*3] = (*dx_p) * (*dx_p);
      cov_data[c*3+1] = (*dx_p) * (*dy_p);
      cov_data[c*3+2] = (*dy_p) * (*dy_p);
    }
  }

  cv::boxFilter( cov, cov, cov.depth(), cv::Size(5, 5),
                 cv::Point(-1,-1), false, cv::BORDER_DEFAULT );

  calcEigenValsVecs( cov, eigenv );

  cv::Mat_<float> eig_o, eig_d;
  eig_o.create( im_size );
  for( int r = 0; r < im_size.height; r++ )
  {
    const cv::Vec6f *eigenv_p =  eigenv.ptr<const cv::Vec6f>(r);
    float *eig_o_p =  eig_o.ptr<float>(r);
    for( int c = 0; c < im_size.width; c++, eigenv_p++, eig_o_p++ )
    {
      if( (*eigenv_p)[2] || (*eigenv_p)[3] )
        *eig_o_p = atan2((*eigenv_p)[3], (*eigenv_p)[2] );
      else
        *eig_o_p = 0;
    }
  }

  timer.reset();
  const cv::Mat &ig_eig_o = ig.getEigenOrientations();
  GTEST_COUT<< "Elapsed time getEigenOrientations() : " << timer.elapsedTimeMs()<< " ms" << std::endl;

  ASSERT_TRUE(quasiIdenticalMats(ig_eig_o,eig_o, 5e-3 ));

  eig_d.create( im_size );
  for( int r = 0; r < im_size.height; r++ )
  {
    const cv::Vec6f *eigenv_p =  eigenv.ptr<const cv::Vec6f>(r);
    float *eig_d_p = eig_d.ptr<float>(r);
    for( int c = 0; c < im_size.width; c++, eigenv_p++, eig_d_p++ )
    {
      if( (*eigenv_p)[2] )
        *eig_d_p = atan((*eigenv_p)[3] / (*eigenv_p)[2] );
      else if( (*eigenv_p)[3] > 0 )
        *eig_d_p = M_PI/2;
      else if( (*eigenv_p)[3] < 0 )
        *eig_d_p = -M_PI/2;
      else
        *eig_d_p = 0;
    }
  }

  timer.reset();
  const cv::Mat &ig_eig_d = ig.getEigenDirections();
  GTEST_COUT<< "Elapsed time getEigenDirections() : " << timer.elapsedTimeMs()<< " ms" << std::endl;

  ASSERT_TRUE(quasiIdenticalMats(ig_eig_d,eig_d, 5e-3 ));

  cv::Mat_<float> grad_m;
  grad_m = cv::abs(dx) + cv::abs(dy);
  cv::normalize(grad_m, grad_m, 0, 1.0, cv::NORM_MINMAX, cv::DataType<float>::type);

  ASSERT_TRUE(identicalMats(ig.getGradientMagnitudes(),grad_m));

  ig.create( m );
  ig.enableFastMagnitude(false);

  ASSERT_TRUE(identicalMats(ig.getGradientX(),dx));
  ASSERT_TRUE(identicalMats(ig.getGradientY(),dy));
  cv::magnitude(dx, dy, grad_m );
  cv::normalize(grad_m, grad_m, 0, 1.0, cv::NORM_MINMAX, cv::DataType<float>::type);

  ASSERT_TRUE(identicalMats(ig.getGradientMagnitudes(),grad_m));

  const int num_pts = 1000;
  std::vector<cv::Point2f> coords( num_pts );
  for( auto &c : coords )
  {
    c.x = 2 + (im_size.width - 5)*static_cast<float>(rand())/RAND_MAX;
    c.y = 2 + (im_size.height - 5)*static_cast<float>(rand())/RAND_MAX;
  }

  std::vector<float> nn_vals, li_vals, ci_vals;

  nn_vals = ig.getGradientMagnitudes(coords, 0, cv_ext::INTERP_NEAREST_NEIGHBOR);
  li_vals = ig.getGradientMagnitudes(coords, 0, cv_ext::INTERP_BILINEAR);
  ci_vals = ig.getGradientMagnitudes(coords, 0, cv_ext::INTERP_BICUBIC);

  for( int i = 0; i < num_pts; i++ )
  {
    if( nn_vals[i] != cv_ext::getPix<float>( grad_m, coords[i].x, coords[i].y ) )
    {
      std::cout<<i<<"("<<coords[i].x<<", "<<coords[i].y<<") = "<<nn_vals[i]<<" : "<<cv_ext::getPix<float>( grad_m, coords[i].x, coords[i].y )<<std::endl;
    }
    ASSERT_TRUE( nn_vals[i] == cv_ext::getPix<float>( grad_m, coords[i].x, coords[i].y ) );
    ASSERT_TRUE( li_vals[i] == cv_ext::bilinearInterp<float>( grad_m, coords[i].x, coords[i].y ) );
    ASSERT_TRUE( ci_vals[i] == cv_ext::bicubicInterp<float>( grad_m, coords[i].x, coords[i].y ) );
  }
}

TEST (ImageGradientTest, OperatorsTest)
{
  testOperators<uchar>();
  testOperators<ushort >();
  testOperators<float>();
}

TEST (ImageGradientTest, AlignmentTest)
{
  testAlignment<MemoryAlignment::MEM_ALIGN_16>();
  testAlignment<MemoryAlignment::MEM_ALIGN_32>();
  testAlignment<MemoryAlignment::MEM_ALIGN_64>();
  testAlignment<MemoryAlignment::MEM_ALIGN_128>();
  testAlignment<MemoryAlignment::MEM_ALIGN_256>();
  testAlignment<MemoryAlignment::MEM_ALIGN_512>();
}