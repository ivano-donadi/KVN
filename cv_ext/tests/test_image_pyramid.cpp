#include <gtest/gtest.h>

#include <vector>
#include <cstdlib>
#include <ctime>

#include <cv_ext/image_pyramid.h>
#include <cv_ext/memory.h>

#include "tests_utils.h"

using namespace cv_ext;

template < MemoryAlignment _align > static void testAlignment()
{
  const int num_levels = 5;
  cv::Mat m(cv::Size(317,199), cv::DataType<ushort>::type);
  cv_ext::ImagePyramidBase<_align> pyr1(m, num_levels, -1, true );
  cv_ext::ImagePyramidBase<_align> pyr2(m, num_levels, cv::DataType<float>::type, true );
  for( int i = 0; i < num_levels; i++ )
  {
    ASSERT_TRUE(CV_EXT_IS_ALIGNED(pyr1[i].data, _align));
    ASSERT_EQ(pyr1[i].step[0]%_align,0);
    ASSERT_TRUE(CV_EXT_IS_ALIGNED(pyr2[i].data, _align));
    ASSERT_EQ(pyr2[i].step[0]%_align,0);
  }

  std::vector< cv_ext::ImagePyramidBase<_align> > pyr_vec;
  pyr_vec.reserve(2);
  for( int i = 0; i < 2; i++ )
    pyr_vec.emplace_back( m, num_levels, -1, true );

  for( int i = 0; i < 2; i++ )
  {
    for( int j = 0; j < num_levels; j++ )
    {
      ASSERT_TRUE(CV_EXT_IS_ALIGNED(pyr_vec[i][j].data, _align));
      ASSERT_EQ(pyr_vec[i][j].step[0]%_align,0);
    }
  }
}

TEST (ImagePyramidTest, GaussianPyramidTest)
{
  const int num_levels = 5;
  cv::Mat m(cv::Size(320,200), cv::DataType<float>::type);
  fillRandom<float>(m);
  cv_ext::ImagePyramid pyr( m, num_levels );
  std::vector< cv::Mat > pyr_levels(num_levels);

  pyr_levels[0] = m;
  for(int i = 1; i < num_levels; i++ )
    cv::pyrDown(pyr_levels[i-1], pyr_levels[i]);

  for( int i = 0; i < num_levels; i++ )
    ASSERT_TRUE(identicalMats(pyr_levels[i],pyr[i]));
}

TEST (ImagePyramidTest, CustomPyramidTest)
{
  const int num_levels = 5;
  srand(time(NULL));
  double scale_factor = 1.0 + static_cast<double >(rand())/RAND_MAX, cur_scale = 1;
  cv::Mat m(cv::Size(317,199), cv::DataType<uchar>::type);
  cv_ext::ImagePyramid pyr;
  pyr.createCustom(m, num_levels, scale_factor, -1, cv_ext::INTERP_BILINEAR );
  std::vector< cv::Mat > pyr_levels(num_levels);

  pyr_levels[0] = m;
  for(int i = 1; i < num_levels; i++ )
  {
    cur_scale *= scale_factor;
    cv::Size cur_size(cvRound(m.cols*(1.0/cur_scale)), cvRound(m.rows*(1.0/cur_scale)));
    cv::resize(pyr_levels[i-1], pyr_levels[i], cur_size, 0, 0, cv::INTER_LINEAR );
  }

  for(int i = 0; i < num_levels; i++ )
    ASSERT_TRUE(identicalMats(pyr_levels[i],pyr[i]));
}

TEST (ImagePyramidTest, AlignmentTest)
{
  testAlignment<MemoryAlignment::MEM_ALIGN_16>();
  testAlignment<MemoryAlignment::MEM_ALIGN_32>();
  testAlignment<MemoryAlignment::MEM_ALIGN_64>();
  testAlignment<MemoryAlignment::MEM_ALIGN_128>();
  testAlignment<MemoryAlignment::MEM_ALIGN_256>();
  testAlignment<MemoryAlignment::MEM_ALIGN_512>();
}
