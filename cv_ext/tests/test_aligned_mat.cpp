#include <gtest/gtest.h>

#include <vector>

#include <cv_ext/aligned_mat.h>
#include <cv_ext/memory.h>

#include "tests_utils.h"

using namespace cv_ext;

template < MemoryAlignment _align > static void testAlignment()
{
  cv_ext::AlignedMatBase<_align> m1;

  m1.create(cv::Size(1,1),cv::DataType<uchar>::type);
  ASSERT_TRUE(CV_EXT_IS_ALIGNED(m1.data, _align));
  ASSERT_EQ(m1.step[0]%_align,0);

  cv_ext::AlignedMatBase<_align> m2(1, 1, cv::DataType<uchar>::type, 255 );
  ASSERT_TRUE(CV_EXT_IS_ALIGNED(m2.data, _align));
  ASSERT_EQ(m2.step[0]%_align,0);

  m1 = AlignedMatBase<_align>(11, 7, cv::DataType<uchar>::type );
  ASSERT_TRUE(CV_EXT_IS_ALIGNED(m1.data, _align));
  ASSERT_EQ(m1.step[0]%_align,0);

  m2.create(cv::Size(200,100),cv::DataType<uchar>::type);
  ASSERT_TRUE(CV_EXT_IS_ALIGNED(m2.data, _align));
  ASSERT_EQ(m2.step[0]%_align,0);

  m1 = AlignedMatBase<_align>(11, 7, CV_8UC(3) );
  ASSERT_TRUE(CV_EXT_IS_ALIGNED(m1.data, _align));
  ASSERT_EQ(m1.step[0]%_align,0);

  m2.create(cv::Size(200,100),CV_8UC(5));
  ASSERT_TRUE(CV_EXT_IS_ALIGNED(m2.data, _align));
  ASSERT_EQ(m2.step[0]%_align,0);

  cv::Mat m3( 317,199, cv::DataType<double>::type );
  fillRandom<float>( m3 );
  cv_ext::AlignedMatBase<_align> m4(m3, true);
  cv_ext::AlignedMatBase<_align> m5(m4);
  ASSERT_TRUE(CV_EXT_IS_ALIGNED(m4.data, _align));
  ASSERT_EQ(m4.step[0]%_align,0);
  ASSERT_TRUE(identicalMats(m3,m4));
  ASSERT_TRUE(m3.data != m4.data);
  ASSERT_TRUE(m4.data == m5.data);

  std::vector<AlignedMatBase<_align> > m_vec;
  m_vec.reserve(2);
  for( int i = 0; i < 2; i++ )
  {
    m_vec.emplace_back( 11, 7, cv::DataType<uchar>::type, _align );
    ASSERT_TRUE(CV_EXT_IS_ALIGNED(m_vec[i].data, _align));
    ASSERT_EQ(m_vec[i].step[0]%_align,0);
  }
}

template < MemoryAlignment _align > static void testAlignmentWithFunctions()
{
  cv_ext::AlignedMatBase<_align> m1(11, 7, cv::DataType<float>::type );
  cv::Mat m2( 11, 7, cv::DataType<float>::type );

  fillRandom<float>( m2 );

  m2.copyTo(m1);

  cv::blur(m1,m1,cv::Size(3,3));
  cv::blur(m2,m2,cv::Size(3,3));

  ASSERT_TRUE(CV_EXT_IS_ALIGNED(m1.data, _align));
  ASSERT_EQ(m1.step[0]%_align,0);
  ASSERT_TRUE(identicalMats(m1,m2));

  cv::blur(m2,m1,cv::Size(3,3));
  ASSERT_TRUE(CV_EXT_IS_ALIGNED(m1.data, _align));
  ASSERT_EQ(m1.step[0]%_align,0);
}

TEST (AlignedMatTest, ConstructorTest)
{
  const int vec_size = 3;
  std::vector < cv::Mat > in_m(vec_size);
  for( auto &m : in_m )
  {
    m.create(cv::Size(1920,1080), cv::DataType<float>::type);
    fillRandom<float>(m);
  }

  std::vector < cv_ext::AlignedMat > a_m;
  a_m.reserve(vec_size);
  for( auto &m : in_m )
    a_m.emplace_back(m, true );

  for( int i = 0; i < vec_size; i++ )
    ASSERT_TRUE(identicalMats(in_m[i], a_m[i]));
}

TEST (AlignedMatTest, AlignmentTest)
{
  testAlignment<MemoryAlignment::MEM_ALIGN_16>();
  testAlignment<MemoryAlignment::MEM_ALIGN_32>();
  testAlignment<MemoryAlignment::MEM_ALIGN_64>();
  testAlignment<MemoryAlignment::MEM_ALIGN_128>();
  testAlignment<MemoryAlignment::MEM_ALIGN_256>();
  testAlignment<MemoryAlignment::MEM_ALIGN_512>();
}

TEST (AlignedMatTest, AlignmentTestWithFunctions)
{
  testAlignmentWithFunctions<MemoryAlignment::MEM_ALIGN_16>();
  testAlignmentWithFunctions<MemoryAlignment::MEM_ALIGN_32>();
  testAlignmentWithFunctions<MemoryAlignment::MEM_ALIGN_64>();
  testAlignmentWithFunctions<MemoryAlignment::MEM_ALIGN_128>();
  testAlignmentWithFunctions<MemoryAlignment::MEM_ALIGN_256>();
  testAlignmentWithFunctions<MemoryAlignment::MEM_ALIGN_512>();
}
