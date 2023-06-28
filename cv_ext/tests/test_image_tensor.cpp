#include <gtest/gtest.h>

#include <cv_ext/image_tensor.h>
#include <cv_ext/memory.h>

#include "tests_utils.h"

using namespace cv_ext;

template < MemoryAlignment _align > static void testAlignment()
{
  const int depth = 7;

  cv_ext::ImageTensorBase<_align > tens1(317, 199, depth, cv::DataType<float>::type ), tens2;
  tens2.create(1023, 769,  depth, cv::DataType<float>::type );

  for( int i = 0; i < depth; i++ )
  {
    ASSERT_TRUE(CV_EXT_IS_ALIGNED(tens1[i].data, _align));
    ASSERT_EQ(tens1[i].step[0]%_align,0);
    ASSERT_TRUE(CV_EXT_IS_ALIGNED(tens2[i].data, _align));
    ASSERT_EQ(tens2[i].step[0]%_align,0);
  }
}

TEST (ImageTensorTest, AlignmentTest)
{
  testAlignment<MemoryAlignment::MEM_ALIGN_16>();
  testAlignment<MemoryAlignment::MEM_ALIGN_32>();
  testAlignment<MemoryAlignment::MEM_ALIGN_64>();
  testAlignment<MemoryAlignment::MEM_ALIGN_128>();
  testAlignment<MemoryAlignment::MEM_ALIGN_256>();
  testAlignment<MemoryAlignment::MEM_ALIGN_512>();
}
