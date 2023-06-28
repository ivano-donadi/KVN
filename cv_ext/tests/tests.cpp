#include <cstdlib>
#include <ctime>

#include <gtest/gtest.h>

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  // Randomize
  srand(time(NULL));

  return RUN_ALL_TESTS();
}