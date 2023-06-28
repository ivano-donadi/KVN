/*
 * cv_ext - openCV EXTensions
 *
 *  Copyright (c) 2020, Alberto Pretto <alberto.pretto@flexsight.eu>
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <opencv2/opencv.hpp>

#if defined(__arm__) || defined(__aarch64__)

#include <cstdlib>
#define CV_EXT_ALIGNED_MALLOC(size, alignment) aligned_alloc((size_t)(alignment), (size_t)(size))
#define CV_EXT_ALIGNED_FREE(p) free((void *)(p))

#else

#include <mm_malloc.h>
#define CV_EXT_ALIGNED_MALLOC(size, alignment) _mm_malloc((size_t)(size), (size_t)(alignment))
#define CV_EXT_ALIGNED_FREE(p) _mm_free((void *)(p))

#endif

#define CV_EXT_IS_ALIGNED(pointer, alignment) (((uintptr_t)(const void *)(pointer)) % (alignment) == 0)

namespace cv_ext
{
enum MemoryAlignment{ MEM_ALIGN_NONE = 1, MEM_ALIGN_16 = 16, MEM_ALIGN_32 = 32, MEM_ALIGN_64 = 64,
  MEM_ALIGN_128 = 128, MEM_ALIGN_256 = 256, MEM_ALIGN_512 = 512 };

inline bool isMemoryAligned( const cv::Mat &m, MemoryAlignment alignment )
{
  return ( CV_EXT_IS_ALIGNED(m.data, alignment) && CV_EXT_IS_ALIGNED(m.step[0], alignment) );
}

}

#if ( defined(CV_EXT_USE_SSE)  || defined(CV_EXT_USE_NEON) )
#define CV_EXT_DEFAULT_ALIGNMENT (cv_ext::MEM_ALIGN_16)
#elif defined(CV_EXT_USE_AVX)
#define CV_EXT_DEFAULT_ALIGNMENT (cv_ext::MEM_ALIGN_32)
#else
#define CV_EXT_DEFAULT_ALIGNMENT (cv_ext::MEM_ALIGN_NONE)
#endif
