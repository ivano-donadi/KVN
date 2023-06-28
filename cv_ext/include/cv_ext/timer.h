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

#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>

namespace cv_ext
{
class BasicTimer
{
public:
  
  BasicTimer(){ reset(); };
  
  void reset()
  {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    time = (uint64_t)tv.tv_sec*1000000 + (uint64_t)tv.tv_usec;
  }
  
  uint64_t elapsedTimeMs()
  {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    uint64_t cur_time = (uint64_t)tv.tv_sec*1000 + (uint64_t)tv.tv_usec/1000;
    
    return cur_time - time/1000;
  }
  
  uint64_t elapsedTimeUs()
  {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    uint64_t cur_time = (uint64_t)tv.tv_sec*1000000 + (uint64_t)tv.tv_usec;
    
    return cur_time - time;
  }

private:
  uint64_t time;
};
}
