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

#include <cv_ext/memory.h>

namespace cv_ext
{
/**
 * @brief Utility class used to allocate a cv::Mat with specific memory aligned (16, 32, 64, .. bytes)
 *
 * @tparam _align The data will be aligned at the requested boundary, see cv_ext::MemoryAlignment
 */ 
template < MemoryAlignment _align > class AlignedMatBase : public cv::Mat
{
public:
    
  /**
   * @brief Empty constructor: the mat should be created using the method create()
   */  
  AlignedMatBase();

  /**
   * @brief Constructor: create a preallocated matrix with the size provided in input
   * 
   * @param [in] rows New matrix number of rows
   * @param [in] cols New matrix number of columns
   * @param [in] data_type New matrix type, use cv::DataType< T >::type 
   */
  AlignedMatBase(int rows, int cols, int data_type );

  /**
   * @brief Constructor: create a preallocated matrix with the size provided in input
   * 
   * @param [in] size New matrix size: cv::Size(cols, rows)
   * @param [in] data_type New matrix type, use cv::DataType< T >::type 
   */
  AlignedMatBase(cv::Size size, int data_type );

  /**
   * @brief Constructor: create a preallocated matrix with the size provided in input
   * 
   * @param [in] rows New matrix number of rows
   * @param [in] cols New matrix number of columns
   * @param [in] data_type New matrix type, use cv::DataType< T >::type 
   * @param [in] s A value used to initialize each matrix element 
   */
  AlignedMatBase(int rows, int cols, int data_type, const cv::Scalar& s );

  /**
   * @brief Constructor: create a preallocated matrix with the size provided in input
   * 
   * @param [in] size New matrix size: cv::Size(cols, rows)
   * @param [in] data_type New matrix type, use cv::DataType< T >::type
   * @param [in] s A value used to initialize each matrix element 
   */
  AlignedMatBase(cv::Size size, int data_type, const cv::Scalar& s );
  
  /**
   * @brief Destructor: in case, release the manually allocated aigned memory
   */
  virtual ~AlignedMatBase(){};


  /**
   * @brief Copy constructor with a cv::Mat
   *
   * @param [in] other Source cv::Mat
   *
   * No data is copied by this constructor: the data is shared between the current and the source cv::Mat.
   *
   * @warning If the source cv::Mat has not proper memory alignment, the constructor raises an exception.
   */
  AlignedMatBase(const cv::Mat& other, bool copy_data=false );

  /**
   * @brief Assignment operator with a cv::Mat
   *
   * @param [in] other Assigned cv::Mat
   * @param [in] copy_data This flag specifies whether the image data should be copied (true) or shared (false)
   *
   * If copy_data is false, no data is copied by this operator: in this case the data is shared between the current
   * and the input cv::Mat.
   *
   * @warning If the input cv::Mat has not proper memory alignment, this method raises an exception.
   */
  AlignedMatBase& operator=(const cv::Mat& other );

private:

  void setAllocator();
};

typedef AlignedMatBase<CV_EXT_DEFAULT_ALIGNMENT> AlignedMat;

}
