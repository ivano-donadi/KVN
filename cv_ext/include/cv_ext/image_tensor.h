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
#include <memory>

namespace cv_ext
{
/**
 * @brief ImageTensorBase represents a 3D tensor (i.e., a 3D matrix) accessed by means a vector of OpenCV images (cv::Mat).
 *        It supports to specify the memory alignment (16, 32, 64, .. bytes).
 *
 * @tparam _align The internal data will be aligned at the requested boundary, see cv_ext::MemoryAlignment
 */
template < MemoryAlignment _align > class ImageTensorBase
{
public:

  /**
   * @brief Empty constructor: the tensor should be created using the method create()
   */
  ImageTensorBase() = default;
  
  /**
   * @brief Destructor: in case, release the manually allocated aigned memory
   */  
  ~ImageTensorBase();
  
  /**
   * @brief Constructor: create a preallocated tensor with the size provided in input
   * 
   * @param [in] rows New tensor number of rows
   * @param [in] cols New tensor number of columns
   * @param [in] depth New tensor number of depths (i.e., the number of images)
   * @param [in] data_type Tensor elements type, use cv::DataType< T >::type, where T a basic type (char, float, ,...)
   *
   * Allocate a vector of depth cv::Mat, each one with size rows X cols and type provided by tha parameter data_type
   */
  ImageTensorBase(int rows, int cols, int depth, int data_type );

  /**
   * @brief Allocates a new tensor with the size provided in input
   * 
   * @param [in] rows New number of rows
   * @param [in] cols New number of columns
   * @param [in] depth New number of depths (i.e., the number of images)
   * @param [in] data_type Tensor elements type, use cv::DataType< T >::type, where T a basic type (char, float, ,...)
   * 
   * Allocate a vector of depth cv::Mat, each one with size rows X cols and type provided by tha parameter data_type
   */
  void create( int rows, int cols, int depth, int data_type );
  
  /**
   * @brief Provide the number of rows of each matrix
   */
  int rows() const{ return rows_; };
  
  /**
   * @brief Provide the number of cols of each matrix
   */
  int cols() const{ return cols_; };

  /**
   * @brief Provide the depth of the tensor, i.e. the number of matrices that compose it
   */
  int depth() const{ return data_.size(); };
  
  /**
   * @brief Provide a reference to the i-th matrix, i.e. the i-th level of the tensor
   */  
  inline cv::Mat &operator[]( int i ){ return data_[i]; };

  /**
   * @brief Provide a reference to the i-th matrix, i.e. the i-th level of the tensor
   */    
  inline const cv::Mat &operator[]( int i ) const { return data_[i]; };

  /**
   * @brief Provide a reference to the i-th matrix
   *
   * @param i The i-th level of the tensor
   */  
  inline cv::Mat &at( int i ){ return data_[i]; };

  /**
   * @brief Provide a reference to the i-th matrix
   *
   * @param i The i-th level of the tensor
   */  
  inline const cv::Mat &at( int i ) const { return data_[i]; };
  
private:
    
  void releaseBuf();

  int rows_ = 0, cols_ = 0;
  std::vector< cv::Mat > data_;
  void *data_buf_ = nullptr;
};

typedef ImageTensorBase<CV_EXT_DEFAULT_ALIGNMENT> ImageTensor;
typedef std::shared_ptr< ImageTensor > ImageTensorPtr;

}
