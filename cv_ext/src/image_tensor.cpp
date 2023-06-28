#include "cv_ext/image_tensor.h"
#include "cv_ext/memory.h"

using namespace cv;
using namespace cv_ext;

template class cv_ext::ImageTensorBase<MEM_ALIGN_NONE>;
template class cv_ext::ImageTensorBase<MEM_ALIGN_16>;
template class cv_ext::ImageTensorBase<MEM_ALIGN_32>;
template class cv_ext::ImageTensorBase<MEM_ALIGN_64>;
template class cv_ext::ImageTensorBase<MEM_ALIGN_128>;
template class cv_ext::ImageTensorBase<MEM_ALIGN_256>;
template class cv_ext::ImageTensorBase<MEM_ALIGN_512>;

template < MemoryAlignment _align >
  ImageTensorBase<_align >::ImageTensorBase (int rows, int cols, int depth, int data_type )
{
  create ( rows, cols, depth, data_type );
}

template < MemoryAlignment _align >
  ImageTensorBase<_align >::~ImageTensorBase()
{
  releaseBuf();
}

template < MemoryAlignment _align >
  void ImageTensorBase<_align >::releaseBuf()
{
  if( data_buf_ )
    CV_EXT_ALIGNED_FREE( data_buf_ );
  data_buf_ = nullptr;
}

template < MemoryAlignment _align >
  void ImageTensorBase<_align >::create (int rows, int cols, int depth, int data_type )
{
  rows_ = rows;
  cols_ = cols;
  
  releaseBuf();
  data_.clear();
  data_.reserve( depth );

  if( _align == MEM_ALIGN_NONE )
  {
    for( int i = 0; i < depth; i++ )
      data_.push_back( Mat(Size(cols_,rows_), data_type ) );
  }
  else
  {
    // Element size in byte
    int elem_size = int(CV_ELEM_SIZE((data_type & CV_MAT_TYPE_MASK)));
    // Size of "vectorized" element (under a SIMD point of view)
    int vec_size = _align/elem_size;
    // Data step for each (aligned) row
    int tensor_data_step = cols_ + ((cols_%vec_size) ? (vec_size - (cols_%vec_size)) : 0);
    // Data size for each 2D matrix of the sensor
    int data_chunk_size = rows_ * tensor_data_step * elem_size;
    // Allocate just one memory portion for the whole vector of tensors
    data_buf_ = CV_EXT_ALIGNED_MALLOC (data_chunk_size*depth, (size_t)_align );
    char *tensor_data_chunk = (char *)data_buf_;
    
    for( int i = 0; i < depth; i++ )
    {
      data_.push_back( Mat( rows_ , cols_, data_type, tensor_data_chunk, tensor_data_step * elem_size ) ); 
      tensor_data_chunk += data_chunk_size;
    }
  }
}
