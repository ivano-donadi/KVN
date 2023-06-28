#include "cv_ext/aligned_mat.h"
#include "cv_ext/memory.h"

#include <opencv2/core/types_c.h>

// Maintain backwards compatibility with opencv 3
#ifdef CV_EXT_USING_OPENCV3
#define AccessFlag int
#endif

using namespace cv;
using namespace cv_ext;

class AlignedAllocator : public MatAllocator
{
public:
  
  AlignedAllocator(MemoryAlignment align) : align_(align){}; 
  
  // Code modified from OpenCV cv::StdMatAllocator
  UMatData* allocate ( int dims, const int* sizes, int type,
                       void* data0, size_t* step, AccessFlag /*flags*/, UMatUsageFlags /*usageFlags*/ ) const override
  {
    
    size_t total = CV_ELEM_SIZE ( type );
    for ( int i = dims-1; i >= 0; i-- )
    {
      if ( step )
      {
        if ( data0 && step[i] != CV_AUTO_STEP )
        {
          CV_Assert ( total <= step[i] );
          total = step[i];
        }
        else
        {
          step[i] = total;
        }
      }
      total *= sizes[i];
      // Align "rows"
      if(i == dims - 1)
      {
        if( total%align_ )
        {
          total += align_ - total%align_;
        }
      }
    }
    uchar* data = data0 ? ( uchar* ) data0 : ( uchar* ) CV_EXT_ALIGNED_MALLOC (total , (size_t)align_ );
    UMatData* u = new UMatData ( this );
    u->data = u->origdata = data;
    u->size = total;
    if ( data0 )
    {
      u->flags |= UMatData::USER_ALLOCATED;
    }

    return u;
  }

  bool allocate ( UMatData* u, AccessFlag /*accessFlags*/, UMatUsageFlags /*usageFlags*/ ) const override
  {
    if ( !u )
    {
      return false;
    }
    return true;
  }

  void deallocate ( UMatData* u ) const override
  {
    if ( !u )
    {
      return;
    }

    CV_Assert ( u->urefcount == 0 );
    CV_Assert ( u->refcount == 0 );
    if ( ! ( u->flags & UMatData::USER_ALLOCATED ) )
    {
      CV_EXT_ALIGNED_FREE ( u->origdata );
      u->origdata = 0;
    }
    delete u;
  }
private:
  
  MemoryAlignment align_;
  
};


template class cv_ext::AlignedMatBase<MEM_ALIGN_NONE>;
template class cv_ext::AlignedMatBase<MEM_ALIGN_16>;
template class cv_ext::AlignedMatBase<MEM_ALIGN_32>;
template class cv_ext::AlignedMatBase<MEM_ALIGN_64>;
template class cv_ext::AlignedMatBase<MEM_ALIGN_128>;
template class cv_ext::AlignedMatBase<MEM_ALIGN_256>;
template class cv_ext::AlignedMatBase<MEM_ALIGN_512>;

static AlignedAllocator& getAlignedAllocator16()
{
  static AlignedAllocator instance(MEM_ALIGN_16);
  return instance;
}
static AlignedAllocator& getAlignedAllocator32()
{
  static AlignedAllocator instance(MEM_ALIGN_32);
  return instance;
}
static AlignedAllocator& getAlignedAllocator64()
{
  static AlignedAllocator instance(MEM_ALIGN_64);
  return instance;
}

static AlignedAllocator& getAlignedAllocator128()
{
  static AlignedAllocator instance(MEM_ALIGN_128);
  return instance;
}
static AlignedAllocator& getAlignedAllocator256()
{
  static AlignedAllocator instance(MEM_ALIGN_256);
  return instance;
}
static AlignedAllocator& getAlignedAllocator512()
{
  static AlignedAllocator instance(MEM_ALIGN_512);
  return instance;
}

template < MemoryAlignment _align >
  AlignedMatBase<_align >::AlignedMatBase() : Mat ()
{
  setAllocator();
}

template < MemoryAlignment _align >
  AlignedMatBase<_align >::AlignedMatBase (int rows, int cols, int data_type ) : Mat ()
{
  setAllocator();
  create(cv::Size(cols,rows), data_type );
}

template < MemoryAlignment _align >
  AlignedMatBase<_align >::AlignedMatBase (Size size, int data_type ) : Mat ()
{
  setAllocator();
  create(size, data_type);
}

template < MemoryAlignment _align >
  AlignedMatBase<_align >::AlignedMatBase (int rows, int cols, int data_type, const Scalar& s ) : Mat ()
{
  setAllocator();
  create(cv::Size(cols,rows), data_type );
  this->Mat::operator=(s);
}

template < MemoryAlignment _align >
  AlignedMatBase<_align >::AlignedMatBase (Size size, int data_type, const Scalar& s ) : Mat ()
{
  setAllocator();
  create(size, data_type);
  this->Mat::operator=(s);
}

template < MemoryAlignment _align >
  AlignedMatBase<_align >::AlignedMatBase(const cv::Mat& other, bool copy_data ) : Mat (other)
{
  if( copy_data )
  {
    this->release();
    setAllocator();
    other.copyTo(*this);
  }
  else
  {
    setAllocator();
    if( !isMemoryAligned(other, _align) )
      throw std::invalid_argument( "cv_ext::AlignedMatBase::AlignedMatBase( const cv::Mat& ) : "
                                   "Image is not correctly aligned");
  }
}

template < MemoryAlignment _align >
  AlignedMatBase<_align > &AlignedMatBase<_align >::operator=(const cv::Mat &other)
{
  if( !isMemoryAligned(other, _align) )
    throw std::invalid_argument("cv_ext::AlignedMatBase::operator= : Image is not correctly aligned");

  cv::Mat::operator=(other);
  setAllocator();
  return *this;
}

template < MemoryAlignment _align >
  void AlignedMatBase<_align >::setAllocator()
{
  switch(_align)
  {
    case MEM_ALIGN_16:
      allocator = &getAlignedAllocator16();
      break;
    case MEM_ALIGN_32:
      allocator = &getAlignedAllocator32();
      break;
    case MEM_ALIGN_64:
      allocator = &getAlignedAllocator64();
      break;
    case MEM_ALIGN_128:
      allocator = &getAlignedAllocator128();
      break;
    case MEM_ALIGN_256:
      allocator = &getAlignedAllocator256();
      break;
    case MEM_ALIGN_512:
      allocator = &getAlignedAllocator512();
      break;
    default:
      break;
  }
}

#ifdef CV_EXT_USING_OPENCV3
#undef AccessFlag
#endif