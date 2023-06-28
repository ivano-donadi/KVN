#include "cv_ext/image_gradient.h"

#include "cv_ext/interpolations.h"
#include "cv_ext/macros.h"
#include "cv_ext/debug_tools.h"

#include <stdexcept>
#include <limits>

/* TODO
 * Implement NEON optimizations.
 */

using namespace cv_ext;

#if defined(CV_EXT_USE_SSE) || defined(CV_EXT_USE_AVX)
#include <immintrin.h>

#ifdef CV_EXT_USING_OPENCV4
#include <opencv2/core/cv_cpu_helper.h>
#endif

#include <opencv2/core/hal/intrin.hpp>

#elif defined(CV_EXT_USE_NEON)
#include <arm_neon.h>
#endif


// Implemented using the approximated arctan function of eq. (7) in:
// Sreeraman Rajan, Sichun Wang, Robert Inkol, and Alain Joyal
// "Efficient Approximations for the Arctangent Function"
// http://www-labs.iro.umontreal.ca/~mignotte/IFT2425/Documents/EfficientApproximationArctgFunction.pdf
// Non vectorized version, i
static inline float approxAtan( const float &x, const float &y )
{
  float ax = std::abs(x), ay = std::abs(y);
  float z = std::min( ax, ay )/(std::max( ax, ay ) + FLT_EPSILON);
  float r = static_cast<float>(M_PI/4)*z + 0.273f*z*(1.f - z);

  if ( ay > ax )
    r = static_cast<float>(M_PI/2) - r;
  if(r > static_cast<float>(M_PI/2) )
    r -= static_cast<float>(M_PI);

  //  if( (x < 0) !=  (y < 0) )
  //    r = -r;
  if (x < 0)
    r = -r;
  if  (y < 0)
    r = -r;

  return r;
}

// Implemented using the approximated arctan function of eq. (7) in:
// Sreeraman Rajan, Sichun Wang, Robert Inkol, and Alain Joyal
// "Efficient Approximations for the Arctangent Function"
// http://www-labs.iro.umontreal.ca/~mignotte/IFT2425/Documents/EfficientApproximationArctgFunction.pdf
// Non vectorized version, i
static inline float approxAtan2( const float &x, const float &y )
{
  float ax = std::abs(x), ay = std::abs(y);
  float z = std::min( ax, ay )/(std::max( ax, ay ) + FLT_EPSILON);
  float r = static_cast<float>(M_PI/4)*z + 0.273f*z*(1.f - z);

  // More clean implementation
//  if ( ay > ax )
//  {
//    if( x >= 0 )
//      r = static_cast<float>(M_PI/2) - r;
//    else
//      r = static_cast<float>(M_PI/2) + r;
//  }
//  else if( x < 0 )
//  {
//    r = static_cast<float>(M_PI) - r;
//  }
//  if( y < 0 )
//    r = -r;

  // Easily vectorizable implementation
  if ( ay > ax && x >= 0 )
    r = static_cast<float>(M_PI/2) -   r;
  if ( ay > ax && x < 0 )
    r = static_cast<float>(M_PI/2) + r;
  if ( ay <= ax && x < 0 )
    r = static_cast<float>(M_PI) - r;
  if( y < 0 )
    r = -r;

  return r;
}

// Code extracted from the eigen2x2() OpenCV function
static void extractFirstEigenvecLine(int len, const float *cov, float *evx, float *evy)
{
  for(int j = 0; j < len; j++ )
  {
    double a = cov[j*3];
    double b = cov[j*3+1];
    double c = cov[j*3+2];

    double u = (a + c)*0.5;
    double v = std::sqrt((a - c)*(a - c)*0.25 + b*b);
    double l1 = u + v;

    double x = b;
    double y = l1 - a;
    double e = fabs(x);

    if( e + fabs(y) < 1e-4 )
    {
      y = b;
      x = l1 - c;
      e = fabs(x);
      if( e + fabs(y) < 1e-4 )
      {
        e = 1./(e + fabs(y) + FLT_EPSILON);
        x *= e, y *= e;
      }
    }

    double d = 1./std::sqrt(x*x + y*y + DBL_EPSILON);
    evx[j] = (float)(x*d);
    evy[j] = (float)(y*d);
  }
}

#if defined( CV_EXT_USE_SSE ) || defined( CV_EXT_USE_AVX )

// Implemented using the approximated arctan function of eq. (7) in:
// Sreeraman Rajan, Sichun Wang, Robert Inkol, and Alain Joyal
// "Efficient Approximations for the Arctangent Function"
// http://www-labs.iro.umontreal.ca/~mignotte/IFT2425/Documents/EfficientApproximationArctgFunction.pdf
// Vectorized version, inspired by OpenCv struct v_atan (see approxAtan above)
static inline void approxAtanLine(int len, const float *dx, const float *dy, float *res )
{
#if defined( CV_EXT_USE_SSE )

  const __m128 zeros = _mm_set1_ps(0.f),
        nzeros = _mm_set1_ps(-0.0f),
        ones = _mm_set1_ps(1.f),
        pi_4 = _mm_set1_ps(M_PI / 4),
        pi_2 = _mm_set1_ps(M_PI / 2),
        alpha = _mm_set1_ps(0.273f),
        eps = _mm_set1_ps(FLT_EPSILON);

  const __m128 *x_p = (const __m128 *) dx, *y_p = (const __m128 *) dy;
  __m128 *res_p = (__m128 *) res;
  for (int i = 0; i < len; i += 4, x_p++, y_p++, res_p++)
  {
    // float ax = std::abs(x), ay = std::abs(y), a;
    __m128 ax = _mm_andnot_ps(nzeros, *x_p);
    __m128 ay = _mm_andnot_ps(nzeros, *y_p);

    // float z = std::min( ax, ay )/(std::max( ax, ay ) + FLT_EPSILON);
    __m128 min_d = _mm_min_ps(ax, ay);
    __m128 max_d = _mm_max_ps(ax, ay);
    max_d = _mm_add_ps(max_d, eps);
    __m128 z = _mm_div_ps(min_d, max_d);

    // r = static_cast<float>(M_PI/4)*z + 0.273f*z*(1.f - z);
    __m128 r = _mm_sub_ps(ones, z);
    r = _mm_mul_ps(alpha, r);
    r = _mm_add_ps(pi_4, r);
    r = _mm_mul_ps(z, r);

    // if ( ay > ax )
    //   r = static_cast<float>(M_PI/2) - r;
    __m128 tmp_r = _mm_sub_ps(pi_2, r);
    __m128 mask1 = _mm_cmpgt_ps(ay, ax);
    r = _mm_blendv_ps(r, tmp_r, mask1);

    // if(a > static_cast<float>(M_PI/2) )
    //   r -= static_cast<float>(M_PI);
    mask1 = _mm_cmpgt_ps(r, pi_2);
    tmp_r = _mm_and_ps(mask1, pi_2);
    r = _mm_sub_ps(r, tmp_r);

    // if( (x < 0) !=  (y < 0) )
    //   r = -r;
    mask1 = _mm_cmplt_ps(*x_p, zeros);
    __m128 mask2 = _mm_cmplt_ps(*y_p, zeros);
    mask1 = _mm_xor_ps(mask1, mask2);
    tmp_r = _mm_xor_ps(nzeros, r); // negate
    *res_p = _mm_blendv_ps(r, tmp_r, mask1);
  }

#elif defined( CV_EXT_USE_AVX )

  const __m256 zeros = _mm256_set1_ps(0.f),
        nzeros = _mm256_set1_ps(-0.0f),
        ones = _mm256_set1_ps(1.f),
        pi_4 = _mm256_set1_ps(M_PI / 4),
        pi_2 = _mm256_set1_ps(M_PI / 2),
        alpha = _mm256_set1_ps(0.273f),
        eps = _mm256_set1_ps(FLT_EPSILON);

  const __m256 *x_p = (const __m256 *) dx, *y_p = (const __m256 *) dy;
  __m256 *res_p = (__m256 *) res;
  for (int i = 0; i < len; i += 8, x_p++, y_p++, res_p++)
  {
    // float ax = std::abs(x), ay = std::abs(y), a;
    __m256 ax = _mm256_andnot_ps(nzeros, *x_p);
    __m256 ay = _mm256_andnot_ps(nzeros, *y_p);

    // float z = std::min( ax, ay )/(std::max( ax, ay ) + FLT_EPSILON);
    __m256 min_d = _mm256_min_ps(ax, ay);
    __m256 max_d = _mm256_max_ps(ax, ay);
    max_d = _mm256_add_ps(max_d, eps);
    __m256 z = _mm256_div_ps(min_d, max_d);

    // r = static_cast<float>(M_PI/4)*z + 0.273f*z*(1.f - z);
    __m256 r = _mm256_sub_ps(ones, z);
    r = _mm256_mul_ps(alpha, r);
    r = _mm256_add_ps(pi_4, r);
    r = _mm256_mul_ps(z, r);

    // if ( ay > ax )
    //   r = static_cast<float>(M_PI/2) - r;
    __m256 tmp_r = _mm256_sub_ps(pi_2, r);
    __m256 mask1 = _mm256_cmp_ps(ay, ax,_CMP_GT_OS);
    r = _mm256_blendv_ps(r, tmp_r, mask1);

    // if(a > static_cast<float>(M_PI/2) )
    //   r -= static_cast<float>(M_PI);
    mask1 = _mm256_cmp_ps(r, pi_2, _CMP_GT_OS);
    tmp_r = _mm256_and_ps(mask1, pi_2);
    r = _mm256_sub_ps(r, tmp_r);

    // if( (x < 0) !=  (y < 0) )
    //   r = -r;
    mask1 = _mm256_cmp_ps(*x_p, zeros, _CMP_LT_OS);
    __m256 mask2 = _mm256_cmp_ps(*y_p, zeros, _CMP_LT_OS);
    mask1 = _mm256_xor_ps(mask1, mask2);
    tmp_r = _mm256_xor_ps(nzeros, r); // negate
    *res_p = _mm256_blendv_ps(r, tmp_r, mask1);
  }

#endif

}

// Implemented using the approximated arctan function of eq. (7) in:
// Sreeraman Rajan, Sichun Wang, Robert Inkol, and Alain Joyal
// "Efficient Approximations for the Arctangent Function"
// http://www-labs.iro.umontreal.ca/~mignotte/IFT2425/Documents/EfficientApproximationArctgFunction.pdf
// Vectorized version, inspired by OpenCv struct v_atan (see approxAtan above)
static inline void approxAtan2Line(int len, const float *dx, const float *dy, float *res )
{

#if defined( CV_EXT_USE_SSE )

  const __m128 zeros = _mm_set1_ps(0.f),
      nzeros = _mm_set1_ps(-0.0f),
      ones = _mm_set1_ps(1.f),
      pi_4 = _mm_set1_ps(M_PI / 4),
      pi_2 = _mm_set1_ps(M_PI / 2),
      pi = _mm_set1_ps(M_PI),
      alpha = _mm_set1_ps(0.273f),
      eps = _mm_set1_ps(FLT_EPSILON);

  const __m128 *x_p = (const __m128 *) dx, *y_p = (const __m128 *) dy;
  __m128 *res_p = (__m128 *) res;
  for (int i = 0; i < len; i += 4, x_p++, y_p++, res_p++)
  {
    // float ax = std::abs(x), ay = std::abs(y), a;
    __m128 ax = _mm_andnot_ps(nzeros, *x_p);
    __m128 ay = _mm_andnot_ps(nzeros, *y_p);


    // float z = std::min( ax, ay )/(std::max( ax, ay ) + FLT_EPSILON);
    __m128 min_d = _mm_min_ps(ax, ay);
    __m128 max_d = _mm_max_ps(ax, ay);
    max_d = _mm_add_ps(max_d, eps);
    __m128 z = _mm_div_ps(min_d, max_d);

    // r = static_cast<float>(M_PI/4)*z + 0.273f*z*(1.f - z);
    __m128 r = _mm_sub_ps(ones, z);
    r = _mm_mul_ps(alpha, r);
    r = _mm_add_ps(pi_4, r);
    r = _mm_mul_ps(z, r);

    // x < 0 mask
    __m128 x_lt_mask = _mm_cmplt_ps(*x_p, zeros);
    // x >= 0 mask
    __m128 x_ge_mask = _mm_cmpge_ps(*x_p, zeros);
    // ay > ax mask
    __m128 ay_gt_mask = _mm_cmpgt_ps(ay, ax);
    // ay <= ax
    __m128 ay_le_mask = _mm_cmple_ps(ay, ax);
    // y < 0 mask
    __m128 y_lt_mask = _mm_cmplt_ps(*y_p, zeros);

    // if ( ay > ax && x >= 0 )
    //  r = static_cast<float>(M_PI/2) - r;
    __m128 mask = _mm_and_ps(ay_gt_mask, x_ge_mask);
    __m128 tmp_r = _mm_sub_ps(pi_2, r);
    r = _mm_blendv_ps(r, tmp_r, mask);

    // if ( ay > ax && x < 0 )
    //   r = static_cast<float>(M_PI/2) + r;
    mask = _mm_and_ps(ay_gt_mask, x_lt_mask);
    tmp_r = _mm_add_ps(pi_2, r);
    r = _mm_blendv_ps(r, tmp_r, mask);

    // if ( ay <= ax && x < 0 )
    //   r = static_cast<float>(M_PI) - r;
    mask = _mm_and_ps(ay_le_mask, x_lt_mask);
    tmp_r = _mm_sub_ps(pi, r);
    r = _mm_blendv_ps(r, tmp_r, mask);

    // if( y < 0 )
    //   r = -r;
    tmp_r = _mm_xor_ps(nzeros, r); // negate
    *res_p = _mm_blendv_ps(r, tmp_r, y_lt_mask);
  }

#elif defined( CV_EXT_USE_AVX )

  const __m256 zeros = _mm256_set1_ps(0.f),
      nzeros = _mm256_set1_ps(-0.0f),
      ones = _mm256_set1_ps(1.f),
      pi_4 = _mm256_set1_ps(M_PI / 4),
      pi_2 = _mm256_set1_ps(M_PI / 2),
      pi = _mm256_set1_ps(M_PI),
      alpha = _mm256_set1_ps(0.273f),
      eps = _mm256_set1_ps(FLT_EPSILON);

  const __m256 *x_p = (const __m256 *) dx, *y_p = (const __m256 *) dy;
  __m256 *res_p = (__m256 *) res;
  for (int i = 0; i < len; i += 8, x_p++, y_p++, res_p++)
  {
    // float ax = std::abs(x), ay = std::abs(y), a;
    __m256 ax = _mm256_andnot_ps(nzeros, *x_p);
    __m256 ay = _mm256_andnot_ps(nzeros, *y_p);


    // float z = std::min( ax, ay )/(std::max( ax, ay ) + FLT_EPSILON);
    __m256 min_d = _mm256_min_ps(ax, ay);
    __m256 max_d = _mm256_max_ps(ax, ay);
    max_d = _mm256_add_ps(max_d, eps);
    __m256 z = _mm256_div_ps(min_d, max_d);

    // r = static_cast<float>(M_PI/4)*z + 0.273f*z*(1.f - z);
    __m256 r = _mm256_sub_ps(ones, z);
    r = _mm256_mul_ps(alpha, r);
    r = _mm256_add_ps(pi_4, r);
    r = _mm256_mul_ps(z, r);

    // x < 0 mask
    __m256 x_lt_mask = _mm256_cmp_ps(*x_p, zeros, _CMP_LT_OS);
    // x >= 0 mask
    __m256 x_ge_mask = _mm256_cmp_ps(*x_p, zeros, _CMP_GE_OS);
    // ay > ax mask
    __m256 ay_gt_mask = _mm256_cmp_ps(ay, ax, _CMP_GT_OS);
    // ay <= ax
    __m256 ay_le_mask = _mm256_cmp_ps(ay, ax, _CMP_LE_OS);
    // y < 0 mask
    __m256 y_lt_mask = _mm256_cmp_ps(*y_p, zeros, _CMP_LT_OS);

    // if ( ay > ax && x >= 0 )
    //  r = static_cast<float>(M_PI/2) - r;
    __m256 mask = _mm256_and_ps(ay_gt_mask, x_ge_mask);
    __m256 tmp_r = _mm256_sub_ps(pi_2, r);
    r = _mm256_blendv_ps(r, tmp_r, mask);

    // if ( ay > ax && x < 0 )
    //   r = static_cast<float>(M_PI/2) + r;
    mask = _mm256_and_ps(ay_gt_mask, x_lt_mask);
    tmp_r = _mm256_add_ps(pi_2, r);
    r = _mm256_blendv_ps(r, tmp_r, mask);

    // if ( ay <= ax && x < 0 )
    //   r = static_cast<float>(M_PI) - r;
    mask = _mm256_and_ps(ay_le_mask, x_lt_mask);
    tmp_r = _mm256_sub_ps(pi, r);
    r = _mm256_blendv_ps(r, tmp_r, mask);

    // if( y < 0 )
    //   r = -r;
    tmp_r = _mm256_xor_ps(nzeros, r); // negate
    *res_p = _mm256_blendv_ps(r, tmp_r, y_lt_mask);
  }

#endif

}

#if defined( CV_EXT_USE_AVX )

// Separate high and low 128 bit and cast to __m128i (taken from OpenCV)
static void v_separate_lo_hi(const __m256& src, __m128i& lo, __m128i& hi)
{
  lo = _mm_castps_si128(_mm256_castps256_ps128(src));
  hi = _mm_castps_si128(_mm256_extractf128_ps(src, 1));
}

// Realign four 3-packed vector to three 4-packed vector (taken from OpenCV)
static void v_pack4x3to3x4(const __m128i& s0, const __m128i& s1, const __m128i& s2, const __m128i& s3, __m128i& d0, __m128i& d1, __m128i& d2)
{
  d0 = _mm_or_si128(s0, _mm_slli_si128(s1, 12));
  d1 = _mm_or_si128(_mm_srli_si128(s1, 4), _mm_slli_si128(s2, 8));
  d2 = _mm_or_si128(_mm_srli_si128(s2, 8), _mm_slli_si128(s3, 4));
}

// Interleave three float vector and store (taken from OpenCV)
static void store_interleave(float* ptr, const __m256& a, const __m256& b, const __m256& c)
{
  __m128i a0, a1, b0, b1, c0, c1;
  v_separate_lo_hi(a, a0, a1);
  v_separate_lo_hi(b, b0, b1);
  v_separate_lo_hi(c, c0, c1);

  cv::v_uint32x4 z = cv::v_setzero_u32();
  cv::v_uint32x4 u0, u1, u2, u3;
  cv::v_transpose4x4(cv::v_uint32x4(a0), cv::v_uint32x4(b0), cv::v_uint32x4(c0), z, u0, u1, u2, u3);
  v_pack4x3to3x4(u0.val, u1.val, u2.val, u3.val, a0, b0, c0);
  cv::v_transpose4x4(cv::v_uint32x4(a1), cv::v_uint32x4(b1), cv::v_uint32x4(c1), z, u0, u1, u2, u3);
  v_pack4x3to3x4(u0.val, u1.val, u2.val, u3.val, a1, b1, c1);

#if !defined(__GNUC__) || defined(__INTEL_COMPILER)
  _mm256_storeu_ps(ptr, _mm256_setr_m128(_mm_castsi128_ps(a0), _mm_castsi128_ps(b0)));
    _mm256_storeu_ps(ptr + 8, _mm256_setr_m128(_mm_castsi128_ps(c0), _mm_castsi128_ps(a1)));
    _mm256_storeu_ps(ptr + 16,  _mm256_setr_m128(_mm_castsi128_ps(b1), _mm_castsi128_ps(c1)));
#else
  // GCC: workaround for missing AVX intrinsic: "_mm256_setr_m128()"
  _mm256_storeu_ps(ptr, _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(a0)), _mm_castsi128_ps(b0), 1));
  _mm256_storeu_ps(ptr + 8, _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(c0)), _mm_castsi128_ps(a1), 1));
  _mm256_storeu_ps(ptr + 16,  _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(b1)), _mm_castsi128_ps(c1), 1));
#endif
}

#endif

static void computeCovMats(const float* dxdata, const float* dydata, float* cov_data, int width)
{
#if defined (CV_EXT_USE_SSE)

  for (int j = 0; j <= width; j += 4)
  {
    cv::v_float32x4 v_dx = cv::v_load(dxdata + j);
    cv::v_float32x4 v_dy = cv::v_load(dydata + j);

    cv::v_float32x4 v_dst0, v_dst1, v_dst2;
    v_dst0 = v_dx * v_dx;
    v_dst1 = v_dx * v_dy;
    v_dst2 = v_dy * v_dy;

    cv::v_store_interleave(cov_data + j * 3, v_dst0, v_dst1, v_dst2);
  }

#else

  const __m256 *dx_p = (const __m256 *) dxdata,
               *dy_p = (const __m256 *) dydata;

  for (int j = 0; j < width; j += 8, dx_p++, dy_p++ )
  {
    __m256 v_dst0, v_dst1, v_dst2;
    v_dst0 = _mm256_mul_ps(*dx_p, *dx_p);
    v_dst1 = _mm256_mul_ps(*dx_p, *dy_p);
    v_dst2 = _mm256_mul_ps(*dy_p, *dy_p);

    store_interleave(cov_data + j * 3, v_dst0, v_dst1, v_dst2);
  }

#endif
}


#endif

template class cv_ext::ImageGradientBase<MEM_ALIGN_NONE>;
template class cv_ext::ImageGradientBase<MEM_ALIGN_16>;
template class cv_ext::ImageGradientBase<MEM_ALIGN_32>;
template class cv_ext::ImageGradientBase<MEM_ALIGN_64>;
template class cv_ext::ImageGradientBase<MEM_ALIGN_128>;
template class cv_ext::ImageGradientBase<MEM_ALIGN_256>;
template class cv_ext::ImageGradientBase<MEM_ALIGN_512>;

template < MemoryAlignment _align >
  ImageGradientBase<_align > :: ImageGradientBase (ImageGradientBase &&other ) :
    pyr_num_levels_(std::move(other.pyr_num_levels_)),
    pyr_scale_factor_(std::move(other.pyr_scale_factor_)),
    use_scharr_operator_(std::move(other.use_scharr_operator_)),
    fast_magnitude_(std::move(other.fast_magnitude_)),
    gl_pyr_(std::move(other.gl_pyr_)),
    dx_imgs_(std::move(other.dx_imgs_)),
    dy_imgs_(std::move(other.dy_imgs_)),
    gradient_dir_imgs_(std::move(other.gradient_dir_imgs_)),
    gradient_ori_imgs_(std::move(other.gradient_ori_imgs_)),
    eigen_dir_imgs_(std::move(other.eigen_dir_imgs_)),
    eigen_ori_imgs_(std::move(other.eigen_ori_imgs_)),
    gradient_mag_imgs_(std::move(other.gradient_mag_imgs_))
{}

template < MemoryAlignment _align >
ImageGradientBase<_align > :: ImageGradientBase( const cv::Mat &img, unsigned int pyr_levels, double pyr_scale_factor,
                                                 bool deep_copy, bool bgr_color_order )
{
  create( img, pyr_levels, pyr_scale_factor, deep_copy, bgr_color_order );
}

template < MemoryAlignment _align >
  void ImageGradientBase<_align > :: create( const cv::Mat &img, unsigned int pyr_levels,
                                             double pyr_scale_factor, bool deep_copy, bool bgr_color_order )
{
  cv_ext_assert( img.rows && img.cols );
  cv_ext_assert( img.channels() == 3 || img.channels() == 1 );

  pyr_num_levels_ = pyr_levels;
  if(pyr_num_levels_ < 1 ) pyr_num_levels_ = 1;
  pyr_scale_factor_ = pyr_scale_factor;
  if( pyr_scale_factor_ < 1.0 || pyr_scale_factor_ > 2.0 ) pyr_scale_factor_ = 2.0;

  dx_imgs_.clear();
  dy_imgs_.clear();
  gradient_dir_imgs_.clear();
  gradient_mag_imgs_.clear();

  dx_imgs_.resize( pyr_num_levels_ );
  dy_imgs_.resize( pyr_num_levels_ );
  gradient_dir_imgs_.resize( pyr_num_levels_ );
  gradient_ori_imgs_.resize( pyr_num_levels_ );
  eigen_dir_imgs_.resize( pyr_num_levels_ );
  eigen_ori_imgs_.resize( pyr_num_levels_ );
  gradient_mag_imgs_.resize( pyr_num_levels_ );

  setImage( img, deep_copy, bgr_color_order );
}

template < MemoryAlignment _align >
  void ImageGradientBase<_align > :: setImage(const cv::Mat& img, bool deep_copy, bool bgr_color_order )
{
  if( pyr_scale_factor_ == 2.0 )
  {
    if( img.channels() == 3 )
    {
      cv_ext::AlignedMatBase <_align > gl_img(img.size(), cv::DataType<float>::type );
      cv::cvtColor( img, gl_img, bgr_color_order?cv::COLOR_BGR2GRAY:cv::COLOR_RGB2GRAY );
      gl_pyr_.create( gl_img, pyr_num_levels_, cv::DataType<float>::type, false );
    }
    else
    {
      // GL input -> initialize only the GL pyramids
      gl_pyr_.create( img, pyr_num_levels_, cv::DataType<float>::type, deep_copy );
    }
  }
  else
  {
    if( img.channels() == 3 )
    {
      cv_ext::AlignedMatBase <_align > gl_img(img.size(), cv::DataType<float>::type );
      cv::cvtColor( img, gl_img, bgr_color_order?cv::COLOR_BGR2GRAY:cv::COLOR_RGB2GRAY );
      gl_pyr_.createCustom( gl_img, pyr_num_levels_, pyr_scale_factor_,
                            cv::DataType<float>::type, INTERP_BILINEAR, false );
    }
    else
    {
      // GL input -> initialize only the GL pyramids
      gl_pyr_.createCustom( img, pyr_num_levels_, pyr_scale_factor_,
                            cv::DataType<float>::type, INTERP_BILINEAR, deep_copy );
    }
  }
}

template < MemoryAlignment _align >
  void ImageGradientBase<_align > :: computeGradientImages(int scale_index )
{
  if( use_scharr_operator_ )
  {
    cv::Scharr( getIntensities( scale_index ), dx_imgs_[scale_index], cv::DataType<float>::type, 1, 0 );
    cv::Scharr( getIntensities( scale_index ), dy_imgs_[scale_index], cv::DataType<float>::type, 0, 1 );
  }
  else
  {
    cv::Sobel( getIntensities( scale_index ), dx_imgs_[scale_index], cv::DataType<float>::type, 1, 0, 3);
    cv::Sobel( getIntensities( scale_index ), dy_imgs_[scale_index], cv::DataType<float>::type, 0, 1, 3);
  }
}

template < MemoryAlignment _align >
  void ImageGradientBase<_align > :: computeGradientDirections(int scale_index )
{
  if( dx_imgs_[scale_index].empty() || dy_imgs_[scale_index].empty() )
    computeGradientImages( scale_index );

  gradient_dir_imgs_[scale_index].create( dx_imgs_[scale_index].size(), cv::DataType<float>::type );
  computeDirections( dx_imgs_[scale_index], dy_imgs_[scale_index], gradient_dir_imgs_[scale_index] );
}

template < MemoryAlignment _align >
void ImageGradientBase<_align > ::computeGradientOrientations(int scale_index )
{
  if( dx_imgs_[scale_index].empty() || dy_imgs_[scale_index].empty() )
    computeGradientImages( scale_index );

  gradient_ori_imgs_[scale_index].create( dx_imgs_[scale_index].size(), cv::DataType<float>::type );
  computeOrientations( dx_imgs_[scale_index], dy_imgs_[scale_index], gradient_ori_imgs_[scale_index] );
}

template < MemoryAlignment _align >
void ImageGradientBase<_align > :: computeEigenDirections(int scale_index )
{
  cv_ext::AlignedMatBase<_align> vx, vy;
  computeEigenVector(scale_index, vx, vy);

  eigen_dir_imgs_[scale_index].create( dx_imgs_[scale_index].size(), cv::DataType<float>::type );
  computeDirections( vx, vy, eigen_dir_imgs_[scale_index] );
}

template < MemoryAlignment _align >
void ImageGradientBase<_align > ::computeEigenOrientations(int scale_index)
{
  cv_ext::AlignedMatBase<_align> vx, vy;
  computeEigenVector(scale_index, vx, vy);

  eigen_ori_imgs_[scale_index].create( dx_imgs_[scale_index].size(), cv::DataType<float>::type );
  computeOrientations( vx, vy, eigen_ori_imgs_[scale_index] );
}

template < MemoryAlignment _align >
  void ImageGradientBase<_align > :: computeGradientMagnitude(int scale_index )
{
  if( dx_imgs_[scale_index].empty() || dy_imgs_[scale_index].empty() )
    computeGradientImages( scale_index );

  cv_ext::AlignedMatBase <_align > gradient_mag_img;

  if( fast_magnitude_ )
  {
    cv_ext::AlignedMatBase <_align > abs_dx_img(dx_imgs_[scale_index].size(), dx_imgs_[scale_index].type() ),
                                       abs_dy_img( dy_imgs_[scale_index].size(), dy_imgs_[scale_index].type() );
    // Compute the (abs) magnitude image, a quickest version that avoids to use the sqrt() operator
    // Here we are not using cv::abs() to avoid unaligned re-allocations
    cv::absdiff(dx_imgs_[scale_index], cv::Scalar::all(0), abs_dx_img);
    cv::absdiff(dy_imgs_[scale_index], cv::Scalar::all(0), abs_dy_img);

    cv::add(abs_dx_img, abs_dy_img, gradient_mag_img );
  }
  else
  {
    cv::magnitude(dx_imgs_[scale_index], dy_imgs_[scale_index], gradient_mag_img);
  }

  cv::normalize(gradient_mag_img, gradient_mag_imgs_[scale_index], 0, 1.0, cv::NORM_MINMAX, cv::DataType<float>::type);
}

template < MemoryAlignment _align >
void ImageGradientBase<_align > ::computeDirections( const cv::Mat &vx, const cv::Mat &vy, cv::Mat &dst )
{
  int width = vx.cols, height = vy.rows;
#if defined( CV_EXT_USE_SSE ) || defined( CV_EXT_USE_AVX )

  for( int y = 0; y < height; y++)
  {
    const float *vx_p = vx.ptr<float>(y), *vy_p = vy.ptr<float>(y);
    float *dst_p = dst.ptr<float>(y);
    if( _align >= CV_EXT_DEFAULT_ALIGNMENT )
    {
      approxAtanLine(width, vx_p, vy_p, dst_p);
    }
    else
    {
      for( int x = 0; x < width; x++, vx_p++, vy_p++, dst_p++)
        *dst_p = approxAtan(*vx_p, *vy_p);
    }
  }

#else

  for( int y = 0; y < height; y++)
  {
    const float *vx_p = vx.ptr<float>(y), *vy_p = vy.ptr<float>(y);
    float *dst_p = dst.ptr<float>(y);
    for( int x = 0; x < width; x++, vx_p++, vy_p++, dst_p++)
      *dst_p = approxAtan(*vx_p, *vy_p);
  }

#endif
}

template < MemoryAlignment _align >
void ImageGradientBase<_align > ::computeOrientations( const cv::Mat &vx, const cv::Mat &vy, cv::Mat &dst )
{
  int width = vx.cols, height = vx.rows;
#if defined( CV_EXT_USE_SSE ) || defined( CV_EXT_USE_AVX )

  for( int y = 0; y < height; y++)
  {
    const float *vx_p = vx.ptr<float>(y), *vy_p = vy.ptr<float>(y);
    float *dst_p = dst.ptr<float>(y);
    if( _align >= CV_EXT_DEFAULT_ALIGNMENT )
    {
      approxAtan2Line(width, vx_p, vy_p, dst_p );
    }
    else
    {
      for( int x = 0; x < width; x++, vx_p++, vy_p++, dst_p++)
        *dst_p = approxAtan2(*vx_p, *vy_p);
    }
  }

#else

  for( int y = 0; y < height; y++)
  {
    const float *vx_p = vx.ptr<float>(y), *vy_p = vy.ptr<float>(y);
    float *dst_p = dst.ptr<float>(y);
    for( int x = 0; x < width; x++, vx_p++, vy_p++, dst_p++)
      *dst_p = approxAtan2(*vx_p, *vy_p);
  }

#endif
}

template < MemoryAlignment _align >
  void ImageGradientBase<_align > :: computeEigenVector(int scale_index, cv::Mat &vx, cv::Mat &vy )
{
  if( dx_imgs_[scale_index].empty() || dy_imgs_[scale_index].empty() )
    computeGradientImages( scale_index );

  cv::Size size = gl_pyr_[scale_index].size();
  vx.create(size, cv::DataType<float>::type );
  vy.create(size, cv::DataType<float>::type );
  cv_ext::AlignedMatBase <_align > cov( size, CV_32FC3 );
  const cv::Mat &dx_mat = dx_imgs_[scale_index], &dy_mat = dy_imgs_[scale_index];

#if defined( CV_EXT_USE_SSE ) || defined( CV_EXT_USE_AVX )

  for( int y = 0; y < size.height; y++)
  {
    float* cov_p = cov.template ptr<float>(y);
    const float* dx_p = dx_mat.ptr<float>(y);
    const float* dy_p = dy_mat.ptr<float>(y);

    if( _align >= CV_EXT_DEFAULT_ALIGNMENT )
    {
      computeCovMats(dx_p, dy_p, cov_p, size.width );
    }
    else
    {
      for( int x = 0; x < size.width; x++)
      {
        float dx = dx_p[x];
        float dy = dy_p[x];

        cov_p[x*3] = dx*dx;
        cov_p[x*3+1] = dx*dy;
        cov_p[x*3+2] = dy*dy;
      }
    }
  }

#else

  for( int y = 0; y < size.height; y++)
  {
    float* cov_p = cov.template ptr<float>(y);
    const float* dx_p = dx_mat.ptr<float>(y);
    const float* dy_p = dy_mat.ptr<float>(y);

    for( int x = 0; x < size.width; x++)
    {
      float dx = dx_p[x];
      float dy = dy_p[x];

      cov_p[x*3] = dx*dx;
      cov_p[x*3+1] = dx*dy;
      cov_p[x*3+2] = dy*dy;
    }
  }

#endif

  const int block_size = 5;
  cv::boxFilter( cov, cov, cov.depth(), cv::Size(block_size, block_size),
                 cv::Point(-1,-1), false, cv::BORDER_DEFAULT );

  for( int y = 0; y < size.height; y++ )
  {
    const float* cov_p = cov.template ptr<float>(y);
    float *vx_p = vx.ptr<float>(y);
    float *vy_p = vy.ptr<float>(y);

    extractFirstEigenvecLine(size.width, cov_p, vx_p, vy_p);
  }
}

template < MemoryAlignment _align >
  const cv::Mat & ImageGradientBase<_align > :: getIntensities(int scale_index )
{
  return gl_pyr_[scale_index ];
}

template < MemoryAlignment _align >
  const cv::Mat &ImageGradientBase<_align >::getGradientX(int scale_index )
{
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if( dx_imgs_[scale_index].empty() )
    computeGradientImages( scale_index );

  return dx_imgs_[scale_index ];
}

template < MemoryAlignment _align >
  const cv::Mat &ImageGradientBase<_align >::getGradientY(int scale_index )
{
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if( dy_imgs_[scale_index].empty() )
    computeGradientImages( scale_index );

  return dy_imgs_[scale_index ];
}

template < MemoryAlignment _align >
  const cv::Mat & ImageGradientBase<_align > :: getGradientDirections(int scale_index )
{
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if( gradient_dir_imgs_[scale_index].empty() )
    computeGradientDirections( scale_index );
  return gradient_dir_imgs_[scale_index];
}

template < MemoryAlignment _align >
const cv::Mat & ImageGradientBase<_align > :: getGradientOrientations(int scale_index )
{
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if( gradient_ori_imgs_[scale_index].empty() )
    computeGradientOrientations( scale_index );
  return gradient_ori_imgs_[scale_index];
}

template < MemoryAlignment _align >
const cv::Mat & ImageGradientBase<_align > :: getEigenDirections(int scale_index )
{
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if( eigen_dir_imgs_[scale_index].empty() )
    computeEigenDirections( scale_index );
  return eigen_dir_imgs_[scale_index];
}

template < MemoryAlignment _align >
const cv::Mat & ImageGradientBase<_align > :: getEigenOrientations(int scale_index )
{
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if( eigen_ori_imgs_[scale_index].empty() )
    computeEigenOrientations( scale_index );
  return eigen_ori_imgs_[scale_index];
}

template < MemoryAlignment _align >
  const cv::Mat & ImageGradientBase<_align > :: getGradientMagnitudes(int scale_index )
{
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if( gradient_mag_imgs_[scale_index].empty() )
    computeGradientMagnitude( scale_index );
  return gradient_mag_imgs_[scale_index];
}

template < MemoryAlignment _align >
  std::vector<float> ImageGradientBase<_align > :: getIntensities(const std::vector<cv::Point2f> &coords,
                                                                  int scale_index, InterpolationType type )
{
  if( type == INTERP_NEAREST_NEIGHBOR )
    return getResultFromCordsNN<float>( getIntensities( scale_index ), coords );
  else if( type == INTERP_BILINEAR )
    return getResultFromCordsBL<float>( getIntensities( scale_index ), coords );
  else if( type == INTERP_BICUBIC )
    return getResultFromCordsBC<float>( getIntensities( scale_index ), coords );
  else
    throw std::invalid_argument("ImageGradientBase::getIntensities : Unknown interpolation type");
}

template < MemoryAlignment _align >
  std::vector<float> ImageGradientBase<_align >::getGradientX(const std::vector<cv::Point2f> &coords,
                                                              int scale_index, InterpolationType type)
{
  if( type == INTERP_NEAREST_NEIGHBOR )
    return getResultFromCordsNN<float>( getGradientX( scale_index ), coords );
  else if( type == INTERP_BILINEAR )
    return getResultFromCordsBL<float>( getGradientX( scale_index ), coords );
  else if( type == INTERP_BICUBIC )
    return getResultFromCordsBC<float>( getGradientX( scale_index ), coords );
  else
    throw std::invalid_argument("ImageGradientBase::getGradientX : Unknown interpolation type");
}

template < MemoryAlignment _align >
  std::vector<float> ImageGradientBase<_align >::getGradientY(const std::vector<cv::Point2f> &coords,
                                                              int scale_index, InterpolationType type)
{
  if( type == INTERP_NEAREST_NEIGHBOR )
    return getResultFromCordsNN<float>( getGradientY( scale_index ), coords );
  else if( type == INTERP_BILINEAR )
    return getResultFromCordsBL<float>( getGradientY( scale_index ), coords );
  else if( type == INTERP_BICUBIC )
    return getResultFromCordsBC<float>( getGradientY( scale_index ), coords );
  else
    throw std::invalid_argument("ImageGradientBase::getGradientY : Unknown interpolation type");
}

template < MemoryAlignment _align >
  std::vector<float> ImageGradientBase<_align > :: getGradientDirections(const std::vector<cv::Point2f> &coords,
                                                                         int scale_index, InterpolationType type )
{
  if( type == INTERP_NEAREST_NEIGHBOR )
    return getResultFromCordsNN<float>( getGradientDirections( scale_index ), coords );
  else if( type == INTERP_BILINEAR )
    return getResultFromCordsBL<float>( getGradientDirections( scale_index ), coords );
  else if( type == INTERP_BICUBIC )
    return getResultFromCordsBC<float>( getGradientDirections( scale_index ), coords );
  else
    throw std::invalid_argument("ImageGradientBase::getGradientDirections : Unknown interpolation type");
}

template < MemoryAlignment _align >
std::vector<float> ImageGradientBase<_align > :: getGradientOrientations(const std::vector<cv::Point2f> &coords,
                                                                         int scale_index, InterpolationType type )
{
  if( type == INTERP_NEAREST_NEIGHBOR )
    return getResultFromCordsNN<float>( getGradientOrientations( scale_index ), coords );
  else if( type == INTERP_BILINEAR )
    return getResultFromCordsBL<float>( getGradientOrientations( scale_index ), coords );
  else if( type == INTERP_BICUBIC )
    return getResultFromCordsBC<float>( getGradientOrientations( scale_index ), coords );
  else
    throw std::invalid_argument("ImageGradientBase::getGradientOrientations : Unknown interpolation type");
}

template < MemoryAlignment _align >
std::vector<float> ImageGradientBase<_align > :: getEigenDirections( const std::vector<cv::Point2f> &coords,
                                                                     int scale_index, InterpolationType type )
{
  if( type == INTERP_NEAREST_NEIGHBOR )
    return getResultFromCordsNN<float>( getEigenDirections( scale_index ), coords );
  else if( type == INTERP_BILINEAR )
    return getResultFromCordsBL<float>( getEigenDirections( scale_index ), coords );
  else if( type == INTERP_BICUBIC )
    return getResultFromCordsBC<float>( getEigenDirections( scale_index ), coords );
  else
    throw std::invalid_argument("ImageGradientBase::getEigenDirections : Unknown interpolation type");
}

template < MemoryAlignment _align >
std::vector<float> ImageGradientBase<_align > :: getEigenOrientations( const std::vector<cv::Point2f> &coords,
                                                                                 int scale_index, InterpolationType type )
{
  if( type == INTERP_NEAREST_NEIGHBOR )
    return getResultFromCordsNN<float>( getEigenOrientations( scale_index ), coords );
  else if( type == INTERP_BILINEAR )
    return getResultFromCordsBL<float>( getEigenOrientations( scale_index ), coords );
  else if( type == INTERP_BICUBIC )
    return getResultFromCordsBC<float>( getEigenOrientations( scale_index ), coords );
  else
    throw std::invalid_argument("ImageGradientBase::getEigenOrientations : Unknown interpolation type");
}

template < MemoryAlignment _align >
  std::vector<float> ImageGradientBase<_align > :: getGradientMagnitudes(const std::vector<cv::Point2f> &coords,
                                                                         int scale_index, InterpolationType type )
{
  if( type == INTERP_NEAREST_NEIGHBOR )
    return getResultFromCordsNN<float>( getGradientMagnitudes( scale_index ), coords );
  else if( type == INTERP_BILINEAR )
    return getResultFromCordsBL<float>( getGradientMagnitudes( scale_index ), coords );
  else if( type == INTERP_BICUBIC )
    return getResultFromCordsBC<float>( getGradientMagnitudes( scale_index ), coords );
  else
    throw std::invalid_argument("ImageGradientBase::getGradientMagnitudes : Unknown interpolation type");
}

template < MemoryAlignment _align >
  template <typename T> std::vector<float>
    ImageGradientBase<_align > :: getResultFromCordsNN(const AlignedMatBase <_align > &res_img,
                                                       const std::vector<cv::Point2f> &coords ) const
{
  int v_size = coords.size();
  std::vector<float> res_vec(v_size );

  for( int i = 0; i < v_size; i++ )
  {
    const cv::Point2f &coord = coords.at(i);
    const int x = roundPositive(coord.x), y = roundPositive(coord.y);
    if( x >= 0 && y >= 0 && x < res_img.cols && y < res_img.rows )
      res_vec[i] = static_cast<float>(res_img.template at<T> (y, x ) );
    else
      res_vec[i] =  OUT_OF_IMG_VAL;
  }
  
  return res_vec;
}

template < MemoryAlignment _align >
  template <typename T> std::vector<float>
    ImageGradientBase<_align > :: getResultFromCordsBL(const AlignedMatBase <_align > &res_img,
                                                       const std::vector<cv::Point2f> &coords ) const
{
  int v_size = coords.size();
  std::vector<float> res_vec(v_size );

  for( int i = 0; i < v_size; i++ )
  {
    const cv::Point2f &coord = coords.at(i);
    const float &x = coord.x, &y = coord.y,
                w = static_cast<float>(res_img.cols) - 1.0f,
                h = static_cast<float>(res_img.rows) - 1.0f;
    if( x >= 1.0f && y >= 1.0f && x < w && y < h )
      res_vec[i] =  bilinearInterp<T>(res_img, x, y );
    else
      res_vec[i] =  OUT_OF_IMG_VAL;
  }
  
  return res_vec;
}

template < MemoryAlignment _align >
  template <typename T> std::vector<float>
      ImageGradientBase<_align > :: getResultFromCordsBC(const AlignedMatBase <_align > &res_img,
                                                         const std::vector<cv::Point2f> &coords ) const
{
  int v_size = coords.size();
  std::vector<float> res_vec(v_size );

  for( int i = 0; i < v_size; i++ )
  {
    const cv::Point2f &coord = coords.at(i);
    const float &x = coord.x, &y = coord.y,
        w = static_cast<float>(res_img.cols) - 2.0f,
        h = static_cast<float>(res_img.rows) - 2.0f;
    if( x >= 2.0f && y >= 2.0f && x < w && y < h )
      res_vec[i] =  bicubicInterp<T>(res_img, x, y );
    else
      res_vec[i] =  OUT_OF_IMG_VAL;
  }

  return res_vec;
}
