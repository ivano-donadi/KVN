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

#include <cmath>
#include <opencv2/opencv.hpp>
#include <ceres/jet.h>

#include <cv_ext/base.h>
#include <cv_ext/image_tensor.h>

/* TODO
 *
 * IMPORTNAT: Check derivatives computation
 * Implement in SSE
*/

static inline float cubicInterp ( const float &offset, const float &v0, const float &v1, 
                                  const float &v2, const float &v3 )
{
  return (1.0f/18.0f) * ( ( ( ( -7.0f * v0 + 21.0f * v1 - 21.0f * v2 + 7.0f * v3 ) * offset +
                        ( 15.0f * v0 - 36.0f * v1 + 27.0f * v2 - 6.0f * v3 ) ) * offset +
                        ( -9.0f * v0 + 9.0f * v2 ) ) * offset + ( v0 + 16.0f * v1 + v2 ) );
}

namespace cv_ext
{

//! @brief Infinetesimal used for computed interpolated image derivatives
static const double IMG_DERIVS_EPS = 1.0e-3;


template < typename _Ti , typename _To  > inline 
  _To getPix ( const cv::Mat &img, const _To &x, const _To &y )
{
  int x0 = roundPositive( x ), y0 = roundPositive( y );
  return img.at<_Ti> ( y0, x0 );
}

template < typename _Ti , typename _To, MemoryAlignment _align > inline
  _To tensorGetPix ( const ImageTensorBase<_align > &tensor,
                     const _To &x, const _To &y, const _To &z, 
                     bool cyclic_z = true )
{
  int x0 = roundPositive( x ),
      y0 = roundPositive( y ),
      z0 = cvRound( double ( z ) ); // z could be negative, so use opencv cvRound() function
      
  if( cyclic_z )
  {
    if( z0 < 0 )
      z0 = tensor.depth() -1;
    else
      z0 %= tensor.depth();
  }
  return tensor[z0].template at<_Ti> ( y0, x0 );
}


// template < typename _Ti , typename _To  > inline 
//   _To bilinearInterp ( const cv::Mat &img, const _To &x, const _To &y )
// {
//   int x0 = floor ( double ( x ) ), y0 = floor ( double ( y ) );
//   int x1 = x0 + 1, y1 = y0 + 1;
// 
//   _To bilienar_mat[] = { _To ( img.at<_Ti> ( y0, x0 ) ), _To ( img.at<_Ti> ( y1, x0 ) ),
//                          _To ( img.at<_Ti> ( y0, x1 ) ), _To ( img.at<_Ti> ( y1, x1 ) ) };
//                          
//   _To x_mat[] = { 1.0f- ( x - ( _To ) x0 ) , ( x - ( _To ) x0 ) };
//   _To y_mat[] = { 1.0f- ( y - ( _To ) y0 ) , ( y - ( _To ) y0 ) };
// 
//   return x_mat[0]* ( bilienar_mat[0]*y_mat[0] + bilienar_mat[1]*y_mat[1] )
//          + x_mat[1]* ( bilienar_mat[2]*y_mat[0] + bilienar_mat[3]*y_mat[1] );
// };

template < typename _Ti , typename _To  > inline 
  _To bilinearInterp ( const cv::Mat &img, const _To &x, const _To &y )
{
  int x0 = floor ( double ( x ) ), y0 = floor ( double ( y ) );
  int x1 = x0 + 1, y1 = y0 + 1;

  Eigen::Matrix<_To, 2, 2> bilienar_mat;
  bilienar_mat << _To ( img.at<_Ti> ( y0, x0 ) ), _To ( img.at<_Ti> ( y1, x0 ) ),
                  _To ( img.at<_Ti> ( y0, x1 ) ), _To ( img.at<_Ti> ( y1, x1 ) );
  
  Eigen::Matrix<_To, 2, 1> x_mat, y_mat;
  
  x_mat << ( _To ) 1.0 - ( x - ( _To ) x0 ) , ( x - ( _To ) x0 );
  y_mat << ( _To ) 1.0 - ( y - ( _To ) y0 ) , ( y - ( _To ) y0 );

  return ( x_mat.array() * (bilienar_mat*y_mat).array() ).sum();
}

template < typename _Ti , typename _To, MemoryAlignment _align > inline
  _To tensorbilinearInterp ( const ImageTensorBase<_align > &tensor,
                             const _To &x, const _To &y, const _To &z, 
                             bool cyclic_z = true )
{
  int x0 = floor ( double ( x ) ), y0 = floor ( double ( y ) ), z0 = floor ( double ( z ) );
  int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
  
  if( cyclic_z )
  {
    if( z0 < 0 )
      z0 = tensor.depth() -1;
    else
      z0 %= tensor.depth();
    z1 %= tensor.depth();
  }
  
  const cv::Mat &img0 = tensor[z0], &img1 = tensor[z1];
  Eigen::Matrix<_To, 2, 2> bilienar_mat0, bilienar_mat1;
  bilienar_mat0 << _To ( img0.at<_Ti> ( y0, x0 ) ), _To ( img0.at<_Ti> ( y1, x0 ) ),
                   _To ( img0.at<_Ti> ( y0, x1 ) ), _To ( img0.at<_Ti> ( y1, x1 ) );
  bilienar_mat1 << _To ( img1.at<_Ti> ( y0, x0 ) ), _To ( img1.at<_Ti> ( y1, x0 ) ),
                   _To ( img1.at<_Ti> ( y0, x1 ) ), _To ( img1.at<_Ti> ( y1, x1 ) );
                   
  Eigen::Matrix<_To, 2, 1> x_mat, y_mat, z_mat;
  
  x_mat << ( _To ) 1.0 - ( x - ( _To ) x0 ) , ( x - ( _To ) x0 ) ;
  y_mat << ( _To ) 1.0 - ( y - ( _To ) y0 ) , ( y - ( _To ) y0 ) ;
  
  _To val0 = ( x_mat.array() * (bilienar_mat0*y_mat).array() ).sum(),
      val1 = ( x_mat.array() * (bilienar_mat1*y_mat).array() ).sum();
      
  return (( _To ) 1.0 - ( z - ( _To ) z0 ) ) * val0 + ( z - ( _To ) z0 ) * val1;
}

// template < typename _Ti , typename _To  > inline 
//   _To tensorBilinearInterp ( const std::vector< cv::Mat > &tensor, 
//                              const _To &x, const _To &y, const _To &z, 
//                              bool cyclic_z = true )
// {
//   int z0 = floor ( double ( z ) ), z1 = z0 + 1;
//   if( cyclic_z && z1 >= tensor.depth() )
//   {
//     z0 %= tensor.depth() - 1;
//     z1 %= tensor.depth() - 1;
//   }
//   // BUG Devi usare mod()
//   z1 = z0 + 1;
//   
//     
//     && z0 >= 
//   
//   int x0 = floor ( double ( x ) ), y0 = floor ( double ( y ) );
//   int x1 = x0 + 1, y1 = y0 + 1;
// 
//   _To bilienar_mat[] = { _To ( img.at<_Ti> ( y0, x0 ) ), _To ( img.at<_Ti> ( y1, x0 ) ),
//                          _To ( img.at<_Ti> ( y0, x1 ) ), _To ( img.at<_Ti> ( y1, x1 ) ) };
//                          
//   _To x_mat[] = { 1.0f- ( x - ( _To ) x0 ) , ( x - ( _To ) x0 ) };
//   _To y_mat[] = { 1.0f- ( y - ( _To ) y0 ) , ( y - ( _To ) y0 ) };
// 
//   return x_mat[0]* ( bilienar_mat[0]*y_mat[0] + bilienar_mat[1]*y_mat[1] )
//          + x_mat[1]* ( bilienar_mat[2]*y_mat[0] + bilienar_mat[3]*y_mat[1] );
// };

template < typename _Ti, typename _To, typename _Tj, int _Nj > inline
  ceres::Jet<_Tj, _Nj> getPix ( const cv::Mat &img,
                                const ceres::Jet<_Tj, _Nj> &x, 
                                const ceres::Jet<_Tj, _Nj> &y )
{
  _Tj real_x = x.a, real_y = y.a;
  ceres::Jet<_Tj, _Nj> out;
  out.a = getPix< _Ti , _Tj >( img, real_x, real_y );
  
  _Tj img_derivs[2];
  
  // Compute the numerical derivatives in the image
  img_derivs[0] = _Tj(0.5)* ( getPix< _Ti , _Tj >( img, real_x + _Tj(1.0), real_y ) -
                              getPix< _Ti , _Tj >( img, real_x - _Tj(1.0), real_y ));
  img_derivs[1] = _Tj(0.5)* ( getPix< _Ti , _Tj >( img, real_x, real_y + _Tj(1.0) ) -
                              getPix< _Ti , _Tj >( img, real_x, real_y - _Tj(1.0) ));
                    
  for( int i = 0; i < _Nj; i++)
    out.v(i,0) = img_derivs[0]*x.v(i,0) + img_derivs[1]*y.v(i,0);
  
  return out;  
}

template < typename _Ti, typename _To, typename _Tj, int _Nj, MemoryAlignment _align > inline
  ceres::Jet<_Tj, _Nj> tensorGetPix ( const ImageTensorBase<_align > &tensor,
                                      const ceres::Jet<_Tj, _Nj> &x, 
                                      const ceres::Jet<_Tj, _Nj> &y,
                                      const ceres::Jet<_Tj, _Nj> &z,
                                      bool cyclic_z = true )
{
  _Tj real_x = x.a, real_y = y.a, real_z = z.a;
  ceres::Jet<_Tj, _Nj> out;
  out.a = tensorGetPix< _Ti , _Tj >( tensor, real_x, real_y, real_z, cyclic_z );
  
  _Tj tensor_derivs[3];
  
  // Compute the numerical derivatives in the tensor
  tensor_derivs[0] = _Tj(0.5)*( tensorGetPix< _Ti , _Tj >( tensor, real_x + _Tj(1.0), real_y, real_z, cyclic_z  ) -
                                tensorGetPix< _Ti , _Tj >( tensor, real_x - _Tj(1.0), real_y, real_z, cyclic_z  ));
  tensor_derivs[1] = _Tj(0.5)*( tensorGetPix< _Ti , _Tj >( tensor, real_x, real_y + _Tj(1.0), real_z, cyclic_z  ) -
                                tensorGetPix< _Ti , _Tj >( tensor, real_x, real_y - _Tj(1.0), real_z, cyclic_z  ));
  tensor_derivs[2] = _Tj(0.5)*( tensorGetPix< _Ti , _Tj >( tensor, real_x, real_y, real_z + _Tj(1.0), cyclic_z  ) -
                                tensorGetPix< _Ti , _Tj >( tensor, real_x, real_y, real_z - _Tj(1.0), cyclic_z  ));
                   
  for( int i = 0; i < _Nj; i++)
    out.v(i,0) = tensor_derivs[0]*x.v(i,0) + tensor_derivs[1]*y.v(i,0) + tensor_derivs[2]*z.v(i,0);
  
  return out;  
}

/* 
 * Pixel coordinates depends on the _Nj input (optimization) parameters x, y, z, t, ...:
 * 
 * u( x, y, z, t, ...) -> i.e., the "x" coordinate in the image
 * v( x, y, z, t, .. ) -> i.e., the "y" coordinate in the image
 * 
 * As Jet input, we also have the following Jacobian ( through the infinitesimal parts):
 * 
 *       | du/dx  du/dy  du/dz  du/dz ... | 
 * J_g = |                                |
 *       | dv/dx  dv/dy  dv/dz  dv/dz ... | 
 * 
 * We should sample in the input image a pixel value f(u,v), 
 * so we need to compute the following derivatives:
 * 
 * J = | df/dx  df/dy  df/dz  df/dt  ... |
 * 
 * J = J_f * J_g
 * 
 * where:
 * 
 * J_f = | df/du  df/dv |
 * 
 */
template < typename _Ti, typename _To, typename _Tj, int _Nj > inline
  ceres::Jet<_Tj, _Nj> bilinearInterp ( const cv::Mat &img,
                                        const ceres::Jet<_Tj, _Nj> &x, 
                                        const ceres::Jet<_Tj, _Nj> &y )
{
  _Tj real_x = x.a, real_y = y.a;
  ceres::Jet<_Tj, _Nj> out;
  out.a = bilinearInterp< _Ti , _Tj >( img, real_x, real_y );
  
  _Tj img_derivs[2], idt = _Tj(1.0/(2.0*IMG_DERIVS_EPS));
  
  // Compute the numerical derivatives in the image
  img_derivs[0] = (bilinearInterp< _Ti , _Tj >( img, real_x + _Tj(IMG_DERIVS_EPS), real_y ) -
                   bilinearInterp< _Ti , _Tj >( img, real_x - _Tj(IMG_DERIVS_EPS), real_y )) * idt;
  img_derivs[1] = (bilinearInterp< _Ti , _Tj >( img, real_x, real_y + _Tj(IMG_DERIVS_EPS) ) -
                   bilinearInterp< _Ti , _Tj >( img, real_x, real_y - _Tj(IMG_DERIVS_EPS) )) * idt;

//   img_derivs[0] = (bilinearInterp< _Ti , _Tj >( img, real_x + _Tj(IMG_DERIVS_EPS), real_y ) - 
//                    real_out) / _Tj(IMG_DERIVS_EPS);
//   img_derivs[1] = (bilinearInterp< _Ti , _Tj >( img, real_x, real_y + _Tj(IMG_DERIVS_EPS) ) -
//                    real_out) / _Tj(IMG_DERIVS_EPS);
                   

  
  for( int i = 0; i < _Nj; i++)
    out.v(i,0) = img_derivs[0]*x.v(i,0) + img_derivs[1]*y.v(i,0);
  
  return out;
}

template < typename _Ti, typename _To, typename _Tj, int _Nj, MemoryAlignment _align > inline
  ceres::Jet<_Tj, _Nj> tensorbilinearInterp ( const ImageTensorBase<_align > &tensor,
                                              const ceres::Jet<_Tj, _Nj> &x, 
                                              const ceres::Jet<_Tj, _Nj> &y,
                                              const ceres::Jet<_Tj, _Nj> &z,
                                              bool cyclic_z = true )
{
  _Tj real_x = x.a, real_y = y.a, real_z = z.a, real_z_0, real_z_1;
  ceres::Jet<_Tj, _Nj> out;
  out.a = tensorGetPix< _Ti , _Tj >( tensor, real_x, real_y, real_z, cyclic_z );
  
  _Tj tensor_derivs[3], idt = _Tj(1.0/(2.0*IMG_DERIVS_EPS));
  
  // Compute the numerical derivatives in the tensor
  tensor_derivs[0] = ( tensorbilinearInterp< _Ti , _Tj >( tensor, real_x + _Tj(IMG_DERIVS_EPS), real_y, real_z, cyclic_z  ) - 
                       tensorbilinearInterp< _Ti , _Tj >( tensor, real_x - _Tj(IMG_DERIVS_EPS), real_y, real_z, cyclic_z  )) * idt;
  tensor_derivs[1] = ( tensorbilinearInterp< _Ti , _Tj >( tensor, real_x, real_y + _Tj(IMG_DERIVS_EPS), real_z, cyclic_z  ) -
                       tensorbilinearInterp< _Ti , _Tj >( tensor, real_x, real_y - _Tj(IMG_DERIVS_EPS), real_z, cyclic_z  )) * idt;
  tensor_derivs[2] = ( tensorbilinearInterp< _Ti , _Tj >( tensor, real_x, real_y, real_z + _Tj(IMG_DERIVS_EPS), cyclic_z  ) -
                       tensorbilinearInterp< _Ti , _Tj >( tensor, real_x, real_y, real_z - _Tj(IMG_DERIVS_EPS), cyclic_z  )) * idt;

  for( int i = 0; i < _Nj; i++)
    out.v(i,0) = tensor_derivs[0]*x.v(i,0) + tensor_derivs[1]*y.v(i,0) + tensor_derivs[2]*z.v(i,0);
  
  return out;  
}

template < typename _Ti > inline float bicubicInterp ( const cv::Mat &img, const float &x, const float &y )
{
  int x0 = floor ( double ( x ) ), y0 = floor ( double ( y ) );
  float dx = x - x0, dy = y - y0;
  float v[4];

  int tx = x0 - 1, ty = y0 - 1;
  for ( int i = 0; i < 4; i++ )
  {

    v[i] = cubicInterp ( dx, float ( img.at<_Ti> ( ty, tx ) ),
                         float ( img.at<_Ti> ( ty, tx + 1 ) ),
                         float ( img.at<_Ti> ( ty, tx + 2 ) ),
                         float ( img.at<_Ti> ( ty, tx + 3 ) ) );
    ty++;
  }

  return cubicInterp ( dy, v[0],v[1],v[2],v[3] );
}

}
