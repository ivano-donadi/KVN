/*
 * d2co - Direct Directional Chamfer Optimization
 *
 *  Copyright (c) 2020, Alberto Pretto <alberto.pretto@flexsight.eu>
 *                      Marco Imperoli <marco.imperoli@flexsight.eu>
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

#include <vector>
#include <opencv2/opencv.hpp>

class HDRCreator
{

public:
  enum HDRCreatorWeightsType
  {
    HDRCreatorWeightsType_Triangular = 0,
    HDRCreatorWeightsType_Gaussian,
    HDRCreatorWeightsType_Plateau
  };

  enum HDRCreatorResponseType
  {
    HDRCreatorResponseType_Linear = 0,
    HDRCreatorResponseType_Gamma,
    HDRCreatorResponseType_Log10
  };

  enum HDRCreatorMappingMethod
  {
    HDRCreatorMappingMethod_Linear = 0,
    HDRCreatorMappingMethod_Gamma_1_4,
    HDRCreatorMappingMethod_Gamma_1_8,
    HDRCreatorMappingMethod_Gamma_2_2,
    HDRCreatorMappingMethod_Gamma_2_6,
    HDRCreatorMappingMethod_Log10
  };


private:


  HDRCreatorWeightsType e_WeightsType;
  HDRCreatorResponseType e_ResponseType;
  HDRCreatorMappingMethod e_MappingMethod;

  float hist_win_min, hist_win_max;
  int min_response, max_response;

  // ---- Constructors - Destructors - Initialization
public:
  HDRCreator ();
  ~HDRCreator();

private:
  void InitBasicVars();

  // ---- Methods
public:

  cv::Mat ComputeHDR ( std::vector<cv::Mat> &ldr_images, std::vector<float> &arrayofexptime );
  cv::Mat MapToLDR ( cv::Mat &hdrimage, double targetexposure = 0 );

  // ---- Calculation Methods
private:
  void applyResponse ( std::vector<cv::Mat> &ldr_images, const std::vector<float> &arrayofexptime, cv::Mat &hdr_out,
                       const float* response, const float* w, const int pix_levels );

  void weightsTriangle ( float* w, int M );
  void weightsGauss ( float* w, int M, int Mmin, int Mmax, float sigma  = 8.0f );
  void exposureWeightsIcip06 ( float* w, int M, int Mmin, int Mmax );

  void responseLinear ( float* response, int M );
  void responseGamma ( float* response, int M );
  void responseLog10 ( float* response, int M );

  typedef int ( HDRCreator::*MappingFunc ) ( float x );
  inline int getMappingLinear ( float x );
  inline int getMappingGamma_1_4 ( float x );
  inline int getMappingGamma_1_8 ( float x );
  inline int getMappingGamma_2_2 ( float x );
  inline int getMappingGamma_2_6 ( float x );
  inline int getMappingLog10 ( float x );

  void fitToDynamicRange ( cv::Mat &hdrimage );

  template<class T> T clamp ( T val, T min, T max );
};
