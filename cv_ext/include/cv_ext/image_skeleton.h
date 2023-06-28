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

/* TODO
 *
 * WARNING
 *
 * - In shrinkIntersections() unused s_p pointer
 */
namespace cv_ext
{
/**
  * @brief Simple thinning algorithm
  * 
  * @param[in] src Input image: it should be a binary (1 and 0 values) 8 bits single channel image.
  *            In case, use the binarize flag.
  * @param[out] dst Ouput (binary) image
  * @param[in] binarize true if you want to binarize the input image, defaults to true.
  * @param[in] thresh Threshold for the binarization, defaults to 0.
  * 
  * Implementation of the thinning algorithm presented in the paper 
  * "A Fast Parallel Algorithm for Thinning Digital Patterns"
  * T. Y. Zhang and C. Y. Suen
  * Communications of the ACM March 1984 Volume 27 Number 3
  */
void morphThinning( const cv::Mat &src, cv::Mat &dst, bool binarize = true, uchar thresh = 0 );


/**
  * @brief Extract a (possibly not connected) graph from a binary skeleton
  * 
  * @param[in] skeleton Input skeleton image: it should be a general, 0 and X values binary
  *                 8 bits single channel image.
  * @param[out] nodes Output graph nodes
  * @param[out] edges Output graph edges, by means of a vecto of vectors of index
  * @param[in] find_leafs true if you want to extract also the leaf nodes, defaults to true
  * @param[in] min_dist minimum distance between nodes, it should be >= 1, default to 1
  */
void graphExtraction( const cv::Mat &skeleton, std::vector<cv::Point2f> &nodes,
                      std::vector< std::vector<int> > &edges, 
                      bool find_leafs = true, int min_dist = 1 );

}
