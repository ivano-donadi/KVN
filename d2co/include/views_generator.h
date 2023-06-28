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

#include <iostream>
#include <vector>

#include "cv_ext/cv_ext.h"

/**
 * @brief ViewsGenerator is an utilty class used to sample a set of views (i.e., rigid body transformations) 
 *        given a set of roll, pitch, yaw and depths intervals, and angular and depth sample steps.
 * 
 * This class is used to generate a set of views of an object, to be used for instance to generate a set of template
 * (one for eaach view) of such object.
 */
class ViewsGenerator
{
public:
 
  /**
   * @brief Provide the current roll and pitch ranges (in radians). 
   * 
   * The roll and pitch represent rotations around the x and y axes.
   */
  std::vector< cv_ext::IntervalD > rollPitchRanges() const { return rp_ranges_; }
  
  /**
   * @brief Provide the current yaw ranges (in radians). 
   * 
   * The yaw represents the rotation around the z axis
   */
  std::vector< cv_ext::IntervalD > yawRanges() const { return yaw_ranges_; }
  
  /**
   * @brief Provide the current depth ranges (in meters). 
   * 
   * The depth represents translation along the z axis
   */  
  std::vector< cv_ext::IntervalD > depthRanges() const { return z_ranges_; }
  
  /**
   * @brief Provide the angular step (in radias) used to sample the roll, pitch, and yaw rotations. 
   * 
   * @note This step represents the maximum allowed step.
   */    
  double angStep() const { return ang_step_; };
  
  /**
   * @brief Provide the depth step (in meters) used to sample the depth translations. 
   * 
   * @note This step represents the maximum allowed step.
   */  
  double depthStep() const { return z_step_; };
  
  /**
   * @brief Add a roll and pitch range to the current set of roll and pitch ranges. 
   * 
   * @param[in] range The new roll and pitch range (in radians)
   * 
   * The roll and pitch represent rotations around the x and y axes.
   */
  void addRollPitchRange( const cv_ext::IntervalD &range ){ rp_ranges_.push_back(range); }
  
  /**
   * @brief Add a yaw range to the current set of yaw ranges. 
   * 
   * @param[in] range The new yaw range (in radians)
   * 
   * The yaw represents the rotation around the z axis
   */  
  void addYawRange( const cv_ext::IntervalD &range ){ yaw_ranges_.push_back(range); }
  
  /**
   * @brief Add a depth range to the current set of depth ranges. 
   * 
   * @param[in] range The new depth range (in meters) 
   * 
   * The depth represents translation along the z axis
   */   
  void addDepthRange( const cv_ext::IntervalD &range ){ z_ranges_.push_back(range); }
  
  /**
   * @brief Set the angular step used to sample the roll, pitch, and yaw rotations. 
   * 
   * @param[in] step The angular step (in radias) 
   * 
   * @note This step represents the maximum allowed step.
   */
  void setAngStep( double step ){ ang_step_ = step; };

  /**
   * @brief Set the depth step used to sample the depth translations. 
   * 
   * @param[in] step The depth step (in meters) 
   * 
   * @note This step represents the maximum allowed step.
   */  
  void setDepthStep( double step ){ z_step_ = step; };
  
  /**
   * @brief This method samples a set of views (i.e., rigid body transformations) given 
   * the provided intervals and sample steps (see addRollPitchRange(), setMinAngStep(), ...).
   * 
   * @param[out] r_quats Output vector of quaternios representing the rotations of each generated view
   * @param[out] t_vecs Output vector of 3D vectrors representing the translation of each generated view
   * 
   * Ypu should provide all the intervals and sample steps, otherwise this method will provide 
   * empty out vectors. 
   * @note The translations are alsways zero along the x and y axes
   */
  void generate( cv_ext::vector_Quaterniond &r_quats, cv_ext::vector_Vector3d &t_vecs );
  
  /**
   * @brief Clear all the intervals and reset the steps to the defult value.
   */  
  void reset();
  
private:
 
  std::vector< cv_ext::IntervalD > rp_ranges_, yaw_ranges_, z_ranges_;
  double ang_step_ = 0, z_step_ = 0; 
};
