#pragma once

#include "cv_ext/cv_ext.h"

#include <string>
#include <vector>

bool readFileNamesFromFolder ( const std::string& input_folder_name, std::vector< std::string >& names );
void objectPoseControlsHelp();
void parseObjectPoseControls ( int key ,cv::Mat &r_vec, cv::Mat &t_vec,
                               double r_inc = 0.01, double t_inc = 0.01 );
